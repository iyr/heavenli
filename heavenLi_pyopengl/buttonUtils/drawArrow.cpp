#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
   // These undefs necessary because microsoft
   #undef near
   #undef far
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#include <vector>
#include <math.h>

using namespace std;

GLfloat     *arrowCoordBuffer  = NULL; // Stores (X, Y) (float) for each vertex
GLfloat     *arrowColorBuffer  = NULL; // Stores (R, G, B) (float) for each vertex
GLushort    *arrowIndices      = NULL; // Stores index corresponding to each vertex
GLuint      arrowVerts;                // Total number of vertices
GLuint      extraArrowVerts;
Matrix      arrowMVP;                  // Transformation matrix passed to shader
Params      arrowPrevState;            // Stores transformations to avoid redundant recalculation
GLuint      arrowVBO;                  // Vertex Buffer Object ID
GLboolean   arrowFirstRun = GL_TRUE;   // Determines if function is running for the first time (for VBO initialization)

PyObject* drawArrow_drawButtons(PyObject* self, PyObject *args) {
   PyObject *faceColorPyTup;
   PyObject *extraColorPyTup;
   PyObject *detailColorPyTup;
   float gx, gy, ao, scale, w2h;
   float faceColor[3];
   float extraColor[3];
   float detailColor[3];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "fffffOOO",
            &gx, &gy,
            &ao,
            &scale,
            &w2h,
            &faceColorPyTup,
            &extraColorPyTup,
            &detailColorPyTup) )
   {
      Py_RETURN_NONE;
   }

   faceColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 0)));
   faceColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 1)));
   faceColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 2)));

   extraColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 0)));
   extraColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 1)));
   extraColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 2)));

   detailColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0)));
   detailColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1)));
   detailColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2)));

   // Allocate and Define Geometry/Color buffers
   if (  arrowCoordBuffer  == NULL  ||
         arrowColorBuffer  == NULL  ||
         arrowIndices      == NULL  ){

      printf("Initializing Geometry for Arrow Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      float px, py, qx, qy, radius;
      int circleSegments = 60;
      drawEllipse(0.0f, 0.0f, 1.0f, circleSegments, faceColor, verts, colrs);

      px = -0.125f, py = 0.625f;
      qx =  0.500f, qy =  0.00f;
      radius = float(sqrt(2.0)*0.125f);
      drawPill(px, py, qx, qy, radius, detailColor, verts, colrs);

      px = -0.125f, py = -0.625f;
      extraArrowVerts = drawPill(px, py, qx, qy, radius, detailColor, verts, colrs);

      px = -0.125f, py =  0.625f;
      radius = 0.125f;
      drawPill(px, py, qx, qy, radius, extraColor, verts, colrs);

      px = -0.125f, py = -0.625f;
      drawPill(px, py, qx, qy, radius, extraColor, verts, colrs);

      arrowVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (arrowCoordBuffer == NULL) {
         arrowCoordBuffer = new GLfloat[arrowVerts*2];
      } else {
         delete [] arrowCoordBuffer;
         arrowCoordBuffer = new GLfloat[arrowVerts*2];
      }

      if (arrowColorBuffer == NULL) {
         arrowColorBuffer = new GLfloat[arrowVerts*3];
      } else {
         delete [] arrowColorBuffer;
         arrowColorBuffer = new GLfloat[arrowVerts*3];
      }

      if (arrowIndices == NULL) {
         arrowIndices = new GLushort[arrowVerts];
      } else {
         delete [] arrowIndices;
         arrowIndices = new GLushort[arrowVerts];
      }

      for (unsigned int i = 0; i < arrowVerts; i++) {
         arrowCoordBuffer[i*2]   = verts[i*2];
         arrowCoordBuffer[i*2+1] = verts[i*2+1];
         arrowIndices[i]          = i;
         arrowColorBuffer[i*3+0]  = colrs[i*3+0];
         arrowColorBuffer[i*3+1]  = colrs[i*3+1];
         arrowColorBuffer[i*3+2]  = colrs[i*3+2];
      }

      // Calculate Initial Transformation Matrix
      Matrix Ortho;
      Matrix ModelView;

      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, 1.0f*gx, 1.0f*gy, 0.0f );
      if (w2h <= 1.0f) {
         MatrixScale( &ModelView, scale, scale*w2h, 1.0f );
      } else {
         MatrixScale( &ModelView, scale/w2h, scale, 1.0f );
      }
      MatrixRotate( &ModelView, ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &arrowMVP, &ModelView, &Ortho );

      arrowPrevState.ao = ao;
      arrowPrevState.dx = gx;
      arrowPrevState.dy = gy;
      arrowPrevState.sx = scale;
      arrowPrevState.sy = scale;
      arrowPrevState.w2h = w2h;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (arrowFirstRun == GL_TRUE) {
         arrowFirstRun = GL_FALSE;
         glGenBuffers(1, &arrowVBO);
      } else {
         glDeleteBuffers(1, &arrowVBO);
         glGenBuffers(1, &arrowVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, arrowVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*arrowVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLuint64 offset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*arrowVerts, arrowCoordBuffer);
      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLuint64*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*arrowVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*arrowVerts, arrowColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (GLuint64*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   }

   // Update Color, if needed
   for (int i = 0; i < 3; i++) {
      if ( arrowColorBuffer[extraArrowVerts*3+i] != extraColor[i] ) {
         for (unsigned int k = extraArrowVerts; k < arrowVerts; k++) {
            arrowColorBuffer[k*3 + i] = extraColor[i];
         }
         // Update Contents of VBO
         // Set active VBO
         glBindBuffer(GL_ARRAY_BUFFER, arrowVBO);
         // Convenience variable
         GLuint64 offset = 2*sizeof(GLfloat)*arrowVerts;
         // Load Vertex Color data into VBO
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*arrowVerts, arrowColorBuffer);
      }
   }

   // Update Transfomation Matrix if any chaoe in parameters
   if (  arrowPrevState.ao != ao     ||
         arrowPrevState.dx != gx     ||
         arrowPrevState.dy != gy     ||
         arrowPrevState.sx != scale  ||
         arrowPrevState.sy != scale  ||
         arrowPrevState.w2h != w2h   ){
      
      Matrix Ortho;
      Matrix ModelView;

      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, 1.0f*gx, 1.0f*gy, 0.0f );
      if (w2h <= 1.0f) {
         MatrixScale( &ModelView, scale, scale*w2h, 1.0f );
      } else {
         MatrixScale( &ModelView, scale/w2h, scale, 1.0f );
      }
      MatrixRotate( &ModelView, ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &arrowMVP, &ModelView, &Ortho );

      arrowPrevState.ao = ao;
      arrowPrevState.dx = gx;
      arrowPrevState.dy = gy;
      arrowPrevState.sx = scale;
      arrowPrevState.sy = scale;
      arrowPrevState.w2h = w2h;

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, arrowVBO);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*arrowVerts));
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &arrowMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, arrowVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*arrowVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, arrowVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   Py_RETURN_NONE;
}
