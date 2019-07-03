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

GLfloat     *confirmCoordBuffer  = NULL;  // Stores (X, Y) (float) for each vertex
GLfloat     *confirmColorBuffer  = NULL;  // Stores (R, G, B) (float) for each vertex
GLushort    *confirmIndices      = NULL;  // Stores index corresponding to each vertex
GLuint      confirmVerts;                 // Total number of vertices
GLuint      extraConfirmVerts;
Matrix      confirmMVP;                   // Transformation matrix passed to shader
Params      confirmPrevState;             // Stores transformations to avoid redundant recalculation
GLuint      confirmVBO;                   // Vertex Buffer Object ID
GLboolean   confirmFirstRun = GL_TRUE;    // Determines if function is running for the first time (for VBO initialization)

PyObject* drawConfirm_drawButtons(PyObject* self, PyObject *args) {
   PyObject *faceColorPyTup;
   PyObject *extraColorPyTup;
   PyObject *detailColorPyTup;
   GLfloat gx, gy, scale, w2h, ao=0.0f;
   GLfloat faceColor[3];
   GLfloat extraColor[3];
   GLfloat detailColor[3];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "ffffOOO",
            &gx, &gy,
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
   if (  confirmCoordBuffer   == NULL  ||
         confirmColorBuffer   == NULL  ||
         confirmIndices       == NULL  ){

      printf("Initializing Geometry for Confirm Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      float px, py, qx, qy, radius;
      int circleSegments = 60;
      defineEllipse(
            0.0f, 0.0f,
            1.0f, 1.0f,
            circleSegments,
            faceColor,
            verts, colrs);

      px = -0.75f, py =  0.0f;
      qx = -0.25f, qy = -0.5f;
      radius = float(sqrt(2.0)*0.125f);
      definePill(
            px, py,
            qx, qy,
            radius,
            circleSegments/2,
            detailColor,
            verts, colrs);


      px = 0.625f, py = 0.375f;
      extraConfirmVerts = definePill(
            px, py, 
            qx, qy, 
            radius, 
            circleSegments/2,
            detailColor, 
            verts, colrs);

      px = -0.75f, py =  0.0f;
      radius = 0.125f;
      definePill(
            px, py,
            qx, qy,
            radius,
            circleSegments/2,
            extraColor,
            verts, colrs);

      px = 0.625f, py = 0.375f;
      definePill(
            px, py,
            qx, qy,
            radius,
            circleSegments/2,
            extraColor,
            verts, colrs);

      confirmVerts = verts.size()/2;

      // Pack Vertices and Colors into global array buffers
      if (confirmCoordBuffer == NULL) {
         confirmCoordBuffer = new GLfloat[confirmVerts*2];
      } else {
         delete [] confirmCoordBuffer;
         confirmCoordBuffer = new GLfloat[confirmVerts*2];
      }

      if (confirmColorBuffer == NULL) {
         confirmColorBuffer = new GLfloat[confirmVerts*3];
      } else {
         delete [] confirmColorBuffer;
         confirmColorBuffer = new GLfloat[confirmVerts*3];
      }

      if (confirmIndices == NULL) {
         confirmIndices = new GLushort[confirmVerts];
      } else {
         delete [] confirmIndices;
         confirmIndices = new GLushort[confirmVerts];
      }

      for (unsigned int i = 0; i < confirmVerts; i++) {
         confirmCoordBuffer[i*2]   = verts[i*2];
         confirmCoordBuffer[i*2+1] = verts[i*2+1];
         confirmIndices[i]          = i;
         confirmColorBuffer[i*3+0]  = colrs[i*3+0];
         confirmColorBuffer[i*3+1]  = colrs[i*3+1];
         confirmColorBuffer[i*3+2]  = colrs[i*3+2];
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
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);

      // Cache Matrix to global memory
      MatrixMultiply( &confirmMVP, &ModelView, &Ortho );

      // Cache transformations to avoid recalculation
      confirmPrevState.ao = ao;
      confirmPrevState.dx = gx;
      confirmPrevState.dy = gy;
      confirmPrevState.sx = scale;
      confirmPrevState.sy = scale;
      confirmPrevState.w2h = w2h;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (confirmFirstRun == GL_TRUE) {
         confirmFirstRun = GL_FALSE;
         glGenBuffers(1, &confirmVBO);
      } else {
         glDeleteBuffers(1, &confirmVBO);
         glGenBuffers(1, &confirmVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, confirmVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*confirmVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLintptr offset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*confirmVerts, confirmCoordBuffer);

      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);

      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*confirmVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*confirmVerts, confirmColorBuffer);

      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (GLintptr*)offset);

      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   }

   // Geometry allocated, check if color needs to be updated
   for (int i = 0; i < 3; i++) {
      if ( confirmColorBuffer[extraConfirmVerts*3+i] != extraColor[i] ) {
         for (unsigned int k = extraConfirmVerts; k < confirmVerts; k++) {
            confirmColorBuffer[k*3 + i] = extraColor[i];
         }
         // Update Contents of VBO
         // Set active VBO
         glBindBuffer(GL_ARRAY_BUFFER, confirmVBO);
         // Convenience variable
         GLintptr offset = 2*sizeof(GLfloat)*confirmVerts;
         // Load Vertex Color data into VBO
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*confirmVerts, confirmColorBuffer);
      }
   }

   // Update Transfomation Matrix if any change in parameters
   if (  confirmPrevState.ao != ao     ||
         confirmPrevState.dx != gx     ||
         confirmPrevState.dy != gy     ||
         confirmPrevState.sx != scale  ||
         confirmPrevState.sy != scale  ||
         confirmPrevState.w2h != w2h   ){
      
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
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &confirmMVP, &ModelView, &Ortho );

      confirmPrevState.ao = ao;
      confirmPrevState.dx = gx;
      confirmPrevState.dy = gy;
      confirmPrevState.sx = scale;
      confirmPrevState.sy = scale;
      confirmPrevState.w2h = w2h;

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, confirmVBO);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*confirmVerts));
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &confirmMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, confirmVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*confirmVerts));

   // Not sure if I actually need these
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, confirmVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   Py_RETURN_NONE;
}
