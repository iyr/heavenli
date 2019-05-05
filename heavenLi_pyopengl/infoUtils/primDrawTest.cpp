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

GLfloat     *primTestCoordBuffer  = NULL;  // Stores (X, Y) (float) for each vertex
GLfloat     *primTestColorBuffer  = NULL;  // Stores (R, G, B) (float) for each vertex
GLushort    *primTestIndices      = NULL;  // Stores index corresponding to each vertex
GLuint      primTestVerts;                 // Total number of vertices
GLuint      extraPrimTestVerts;
Matrix      primTestMVP;                   // Transformation matrix passed to shader
Params      primTestPrevState;             // Stores transformations to avoid redundant recalculation
GLuint      primTestVBO;                   // Vertex Buffer Object ID
GLboolean   primTestFirstRun = GL_TRUE;    // Determines if function is running for the first time (for VBO initialization)

PyObject* primTest_drawButtons(PyObject* self, PyObject *args) {
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
   if (  primTestCoordBuffer   == NULL  ||
         primTestColorBuffer   == NULL  ||
         primTestIndices       == NULL  ){

      printf("Initializing Geometry for PrimTest Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      defineArch( 0.0f,+0.5f, 0.3f, 0.3f, 9.0f, 260.0f, 0.1f, 15, detailColor, verts, colrs);
      defineEllipse(-1.0f,-1.0f, 0.4f, 0.4f, 32, faceColor, verts, colrs);
      defineArch( 0.0f,-0.4f, 0.3f, 0.3f, 0.0f, 180.0f, 0.1f, 15, detailColor, verts, colrs);
      defineEllipse(-1.0f, 1.0f, 0.4f, 0.4f, 32, faceColor, verts, colrs);
      defineArch(+0.5f, 0.0f, 0.3f, 0.3f, 0.0f, 360.0f, 0.1f, 12, faceColor, verts, colrs);
      defineEllipse( 1.0f,-1.0f, 0.4f, 0.4f, 32, faceColor, verts, colrs);
      defineEllipse( 0.6f, 0.6f, 0.4f, 0.4f, 32, faceColor, verts, colrs);
      defineArch(-0.5f, 0.0f, 0.3f, 0.3f, 0.0f, 160.0f, 0.1f, 24, extraColor, verts, colrs);

      primTestVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (primTestCoordBuffer == NULL) {
         primTestCoordBuffer = new GLfloat[primTestVerts*2];
      } else {
         delete [] primTestCoordBuffer;
         primTestCoordBuffer = new GLfloat[primTestVerts*2];
      }

      if (primTestColorBuffer == NULL) {
         primTestColorBuffer = new GLfloat[primTestVerts*3];
      } else {
         delete [] primTestColorBuffer;
         primTestColorBuffer = new GLfloat[primTestVerts*3];
      }

      if (primTestIndices == NULL) {
         primTestIndices = new GLushort[primTestVerts];
      } else {
         delete [] primTestIndices;
         primTestIndices = new GLushort[primTestVerts];
      }

      for (unsigned int i = 0; i < primTestVerts; i++) {
         primTestCoordBuffer[i*2]   = verts[i*2];
         primTestCoordBuffer[i*2+1] = verts[i*2+1];
         primTestIndices[i]          = i;
         primTestColorBuffer[i*3+0]  = colrs[i*3+0];
         primTestColorBuffer[i*3+1]  = colrs[i*3+1];
         primTestColorBuffer[i*3+2]  = colrs[i*3+2];
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
      MatrixMultiply( &primTestMVP, &ModelView, &Ortho );

      // Cache transformations to avoid recalculation
      primTestPrevState.ao = ao;
      primTestPrevState.dx = gx;
      primTestPrevState.dy = gy;
      primTestPrevState.sx = scale;
      primTestPrevState.sy = scale;
      primTestPrevState.w2h = w2h;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (primTestFirstRun == GL_TRUE) {
         primTestFirstRun = GL_FALSE;
         glGenBuffers(1, &primTestVBO);
      } else {
         glDeleteBuffers(1, &primTestVBO);
         glGenBuffers(1, &primTestVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, primTestVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*primTestVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLintptr offset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*primTestVerts, primTestCoordBuffer);

      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);

      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*primTestVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*primTestVerts, primTestColorBuffer);

      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (GLintptr*)offset);

      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   }

   /*
   // Geometry allocated, check if color needs to be updated
   for (int i = 0; i < 3; i++) {
      if ( primTestColorBuffer[extraPrimTestVerts*3+i] != extraColor[i] ) {
         for (unsigned int k = extraPrimTestVerts; k < primTestVerts; k++) {
            primTestColorBuffer[k*3 + i] = extraColor[i];
         }
         // Update Contents of VBO
         // Set active VBO
         glBindBuffer(GL_ARRAY_BUFFER, primTestVBO);
         // Convenience variable
         GLintptr offset = 2*sizeof(GLfloat)*primTestVerts;
         // Load Vertex Color data into VBO
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*primTestVerts, primTestColorBuffer);
      }
   }
   */

   // Update Transfomation Matrix if any change in parameters
   if (  primTestPrevState.ao != ao     ||
         primTestPrevState.dx != gx     ||
         primTestPrevState.dy != gy     ||
         primTestPrevState.sx != scale  ||
         primTestPrevState.sy != scale  ||
         primTestPrevState.w2h != w2h   ){
      
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
      MatrixMultiply( &primTestMVP, &ModelView, &Ortho );

      primTestPrevState.ao = ao;
      primTestPrevState.dx = gx;
      primTestPrevState.dy = gy;
      primTestPrevState.sx = scale;
      primTestPrevState.sy = scale;
      primTestPrevState.w2h = w2h;

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, primTestVBO);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*primTestVerts));
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &primTestMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, primTestVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*primTestVerts));

   // Not sure if I actually need these
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, primTestVerts);
   //glDrawElements(GL_TRIANGLE_STRIP, primTestVerts, GL_UNSIGNED_SHORT, colrTriIndices);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   Py_RETURN_NONE;
}
