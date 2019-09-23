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

GLfloat     *granRockerCoordBuffer  = NULL;
GLfloat     *granRockerColorBuffer   = NULL;
GLushort    *granRockerIndices       = NULL;
GLuint      granRockerVerts;
Matrix      granRockerMVP;                   // Transformation matrix passed to shader
Params      granRockerPrevState;             // Stores transformations to avoid redundant recalculation
GLuint      granRockerVBO;                   // Vertex Buffer Object ID
GLboolean   granRockerFirstRun = GL_TRUE;    // Determines if function is running for the first time (for VBO initialization)

PyObject* drawGranRocker_drawUtils(PyObject *self, PyObject *args) {
   PyObject *py_faceColor;
   PyObject *py_detailColor;
   int   numHues;
   GLfloat gx, gy, rotation, w2h, scale, tDiff, ao=0.0f;
   GLfloat faceColor[3];
   GLfloat detailColor[3];
   GLfloat black[3] = {0.0, 0.0, 0.0};
   GLfloat white[3] = {1.0, 1.0, 1.0};
   char circleSegments = 45;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "ffOOlffff",
            &gx,
            &gy,
            &py_faceColor,
            &py_detailColor,
            &numHues,
            &rotation,
            &w2h,
            &scale,
            &tDiff)) {
      Py_RETURN_NONE;
   }

   // Parse Colors
   faceColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 0)));
   faceColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 1)));
   faceColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 2)));
   detailColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 0)));
   detailColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 1)));
   detailColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 2)));

   // Allocate and Define Geometry/Color buffers
   if (  granRockerCoordBuffer   == NULL  ||
         granRockerColorBuffer   == NULL  ||
         granRockerIndices       == NULL  ){
      printf("Initializing Geometry for Granularity Rocker\n");
      float unit = float(1.0/36.0);
      float R, G, B, buttonSize = 0.8f;
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      R = black[0];
      G = black[1];
      B = black[2];

      // Upper Background Mask (quad)
      /* X */ verts.push_back(-1.0);
      /* Y */ verts.push_back( 1.0);
      /* X */ verts.push_back(-1.0);
      /* Y */ verts.push_back( 0.0);
      /* X */ verts.push_back( 1.0);
      /* Y */ verts.push_back( 1.0);

      /* X */ verts.push_back( 1.0);
      /* Y */ verts.push_back( 1.0);
      /* X */ verts.push_back(-1.0);
      /* Y */ verts.push_back( 0.0);
      /* X */ verts.push_back( 1.0);
      /* Y */ verts.push_back( 0.0);

      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);

      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);

      // Lower Background Mask (Pill)
      drawPill(
            -24.0f*unit, 0.0f,   /* X, Y */
             24.0f*unit, 0.0f,   /* X, Y */
             12.0f*unit,         /* Radius */
             black,              /* Color */
             verts,
             colrs);

      // Left (Minus) Button
      drawCircle(
            -24.0f*unit, 0.0f, /* X, Y */
             12.0f*unit*buttonSize,       /* Radius */
             circleSegments,              /* Number of Circle Triangles */
             faceColor,                   /* Colors */
             verts,
             colrs);

      // Right (Plus) Button
      drawCircle(
             24.0f*unit, 0.0f, /* X, Y */
             12.0f*unit*buttonSize,       /* Radius */
             circleSegments,              /* Number of Circle Triangles */
             faceColor,                   /* Colors */
             verts,
             colrs);

      // Iconography
      drawCircle(-5.0f*unit*buttonSize,  6.0f*unit*buttonSize, 4.0f*unit*buttonSize, circleSegments, white, verts, colrs);
      drawCircle( 5.0f*unit*buttonSize,  0.0f*unit*buttonSize, 4.0f*unit*buttonSize, circleSegments, white, verts, colrs);
      drawCircle(-5.0f*unit*buttonSize, -6.0f*unit*buttonSize, 4.0f*unit*buttonSize, circleSegments, white, verts, colrs);

      // Minus Symbol
      float tmo = 18.0f;
      drawPill(
            -32.0f*unit + tmo*unit*buttonSize, 0.0f,  /* X, Y */
            -16.0f*unit - tmo*unit*buttonSize, 0.0f,  /* X, Y */
            2.0f*unit*buttonSize,                     /* Radius */
            detailColor,                              /* Color */
            verts,
            colrs);

      // Plus Symbol
      drawPill(
            32.0f*unit - tmo*unit*buttonSize, 0.0f,   /* X, Y */
            16.0f*unit + tmo*unit*buttonSize, 0.0f,   /* X, Y */
            2.0f*unit*buttonSize,                     /* Radius */
            detailColor,                              /* Color */
            verts,
            colrs);
      drawPill(
            24.0f*unit,  8.0f*unit*buttonSize,  /* X, Y */
            24.0f*unit, -8.0f*unit*buttonSize,  /* X, Y */
            2.0f*unit*buttonSize,               /* Radius */
            detailColor,                        /* Color */
            verts,
            colrs);

      granRockerVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (granRockerCoordBuffer == NULL) {
         granRockerCoordBuffer = new GLfloat[granRockerVerts*2];
      } else {
         delete [] granRockerCoordBuffer;
         granRockerCoordBuffer = new GLfloat[granRockerVerts*2];
      }

      if (granRockerColorBuffer == NULL) {
         granRockerColorBuffer = new GLfloat[granRockerVerts*3];
      } else {
         delete [] granRockerColorBuffer;
         granRockerColorBuffer = new GLfloat[granRockerVerts*3];
      }

      if (granRockerIndices == NULL) {
         granRockerIndices = new GLushort[granRockerVerts];
      } else {
         delete [] granRockerIndices;
         granRockerIndices = new GLushort[granRockerVerts];
      }

      for (unsigned int i = 0; i < granRockerVerts; i++) {
         granRockerCoordBuffer[i*2]    = verts[i*2];
         granRockerCoordBuffer[i*2+1]  = verts[i*2+1];
         granRockerIndices[i]          = i;
         granRockerColorBuffer[i*3+0]  = colrs[i*3+0];
         granRockerColorBuffer[i*3+1]  = colrs[i*3+1];
         granRockerColorBuffer[i*3+2]  = colrs[i*3+2];
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
      MatrixMultiply( &granRockerMVP, &ModelView, &Ortho );

      granRockerPrevState.ao = ao;
      granRockerPrevState.dx = gx;
      granRockerPrevState.dy = gy;
      granRockerPrevState.sx = scale;
      granRockerPrevState.sy = scale;
      granRockerPrevState.w2h = w2h;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (granRockerFirstRun == GL_TRUE) {
         granRockerFirstRun = GL_FALSE;
         glGenBuffers(1, &granRockerVBO);
      } else {
         glDeleteBuffers(1, &granRockerVBO);
         glGenBuffers(1, &granRockerVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, granRockerVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*granRockerVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLintptr offset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*granRockerVerts, granRockerCoordBuffer);
      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*granRockerVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*granRockerVerts, granRockerColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   }

   // Update Transfomation Matrix if any change in parameters
   if (  granRockerPrevState.ao != ao     ||
         granRockerPrevState.dx != gx     ||
         granRockerPrevState.dy != gy     ||
         granRockerPrevState.sx != scale  ||
         granRockerPrevState.sy != scale  ||
         granRockerPrevState.w2h != w2h   ){
      
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
      MatrixMultiply( &granRockerMVP, &ModelView, &Ortho );

      granRockerPrevState.ao = ao;
      granRockerPrevState.dx = gx;
      granRockerPrevState.dy = gy;
      granRockerPrevState.sx = scale;
      granRockerPrevState.sy = scale;
      granRockerPrevState.w2h = w2h;

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, granRockerVBO);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*granRockerVerts));
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &granRockerMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, granRockerVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*granRockerVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, granRockerVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   Py_RETURN_NONE;
}
