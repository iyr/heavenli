#include <Python.h>
#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
   // These undefs necessary because microsoft
   #undef near
   #undef far
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>
using namespace std;
extern float offScreen;

float constrain(float value, float min, float max) {
   if (value > max)
      return max;
   else if (value < min)
      return min;
   else
      return value;
}

GLfloat     *homeLinearCoordBuffer  = NULL; // Stores (X, Y) (float) for each vertex
GLfloat     *homeLinearColorBuffer  = NULL; // Stores (R, G, B) (float) for each vertex
GLushort    *homeLinearIndices      = NULL; // Stores index corresponding to each vertex
GLuint      homeLinearVerts;
GLint       prevHomeLinearNumbulbs;
Matrix      homeLinearMVP;                  // Transformation matrix passed to shader
Params      homeLinearPrevState;            // Stores transformations to avoid redundant recalculation
GLuint      homeLinearVBO;                  // Vertex Buffer Object ID
GLboolean   homeLinearFirstRun = GL_TRUE;   // Determines if function is running for the first time (for VBO initialization)

PyObject* drawHomeLinear_drawArn(PyObject *self, PyObject *args) {
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   GLfloat *bulbColors;
   GLfloat gx, gy, wx, wy, ao, w2h; 
   GLfloat R, G, B;
   GLint numBulbs;
   if (!PyArg_ParseTuple(args,
            "fffflffO",
            &gx, &gy,
            &wx, &wy,
            &numBulbs,
            &ao,
            &w2h,
            &py_list
            ))
   {
      Py_RETURN_NONE;
   }

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new float[numBulbs*3];
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = float(PyFloat_AsDouble(py_float));
      }
   }

   // Allocate and Define Geometry/Color buffers
   if (homeLinearCoordBuffer    == NULL ||
       homeLinearColorBuffer     == NULL ||
       homeLinearIndices         == NULL ){

      printf("Generating geometry for homeLinear\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy;
      float offset = float(4.0/60.0);
      R = 0.0;
      G = 0.0;
      B = 0.0;
      for (int i = 0; i < 60; i++) {
         if (i == 0) {
            TLx = -4.0;
            TLy =  4.0;

            BLx = -4.0;
            BLy = -4.0;
         } else {
            TLx = float(-2.0 + i*offset);
            TLy =  4.0;

            BLx = float(-2.0 + i*offset);
            BLy = -4.0;
         }

         if (i == 60-1) {
            TRx =  4.0;
            TRy =  4.0;

            BRx =  4.0;
            BRy = -4.0;
         } else {
            TRx = float(-2.0 + (i+1)*offset);
            TRy =  4.0;

            BRx = float(-2.0 + (i+1)*offset);
            BRy = -4.0;
         }

         /* X */ verts.push_back(TLx);   /* Y */ verts.push_back(TLy);
         /* X */ verts.push_back(BLx);   /* Y */ verts.push_back(BLy);
         /* X */ verts.push_back(TRx);   /* Y */ verts.push_back(TRy);

         /* X */ verts.push_back(TRx);   /* Y */ verts.push_back(TRy);
         /* X */ verts.push_back(BLx);   /* Y */ verts.push_back(BLy);
         /* X */ verts.push_back(BRx);   /* Y */ verts.push_back(BRy);

         for (int j = 0; j < 6; j++) {
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
         }
      }

      homeLinearVerts = verts.size()/2;
      printf("homeLinear vertexBuffer length: %.i, Number of vertices: %.i, tris: %.i\n", homeLinearVerts*2, homeLinearVerts, homeLinearVerts/3);

      if (homeLinearCoordBuffer == NULL) {
         homeLinearCoordBuffer = new GLfloat[homeLinearVerts*2];
      } else {
         delete [] homeLinearCoordBuffer;
         homeLinearCoordBuffer = new GLfloat[homeLinearVerts*2];
      }

      if (homeLinearColorBuffer == NULL) {
         homeLinearColorBuffer = new GLfloat[homeLinearVerts*3];
      } else {
         delete [] homeLinearColorBuffer;
         homeLinearColorBuffer = new GLfloat[homeLinearVerts*3];
      }

      if (homeLinearIndices == NULL) {
         homeLinearIndices = new GLushort[homeLinearVerts];
      } else {
         delete [] homeLinearIndices;
         homeLinearIndices = new GLushort[homeLinearVerts];
      }

      for (unsigned int i = 0; i < homeLinearVerts; i++) {
         homeLinearCoordBuffer[i*2+0] = verts[i*2+0];
         homeLinearCoordBuffer[i*2+1] = verts[i*2+1];
         homeLinearColorBuffer[i*3+0]  = colrs[i*3+0];
         homeLinearColorBuffer[i*3+1]  = colrs[i*3+1];
         homeLinearColorBuffer[i*3+2]  = colrs[i*3+2];
         homeLinearIndices[i]          = i;
      }

      // Calculate initial transformation matrix
      Matrix Ortho;
      Matrix ModelView;
      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixLoadIdentity( &ModelView );
      MatrixScale( &ModelView, 0.5f, 0.5f, 1.0f );
      MatrixRotate( &ModelView, 180-ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &homeLinearMVP, &ModelView, &Ortho );
      homeLinearPrevState.ao = ao;

      homeLinearPrevState.ao = ao;
      prevHomeLinearNumbulbs = numBulbs;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (homeLinearFirstRun == GL_TRUE) {
         homeLinearFirstRun = GL_FALSE;
         glGenBuffers(1, &homeLinearVBO);
      } else {
         glDeleteBuffers(1, &homeLinearVBO);
         glGenBuffers(1, &homeLinearVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, homeLinearVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*homeLinearVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLuint64 bytesOffset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, bytesOffset, sizeof(GLfloat)*2*homeLinearVerts, homeLinearCoordBuffer);
      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLuint64*)bytesOffset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*homeLinearVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*homeLinearVerts, homeLinearColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (GLuint64*)bytesOffset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   } 
   // Geometry already calculated, check if any colors need to be updated.
   for (int i = 0; i < 3; i++) {
      for (int j = 0; j < numBulbs; j++) {
         // 3*2*3:
         // 3 (R,G,B) color values per vertex
         // 2 Triangles per Quad
         // 3 Vertices per Triangle
         if (bulbColors[i+j*3] != homeLinearColorBuffer[i + j*(60/numBulbs)*9*2 ] || 
               prevHomeLinearNumbulbs != numBulbs) {
            for (int k = 0; k < (60/numBulbs)*3*2; k++) {  
               if (bulbColors[i+j*3] != homeLinearColorBuffer[i + k*3 + j*(60/numBulbs)*9*2 ]) {
                  homeLinearColorBuffer[ j*(60/numBulbs)*9*2 + k*3 + i ] = bulbColors[i+j*3];
               }
            }
            // Update Contents of VBO
            // Set active VBO
            glBindBuffer(GL_ARRAY_BUFFER, homeLinearVBO);
            // Convenience variable
            GLuint64 offset = 2*sizeof(GLfloat)*homeLinearVerts;
            // Load Vertex Color data into VBO
            glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*homeLinearVerts, homeLinearColorBuffer);
         }
      }
   }

   prevHomeLinearNumbulbs = numBulbs;
   delete [] bulbColors;

   // Update Transfomation Matrix if any change in parameters
   if (  homeLinearPrevState.ao != ao                       ||
         homeLinearPrevState.ao != homeLinearPrevState.ao   ){
      Matrix Ortho;
      Matrix ModelView;
      float left = -w2h, right = w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixLoadIdentity( &ModelView );

      MatrixScale( &ModelView, 0.5f, 0.5f, 1.0f );
      MatrixRotate( &ModelView, 180-ao, 0.0f, 0.0f, 1.0f);

      MatrixMultiply( &homeLinearMVP, &ModelView, &Ortho );
      homeLinearPrevState.ao = ao;
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &homeLinearMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, homeLinearVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*homeLinearVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, homeLinearVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   Py_RETURN_NONE;
}

/*
 * Explanation of features:
 * <= 0: just the color representation
 * <= 1: color representation + outline
 * <= 2: color representation + outline + bulb markers
 * <= 3: color representation + outline + bulb markers + bulb marker halos
 * <= 4: color representation + outline + bulb markers + bulb marker halos + grand halo
 */

GLfloat     *iconLinearCoordBuffer  = NULL; // Stores (X, Y) (float) for each vertex
GLfloat     *iconLinearColorBuffer  = NULL; // Stores (R, G, B) (float) for each vertex
GLushort    *iconLinearIndices      = NULL; // Stores index corresponding to each vertex
GLfloat     *iconLinearBulbVertices = NULL;
GLuint      iconLinearVerts;
GLint       prevIconLinearNumBulbs;
GLint       prevIconLinearFeatures;
Matrix      iconLinearMVP;                  // Transformation matrix passed to shader
Params      iconLinearPrevState;            // Stores transformations to avoid redundant recalculation
GLuint      iconLinearVBO;                  // Vertex Buffer Object ID
GLboolean   iconLinearFirstRun = GL_TRUE;   // Determines if function is running for the first time (for VBO initialization)

PyObject* drawIconLinear_drawArn(PyObject *self, PyObject *args) {
   PyObject*   detailColorPyTup;
   PyObject*   py_list;
   PyObject*   py_tuple;
   PyObject*   py_float;
   GLfloat*    bulbColors;
   GLfloat     detailColor[3];
   GLfloat     gx, gy, scale, ao, w2h; 
   GLfloat     R, G, B;
   long        numBulbs, features;
   GLint       vertIndex = 0;
   if (!PyArg_ParseTuple(args,
            "ffflOlffO",
            &gx, &gy,
            &scale, 
            &features,
            &detailColorPyTup,
            &numBulbs,
            &ao,
            &w2h,
            &py_list
            ))
   {
      Py_RETURN_NONE;
   }

   char circleSegments = 20;

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new float[numBulbs*3];
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = float(PyFloat_AsDouble(py_float));
      }
   }

   // Parse RGB detail colors
   detailColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0)));
   detailColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1)));
   detailColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2)));

   // Allocate and Define Geometry/Color buffers
   if (iconLinearCoordBuffer  == NULL     ||
       iconLinearColorBuffer  == NULL     ||
       iconLinearIndices      == NULL     ){

      printf("Generating geometry for iconLinear\n");
      vector<GLfloat> markerVerts;
      vector<GLfloat> markerColrs;
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy, tmx, tmy, ri, ro;
      float offset = float(2.0/60.0);
      float degSegment = float(360.0/float(circleSegments));
      float delta = float(degSegment/4.0);

      drawEllipse(float(0.0), float(0.0), float(0.16), circleSegments, detailColor, markerVerts, markerColrs);
      drawHalo(float(0.0), float(0.0), float(0.22), float(0.22), float(0.07), circleSegments, detailColor, markerVerts, markerColrs);

      // Safely (Re)allocate memory for bulb marker vertices
      if (iconLinearBulbVertices == NULL) {
         iconLinearBulbVertices = new GLfloat[markerVerts.size()];
      } else {
         delete [] iconLinearBulbVertices;
         iconLinearBulbVertices = new GLfloat[markerVerts.size()];
      }

      for (unsigned int i = 0; i < markerVerts.size()/2; i++) {
         iconLinearBulbVertices[i*2+0] = markerVerts[i*2+0];
         iconLinearBulbVertices[i*2+1] = markerVerts[i*2+1];
      }

      // Define Square of Stripes with Rounded Corners
      int tmb = 0;
      for (int i = 0; i < 60; i++) {
         if (i%10 == 0) {
            tmb++;
         }
         R = bulbColors[tmb*3+0];
         G = bulbColors[tmb*3+1];
         B = bulbColors[tmb*3+2];

         // Define end-slice with rounded corners
         if (i == 0) {
            TLx = -0.75;
            TLy =  1.00;

            BLx = -0.75;
            BLy = -1.00;

            /* X */ verts.push_back(-1.00);   /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back(-1.00);   /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back(-0.75);   /* Y */ verts.push_back( 0.75);

            /* X */ verts.push_back(-0.75);   /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back(-1.00);   /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back(-0.75);   /* Y */ verts.push_back(-0.75);

            // Defines Rounded Corners
            for (int j = 0; j < circleSegments; j++) {
               /* X */ verts.push_back(-0.75);
               /* Y */ verts.push_back( 0.75);
               /* X */ verts.push_back(float(-0.75 + 0.25*cos(degToRad(90+j*delta))));
               /* Y */ verts.push_back(float( 0.75 + 0.25*sin(degToRad(90+j*delta))));
               /* X */ verts.push_back(float(-0.75 + 0.25*cos(degToRad(90+(j+1)*delta))));
               /* Y */ verts.push_back(float( 0.75 + 0.25*sin(degToRad(90+(j+1)*delta))));

               /* X */ verts.push_back(-0.75);
               /* Y */ verts.push_back(-0.75);
               /* X */ verts.push_back(float(-0.75 + 0.25*cos(degToRad(180+j*delta))));
               /* Y */ verts.push_back(float(-0.75 + 0.25*sin(degToRad(180+j*delta))));
               /* X */ verts.push_back(float(-0.75 + 0.25*cos(degToRad(180+(j+1)*delta))));
               /* Y */ verts.push_back(float(-0.75 + 0.25*sin(degToRad(180+(j+1)*delta))));
               for (int j = 0; j < 6; j++) {
                  /* R */ colrs.push_back(R);
                  /* G */ colrs.push_back(G);
                  /* B */ colrs.push_back(B);
               }
            }

            for (int j = 0; j < 6; j++) {
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
            }
         } else {
            TLx = float(-1.0 + i*offset);
            TLy =  1.0;

            BLx = float(-1.0 + i*offset);
            BLy = -1.0;
         }

         // Define end-slice with rounded corners
         if (i == 60-1) {
            TRx =  0.75;
            TRy =  1.00;

            BRx =  0.75;
            BRy = -1.00;
            /* X */ verts.push_back( 1.00);   /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back( 1.00);   /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back( 0.75);   /* Y */ verts.push_back( 0.75);

            /* X */ verts.push_back( 0.75);   /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back( 1.00);   /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back( 0.75);   /* Y */ verts.push_back(-0.75);

            // Defines Rounded Corners
            for (int j = 0; j < circleSegments; j++) {
               /* X */ verts.push_back( 0.75);
               /* Y */ verts.push_back( 0.75);
               /* X */ verts.push_back(float( 0.75 + 0.25*cos(degToRad(j*delta))));
               /* Y */ verts.push_back(float( 0.75 + 0.25*sin(degToRad(j*delta))));
               /* X */ verts.push_back(float( 0.75 + 0.25*cos(degToRad((j+1)*delta))));
               /* Y */ verts.push_back(float( 0.75 + 0.25*sin(degToRad((j+1)*delta))));

               /* X */ verts.push_back( 0.75);
               /* Y */ verts.push_back(-0.75);
               /* X */ verts.push_back(float( 0.75 + 0.25*cos(degToRad(270+j*delta))));
               /* Y */ verts.push_back(float(-0.75 + 0.25*sin(degToRad(270+j*delta))));
               /* X */ verts.push_back(float( 0.75 + 0.25*cos(degToRad(270+(j+1)*delta))));
               /* Y */ verts.push_back(float(-0.75 + 0.25*sin(degToRad(270+(j+1)*delta))));
               for (int j = 0; j < 6; j++) {
                  /* R */ colrs.push_back(R);
                  /* G */ colrs.push_back(G);
                  /* B */ colrs.push_back(B);
               }
            }
            for (int j = 0; j < 6; j++) {
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
            }
         } else {
            TRx = float(-1.0 + (i+1)*offset);
            TRy =  1.0;

            BRx = float(-1.0 + (i+1)*offset);
            BRy = -1.0;
         }

         // Draw normal rectangular strip for non-end segments
         /* X */ verts.push_back(constrain(TLx, -0.75, 0.75));   /* Y */ verts.push_back(TLy);
         /* X */ verts.push_back(constrain(BLx, -0.75, 0.75));   /* Y */ verts.push_back(BLy);
         /* X */ verts.push_back(constrain(TRx, -0.75, 0.75));   /* Y */ verts.push_back(TRy);

         /* X */ verts.push_back(constrain(TRx, -0.75, 0.75));   /* Y */ verts.push_back(TRy);
         /* X */ verts.push_back(constrain(BLx, -0.75, 0.75));   /* Y */ verts.push_back(BLy);
         /* X */ verts.push_back(constrain(BRx, -0.75, 0.75));   /* Y */ verts.push_back(BRy);

         for (int j = 0; j < 6; j++) {
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
         }
      }

      R = detailColor[0];
      G = detailColor[1];
      B = detailColor[2];

      // Define OutLine
      if (features >= 1) {
         tmx = 0.0;
         tmy = 0.0;
      } else {
         tmx = offScreen; 
         tmy = offScreen;
      }

      /*
       * Draw Outer Straights
       */
      //---------//
      /* X */ verts.push_back(float(tmx - 9.0/8.0));   /* Y */ verts.push_back(float(tmy + 0.75));
      /* X */ verts.push_back(float(tmx - 9.0/8.0));   /* Y */ verts.push_back(float(tmy - 0.75));
      /* X */ verts.push_back(float(tmx - 1.00));      /* Y */ verts.push_back(float(tmy + 0.75));

      /* X */ verts.push_back(float(tmx - 1.00));      /* Y */ verts.push_back(float(tmy + 0.75));
      /* X */ verts.push_back(float(tmx - 9.0/8.0));   /* Y */ verts.push_back(float(tmy - 0.75));
      /* X */ verts.push_back(float(tmx - 1.00));      /* Y */ verts.push_back(float(tmy - 0.75));

      //---------//
      /* X */ verts.push_back(float(tmx + 9.0/8.0));   /* Y */ verts.push_back(float(tmy + 0.75));
      /* X */ verts.push_back(float(tmx + 9.0/8.0));   /* Y */ verts.push_back(float(tmy - 0.75));
      /* X */ verts.push_back(float(tmx + 1.00));      /* Y */ verts.push_back(float(tmy + 0.75));

      /* X */ verts.push_back(float(tmx + 1.00));      /* Y */ verts.push_back(float(tmy + 0.75));
      /* X */ verts.push_back(float(tmx + 9.0/8.0));   /* Y */ verts.push_back(float(tmy - 0.75));
      /* X */ verts.push_back(float(tmx + 1.00));      /* Y */ verts.push_back(float(tmy - 0.75));

      //---------//
      /* X */ verts.push_back(float(tmx + 0.75));   /* Y */ verts.push_back(float(tmy - 9.0/8.0));
      /* X */ verts.push_back(float(tmx - 0.75));   /* Y */ verts.push_back(float(tmy - 9.0/8.0));
      /* X */ verts.push_back(float(tmx + 0.75));   /* Y */ verts.push_back(float(tmy - 1.00));

      /* X */ verts.push_back(float(tmx + 0.75));   /* Y */ verts.push_back(float(tmy - 1.00));
      /* X */ verts.push_back(float(tmx - 0.75));   /* Y */ verts.push_back(float(tmy - 9.0/8.0));
      /* X */ verts.push_back(float(tmx - 0.75));   /* Y */ verts.push_back(float(tmy - 1.00));

      //---------//
      /* X */ verts.push_back(float(tmx + 0.75));   /* Y */ verts.push_back(float(tmy + 9.0/8.0));
      /* X */ verts.push_back(float(tmx - 0.75));   /* Y */ verts.push_back(float(tmy + 9.0/8.0));
      /* X */ verts.push_back(float(tmx + 0.75));   /* Y */ verts.push_back(float(tmy + 1.00));

      /* X */ verts.push_back(float(tmx + 0.75));   /* Y */ verts.push_back(float(tmy + 1.00));
      /* X */ verts.push_back(float(tmx - 0.75));   /* Y */ verts.push_back(float(tmy + 9.0/8.0));
      /* X */ verts.push_back(float(tmx - 0.75));   /* Y */ verts.push_back(float(tmy + 1.00));
      for (int j = 0; j < 24; j++) {
         /* R */ colrs.push_back(R);
         /* G */ colrs.push_back(G);
         /* B */ colrs.push_back(B);
      }

      /*
       * Draw Rounded Corners
       */
      ri = 0.25;
      ro = 0.125 + 0.25;
      float tmo;
      if (features >= 1) {
         tmo = 0.0;
      } else {
         tmo = offScreen;
      }
      for (int i = 0; i < 4; i++) {
         switch(i) {
            case 0:
               tmx = float( 0.75 + tmo);
               tmy = float( 0.75 + tmo);
               break;
            case 1:
               tmx = float(-0.75 + tmo);
               tmy = float( 0.75 + tmo);
               break;
            case 2:
               tmx = float(-0.75 + tmo);
               tmy = float(-0.75 + tmo);
               break;
            case 3:
               tmx = float( 0.75 + tmo);
               tmy = float(-0.75 + tmo);
               break;
         }

         for (int j = 0; j < circleSegments; j++) {
            /* X */ verts.push_back(float(tmx + ri*cos(degToRad(i*90 + j*delta))));
            /* Y */ verts.push_back(float(tmy + ri*sin(degToRad(i*90 + j*delta))));
            /* X */ verts.push_back(float(tmx + ro*cos(degToRad(i*90 + j*delta))));
            /* Y */ verts.push_back(float(tmy + ro*sin(degToRad(i*90 + j*delta))));
            /* X */ verts.push_back(float(tmx + ri*cos(degToRad(i*90 + (j+1)*delta))));
            /* Y */ verts.push_back(float(tmy + ri*sin(degToRad(i*90 + (j+1)*delta))));

            /* X */ verts.push_back(float(tmx + ri*cos(degToRad(i*90 + (j+1)*delta))));
            /* Y */ verts.push_back(float(tmy + ri*sin(degToRad(i*90 + (j+1)*delta))));
            /* X */ verts.push_back(float(tmx + ro*cos(degToRad(i*90 + j*delta))));
            /* Y */ verts.push_back(float(tmy + ro*sin(degToRad(i*90 + j*delta))));
            /* X */ verts.push_back(float(tmx + ro*cos(degToRad(i*90 + (j+1)*delta))));
            /* Y */ verts.push_back(float(tmy + ro*sin(degToRad(i*90 + (j+1)*delta))));
            for (int k = 0; k < 6; k++) {
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
            }
         }
      }

      // Define Bulb Markers
      for (int i = 0; i < 6; i++) {
         if (features >= 2.0 && i < numBulbs) {
            if (numBulbs == 1) {
               tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs));
               tmy = (17.0/16.0);
            } else {
               tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs));
               tmy = -(17.0/16.0);
            }
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
         drawEllipse(tmx, tmy, float(1.0/6.0), circleSegments, detailColor, verts, colrs);
      }

      // Define Bulb Halos
      float limit = float(1.0/float(numBulbs));
      for (int i = 0; i < 6; i++) {
         if (features >= 3 && i < numBulbs) {
            tmo = 0.0;
         } else { 
            tmo = offScreen;
         }
         if (numBulbs == 1) {
            tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs)) + tmo;
            tmy = float( (17.0/16.0) + tmo);
         } else {
            tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs)) + tmo;
            tmy = float(-(17.0/16.0) + tmo);
         }
         for (int j = 0; j < circleSegments; j++) {
            if (i == 0) {
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  0], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  1]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  2], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  3]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  4], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  5]);

               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  6], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  7]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  8], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  9]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 + 10], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 + 11]);
            } else if (i == numBulbs-1) {
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  0], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  1]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  2], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  3]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  4], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  5]);

               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  6], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  7]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  8], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  9]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 + 10], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 + 11]);
            } else {
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  0], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  1]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  2], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  3]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  4], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  5]);

               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  6], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  7]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 +  8], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 +  9]);
               /* X */ verts.push_back(constrain(tmx + iconLinearBulbVertices[circleSegments*6 + j*12 + 10], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconLinearBulbVertices[circleSegments*6 + j*12 + 11]);
            }
         }

         for (int j = 0; j < circleSegments*3; j++) {
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
         }
      }

      // Define Grand Outline
      if (features >= 4) {
         tmo = 0.0;
      } else {
         tmo = offScreen;
      }

      /*
       * Draw Outer Straights
       */

      /* X */ verts.push_back(float(tmo-0.75));  /* Y */ verts.push_back(float(tmo+(17.0/16.0 + 17.0/60.0)));
      /* X */ verts.push_back(float(tmo-0.75));  /* Y */ verts.push_back(float(tmo+(17.0/16.0 + 13.0/60.0)));
      /* X */ verts.push_back(float(tmo+0.75));  /* Y */ verts.push_back(float(tmo+(17.0/16.0 + 17.0/60.0)));

      /* X */ verts.push_back(float(tmo+0.75));  /* Y */ verts.push_back(float(tmo+(17.0/16.0 + 13.0/60.0)));
      /* X */ verts.push_back(float(tmo+0.75));  /* Y */ verts.push_back(float(tmo+(17.0/16.0 + 17.0/60.0)));
      /* X */ verts.push_back(float(tmo-0.75));  /* Y */ verts.push_back(float(tmo+(17.0/16.0 + 13.0/60.0)));

      /* X */ verts.push_back(float(tmo-0.75));  /* Y */ verts.push_back(float(tmo-(17.0/16.0 + 17.0/60.0)));
      /* X */ verts.push_back(float(tmo-0.75));  /* Y */ verts.push_back(float(tmo-(17.0/16.0 + 13.0/60.0)));
      /* X */ verts.push_back(float(tmo+0.75));  /* Y */ verts.push_back(float(tmo-(17.0/16.0 + 17.0/60.0)));

      /* X */ verts.push_back(float(tmo+0.75));  /* Y */ verts.push_back(float(tmo-(17.0/16.0 + 13.0/60.0)));
      /* X */ verts.push_back(float(tmo+0.75));  /* Y */ verts.push_back(float(tmo-(17.0/16.0 + 17.0/60.0)));
      /* X */ verts.push_back(float(tmo-0.75));  /* Y */ verts.push_back(float(tmo-(17.0/16.0 + 13.0/60.0)));

      /* X */ verts.push_back(float(tmo+(17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back(float(tmo-0.75));
      /* X */ verts.push_back(float(tmo+(17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back(float(tmo-0.75));
      /* X */ verts.push_back(float(tmo+(17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back(float(tmo+0.75));

      /* X */ verts.push_back(float(tmo+(17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back(float(tmo+0.75));
      /* X */ verts.push_back(float(tmo+(17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back(float(tmo+0.75));
      /* X */ verts.push_back(float(tmo+(17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back(float(tmo-0.75));

      /* X */ verts.push_back(float(tmo-(17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back(float(tmo-0.75));
      /* X */ verts.push_back(float(tmo-(17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back(float(tmo-0.75));
      /* X */ verts.push_back(float(tmo-(17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back(float(tmo+0.75));

      /* X */ verts.push_back(float(tmo-(17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back(float(tmo+0.75));
      /* X */ verts.push_back(float(tmo-(17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back(float(tmo+0.75));
      /* X */ verts.push_back(float(tmo-(17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back(float(tmo-0.75));

      for (int j = 0; j < 24; j++) {
         /* R */ colrs.push_back(R);
         /* G */ colrs.push_back(G);
         /* B */ colrs.push_back(B);
      }

      /*
       * Draw Rounded Corners
       */
      ri = float(5.0/16.0+13.0/60.0);
      ro = float(5.0/16.0+17.0/60.0);
      delta = float(degSegment/4.0);
      for (int i = 0; i < 4; i++) {
         switch(i) {
            case 0:
               tmx = float( 0.75 + tmo);
               tmy = float( 0.75 + tmo);
               break;
            case 1:
               tmx = float(-0.75 + tmo);
               tmy = float( 0.75 + tmo);
               break;
            case 2:
               tmx = float(-0.75 + tmo);
               tmy = float(-0.75 + tmo);
               break;
            case 3:
               tmx = float( 0.75 + tmo);
               tmy = float(-0.75 + tmo);
            break;
         }

         for (int j = 0; j < circleSegments; j++) {
            float j0 = float(degToRad(i*90 + j*delta));
            float j1 = float(degToRad(i*90 + (j+1)*delta));
            /* X */ verts.push_back(float(tmx + ri*cos(j0)));  /* Y */ verts.push_back(float(tmy + ri*sin(j0)));
            /* X */ verts.push_back(float(tmx + ro*cos(j0)));  /* Y */ verts.push_back(float(tmy + ro*sin(j0)));
            /* X */ verts.push_back(float(tmx + ri*cos(j1)));  /* Y */ verts.push_back(float(tmy + ri*sin(j1)));

            /* X */ verts.push_back(float(tmx + ri*cos(j1)));  /* Y */ verts.push_back(float(tmy + ri*sin(j1)));
            /* X */ verts.push_back(float(tmx + ro*cos(j0)));  /* Y */ verts.push_back(float(tmy + ro*sin(j0)));
            /* X */ verts.push_back(float(tmx + ro*cos(j1)));  /* Y */ verts.push_back(float(tmy + ro*sin(j1)));
            for (int k = 0; k < 6; k++) {
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
            }
         }
      }

      iconLinearVerts = verts.size()/2;
      printf("iconLinear vertexBuffer length: %.i, Number of vertices: %.i, tris: %.i\n", iconLinearVerts*2, iconLinearVerts, iconLinearVerts/3);

      // Safely (Re)allocate memory for icon Vertex Buffer
      if (iconLinearCoordBuffer == NULL) {
         iconLinearCoordBuffer = new GLfloat[iconLinearVerts*2];
      } else {
         delete [] iconLinearCoordBuffer;
         iconLinearCoordBuffer = new GLfloat[iconLinearVerts*2];
      }

      // Safely (Re)allocate memory for icon Color Buffer
      if (iconLinearColorBuffer == NULL) {
         iconLinearColorBuffer = new GLfloat[iconLinearVerts*3];
      } else {
         delete [] iconLinearColorBuffer;
         iconLinearColorBuffer = new GLfloat[iconLinearVerts*3];
      }

      // Safely (Re)allocate memory for icon indices
      if (iconLinearIndices == NULL) {
         iconLinearIndices = new GLushort[iconLinearVerts];
      } else {
         delete [] iconLinearIndices;
         iconLinearIndices = new GLushort[iconLinearVerts];
      }

      for (unsigned int i = 0; i < iconLinearVerts; i++) {
         iconLinearCoordBuffer[i*2+0] = verts[i*2+0];
         iconLinearCoordBuffer[i*2+1] = verts[i*2+1];
         iconLinearColorBuffer[i*3+0]  = colrs[i*3+0];
         iconLinearColorBuffer[i*3+1]  = colrs[i*3+1];
         iconLinearColorBuffer[i*3+2]  = colrs[i*3+2];
         iconLinearIndices[i]          = i;
      }

      // Calculate initial transformation matrix
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
      MatrixRotate( &ModelView, 180-ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &iconLinearMVP, &ModelView, &Ortho );

      iconLinearPrevState.ao = ao;
      iconLinearPrevState.dx = gx;
      iconLinearPrevState.dy = gy;
      iconLinearPrevState.sx = scale;
      iconLinearPrevState.sy = scale;
      iconLinearPrevState.w2h = w2h;

      // Update State machine variables
      prevIconLinearNumBulbs = numBulbs;
      prevIconLinearFeatures = features;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (iconLinearFirstRun == GL_TRUE) {
         iconLinearFirstRun = GL_FALSE;
         glGenBuffers(1, &iconLinearVBO);
      } else {
         glDeleteBuffers(1, &iconLinearVBO);
         glGenBuffers(1, &iconLinearVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, iconLinearVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*iconLinearVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLuint64 bytesOffset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*iconLinearVerts, iconLinearCoordBuffer);
      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLuint64*)bytesOffset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*iconLinearVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*iconLinearVerts, iconLinearColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (GLuint64*)bytesOffset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   } 

   // Update features
   if (prevIconLinearFeatures != features ||
       prevIconLinearNumBulbs != numBulbs ){

      prevIconLinearFeatures = features;
      float tmx, tmy;
      //float degSegment = float(360.0/float(circleSegments));
      tmx = 0.0;
      tmy = 0.0;
      vertIndex = 0;

      // Define Square of Stripes with Rounded Corners
      for (int i = 0; i < 60; i++) {
         if (i == 0 || i == 60-1) {
            vertIndex += 12;

            // Defines Rounded Corners
            for (int j = 0; j < circleSegments; j++) {
               vertIndex += 12;
            }
         }

         // Draw normal rectangular strip for non-end segments
         vertIndex += 12;
      }

      // Define OutLine
      // Move outline on-screen if off-screen
      if (features >= 1) {
         if (iconLinearCoordBuffer[vertIndex+1] > offScreen/2) {
            tmx = -offScreen;
            tmy = -offScreen;
         } else {
            tmx = 0.0;
            tmy = 0.0;
         }
      } 
      // Move outline off-screen if on-screen
      else {
         if (iconLinearCoordBuffer[vertIndex+1] > offScreen/2) {
            tmx = 0.0;
            tmy = 0.0;
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
      }

      /*
       * Draw Outer Straights
       */
      for (int i = 0; i < 4; i++ ) {
         /* X */ iconLinearCoordBuffer[vertIndex +  0] = iconLinearCoordBuffer[vertIndex +  0] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex +  1] = iconLinearCoordBuffer[vertIndex +  1] + tmy;
         /* X */ iconLinearCoordBuffer[vertIndex +  2] = iconLinearCoordBuffer[vertIndex +  2] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex +  3] = iconLinearCoordBuffer[vertIndex +  3] + tmy;
         /* X */ iconLinearCoordBuffer[vertIndex +  4] = iconLinearCoordBuffer[vertIndex +  4] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex +  5] = iconLinearCoordBuffer[vertIndex +  5] + tmy;

         /* X */ iconLinearCoordBuffer[vertIndex +  6] = iconLinearCoordBuffer[vertIndex +  6] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex +  7] = iconLinearCoordBuffer[vertIndex +  7] + tmy;
         /* X */ iconLinearCoordBuffer[vertIndex +  8] = iconLinearCoordBuffer[vertIndex +  8] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex +  9] = iconLinearCoordBuffer[vertIndex +  9] + tmy;
         /* X */ iconLinearCoordBuffer[vertIndex + 10] = iconLinearCoordBuffer[vertIndex + 10] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex + 11] = iconLinearCoordBuffer[vertIndex + 11] + tmy;
         vertIndex += 12;
      }

      /*
       * Draw Rounded Corners
       */
      for (int i = 0; i < 4; i++) {
         for (int j = 0; j < circleSegments; j++) {
            /* X */ iconLinearCoordBuffer[vertIndex +  0] = iconLinearCoordBuffer[vertIndex +  0] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex +  1] = iconLinearCoordBuffer[vertIndex +  1] + tmy;
            /* X */ iconLinearCoordBuffer[vertIndex +  2] = iconLinearCoordBuffer[vertIndex +  2] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex +  3] = iconLinearCoordBuffer[vertIndex +  3] + tmy;
            /* X */ iconLinearCoordBuffer[vertIndex +  4] = iconLinearCoordBuffer[vertIndex +  4] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex +  5] = iconLinearCoordBuffer[vertIndex +  5] + tmy;

            /* X */ iconLinearCoordBuffer[vertIndex +  6] = iconLinearCoordBuffer[vertIndex +  6] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex +  7] = iconLinearCoordBuffer[vertIndex +  7] + tmy;
            /* X */ iconLinearCoordBuffer[vertIndex +  8] = iconLinearCoordBuffer[vertIndex +  8] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex +  9] = iconLinearCoordBuffer[vertIndex +  9] + tmy;
            /* X */ iconLinearCoordBuffer[vertIndex + 10] = iconLinearCoordBuffer[vertIndex + 10] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex + 11] = iconLinearCoordBuffer[vertIndex + 11] + tmy;
            vertIndex += 12;
         }
      }

      // Define Bulb Markers
      for (int i = 0; i < 6; i++) {
         if (features >= 2 && i < numBulbs) {
            if (numBulbs == 1) {
               tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs));
               tmy = (17.0/16.0);
            } else {
               tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs));
               tmy = -(17.0/16.0);
            }
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
//#        pragma omp parallel for
         for (int j = 0; j < circleSegments; j++) {
            /* X */ iconLinearCoordBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 0] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 1] + tmy;
            /* X */ iconLinearCoordBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 2] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 3] + tmy;
            /* X */ iconLinearCoordBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 4] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 5] + tmy;
         }
      }

      // Define Bulb Halos
      float limit;
      for (int i = 0; i < 6; i++) {
         if (features >= 3 && i < numBulbs) {
            if (numBulbs == 1) {
               tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs));
               tmy = (17.0/16.0);
            } else {
               tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs));
               tmy = -(17.0/16.0);
            }
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
         limit = float(1.0/float(numBulbs));
         int tmj;
         for (int j = 0; j < circleSegments; j++) {
            tmj = 6*circleSegments + j*12;
            if (i == 0) {
               /* X */ iconLinearCoordBuffer[vertIndex +  0] = constrain( tmx + iconLinearBulbVertices[  0 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex +  1] =            tmy + iconLinearBulbVertices[  1 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex +  2] = constrain( tmx + iconLinearBulbVertices[  2 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex +  3] =            tmy + iconLinearBulbVertices[  3 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex +  4] = constrain( tmx + iconLinearBulbVertices[  4 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex +  5] =            tmy + iconLinearBulbVertices[  5 + tmj];

               /* X */ iconLinearCoordBuffer[vertIndex +  6] = constrain( tmx + iconLinearBulbVertices[  6 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex +  7] =            tmy + iconLinearBulbVertices[  7 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex +  8] = constrain( tmx + iconLinearBulbVertices[  8 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex +  9] =            tmy + iconLinearBulbVertices[  9 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex + 10] = constrain( tmx + iconLinearBulbVertices[ 10 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex + 11] =            tmy + iconLinearBulbVertices[ 11 + tmj];
               vertIndex += 12;
            } else if (i == numBulbs-1) {
               /* X */ iconLinearCoordBuffer[vertIndex +  0] = constrain( tmx + iconLinearBulbVertices[  0 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearCoordBuffer[vertIndex +  1] =            tmy + iconLinearBulbVertices[  1 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex +  2] = constrain( tmx + iconLinearBulbVertices[  2 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearCoordBuffer[vertIndex +  3] =            tmy + iconLinearBulbVertices[  3 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex +  4] = constrain( tmx + iconLinearBulbVertices[  4 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearCoordBuffer[vertIndex +  5] =            tmy + iconLinearBulbVertices[  5 + tmj];

               /* X */ iconLinearCoordBuffer[vertIndex +  6] = constrain( tmx + iconLinearBulbVertices[  6 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearCoordBuffer[vertIndex +  7] =            tmy + iconLinearBulbVertices[  7 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex +  8] = constrain( tmx + iconLinearBulbVertices[  8 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearCoordBuffer[vertIndex +  9] =            tmy + iconLinearBulbVertices[  9 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex + 10] = constrain( tmx + iconLinearBulbVertices[ 10 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearCoordBuffer[vertIndex + 11] =            tmy + iconLinearBulbVertices[ 11 + tmj];
               vertIndex += 12;
            } else {
               /* X */ iconLinearCoordBuffer[vertIndex +  0] = constrain( tmx + iconLinearBulbVertices[  0 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex +  1] =            tmy + iconLinearBulbVertices[  1 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex +  2] = constrain( tmx + iconLinearBulbVertices[  2 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex +  3] =            tmy + iconLinearBulbVertices[  3 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex +  4] = constrain( tmx + iconLinearBulbVertices[  4 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex +  5] =            tmy + iconLinearBulbVertices[  5 + tmj];

               /* X */ iconLinearCoordBuffer[vertIndex +  6] = constrain( tmx + iconLinearBulbVertices[  6 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex +  7] =            tmy + iconLinearBulbVertices[  7 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex +  8] = constrain( tmx + iconLinearBulbVertices[  8 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex +  9] =            tmy + iconLinearBulbVertices[  9 + tmj];
               /* X */ iconLinearCoordBuffer[vertIndex + 10] = constrain( tmx + iconLinearBulbVertices[ 10 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearCoordBuffer[vertIndex + 11] =            tmy + iconLinearBulbVertices[ 11 + tmj];
               vertIndex += 12;
            }
         }
      }

      // Define Grand Outline
      if (features >= 4) {
         if (iconLinearCoordBuffer[vertIndex] > offScreen/2) {
            tmx = -offScreen;
            tmy = -offScreen;
         } else {
            tmx = 0.0;
            tmy = 0.0;
         }
      } else {
         if (iconLinearCoordBuffer[vertIndex] > offScreen/2) {
            tmx = 0.0;
            tmy = 0.0;
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
      }

      /*
       * Draw Outer Straights
       */

      for (int i = 0; i < 4; i++ ) {
         /* X */ iconLinearCoordBuffer[vertIndex +  0] = iconLinearCoordBuffer[vertIndex +  0] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex +  1] = iconLinearCoordBuffer[vertIndex +  1] + tmy;
         /* X */ iconLinearCoordBuffer[vertIndex +  2] = iconLinearCoordBuffer[vertIndex +  2] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex +  3] = iconLinearCoordBuffer[vertIndex +  3] + tmy;
         /* X */ iconLinearCoordBuffer[vertIndex +  4] = iconLinearCoordBuffer[vertIndex +  4] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex +  5] = iconLinearCoordBuffer[vertIndex +  5] + tmy;

         /* X */ iconLinearCoordBuffer[vertIndex +  6] = iconLinearCoordBuffer[vertIndex +  6] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex +  7] = iconLinearCoordBuffer[vertIndex +  7] + tmy;
         /* X */ iconLinearCoordBuffer[vertIndex +  8] = iconLinearCoordBuffer[vertIndex +  8] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex +  9] = iconLinearCoordBuffer[vertIndex +  9] + tmy;
         /* X */ iconLinearCoordBuffer[vertIndex + 10] = iconLinearCoordBuffer[vertIndex + 10] + tmx;
         /* Y */ iconLinearCoordBuffer[vertIndex + 11] = iconLinearCoordBuffer[vertIndex + 11] + tmy;
         vertIndex += 12;
      }

      /*
       * Draw Rounded Corners
       */
      for (int i = 0; i < 4; i++) {
         for (int j = 0; j < circleSegments; j++) {
            /* X */ iconLinearCoordBuffer[vertIndex +  0] = iconLinearCoordBuffer[vertIndex +  0] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex +  1] = iconLinearCoordBuffer[vertIndex +  1] + tmy;
            /* X */ iconLinearCoordBuffer[vertIndex +  2] = iconLinearCoordBuffer[vertIndex +  2] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex +  3] = iconLinearCoordBuffer[vertIndex +  3] + tmy;
            /* X */ iconLinearCoordBuffer[vertIndex +  4] = iconLinearCoordBuffer[vertIndex +  4] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex +  5] = iconLinearCoordBuffer[vertIndex +  5] + tmy;

            /* X */ iconLinearCoordBuffer[vertIndex +  6] = iconLinearCoordBuffer[vertIndex +  6] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex +  7] = iconLinearCoordBuffer[vertIndex +  7] + tmy;
            /* X */ iconLinearCoordBuffer[vertIndex +  8] = iconLinearCoordBuffer[vertIndex +  8] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex +  9] = iconLinearCoordBuffer[vertIndex +  9] + tmy;
            /* X */ iconLinearCoordBuffer[vertIndex + 10] = iconLinearCoordBuffer[vertIndex + 10] + tmx;
            /* Y */ iconLinearCoordBuffer[vertIndex + 11] = iconLinearCoordBuffer[vertIndex + 11] + tmy;
            vertIndex += 12;
         }
      }

      prevIconLinearFeatures = features;
      // Update Contents of VBO
      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, iconLinearVBO);
      // Convenience variable
      GLuint64 offset = 0;
      // Load Vertex Color data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*iconLinearVerts, iconLinearCoordBuffer);
   }

   // Geometry allocated/calculated, check if colors need to be updated
   for (int i = 0; i < 3; i++) {
      for (int j = 0; j < numBulbs; j++) {
         float tmc = float(bulbColors[j*3+i]);
         int tmp = (j*(60/numBulbs)*3*2 + circleSegments*3*2 + 6);
         // Special Case for Rounded Corner Segments
         if (j == 0) {
            if (  tmc != iconLinearColorBuffer[i]     || 
                  prevIconLinearNumBulbs != numBulbs  ){
               for (int k = 0; k < tmp*3; k++) {
                  if (tmc != iconLinearColorBuffer[i + k*3]) {
                     iconLinearColorBuffer[i + k*3] = tmc;
                  }
               }
               // Update Contents of VBO
               // Set active VBO
               glBindBuffer(GL_ARRAY_BUFFER, iconLinearVBO);
               // Convenience variable
               GLuint64 offset = 2*sizeof(GLfloat)*iconLinearVerts;
               // Load Vertex Color data into VBO
               glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*iconLinearVerts, iconLinearColorBuffer);
            }
         } 
   
         // Special Case for Rounded Corner Segments
         if (j == numBulbs-1) {

            // Fixes edge-case bug
            if (numBulbs == 1)
               tmp *= 3;

            if (  tmc != iconLinearColorBuffer[i + tmp*3] || 
                  prevIconLinearNumBulbs != numBulbs ) {
               for (int k = 0; k < ((60/numBulbs)*3*2 + 2*3*circleSegments + 2*3); k++) {
                  if (tmc != iconLinearColorBuffer[i + k*3 + tmp*3] ) {
                     iconLinearColorBuffer[i + k*3 + tmp*3] = tmc;
                  }
               }
               // Update Contents of VBO
               // Set active VBO
               glBindBuffer(GL_ARRAY_BUFFER, iconLinearVBO);
               // Convenience variable
               GLuint64 offset = 2*sizeof(GLfloat)*iconLinearVerts;
               // Load Vertex Color data into VBO
               glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*iconLinearVerts, iconLinearColorBuffer);
            }
         } 
         else
         // General Case for middle segments
         {
            if (  tmc != iconLinearColorBuffer[i + (j*(60/numBulbs)*3*2 + circleSegments*3*2 + 6)*3] || 
                  prevIconLinearNumBulbs != numBulbs) {
               for (int k = 0; k < (60/numBulbs)*3*2; k++) {
                  if (tmc != iconLinearColorBuffer[i + k*3 + (j*(60/numBulbs)*3*2 + circleSegments*3*2 + 6)*3] ) {
                     iconLinearColorBuffer[i + k*3 + (j*(60/numBulbs)*3*2 + circleSegments*3*2 + 6)*3] = tmc;
                  }
               }
               // Update Contents of VBO
               // Set active VBO
               glBindBuffer(GL_ARRAY_BUFFER, iconLinearVBO);
               // Convenience variable
               GLuint64 offset = 2*sizeof(GLfloat)*iconLinearVerts;
               // Load Vertex Color data into VBO
               glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*iconLinearVerts, iconLinearColorBuffer);
            }
         }
      }

      // Check if detail color needs to be updated
      if (float(detailColor[i]) != iconLinearColorBuffer[i+(60*2*3 + 4*circleSegments*3 + 2*6)*3]) {
         for (unsigned int k = (60*2*3 + 4*circleSegments*3 + 2*6); k < iconLinearVerts; k++) {
            iconLinearColorBuffer[k*3+i] = float(detailColor[i]);
         }
         // Update Contents of VBO
         // Set active VBO
         glBindBuffer(GL_ARRAY_BUFFER, iconLinearVBO);
         // Convenience variable
         GLuint64 offset = 2*sizeof(GLfloat)*iconLinearVerts;
         // Load Vertex Color data into VBO
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*iconLinearVerts, iconLinearColorBuffer);
      }
   }
   prevIconLinearNumBulbs = numBulbs;
   
   delete [] bulbColors;

   // Update Transfomation Matrix if any change in parameters
   if (  iconLinearPrevState.ao != ao     ||
         iconLinearPrevState.dx != gx     ||
         iconLinearPrevState.dy != gy     ||
         iconLinearPrevState.sx != scale  ||
         iconLinearPrevState.sy != scale  ||
         iconLinearPrevState.w2h != w2h   ){
      
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
      MatrixRotate( &ModelView, 180-ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &iconLinearMVP, &ModelView, &Ortho );

      iconLinearPrevState.ao = ao;
      iconLinearPrevState.dx = gx;
      iconLinearPrevState.dy = gy;
      iconLinearPrevState.sx = scale;
      iconLinearPrevState.sy = scale;
      iconLinearPrevState.w2h = w2h;
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &iconLinearMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, iconLinearVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*iconLinearVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, iconLinearVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   Py_RETURN_NONE;
}
