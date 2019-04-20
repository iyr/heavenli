#include <Python.h>
#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glext.h>
//#include "matrixUtils.h"
#include <vector>
#include <math.h>
using namespace std;

GLfloat  *homeCircleVertexBuffer       = NULL;
GLfloat  *homeCircleColorBuffer        = NULL;
GLushort *homeCircleIndices            = NULL;
GLuint   homeCircleVerts;
GLuint   homeCircleBuffer;
GLuint   homeCircleIBO;
GLint    prevHomeCircleNumBulbs;
GLint    attribVertexPosition;
GLint    attribVertexColor;
Matrix MVP;
//GLfloat  ModelViewMatrix[4][4];
//GLfloat  OrthoMatrix[4][4];

PyObject* drawHomeCircle_drawArn(PyObject *self, PyObject *args) {
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   float *bulbColors;
   float gx, gy, wx, wy, ao, w2h; 
   float R, G, B;
   int numBulbs;

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
   //char circleSegments = 60/numBulbs;
   char circleSegments = 60;

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new float[numBulbs*3];
//#  pragma omp parallel for
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = float(PyFloat_AsDouble(py_float));
      }
   }

   if (homeCircleVertexBuffer == NULL  ||
       homeCircleColorBuffer  == NULL  ||
       homeCircleIndices      == NULL  ){

      printf("Generating geometry for homeCircle\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      //char degSegment = 360 / circleSegments;
      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma;
      
      for (int j = 0; j < numBulbs; j++) {
         R = float(bulbColors[j*3+0]);
         G = float(bulbColors[j*3+1]);
         B = float(bulbColors[j*3+2]);
//#        pragma omp parallel for
         for (int i = 0; i < circleSegments/numBulbs; i++) {
            /* X */ verts.push_back(float(0.0));
            /* Y */ verts.push_back(float(0.0));

            tma = float(degToRad(i*float(degSegment) + j*angOffset - 90.0));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));

            tma = float(degToRad((i+1)*float(degSegment) + j*angOffset - 90.0));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));

            /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
            /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
            /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
         }
      }

      homeCircleVerts = verts.size()/2;
      printf("homeCircle vertexBuffer length: %.i, Number of vertices: %.i, tris: %.i\n", homeCircleVerts*2, homeCircleVerts, homeCircleVerts/3);

      if (homeCircleVertexBuffer == NULL) {
         homeCircleVertexBuffer = new GLfloat[homeCircleVerts*2];
      } else {
         delete [] homeCircleVertexBuffer;
         homeCircleVertexBuffer = new GLfloat[homeCircleVerts*2];
      }

      if (homeCircleColorBuffer == NULL) {
         homeCircleColorBuffer = new GLfloat[homeCircleVerts*3];
      } else {
         delete [] homeCircleColorBuffer;
         homeCircleColorBuffer = new GLfloat[homeCircleVerts*3];
      }

      if (homeCircleIndices == NULL) {
         homeCircleIndices = new GLushort[homeCircleVerts];
      } else {
         delete [] homeCircleIndices;
         homeCircleIndices = new GLushort[homeCircleVerts];
      }

//#     pragma omp parallel for
      for (unsigned int i = 0; i < homeCircleVerts; i++) {
         homeCircleVertexBuffer[i*2+0] = verts[i*2+0];
         homeCircleVertexBuffer[i*2+1] = verts[i*2+1];
         homeCircleColorBuffer[i*3+0]  = colrs[i*3+0];
         homeCircleColorBuffer[i*3+1]  = colrs[i*3+1];
         homeCircleColorBuffer[i*3+2]  = colrs[i*3+2];
         homeCircleIndices[i]          = i;
      }

      prevHomeCircleNumBulbs = numBulbs;

      //glScalef(sqrt(w2h)*hypot(wx, wy), sqrt(wy/wx)*hypot(wx, wy), 1.0);
      //glRotatef(ao, 0, 0, 1);
      /*
      // Set Identity Matrix
      memset(ModelViewMatrix, 0x0, sizeof(ModelViewMatrix));
      ModelViewMatrix[0][0] = 1.0f;
      ModelViewMatrix[1][1] = 1.0f;
      ModelViewMatrix[2][2] = 1.0f;
      ModelViewMatrix[3][3] = 1.0f;

      // Set Identity Matrix
      memset(OrthoMatrix, 0x0, sizeof(OrthoMatrix));
      OrthoMatrix[0][0] = 1.0f;
      OrthoMatrix[1][1] = 1.0f;
      OrthoMatrix[2][2] = 1.0f;
      OrthoMatrix[3][3] = 1.0f;

      float deltaX = right*w2h - left*w2h;
      float deltaY = top - bottom;
      float deltaZ = far - near;

      OrthoMatrix[0][0] = 2.0f / deltaX;
      OrthoMatrix[3][0] = -(right - left) / deltaX;
      OrthoMatrix[1][1] = 2.0f / deltaY;
      OrthoMatrix[3][1] = -(top - bottom) / deltaY;
      OrthoMatrix[2][2] = 2.0f / deltaZ;
      OrthoMatrix[3][2] = -(near - far)   / deltaZ;
      */
      
   } 
   // Geometry already calculated, update colors
   /*
    * Iterate through each color channel 
    * 0 - RED
    * 1 - GREEN
    * 2 - BLUE
    */
   for (int i = 0; i < 3; i++) {
         
      // Update color, if needed
      for (int j = 0; j < numBulbs; j++) {
         if (float(bulbColors[ i + j*3 ]) != homeCircleColorBuffer[ i + j*(60/numBulbs)*9 ] ||
               prevHomeCircleNumBulbs != numBulbs) {
//#           pragma omp parallel for
            for (int k = 0; k < (60/numBulbs)*3; k++) {
               if (float(bulbColors[ i + j*3 ]) != homeCircleColorBuffer[ i + k*3 + j*(60/numBulbs)*9 ]) {
                  homeCircleColorBuffer[ j*(60/numBulbs)*9 + k*3 + i ] = float(bulbColors[i+j*3]);
               }
            }
         }
      }
   }

   prevHomeCircleNumBulbs = numBulbs;
   delete [] bulbColors;

   /*
   glPushMatrix();
   glScalef(sqrt(w2h)*hypot(wx, wy), sqrt(wy/wx)*hypot(wx, wy), 1.0);
   glRotatef(ao, 0, 0, 1);
   glColorPointer(3, GL_FLOAT, 0, homeCircleColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, homeCircleVertexBuffer);
   glDrawElements( GL_TRIANGLES, homeCircleVerts, GL_UNSIGNED_SHORT, homeCircleIndices);
   glPopMatrix();
   */
   
   Matrix Ortho;
   Matrix ModelView;

   float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
   MatrixLoadIdentity( &Ortho );
   MatrixOrtho( &Ortho, left, right, bottom, top, near, far );

   MatrixLoadIdentity( &ModelView );

   MatrixScale( &ModelView, 2.0f, 2.0f , 1.0f );
   MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);

   MatrixMultiply( &MVP, &ModelView, &Ortho );

   GLint mvpLoc;
   mvpLoc = glGetUniformLocation( 3, "MVP" );
   glUniformMatrix4fv( mvpLoc, 1, GL_FALSE, &MVP.mat[0][0] );
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, homeCircleVertexBuffer);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, homeCircleColorBuffer);
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, homeCircleVerts);
   //glDisableVertexAttribArray(0);
   //glDisableVertexAttribArray(1);

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
GLfloat  *iconCircleVertexBuffer = NULL;
GLfloat  *iconCircleColorBuffer  = NULL;
GLushort *iconCircleIndices      = NULL;
GLfloat  *iconCircleBulbVertices = NULL;
GLuint   iconCircleVerts;
int      prevIconCircleNumBulbs;
int      prevIconCircleFeatures;
extern float offScreen;

PyObject* drawIconCircle_drawArn(PyObject *self, PyObject *args) {
   PyObject*   detailColorPyTup;
   PyObject*   py_list;
   PyObject*   py_tuple;
   PyObject*   py_float;
   float*      bulbColors;
   float       detailColor[3];
   float       gx, gy, scale, ao, w2h;
   long        numBulbs, features;
   int         vertIndex = 0;
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

   char circleSegments = 60;

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new float[numBulbs*3];
//#  pragma omp parallel for
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

   if (iconCircleVertexBuffer == NULL     ||
       iconCircleColorBuffer  == NULL     ||
       iconCircleIndices      == NULL     ||
       iconCircleBulbVertices == NULL     ){

      printf("Generating geometry for iconCircle\n");
      vector<GLfloat> markerVerts;
      vector<GLfloat> markerColrs;

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma, tmx, tmy, delta, R, G, B;

      drawEllipse(float(0.0), float(0.0), float(0.16), circleSegments/3, detailColor, markerVerts, markerColrs);
      drawHalo(float(0.0), float(0.0), float(0.22), float(0.22), float(0.07), circleSegments/3, detailColor, markerVerts, markerColrs);

      // Draw Only the color wheel if 'features' <= 0
      delta = degSegment;
//#     pragma omp parallel for
      for (int j = 0; j < numBulbs; j++) {
         R = bulbColors[j*3+0];
         G = bulbColors[j*3+1];
         B = bulbColors[j*3+2];
         for (int i = 0; i < circleSegments/numBulbs; i++) {
            /* X */ verts.push_back(float(0.0));
            /* Y */ verts.push_back(float(0.0));

            tma = float(degToRad(i*delta + j*angOffset - 90.0));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));

            tma = float(degToRad((i+1)*delta + j*angOffset - 90.0));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));

            /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
            /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
            /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
         }
      }

      // Draw Color Wheel + Outline if 'features' >= 1
      R = detailColor[0];
      G = detailColor[1];
      B = detailColor[2];
      delta = float(degSegment);
      if (features >= 1) {
         tmx = 0.0;
         tmy = 0.0;
      } else {
         tmx = offScreen;
         tmy = offScreen;
      }
      drawHalo(
            tmx, tmy,
            float(1.0), float(1.0),
            float(0.1),
            circleSegments,
            detailColor,
            verts, colrs);

      // Draw Color Wheel + Outline + BulbMarkers if 'features' >= 2
      int iUlim = circleSegments/3;
      int tmo = 180/numBulbs;
      degSegment = 360/iUlim;
//#     pragma omp parallel for
      for (int j = 0; j < 6; j++) {
         if ( (j < numBulbs) && (features >= 2) ) {
            tmx = float(cos(degToRad(-90 - j*(angOffset) + tmo))*1.05);
            tmy = float(sin(degToRad(-90 - j*(angOffset) + tmo))*1.05);
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
         drawEllipse(tmx, tmy, float(0.16), circleSegments/3, detailColor, verts, colrs);
      }

      // Draw Halos for bulb Markers
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos if 'features' == 3
//#     pragma omp parallel for
      for (int j = 0; j < 6; j++) {
         if (j < numBulbs && features >= 3) {
            tmx = float(cos(degToRad(-90 - j*(angOffset) + tmo))*1.05);
            tmy = float(sin(degToRad(-90 - j*(angOffset) + tmo))*1.05);
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
         drawHalo(
               tmx, tmy, 
               float(0.22), float(0.22), 
               float(0.07), 
               circleSegments/3, 
               detailColor, 
               verts, colrs);
      }
       
      // Draw Grand (Room) Halo
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos + Grand Halo if 'features' == 4
      if (features >= 4) {
         tmx = 0.0;
         tmy = 0.0;
      } else {
         tmx = offScreen;
         tmy = offScreen;
      }
      drawHalo(
            tmx, tmy,
            float(1.28), float(1.28),
            float(0.08),
            circleSegments,
            detailColor,
            verts, colrs);

      iconCircleVerts = verts.size()/2;
      printf("iconCircle vertexBuffer length: %.i, Number of vertices: %.i, tris: %.i\n", iconCircleVerts*2, iconCircleVerts, iconCircleVerts/3);

      // Safely (Re)allocate memory for bulb marker vertices
      if (iconCircleBulbVertices == NULL) {
         iconCircleBulbVertices = new GLfloat[markerVerts.size()];
      } else {
         delete [] iconCircleBulbVertices;
         iconCircleBulbVertices = new GLfloat[markerVerts.size()];
      }

      // Safely (Re)allocate memory for icon Vertex Buffer
      if (iconCircleVertexBuffer == NULL) {
         iconCircleVertexBuffer = new GLfloat[iconCircleVerts*2];
      } else {
         delete [] iconCircleVertexBuffer;
         iconCircleVertexBuffer = new GLfloat[iconCircleVerts*2];
      }

      // Safely (Re)allocate memory for icon Color Buffer
      if (iconCircleColorBuffer == NULL) {
         iconCircleColorBuffer = new GLfloat[iconCircleVerts*3];
      } else {
         delete [] iconCircleColorBuffer;
         iconCircleColorBuffer = new GLfloat[iconCircleVerts*3];
      }

      // Safely (Re)allocate memory for icon indices
      if (iconCircleIndices == NULL) {
         iconCircleIndices = new GLushort[iconCircleVerts];
      } else {
         delete [] iconCircleIndices;
         iconCircleIndices = new GLushort[iconCircleVerts];
      }

//#     pragma omp parallel for
      for (unsigned int i = 0; i < markerVerts.size()/2; i++) {
         iconCircleBulbVertices[i*2+0] = markerVerts[i*2+0];
         iconCircleBulbVertices[i*2+1] = markerVerts[i*2+1];
      }

//#     pragma omp parallel for
      for (unsigned int i = 0; i < iconCircleVerts; i++) {
         iconCircleVertexBuffer[i*2+0] = verts[i*2+0];
         iconCircleVertexBuffer[i*2+1] = verts[i*2+1];
         iconCircleColorBuffer[i*3+0]  = colrs[i*3+0];
         iconCircleColorBuffer[i*3+1]  = colrs[i*3+1];
         iconCircleColorBuffer[i*3+2]  = colrs[i*3+2];
         iconCircleIndices[i]          = i;
      }

      prevIconCircleNumBulbs = numBulbs;
      prevIconCircleFeatures = features;
   } 
   // Update Geometry, if alreay allocated
   if (prevIconCircleFeatures != features ||
       prevIconCircleNumBulbs != numBulbs ){

      float angOffset = float(360.0 / float(numBulbs));
      float tmx, tmy;
      vertIndex = 0;
      
      /*
       * Explanation of features:
       * <= 0: just the color representation
       * <= 1: color representation + outline
       * <= 2: color representation + outline + bulb markers
       * <= 3: color representation + outline + bulb markers + bulb marker halos
       * <= 4: color representation + outline + bulb markers + bulb marker halos + grand halo
       */
      // Draw Only the color wheel if 'features' <= 0
      // Update Color Wheel
      for (int j = 0; j < numBulbs; j++) {
         for (int i = 0; i < circleSegments/numBulbs; i++) {
            vertIndex += 6;
         }
      }

      // Draw Color Wheel + Outline if 'features' == 1
      // Update Outline
      if (features >= 1) {
         // Move Outline on-screen if off-screen
         if (iconCircleVertexBuffer[vertIndex] > offScreen/2) {
            tmx = -offScreen;
            tmy = -offScreen;
         } else {
            tmx = 0.0;
            tmy = 0.0;
         }
      } else {
         // Move Outline of screen if on screen
         if (iconCircleVertexBuffer[vertIndex] > offScreen/2) {
            tmx = 0.0;
            tmy = 0.0;
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
      }
      //float delta = float(degSegment);
//#     pragma omp parallel for
      for (int i = 0; i < circleSegments; i++) {
         /* X */ iconCircleVertexBuffer[vertIndex +  0] = iconCircleVertexBuffer[vertIndex +  0] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex +  1] = iconCircleVertexBuffer[vertIndex +  1] + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex +  2] = iconCircleVertexBuffer[vertIndex +  2] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex +  3] = iconCircleVertexBuffer[vertIndex +  3] + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex +  4] = iconCircleVertexBuffer[vertIndex +  4] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex +  5] = iconCircleVertexBuffer[vertIndex +  5] + tmy;

         /* X */ iconCircleVertexBuffer[vertIndex +  6] = iconCircleVertexBuffer[vertIndex +  6] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex +  7] = iconCircleVertexBuffer[vertIndex +  7] + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex +  8] = iconCircleVertexBuffer[vertIndex +  8] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex +  9] = iconCircleVertexBuffer[vertIndex +  9] + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex + 10] = iconCircleVertexBuffer[vertIndex + 10] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex + 11] = iconCircleVertexBuffer[vertIndex + 11] + tmy;
         vertIndex += 12;
      }

      // Update Bulb Markers
      // Draw Color Wheel + Outline + BulbMarkers if 'features' >= 2
      int iUlim = circleSegments/3;
      //int degSegment = 360/iUlim;
      for (int j = 0; j < 6; j++) {
         if (j < numBulbs && features >= 2) {
            tmx = float(cos(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
            tmy = float(sin(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
//#        pragma omp parallel for
         for (int i = 0; i < iUlim; i++) {
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconCircleBulbVertices[i*6+0];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconCircleBulbVertices[i*6+1];

            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconCircleBulbVertices[i*6+2];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconCircleBulbVertices[i*6+3];

            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconCircleBulbVertices[i*6+4];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconCircleBulbVertices[i*6+5];
         }
      }

      // Draw Halos for bulb Markers
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos if 'features' == 3
      for (int j = 0; j < 6; j++) {
         if (j < numBulbs && features >= 3) {
            tmx = float(cos(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
            tmy = float(sin(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
//#        pragma omp parallel for
         for (int i = 0; i < iUlim; i++) {
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 +  0];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 +  1];
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 +  2];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 +  3];
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 +  4];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 +  5];
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 +  6];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 +  7];
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 +  8];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 +  9];
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 + 10];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 + 11];
         }
      }

      // Update Grand (Room) Outline
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos + Grand Halo if 'features' == 4
      if (features >= 4) {
         // Move Outline on-screen if off-screen
         if (iconCircleVertexBuffer[vertIndex] > offScreen/2) {
            tmx = -offScreen;
            tmy = -offScreen;
         } else {
            tmx = 0.0;
            tmy = 0.0;
         }
      } else {
         // Move Outline of screen if on screen
         if (iconCircleVertexBuffer[vertIndex] > offScreen/2) {
            tmx = 0.0;
            tmy = 0.0;
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
      }
//#     pragma omp parallel for
      for (int i = 0; i < circleSegments; i++) {
         /* X */ iconCircleVertexBuffer[vertIndex +  0] = iconCircleVertexBuffer[vertIndex +  0]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex +  1] = iconCircleVertexBuffer[vertIndex +  1]  + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex +  2] = iconCircleVertexBuffer[vertIndex +  2]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex +  3] = iconCircleVertexBuffer[vertIndex +  3]  + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex +  4] = iconCircleVertexBuffer[vertIndex +  4]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex +  5] = iconCircleVertexBuffer[vertIndex +  5]  + tmy;

         /* X */ iconCircleVertexBuffer[vertIndex +  6] = iconCircleVertexBuffer[vertIndex +  6]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex +  7] = iconCircleVertexBuffer[vertIndex +  7]  + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex +  8] = iconCircleVertexBuffer[vertIndex +  8]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex +  9] = iconCircleVertexBuffer[vertIndex +  9]  + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex + 10] = iconCircleVertexBuffer[vertIndex + 10]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex + 11] = iconCircleVertexBuffer[vertIndex + 11]  + tmy;
         vertIndex += 12;
      }

      prevIconCircleFeatures = features;
      //printf("iconCircle vertIndex after update: %.i, vertices updated: %.i, tris updated: %.i\n", vertIndex, vertIndex/2, vertIndex/6);
   }
   // Geometry already calculated, update colors
   /*
    * Iterate through each color channel 
    * 0 - RED
    * 1 - GREEN
    * 2 - BLUE
    */
   for (int i = 0; i < 3; i++) {
         
      // Update color wheel, if needed
      for (int j = 0; j < numBulbs; j++) {
         int tmo = (60/numBulbs)*9;
         if (float(bulbColors[i+j*3]) != iconCircleColorBuffer[i + j*tmo]
               || prevIconCircleNumBulbs != numBulbs){
//#           pragma omp parallel for
            for (int k = 0; k < tmo/3; k++) {
               if (float(bulbColors[i+j*3]) != iconCircleColorBuffer[i + k*3 + j*tmo]){
                  iconCircleColorBuffer[j*tmo + k*3 + i] = float(bulbColors[i+j*3]);
               }
            }
         }
      }

      // Update outline, bulb markers, if needed
      if (float(detailColor[i]) != float(iconCircleColorBuffer[i + circleSegments*9])) {
         for (unsigned int k = circleSegments*3; k < iconCircleVerts; k++) {
            iconCircleColorBuffer[ k*3 + i ] = float(detailColor[i]);
         }
      }
   }

   prevIconCircleNumBulbs = numBulbs;
   delete [] bulbColors;

   glPushMatrix();
   glTranslatef(gx*w2h, gy, 0);
   if (w2h >= 1) {
      glScalef(scale, scale, 1);
   } else {
      glScalef(scale*w2h, scale*w2h, 1);
   }
   glRotatef(ao, 0, 0, 1);
   glColorPointer(3, GL_FLOAT, 0, iconCircleColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, iconCircleVertexBuffer);
   glDrawElements( GL_TRIANGLES, iconCircleVerts, GL_UNSIGNED_SHORT, iconCircleIndices);
   glPopMatrix();

   Py_RETURN_NONE;
}
