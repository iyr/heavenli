#include <Python.h>
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>
#include "drawUtils.h"
#define degToRad(angleInDegrees) ((angleInDegrees) * 3.1415926535 / 180.0)
using namespace std;

float constrain(float value, float min, float max) {
   if (value > max)
      return max;
   else if (value < min)
      return min;
   else
      return value;
}

GLfloat  *homeCircleVertexBuffer       = NULL;
GLfloat  *homeCircleColorBuffer        = NULL;
GLushort *homeCircleIndices            = NULL;
GLuint   homeCircleVerts;
GLint    prevHomeCircleNumBulbs;

PyObject* drawHomeCircle_drawArn(PyObject *self, PyObject *args) {
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double *bulbColors;
   float gx, gy, wx, wy, ao, w2h, R, G, B;
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
   bulbColors = new double[numBulbs*3];
#  pragma omp parallel for
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = double(PyFloat_AsDouble(py_float));
      }
   }

   if (homeCircleVertexBuffer == NULL  ||
       homeCircleColorBuffer  == NULL  ||
       homeCircleIndices      == NULL  ){

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
#        pragma omp parallel for
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

#     pragma omp parallel for
      for (unsigned int i = 0; i < homeCircleVerts; i++) {
         homeCircleVertexBuffer[i*2+0] = verts[i*2+0];
         homeCircleVertexBuffer[i*2+1] = verts[i*2+1];
         homeCircleColorBuffer[i*3+0]  = colrs[i*3+0];
         homeCircleColorBuffer[i*3+1]  = colrs[i*3+1];
         homeCircleColorBuffer[i*3+2]  = colrs[i*3+2];
         homeCircleIndices[i]          = i;
      }

      prevHomeCircleNumBulbs = numBulbs;
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
#           pragma omp parallel for
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

   glPushMatrix();
   glScalef(sqrt(w2h)*hypot(wx, wy), sqrt(wy/wx)*hypot(wx, wy), 1.0);
   glRotatef(ao, 0, 0, 1);
   glColorPointer(3, GL_FLOAT, 0, homeCircleColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, homeCircleVertexBuffer);
   glDrawElements( GL_TRIANGLES, homeCircleVerts, GL_UNSIGNED_SHORT, homeCircleIndices);
   glPopMatrix();

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
GLfloat  *iconCircleVertexBuffer       = NULL;
GLfloat  *iconCircleColorBuffer        = NULL;
GLfloat  *iconBulbMarkerVertices       = NULL;
GLushort *iconCircleIndices            = NULL;
GLuint   iconCircleVerts;
int      prevIconCircleNumBulbs;
int      prevIconCircleFeatures;
float    offScreen                     = 100.0;

PyObject* drawIconCircle_drawArn(PyObject *self, PyObject *args) {
   PyObject* detailColorPyTup;
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double *bulbColors;
   double detailColor[3];
   float gx, gy, scale, ao, w2h, delta;
   int numBulbs, features;
   int vertIndex = 0;
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
   bulbColors = new double[numBulbs*3];
#  pragma omp parallel for
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = double(PyFloat_AsDouble(py_float));
      }
   }

   // Parse RGB detail colors
   detailColor[0] = PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0));
   detailColor[1] = PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1));
   detailColor[2] = PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2));

   if (iconCircleVertexBuffer == NULL     ||
       iconCircleColorBuffer  == NULL     ||
       iconCircleIndices      == NULL     ||
       iconBulbMarkerVertices == NULL     ){

      printf("Generating geometry for iconCircle\n");
      vector<GLfloat> markerVerts;
      vector<GLfloat> markerColrs;

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma, tmx, tmy, R, G, B, delta;

      drawEllipse(float(0.0), float(0.0), float(0.16), circleSegments/3, detailColor, markerVerts, markerColrs);
      drawHalo(float(0.0), float(0.0), float(0.22), float(0.22), float(0.07), circleSegments/3, detailColor, markerVerts, markerColrs);

      // Draw Only the color wheel if 'features' <= 0
      delta = degSegment;
#     pragma omp parallel for
      for (int j = 0; j < numBulbs; j++) {
         R = float(bulbColors[j*3+0]);
         G = float(bulbColors[j*3+1]);
         B = float(bulbColors[j*3+2]);
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
      R = float(detailColor[0]);
      G = float(detailColor[1]);
      B = float(detailColor[2]);
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
#     pragma omp parallel for
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
#     pragma omp parallel for
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
      if (iconBulbMarkerVertices == NULL) {
         iconBulbMarkerVertices = new GLfloat[markerVerts.size()];
      } else {
         delete [] iconBulbMarkerVertices;
         iconBulbMarkerVertices = new GLfloat[markerVerts.size()];
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

#     pragma omp parallel for
      for (unsigned int i = 0; i < markerVerts.size()/2; i++) {
         iconBulbMarkerVertices[i*2+0] = markerVerts[i*2+0];
         iconBulbMarkerVertices[i*2+1] = markerVerts[i*2+1];
      }

#     pragma omp parallel for
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

      char degSegment = 360 / circleSegments;
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
      delta = float(degSegment);
#     pragma omp parallel for
      for (int i = 0; i < circleSegments; i++) {
         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmy;

         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex] + tmy;
      }

      // Update Bulb Markers
      // Draw Color Wheel + Outline + BulbMarkers if 'features' >= 2
      int iUlim = circleSegments/3;
      degSegment = 360/iUlim;
      for (int j = 0; j < 6; j++) {
         if (j < numBulbs && features >= 2) {
            tmx = float(cos(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
            tmy = float(sin(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
#        pragma omp parallel for
         for (int i = 0; i < iUlim; i++) {
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconBulbMarkerVertices[i*6+0];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconBulbMarkerVertices[i*6+1];

            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconBulbMarkerVertices[i*6+2];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconBulbMarkerVertices[i*6+3];

            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconBulbMarkerVertices[i*6+4];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconBulbMarkerVertices[i*6+5];
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
#        pragma omp parallel for
         for (int i = 0; i < iUlim; i++) {
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconBulbMarkerVertices[iUlim*6 + i*12 +  0];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconBulbMarkerVertices[iUlim*6 + i*12 +  1];
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconBulbMarkerVertices[iUlim*6 + i*12 +  2];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconBulbMarkerVertices[iUlim*6 + i*12 +  3];
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconBulbMarkerVertices[iUlim*6 + i*12 +  4];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconBulbMarkerVertices[iUlim*6 + i*12 +  5];
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconBulbMarkerVertices[iUlim*6 + i*12 +  6];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconBulbMarkerVertices[iUlim*6 + i*12 +  7];
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconBulbMarkerVertices[iUlim*6 + i*12 +  8];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconBulbMarkerVertices[iUlim*6 + i*12 +  9];
            /* X */ iconCircleVertexBuffer[vertIndex++] = tmx + iconBulbMarkerVertices[iUlim*6 + i*12 + 10];
            /* Y */ iconCircleVertexBuffer[vertIndex++] = tmy + iconBulbMarkerVertices[iUlim*6 + i*12 + 11];
         }
      }

      // Update Grand (Room) Outline
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos + Grand Halo if 'features' == 4
      circleSegments = 60;
      degSegment = 360/60;
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
#     pragma omp parallel for
      for (int i = 0; i < circleSegments; i++) {
         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmy;


         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmy;
         /* X */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmx;
         /* Y */ iconCircleVertexBuffer[vertIndex++] = iconCircleVertexBuffer[vertIndex]  + tmy;
      }

      prevIconCircleFeatures = features;
      printf("iconCircle vertIndex after update: %.i, vertices updated: %.i, tris updated: %.i\n", vertIndex, vertIndex/2, vertIndex/6);
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
#           pragma omp parallel for
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

GLfloat  *homeLinearVertexBuffer = NULL;
GLfloat  *homeLinearColorBuffer  = NULL;
GLushort *homeLinearIndices      = NULL;
GLuint   homeLinearVerts;
int      prevHomeLinearNumbulbs;

PyObject* drawHomeLinear_drawArn(PyObject *self, PyObject *args) {
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double *bulbColors;
   float gx, gy, wx, wy, ao, w2h, R, G, B;
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
   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new double[numBulbs*3];
#  pragma omp parallel for
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = double(PyFloat_AsDouble(py_float));
      }
   }

   if (homeLinearVertexBuffer    == NULL ||
       homeLinearColorBuffer     == NULL ||
       homeLinearIndices         == NULL ){

      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy;
      float offset = float(4.0/60.0);
      R = float(0.0);
      G = float(0.0);
      B = float(0.0);
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

      if (homeLinearVertexBuffer == NULL) {
         homeLinearVertexBuffer = new GLfloat[homeLinearVerts*2];
      } else {
         delete [] homeLinearVertexBuffer;
         homeLinearVertexBuffer = new GLfloat[homeLinearVerts*2];
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

#     pragma omp parallel for
      for (unsigned int i = 0; i < homeLinearVerts; i++) {
         homeLinearVertexBuffer[i*2+0] = verts[i*2+0];
         homeLinearVertexBuffer[i*2+1] = verts[i*2+1];
         homeLinearColorBuffer[i*3+0]  = colrs[i*3+0];
         homeLinearColorBuffer[i*3+1]  = colrs[i*3+1];
         homeLinearColorBuffer[i*3+2]  = colrs[i*3+2];
         homeLinearIndices[i]          = i;
      }

      prevHomeLinearNumbulbs = numBulbs;
   } 
   // Geometry already calculated, check if any colors need to be updated.
   else {
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < numBulbs; j++) {
            // 3*2*3:
            // 3 (R,G,B) color values per vertex
            // 2 Triangles per Quad
            // 3 Vertices per Triangle
            if (float(bulbColors[i+j*3]) != homeLinearColorBuffer[i + j*(60/numBulbs)*9*2 ] || prevHomeLinearNumbulbs != numBulbs) {
#              pragma omp parallel for
               for (int k = 0; k < (60/numBulbs)*3*2; k++) {  
                  if (float(bulbColors[i+j*3]) != homeLinearColorBuffer[i + k*3 + j*(60/numBulbs)*9*2 ]) {
                     homeLinearColorBuffer[ j*(60/numBulbs)*9*2 + k*3 + i ] = float(bulbColors[i+j*3]);
                  }
               }
            }
         }
      }
   }

   prevHomeLinearNumbulbs = numBulbs;
   delete [] bulbColors;

   glPushMatrix();
   glRotatef(90, 0, 0, 1);
   glScalef(0.5, float(w2h/2.0), 1);
   glRotatef(ao+90, 0, 0, 1);
   glColorPointer(3, GL_FLOAT, 0, homeLinearColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, homeLinearVertexBuffer);
   glDrawElements( GL_TRIANGLES, homeLinearVerts, GL_UNSIGNED_SHORT, homeLinearIndices);
   glPopMatrix();

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

GLfloat  *iconLinearVertexBuffer = NULL;
GLfloat  *iconLinearColorBuffer  = NULL;
GLushort *iconLinearIndices      = NULL;
GLuint   iconLinearVerts;
int      prevIconLinearNumBulbs;
int      prevIconLinearFeatures;

PyObject* drawIconLinear_drawArn(PyObject *self, PyObject *args) {
   PyObject* detailColorPyTup;
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double *bulbColors;
   double detailColor[3];
   float gx, gy, scale, ao, w2h, R, G, B, delta;
   int numBulbs, features;
   int vertIndex = 0;
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
   bulbColors = new double[numBulbs*3];
#  pragma omp parallel for
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = double(PyFloat_AsDouble(py_float));
      }
   }

   // Parse RGB detail colors
   detailColor[0] = PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0));
   detailColor[1] = PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1));
   detailColor[2] = PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2));

   if (iconLinearVertexBuffer == NULL     ||
       iconLinearColorBuffer  == NULL     ||
       iconLinearIndices      == NULL     ){

      printf("Generating geometry for iconLinear\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy, tmx, tmy, ri, ro;
      float offset = float(2.0/60.0);
      float degSegment = float(360.0/float(circleSegments));
      float delta = float(degSegment/4.0);

      // Define Square of Stripes with Rounded Corners
      int tmb = 0;
      for (int i = 0; i < 60; i++) {
         if (i%10 == 0) {
            tmb++;
         }
         R = float(bulbColors[tmb*3+0]);
         G = float(bulbColors[tmb*3+1]);
         B = float(bulbColors[tmb*3+2]);

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

      R = float(detailColor[0]);
      G = float(detailColor[1]);
      B = float(detailColor[2]);

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
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  0], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  1]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  2], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  3]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  4], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  5]);

               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  6], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  7]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  8], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  9]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 + 10], -2.0, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 + 11]);
            } else if (i == numBulbs-1) {
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  0], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  1]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  2], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  3]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  4], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  5]);

               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  6], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  7]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  8], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  9]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 + 10], tmx-limit, 2.0));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 + 11]);
            } else {
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  0], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  1]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  2], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  3]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  4], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  5]);

               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  6], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  7]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 +  8], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 +  9]);
               /* X */ verts.push_back(constrain(tmx + iconBulbMarkerVertices[circleSegments*6 + j*12 + 10], tmx-limit, tmx+limit));
               /* Y */ verts.push_back(          tmy + iconBulbMarkerVertices[circleSegments*6 + j*12 + 11]);
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

      if (iconLinearVertexBuffer == NULL) {
         iconLinearVertexBuffer = new GLfloat[iconLinearVerts*2];
      } else {
         delete [] iconLinearVertexBuffer;
         iconLinearVertexBuffer = new GLfloat[iconLinearVerts*2];
      }

      if (iconLinearColorBuffer == NULL) {
         iconLinearColorBuffer = new GLfloat[iconLinearVerts*3];
      } else {
         delete [] iconLinearColorBuffer;
         iconLinearColorBuffer = new GLfloat[iconLinearVerts*3];
      }

      if (iconLinearIndices == NULL) {
         iconLinearIndices = new GLushort[iconLinearVerts];
      } else {
         delete [] iconLinearIndices;
         iconLinearIndices = new GLushort[iconLinearVerts];
      }

#     pragma omp parallel for
      for (unsigned int i = 0; i < iconLinearVerts; i++) {
         iconLinearVertexBuffer[i*2+0] = verts[i*2+0];
         iconLinearVertexBuffer[i*2+1] = verts[i*2+1];
         iconLinearColorBuffer[i*3+0]  = colrs[i*3+0];
         iconLinearColorBuffer[i*3+1]  = colrs[i*3+1];
         iconLinearColorBuffer[i*3+2]  = colrs[i*3+2];
         iconLinearIndices[i]          = i;
      }

      prevIconLinearNumBulbs = numBulbs;
      prevIconLinearFeatures = features;
   } 

   // Update features
   if (prevIconLinearFeatures != features ||
       prevIconLinearNumBulbs != numBulbs ){

      prevIconLinearFeatures = features;
      float tmx, tmy;
      float degSegment = float(360.0/float(circleSegments));
      delta = float(degSegment/4.0);
      tmx = 0.0;
      tmy = 0.0;
      vertIndex = 0;

      // Define Square of Stripes with Rounded Corners
#     pragma omp parallel for
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
         if (iconLinearVertexBuffer[vertIndex+1] > offScreen/2) {
            tmx = -offScreen;
            tmy = -offScreen;
         } else {
            tmx = 0.0;
            tmy = 0.0;
         }
      } 
      // Move outline off-screen if on-screen
      else {
         if (iconLinearVertexBuffer[vertIndex+1] > offScreen/2) {
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
#     pragma omp parallel for
      for (int i = 0; i < 4; i++ ) {
         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;

         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
      }

      /*
       * Draw Rounded Corners
       */
      for (int i = 0; i < 4; i++) {
#        pragma omp parallel for
         for (int j = 0; j < circleSegments; j++) {
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;

            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
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
#        pragma omp parallel for
         for (int j = 0; j < circleSegments; j++) {
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconBulbMarkerVertices[j*6 + 0] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconBulbMarkerVertices[j*6 + 1] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconBulbMarkerVertices[j*6 + 2] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconBulbMarkerVertices[j*6 + 3] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconBulbMarkerVertices[j*6 + 4] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconBulbMarkerVertices[j*6 + 5] + tmy;
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
#        pragma omp parallel for
         for (int j = 0; j < circleSegments; j++) {
            tmj = 6*circleSegments + j*12;
            if (i == 0) {
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  0 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  1 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  2 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  3 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  4 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  5 + tmj];

               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  6 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  7 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  8 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  9 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[ 10 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[ 11 + tmj];
            } else if (i == numBulbs-1) {
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  0 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  1 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  2 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  3 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  4 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  5 + tmj];

               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  6 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  7 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  8 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  9 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[ 10 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[ 11 + tmj];
            } else {
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  0 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  1 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  2 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  3 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  4 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  5 + tmj];

               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  6 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  7 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[  8 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[  9 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx + iconBulbMarkerVertices[ 10 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy + iconBulbMarkerVertices[ 11 + tmj];
            }
         }
      }

      // Define Grand Outline
      if (features >= 4) {
         if (iconLinearVertexBuffer[vertIndex] > offScreen/2) {
            tmx = -offScreen;
            tmy = -offScreen;
         } else {
            tmx = 0.0;
            tmy = 0.0;
         }
      } else {
         if (iconLinearVertexBuffer[vertIndex] > offScreen/2) {
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

#     pragma omp parallel for
      for (int i = 0; i < 4; i ++ ) {
         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;

         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
      }

      /*
       * Draw Rounded Corners
       */
      for (int i = 0; i < 4; i++) {
#        pragma omp parallel for
         for (int j = 0; j < circleSegments; j++) {
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;

            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearVertexBuffer[vertIndex] + tmy;
         }
      }

      prevIconLinearFeatures = features;
   }

   // Geometry allocated/calculated, check if colors need to be updated
   int tmb = 0;
   for (int i = 0; i < 3; i++) {
      for (int j = 0; j < numBulbs; j++) {
         float tmc = float(bulbColors[j*3+i]);
         // Special Case for Rounded Corner Segments
         if (j == 0) {
            if (tmc != iconLinearColorBuffer[i] || prevIconLinearNumBulbs != numBulbs) {
#              pragma omp parallel for
               for (int k = 0; k < (j*(60/numBulbs)*3*2 + circleSegments*2*3 + 6)*3; k++) {
                  if (tmc != iconLinearColorBuffer[i + k*3]) {
                     iconLinearColorBuffer[i + k*3] = tmc;
                  }
               }
            }
         } 
   
         // Special Case for Rounded Corner Segments
         if (j == numBulbs-1) {
            if (tmc != iconLinearColorBuffer[i + (j*(60/numBulbs)*3*2 + circleSegments*3*2 + 6)*3] || prevIconLinearNumBulbs != numBulbs ) {
#              pragma omp parallel for
               for (int k = 0; k < ((60/numBulbs)*3*2 + 2*3*circleSegments + 2*3); k++) {
                  if (tmc != iconLinearColorBuffer[i + k*3 + (j*(60/numBulbs)*3*2 + circleSegments*3*2 + 6)*3] ) {
                     iconLinearColorBuffer[i + k*3 + (j*(60/numBulbs)*3*2 + circleSegments*3*2 + 6)*3] = tmc;
                  }
               }
            }
         } 
         else
         // General Case for middle segments
         {
            if (tmc != iconLinearColorBuffer[i + (j*(60/numBulbs)*3*2 + circleSegments*3*2 + 6)*3] || prevIconLinearNumBulbs != numBulbs) {
#              pragma omp parallel for
               for (int k = 0; k < (60/numBulbs)*3*2; k++) {
                  if (tmc != iconLinearColorBuffer[i + k*3 + (j*(60/numBulbs)*3*2 + circleSegments*3*2 + 6)*3] ) {
                     iconLinearColorBuffer[i + k*3 + (j*(60/numBulbs)*3*2 + circleSegments*3*2 + 6)*3] = tmc;
                  }
               }
            }
         }
      }

      // Check if detail color needs to be updated
      if (float(detailColor[i]) != iconLinearColorBuffer[i+(60*2*3 + 4*circleSegments*3 + 2*6)*3]) {
#           pragma omp parallel for
         for (unsigned int k = (60*2*3 + 4*circleSegments*3 + 2*6); k < iconLinearVerts; k++) {
            iconLinearColorBuffer[k*3+i] = float(detailColor[i]);
         }
      }
   }
   prevIconLinearNumBulbs = numBulbs;
   
   delete [] bulbColors;

   glPushMatrix();
   glTranslatef(gx*w2h, gy, 0);
   glRotatef(90, 0, 0, 1);
   if (w2h >= 1.0) {
      glScalef(scale, scale, 1);
   } else {
      glScalef(scale*w2h, scale*w2h, 1);
   }
   glRotatef(ao+90, 0, 0, 1);
   glColorPointer(3, GL_FLOAT, 0, iconLinearColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, iconLinearVertexBuffer);
   glDrawElements( GL_TRIANGLES, iconLinearVerts, GL_UNSIGNED_SHORT, iconLinearIndices);
   glPopMatrix();

   Py_RETURN_NONE;
}

static PyMethodDef drawArn_methods[] = {
   { "drawHomeCircle", (PyCFunction)drawHomeCircle_drawArn, METH_VARARGS },
   { "drawIconCircle", (PyCFunction)drawIconCircle_drawArn, METH_VARARGS },
   { "drawHomeLinear", (PyCFunction)drawHomeLinear_drawArn, METH_VARARGS },
   { "drawIconLinear", (PyCFunction)drawIconLinear_drawArn, METH_VARARGS },
   { NULL, NULL, 0, NULL }
};

static PyModuleDef drawArn_module = {
   PyModuleDef_HEAD_INIT,
   "drawArn",
   "Functions for drawing the background and lamp iconography",
   0,
   drawArn_methods
};

PyMODINIT_FUNC PyInit_drawArn() {
   PyObject* m = PyModule_Create(&drawArn_module);
   if (m == NULL) {
      return NULL;
   }
   return m;
}
