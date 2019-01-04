#include <Python.h>
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>
#define degToRad(angleInDegrees) ((angleInDegrees) * 3.1415926535 / 180.0)
using namespace std;

GLfloat  *homeCircleVertexBuffer = NULL;
GLfloat  *homeCircleColorBuffer  = NULL;
GLushort *homeCircleIndices      = NULL;
GLuint   homeCircleVerts         = NULL;
int      prevHomeCircleNumBulbs            = NULL;
float    prevHomeCircleAngularOffset       = NULL;
float    prevHomeCircleWx                  = NULL;
float    prevHomeCircleWy                  = NULL;

PyObject* drawHomeCircle_drawArn(PyObject *self, PyObject *args) {
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double *bulbColors;
   float gx, gy, wx, wy, ao, w2h, sx, sy;
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
   char circleSegments = 60/numBulbs;

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new double[numBulbs*3];
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = double(PyFloat_AsDouble(py_float));
      }
   }

   if (homeCircleVertexBuffer == NULL  ||
       homeCircleColorBuffer  == NULL  ||
       homeCircleIndices      == NULL ) {

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma, tmx, tmy;
      sx = sqrt(w2h)*hypot(wx, wy);
      sy = sqrt(wy/wx)*hypot(wx, wy);
      
      for (int j = 0; j < numBulbs; j++) {
         for (int i = 0; i < circleSegments; i++) {
            /* X */ verts.push_back(float(0.0));
            /* Y */ verts.push_back(float(0.0));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            tma = float(degToRad(i*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = cos(tma)*sx;
            tmy = sin(tma)*sy;
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            tma = float(degToRad((i+1)*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = cos(tma)*sx;
            tmy = sin(tma)*sy;
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));
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
      prevHomeCircleAngularOffset = ao;
      prevHomeCircleWx = wx;
      prevHomeCircleWy = wy;
   } 
   else
   // Update Geometry, if alreay allocated
   if (prevHomeCircleNumBulbs        != numBulbs   ||
       prevHomeCircleAngularOffset   != ao         ||
       prevHomeCircleWx              != wx         ||
       prevHomeCircleWy              != wy         ){

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma, tmx, tmy;
      int vertIndex = 0;
      int colorIndex = 0;
      sx = sqrt(w2h)*hypot(wx, wy);
      sy = sqrt(wy/wx)*hypot(wx, wy);
      
      for (int j = 0; j < numBulbs; j++) {
         for (int i = 0; i < circleSegments; i++) {
            /* X */ homeCircleVertexBuffer[vertIndex++] = (float(0.0));
            /* Y */ homeCircleVertexBuffer[vertIndex++] = (float(0.0));
            /* R */ homeCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+0]));
            /* G */ homeCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+1]));
            /* B */ homeCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+2]));

            tma = float(degToRad(i*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = cos(tma)*sx;
            tmy = sin(tma)*sy;
            /* X */ homeCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ homeCircleVertexBuffer[vertIndex++] = (float(tmy));
            /* R */ homeCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+0]));
            /* G */ homeCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+1]));
            /* B */ homeCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+2]));

            tma = float(degToRad((i+1)*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = cos(tma)*sx;
            tmy = sin(tma)*sy;
            /* X */ homeCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ homeCircleVertexBuffer[vertIndex++] = (float(tmy));
            /* R */ homeCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+0]));
            /* G */ homeCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+1]));
            /* B */ homeCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+2]));
         }
      }

      prevHomeCircleNumBulbs = numBulbs;
      prevHomeCircleAngularOffset = ao;
      prevHomeCircleWx = wx;
      prevHomeCircleWy = wy;

   }
   // Geometry already calculated, update colors
   else 
   {
      /*
       * Iterate through each color channel 
       * 0 - RED
       * 1 - GREEN
       * 2 - BLUE
       */
      for (int i = 0; i < 3; i++) {
         
         // Update color, if needed
         for (int j = 0; j < numBulbs; j++) {

            if (float(bulbColors[i+j*3]) != homeCircleColorBuffer[ i + j*circleSegments*9 ]) {
               unsigned int tmu = circleSegments*3;
#              pragma omp parallel for
               for (unsigned int k = 0; k < tmu; k++) {
                     homeCircleColorBuffer[ j*circleSegments*9 + k*3 + i ] = float(bulbColors[i+j*3]);
               }
            }
         }
      }
   }

   delete [] bulbColors;

   glColorPointer(3, GL_FLOAT, 0, homeCircleColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, homeCircleVertexBuffer);
   glDrawElements( GL_TRIANGLES, homeCircleVerts, GL_UNSIGNED_SHORT, homeCircleIndices);

   Py_RETURN_NONE;
}

GLfloat  *iconCircleVertexBuffer = NULL;
GLfloat  *iconCircleColorBuffer  = NULL;
GLushort *iconCircleIndices      = NULL;
GLuint   iconCircleVerts         = NULL;
int      prevIconCircleNumBulbs        = NULL;
int      prevIconCircleFeatures        = NULL;
float    prevIconCircleAngularOffset   = NULL;
float    prevIconCircleWx              = NULL;
float    prevIconCircleWy              = NULL;
float    prevIconCircleW2H             = NULL;

PyObject* drawIconCircle_drawArn(PyObject *self, PyObject *args) {
   PyObject* detailColorPyTup;
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double *bulbColors;
   double detailColor[3];
   float gx, gy, scale, ao, w2h; 
   int numBulbs, features;
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

   char circleSegments = 60/numBulbs;

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new double[numBulbs*3];
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

   if (prevIconCircleNumBulbs != numBulbs ||
       iconCircleVertexBuffer == NULL     ||
       iconCircleColorBuffer  == NULL     ||
       iconCircleIndices      == NULL     ){

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma, tmx, tmy;
      gx *= w2h;
      if (w2h >= 1.0) {
         scale = scale;
      } else {
         scale *= w2h;
      }

      /*
       * Explanation of features:
       * <= 0: just the color representation
       * <= 1: color representation + outline
       * <= 2: color representation + outline + bulb markers
       * <= 3: color representation + outline + bulb markers + bulb marker halos
       * <= 4: color representation + outline + bulb markers + bulb marker halos + grand halo
       */
      // Draw Only the color wheel if 'features' <= 0
      for (int j = 0; j < numBulbs; j++) {
         for (int i = 0; i < circleSegments; i++) {
            /* X */ verts.push_back(float(gx));
            /* Y */ verts.push_back(float(gy));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            tma = float(degToRad(i*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = gx+cos(tma)*scale;
            tmy = gy+sin(tma)*scale;
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            tma = float(degToRad((i+1)*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = gx+cos(tma)*scale;
            tmy = gy+sin(tma)*scale;
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));
         }
      }

      // Draw Color Wheel + Outline if 'features' == 1
      if (features >= 1) {
         for (int i = 0; i < circleSegments*3; i++) {
            tma = float(degToRad(i*float(degSegment/3)) + ao - 90.0);
            tmx = gx+cos(tma)*scale;
            tmy = gy+sin(tma)*scale;
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));

            tmx = float(gx+cos(tma)*scale*1.1);
            tmy = float(gy+sin(tma)*scale*1.1);
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));

            tma = float(degToRad((i+1)*float(degSegment/3)) + ao - 90.0);
            tmx = gx+cos(tma)*scale;
            tmy = gy+sin(tma)*scale;
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));

            tmx = float(gx+cos(tma)*scale*1.1);
            tmy = float(gy+sin(tma)*scale*1.1);
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));

            tmx = gx+cos(tma)*scale;
            tmy = gy+sin(tma)*scale;
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));

            tma = float(degToRad((i+0)*float(degSegment/3)) + ao - 90.0);
            tmx = float(gx+cos(tma)*scale*1.1);
            tmy = float(gy+sin(tma)*scale*1.1);
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));
         }
      } else {
         for (int i = 0; i < circleSegments*3; i++) {
            for (int j = 0; j < 6; j++) {
               /* X */ verts.push_back(float(100.0));
               /* Y */ verts.push_back(float(100.0));
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));
            }
         }
      }

      // Draw Color Wheel + Outline + BulbMarkers if 'features' == 2
      int iUlim = (circleSegments*numBulbs)/3;
      if (features >= 2) {
         degSegment = 360/((circleSegments*numBulbs)/3);
         for (int j = 0; j < numBulbs; j++) {
            tmx = float(gx+cos(degToRad(-90 - j*(angOffset) + 180/numBulbs + ao))*1.05*scale);
            tmy = float(gy+sin(degToRad(-90 - j*(angOffset) + 180/numBulbs + ao))*1.05*scale);
            for (int i = 0; i < iUlim; i++) {
               /* X */ verts.push_back(float(tmx));
               /* Y */ verts.push_back(float(tmy));
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));

               /* X */ verts.push_back(float(tmx + scale*0.16*cos(degToRad(i*degSegment))));
               /* Y */ verts.push_back(float(tmy + scale*0.16*sin(degToRad(i*degSegment))));
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));

               /* X */ verts.push_back(float(tmx + scale*0.16*cos(degToRad((i+1)*degSegment))));
               /* Y */ verts.push_back(float(tmy + scale*0.16*sin(degToRad((i+1)*degSegment))));
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));
            }
         }
      } else {
         for (int k = 0; k < 3; k++) {
            for (int j = 0; j < numBulbs; j++) {
               for (int i = 0; i < iUlim; i++) {
                  /* X */ verts.push_back(float(100.0));
                  /* Y */ verts.push_back(float(100.0));
                  /* R */ colrs.push_back(float(detailColor[0]));
                  /* G */ colrs.push_back(float(detailColor[1]));
                  /* B */ colrs.push_back(float(detailColor[2]));
               }
            }
         }
      }

      // Draw Halos for bulb Markers
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos if 'features' == 3
      if (features >= 3) {
         for (int j = 0; j < numBulbs; j++) {
            tmx = float(gx+cos(degToRad(-90 - j*(angOffset) + 180/numBulbs + ao))*1.05*scale);
            tmy = float(gy+sin(degToRad(-90 - j*(angOffset) + 180/numBulbs + ao))*1.05*scale);
            for (int i = 0; i < iUlim; i++) {
               tma = float(degToRad(i*float(degSegment)) + ao - 90.0);
               /* X */ verts.push_back(float(tmx+cos(tma)*scale*0.22));
               /* Y */ verts.push_back(float(tmy+sin(tma)*scale*0.22));
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));

               /* X */ verts.push_back(float(tmx+cos(tma)*scale*0.29));
               /* Y */ verts.push_back(float(tmy+sin(tma)*scale*0.29));
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));

               tma = float(degToRad((i+1)*float(degSegment)) + ao - 90.0);
               /* X */ verts.push_back(float(tmx+cos(tma)*scale*0.22));
               /* Y */ verts.push_back(float(tmy+sin(tma)*scale*0.22));
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));

               /* X */ verts.push_back(float(tmx+cos(tma)*scale*0.29));
               /* Y */ verts.push_back(float(tmy+sin(tma)*scale*0.29));
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));

               /* X */ verts.push_back(float(tmx+cos(tma)*scale*0.22));
               /* Y */ verts.push_back(float(tmy+sin(tma)*scale*0.22));
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));

               tma = float(degToRad((i+0)*float(degSegment)) + ao - 90.0);
               /* X */ verts.push_back(float(tmx+cos(tma)*scale*0.29));
               /* Y */ verts.push_back(float(tmy+sin(tma)*scale*0.29));
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));
            }
         }
      } else {
         for (int j = 0; j < numBulbs; j++) {
            for (int k = 0; k < 6; k++) {
               for (int i = 0; i < iUlim; i++) {
                  /* X */ verts.push_back(float(100.0));
                  /* Y */ verts.push_back(float(100.0));
                  /* R */ colrs.push_back(float(detailColor[0]));
                  /* G */ colrs.push_back(float(detailColor[1]));
                  /* B */ colrs.push_back(float(detailColor[2]));
               }
            }
         }
      }
      
      // Draw Grand (Room) Halo
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos + Grand Halo if 'features' == 4
      circleSegments = 60;
      if (features >= 4) {
         degSegment = 360/60;
         for (int i = 0; i < circleSegments; i++) {
            tma = float(degToRad(i*float(degSegment)) + ao - 90.0);
            tmx = float(gx+cos(tma)*scale*1.28);
            tmy = float(gy+sin(tma)*scale*1.28);
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));

            tmx = float(gx+cos(tma)*scale*1.36);
            tmy = float(gy+sin(tma)*scale*1.36);
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));

            tma = float(degToRad((i+1)*float(degSegment)) + ao - 90.0);
            tmx = float(gx+cos(tma)*scale*1.28);
            tmy = float(gy+sin(tma)*scale*1.28);
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));

            tmx = float(gx+cos(tma)*scale*1.36);
            tmy = float(gy+sin(tma)*scale*1.36);
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));

            tmx = float(gx+cos(tma)*scale*1.28);
            tmy = float(gy+sin(tma)*scale*1.28);
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));

            tma = float(degToRad((i+0)*float(degSegment)) + ao - 90.0);
            tmx = float(gx+cos(tma)*scale*1.36);
            tmy = float(gy+sin(tma)*scale*1.36);
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));
         }
      } else {
         for (int k = 0; k < 6; k++) {
            for (int i = 0; i < circleSegments; i++) {
               /* X */ verts.push_back(float(100.0));
               /* Y */ verts.push_back(float(100.0));
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));
            }
         }
      }

      iconCircleVerts = verts.size()/2;

      if (iconCircleVertexBuffer == NULL) {
         iconCircleVertexBuffer = new GLfloat[iconCircleVerts*2];
      } else {
         delete [] iconCircleVertexBuffer;
         iconCircleVertexBuffer = new GLfloat[iconCircleVerts*2];
      }

      if (iconCircleColorBuffer == NULL) {
         iconCircleColorBuffer = new GLfloat[iconCircleVerts*3];
      } else {
         delete [] iconCircleColorBuffer;
         iconCircleColorBuffer = new GLfloat[iconCircleVerts*3];
      }

      if (iconCircleIndices == NULL) {
         iconCircleIndices = new GLushort[iconCircleVerts];
      } else {
         delete [] iconCircleIndices;
         iconCircleIndices = new GLushort[iconCircleVerts];
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
      prevIconCircleAngularOffset = ao;
      prevIconCircleW2H = w2h;
      prevIconCircleFeatures = features;
   } 
   else
   // Update Geometry, if alreay allocated
   if (prevIconCircleAngularOffset  != ao          ||
       prevIconCircleFeatures       != features    ||
       prevIconCircleW2H            != w2h         ){

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma, tmx, tmy;
      int vertIndex = 0;
      int colorIndex = 0;
      
      gx *= w2h;
      if (w2h >= 1.0) {
         scale = scale;
      } else {
         scale *= w2h;
      }

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
         for (int i = 0; i < circleSegments; i++) {
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(gx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(gy));

            tma = float(degToRad(i*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = gx+cos(tma)*scale;
            tmy = gy+sin(tma)*scale;
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

            tma = float(degToRad((i+1)*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = gx+cos(tma)*scale;
            tmy = gy+sin(tma)*scale;
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));
         }
      }

      // Draw Color Wheel + Outline if 'features' == 1
      // Update Outline
      if (features >= 1) {
         for (int i = 0; i < circleSegments*3; i++) {
            tma = float(degToRad(i*float(degSegment/3)) + ao - 90.0);
            tmx = gx+cos(tma)*scale;
            tmy = gy+sin(tma)*scale;
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

            tmx = float(gx+cos(tma)*scale*1.1);
            tmy = float(gy+sin(tma)*scale*1.1);
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

            tma = float(degToRad((i+1)*float(degSegment/3)) + ao - 90.0);
            tmx = gx+cos(tma)*scale;
            tmy = gy+sin(tma)*scale;
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

            tmx = float(gx+cos(tma)*scale*1.1);
            tmy = float(gy+sin(tma)*scale*1.1);
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

            tmx = gx+cos(tma)*scale;
            tmy = gy+sin(tma)*scale;
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

            tma = float(degToRad((i+0)*float(degSegment/3)) + ao - 90.0);
            tmx = float(gx+cos(tma)*scale*1.1);
            tmy = float(gy+sin(tma)*scale*1.1);
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));
         }
      } else {
         for (int j = 0; j < 6; j++) {
            for (int i = 0; i < circleSegments*3; i++) {
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(100.0));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(100.0));
            }
         }
      }

      // Update Bulb Markers
      // Draw Color Wheel + Outline + BulbMarkers if 'features' == 2
      int iUlim = (circleSegments*numBulbs)/3;
      if (features >= 2) {
         degSegment = 360/((circleSegments*numBulbs)/3);
         for (int j = 0; j < numBulbs; j++) {
            tmx = float(gx+cos(degToRad(-90 - j*(angOffset) + 180/numBulbs + ao))*1.05*scale);
            tmy = float(gy+sin(degToRad(-90 - j*(angOffset) + 180/numBulbs + ao))*1.05*scale);
               for (int i = 0; i < iUlim; i++) {
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx + scale*0.16*cos(degToRad(i*degSegment))));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy + scale*0.16*sin(degToRad(i*degSegment))));

               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx + scale*0.16*cos(degToRad((i+1)*degSegment))));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy + scale*0.16*sin(degToRad((i+1)*degSegment))));
            }
         }
      } else {
         for (int j = 0; j < numBulbs; j++) {
            for (int k = 0; k < 6; k++) {
               for (int i = 0; i < iUlim; i++) {
                  /* X */ iconCircleVertexBuffer[vertIndex++] = (float(100.0));
                  /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(100.0));
               }
            }
         }
      }

      // Draw Halos for bulb Markers
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos if 'features' == 3
      if (features >= 3) {
         for (int j = 0; j < numBulbs; j++) {
            tmx = float(gx+cos(degToRad(-90 - j*(angOffset) + 180/numBulbs + ao))*1.05*scale);
            tmy = float(gy+sin(degToRad(-90 - j*(angOffset) + 180/numBulbs + ao))*1.05*scale);
            for (int i = 0; i < iUlim; i++) {
               tma = float(degToRad(i*float(degSegment)) + ao - 90.0);
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx+cos(tma)*scale*0.22));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy+sin(tma)*scale*0.22));

               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx+cos(tma)*scale*0.29));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy+sin(tma)*scale*0.29));

               tma = float(degToRad((i+1)*float(degSegment)) + ao - 90.0);
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx+cos(tma)*scale*0.22));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy+sin(tma)*scale*0.22));

               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx+cos(tma)*scale*0.29));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy+sin(tma)*scale*0.29));

               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx+cos(tma)*scale*0.22));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy+sin(tma)*scale*0.22));

               tma = float(degToRad((i+0)*float(degSegment)) + ao - 90.0);
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx+cos(tma)*scale*0.29));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy+sin(tma)*scale*0.29));
            }
         }
      } else {
         for (int j = 0; j < numBulbs; j++) {
            for (int k = 0; k < 6; k++) {
               for (int i = 0; i < iUlim; i++) {
                  /* X */ iconCircleVertexBuffer[vertIndex++] = float(100.0);
                  /* Y */ iconCircleVertexBuffer[vertIndex++] = float(100.0);
               }
            }
         }
      }

      // Update Grand (Room) Outline
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos + Grand Halo if 'features' == 4
      circleSegments = 60;
      degSegment = 360/60;
      if (features >= 4) {
         if (prevIconCircleW2H != w2h) {
            for (int i = 0; i < circleSegments; i++) {
               tma = float(degToRad(i*float(degSegment)) + ao - 90.0);
               tmx = float(gx+cos(tma)*scale*1.28);
               tmy = float(gy+sin(tma)*scale*1.28);
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

               tmx = float(gx+cos(tma)*scale*1.36);
               tmy = float(gy+sin(tma)*scale*1.36);
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

               tma = float(degToRad((i+1)*float(degSegment)) + ao - 90.0);
               tmx = float(gx+cos(tma)*scale*1.28);
               tmy = float(gy+sin(tma)*scale*1.28);
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

               tmx = float(gx+cos(tma)*scale*1.36);
               tmy = float(gy+sin(tma)*scale*1.36);
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

               tmx = float(gx+cos(tma)*scale*1.28);
               tmy = float(gy+sin(tma)*scale*1.28);
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));

               tma = float(degToRad((i+0)*float(degSegment)) + ao - 90.0);
               tmx = float(gx+cos(tma)*scale*1.36);
               tmy = float(gy+sin(tma)*scale*1.36);
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));
            }
         }
      } else {
         for (int i = 0; i < circleSegments; i++) {
            for (int j = 0; j < 6; j++) {
               /* X */ iconCircleVertexBuffer[vertIndex++] = (float(100.0));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(100.0));
            }
         }
      }

      prevIconCircleNumBulbs = numBulbs;
      prevIconCircleAngularOffset = ao;
      prevIconCircleW2H = w2h;
      prevIconCircleFeatures = features;
   }
   // Geometry already calculated, update colors
   else 
   {
      /*
       * Iterate through each color channel 
       * 0 - RED
       * 1 - GREEN
       * 2 - BLUE
       */
      for (int i = 0; i < 3; i++) {
         
         // Update color wheel, if needed
         for (int j = 0; j < numBulbs; j++) {

            if (float(bulbColors[i+j*3]) != iconCircleColorBuffer[ i + j*circleSegments*9 ]) {
               unsigned int tmu = circleSegments*3;
#              pragma omp parallel for
               for (unsigned int k = 0; k < tmu; k++) {
                     iconCircleColorBuffer[ j*circleSegments*9 + k*3 + i ] = float(bulbColors[i+j*3]);
               }
            }
         }

         // Update outline, bulb markers, if needed
         if (float(detailColor[i]) != float(iconCircleColorBuffer[i + numBulbs*circleSegments*9])) {
            for (unsigned int k = numBulbs*circleSegments*3; k < iconCircleVerts; k++) {
               iconCircleColorBuffer[ k*3 + i ] = float(detailColor[i]);
            }
         }
      }
   }

   delete [] bulbColors;

   glColorPointer(3, GL_FLOAT, 0, iconCircleColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, iconCircleVertexBuffer);
   glDrawElements( GL_TRIANGLES, iconCircleVerts, GL_UNSIGNED_SHORT, iconCircleIndices);

   Py_RETURN_NONE;
}

GLfloat  *homeLinearVertexBuffer = NULL;
GLfloat  *homeLinearColorBuffer  = NULL;
GLushort *homeLinearIndices      = NULL;
GLuint   homeLinearVerts         = NULL;
int      prevHomeLinearNumBulbs  = NULL;

PyObject* drawHomeLinear_drawArn(PyObject *self, PyObject *args) {
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double *bulbColors;
   float gx, gy, wx, wy, ao, w2h;
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
   char circleSegments = 60/numBulbs;

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new double[numBulbs*3];
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = double(PyFloat_AsDouble(py_float));
      }
   }

   if (homeLinearVertexBuffer    == NULL ||
       homeLinearColorBuffer     == NULL ||
       homeLinearIndices         == NULL ||
       homeLinearVerts           == NULL ||
       prevHomeLinearNumBulbs    != numBulbs 
       ){

      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy;
      float offset = float(4.0/numBulbs);
      for (int i = 0; i < numBulbs; i++) {
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

         if (i == numBulbs-1) {
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

         /* X */ verts.push_back(TLx);
         /* Y */ verts.push_back(TLy);
         /* X */ verts.push_back(BLx);
         /* Y */ verts.push_back(BLy);
         /* X */ verts.push_back(TRx);
         /* Y */ verts.push_back(TRy);

         /* X */ verts.push_back(TRx);
         /* Y */ verts.push_back(TRy);
         /* X */ verts.push_back(BLx);
         /* Y */ verts.push_back(BLy);
         /* X */ verts.push_back(BRx);
         /* Y */ verts.push_back(BRy);

         for (int j = 0; j < 6; j++) {
            /* R */ colrs.push_back(float(bulbColors[i*3+0]));
            /* G */ colrs.push_back(float(bulbColors[i*3+1]));
            /* B */ colrs.push_back(float(bulbColors[i*3+2]));
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

      prevHomeLinearNumBulbs = numBulbs;
   } 
   // Geometry already calculated, check if any colors need to be updated.
   else {
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < numBulbs; j++) {
            // 3*2*3:
            // 3 (R,G,B) color values per vertex
            // 2 Triangles per Quad
            // 3 Vertices per Triangle
            if (float(bulbColors[i+j*3]) != homeLinearColorBuffer[i+j*3*2*3]) {
               float tmc = float(bulbColors[j*3+i]);
               for (int k = 0; k < 6; k++) {
                  homeLinearColorBuffer[j*3*2*3 + k*3 + i] = tmc;
               }
            }
         }
      }
   }
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

GLfloat  *iconLinearVertexBuffer = NULL;
GLfloat  *iconLinearColorBuffer  = NULL;
GLushort *iconLinearIndices      = NULL;
GLuint   iconLinearVerts         = NULL;
int      prevIconLinearNumBulbs        = NULL;
int      prevIconLinearFeatures        = NULL;
float    prevIconLinearAngularOffset   = NULL;
float    prevIconLinearWx              = NULL;
float    prevIconLinearWy              = NULL;
float    prevIconLinearW2H             = NULL;

PyObject* drawIconLinear_drawArn(PyObject *self, PyObject *args) {
   PyObject* detailColorPyTup;
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double *bulbColors;
   double detailColor[3];
   float gx, gy, scale, ao, w2h; 
   int numBulbs, features;
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

   if (prevIconLinearNumBulbs != numBulbs ||
       iconLinearVertexBuffer == NULL     ||
       iconLinearColorBuffer  == NULL     ||
       iconLinearIndices      == NULL     ){

      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy;
      float offset = float(2.0/numBulbs);
      char degSegment = 360/circleSegments;

      /*
       * Explanation of features:
       * <= 0: just the color representation
       * <= 1: color representation + outline
       * <= 2: color representation + outline + bulb markers
       * <= 3: color representation + outline + bulb markers + bulb marker halos
       * <= 4: color representation + outline + bulb markers + bulb marker halos + grand halo
       */

      // Define Square of Stripes with Rounded Corners
      for (int i = 0; i < numBulbs; i++) {
         if (i == 0) {
            TLx = -0.75;
            TLy =  1.00;

            BLx = -0.75;
            BLy = -1.00;

            /* X */ verts.push_back(-1.00);
            /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back(-1.00);
            /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back(-0.75);
            /* Y */ verts.push_back( 0.75);

            /* X */ verts.push_back(-0.75);
            /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back(-1.00);
            /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back(-0.75);
            /* Y */ verts.push_back(-0.75);

            // Defines Rounded Corners
            for (int j = 0; j < circleSegments; j++) {
               /* X */ verts.push_back(-0.75);
               /* Y */ verts.push_back( 0.75);
               /* X */ verts.push_back(float(-0.75 + 0.25*cos(degToRad(90+j*(degSegment/4.0)))));
               /* Y */ verts.push_back(float( 0.75 + 0.25*sin(degToRad(90+j*(degSegment/4.0)))));
               /* X */ verts.push_back(float(-0.75 + 0.25*cos(degToRad(90+(j+1)*(degSegment/4.0)))));
               /* Y */ verts.push_back(float( 0.75 + 0.25*sin(degToRad(90+(j+1)*(degSegment/4.0)))));

               /* X */ verts.push_back(-0.75);
               /* Y */ verts.push_back(-0.75);
               /* X */ verts.push_back(float(-0.75 + 0.25*cos(degToRad(180+j*(degSegment/4.0)))));
               /* Y */ verts.push_back(float(-0.75 + 0.25*sin(degToRad(180+j*(degSegment/4.0)))));
               /* X */ verts.push_back(float(-0.75 + 0.25*cos(degToRad(180+(j+1)*(degSegment/4.0)))));
               /* Y */ verts.push_back(float(-0.75 + 0.25*sin(degToRad(180+(j+1)*(degSegment/4.0)))));
               for (int j = 0; j < 6; j++) {
                  /* R */ colrs.push_back(float(bulbColors[i*3+0]));
                  /* G */ colrs.push_back(float(bulbColors[i*3+1]));
                  /* B */ colrs.push_back(float(bulbColors[i*3+2]));
               }
            }

            for (int j = 0; j < 6; j++) {
               /* R */ colrs.push_back(float(bulbColors[i*3+0]));
               /* G */ colrs.push_back(float(bulbColors[i*3+1]));
               /* B */ colrs.push_back(float(bulbColors[i*3+2]));
            }
         } else {
            TLx = float(-1.0 + i*offset);
            TLy =  1.0;

            BLx = float(-1.0 + i*offset);
            BLy = -1.0;
         }

         if (i == numBulbs-1) {
            TRx =  0.75;
            TRy =  1.00;

            BRx =  0.75;
            BRy = -1.00;
            /* X */ verts.push_back( 1.00);
            /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back( 1.00);
            /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back( 0.75);
            /* Y */ verts.push_back( 0.75);

            /* X */ verts.push_back( 0.75);
            /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back( 1.00);
            /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back( 0.75);
            /* Y */ verts.push_back(-0.75);

            // Defines Rounded Corners
            for (int j = 0; j < circleSegments; j++) {
               /* X */ verts.push_back( 0.75);
               /* Y */ verts.push_back( 0.75);
               /* X */ verts.push_back(float( 0.75 + 0.25*cos(degToRad(j*(degSegment/4.0)))));
               /* Y */ verts.push_back(float( 0.75 + 0.25*sin(degToRad(j*(degSegment/4.0)))));
               /* X */ verts.push_back(float( 0.75 + 0.25*cos(degToRad((j+1)*(degSegment/4.0)))));
               /* Y */ verts.push_back(float( 0.75 + 0.25*sin(degToRad((j+1)*(degSegment/4.0)))));

               /* X */ verts.push_back( 0.75);
               /* Y */ verts.push_back(-0.75);
               /* X */ verts.push_back(float( 0.75 + 0.25*cos(degToRad(270+j*(degSegment/4.0)))));
               /* Y */ verts.push_back(float(-0.75 + 0.25*sin(degToRad(270+j*(degSegment/4.0)))));
               /* X */ verts.push_back(float( 0.75 + 0.25*cos(degToRad(270+(j+1)*(degSegment/4.0)))));
               /* Y */ verts.push_back(float(-0.75 + 0.25*sin(degToRad(270+(j+1)*(degSegment/4.0)))));
               for (int j = 0; j < 6; j++) {
                  /* R */ colrs.push_back(float(bulbColors[i*3+0]));
                  /* G */ colrs.push_back(float(bulbColors[i*3+1]));
                  /* B */ colrs.push_back(float(bulbColors[i*3+2]));
               }
            }
            for (int j = 0; j < 6; j++) {
               /* R */ colrs.push_back(float(bulbColors[i*3+0]));
               /* G */ colrs.push_back(float(bulbColors[i*3+1]));
               /* B */ colrs.push_back(float(bulbColors[i*3+2]));
            }
         } else {
            TRx = float(-1.0 + (i+1)*offset);
            TRy =  1.0;

            BRx = float(-1.0 + (i+1)*offset);
            BRy = -1.0;
         }

         // Draw normal rectangular strip for none end segments
         /* X */ verts.push_back(TLx);
         /* Y */ verts.push_back(TLy);
         /* X */ verts.push_back(BLx);
         /* Y */ verts.push_back(BLy);
         /* X */ verts.push_back(TRx);
         /* Y */ verts.push_back(TRy);

         /* X */ verts.push_back(TRx);
         /* Y */ verts.push_back(TRy);
         /* X */ verts.push_back(BLx);
         /* Y */ verts.push_back(BLy);
         /* X */ verts.push_back(BRx);
         /* Y */ verts.push_back(BRy);

         for (int j = 0; j < 6; j++) {
            /* R */ colrs.push_back(float(bulbColors[i*3+0]));
            /* G */ colrs.push_back(float(bulbColors[i*3+1]));
            /* B */ colrs.push_back(float(bulbColors[i*3+2]));
         }

         // Define OutLine
         if (features >= 1) {

            /*
             * Draw Outer Straights
             */
            /* X */ verts.push_back(-9.0/8.0);
            /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back(-9.0/8.0);
            /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back(-1.00);
            /* Y */ verts.push_back( 0.75);

            /* X */ verts.push_back(-1.00);
            /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back(-9.0/8.0);
            /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back(-1.00);
            /* Y */ verts.push_back(-0.75);

            /* X */ verts.push_back( 9.0/8.0);
            /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back( 9.0/8.0);
            /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back( 1.00);
            /* Y */ verts.push_back( 0.75);

            /* X */ verts.push_back( 1.00);
            /* Y */ verts.push_back( 0.75);
            /* X */ verts.push_back( 9.0/8.0);
            /* Y */ verts.push_back(-0.75);
            /* X */ verts.push_back( 1.00);
            /* Y */ verts.push_back(-0.75);

            /* X */ verts.push_back( 0.75);
            /* Y */ verts.push_back(-9.0/8.0);
            /* X */ verts.push_back(-0.75);
            /* Y */ verts.push_back(-9.0/8.0);
            /* X */ verts.push_back( 0.75);
            /* Y */ verts.push_back(-1.00);

            /* X */ verts.push_back( 0.75);
            /* Y */ verts.push_back(-1.00);
            /* X */ verts.push_back(-0.75);
            /* Y */ verts.push_back(-9.0/8.0);
            /* X */ verts.push_back(-0.75);
            /* Y */ verts.push_back(-1.00);

            /* X */ verts.push_back( 0.75);
            /* Y */ verts.push_back( 9.0/8.0);
            /* X */ verts.push_back(-0.75);
            /* Y */ verts.push_back( 9.0/8.0);
            /* X */ verts.push_back( 0.75);
            /* Y */ verts.push_back( 1.00);

            /* X */ verts.push_back( 0.75);
            /* Y */ verts.push_back( 1.00);
            /* X */ verts.push_back(-0.75);
            /* Y */ verts.push_back( 9.0/8.0);
            /* X */ verts.push_back(-0.75);
            /* Y */ verts.push_back( 1.00);
            for (int j = 0; j < 24; j++) {
               /* R */ colrs.push_back(float(detailColor[0]));
               /* G */ colrs.push_back(float(detailColor[1]));
               /* B */ colrs.push_back(float(detailColor[2]));
            }

            /*
             * Draw Rounded Corners
             */
            float tmx, tmy;
            for (int i = 0; i < 4; i++) {
               switch(i) {
                  case 0:
                     tmx =  0.75;
                     tmy =  0.75;
                     break;
                  case 1:
                     tmx = -0.75;
                     tmy =  0.75;
                     break;
                  case 2:
                     tmx = -0.75;
                     tmy = -0.75;
                     break;
                  case 3:
                     tmx =  0.75;
                     tmy = -0.75;
                     break;
               }

               for (int j = 0; j < circleSegments; j++) {
                  /* X */ verts.push_back(float(tmx + 0.25*cos(degToRad(i*90 + j*(degSegment/4.0)))));
                  /* Y */ verts.push_back(float(tmy + 0.25*sin(degToRad(i*90 + j*(degSegment/4.0)))));
                  /* X */ verts.push_back(float(tmx + (0.25+0.125)*cos(degToRad(i*90 + j*(degSegment/4.0)))));
                  /* Y */ verts.push_back(float(tmy + (0.25+0.125)*sin(degToRad(i*90 + j*(degSegment/4.0)))));
                  /* X */ verts.push_back(float(tmx + 0.25*cos(degToRad(i*90 + (j+1)*(degSegment/4.0)))));
                  /* Y */ verts.push_back(float(tmy + 0.25*sin(degToRad(i*90 + (j+1)*(degSegment/4.0)))));


                  /* X */ verts.push_back(float(tmx + 0.25*cos(degToRad(i*90 + (j+1)*(degSegment/4.0)))));
                  /* Y */ verts.push_back(float(tmy + 0.25*sin(degToRad(i*90 + (j+1)*(degSegment/4.0)))));
                  /* X */ verts.push_back(float(tmx + (0.25+0.125)*cos(degToRad(i*90 + j*(degSegment/4.0)))));
                  /* Y */ verts.push_back(float(tmy + (0.25+0.125)*sin(degToRad(i*90 + j*(degSegment/4.0)))));
                  /* X */ verts.push_back(float(tmx + (0.25+0.125)*cos(degToRad(i*90 + (j+1)*(degSegment/4.0)))));
                  /* Y */ verts.push_back(float(tmy + (0.25+0.125)*sin(degToRad(i*90 + (j+1)*(degSegment/4.0)))));
                  for (int k = 0; k < 6; k++) {
                     /* R */ colrs.push_back(float(detailColor[0]));
                     /* G */ colrs.push_back(float(detailColor[1]));
                     /* B */ colrs.push_back(float(detailColor[2]));
                  }
               }
            }
         }

         // Define Bulb Markers
         if (features >= 2) {

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
   }
   
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
