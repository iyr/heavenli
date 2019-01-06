#include <Python.h>
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>
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
GLuint   homeCircleVerts               = NULL;

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
         for (int i = 0; i < circleSegments/numBulbs; i++) {
            /* X */ verts.push_back(float(0.0));
            /* Y */ verts.push_back(float(0.0));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            tma = float(degToRad(i*float(degSegment) + j*angOffset - 90.0));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            tma = float(degToRad((i+1)*float(degSegment) + j*angOffset - 90.0));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
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
#        pragma omp parallel for
         for (int k = 0; k < (60/numBulbs)*3; k++) {
            if (float(bulbColors[ i + j*3 ]) != homeCircleColorBuffer[ i + k*3 + j*(60/numBulbs)*9 ]) {
               homeCircleColorBuffer[ j*(60/numBulbs)*9 + k*3 + i ] = float(bulbColors[i+j*3]);
            }
         }
      }
   }

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

GLfloat  *iconCircleVertexBuffer       = NULL;
GLfloat  *iconCircleColorBuffer        = NULL;
GLushort *iconCircleIndices            = NULL;
GLuint   iconCircleVerts               = NULL;
int      prevIconCircleNumBulbs        = NULL;
int      prevIconCircleFeatures        = NULL;

PyObject* drawIconCircle_drawArn(PyObject *self, PyObject *args) {
   PyObject* detailColorPyTup;
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double *bulbColors;
   double detailColor[3];
   float gx, gy, scale, ao, w2h, delta;
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

   char circleSegments = 60;

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

   if (iconCircleVertexBuffer == NULL     ||
       iconCircleColorBuffer  == NULL     ||
       iconCircleIndices      == NULL     ){

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma, tmx, tmy, R, G, B, delta;
      /*
       * Explanation of features:
       * <= 0: just the color representation
       * <= 1: color representation + outline
       * <= 2: color representation + outline + bulb markers
       * <= 3: color representation + outline + bulb markers + bulb marker halos
       * <= 4: color representation + outline + bulb markers + bulb marker halos + grand halo
       */
      // Draw Only the color wheel if 'features' <= 0
      delta = degSegment;
      for (int j = 0; j < numBulbs; j++) {
         R = float(bulbColors[j*3+0]);
         G = float(bulbColors[j*3+1]);
         B = float(bulbColors[j*3+2]);
         for (int i = 0; i < circleSegments/numBulbs; i++) {
            /* X */ verts.push_back(float(0.0));
            /* Y */ verts.push_back(float(0.0));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            tma = float(degToRad(i*delta + j*angOffset - 90.0));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            tma = float(degToRad((i+1)*delta + j*angOffset - 90.0));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
         }
      }

      // Draw Color Wheel + Outline if 'features' == 1
      R = float(detailColor[0]);
      G = float(detailColor[1]);
      B = float(detailColor[2]);
      if (features >= 1) {
         delta = float(degSegment);
#        pragma omp parallel for
         for (int i = 0; i < circleSegments; i++) {
            tma = float(degToRad(i*delta));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            /* X */ verts.push_back(float(cos(tma)*1.1));
            /* Y */ verts.push_back(float(sin(tma)*1.1));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            tma = float(degToRad((i+1)*delta));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            /* X */ verts.push_back(float(cos(tma)*1.1));
            /* Y */ verts.push_back(float(sin(tma)*1.1));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            tma = float(degToRad((i+0)*delta));
            /* X */ verts.push_back(float(cos(tma)*1.1));
            /* Y */ verts.push_back(float(sin(tma)*1.1));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
         }
      } else {
         for (int j = 0; j < 6; j++) {
#           pragma omp parallel for
            for (int i = 0; i < circleSegments; i++) {
               /* X */ verts.push_back(float(100.0));
               /* Y */ verts.push_back(float(100.0));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
            }
         }
      }

      // Draw Color Wheel + Outline + BulbMarkers if 'features' >= 2
      int iUlim = circleSegments/3;
      int tmo = 180/numBulbs;
      if (features >= 2) {
         degSegment = 360/iUlim;
         for (int j = 0; j < 6; j++) {
            if (j < numBulbs) {
               tmx = float(cos(degToRad(-90 - j*(angOffset) + tmo))*1.05);
               tmy = float(sin(degToRad(-90 - j*(angOffset) + tmo))*1.05);
            } else {
               tmx = 100;
               tmy = 100;
            }
#           pragma omp parallel for
            for (int i = 0; i < iUlim; i++) {
               /* X */ verts.push_back(float(tmx));
               /* Y */ verts.push_back(float(tmy));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);

               /* X */ verts.push_back(float(tmx + 0.16*cos(degToRad(i*degSegment))));
               /* Y */ verts.push_back(float(tmy + 0.16*sin(degToRad(i*degSegment))));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);

               /* X */ verts.push_back(float(tmx + 0.16*cos(degToRad((i+1)*degSegment))));
               /* Y */ verts.push_back(float(tmy + 0.16*sin(degToRad((i+1)*degSegment))));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
            }
         }
      } else {
         for (int j = 0; j < numBulbs; j++) {
#           pragma omp parallel for
            for (int i = 0; i < iUlim; i++) {
               /* X */ verts.push_back(float(100.0));
               /* Y */ verts.push_back(float(100.0));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
               /* X */ verts.push_back(float(100.0));
               /* Y */ verts.push_back(float(100.0));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
               /* X */ verts.push_back(float(100.0));
               /* Y */ verts.push_back(float(100.0));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
            }
         }
      }

      // Draw Halos for bulb Markers
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos if 'features' == 3
      if (features >= 3) {
         for (int j = 0; j < 6; j++) {
            if (j < numBulbs) {
               tmx = float(cos(degToRad(-90 - j*(angOffset) + tmo))*1.05);
               tmy = float(sin(degToRad(-90 - j*(angOffset) + tmo))*1.05);
            } else {
               tmx = 100.0;
               tmy = 100.0;
            }
#           pragma omp parallel for
            for (int i = 0; i < iUlim; i++) {
               tma = float(degToRad(i*float(degSegment)));
               /* X */ verts.push_back(float(tmx+cos(tma)*0.22));
               /* Y */ verts.push_back(float(tmy+sin(tma)*0.22));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);

               /* X */ verts.push_back(float(tmx+cos(tma)*0.29));
               /* Y */ verts.push_back(float(tmy+sin(tma)*0.29));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);

               tma = float(degToRad((i+1)*float(degSegment)));
               /* X */ verts.push_back(float(tmx+cos(tma)*0.22));
               /* Y */ verts.push_back(float(tmy+sin(tma)*0.22));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);

               /* X */ verts.push_back(float(tmx+cos(tma)*0.29));
               /* Y */ verts.push_back(float(tmy+sin(tma)*0.29));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);

               /* X */ verts.push_back(float(tmx+cos(tma)*0.22));
               /* Y */ verts.push_back(float(tmy+sin(tma)*0.22));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);

               tma = float(degToRad((i)*float(degSegment)));
               /* X */ verts.push_back(float(tmx+cos(tma)*0.29));
               /* Y */ verts.push_back(float(tmy+sin(tma)*0.29));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
            }
         }
      } else {
         for (int j = 0; j < numBulbs; j++) {
            for (int k = 0; k < 6; k++) {
#              pragma omp parallel for
               for (int i = 0; i < iUlim; i++) {
                  /* X */ verts.push_back(float(100.0));
                  /* Y */ verts.push_back(float(100.0));
                  /* R */ colrs.push_back(R);
                  /* G */ colrs.push_back(G);
                  /* B */ colrs.push_back(B);
               }
            }
         }
      }
      
      // Draw Grand (Room) Halo
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos + Grand Halo if 'features' == 4
      circleSegments = 60;
      if (features >= 4) {
         degSegment = 360/60;
#        pragma omp parallel for
         for (int i = 0; i < circleSegments; i++) {
            tma = float(degToRad(i*float(degSegment)));
            /* X */ verts.push_back(float(cos(tma)*1.28));
            /* Y */ verts.push_back(float(sin(tma)*1.28));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            /* X */ verts.push_back(float(cos(tma)*1.36));
            /* Y */ verts.push_back(float(sin(tma)*1.36));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            tma = float(degToRad((i+1)*float(degSegment)));
            /* X */ verts.push_back(float(cos(tma)*1.28));
            /* Y */ verts.push_back(float(sin(tma)*1.28));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);


            /* X */ verts.push_back(float(cos(tma)*1.36));
            /* Y */ verts.push_back(float(sin(tma)*1.36));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            /* X */ verts.push_back(float(cos(tma)*1.28));
            /* Y */ verts.push_back(float(sin(tma)*1.28));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);

            tma = float(degToRad((i)*float(degSegment)));
            /* X */ verts.push_back(float(cos(tma)*1.36));
            /* Y */ verts.push_back(float(sin(tma)*1.36));
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
         }
      } else {
         for (int k = 0; k < 6; k++) {
#           pragma omp parallel for
            for (int i = 0; i < circleSegments; i++) {
               /* X */ verts.push_back(float(100.0));
               /* Y */ verts.push_back(float(100.0));
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
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
      prevIconCircleFeatures = features;
   } 
   else
   // Update Geometry, if alreay allocated
   if (prevIconCircleFeatures != features ||
       prevIconCircleNumBulbs != numBulbs ){

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma, tmx, tmy;
      int vertIndex = 0;
      int colorIndex = 0;
      
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
         delta = float(degSegment);
         for (int i = 0; i < circleSegments; i++) {
            tma = float(degToRad(i*delta));
            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma));

            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma)*1.1);
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma)*1.1);

            tma = float(degToRad((i+1)*delta));
            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma));

            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma)*1.1);
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma)*1.1);

            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma));

            tma = float(degToRad((i+0)*delta));
            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma)*1.1);
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma)*1.1);
         }
      } else {
         for (int j = 0; j < 6; j++) {
            for (int i = 0; i < circleSegments; i++) {
               /* X */ iconCircleVertexBuffer[vertIndex++] = float(100.0);
               /* Y */ iconCircleVertexBuffer[vertIndex++] = float(100.0);
            }
         }
      }

      // Update Bulb Markers
      // Draw Color Wheel + Outline + BulbMarkers if 'features' == 2
      int iUlim = circleSegments/3;
      if (features >= 2) {
         degSegment = 360/iUlim;
         for (int j = 0; j < 6; j++) {
            if (j < numBulbs) {
               tmx = float(cos(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
               tmy = float(sin(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
            } else {
               tmx = 100;
               tmy = 100;
            }
            for (int i = 0; i < iUlim; i++) {
               /* X */ iconCircleVertexBuffer[vertIndex++] = float(tmx);
               /* Y */ iconCircleVertexBuffer[vertIndex++] = float(tmy);

               /* X */ iconCircleVertexBuffer[vertIndex++] = float(tmx + 0.16*cos(degToRad(i*degSegment)));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = float(tmy + 0.16*sin(degToRad(i*degSegment)));

               /* X */ iconCircleVertexBuffer[vertIndex++] = float(tmx + 0.16*cos(degToRad((i+1)*degSegment)));
               /* Y */ iconCircleVertexBuffer[vertIndex++] = float(tmy + 0.16*sin(degToRad((i+1)*degSegment)));
            }
         }
      } else {
         for (int j = 0; j < numBulbs; j++) {
            for (int k = 0; k < 3; k++) {
               for (int i = 0; i < iUlim; i++) {
                  /* X */ iconCircleVertexBuffer[vertIndex++] = float(100.0);
                  /* Y */ iconCircleVertexBuffer[vertIndex++] = float(100.0);
               }
            }
         }
      }

      // Draw Halos for bulb Markers
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos if 'features' == 3
      if (features >= 3) {
         for (int j = 0; j < 6; j++) {
            if (j < numBulbs) {
               tmx = float(cos(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
               tmy = float(sin(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
            } else {
               tmx = 100.0;
               tmy = 100.0;
            }
#           pragma omp parallel for
            for (int i = 0; i < iUlim; i++) {
               tma = float(degToRad(i*float(degSegment)));
               /* X */ iconCircleVertexBuffer[vertIndex++] = float(tmx+cos(tma)*0.22);
               /* Y */ iconCircleVertexBuffer[vertIndex++] = float(tmy+sin(tma)*0.22);
               /* X */ iconCircleVertexBuffer[vertIndex++] = float(tmx+cos(tma)*0.29);
               /* Y */ iconCircleVertexBuffer[vertIndex++] = float(tmy+sin(tma)*0.29);
               tma = float(degToRad((i+1)*float(degSegment)));
               /* X */ iconCircleVertexBuffer[vertIndex++] = float(tmx+cos(tma)*0.22);
               /* Y */ iconCircleVertexBuffer[vertIndex++] = float(tmy+sin(tma)*0.22);

               /* X */ iconCircleVertexBuffer[vertIndex++] = float(tmx+cos(tma)*0.29);
               /* Y */ iconCircleVertexBuffer[vertIndex++] = float(tmy+sin(tma)*0.29);
               /* X */ iconCircleVertexBuffer[vertIndex++] = float(tmx+cos(tma)*0.22);
               /* Y */ iconCircleVertexBuffer[vertIndex++] = float(tmy+sin(tma)*0.22);
               tma = float(degToRad((i+0)*float(degSegment)));
               /* X */ iconCircleVertexBuffer[vertIndex++] = float(tmx+cos(tma)*0.29);
               /* Y */ iconCircleVertexBuffer[vertIndex++] = float(tmy+sin(tma)*0.29);
            }
         }
      } else {
         for (int j = 0; j < numBulbs; j++) {
            for (int k = 0; k < 6; k++) {
#              pragma omp parallel for
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
#        pragma omp parallel for
         for (int i = 0; i < circleSegments; i++) {
            tma = float(degToRad(i*float(degSegment)));
            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma)*1.28);
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma)*1.28);
            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma)*1.36);
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma)*1.36);
            tma = float(degToRad((i+1)*float(degSegment)));
            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma)*1.28);
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma)*1.28);


            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma)*1.36);
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma)*1.36);
            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma)*1.28);
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma)*1.28);
            tma = float(degToRad((i)*float(degSegment)));
            /* X */ iconCircleVertexBuffer[vertIndex++] = float(cos(tma)*1.36);
            /* Y */ iconCircleVertexBuffer[vertIndex++] = float(sin(tma)*1.36);
         }
      } else {
         for (int j = 0; j < 6; j++) {
#           pragma omp parallel for
            for (int i = 0; i < circleSegments; i++) {
               /* X */ iconCircleVertexBuffer[vertIndex++] = float(100.0);
               /* Y */ iconCircleVertexBuffer[vertIndex++] = float(100.0);
            }
         }
      }

      prevIconCircleNumBulbs = numBulbs;
      prevIconCircleFeatures = features;
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
#        pragma omp parallel for
         for (int k = 0; k < tmo/3; k++) {
            if (float(bulbColors[i+j*3]) != iconCircleColorBuffer[i + k*3 + j*tmo]){
               iconCircleColorBuffer[j*tmo + k*3 + i] = float(bulbColors[i+j*3]);
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
GLuint   homeLinearVerts         = NULL;
int      prevHomeLinearNumBulbs  = NULL;

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
         R = float(bulbColors[i*3+0]);
         G = float(bulbColors[i*3+1]);
         B = float(bulbColors[i*3+2]);
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

GLfloat  *iconLinearVertexBuffer       = NULL;
GLfloat  *iconLinearColorBuffer        = NULL;
GLushort *iconLinearIndices            = NULL;
GLuint   iconLinearVerts               = NULL;
GLuint   iconLinearColrs0              = NULL;
GLuint   iconLinearColrs1              = NULL;
GLuint   iconLinearColrs2              = NULL;
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
   float gx, gy, scale, ao, w2h, R, G, B, delta;
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
       prevIconLinearFeatures != features ||
       iconLinearVertexBuffer == NULL     ||
       iconLinearColorBuffer  == NULL     ||
       iconLinearIndices      == NULL     ){

      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy;
      float offset = float(2.0/numBulbs);
      float degSegment = float(360.0/float(circleSegments));

      /*
       * Explanation of features:
       * <= 0: just the color representation
       * <= 1: color representation + outline
       * <= 2: color representation + outline + bulb markers
       * <= 3: color representation + outline + bulb markers + bulb marker halos
       * <= 4: color representation + outline + bulb markers + bulb marker halos + grand halo
       */

      // Define Square of Stripes with Rounded Corners
      delta = float(degSegment/4.0);
      for (int i = 0; i < numBulbs; i++) {
         R = float(bulbColors[i*3+0]);
         G = float(bulbColors[i*3+1]);
         B = float(bulbColors[i*3+2]);
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

         if (i == numBulbs-1) {
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
         /* X */ verts.push_back(TLx);   /* Y */ verts.push_back(TLy);
         /* X */ verts.push_back(BLx);   /* Y */ verts.push_back(BLy);
         /* X */ verts.push_back(TRx);   /* Y */ verts.push_back(TRy);

         /* X */ verts.push_back(TRx);   /* Y */ verts.push_back(TRy);
         /* X */ verts.push_back(BLx);   /* Y */ verts.push_back(BLy);
         /* X */ verts.push_back(BRx);   /* Y */ verts.push_back(BRy);

         if (i == 0) {
            iconLinearColrs0 = verts.size()/2;
         } 
         if (i == 1) {
            iconLinearColrs1 = verts.size()/2;
         } 
         if (i == numBulbs-1 ) {
            iconLinearColrs2 = verts.size()/2;
         }

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
         /*
          * Draw Outer Straights
          */
         //---------//
         /* X */ verts.push_back(-9.0/8.0);   /* Y */ verts.push_back( 0.75);
         /* X */ verts.push_back(-9.0/8.0);   /* Y */ verts.push_back(-0.75);
         /* X */ verts.push_back(-1.00);      /* Y */ verts.push_back( 0.75);

         /* X */ verts.push_back(-1.00);      /* Y */ verts.push_back( 0.75);
         /* X */ verts.push_back(-9.0/8.0);   /* Y */ verts.push_back(-0.75);
         /* X */ verts.push_back(-1.00);      /* Y */ verts.push_back(-0.75);

         //---------//
         /* X */ verts.push_back( 9.0/8.0);   /* Y */ verts.push_back( 0.75);
         /* X */ verts.push_back( 9.0/8.0);   /* Y */ verts.push_back(-0.75);
         /* X */ verts.push_back( 1.00);      /* Y */ verts.push_back( 0.75);

         /* X */ verts.push_back( 1.00);      /* Y */ verts.push_back( 0.75);
         /* X */ verts.push_back( 9.0/8.0);   /* Y */ verts.push_back(-0.75);
         /* X */ verts.push_back( 1.00);      /* Y */ verts.push_back(-0.75);

         //---------//
         /* X */ verts.push_back( 0.75);   /* Y */ verts.push_back(-9.0/8.0);
         /* X */ verts.push_back(-0.75);   /* Y */ verts.push_back(-9.0/8.0);
         /* X */ verts.push_back( 0.75);   /* Y */ verts.push_back(-1.00);

         /* X */ verts.push_back( 0.75);   /* Y */ verts.push_back(-1.00);
         /* X */ verts.push_back(-0.75);   /* Y */ verts.push_back(-9.0/8.0);
         /* X */ verts.push_back(-0.75);   /* Y */ verts.push_back(-1.00);

         //---------//
         /* X */ verts.push_back( 0.75);   /* Y */ verts.push_back( 9.0/8.0);
         /* X */ verts.push_back(-0.75);   /* Y */ verts.push_back( 9.0/8.0);
         /* X */ verts.push_back( 0.75);   /* Y */ verts.push_back( 1.00);

         /* X */ verts.push_back( 0.75);   /* Y */ verts.push_back( 1.00);
         /* X */ verts.push_back(-0.75);   /* Y */ verts.push_back( 9.0/8.0);
         /* X */ verts.push_back(-0.75);   /* Y */ verts.push_back( 1.00);
         for (int j = 0; j < 24; j++) {
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
         }

         /*
          * Draw Rounded Corners
          */
         float tmx, tmy, ri, ro;
         ri = 0.25;
         ro = 0.125 + 0.25;
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
      } else {
         for (int i = 0; i < (4*6*4 + 4*6*circleSegments); i++) {
            /* X */ verts.push_back(100.0);
            /* Y */ verts.push_back(100.0);
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
         }
      }

      // Define Bulb Markers
      if (features >= 2) {
         float tmx, tmy;
         for (int i = 0; i < numBulbs; i++) {
            if (numBulbs == 1) {
               tmy = (17.0/16.0);
            } else {
               tmy = -(17.0/16.0);
            }
            tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs));
            for (int j = 0; j < circleSegments; j++) {
               /* X */ verts.push_back(tmx);
               /* Y */ verts.push_back(tmy);
               /* X */ verts.push_back(tmx+float((1.0/6.0)*cos(degToRad(j*degSegment))));
               /* Y */ verts.push_back(tmy+float((1.0/6.0)*sin(degToRad(j*degSegment))));
               /* X */ verts.push_back(tmx+float((1.0/6.0)*cos(degToRad((j+1)*degSegment))));
               /* Y */ verts.push_back(tmy+float((1.0/6.0)*sin(degToRad((j+1)*degSegment))));
            }

            for (int j = 0; j < circleSegments*3; j++) {
               /* R */ colrs.push_back(R);
               /* G */ colrs.push_back(G);
               /* B */ colrs.push_back(B);
            }
         }
      } else {
         for (int i = 0; i < circleSegments*numBulbs*3; i++) {
            /* X */ verts.push_back(100.0);
            /* Y */ verts.push_back(100.0);
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
         }
      }

      // Define Bulb Halos
      if (features >= 3) {
         float tmx, tmy, ri, ro, limit;
         for (int i = 0; i < numBulbs; i++) {
            if (numBulbs == 1) {
               tmy = (17.0/16.0);
            } else {
               tmy = -(17.0/16.0);
            }
            tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs));
            limit = float(1.0/float(numBulbs));
            ri = float(13.0/60.0);
            ro = float(17.0/60.0);
            for (int j = 0; j < circleSegments; j++) {
               if (i == 0) {
                  /* X */ verts.push_back(constrain(tmx+float(ri*cos(degToRad(j*degSegment))),       -2.0, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ri*sin(degToRad(j*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ro*cos(degToRad(j*degSegment))),       -2.0, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ro*sin(degToRad(j*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ri*cos(degToRad((j+1)*degSegment))),   -2.0, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ri*sin(degToRad((j+1)*degSegment))));

                  /* X */ verts.push_back(constrain(tmx+float(ri*cos(degToRad((j+1)*degSegment))),   -2.0, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ri*sin(degToRad((j+1)*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ro*cos(degToRad(j*degSegment))),       -2.0, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ro*sin(degToRad(j*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ro*cos(degToRad((j+1)*degSegment))),   -2.0, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ro*sin(degToRad((j+1)*degSegment))));
               } else if (i == numBulbs-1) {
                  /* X */ verts.push_back(constrain(tmx+float(ri*cos(degToRad(j*degSegment))),       tmx-limit, 2.0));
                  /* Y */ verts.push_back(tmy+float(ri*sin(degToRad(j*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ro*cos(degToRad(j*degSegment))),       tmx-limit, 2.0));
                  /* Y */ verts.push_back(tmy+float(ro*sin(degToRad(j*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ri*cos(degToRad((j+1)*degSegment))),   tmx-limit, 2.0));
                  /* Y */ verts.push_back(tmy+float(ri*sin(degToRad((j+1)*degSegment))));

                  /* X */ verts.push_back(constrain(tmx+float(ri*cos(degToRad((j+1)*degSegment))),   tmx-limit, 2.0));
                  /* Y */ verts.push_back(tmy+float(ri*sin(degToRad((j+1)*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ro*cos(degToRad(j*degSegment))),       tmx-limit, 2.0));
                  /* Y */ verts.push_back(tmy+float(ro*sin(degToRad(j*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ro*cos(degToRad((j+1)*degSegment))),   tmx-limit, 2.0));
                  /* Y */ verts.push_back(tmy+float(ro*sin(degToRad((j+1)*degSegment))));
               } else {
                  /* X */ verts.push_back(constrain(tmx+float(ri*cos(degToRad(j*degSegment))),       tmx-limit, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ri*sin(degToRad(j*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ro*cos(degToRad(j*degSegment))),       tmx-limit, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ro*sin(degToRad(j*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ri*cos(degToRad((j+1)*degSegment))),   tmx-limit, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ri*sin(degToRad((j+1)*degSegment))));

                  /* X */ verts.push_back(constrain(tmx+float(ri*cos(degToRad((j+1)*degSegment))),   tmx-limit, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ri*sin(degToRad((j+1)*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ro*cos(degToRad(j*degSegment))),       tmx-limit, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ro*sin(degToRad(j*degSegment))));
                  /* X */ verts.push_back(constrain(tmx+float(ro*cos(degToRad((j+1)*degSegment))),   tmx-limit, tmx+limit));
                  /* Y */ verts.push_back(tmy+float(ro*sin(degToRad((j+1)*degSegment))));
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
      } else {
         for (int i = 0; i < numBulbs*6*circleSegments; i++) {
            /* X */ verts.push_back(100.0);
            /* Y */ verts.push_back(100.0);
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
         }
      }

      // Define Grand Outline
      if (features >= 4) {

         /*
          * Draw Outer Straights
          */

         /* X */ verts.push_back(-0.75);  /* Y */ verts.push_back(float( (17.0/16.0 + 17.0/60.0)));
         /* X */ verts.push_back(-0.75);  /* Y */ verts.push_back(float( (17.0/16.0 + 13.0/60.0)));
         /* X */ verts.push_back( 0.75);  /* Y */ verts.push_back(float( (17.0/16.0 + 17.0/60.0)));

         /* X */ verts.push_back( 0.75);  /* Y */ verts.push_back(float( (17.0/16.0 + 13.0/60.0)));
         /* X */ verts.push_back( 0.75);  /* Y */ verts.push_back(float( (17.0/16.0 + 17.0/60.0)));
         /* X */ verts.push_back(-0.75);  /* Y */ verts.push_back(float( (17.0/16.0 + 13.0/60.0)));

         /* X */ verts.push_back(-0.75);  /* Y */ verts.push_back(float(-(17.0/16.0 + 17.0/60.0)));
         /* X */ verts.push_back(-0.75);  /* Y */ verts.push_back(float(-(17.0/16.0 + 13.0/60.0)));
         /* X */ verts.push_back( 0.75);  /* Y */ verts.push_back(float(-(17.0/16.0 + 17.0/60.0)));

         /* X */ verts.push_back( 0.75);  /* Y */ verts.push_back(float(-(17.0/16.0 + 13.0/60.0)));
         /* X */ verts.push_back( 0.75);  /* Y */ verts.push_back(float(-(17.0/16.0 + 17.0/60.0)));
         /* X */ verts.push_back(-0.75);  /* Y */ verts.push_back(float(-(17.0/16.0 + 13.0/60.0)));

         /* X */ verts.push_back(float( (17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back(-0.75);
         /* X */ verts.push_back(float( (17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back(-0.75);
         /* X */ verts.push_back(float( (17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back( 0.75);

         /* X */ verts.push_back(float( (17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back( 0.75);
         /* X */ verts.push_back(float( (17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back( 0.75);
         /* X */ verts.push_back(float( (17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back(-0.75);

         /* X */ verts.push_back(float(-(17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back(-0.75);
         /* X */ verts.push_back(float(-(17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back(-0.75);
         /* X */ verts.push_back(float(-(17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back( 0.75);

         /* X */ verts.push_back(float(-(17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back( 0.75);
         /* X */ verts.push_back(float(-(17.0/16.0 + 17.0/60.0)));  /* Y */ verts.push_back( 0.75);
         /* X */ verts.push_back(float(-(17.0/16.0 + 13.0/60.0)));  /* Y */ verts.push_back(-0.75);

         for (int j = 0; j < 24; j++) {
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
         }

         /*
          * Draw Rounded Corners
          */
         float tmx, tmy, ri, ro;
         ri = float(5.0/16.0+13.0/60.0);
         ro = float(5.0/16.0+17.0/60.0);
         delta = float(degSegment/4.0);
         for (int i = 0; i < 4; i++) {
            switch(i) {
               case 0:
                  tmx =  0.75;   tmy =  0.75;
                  break;
               case 1:
                  tmx = -0.75;   tmy =  0.75;
                  break;
               case 2:
                  tmx = -0.75;   tmy = -0.75;
                  break;
               case 3:
                  tmx =  0.75;   tmy = -0.75;
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
      } else {
         for (int i = 0; i < (4*6*4 + 4*6*circleSegments); i++) {
            /* X */ verts.push_back(100.0);
            /* Y */ verts.push_back(100.0);
            /* R */ colrs.push_back(R);
            /* G */ colrs.push_back(G);
            /* B */ colrs.push_back(B);
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
   if (prevIconLinearFeatures != features) {
      int vertIndex = 0;
      int colorIndex = 0;

      prevIconLinearFeatures = features;
      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy;
      float offset = float(2.0/numBulbs);
      float degSegment = float(360.0/float(circleSegments));

      /*
       * Explanation of features:
       * <= 0: just the color representation
       * <= 1: color representation + outline
       * <= 2: color representation + outline + bulb markers
       * <= 3: color representation + outline + bulb markers + bulb marker halos
       * <= 4: color representation + outline + bulb markers + bulb marker halos + grand halo
       */

      // Define Square of Stripes with Rounded Corners
      delta = float(degSegment/4.0);
      for (int i = 0; i < numBulbs; i++) {
         if (i == 0) {
            TLx = -0.75;
            TLy =  1.00;

            BLx = -0.75;
            BLy = -1.00;

            /* X */ iconLinearVertexBuffer[vertIndex++] = (-1.00);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
            /* X */ iconLinearVertexBuffer[vertIndex++] = (-1.00);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
            /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);

            /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
            /* X */ iconLinearVertexBuffer[vertIndex++] = (-1.00);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
            /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);

            // Defines Rounded Corners
            for (int j = 0; j < circleSegments; j++) {
               /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
               /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
               /* X */ iconLinearVertexBuffer[vertIndex++] = float(-0.75 + 0.25*cos(degToRad(90+j*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float( 0.75 + 0.25*sin(degToRad(90+j*delta)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = float(-0.75 + 0.25*cos(degToRad(90+(j+1)*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float( 0.75 + 0.25*sin(degToRad(90+(j+1)*delta)));

               /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
               /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
               /* X */ iconLinearVertexBuffer[vertIndex++] = float(-0.75 + 0.25*cos(degToRad(180+j*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float(-0.75 + 0.25*sin(degToRad(180+j*delta)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = float(-0.75 + 0.25*cos(degToRad(180+(j+1)*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float(-0.75 + 0.25*sin(degToRad(180+(j+1)*delta)));
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
            /* X */ iconLinearVertexBuffer[vertIndex++] = ( 1.00);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
            /* X */ iconLinearVertexBuffer[vertIndex++] = ( 1.00);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
            /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);

            /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
            /* X */ iconLinearVertexBuffer[vertIndex++] = ( 1.00);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
            /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);

            // Defines Rounded Corners
            for (int j = 0; j < circleSegments; j++) {
               /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
               /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
               /* X */ iconLinearVertexBuffer[vertIndex++] = float( 0.75 + 0.25*cos(degToRad(j*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float( 0.75 + 0.25*sin(degToRad(j*delta)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = float( 0.75 + 0.25*cos(degToRad((j+1)*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float( 0.75 + 0.25*sin(degToRad((j+1)*delta)));

               /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
               /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
               /* X */ iconLinearVertexBuffer[vertIndex++] = float( 0.75 + 0.25*cos(degToRad(270+j*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float(-0.75 + 0.25*sin(degToRad(270+j*delta)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = float( 0.75 + 0.25*cos(degToRad(270+(j+1)*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float(-0.75 + 0.25*sin(degToRad(270+(j+1)*delta)));
            }
         } else {
            TRx = float(-1.0 + (i+1)*offset);
            TRy =  1.0;

            BRx = float(-1.0 + (i+1)*offset);
            BRy = -1.0;
         }

         // Draw normal rectangular strip for non-end segments
         /* X */ iconLinearVertexBuffer[vertIndex++] = TLx;   /* Y */ iconLinearVertexBuffer[vertIndex++] = TLy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = BLx;   /* Y */ iconLinearVertexBuffer[vertIndex++] = BLy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = TRx;   /* Y */ iconLinearVertexBuffer[vertIndex++] = TRy;

         /* X */ iconLinearVertexBuffer[vertIndex++] = TRx;   /* Y */ iconLinearVertexBuffer[vertIndex++] = TRy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = BLx;   /* Y */ iconLinearVertexBuffer[vertIndex++] = BLy;
         /* X */ iconLinearVertexBuffer[vertIndex++] = BRx;   /* Y */ iconLinearVertexBuffer[vertIndex++] = BRy;
      }

      // Define OutLine
      if (features >= 1) {
         /*
          * Draw Outer Straights
          */
         //---------//
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-9.0/8.0);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-9.0/8.0);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-1.00);      /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);

         /* X */ iconLinearVertexBuffer[vertIndex++] = (-1.00);      /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-9.0/8.0);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-1.00);      /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);

         //---------//
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 9.0/8.0);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 9.0/8.0);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 1.00);      /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);

         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 1.00);      /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 9.0/8.0);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 1.00);      /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);

         //---------//
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-9.0/8.0);
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-9.0/8.0);
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-1.00);

         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-1.00);
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-9.0/8.0);
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = (-1.00);

         //---------//
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 9.0/8.0);
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 9.0/8.0);
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 1.00);

         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 1.00);
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 9.0/8.0);
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);   /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 1.00);

         /*
          * Draw Rounded Corners
          */
         float tmx, tmy, ri, ro;
         ri = 0.25;
         ro = 0.125 + 0.25;
         for (int i = 0; i < 4; i++) {
            switch(i) {
               case 0:
                  tmx = ( 0.75);
                  tmy = ( 0.75);
                  break;
               case 1:
                  tmx = (-0.75);
                  tmy = ( 0.75);
                  break;
               case 2:
                  tmx = (-0.75);
                  tmy = (-0.75);
                  break;
               case 3:
                  tmx = ( 0.75);
                  tmy = (-0.75);
                  break;
            }

            for (int j = 0; j < circleSegments; j++) {
               /* X */ iconLinearVertexBuffer[vertIndex++] = float(tmx + ri*cos(degToRad(i*90 + j*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float(tmy + ri*sin(degToRad(i*90 + j*delta)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = float(tmx + ro*cos(degToRad(i*90 + j*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float(tmy + ro*sin(degToRad(i*90 + j*delta)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = float(tmx + ri*cos(degToRad(i*90 + (j+1)*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float(tmy + ri*sin(degToRad(i*90 + (j+1)*delta)));

               /* X */ iconLinearVertexBuffer[vertIndex++] = float(tmx + ri*cos(degToRad(i*90 + (j+1)*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float(tmy + ri*sin(degToRad(i*90 + (j+1)*delta)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = float(tmx + ro*cos(degToRad(i*90 + j*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float(tmy + ro*sin(degToRad(i*90 + j*delta)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = float(tmx + ro*cos(degToRad(i*90 + (j+1)*delta)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = float(tmy + ro*sin(degToRad(i*90 + (j+1)*delta)));
            }
         }
      } else {
         for (int i = 0; i < (4*6*4 + 4*6*circleSegments); i++) {
            /* X */ iconLinearVertexBuffer[vertIndex++] = (100.0);
            /* Y */ iconLinearVertexBuffer[vertIndex++] = (100.0);
         }
      }

      // Define Bulb Markers
      if (features >= 2) {
         float tmx, tmy;
         for (int i = 0; i < numBulbs; i++) {
            if (numBulbs == 1) {
               tmy = (17.0/16.0);
            } else {
               tmy = -(17.0/16.0);
            }
            tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs));
            for (int j = 0; j < circleSegments; j++) {
               /* X */ iconLinearVertexBuffer[vertIndex++] = (tmx);
               /* Y */ iconLinearVertexBuffer[vertIndex++] = (tmy);
               /* X */ iconLinearVertexBuffer[vertIndex++] = tmx+float((1.0/6.0)*cos(degToRad(j*degSegment)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = tmy+float((1.0/6.0)*sin(degToRad(j*degSegment)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = tmx+float((1.0/6.0)*cos(degToRad((j+1)*degSegment)));
               /* Y */ iconLinearVertexBuffer[vertIndex++] = tmy+float((1.0/6.0)*sin(degToRad((j+1)*degSegment)));
            }
         }
      } else {
         for (int i = 0; i < circleSegments*numBulbs*3; i++) {
            /* X */ iconLinearVertexBuffer[vertIndex++] = (100.0);
            /* Y */ iconLinearVertexBuffer[vertIndex++] = (100.0);
         }
      }

      // Define Bulb Halos
      if (features >= 3) {
         float tmx, tmy, ri, ro, limit;
         for (int i = 0; i < numBulbs; i++) {
            if (numBulbs == 1) {
               tmy = (17.0/16.0);
            } else {
               tmy = -(17.0/16.0);
            }
            tmx = float(-1.0 + 1.0/float(numBulbs) + (i*2.0)/float(numBulbs));
            limit = float(1.0/float(numBulbs));
            ri = float(13.0/60.0);
            ro = float(17.0/60.0);
            for (int j = 0; j < circleSegments; j++) {
               if (i == 0) {
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ri*cos(degToRad(j*degSegment))),       -2.0, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ri*sin(degToRad(j*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ro*cos(degToRad(j*degSegment))),       -2.0, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ro*sin(degToRad(j*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ri*cos(degToRad((j+1)*degSegment))),   -2.0, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ri*sin(degToRad((j+1)*degSegment)));

                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ri*cos(degToRad((j+1)*degSegment))),   -2.0, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ri*sin(degToRad((j+1)*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ro*cos(degToRad(j*degSegment))),       -2.0, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ro*sin(degToRad(j*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ro*cos(degToRad((j+1)*degSegment))),   -2.0, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ro*sin(degToRad((j+1)*degSegment)));
               } else if (i == numBulbs-1) {
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ri*cos(degToRad(j*degSegment))),       tmx-limit, 2.0);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ri*sin(degToRad(j*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ro*cos(degToRad(j*degSegment))),       tmx-limit, 2.0);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ro*sin(degToRad(j*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ri*cos(degToRad((j+1)*degSegment))),   tmx-limit, 2.0);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ri*sin(degToRad((j+1)*degSegment)));

                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ri*cos(degToRad((j+1)*degSegment))),   tmx-limit, 2.0);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ri*sin(degToRad((j+1)*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ro*cos(degToRad(j*degSegment))),       tmx-limit, 2.0);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ro*sin(degToRad(j*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ro*cos(degToRad((j+1)*degSegment))),   tmx-limit, 2.0);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ro*sin(degToRad((j+1)*degSegment)));
               } else {
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ri*cos(degToRad(j*degSegment))),       tmx-limit, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ri*sin(degToRad(j*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ro*cos(degToRad(j*degSegment))),       tmx-limit, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ro*sin(degToRad(j*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ri*cos(degToRad((j+1)*degSegment))),   tmx-limit, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ri*sin(degToRad((j+1)*degSegment)));

                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ri*cos(degToRad((j+1)*degSegment))),   tmx-limit, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ri*sin(degToRad((j+1)*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ro*cos(degToRad(j*degSegment))),       tmx-limit, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ro*sin(degToRad(j*degSegment)));
                  /* X */ iconLinearVertexBuffer[vertIndex++] = constrain( tmx+float(ro*cos(degToRad((j+1)*degSegment))),   tmx-limit, tmx+limit);
                  /* Y */ iconLinearVertexBuffer[vertIndex++] =            tmy+float(ro*sin(degToRad((j+1)*degSegment)));
               }
            }
         }
      } else {
         for (int i = 0; i < numBulbs*6*circleSegments; i++) {
            /* X */ iconLinearVertexBuffer[vertIndex++] = (100.0);
            /* Y */ iconLinearVertexBuffer[vertIndex++] = (100.0);
         }
      }

      // Define Grand Outline
      if (features >= 4) {

         /*
          * Draw Outer Straights
          */

         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 17.0/60.0) );
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 13.0/60.0) );
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 17.0/60.0) );

         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 13.0/60.0) );
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 17.0/60.0) );
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 13.0/60.0) );

         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 17.0/60.0) );
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 13.0/60.0) );
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 17.0/60.0) );

         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 13.0/60.0) );
         /* X */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 17.0/60.0) );
         /* X */ iconLinearVertexBuffer[vertIndex++] = (-0.75);  /* Y */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 13.0/60.0) );

         /* X */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 17.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 13.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 17.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);

         /* X */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 13.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 17.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = float( (17.0/16.0 + 13.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);

         /* X */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 17.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 13.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 17.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);

         /* X */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 13.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 17.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = ( 0.75);
         /* X */ iconLinearVertexBuffer[vertIndex++] = float(-(17.0/16.0 + 13.0/60.0) );  /* Y */ iconLinearVertexBuffer[vertIndex++] = (-0.75);

         /*
          * Draw Rounded Corners
          */
         float tmx, tmy, ri, ro;
         ri = float(5.0/16.0+13.0/60.0);
         ro = float(5.0/16.0+17.0/60.0);
         delta = float(degSegment/4.0);
         for (int i = 0; i < 4; i++) {
            switch(i) {
               case 0:
                  tmx =  0.75;   tmy =  0.75;
                  break;
               case 1:
                  tmx = -0.75;   tmy =  0.75;
                  break;
               case 2:
                  tmx = -0.75;   tmy = -0.75;
                  break;
               case 3:
                  tmx =  0.75;   tmy = -0.75;
                  break;
            }

            for (int j = 0; j < circleSegments; j++) {
               float j0 = float(degToRad(i*90 + j*delta));
               float j1 = float(degToRad(i*90 + (j+1)*delta));
               /* X */ iconLinearVertexBuffer[vertIndex++] = (float(tmx + ri*cos(j0)));  /* Y */ iconLinearVertexBuffer[vertIndex++] = (float(tmy + ri*sin(j0)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = (float(tmx + ro*cos(j0)));  /* Y */ iconLinearVertexBuffer[vertIndex++] = (float(tmy + ro*sin(j0)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = (float(tmx + ri*cos(j1)));  /* Y */ iconLinearVertexBuffer[vertIndex++] = (float(tmy + ri*sin(j1)));

               /* X */ iconLinearVertexBuffer[vertIndex++] = (float(tmx + ri*cos(j1)));  /* Y */ iconLinearVertexBuffer[vertIndex++] = (float(tmy + ri*sin(j1)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = (float(tmx + ro*cos(j0)));  /* Y */ iconLinearVertexBuffer[vertIndex++] = (float(tmy + ro*sin(j0)));
               /* X */ iconLinearVertexBuffer[vertIndex++] = (float(tmx + ro*cos(j1)));  /* Y */ iconLinearVertexBuffer[vertIndex++] = (float(tmy + ro*sin(j1)));
            }
         }
      } else {
         for (int i = 0; i < (4*6*4 + 4*6*circleSegments); i++) {
            /* X */ iconLinearVertexBuffer[vertIndex++] = (100.0);
            /* Y */ iconLinearVertexBuffer[vertIndex++] = (100.0);
         }
      }
   }
   // Geometry allocated/calculated, check if colors need to be updated
   if (true)
   {
      int deltaColrs = iconLinearColrs1 - iconLinearColrs0;
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < numBulbs; j++) {

            // Special Cases for Rounded Corner Segments
            if (j == 0) {
               if (float(bulbColors[i]) != iconLinearColorBuffer[i]) {
                  float tmc = float(bulbColors[i]);
#                 pragma omp parallel for
                  for (unsigned int k = 0; k < iconLinearColrs0; k++) {
                     iconLinearColorBuffer[k*3 + i] = tmc;
                  }
               }
            } else if (j == numBulbs-1) {
               int uLim = iconLinearColrs0 + (j-1)*deltaColrs;
               if (float(bulbColors[i + j*3]) != iconLinearColorBuffer[i + uLim*3]) {
                  float tmc = float(bulbColors[i+j*3]);
#                 pragma omp parallel for
                  for (unsigned int k = uLim; k < iconLinearVerts; k++) {
                     iconLinearColorBuffer[k*3 + i] = tmc;
                  }
               }
            } else

            // General Cases
            {
               if (float(bulbColors[i + j*3]) != iconLinearColorBuffer[i + iconLinearColrs0*3 + (j-1)*deltaColrs*3]) {
                  float tmc = float(bulbColors[i+j*3]);
#                 pragma omp parallel for
                  for (unsigned int k = iconLinearColrs0; k < iconLinearColrs1; k++) {
                     iconLinearColorBuffer[k*3 + i + (j-1)*deltaColrs*3] = tmc;
                  }
               }
            }
         }

         // Check if outline color needs to be updated
         if (float(detailColor[i]) != iconLinearColorBuffer[i+iconLinearColrs2*3]) {
#           pragma omp parallel for
            for (unsigned int k = iconLinearColrs2; k < iconLinearVerts; k++) {
               iconLinearColorBuffer[k*3+i] = float(detailColor[i]);
            }
         }
      }
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
