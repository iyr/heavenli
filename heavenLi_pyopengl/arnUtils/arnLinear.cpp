#include <Python.h>
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>
using namespace std;

float constrain(float value, float min, float max) {
   if (value > max)
      return max;
   else if (value < min)
      return min;
   else
      return value;
}

GLfloat  *homeLinearVertexBuffer = NULL;
GLfloat  *homeLinearColorBuffer  = NULL;
GLushort *homeLinearIndices      = NULL;
GLuint   homeLinearVerts;
int      prevHomeLinearNumbulbs;
extern float offScreen;

PyObject* drawHomeLinear_drawArn(PyObject *self, PyObject *args) {
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   float *bulbColors;
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
   bulbColors = new float[numBulbs*3];
//#  pragma omp parallel for
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = float(PyFloat_AsDouble(py_float));
      }
   }

   if (homeLinearVertexBuffer    == NULL ||
       homeLinearColorBuffer     == NULL ||
       homeLinearIndices         == NULL ){

      printf("Generating geometry for homeLinear\n");
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
      printf("homeLinear vertexBuffer length: %.i, Number of vertices: %.i, tris: %.i\n", homeLinearVerts*2, homeLinearVerts, homeLinearVerts/3);

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

//#     pragma omp parallel for
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
//#              pragma omp parallel for
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
GLfloat  *iconLinearBulbVertices = NULL;
GLuint   iconLinearVerts;
int      prevIconLinearNumBulbs;
int      prevIconLinearFeatures;

PyObject* drawIconLinear_drawArn(PyObject *self, PyObject *args) {
   PyObject* detailColorPyTup;
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   float *bulbColors;
   float detailColor[3];
   float gx, gy, scale, ao, w2h, R, G, B;
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

   if (iconLinearVertexBuffer == NULL     ||
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
      if (iconLinearVertexBuffer == NULL) {
         iconLinearVertexBuffer = new GLfloat[iconLinearVerts*2];
      } else {
         delete [] iconLinearVertexBuffer;
         iconLinearVertexBuffer = new GLfloat[iconLinearVerts*2];
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

//#     pragma omp parallel for
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
      //float degSegment = float(360.0/float(circleSegments));
      tmx = 0.0;
      tmy = 0.0;
      vertIndex = 0;

      // Define Square of Stripes with Rounded Corners
//#     pragma omp parallel for
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
//#     pragma omp parallel for
      for (int i = 0; i < 4; i++ ) {
         /* X */ iconLinearVertexBuffer[vertIndex +  0] = iconLinearVertexBuffer[vertIndex +  0] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex +  1] = iconLinearVertexBuffer[vertIndex +  1] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex +  2] = iconLinearVertexBuffer[vertIndex +  2] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex +  3] = iconLinearVertexBuffer[vertIndex +  3] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex +  4] = iconLinearVertexBuffer[vertIndex +  4] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex +  5] = iconLinearVertexBuffer[vertIndex +  5] + tmy;

         /* X */ iconLinearVertexBuffer[vertIndex +  6] = iconLinearVertexBuffer[vertIndex +  6] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex +  7] = iconLinearVertexBuffer[vertIndex +  7] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex +  8] = iconLinearVertexBuffer[vertIndex +  8] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex +  9] = iconLinearVertexBuffer[vertIndex +  9] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex + 10] = iconLinearVertexBuffer[vertIndex + 10] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex + 11] = iconLinearVertexBuffer[vertIndex + 11] + tmy;
         vertIndex += 12;
      }

      /*
       * Draw Rounded Corners
       */
      for (int i = 0; i < 4; i++) {
//#        pragma omp parallel for
         for (int j = 0; j < circleSegments; j++) {
            /* X */ iconLinearVertexBuffer[vertIndex +  0] = iconLinearVertexBuffer[vertIndex +  0] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex +  1] = iconLinearVertexBuffer[vertIndex +  1] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex +  2] = iconLinearVertexBuffer[vertIndex +  2] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex +  3] = iconLinearVertexBuffer[vertIndex +  3] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex +  4] = iconLinearVertexBuffer[vertIndex +  4] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex +  5] = iconLinearVertexBuffer[vertIndex +  5] + tmy;

            /* X */ iconLinearVertexBuffer[vertIndex +  6] = iconLinearVertexBuffer[vertIndex +  6] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex +  7] = iconLinearVertexBuffer[vertIndex +  7] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex +  8] = iconLinearVertexBuffer[vertIndex +  8] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex +  9] = iconLinearVertexBuffer[vertIndex +  9] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex + 10] = iconLinearVertexBuffer[vertIndex + 10] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex + 11] = iconLinearVertexBuffer[vertIndex + 11] + tmy;
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
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 0] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 1] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 2] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 3] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 4] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex++] = iconLinearBulbVertices[j*6 + 5] + tmy;
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
//#        pragma omp parallel for
         for (int j = 0; j < circleSegments; j++) {
            tmj = 6*circleSegments + j*12;
            if (i == 0) {
               /* X */ iconLinearVertexBuffer[vertIndex +  0] = constrain( tmx + iconLinearBulbVertices[  0 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex +  1] =            tmy + iconLinearBulbVertices[  1 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex +  2] = constrain( tmx + iconLinearBulbVertices[  2 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex +  3] =            tmy + iconLinearBulbVertices[  3 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex +  4] = constrain( tmx + iconLinearBulbVertices[  4 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex +  5] =            tmy + iconLinearBulbVertices[  5 + tmj];

               /* X */ iconLinearVertexBuffer[vertIndex +  6] = constrain( tmx + iconLinearBulbVertices[  6 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex +  7] =            tmy + iconLinearBulbVertices[  7 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex +  8] = constrain( tmx + iconLinearBulbVertices[  8 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex +  9] =            tmy + iconLinearBulbVertices[  9 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex + 10] = constrain( tmx + iconLinearBulbVertices[ 10 + tmj], -2.0, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex + 11] =            tmy + iconLinearBulbVertices[ 11 + tmj];
               vertIndex += 12;
            } else if (i == numBulbs-1) {
               /* X */ iconLinearVertexBuffer[vertIndex +  0] = constrain( tmx + iconLinearBulbVertices[  0 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex +  1] =            tmy + iconLinearBulbVertices[  1 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex +  2] = constrain( tmx + iconLinearBulbVertices[  2 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex +  3] =            tmy + iconLinearBulbVertices[  3 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex +  4] = constrain( tmx + iconLinearBulbVertices[  4 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex +  5] =            tmy + iconLinearBulbVertices[  5 + tmj];

               /* X */ iconLinearVertexBuffer[vertIndex +  6] = constrain( tmx + iconLinearBulbVertices[  6 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex +  7] =            tmy + iconLinearBulbVertices[  7 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex +  8] = constrain( tmx + iconLinearBulbVertices[  8 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex +  9] =            tmy + iconLinearBulbVertices[  9 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex + 10] = constrain( tmx + iconLinearBulbVertices[ 10 + tmj], tmx-limit,  2.0);
               /* Y */ iconLinearVertexBuffer[vertIndex + 11] =            tmy + iconLinearBulbVertices[ 11 + tmj];
               vertIndex += 12;
            } else {
               /* X */ iconLinearVertexBuffer[vertIndex +  0] = constrain( tmx + iconLinearBulbVertices[  0 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex +  1] =            tmy + iconLinearBulbVertices[  1 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex +  2] = constrain( tmx + iconLinearBulbVertices[  2 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex +  3] =            tmy + iconLinearBulbVertices[  3 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex +  4] = constrain( tmx + iconLinearBulbVertices[  4 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex +  5] =            tmy + iconLinearBulbVertices[  5 + tmj];

               /* X */ iconLinearVertexBuffer[vertIndex +  6] = constrain( tmx + iconLinearBulbVertices[  6 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex +  7] =            tmy + iconLinearBulbVertices[  7 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex +  8] = constrain( tmx + iconLinearBulbVertices[  8 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex +  9] =            tmy + iconLinearBulbVertices[  9 + tmj];
               /* X */ iconLinearVertexBuffer[vertIndex + 10] = constrain( tmx + iconLinearBulbVertices[ 10 + tmj], tmx-limit, tmx+limit);
               /* Y */ iconLinearVertexBuffer[vertIndex + 11] =            tmy + iconLinearBulbVertices[ 11 + tmj];
               vertIndex += 12;
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

//#     pragma omp parallel for
      for (int i = 0; i < 4; i++ ) {
         /* X */ iconLinearVertexBuffer[vertIndex +  0] = iconLinearVertexBuffer[vertIndex +  0] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex +  1] = iconLinearVertexBuffer[vertIndex +  1] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex +  2] = iconLinearVertexBuffer[vertIndex +  2] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex +  3] = iconLinearVertexBuffer[vertIndex +  3] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex +  4] = iconLinearVertexBuffer[vertIndex +  4] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex +  5] = iconLinearVertexBuffer[vertIndex +  5] + tmy;

         /* X */ iconLinearVertexBuffer[vertIndex +  6] = iconLinearVertexBuffer[vertIndex +  6] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex +  7] = iconLinearVertexBuffer[vertIndex +  7] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex +  8] = iconLinearVertexBuffer[vertIndex +  8] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex +  9] = iconLinearVertexBuffer[vertIndex +  9] + tmy;
         /* X */ iconLinearVertexBuffer[vertIndex + 10] = iconLinearVertexBuffer[vertIndex + 10] + tmx;
         /* Y */ iconLinearVertexBuffer[vertIndex + 11] = iconLinearVertexBuffer[vertIndex + 11] + tmy;
         vertIndex += 12;
      }

      /*
       * Draw Rounded Corners
       */
      for (int i = 0; i < 4; i++) {
//#        pragma omp parallel for
         for (int j = 0; j < circleSegments; j++) {
            /* X */ iconLinearVertexBuffer[vertIndex +  0] = iconLinearVertexBuffer[vertIndex +  0] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex +  1] = iconLinearVertexBuffer[vertIndex +  1] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex +  2] = iconLinearVertexBuffer[vertIndex +  2] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex +  3] = iconLinearVertexBuffer[vertIndex +  3] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex +  4] = iconLinearVertexBuffer[vertIndex +  4] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex +  5] = iconLinearVertexBuffer[vertIndex +  5] + tmy;

            /* X */ iconLinearVertexBuffer[vertIndex +  6] = iconLinearVertexBuffer[vertIndex +  6] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex +  7] = iconLinearVertexBuffer[vertIndex +  7] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex +  8] = iconLinearVertexBuffer[vertIndex +  8] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex +  9] = iconLinearVertexBuffer[vertIndex +  9] + tmy;
            /* X */ iconLinearVertexBuffer[vertIndex + 10] = iconLinearVertexBuffer[vertIndex + 10] + tmx;
            /* Y */ iconLinearVertexBuffer[vertIndex + 11] = iconLinearVertexBuffer[vertIndex + 11] + tmy;
            vertIndex += 12;
         }
      }

      prevIconLinearFeatures = features;
   }

   // Geometry allocated/calculated, check if colors need to be updated
   for (int i = 0; i < 3; i++) {
      for (int j = 0; j < numBulbs; j++) {
         float tmc = float(bulbColors[j*3+i]);
         // Special Case for Rounded Corner Segments
         if (j == 0) {
            if (tmc != iconLinearColorBuffer[i] || prevIconLinearNumBulbs != numBulbs) {
//#              pragma omp parallel for
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
//#              pragma omp parallel for
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
//#              pragma omp parallel for
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
//#           pragma omp parallel for
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
