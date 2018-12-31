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
int      prevIconCircleNumBulbs            = NULL;
float    prevIconCircleAngularOffset       = NULL;
float    prevIconCircleWx                  = NULL;
float    prevIconCircleWy                  = NULL;

PyObject* drawIconCircle_drawArn(PyObject *self, PyObject *args) {
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

   if (iconCircleVertexBuffer == NULL  ||
       iconCircleColorBuffer  == NULL  ||
       iconCircleIndices      == NULL ) {

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma, tmx, tmy;
      sx = sqrt(w2h)*hypot(wx, wy);
      sy = sqrt(wy/wx)*hypot(wx, wy);
      
      for (int j = 0; j < numBulbs; j++) {
         for (int i = 0; i < circleSegments; i++) {
            /* X */ verts.push_back(float(gx));
            /* Y */ verts.push_back(float(gy));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            tma = float(degToRad(i*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = gx+cos(tma)*sx;
            tmy = gy+sin(tma)*sy;
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            tma = float(degToRad((i+1)*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = gx+cos(tma)*sx;
            tmy = gy+sin(tma)*sy;
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));
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
      prevIconCircleWx = wx;
      prevIconCircleWy = wy;
   } 
   else
   // Update Geometry, if alreay allocated
   if (prevIconCircleNumBulbs        != numBulbs   ||
       prevIconCircleAngularOffset   != ao         ||
       prevIconCircleWx              != wx         ||
       prevIconCircleWy              != wy         ){

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma, tmx, tmy;
      int vertIndex = 0;
      int colorIndex = 0;
      sx = sqrt(w2h)*hypot(wx, wy);
      sy = sqrt(wy/wx)*hypot(wx, wy);
      
      for (int j = 0; j < numBulbs; j++) {
         for (int i = 0; i < circleSegments; i++) {
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(gx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(gy));
            /* R */ iconCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+0]));
            /* G */ iconCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+1]));
            /* B */ iconCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+2]));

            tma = float(degToRad(i*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = gx+cos(tma)*sx;
            tmy = gy+sin(tma)*sy;
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));
            /* R */ iconCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+0]));
            /* G */ iconCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+1]));
            /* B */ iconCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+2]));

            tma = float(degToRad((i+1)*float(degSegment/(numBulbs)) + ao + (j)*(angOffset) - 90.0));
            tmx = gx+cos(tma)*sx;
            tmy = gy+sin(tma)*sy;
            /* X */ iconCircleVertexBuffer[vertIndex++] = (float(tmx));
            /* Y */ iconCircleVertexBuffer[vertIndex++] = (float(tmy));
            /* R */ iconCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+0]));
            /* G */ iconCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+1]));
            /* B */ iconCircleColorBuffer[colorIndex++] = (float(bulbColors[j*3+2]));
         }
      }

      prevIconCircleNumBulbs = numBulbs;
      prevIconCircleAngularOffset = ao;
      prevIconCircleWx = wx;
      prevIconCircleWy = wy;

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

            if (float(bulbColors[i+j*3]) != iconCircleColorBuffer[ i + j*circleSegments*9 ]) {
               unsigned int tmu = circleSegments*3;
#              pragma omp parallel for
               for (unsigned int k = 0; k < tmu; k++) {
                     iconCircleColorBuffer[ j*circleSegments*9 + k*3 + i ] = float(bulbColors[i+j*3]);
               }
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

PyObject* drawHomeLinear_drawArn(PyObject *self, PyObject *args) {
   Py_RETURN_NONE;
}

PyObject* drawIconLinear_drawArn(PyObject *self, PyObject *args) {
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
