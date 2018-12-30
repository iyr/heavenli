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
int prevNumBulbs                 = NULL;
float prevAngularOffset         = NULL;

PyObject* drawHomeCircle_drawArn(PyObject *self, PyObject *args) {
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double *bulbColors;
   float gx, gy, dx, dy, ao, w2h;
   int numBulbs;
   if (!PyArg_ParseTuple(args,
            "fffflffO",
            &gx, &gy,
            &dx, &dy,
            &numBulbs,
            &ao,
            &w2h,
            &py_list
            ))
   {
      Py_RETURN_NONE;
   }
   int wx = 300;
   int wy = 300;
   float angOffset = float(360.0 / float(numBulbs));

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new double[numBulbs*3];
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = double(PyFloat_AsDouble(py_float));
      }
   }

   glPushMatrix();
   glScalef(sqrt(w2h)*hypot(wx, wy), sqrt(wy/wx)*hypot(wx, wy), 1);
   if (prevNumBulbs != numBulbs       ||
       prevAngularOffset != ao        ||
       homeCircleVertexBuffer == NULL ||
       homeCircleColorBuffer  == NULL ||
       homeCircleIndices      == NULL) {

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      char circleSegments = 30;
      char degSegment = 360 / circleSegments;
      float tma, tmx, tmy;
      
      for (int j = 0; j < numBulbs; j++) {
         for (int i = 0; i < circleSegments; i++) {
            /* X */ verts.push_back(float(0.0));
            /* Y */ verts.push_back(float(0.0));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            tma = degToRad(i*float(12.0/float(numBulbs)) + ao + float(j)*(angOffset) - 90.0);
            tmx = cos(tma);
            tmy = sin(tma);
            /* X */ verts.push_back(float(tmx));
            /* Y */ verts.push_back(float(tmy));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            tma = degToRad((i+1)*float(12.0/float(numBulbs)) + ao + float(j)*(angOffset) - 90.0);
            tmx = cos(tma);
            tmy = sin(tma);
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

      prevNumBulbs = numBulbs;
      prevAngularOffset = ao;
   }

   delete [] bulbColors;

   glColorPointer(3, GL_FLOAT, 0, homeCircleColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, homeCircleVertexBuffer);
   glDrawElements( GL_TRIANGLES, homeCircleVerts, GL_UNSIGNED_SHORT, homeCircleIndices);
   glPopMatrix();

   Py_RETURN_NONE;
}

PyObject* drawIconCircle_drawArn(PyObject *self, PyObject *args) {
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
