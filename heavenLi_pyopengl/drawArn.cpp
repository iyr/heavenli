#include <Python.h>
//#define degToRad(angleInDegrees) ((angleInDegrees) * 3.1415926535 / 180.0)
#include "matrixUtils.c"
#include "drawUtils.h"
#include "arnUtils/arnCircle.cpp"
#include "arnUtils/arnLinear.cpp"

float offScreen = 100.0;
PyObject* drawHomeCircle_drawArn(PyObject *self, PyObject *args);
PyObject* drawIconCircle_drawArn(PyObject *self, PyObject *args);
PyObject* drawHomeLinear_drawArn(PyObject *self, PyObject *args);
PyObject* drawIconLinear_drawArn(PyObject *self, PyObject *args);

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
