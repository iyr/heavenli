#include <Python.h>
#include "drawUtils.h"
#include "buttonUtils/drawClock.cpp"
#include "buttonUtils/drawBulbButtons.cpp"
#include "buttonUtils/drawHueRing.cpp"

PyObject* drawClock_drawButtons     (PyObject *self, PyObject *args);
PyObject* drawBulbButton_drawButtons(PyObject *self, PyObject *args);
PyObject* drawHueRing_drawButtons   (PyObject *self, PyObject *args);

static PyMethodDef drawButtons_methods[] = {
   { "drawClock",       (PyCFunction)drawClock_drawButtons,       METH_VARARGS },
   { "drawBulbButton",  (PyCFunction)drawBulbButton_drawButtons,  METH_VARARGS },
   { "drawHueRing",     (PyCFunction)drawHueRing_drawButtons,     METH_VARARGS },
   { NULL, NULL, 0, NULL}
};

static PyModuleDef drawButtons_module = {
   PyModuleDef_HEAD_INIT,
   "drawButtons",
   "HeavenLi Button Library",
   0,
   drawButtons_methods
};

PyMODINIT_FUNC PyInit_drawButtons() {
   //return PyModule_Create(&drawButtons_module);
   PyObject* m = PyModule_Create(&drawButtons_module);
   if (m == NULL) {
      return NULL;
   }
   return m;
}
