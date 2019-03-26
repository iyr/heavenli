#include <Python.h>
#include "drawUtils.h"

#include "buttonUtils/drawClock.cpp"
#include "buttonUtils/drawArrow.cpp"
#include "buttonUtils/drawBulbButtons.cpp"
#include "buttonUtils/drawGranRocker.cpp"
#include "buttonUtils/drawHueRing.cpp"
#include "buttonUtils/drawColrTri.cpp"
#include "buttonUtils/drawConfirm.cpp"
#include "infoUtils/printText.cpp"

PyObject* drawClock_drawButtons     (PyObject *self, PyObject *args);
PyObject* drawArrow_drawButtons     (PyObject *self, PyObject *args);
PyObject* drawBulbButton_drawButtons(PyObject *self, PyObject *args);
PyObject* drawGranRocker_drawButtons(PyObject *self, PyObject *args);
PyObject* drawHueRing_drawButtons   (PyObject *self, PyObject *args);
PyObject* drawColrTri_drawButtons   (PyObject *self, PyObject *args);
PyObject* drawConfirm_drawButtons   (PyObject *self, PyObject *args);
PyObject* printText_drawButtons     (PyObject *self, PyObject *args);

static PyMethodDef drawButtons_methods[] = {
   { "drawClock",       (PyCFunction)drawClock_drawButtons,       METH_VARARGS },
   { "drawArrow",       (PyCFunction)drawArrow_drawButtons,       METH_VARARGS },
   { "drawBulbButton",  (PyCFunction)drawBulbButton_drawButtons,  METH_VARARGS },
   { "drawGranRocker",  (PyCFunction)drawGranRocker_drawButtons,  METH_VARARGS },
   { "drawHueRing",     (PyCFunction)drawHueRing_drawButtons,     METH_VARARGS },
   { "drawColrTri",     (PyCFunction)drawColrTri_drawButtons,     METH_VARARGS },
   { "drawConfirm",     (PyCFunction)drawConfirm_drawButtons,     METH_VARARGS },
   { "printText",       (PyCFunction)printText_drawButtons,       METH_VARARGS },
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
   PyObject* m = PyModule_Create(&drawButtons_module);
   if (m == NULL) {
      return NULL;
   }
   return m;
}
