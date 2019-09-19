#include <Python.h>
#include "matrixUtils.c"
#include "Params.h"
#include "primUtils.h"
#include <ft2build.h>
#include FT_FREETYPE_H

#include "textUtils/printText.cpp"
PyObject* printText_drawText     (PyObject *self, PyObject *args);

static PyMethodDef drawText_methods[] = {
   { "printText",       (PyCFunction)printText_drawText,       METH_VARARGS },
   { NULL, NULL, 0, NULL}
};

static PyModuleDef drawText_module = {
   PyModuleDef_HEAD_INIT,
   "drawText",
   "HeavenLi Character/Text Library",
   0,
   drawText_methods
};

PyMODINIT_FUNC PyInit_drawText() {
   PyObject* m = PyModule_Create(&drawText_module);
   if (m == NULL) {
      return NULL;
   }
   return m;
}
