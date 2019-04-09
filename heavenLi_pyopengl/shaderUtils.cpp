#include <Python.h>

#include "initShaders.cpp"

PyObject* initShaders_shaderUtils(PyObject *self, PyObject *args);

static PyMethodDef shaderUtils_methods[] = {
   { "initShaders", (PyCFunction)initShaders_shaderUtils, METH_NOARGS },
   { NULL, NULL, 0, NULL}
};

static PyModuleDef shaderUtils_module = {
   PyModuleDef_HEAD_INIT,
   "shaderUtils",
   "HeavenLi Shader Utilities",
   0,
   shaderUtils_methods 
};

PyMODINIT_FUNC PyInit_shaderUtils() {
   PyObject* m = PyModule_Create(&shaderUtils_module);
   if (m == NULL) {
      return NULL;
   }
   return m;
}
