#include <Python.h>
#include "matrixUtils.c"
#include "Params.h"
#include "primUtils.h"
#include "textUtils/characterClass.h"
#include <map>

using namespace std;
std::map<GLchar, Character> Characters;
#include "textUtils/loadChar.cpp"
#include "textUtils/drawText.cpp"

PyObject* drawText_textUtils  (PyObject *self, PyObject *args);
PyObject* loadChar_textUtils  (PyObject *self, PyObject *args);

static PyMethodDef textUtils_methods[] = {
   { "drawText",  (PyCFunction)drawText_textUtils, METH_VARARGS },
   { "loadChar",  (PyCFunction)loadChar_textUtils, METH_VARARGS },
   { NULL, NULL, 0, NULL}
};

static PyModuleDef textUtils_module = {
   PyModuleDef_HEAD_INIT,
   "textUtils",
   "HeavenLi Character/Text Library",
   0,
   textUtils_methods
};

PyMODINIT_FUNC PyInit_textUtils() {
   PyObject* m = PyModule_Create(&textUtils_module);
   if (m == NULL) {
      return NULL;
   }
   return m;
}
