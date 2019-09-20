#include <Python.h>
#include "matrixUtils.c"
#include "Params.h"
#include "primUtils.h"
#include "fontUtils/characterClass.h"
#include <map>

using namespace std;
//std::map<uint8_t, Character> Characters;
Character* Characters = new Character[128];
#include "fontUtils/loadChar.cpp"
#include "fontUtils/drawText.cpp"

PyObject* drawText_fontUtils  (PyObject *self, PyObject *args);
PyObject* loadChar_fontUtils  (PyObject *self, PyObject *args);

static PyMethodDef fontUtils_methods[] = {
   { "drawText",  (PyCFunction)drawText_fontUtils, METH_VARARGS },
   { "loadChar",  (PyCFunction)loadChar_fontUtils, METH_VARARGS },
   { NULL, NULL, 0, NULL}
};

static PyModuleDef fontUtils_module = {
   PyModuleDef_HEAD_INIT,
   "fontUtils",
   "HeavenLi Character/Text Library",
   0,
   fontUtils_methods
};

PyMODINIT_FUNC PyInit_fontUtils() {
   PyObject* m = PyModule_Create(&fontUtils_module);
   if (m == NULL) {
      return NULL;
   }
   return m;
}
