#include <Python.h>
#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
   // These undefs necessary because microsoft
   #undef near
   #undef far
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#include "matrixUtils.c"
#include "Params.h"
#include "primUtils.h"
#include "fontUtils/characterStruct.h"
#include <map>

using namespace std;

#include "fontUtils/loadChar.cpp"
#include "fontUtils/drawText.cpp"
#include "fontUtils/buildAtlas.cpp"

PyObject* drawText_fontUtils  (PyObject *self, PyObject *args);
PyObject* loadChar_fontUtils  (PyObject *self, PyObject *args);
PyObject* buildAtlas_fontUtils  (PyObject *self, PyObject *args);

static PyMethodDef fontUtils_methods[] = {
   { "drawText",  (PyCFunction)drawText_fontUtils, METH_VARARGS },
   { "loadChar",  (PyCFunction)loadChar_fontUtils, METH_VARARGS },
   { "buildAtlas",  (PyCFunction)buildAtlas_fontUtils, METH_VARARGS },
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
