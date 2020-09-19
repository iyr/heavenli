/*
 * HeavenLi OpenGL Utilities, draw code, and helper functions */

#include <Python.h>           // Main Python C/C++ API Library

#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
   // These undefs necessary because microsoft
   #undef near
   #undef far
#endif
#include <GL/gl.h>            // OpenGL
#include <GL/glext.h>         // OpenGL extensions because microsoft

#include <vector>             // Used to dynamically build vertex data arrays
#include <string>             // "quack"
#include <math.h>             // Always useful
#include <map>
#include <iterator>

#define GET_VARIABLE_NAME(Variable) (#Variable)
#define GET_CLASS_NAME(ClassName) (#ClassName)

using namespace std;

// Helper function for allocating memory based on datatype as argument
GLuint GLsizeof(GLenum type) {
   switch(type) {
      case GL_TRUE:
      case GL_FALSE:
         return sizeof(GLboolean); // Usually 1 byte, usually...

      case GL_BYTE:
      case GL_UNSIGNED_BYTE:
         return 1;

      case GL_SHORT:
      case GL_UNSIGNED_SHORT:
      case GL_HALF_FLOAT:
         return 2;

      case GL_INT:
      case GL_UNSIGNED_INT:
      case GL_FIXED:
      case GL_FLOAT:
         return 4;

      case GL_DOUBLE:
         return 8;
   }

   return 0;
}

#include "hliGLutils/vertexAttribStrings.h"  // Struct of strings to map data buffers to vertex shader attribs
#include "hliGLutils/vertAttribStruct.h"     // Helper struct for managing shader data
#include "hliGLutils/attribCacheClass.h"     // Derived class for caching vert data per attribute
#include "hliGLutils/shaderProgramClass.h"   // Helper class for building/managing shaders
#include "hliGLutils/matrixUtils.c"    // Minimal Matrix math library for basic 2D graphics
#include "hliGLutils/Params.h"         // Caches Matrix Transformation Calculations
#include "hliGLutils/drawCallClass.h"  // Helper class that abstracts VBO, Matrix Ops, etc.
#include "hliGLutils/primUtils.h"      // Library of Geometric Primitives for drawing shapes
#include "hliGLutils/fontUtils.h"      // Helper utilities for loading fonts and drawing text
#include "hliGLutils/initUtils.h"      // Utilities usually run once that stage/setup/initialize objects
#include "hliGLutils/metaUtils.h"

#include "hliGLutils/buttonUtils.h"    // Utilities for drawing UI elements
#include "hliGLutils/arnUtils.h"       // Dynamic iconography draw code

/* END OF INCLUDES   */

/*
 * Python Function forward declarations
 */
PyObject* initShaders_hliGLutils    (PyObject* self, PyObject* args);

PyObject* drawText_hliGLutils       (PyObject* self, PyObject* args);
PyObject* buildAtlas_hliGLutils     (PyObject* self, PyObject* args);

PyObject* drawArch_hliGLutils       (PyObject* self, PyObject* args);
PyObject* drawEllipse_hliGLutils    (PyObject* self, PyObject* args);
PyObject* drawPill_hliGLutils       (PyObject* self, PyObject* args);

PyObject* drawHomeCircle_hliGLutils (PyObject* self, PyObject* args);
PyObject* drawIconCircle_hliGLutils (PyObject* self, PyObject* args);
PyObject* drawHomeLinear_hliGLutils (PyObject* self, PyObject* args);
PyObject* drawIconLinear_hliGLutils (PyObject* self, PyObject* args);
PyObject* drawIcon_hliGLutils       (PyObject* self, PyObject* args);

PyObject* drawMenu_hliGLutils       (PyObject* self, PyObject* args);

PyObject* drawImageSquare_hliGLutils(PyObject* self, PyObject* args);
PyObject* drawImageCircle_hliGLutils(PyObject* self, PyObject* args);

PyObject* drawArrow_hliGLutils      (PyObject* self, PyObject* args);
PyObject* drawBulbButton_hliGLutils (PyObject* self, PyObject* args);
PyObject* drawClock_hliGLutils      (PyObject* self, PyObject* args);
PyObject* drawColrTri_hliGLutils    (PyObject* self, PyObject* args);
PyObject* drawConfirm_hliGLutils    (PyObject* self, PyObject* args);
PyObject* drawGranChanger_hliGLutils(PyObject* self, PyObject* args);
PyObject* drawHueRing_hliGLutils    (PyObject* self, PyObject* args);
PyObject* drawPrim_hliGLutils       (PyObject* self, PyObject* args);

PyObject* doesDrawCallExist_hliGLutils(PyObject* self, PyObject* args);
PyObject* printDrawCalls_hliGLutils  (PyObject* self, PyObject* args);

/*
 * Python Method Definitions
 */
static PyMethodDef hliGLutils_methods[] = {
   { "initShaders",     (PyCFunction)initShaders_hliGLutils,      METH_NOARGS },

   { "drawText",        (PyCFunction)drawText_hliGLutils,         METH_VARARGS },
   { "buildAtlas",      (PyCFunction)buildAtlas_hliGLutils,       METH_VARARGS },

   { "drawArch",        (PyCFunction)drawArch_hliGLutils,         METH_VARARGS },
   { "drawEllipse",     (PyCFunction)drawEllipse_hliGLutils,      METH_VARARGS },
   { "drawPill",        (PyCFunction)drawPill_hliGLutils,         METH_VARARGS },

   { "drawHomeCircle",  (PyCFunction)drawHomeCircle_hliGLutils,   METH_VARARGS },
   { "drawIconCircle",  (PyCFunction)drawIconCircle_hliGLutils,   METH_VARARGS },
   { "drawHomeLinear",  (PyCFunction)drawHomeLinear_hliGLutils,   METH_VARARGS },
   { "drawIconLinear",  (PyCFunction)drawIconLinear_hliGLutils,   METH_VARARGS },
   { "drawIcon",        (PyCFunction)drawIcon_hliGLutils,         METH_VARARGS },

   { "drawMenu",        (PyCFunction)drawMenu_hliGLutils,         METH_VARARGS },

   { "drawImageSquare", (PyCFunction)drawImageSquare_hliGLutils,  METH_VARARGS },
   { "drawImageCircle", (PyCFunction)drawImageCircle_hliGLutils,  METH_VARARGS },

   { "drawArrow",       (PyCFunction)drawArrow_hliGLutils,        METH_VARARGS },
   { "drawBulbButton",  (PyCFunction)drawBulbButton_hliGLutils,   METH_VARARGS },
   { "drawClock",       (PyCFunction)drawClock_hliGLutils,        METH_VARARGS },
   { "drawColrTri",     (PyCFunction)drawColrTri_hliGLutils,      METH_VARARGS },
   { "drawConfirm",     (PyCFunction)drawConfirm_hliGLutils,      METH_VARARGS },
   { "drawGranChanger", (PyCFunction)drawGranChanger_hliGLutils,  METH_VARARGS },
   { "drawHueRing",     (PyCFunction)drawHueRing_hliGLutils,      METH_VARARGS },
   { "drawPrim",        (PyCFunction)drawPrim_hliGLutils,         METH_VARARGS },

   { "doesDrawCallExist",(PyCFunction)doesDrawCallExist_hliGLutils, METH_VARARGS},
   { "printDrawCalls",  (PyCFunction)printDrawCalls_hliGLutils,   METH_VARARGS },

   { NULL, NULL, 0, NULL}
};

/*
 * Python Module Definition
 */
static PyModuleDef hliGLutils_module = {
   PyModuleDef_HEAD_INIT,
   "hliGLutils",
   "HeavenLi OpenGL and drawCode utility set",
   0,
   hliGLutils_methods
};

PyMODINIT_FUNC PyInit_hliGLutils() {
   PyObject* m = PyModule_Create(&hliGLutils_module);
   if (m == NULL) {
      return NULL;
   }
   return m;
}

