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

#include "hliGLutils/matrixUtils.c"      // Minimal Matrix math library for basic 2D graphics
#include "hliGLutils/Params.h"           // Caches Matrix Transformation Calculations
#include "hliGLutils/drawCallClass.h"    // Helper class that abstracts VBO, Matrix Ops, etc.
#include "hliGLutils/primUtils.h"        // Library of Geometric Primitives for drawing shapes
#include "hliGLutils/fontUtils.h"        // Helper utilities for loading fonts and drawing text

/*
 * Sets up shaders for OpenGL
 */
GLuint   whiteTex;
//GLubyte* blankTexture;
#include "hliGLutils/initShaders.cpp"    // Code that builds a shaderProgram (vert+frag) from source

/*
 * Dynamic iconography draw code for different arrangements
 */
float offScreen = 100.0;            // Used for moving geometry offscreen, will likely get depricated
#include "hliGLutils/arnUtils/arnCircle.cpp"   // Utilities for drawing icon+background for circular arrangements
#include "hliGLutils/arnUtils/arnLinear.cpp"   // Utilities for drawing icon+background for linear arrangements
#include "hliGLutils/arnUtils/drawIcon.cpp"    // Utility for drawing lamp icons

/*
 * Graphical buttons draw code
 */
#include "hliGLutils/buttonUtils/drawPrim.cpp"          // Draws a primitive
#include "hliGLutils/buttonUtils/drawClock.cpp"         // Draws the master switch (clock in center of display)
#include "hliGLutils/buttonUtils/drawArrow.cpp"         // Draws a generic arrow that can be oriented in different directions
#include "hliGLutils/buttonUtils/drawBulbButtons.cpp"   // Draws the Color-setting bottons that encircle/straddle the master switch
#include "hliGLutils/buttonUtils/drawGranChanger.cpp"   // Draws the Granularity Rocker on the color picker screen
#include "hliGLutils/buttonUtils/drawHueRing.cpp"       // Draws the ring of colored dots on the color picker
#include "hliGLutils/buttonUtils/drawColrTri.cpp"       // Draws the triangle of colored dots for the color picker
#include "hliGLutils/buttonUtils/drawConfirm.cpp"       // Draws a checkmark button
#include "hliGLutils/buttonUtils/primDrawTest.cpp"      // used for testing primitive draw code

/* END OF INCLUDES   */

/*
 * Python Function forward declarations
 */
PyObject* initShaders_hliGLutils    (PyObject* self, PyObject* args);

PyObject* drawText_hliGLutils       (PyObject* self, PyObject* args);
//PyObject* loadChar_hliGLutils        (PyObject* self, PyObject* args);
PyObject* buildAtlas_hliGLutils     (PyObject* self, PyObject* args);

PyObject* drawHomeCircle_hliGLutils (PyObject* self, PyObject* args);
PyObject* drawIconCircle_hliGLutils (PyObject* self, PyObject* args);
PyObject* drawHomeLinear_hliGLutils (PyObject* self, PyObject* args);
PyObject* drawIconLinear_hliGLutils (PyObject* self, PyObject* args);
PyObject* drawIcon_hliGLutils       (PyObject* self, PyObject* args);

PyObject* drawArrow_hliGLutils      (PyObject* self, PyObject* args);
PyObject* drawBulbButton_hliGLutils (PyObject* self, PyObject* args);
PyObject* drawClock_hliGLutils      (PyObject* self, PyObject* args);
PyObject* drawColrTri_hliGLutils    (PyObject* self, PyObject* args);
PyObject* drawConfirm_hliGLutils    (PyObject* self, PyObject* args);
PyObject* drawGranChanger_hliGLutils(PyObject* self, PyObject* args);
PyObject* drawHueRing_hliGLutils    (PyObject* self, PyObject* args);
PyObject* drawPrim_hliGLutils       (PyObject* self, PyObject* args);
PyObject* primTest_hliGLutils       (PyObject* self, PyObject* args);

/*
 * Python Method Definitions
 */

static PyMethodDef hliGLutils_methods[] = {
   { "initShaders", (PyCFunction)initShaders_hliGLutils, METH_NOARGS },

   { "drawText",        (PyCFunction)drawText_hliGLutils,         METH_VARARGS },
   //{ "loadChar",        (PyCFunction)loadChar_hliGLutils,       METH_VARARGS },
   { "buildAtlas",      (PyCFunction)buildAtlas_hliGLutils,       METH_VARARGS },

   { "drawHomeCircle",  (PyCFunction)drawHomeCircle_hliGLutils,   METH_VARARGS },
   { "drawIconCircle",  (PyCFunction)drawIconCircle_hliGLutils,   METH_VARARGS },
   { "drawHomeLinear",  (PyCFunction)drawHomeLinear_hliGLutils,   METH_VARARGS },
   { "drawIconLinear",  (PyCFunction)drawIconLinear_hliGLutils,   METH_VARARGS },
   { "drawIcon",        (PyCFunction)drawIcon_hliGLutils,         METH_VARARGS },

   { "drawArrow",       (PyCFunction)drawArrow_hliGLutils,        METH_VARARGS },
   { "drawBulbButton",  (PyCFunction)drawBulbButton_hliGLutils,   METH_VARARGS },
   { "drawClock",       (PyCFunction)drawClock_hliGLutils,        METH_VARARGS },
   { "drawColrTri",     (PyCFunction)drawColrTri_hliGLutils,      METH_VARARGS },
   { "drawConfirm",     (PyCFunction)drawConfirm_hliGLutils,      METH_VARARGS },
   { "drawGranChanger", (PyCFunction)drawGranChanger_hliGLutils,  METH_VARARGS },
   { "drawHueRing",     (PyCFunction)drawHueRing_hliGLutils,      METH_VARARGS },
   { "drawPrim",        (PyCFunction)drawPrim_hliGLutils,         METH_VARARGS },
   { "primTest",        (PyCFunction)primTest_hliGLutils,         METH_VARARGS },

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

