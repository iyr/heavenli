#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>

using namespace std;

GLfloat  *hueRingVertexBuffer = NULL;
GLfloat  *hueRingColorBuffer  = NULL;
GLushort *hueRingIndices      = NULL;

PyObject* drawHueRing_drawButtons(PyObject *self, PyObject *args) {
   float w2h, scale, R, G, B;
   char circleSegments = 45;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "ff",
            &w2h,
            &scale))
   {
      Py_RETURN_NONE;
   }

   // Allocate and Define Geometry/Color buffers
   if (  hueRingVertexBuffer == NULL   ||
         hueRingColorBuffer  == NULL   ||
         hueRingIndices      == NULL   ){

   }
         
   // Geometry up to date, check if colors need to be updated
   for (int i = 0; i < 3; i++) {
      if (faceColor[i] != hueRingColorBuffer[i]) {
         for (int k = 0; k < circleSegments*3; k++) {
            hueRingColorBuffer[i + k*3] = faceColor[i];
         }
      }
   }
   
   glPushMatrix();
   if (w2h <= 1.0) {
         scale = scale*w2h;
   }
   glScalef(scale, scale, 1);
   glColorPointer(3, GL_FLOAT, 0, hueRingColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, hueRingVertexBuffer);
   glDrawElements( GL_TRIANGLES, hueRingVerts, GL_UNSIGNED_SHORT, hueRingIndices);
   glPopMatrix();

   Py_RETURN_NONE;
}
