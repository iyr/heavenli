#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>

using namespace std;

GLfloat*    hueRingVertexBuffer = NULL;
GLfloat*   hueRingColorBuffer  = NULL;
GLushort*   hueRingIndices      = NULL;
GLuint      hueRingVerts;
GLuint      prevHueRingNumHues;
float*      hueButtonData       = NULL;  /* X, Y, hue per button */

PyObject* drawHueRing_drawButtons(PyObject *self, PyObject *args) {
   PyObject *py_list;
   PyObject *py_tuple;
   float w2h, scale;
   char circleSegments = 24;
   char numHues = 12;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "lff",
            &numHues,
            &w2h,
            &scale))
   {
      Py_RETURN_NONE;
   }

   // Allocate and Define Geometry/Color buffers
   if (  prevHueRingNumHues  != numHues ||
         hueRingVertexBuffer == NULL    ||
         hueRingColorBuffer  == NULL    ||
         hueRingIndices      == NULL    ){

      printf("Initializing Geometry for Hue Ring\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float ang, tmx, tmy, tmf, tmr;
      float tmo;
      tmf = float(1.0f / numHues);
      tmr = float(0.15f);
      float colors[3] = {0.0, 0.0, 0.0};

      if (hueButtonData == NULL) {
         hueButtonData = new float[numHues*2];
      } else {
         delete [] hueButtonData;
         hueButtonData = new float[numHues*2];
      }

      for (int i = 0; i < numHues; i++) {
         tmo = float(i) / float(numHues);
         hsv2rgb(tmo, 1.0, 1.0, colors);
         ang = float(360.0*tmo + 90.0);
         tmx = float(cos(degToRad(ang))*0.67*pow(numHues/12.0f, 1.0f/4.0f));
         tmy = float(sin(degToRad(ang))*0.67*pow(numHues/12.0f, 1.0f/4.0f));
         hueButtonData[i*2+0] = tmx;
         hueButtonData[i*2+1] = tmy;
         drawEllipse(tmx, tmy, float(tmr*(12.0/numHues)), circleSegments, colors, verts, colrs);
      }

      hueRingVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (hueRingVertexBuffer == NULL) {
         hueRingVertexBuffer = new GLfloat[hueRingVerts*2];
      } else {
         delete [] hueRingVertexBuffer;
         hueRingVertexBuffer = new GLfloat[hueRingVerts*2];
      }

      if (hueRingColorBuffer == NULL) {
         hueRingColorBuffer = new GLfloat[hueRingVerts*3];
      } else {
         delete [] hueRingColorBuffer;
         hueRingColorBuffer = new GLfloat[hueRingVerts*3];
      }

      if (hueRingIndices == NULL) {
         hueRingIndices = new GLushort[hueRingVerts];
      } else {
         delete [] hueRingIndices;
         hueRingIndices = new GLushort[hueRingVerts];
      }

      for (unsigned int i = 0; i < hueRingVerts; i++) {
         hueRingVertexBuffer[i*2]   = verts[i*2];
         hueRingVertexBuffer[i*2+1] = verts[i*2+1];
         hueRingIndices[i]          = i;
         hueRingColorBuffer[i*3+0]  = colrs[i*3+0];
         hueRingColorBuffer[i*3+1]  = colrs[i*3+1];
         hueRingColorBuffer[i*3+2]  = colrs[i*3+2];
      }
      prevHueRingNumHues = numHues;
   }
         
   glPushMatrix();
   if (w2h <= 1.0) {
         scale = scale*w2h;
   }

   float tmo;
   py_list = PyList_New(numHues);
   for (int i = 0; i < numHues; i++) {
      py_tuple = PyTuple_New(3);
      tmo = float(i) / float(numHues);
      PyTuple_SetItem(py_tuple, 0, PyFloat_FromDouble(hueButtonData[i*2+0]));
      PyTuple_SetItem(py_tuple, 1, PyFloat_FromDouble(hueButtonData[i*2+1]));
      PyTuple_SetItem(py_tuple, 2, PyFloat_FromDouble(tmo));
      PyList_SetItem(py_list, i, py_tuple);
   }

   glScalef(scale, scale, 1);
   glColorPointer(3, GL_FLOAT, 0, hueRingColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, hueRingVertexBuffer);
   glDrawElements( GL_TRIANGLES, hueRingVerts, GL_UNSIGNED_SHORT, hueRingIndices);
   glPopMatrix();

   return py_list;
}
