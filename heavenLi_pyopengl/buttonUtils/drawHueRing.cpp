#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>

using namespace std;

GLfloat*    hueRingVertexBuffer  = NULL;
GLfloat*    hueRingColorBuffer   = NULL;
GLushort*   hueRingIndices       = NULL;
GLuint      hueRingVerts;
GLubyte     prevRingHueRingNumHues;
GLfloat     prevRingHue          = 0.0;
float*      hueButtonData        = NULL;  /* X, Y, hue per button */

PyObject* drawHueRing_drawButtons(PyObject *self, PyObject *args) {
   PyObject *py_list;
   PyObject *py_tuple;
   float w2h, scale, tmo, currentRingHue;
   float ringColor[3];
   char circleSegments = 45;
   unsigned char numHues = 12;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "flOff",
            &currentRingHue,
            &numHues,
            &py_tuple,
            &w2h,
            &scale))
   {
      Py_RETURN_NONE;
   }

   ringColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 0)));
   ringColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 1)));
   ringColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 2)));

   // Allocate and Define Geometry/Color buffers
   if (  prevRingHueRingNumHues   != numHues  ||
         hueRingVertexBuffer  == NULL     ||
         hueRingColorBuffer   == NULL     ||
         hueRingIndices       == NULL     ){

      printf("Initializing Geometry for Hue Ring\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float ang, tmx, tmy;
      float colors[3] = {0.0, 0.0, 0.0};
      float tmr = float(0.15f);

      if (hueButtonData == NULL) {
         hueButtonData = new float[numHues*2];
      } else {
         delete [] hueButtonData;
         hueButtonData = new float[numHues*2];
      }

      float ringX, ringY;
      for (int i = 0; i < numHues; i++) {
         tmo = float(i) / float(numHues);
         hsv2rgb(tmo, 1.0, 1.0, colors);
         ang = float(360.0*tmo + 90.0);
         tmx = float(cos(degToRad(ang))*0.67*pow(numHues/12.0f, 1.0f/4.0f));
         tmy = float(sin(degToRad(ang))*0.67*pow(numHues/12.0f, 1.0f/4.0f));
         hueButtonData[i*2+0] = tmx;
         hueButtonData[i*2+1] = tmy;
         drawEllipse(tmx, tmy, float(tmr*(12.0/numHues)), circleSegments, colors, verts, colrs);
         if (abs(currentRingHue - tmo) <= 1.0f / float(numHues*2)) {
            ringX = tmx;
            ringY = tmy;
         }
      }

      drawHalo(
            ringX, ringY,
            float(1.06*tmr*(12.0/numHues)), float(1.06*tmr*(12.0/numHues)),
            0.02f,
            circleSegments,
            ringColor,
            verts,
            colrs);

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
      prevRingHue = currentRingHue;
      prevRingHueRingNumHues = numHues;
   }

   // Update Ring if hue selection has changed
   if (  prevRingHue != currentRingHue ) {
      float ang, ringX, ringY;
      float tmr = float(0.15f);
      for (int i = 0; i < numHues; i++) {
         tmo = float(i) / float(numHues);
         if (abs(currentRingHue - tmo) <= 1.0f / float(numHues*2)) {
            ang = float(360.0*tmo + 90.0);
            ringX = float(cos(degToRad(ang))*0.67*pow(numHues/12.0f, 1.0f/4.0f));
            ringY = float(sin(degToRad(ang))*0.67*pow(numHues/12.0f, 1.0f/4.0f));
         }
      }

      drawHalo(
            ringX, ringY,
            float(1.06*tmr*(12.0/numHues)), float(1.06*tmr*(12.0/numHues)),
            0.02f,
            circleSegments,
            3*numHues*circleSegments,
            ringColor,
            hueRingVertexBuffer,
            hueRingColorBuffer);
      prevRingHue = currentRingHue;
   }

   // Check if Selection Ring Color needs to be updated
   for (int i = 0; i < 3; i++) {
      if (hueRingColorBuffer[numHues*circleSegments*9+i] != ringColor[i]) {
         printf("quack\n");
         for (int k = numHues*circleSegments*3; k < hueRingVerts; k++) {
            hueRingColorBuffer[k*3+i] = ringColor[i];
         }
      }
   }
         
   glPushMatrix();
   if (w2h <= 1.0) {
         scale = scale*w2h;
   }

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
