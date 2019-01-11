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
GLuint    hueRingVerts;
GLuint    prevHueRingNumHues;

PyObject* drawHueRing_drawButtons(PyObject *self, PyObject *args) {
   float w2h, scale;
   char circleSegments = 45;
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
      float ang, tmo, tmx, tmy, tmf, tmr;
      tmf = float(1.0f / numHues);
      tmr = float(0.15f);
      float colors[3] = {0.0, 0.0, 0.0};
      for (int i = 0; i < numHues; i++) {
         tmo = float(i) / float(numHues);
         hsv2rgb(tmo, 1.0f, 1.0f, colors);
         ang = 360*tmo + 90;
         tmx = float(cos(degToRad(ang))*0.67*pow(numHues/12.0f, 1.0f/4.0f));
         tmy = float(sin(degToRad(ang))*0.67*pow(numHues/12.0f, 1.0f/4.0f));
         drawEllipse(tmx, tmy, tmr*(12.0/numHues), tmr*(12.0/numHues), circleSegments, colors, verts, colrs);
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
         
   // Geometry up to date, check if colors need to be updated
   /*
   for (int i = 0; i < 3; i++) {
      if (faceColor[i] != hueRingColorBuffer[i]) {
         for (int k = 0; k < circleSegments*3; k++) {
            hueRingColorBuffer[i + k*3] = faceColor[i];
         }
      }
   }
   */
   
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
