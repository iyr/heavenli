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

PyObject* drawHueRing_drawButtons(PyObject *self, PyObject *args) {
   float w2h, scale;
   char circleSegments = 45;
   char numHues = 22;

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

      printf("Initializing Geometry for Hue Ring\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float tmo, ang, tmx, tmy, tmf;
      tmf = float(1.0/12.0);
      float colors[3] = {0.0, 0.0, 0.0};
      for (int i = 0; i < 12; i++) {
         tmo = float(float(i)/12.0);
         hsv2rgb(tmo, 1.0, 1.0, colors);
         ang = 360*tmo + 90;
         tmx = float(cos(degToRad(ang))*0.67);
         tmy = float(sin(degToRad(ang))*0.67);
         drawEllipse(tmx, tmy, 0.15, 0.15, circleSegments, colors, verts, colrs);
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
