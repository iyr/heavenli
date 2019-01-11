#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>

using namespace std;

GLfloat  *colrTriVertexBuffer = NULL;
GLfloat  *colrTriColorBuffer  = NULL;
GLushort *colrTriIndices      = NULL;
GLuint    colrTriVerts;
GLint     prevColrTriNumLevels;

PyObject* drawColrTri_drawButtons(PyObject *self, PyObject *args) {
   float w2h, scale, currentHue;
   char circleSegments = 30;
   char numLevels = 6;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "flff",
            &currentHue,
            &numLevels,
            &w2h,
            &scale))
   {
      Py_RETURN_NONE;
   }

   if (numLevels < 5)
      numLevels = 5;
   if (numLevels > 7)
      numLevels = 7;

   if (  prevColrTriNumLevels != numLevels   ||
         colrTriVertexBuffer  == NULL        ||
         colrTriColorBuffer   == NULL        ||
         colrTriIndices       == NULL        ){

      printf("Initializing Geometry for Color Triangle\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float tmx, tmy, tmr;
      float colors[3] = {0.0, 0.0, 0.0};

      tmr = 0.05f;
      for (int i = 0; i < numLevels; i++) {
         for (int j = 0; j < numLevels-i; j++) {
            hsv2rgb(
                  float(currentHue), 
                  float((i+1.0f)/numLevels) - float(1.0f/6.0f), 
                  float(1.0 - j/float(numLevels - 1.0)), 
                  colors);
            //tmx = float(-0.23f + (i*0.130f));
            //tmy = float(+0.37f - (i*0.075f + j*0.15f));
            tmx = float(-0.0383*numLevels + (i*0.13f));
            tmy = float(+0.0616*numLevels - (i*0.075f + j*0.15f));
            drawEllipse(tmx, tmy, tmr, tmr, circleSegments, colors, verts, colrs);
         }
      }

      colrTriVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (colrTriVertexBuffer == NULL) {
         colrTriVertexBuffer = new GLfloat[colrTriVerts*2];
      } else {
         delete [] colrTriVertexBuffer;
         colrTriVertexBuffer = new GLfloat[colrTriVerts*2];
      }

      if (colrTriColorBuffer == NULL) {
         colrTriColorBuffer = new GLfloat[colrTriVerts*3];
      } else {
         delete [] colrTriColorBuffer;
         colrTriColorBuffer = new GLfloat[colrTriVerts*3];
      }

      if (colrTriIndices == NULL) {
         colrTriIndices = new GLushort[colrTriVerts];
      } else {
         delete [] colrTriIndices;
         colrTriIndices = new GLushort[colrTriVerts];
      }

      for (unsigned int i = 0; i < colrTriVerts; i++) {
         colrTriVertexBuffer[i*2]   = verts[i*2];
         colrTriVertexBuffer[i*2+1] = verts[i*2+1];
         colrTriIndices[i]          = i;
         colrTriColorBuffer[i*3+0]  = colrs[i*3+0];
         colrTriColorBuffer[i*3+1]  = colrs[i*3+1];
         colrTriColorBuffer[i*3+2]  = colrs[i*3+2];
      }
      prevColrTriNumLevels = numLevels;
   }

   glPushMatrix();
   if (w2h <= 1.0) {
         scale = scale*w2h;
   }
   glScalef(scale, scale, 1);
   glColorPointer(3, GL_FLOAT, 0, colrTriColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, colrTriVertexBuffer);
   glDrawElements( GL_TRIANGLES, colrTriVerts, GL_UNSIGNED_SHORT, colrTriIndices);
   glPopMatrix();

   Py_RETURN_NONE;

}
