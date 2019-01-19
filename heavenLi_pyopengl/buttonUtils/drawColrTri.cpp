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
GLuint   colrTriVerts;
GLint    prevColrTriNumLevels;
GLfloat prevHue;
float    *triButtonData       = NULL;

PyObject* drawColrTri_drawButtons(PyObject *self, PyObject *args) {
   //PyObject *py_list;
   //PyObject *py_tuple;
   float w2h, scale, currentHue;
   char circleSegments = 18;
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

   /*
    * Granularity Levels:
    * 5: Low
    * 6: Medium
    * 7: High
    */

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
      float tmx, tmy, tmr, saturation, value;
      float colors[3] = {0.0, 0.0, 0.0};

      tmr = 0.05f;
      for (int i = 0; i < numLevels; i++) {        /* Columns */
         for (int j = 0; j < numLevels-i; j++) {   /* Rows */

            // Calculate Discrete Saturation and Value
            value = 1.0f - float(j) / float(numLevels - 1);
            saturation  =  float(i) / float(numLevels - 1 - j);

            // Convert HSV to RGB
            hsv2rgb(
                  currentHue,
                  saturation,
                  value, 
                  colors);

            // Define relative positions of hue buttons
            tmx = float(-0.0383*numLevels + (i*0.13f));
            tmy = float(+0.0616*numLevels - (i*0.075f + j*0.15f));
            drawEllipse(tmx, tmy, tmr, circleSegments, colors, verts, colrs);
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
      prevHue = currentHue;
   }

   if ( prevHue != currentHue ) {
      float rgb[3];
      float hsv[3];
      for (unsigned int i = 0; i < colrTriVerts; i++) {
         // Get current RGB values
         rgb[0] = colrTriColorBuffer[i*3+0];
         rgb[1] = colrTriColorBuffer[i*3+1];
         rgb[2] = colrTriColorBuffer[i*3+2];

         // Convert to Hue/Sat/Val
         rgb2hsv(rgb[0], rgb[1], rgb[2], hsv);

         // Convert to RGB with current Hue
         hsv2rgb(currentHue, hsv[1], hsv[2], rgb);

         // Update Color buffer
         colrTriColorBuffer[i*3+0] = rgb[0];
         colrTriColorBuffer[i*3+1] = rgb[1];
         colrTriColorBuffer[i*3+2] = rgb[2];
      }

      prevHue = currentHue;
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
