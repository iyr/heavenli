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
float    *triButtonData       = NULL;
float    prevTriX             = 0.0;
float    prevTriY             = 0.0;
GLfloat  prevTriSatSel        = 0.0;
GLfloat  prevTriValSel        = 0.0;
GLfloat  prevTriHue;
GLfloat  prevTriSat;
GLfloat  prevTriVal;

PyObject* drawColrTri_drawButtons(PyObject *self, PyObject *args) {
   PyObject *py_list;
   PyObject *py_tuple;
   float w2h, scale, currentTriHue, currentTriSat, currentTriVal, tDiff;
   float ringColor[3];
   char circleSegments = 24;
   long numLevels = 6;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "ffflOfff",
            &currentTriHue,
            &currentTriSat,
            &currentTriVal,
            &numLevels,
            &py_tuple,
            &w2h,
            &scale,
            &tDiff))
   {
      Py_RETURN_NONE;
   }

   ringColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 0)));
   ringColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 1)));
   ringColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 2)));

   long numButtons = (numLevels * (numLevels + 1)) / 2;
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
      if (triButtonData == NULL) {
         triButtonData = new float[4*numButtons];
      } else {
         delete [] triButtonData;
         triButtonData = new float[4*numButtons];
      }

      int index = 0;
      float tmx, tmy, tmr, saturation, value, ringX = 100.0f, ringY = 100.0f;
      float colors[3] = {0.0, 0.0, 0.0};

      tmr = 0.05f;
      for (int i = 0; i < numLevels; i++) {        /* Columns */
         for (int j = 0; j < numLevels-i; j++) {   /* Rows */

            // Calculate Discrete Saturation and Value
            value = 1.0f - float(j) / float(numLevels - 1);
            saturation  =  float(i) / float(numLevels - 1 - j);

            if (saturation != saturation || saturation <= 0.0)
               saturation = 0.000001f;
            if (value != value || value <= 0.0)
               value = 0.000001f;

            // Convert HSV to RGB
            hsv2rgb(
                  currentTriHue,
                  saturation,
                  value, 
                  colors);

            // Define relative positions of sat/val buttons
            tmx = float(-0.0383*numLevels + (i*0.13f));
            tmy = float(+0.0616*numLevels - (i*0.075f + j*0.145f));
            drawEllipse(tmx, tmy, tmr, circleSegments, colors, verts, colrs);
            triButtonData[index*4 + 0] = tmx;
            triButtonData[index*4 + 1] = tmy;
            triButtonData[index*4 + 2] = saturation;
            triButtonData[index*4 + 3] = value;
            index++;
            if (  abs(currentTriSat - saturation) <= 1.0f / float(numLevels*2) &&
                  abs(currentTriVal - value     ) <= 1.0f / float(numLevels*2) ){
               ringX = tmx;
               ringY = tmy;
            }
         }
      }

      drawHalo(
            ringX, ringY,
            float(1.06*tmr), float(1.06*tmr),
            0.03f,
            circleSegments,
            ringColor,
            verts,
            colrs);

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
      prevTriHue = currentTriHue;
      prevTriSat = currentTriSat;
      prevTriVal = currentTriVal;
      prevTriSatSel = currentTriSat;
      prevTriValSel = currentTriVal;
   }

   if (  prevTriSatSel  != currentTriSat  ){
      prevTriSat = prevTriSatSel;
   }
   if (  prevTriValSel  != currentTriVal  ){
      prevTriVal = prevTriValSel;
   }

   float tmr, saturation, value, ringX = 100.0f, ringY = 100.0f;
   float deltaX, deltaY;

   tmr = 0.05f;
   for (int i = 0; i < numLevels; i++) {        // Columns
      for (int j = 0; j < numLevels-i; j++) {   // Rows

         // Calculate Discrete Saturation and Value
         value = 1.0f - float(j) / float(numLevels - 1);
         saturation  =  float(i) / float(numLevels - 1 - j);

         if (saturation != saturation || saturation <= 0.0)
            saturation = 0.000001f;
         if (value != value || value <= 0.0)
            value = 0.000001f;

         // Define relative positions of sat/val buttons
         if (  abs(currentTriSat - saturation) <= 1.0f / float(numLevels*2) &&
               abs(currentTriVal - value     ) <= 1.0f / float(numLevels*2) ){
            ringX = float(-0.0383*numLevels + (i*0.13f));
            ringY = float(+0.0616*numLevels - (i*0.075f + j*0.145f));
         }
      }
   }

   deltaX = ringX-prevTriX;
   deltaY = ringY-prevTriY;
   if (abs(deltaX) > tDiff*0.01) {
      if (deltaX < -0.0) {
         prevTriX -= float(3.0*tDiff*abs(deltaX));
      }
      if (deltaX > -0.0) {
         prevTriX += float(3.0*tDiff*abs(deltaX));
      }
   } else {
      prevTriX = ringX;
   }

   if (abs(deltaY) > tDiff*0.01) {
      if (deltaY < -0.0) {
         prevTriY -= float(3.0*tDiff*abs(deltaY));
      }
      if (deltaY > -0.0) {
         prevTriY += float(3.0*tDiff*abs(deltaY));
      }
   } else {
      prevTriY = ringY;
   }

   if (  prevTriSat  != currentTriSat  ||
         prevTriVal  != currentTriVal  ){
      drawHalo(
            prevTriX, prevTriY,
            float(1.06*tmr), float(1.06*tmr),
            0.03f,
            circleSegments,
            3*numButtons*circleSegments,
            ringColor,
            colrTriVertexBuffer,
            colrTriColorBuffer);
   }

   if (  abs(deltaX) <= tDiff*0.01 &&
         abs(deltaY) <= tDiff*0.01 ){
      prevTriSat = currentTriSat;
      prevTriVal = currentTriVal;
   }

   prevTriSatSel = currentTriSat;
   prevTriValSel = currentTriVal;
   /*
   if (  prevTriSat  != currentTriSat  ||
         prevTriVal  != currentTriVal  ){

      float tmr, saturation, value, ringX = 100.0f, ringY = 100.0f;

      tmr = 0.05f;
      for (int i = 0; i < numLevels; i++) {        // Columns
         for (int j = 0; j < numLevels-i; j++) {   // Rows

            // Calculate Discrete Saturation and Value
            value = 1.0f - float(j) / float(numLevels - 1);
            saturation  =  float(i) / float(numLevels - 1 - j);

            if (saturation != saturation || saturation <= 0.0)
               saturation = 0.000001f;
            if (value != value || value <= 0.0)
               value = 0.000001f;

            // Define relative positions of sat/val buttons
            if (  abs(currentTriSat - saturation) <= 1.0f / float(numLevels*2) &&
                  abs(currentTriVal - value     ) <= 1.0f / float(numLevels*2) ){
               ringX = float(-0.0383*numLevels + (i*0.13f));
               ringY = float(+0.0616*numLevels - (i*0.075f + j*0.145f));
            }
         }
      }

      drawHalo(
            ringX, ringY,
            float(1.06*tmr), float(1.06*tmr),
            0.03f,
            circleSegments,
            3*numButtons*circleSegments,
            ringColor,
            colrTriVertexBuffer,
            colrTriColorBuffer);
      prevTriSat = currentTriSat;
      prevTriVal = currentTriVal;
   }
   */

   // Update colors if current Hue has changed
   if ( prevTriHue != currentTriHue ) {
      float saturation, value;
      float colors[3] = {0.0, 0.0, 0.0};
      int colrIndex = 0;
      for (int i = 0; i < numLevels; i++) {        /* Columns */
         for (int j = 0; j < numLevels-i; j++) {   /* Rows */

            // Calculate Discrete Saturation and Value
            value = 1.0f - float(j) / float(numLevels - 1);
            saturation  =  float(i) / float(numLevels - 1 - j);

            // Convert HSV to RGB
            hsv2rgb(
                  currentTriHue,
                  saturation,
                  value, 
                  colors);

            colrIndex = updateEllipseColor(
                  circleSegments, 
                  colrIndex, 
                  colors,
                  colrTriColorBuffer);
         }
      }
      prevTriHue = currentTriHue;
   }

   // Check if Selection Ring Color needs to be updated
   for (int i = 0; i < 3; i++) {
      if (colrTriColorBuffer[numButtons*circleSegments*9+i] != ringColor[i]) {
         for (unsigned int k = numButtons*circleSegments*3; k < colrTriVerts; k++) {
            colrTriColorBuffer[k*3+i] = ringColor[i];
         }
      }
   }

   py_list = PyList_New(numButtons);
   for (int i = 0; i < numButtons; i++) {
      py_tuple = PyTuple_New(4);
      PyTuple_SetItem(py_tuple, 0, PyFloat_FromDouble(triButtonData[i*4+0]));
      PyTuple_SetItem(py_tuple, 1, PyFloat_FromDouble(triButtonData[i*4+1]));
      PyTuple_SetItem(py_tuple, 2, PyFloat_FromDouble(triButtonData[i*4+2]));
      PyTuple_SetItem(py_tuple, 3, PyFloat_FromDouble(triButtonData[i*4+3]));
      PyList_SetItem(py_list, i, py_tuple);
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

   return py_list;
}
