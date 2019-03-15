#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>

using namespace std;

GLfloat  *colrTriVertexBuffer = NULL;  // Stores (X, Y) (float) for each vertex
GLfloat  *colrTriColorBuffer  = NULL;  // Stores (R, G, B) (float) for each vertex
GLushort *colrTriIndices      = NULL;  // Stores index corresponding to each vertex, could be more space efficient, but meh
GLuint   colrTriVerts;                 // Total number of vertices
GLint    prevColrTriNumLevels;         // Used for updating Granularity changes
float    *triButtonData       = NULL;  // Stores data (X, Y, sat, val) for each button dot
float    prevTriX             = 0.0;   // Used for animating granularity changes
float    prevTriY             = 0.0;   // Used for animating granularity changes
float    prevTriDotScale      = 1.0;   // Used for animating granularity changes
float    prevRingX            = 0.0;   // Used for animating selection ring
float    prevRingY            = 0.0;   // Used for animating selection ring
GLfloat  prevTriHue;                   // Used for animating selection ring
GLfloat  prevTriSat;                   // Used for animating selection ring
GLfloat  prevTriVal;                   // Used for animating selection ring
GLfloat  prevTriSatSel        = 0.0;   // Used to resolve edge-case bug for animating selection ring
GLfloat  prevTriValSel        = 0.0;   // Used to resolve edge-case bug for animating selection ring

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

   long numButtons = (numLevels * (numLevels + 1)) / 2;

   // (Re)Allocate and Define Geometry/Color buffers
   if (  prevColrTriNumLevels != numLevels   ||
         colrTriVertexBuffer  == NULL        ||
         colrTriColorBuffer   == NULL        ||
         colrTriIndices       == NULL        ){

      printf("Initializing Geometry for Color Triangle\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      // Allocate buffer for storing relative positions of each button
      // (Used for processing user input)
      if (triButtonData == NULL) {
         triButtonData = new float[4*numButtons];
         printf("numButtons: %.i\n", numButtons);
      } else {
         delete [] triButtonData;
         triButtonData = new float[4*numButtons];
         printf("numButtons: %.i\n", numButtons);
      }

      if (  prevTriX == 0.0   )
         prevTriX = float(-0.0383*numLevels);

      if (  prevTriY == 0.0   )
         prevTriY = float(+0.0616*numLevels);

      // Actual meat of drawing saturation/value triangle
      int index = 0;
      float tmx, tmy, tmr, saturation, value, ringX = 0.0f, ringY = 0.0f;
      float colors[3] = {0.0, 0.0, 0.0};
      tmr = 0.05f*prevTriDotScale;
      for (int i = 0; i < numLevels; i++) {        /* Columns */
         for (int j = 0; j < numLevels-i; j++) {   /* Rows */

            // Calculate Discrete Saturation and Value
            value = 1.0f - float(j) / float(numLevels - 1);
            saturation  =  float(i) / float(numLevels - 1 - j);

            // Resolve issues that occur when saturation or value are less than zero or NULL
            if (saturation != saturation || saturation <= 0.0)
               saturation = 0.000001f;
            if (value != value || value <= 0.0)
               value = 0.000001f;

            // Convert HSV to RGB
            hsv2rgb(currentTriHue, saturation, value, colors);

            // Define relative positions of sat/val button dots
            tmx = float(prevTriX + (i*0.13f));
            tmy = float(prevTriY - (i*0.075f + j*0.145f));

            // Draw dot
            drawEllipse(tmx, tmy, tmr, circleSegments, colors, verts, colrs);

            // Store position and related data of button dot
            triButtonData[index*4 + 0] = float(-0.0383*numLevels + (i*0.13f));
            triButtonData[index*4 + 1] = float(+0.0616*numLevels - (i*0.075f + j*0.145f));
            triButtonData[index*4 + 2] = saturation;
            triButtonData[index*4 + 3] = value;
            index++;

            // Determine which button dot represents the currently selected saturation and value
            if (  abs(currentTriSat - saturation) <= 1.0f / float(numLevels*2) &&
                  abs(currentTriVal - value     ) <= 1.0f / float(numLevels*2) ){
               ringX = tmx;
               ringY = tmy;
            }
         }
      }

      // Draw a circle around the button dot corresponding to the selected saturation/value
      drawHalo(
            ringX, ringY,
            float(1.06*tmr), float(1.06*tmr),
            0.03f,
            circleSegments,
            ringColor,
            verts,
            colrs);

      // Total number of Vertices, useful for array updating
      colrTriVerts = verts.size()/2;

      // (Re)Allocate buffer for vertex data
      if (colrTriVertexBuffer == NULL) {
         colrTriVertexBuffer = new GLfloat[colrTriVerts*2];
      } else {
         delete [] colrTriVertexBuffer;
         colrTriVertexBuffer = new GLfloat[colrTriVerts*2];
      }

      // (Re)Allocate buffer for color data
      if (colrTriColorBuffer == NULL) {
         colrTriColorBuffer = new GLfloat[colrTriVerts*3];
      } else {
         delete [] colrTriColorBuffer;
         colrTriColorBuffer = new GLfloat[colrTriVerts*3];
      }

      // (Re)Allocate buffer for indices
      if (colrTriIndices == NULL) {
         colrTriIndices = new GLushort[colrTriVerts];
      } else {
         delete [] colrTriIndices;
         colrTriIndices = new GLushort[colrTriVerts];
      }

      // Pack Vertices and Colors into global array buffers
      for (unsigned int i = 0; i < colrTriVerts; i++) {
         colrTriVertexBuffer[i*2]   = verts[i*2];
         colrTriVertexBuffer[i*2+1] = verts[i*2+1];
         colrTriIndices[i]          = i;
         colrTriColorBuffer[i*3+0]  = colrs[i*3+0];
         colrTriColorBuffer[i*3+1]  = colrs[i*3+1];
         colrTriColorBuffer[i*3+2]  = colrs[i*3+2];
      }

      // Update State Machine variables
      prevTriHue = currentTriHue;
      prevTriSat = currentTriSat;
      prevTriVal = currentTriVal;
      prevTriSatSel = currentTriSat;
      prevTriValSel = currentTriVal;
      prevColrTriNumLevels = numLevels;
      printf("Done!\n");
   }

   // Resolves edge case bug animating selection ring
   if (  prevTriSatSel  != currentTriSat  ){
      prevTriSat = prevTriSatSel;
   } 
   if (  prevTriValSel  != currentTriVal  ){
      prevTriVal = prevTriValSel;
   } 

   // Determine distance of selection ring from 
   // current location (prevTri) to target location (ring)
   float tmr, saturation, value, ringX = 0.0f, ringY = 0.0f;
   float deltaX, deltaY;
   tmr = 0.05f;
   for (int i = 0; i < numLevels; i++) {        // Columns
      for (int j = 0; j < numLevels-i; j++) {   // Rows

         // Calculate Discrete Saturation and Value
         value = 1.0f - float(j) / float(numLevels - 1);
         saturation  =  float(i) / float(numLevels - 1 - j);

         // Resolve issues that occur when saturation or value are less than zero or NULL
         if (saturation != saturation || saturation <= 0.0)
            saturation = 0.000001f;
         if (value != value || value <= 0.0)
            value = 0.000001f;

         // Define relative positions of sat/val button dots
         if (  abs(currentTriSat - saturation) <= 1.0f / float(numLevels*2) &&
               abs(currentTriVal - value     ) <= 1.0f / float(numLevels*2) ){
            ringX = float(-0.0383*numLevels + (i*0.13f));
            ringY = float(+0.0616*numLevels - (i*0.075f + j*0.145f));
         }
      }
   }

   deltaX = ringX-prevRingX;
   deltaY = ringY-prevRingY;

   // Update x-position of selection ring if needed
   if (abs(deltaX) > tDiff*0.01) {
      if (deltaX < -0.0) {
         prevRingX -= float(3.0*tDiff*abs(deltaX));
      }
      if (deltaX > -0.0) {
         prevRingX += float(3.0*tDiff*abs(deltaX));
      }
   } else {
      prevRingX = ringX;
   }

   // Update y-position of selection ring if needed
   if (abs(deltaY) > tDiff*0.01) {
      if (deltaY < -0.0) {
         prevRingY -= float(3.0*tDiff*abs(deltaY));
      }
      if (deltaY > -0.0) {
         prevRingY += float(3.0*tDiff*abs(deltaY));
      }
   } else {
      prevRingY = ringY;
   }

   // Update position of the selection ring if needed
   if ( (prevTriSat  != currentTriSat  ||
         prevTriVal  != currentTriVal) ){
      drawHalo(
            prevRingX, prevRingY,
            float(1.06*tmr), float(1.06*tmr),
            0.03f,
            circleSegments,
            3*numButtons*circleSegments,
            ringColor,
            colrTriVertexBuffer,
            colrTriColorBuffer);
   }

   // selection Ring in place, stop updating position
   if (  abs(deltaX) <= tDiff*0.01 &&
         abs(deltaY) <= tDiff*0.01 ){
      prevTriSat = currentTriSat;
      prevTriVal = currentTriVal;
   }

   prevTriSatSel = currentTriSat;
   prevTriValSel = currentTriVal;

   //  Animate granularity changes
   float triX = float(-0.0383f*numLevels); 
   float triY = float(+0.0616f*numLevels);
   deltaX = triX - prevTriX;
   deltaY = triY - prevTriY;

   // Update position of hue/sat dot triangle
   if (  abs(deltaX) > tDiff*0.01   ||
         abs(deltaY) > tDiff*0.01   ){
      if (deltaX < -0.0) {
         prevTriX -= float(3.0*tDiff*abs(deltaX));
      }
      if (deltaX > -0.0) {
         prevTriX += float(3.0*tDiff*abs(deltaX));
      }
      if (deltaY < -0.0) {
         prevTriY -= float(3.0*tDiff*abs(deltaY));
      }
      if (deltaY > -0.0) {
         prevTriY += float(3.0*tDiff*abs(deltaY));
      }

      // Actual meat of drawing saturation/value triangle
      int index = 0;
      float tmx, tmy, tmr, saturation, value, ringX = 0.0f, ringY = 0.0f;
      float colors[3] = {0.0, 0.0, 0.0};
      tmr = 0.05f*prevTriDotScale;
      for (int i = 0; i < numLevels; i++) {        /* Columns */
         for (int j = 0; j < numLevels-i; j++) {   /* Rows */

            // Calculate Discrete Saturation and Value
            value = 1.0f - float(j) / float(numLevels - 1);
            saturation  =  float(i) / float(numLevels - 1 - j);

            // Resolve issues that occur when saturation or value are less than zero or NULL
            if (saturation != saturation || saturation <= 0.0)
               saturation = 0.000001f;
            if (value != value || value <= 0.0)
               value = 0.000001f;

            // Convert HSV to RGB
            hsv2rgb(currentTriHue, saturation, value, colors);

            // Define relative positions of sat/val button dots
            tmx = float(prevTriX + (i*0.13f));
            tmy = float(prevTriY - (i*0.075f + j*0.145f));

            // Draw dot
            //drawEllipse(tmx, tmy, tmr, circleSegments, colors, verts, colrs);
            index = updateEllipseGeometry(tmx, tmy, tmr, circleSegments, index, colrTriVertexBuffer);

            // Determine which button dot represents the currently selected saturation and value
            if (  abs(currentTriSat - saturation) <= 1.0f / float(numLevels) &&
                  abs(currentTriVal - value     ) <= 1.0f / float(numLevels) ){
               ringX = tmx;
               ringY = tmy;
            }
         }
      }

      // Draw a circle around the button dot corresponding to the selected saturation/value
      index = drawHalo(
            prevRingX, prevRingY,
            float(1.06*tmr), float(1.06*tmr),
            0.03f,
            circleSegments,
            index,
            ringColor,
            colrTriVertexBuffer,
            colrTriColorBuffer);
   } else {
      prevTriX = triX;
      prevTriY = triY;
   }

   // Update colors if current Hue has changed
   if ( (prevTriHue != currentTriHue)        &&
        (prevColrTriNumLevels == numLevels)  ){
      printf("Updating Hue");
      float saturation, value;
      float colors[3] = {0.0, 0.0, 0.0};
      int colrIndex = 0;
      for (int i = 0; i < prevColrTriNumLevels; i++) {        /* Columns */
         for (int j = 0; j < prevColrTriNumLevels-i; j++) {   /* Rows */

            // Calculate Discrete Saturation and Value
            value = 1.0f - float(j) / float(prevColrTriNumLevels - 1);
            saturation  =  float(i) / float(prevColrTriNumLevels - 1 - j);

            // Convert HSV to RGB
            hsv2rgb(currentTriHue, saturation, value, colors);

            colrIndex = updateEllipseColor(
                  circleSegments, 
                  colrIndex, 
                  colors,
                  colrTriColorBuffer);
         }
      }
      prevTriHue = currentTriHue;
   }

   // Check if selection Ring Color needs to be updated
   for (int i = 0; i < 3; i++) {
      if ( (colrTriColorBuffer[numButtons*circleSegments*9+i] != ringColor[i])   && 
           (prevColrTriNumLevels == numLevels)                                   &&
           (prevTriDotScale == 1.0)                                              ){
         for (unsigned int k = numButtons*circleSegments*3; k < colrTriVerts; k++) {
            colrTriColorBuffer[k*3+i] = ringColor[i];
         }
      }
   }

   // Create a Python List of tuples containing data of each button dot
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
