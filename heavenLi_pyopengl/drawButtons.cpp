#include <Python.h>
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>
#include "drawUtils.h"
#define degToRad(angleInDegrees) ((angleInDegrees) * 3.1415926535 / 180.0)
using namespace std;

GLfloat  *bulbButtonVertexBuffer = NULL;
GLfloat  *bulbButtonColorBuffer  = NULL;
GLushort *bulbButtonIndices      = NULL;
GLuint   bulbButtonsVerts;
GLuint   vertsPerBulb;
float*   buttonCoords            = NULL;
int      colorsStart;
int      colorsEnd;
int      detailEnd;
int      prevNumBulbs;
int      prevArn;
float    prevAngOffset;
float    prevBulbButtonScale;
float    prevBulbButtonW2H;
PyObject* drawBulbButton_drawButtons(PyObject *self, PyObject *args)
{
   PyObject* faceColorPyTup;
   PyObject* detailColorPyTup;
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double faceColor[3];
   double detailColor[3]; 
   double *bulbColors;
   //double bulbColor[3];
   float angularOffset, scale, w2h;
   int arn, numBulbs;

   // Parse input arguments
   if (!PyArg_ParseTuple(args, 
            "iiffOOOf", 
            &arn,
            &numBulbs,
            &angularOffset,
            &scale,
            &faceColorPyTup,
            &detailColorPyTup,
            &py_list,
            &w2h))
   {
      Py_RETURN_NONE;
   }

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new double[numBulbs*3];
   for (int i = 0; i < numBulbs; i++){
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++){
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = double(PyFloat_AsDouble(py_float));
      }
   }

   // Parse RGB color tuples of face and detail colors
   for (int i = 0; i < 3; i++){
      faceColor[i] = PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, i));
      detailColor[i] = PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, i));
   }

   // Initialize / Update Vertex Geometry and Colors
   if (  prevNumBulbs != numBulbs ||
         bulbButtonVertexBuffer == NULL || 
         bulbButtonColorBuffer == NULL || 
         bulbButtonIndices == NULL || 
         buttonCoords == NULL
         ) {

      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      // Set Number of edges on circles
      char circleSegments = 60;
      char degSegment = 360 / circleSegments;

      // Setup Transformations
      if (w2h <= 1.0)
      {
         scale = w2h*scale;
      }

      if (buttonCoords == NULL) {
         buttonCoords = new float[2*numBulbs];
      } else {
         delete [] buttonCoords;
         buttonCoords = new float[2*numBulbs];
      }

      float tmx, tmy, ang;
      // Define verts / colors for each bulb button
      for (int j = 0; j < numBulbs; j++) {
         if (arn == 0) {
            ang = float(degToRad(j*360/numBulbs - 90 + angularOffset + 180/numBulbs));
         } else if (arn == 1) {
            ang = float(degToRad(
                  (j*180)/(numBulbs-1 < 1 ? 1 : numBulbs-1) + 
                  angularOffset + 
                  (numBulbs == 1 ? -90 : 0)
                  ));
         } else {
            ang = float(0.0);
         }

         // Relative coordinates of each button (from the center of the circle)
         tmx = float(0.75*cos(ang));
         tmy = float(0.75*sin(ang));
         if (w2h >= 1.0) {
            tmx *= float(pow(w2h, 0.5));
         } else {
            tmx *= float(w2h);
            tmy *= float(pow(w2h, 0.5));
         }

         buttonCoords[j*2+0] = tmx;
         buttonCoords[j*2+1] = tmy;

         // Define Vertices / Colors for Button Face
#        pragma omp parallel for
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ verts.push_back(float(tmx+0.0));
            /* Y */ verts.push_back(float(tmy+0.0));
            /* R */ colrs.push_back(float(faceColor[0]));
            /* G */ colrs.push_back(float(faceColor[1]));
            /* B */ colrs.push_back(float(faceColor[2]));

            /* X */ verts.push_back(float(tmx+0.4*cos(degToRad(i*degSegment))*scale));
            /* Y */ verts.push_back(float(tmy+0.4*sin(degToRad(i*degSegment))*scale));
            /* R */ colrs.push_back(float(faceColor[0]));
            /* G */ colrs.push_back(float(faceColor[1]));
            /* B */ colrs.push_back(float(faceColor[2]));

            /* X */ verts.push_back(float(tmx+0.4*cos(degToRad((i+1)*degSegment))*scale));
            /* Y */ verts.push_back(float(tmy+0.4*sin(degToRad((i+1)*degSegment))*scale));
            /* R */ colrs.push_back(float(faceColor[0]));
            /* G */ colrs.push_back(float(faceColor[1]));
            /* B */ colrs.push_back(float(faceColor[2]));
         }

         if (j == 0) {
            colorsStart = colrs.size();
         }
         // Define Vertices for Bulb Icon
#        pragma omp parallel for
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ verts.push_back(float(tmx+0.0*scale));
            /* Y */ verts.push_back(float(tmy+0.1*scale));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            /* X */ verts.push_back(float(tmx+0.2*cos(degToRad(i*degSegment))*scale));
            /* Y */ verts.push_back(float(tmy+(0.1+0.2*sin(degToRad(i*degSegment)))*scale));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            /* X */ verts.push_back(float(tmx+0.2*cos(degToRad((i+1)*degSegment))*scale));
            /* Y */ verts.push_back(float(tmy+(0.1+0.2*sin(degToRad((i+1)*degSegment)))*scale));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));
         }
         if (j == 0) {
            colorsEnd = colrs.size();
         }

         // Define Verts for bulb screw base
         GLfloat tmp[54] = {
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.119*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.153*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.136*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.170*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.204*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.136*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.170*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.204*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.187*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.221*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.255*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.187*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.221*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.255*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.238*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.272*scale),
            /* X, Y */ float(tmx-0.051*scale), float(tmy-0.306*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.238*scale),
            /* X, Y */ float(tmx+0.051*scale), float(tmy-0.306*scale),
            /* X, Y */ float(tmx-0.051*scale), float(tmy-0.306*scale),
         };
   
         for (int i = 0; i < 27; i++) {
            /* X */ verts.push_back(float(tmp[i*2+0]));
            /* Y */ verts.push_back(float(tmp[i*2+1]));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));
         }

         if (j == 0) {
            vertsPerBulb = verts.size()/2;
            detailEnd = colrs.size();
         }
      }
      // Pack Vertices / Colors into global array buffers
      bulbButtonsVerts = verts.size()/2;

      // (Re)allocate vertex buffer
      if (bulbButtonVertexBuffer == NULL) {
         bulbButtonVertexBuffer = new GLfloat[bulbButtonsVerts*2];
      } else {
         delete [] bulbButtonVertexBuffer;
         bulbButtonVertexBuffer = new GLfloat[bulbButtonsVerts*2];
      }

      // (Re)allocate color buffer
      if (bulbButtonColorBuffer == NULL) {
         bulbButtonColorBuffer = new GLfloat[bulbButtonsVerts*3];
      } else {
         delete [] bulbButtonColorBuffer;
         bulbButtonColorBuffer = new GLfloat[bulbButtonsVerts*3];
      }

      // (Re)allocate index array
      if (bulbButtonIndices == NULL) {
         bulbButtonIndices = new GLushort[bulbButtonsVerts];
      } else {
         delete [] bulbButtonIndices;
         bulbButtonIndices = new GLushort[bulbButtonsVerts];
      }

      // Pack bulbButtonIndices, vertex and color bufferes
#     pragma omp parallel for
      for (unsigned int i = 0; i < bulbButtonsVerts; i++){
         bulbButtonVertexBuffer[i*2]   = verts[i*2];
         bulbButtonVertexBuffer[i*2+1] = verts[i*2+1];
         bulbButtonIndices[i]          = i;
         bulbButtonColorBuffer[i*3+0]  = colrs[i*3+0];
         bulbButtonColorBuffer[i*3+1]  = colrs[i*3+1];
         bulbButtonColorBuffer[i*3+2]  = colrs[i*3+2];
      }

      prevNumBulbs = numBulbs;
      prevAngOffset = angularOffset;
      prevBulbButtonW2H = w2h;
      prevArn = arn;
      prevBulbButtonScale = scale;
   } 
   // Recalculate vertex geometry without expensive vertex/array reallocation
   else if (
         prevBulbButtonW2H != w2h ||
         prevArn != arn ||
         prevAngOffset != angularOffset ||
         prevBulbButtonScale != scale
         ) {
      // Set Number of edges on circles
      char circleSegments = 60;
      char degSegment = 360 / circleSegments;

      // Setup Transformations
      if (w2h <= 1.0)
      {
         scale = w2h*scale;
      }

      float tmx, tmy, ang;
      // Define verts / colors for each bulb button
      for (int j = 0; j < numBulbs; j++) {
         if (arn == 0) {
            ang = float(degToRad(j*360/numBulbs - 90 + angularOffset + 180/numBulbs));
         } else if (arn == 1) {
            ang = float(degToRad(
                  (j*180)/(numBulbs-1 < 1 ? 1 : numBulbs-1) + 
                  angularOffset + 
                  (numBulbs == 1 ? -90 : 0)
                  ));
         } else {
            ang = float(0.0);
         }

         // Relative coordinates of each button (from the center of the circle)
         tmx = float(0.75*cos(ang));
         tmy = float(0.75*sin(ang));
         if (w2h >= 1.0) {
            tmx *= float(pow(w2h, 0.5));
         } else {
            tmx *= float(w2h);
            tmy *= float(pow(w2h, 0.5));
         }

         buttonCoords[j*2+0] = tmx;
         buttonCoords[j*2+1] = tmy;

         // Define Vertices / Colors for Button Face
#        pragma omp parallel for
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ bulbButtonVertexBuffer[j*vertsPerBulb*2+i*6+0] = (float(tmx+0.0));
            /* Y */ bulbButtonVertexBuffer[j*vertsPerBulb*2+i*6+1] = (float(tmy+0.0));

            /* X */ bulbButtonVertexBuffer[j*vertsPerBulb*2+i*6+2] = (float(tmx+0.4*cos(degToRad(i*degSegment))*scale));
            /* Y */ bulbButtonVertexBuffer[j*vertsPerBulb*2+i*6+3] = (float(tmy+0.4*sin(degToRad(i*degSegment))*scale));

            /* X */ bulbButtonVertexBuffer[j*vertsPerBulb*2+i*6+4] = (float(tmx+0.4*cos(degToRad((i+1)*degSegment))*scale));
            /* Y */ bulbButtonVertexBuffer[j*vertsPerBulb*2+i*6+5] = (float(tmy+0.4*sin(degToRad((i+1)*degSegment))*scale));
         }

         // Define Vertices for Bulb Icon
#        pragma omp parallel for
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ bulbButtonVertexBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+0] = (float(tmx+0.0*scale));
            /* Y */ bulbButtonVertexBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+1] = (float(tmy+0.1*scale));

            /* X */ bulbButtonVertexBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+2] = (float(tmx+0.2*cos(degToRad(i*degSegment))*scale));
            /* Y */ bulbButtonVertexBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+3] = (float(tmy+(0.1+0.2*sin(degToRad(i*degSegment)))*scale));

            /* X */ bulbButtonVertexBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+4] = (float(tmx+0.2*cos(degToRad((i+1)*degSegment))*scale));
            /* Y */ bulbButtonVertexBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+5] = (float(tmy+(0.1+0.2*sin(degToRad((i+1)*degSegment)))*scale));
         }

         // Define Verts for bulb screw base
         GLfloat tmp[54] = {
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.119*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.153*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.136*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.170*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.204*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.136*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.170*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.204*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.187*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.221*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.255*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.187*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.221*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.255*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.238*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.272*scale),
            /* X, Y */ float(tmx-0.051*scale), float(tmy-0.306*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.238*scale),
            /* X, Y */ float(tmx+0.051*scale), float(tmy-0.306*scale),
            /* X, Y */ float(tmx-0.051*scale), float(tmy-0.306*scale),
         };
   
#        pragma omp parallel for
         for (int i = 0; i < 27; i++) {
            /* X */ bulbButtonVertexBuffer[j*vertsPerBulb*2+i*2+(circleSegments+1)*12+0] = (float(tmp[i*2+0]));
            /* Y */ bulbButtonVertexBuffer[j*vertsPerBulb*2+i*2+(circleSegments+1)*12+1] = (float(tmp[i*2+1]));
         }
      }

      prevAngOffset = angularOffset;
      prevBulbButtonW2H = w2h;
      prevArn = arn;
      prevBulbButtonScale = scale;
   }
   // Vertices / Geometry already calculated
   // Check if colors need to be updated
   else
   {
      /*
       * Iterate through each color channel 
       * 0 - RED
       * 1 - GREEN
       * 2 - BLUE
       */
      for (int i = 0; i < 3; i++) {

         // Update face color, if needed
         if (float(faceColor[i]) != bulbButtonColorBuffer[i]) {
            for (int j = 0; j < numBulbs; j++) {
#              pragma omp parallel for
               for (int k = 0; k < colorsStart/3; k++) {
                  bulbButtonColorBuffer[ j*vertsPerBulb*3 + k*3 + i ] = float(faceColor[i]);
               }
            }
         }

         // Update Detail Color, if needed
         if (float(detailColor[i]) != bulbButtonColorBuffer[colorsEnd+i]) {
            for (int j = 0; j < numBulbs; j++) {
#              pragma omp parallel for
               for (int k = 0; k < (detailEnd - colorsEnd)/3; k++) {
                  bulbButtonColorBuffer[ colorsEnd + j*vertsPerBulb*3 + k*3 + i ] = float(detailColor[i]);
               }
            }
         }
      }
      
      // Update any bulb colors, if needed
      // Iterate through colors (R0, G1, B2)
      for (int i = 0; i < 3; i++) {

         // Iterate though bulbs
         for (int j = 0; j < numBulbs; j++) {

            // Iterate through color buffer to update colors
            if (float(bulbColors[i+j*3]) != bulbButtonColorBuffer[colorsStart + i + j*vertsPerBulb*3]) {
#              pragma omp parallel for
               for (int k = 0; k < (colorsEnd-colorsStart)/3; k++) {
                  bulbButtonColorBuffer[ j*vertsPerBulb*3 + colorsStart + i + k*3 ] = float(bulbColors[i+j*3]);
               }
            }
         }
      }
   } 

   PyList_ClearFreeList();
   py_list = PyList_New(numBulbs);
#  pragma omp parallel for
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyTuple_New(2);
      PyTuple_SetItem(py_tuple, 0, PyFloat_FromDouble(buttonCoords[i*2+0]));
      PyTuple_SetItem(py_tuple, 1, PyFloat_FromDouble(buttonCoords[i*2+1]));
      PyList_SetItem(py_list, i, py_tuple);
   }

   // Cleanup
   delete [] bulbColors;
   
   // Copy Vertex / Color Array Bufferes to GPU, draw
   glColorPointer(3, GL_FLOAT, 0, bulbButtonColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, bulbButtonVertexBuffer);
   glDrawElements( GL_TRIANGLES, bulbButtonsVerts, GL_UNSIGNED_SHORT, bulbButtonIndices);

   return py_list;
}

GLfloat  *clockVertexBuffer = NULL;
GLfloat  *clockColorBuffer  = NULL;
GLushort *clockIndices      = NULL;
GLuint    clockVerts;
GLuint    faceVerts;
float     prevClockScale;
float     prevClockw2h;
float     prevClockHour;
float     prevClockMinute;
PyObject* drawClock_drawButtons(PyObject *self, PyObject *args)
{
   PyObject* faceColorPyTup;
   PyObject* detailColorPyTup;
   GLfloat px, py, qx, qy, radius;
   float scale, w2h, hour, minute;
   double detailColor[3];
   double faceColor[3];

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "ffffOO",
            &hour,
            &minute,
            &scale,
            &w2h,
            &detailColorPyTup,
            &faceColorPyTup)) 
   {
      Py_RETURN_NONE;
   }

   // Parse RGB color tuples of face and detail colors
   for (int i = 0; i < 3; i++){
      faceColor[i] = PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, i));
      detailColor[i] = PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, i));
   }
   
   if (  clockVertexBuffer == NULL  ||
         clockColorBuffer  == NULL  ||
         clockIndices      == NULL  ||
         prevClockScale    != scale ||
         prevClockw2h      != w2h   
         ){

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      // Set Number of edges on circles
      char circleSegments = 60;
      char degSegment = 360 / circleSegments;

      if (w2h <= 1.0)
      {
         scale = scale*w2h;
      }

      //float tmx, tmy, ang;
      for (int i = 0; i < circleSegments+1; i++) {
         /* X */ verts.push_back(float(0.0));
         /* Y */ verts.push_back(float(0.0));
         /* R */ colrs.push_back(float(faceColor[0]));
         /* G */ colrs.push_back(float(faceColor[1]));
         /* B */ colrs.push_back(float(faceColor[2]));

         /* X */ verts.push_back(float(0.5*cos(degToRad(i*degSegment))*scale));
         /* Y */ verts.push_back(float(0.5*sin(degToRad(i*degSegment))*scale));
         /* R */ colrs.push_back(float(faceColor[0]));
         /* G */ colrs.push_back(float(faceColor[1]));
         /* B */ colrs.push_back(float(faceColor[2]));

         /* X */ verts.push_back(float(0.5*cos(degToRad((i+1)*degSegment))*scale));
         /* Y */ verts.push_back(float(0.5*sin(degToRad((i+1)*degSegment))*scale));
         /* R */ colrs.push_back(float(faceColor[0]));
         /* G */ colrs.push_back(float(faceColor[1]));
         /* B */ colrs.push_back(float(faceColor[2]));
      }

      faceVerts = verts.size()/2;

      px = 0.0;
      py = 0.0;
      qx = float(0.2*cos(degToRad(90-360*(hour/12.0)))*scale);
      qy = float(0.2*sin(degToRad(90-360*(hour/12.0)))*scale);
      radius = float(0.02*scale);
      drawPill(px, py, qx, qy, radius, detailColor, verts, colrs);

      qx = float(0.4*cos(degToRad(90-360*(minute/60.0)))*scale);
      qy = float(0.4*sin(degToRad(90-360*(minute/60.0)))*scale);
      radius = float(0.01*scale);
      drawPill(px, py, qx, qy, radius, detailColor, verts, colrs);

      clockVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (clockVertexBuffer == NULL) {
         clockVertexBuffer = new GLfloat[clockVerts*2];
      } else {
         delete [] clockVertexBuffer;
         clockVertexBuffer = new GLfloat[clockVerts*2];
      }

      if (clockColorBuffer == NULL) {
         clockColorBuffer = new GLfloat[clockVerts*3];
      } else {
         delete [] clockColorBuffer;
         clockColorBuffer = new GLfloat[clockVerts*3];
      }

      if (clockIndices == NULL) {
         clockIndices = new GLushort[clockVerts];
      } else {
         delete [] clockIndices;
         clockIndices = new GLushort[clockVerts];
      }

      for (unsigned int i = 0; i < clockVerts; i++) {
         clockVertexBuffer[i*2]   = verts[i*2];
         clockVertexBuffer[i*2+1] = verts[i*2+1];
         clockIndices[i]          = i;
         clockColorBuffer[i*3+0]  = colrs[i*3+0];
         clockColorBuffer[i*3+1]  = colrs[i*3+1];
         clockColorBuffer[i*3+2]  = colrs[i*3+2];
      }

      prevClockScale    = scale;
      prevClockw2h      = w2h;
      prevClockHour     = hour;
      prevClockMinute   = minute;
   } else if (
         prevClockHour     != hour  ||
         prevClockMinute   != minute
         ){
      px = 0.0;
      py = 0.0;
      qx = float(0.2*cos(degToRad(90-360*(hour/12.0)))*scale);
      qy = float(0.2*sin(degToRad(90-360*(hour/12.0)))*scale);
      radius = float(0.02*scale);

      int tmp;
      tmp = drawPill(
            px, py, 
            qx, qy, 
            radius, 
            faceVerts, 
            //faceColor, 
            detailColor, 
            clockVertexBuffer, 
            clockColorBuffer);

      qx = float(0.4*cos(degToRad(90-360*(minute/60.0)))*scale);
      qy = float(0.4*sin(degToRad(90-360*(minute/60.0)))*scale);
      radius = float(0.01*scale);

      tmp = drawPill(
            px, py, 
            qx, qy, 
            radius, 
            tmp, 
            //faceColor, 
            detailColor, 
            clockVertexBuffer, 
            clockColorBuffer);

      prevClockHour     = hour;
      prevClockMinute   = minute;
   }

   glColorPointer(3, GL_FLOAT, 0, clockColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, clockVertexBuffer);
   glDrawElements( GL_TRIANGLES, clockVerts, GL_UNSIGNED_SHORT, clockIndices);

   Py_RETURN_NONE;
}

static PyMethodDef drawButtons_methods[] = {
   { "drawBulbButton", (PyCFunction)drawBulbButton_drawButtons, METH_VARARGS },
   { "drawClock",      (PyCFunction)drawClock_drawButtons,      METH_VARARGS },
   { NULL, NULL, 0, NULL}
};

static PyModuleDef drawButtons_module = {
   PyModuleDef_HEAD_INIT,
   "drawButtons",
   "Draws buttons",
   0,
   drawButtons_methods
};

PyMODINIT_FUNC PyInit_drawButtons() {
   //return PyModule_Create(&drawButtons_module);
   PyObject* m = PyModule_Create(&drawButtons_module);
   if (m == NULL) {
      return NULL;
   }
   return m;
}
