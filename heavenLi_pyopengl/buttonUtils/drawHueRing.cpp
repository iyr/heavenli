#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>

using namespace std;


GLfloat*    hueRingVertexBuffer  = NULL;  // Stores (X, Y) (float) for each vertex
GLfloat*    hueRingColorBuffer   = NULL;  // Stores (R, G, B) (float) for each vertex
GLushort*   hueRingIndices       = NULL;  // Stores index corresponding to each vertex, could be more space efficient, but meh
GLuint      hueRingVerts;                 // Total number of vertices
GLubyte     prevHueRingNumHues;           // Used for updating Granularity changes
float       prevHueDotScale      = 0.0;   // Used for animating granularity changes
float       prevHueDotDist       = 0.0;   // Used for animating granularity changes
float       prevHueRingAng       = 0.0;   // Used for animating selection ring
GLfloat     prevHueRingAni       = 0.0;   // Used to resolve edge-case bug for animating selection ring
GLfloat     prevHueRingSel       = 0.0;   // Used to resolve edge-case bug for animating selection ring
float*      hueButtonData        = NULL;  /* X, Y, hue per button */

PyObject* drawHueRing_drawButtons(PyObject *self, PyObject *args) {
   PyObject *py_list;
   PyObject *py_tuple;
   float w2h, scale, tmo, currentHue, interactionCursor, tDiff;
   float ringColor[3];
   char circleSegments = 45;
   unsigned char numHues = 12;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "flOffff",
            &currentHue,
            &numHues,
            &py_tuple,
            &w2h,
            &scale,
            &tDiff,
            &interactionCursor))
   {
      Py_RETURN_NONE;
   }

   ringColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 0)));
   ringColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 1)));
   ringColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 2)));

   // (Re)Allocate and Define Geometry/Color buffers
   if (  prevHueRingNumHues      != numHues  ||
         hueRingVertexBuffer     == NULL     ||
         hueRingColorBuffer      == NULL     ||
         hueRingIndices          == NULL     ){

      //printf("Initializing Geometry for Hue Ring\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float ang, tmx, tmy, azi;
      float colors[3] = {0.0, 0.0, 0.0};
      float tmr = float(0.15f);

      // Allocate buffer for storing relative positions of each button
      // (Used for processing user input)
      if (hueButtonData == NULL) {
         hueButtonData = new float[numHues*2];
      } else {
         delete [] hueButtonData;
         hueButtonData = new float[numHues*2];
      }

      // Actual meat of drawing hue ring
      float ringX = 100.0, ringY = 100.0;
      for (int i = 0; i < numHues; i++) {

         if (  prevHueDotScale   == 0.0   )
            prevHueDotScale = float(tmr*(12.0/numHues));

         if (  prevHueDotDist    == 0.0   )
            prevHueDotDist = float(0.67*pow(numHues/12.0f, 1.0f/4.0f));

         // Calculate distance between dots about the center
         azi = 1.0f / float(numHues);

         // Convert HSV to RGB
         hsv2rgb(float(azi*i), 1.0, 1.0, colors);

         // Calculate angle (from screen center) of dot
         ang = float(360.0*float(azi*i) + 90.0);

         // Define relative positions of hue button dots
         tmx = float(cos(degToRad(ang))*prevHueDotDist);
         tmy = float(sin(degToRad(ang))*prevHueDotDist);

         // Store position and related data of button dot
         hueButtonData[i*2+0] = tmx;
         hueButtonData[i*2+1] = tmy;
         
         // Draw dot
         drawEllipse(tmx, tmy, prevHueDotScale, circleSegments, colors, verts, colrs);

         // Determine which button dot represents the currently selected hue
         if (abs(currentHue - float(azi*i)) <= 1.0f / float(numHues*2)) {
            ringX = tmx;
            ringY = tmy;
         }
      }

      // Draw a circle around the button dot corresponding to the currently selected hue
      drawHalo(
            ringX, ringY,
            float(1.06*tmr*(12.0/numHues)), float(1.06*tmr*(12.0/numHues)),
            0.03f,
            circleSegments,
            ringColor,
            verts,
            colrs);

      // Total number of Vertices, useful for array updating
      hueRingVerts = verts.size()/2;

      // (Re)Allocate buffer for vertex data
      if (hueRingVertexBuffer == NULL) {
         hueRingVertexBuffer = new GLfloat[hueRingVerts*2];
      } else {
         delete [] hueRingVertexBuffer;
         hueRingVertexBuffer = new GLfloat[hueRingVerts*2];
      }

      // (Re)Allocate buffer for color data
      if (hueRingColorBuffer == NULL) {
         hueRingColorBuffer = new GLfloat[hueRingVerts*3];
      } else {
         delete [] hueRingColorBuffer;
         hueRingColorBuffer = new GLfloat[hueRingVerts*3];
      }

      // (Re)Allocate buffer for indices
      if (hueRingIndices == NULL) {
         hueRingIndices = new GLushort[hueRingVerts];
      } else {
         delete [] hueRingIndices;
         hueRingIndices = new GLushort[hueRingVerts];
      }

      // Pack Vertics and Colors into global array buffers
      for (unsigned int i = 0; i < hueRingVerts; i++) {
         hueRingVertexBuffer[i*2]   = verts[i*2];
         hueRingVertexBuffer[i*2+1] = verts[i*2+1];
         hueRingIndices[i]          = i;
         hueRingColorBuffer[i*3+0]  = colrs[i*3+0];
         hueRingColorBuffer[i*3+1]  = colrs[i*3+1];
         hueRingColorBuffer[i*3+2]  = colrs[i*3+2];
      }

      // Update State Machine variables
      prevHueRingAni = currentHue;
      prevHueRingSel = currentHue;
      prevHueRingNumHues = numHues;
      prevHueRingAng = float(prevHueRingAni*360.0 + 90.0);
   }

   // Resolve an edge case wehre the selection ring can sometimes get stuck
   if (  prevHueRingSel != currentHue  ) {
         prevHueRingAni = prevHueRingSel;
   }

   // Update selection cursor circle if hue selection has changed
   float curAng, deltaAng, ringX = 100.0, ringY = 100.0;
   float tmr = float(0.15f);
   curAng = float(currentHue*360.0 + 90.0);
   deltaAng = curAng - prevHueRingAng;

   // choose shortest path to new target, avoid looping around 0/360 threshold
   if (deltaAng < -180.0)
      deltaAng += 360.0;
   if (deltaAng > 180.0)
      deltaAng -= 360.0;

   tDiff *= float(2.71828);

   // Determine angle of selection ring from
   // current location (prevHueRingAng) and target location (curAng)
   if (abs(deltaAng) > tDiff) {
      if ( deltaAng < -0.0) {
         prevHueRingAng -= float(tDiff*abs(deltaAng));
      }
      if ( deltaAng >  0.0) {
         prevHueRingAng += float(tDiff*abs(deltaAng));
      }
   } else {
      prevHueRingAng = curAng;
      prevHueRingAni = currentHue;
   }

   // Update position of the selection ring if needed
   if (prevHueRingAni != currentHue){
      ringX = float(cos(degToRad(prevHueRingAng))*0.67*pow(numHues/12.0f, 1.0f/4.0f));
      ringY = float(sin(degToRad(prevHueRingAng))*0.67*pow(numHues/12.0f, 1.0f/4.0f));
      drawHalo(
            ringX, ringY,
            float(1.06*tmr*(12.0/numHues)), float(1.06*tmr*(12.0/numHues)),
            0.03f,
            circleSegments,
            3*numHues*circleSegments,
            ringColor,
            hueRingVertexBuffer,
            hueRingColorBuffer);
   }

   prevHueRingSel = currentHue;

   // Animate Granularity changes
   float hueDotScale, hueDotDist, deltaScale, deltaDist;
   hueDotScale = float(tmr*(12.0/numHues));
   hueDotDist  = float(0.67*pow(numHues/12.0f, 1.0f/4.0f));
   deltaScale  = hueDotScale - prevHueDotScale;
   deltaDist   = hueDotDist - prevHueDotDist;

   if (  abs(deltaScale) > tDiff*0.01   ||
         abs(deltaDist)  > tDiff*0.01   ){
      if (deltaScale < -0.0) {
         prevHueDotScale -= float(0.8*tDiff*abs(deltaScale));
      }
      if (deltaScale > -0.0) {
         prevHueDotScale += float(0.8*tDiff*abs(deltaScale));
      }
      if (deltaDist < -0.0) {
         prevHueDotDist -= float(0.8*tDiff*abs(deltaDist));
      }
      if (deltaDist > -0.0) {
         prevHueDotDist += float(0.8*tDiff*abs(deltaDist));
      }

      // Actual meat of drawing hue ring
      float ang, tmx, tmy, azi;
      float colors[3] = {0.0, 0.0, 0.0};
      float tmr = float(0.15f);
      int index = 0;
      float ringX = 100.0, ringY = 100.0;
      for (int i = 0; i < numHues; i++) {

         // Calculate distance between dots about the center
         azi   = 1.0f / float(numHues);

         // Convert HSV to RGB
         hsv2rgb(float(azi*i), 1.0, 1.0, colors);

         // Calculate angle (from screen center) of dot
         ang = float(360.0*float(azi*i) + 90.0);

         // Define relative positions of hue button dots
         tmx = float(cos(degToRad(ang))*prevHueDotDist);
         tmy = float(sin(degToRad(ang))*prevHueDotDist);

         // Draw dot
         index = updateEllipseGeometry(
               tmx, tmy, 
               prevHueDotScale, 
               circleSegments, 
               index, 
               hueRingVertexBuffer);

         // Determine which button dot represents the currently selected hue
         if (abs(currentHue - float(azi*i)) <= 1.0f / float(numHues*2)) {
            ringX = tmx;
            ringY = tmy;
         }
      }

      // Draw a circle around the button dot corresponding to the currently selected hue
      index = drawHalo(
            ringX, ringY,
            float(1.06*tmr*(12.0/numHues)), float(1.06*tmr*(12.0/numHues)),
            0.03f,
            circleSegments,
            index,
            ringColor,
            hueRingVertexBuffer,
            hueRingColorBuffer);

   } else {
      prevHueDotScale   = hueDotScale;
      prevHueDotDist    = hueDotDist;
   }

   // Check if selection Ring Color needs to be updated
   for (int i = 0; i < 3; i++) {
      if (hueRingColorBuffer[numHues*circleSegments*9+i] != ringColor[i]) {
         for (unsigned int k = numHues*circleSegments*3; k < hueRingVerts; k++) {
            hueRingColorBuffer[k*3+i] = ringColor[i];
         }
      }
   }
         
   // Create a Python List of tuples containing data of each button dot
   py_list = PyList_New(numHues);
   for (int i = 0; i < numHues; i++) {
      py_tuple = PyTuple_New(3);
      tmo = float(i) / float(numHues);
      PyTuple_SetItem(py_tuple, 0, PyFloat_FromDouble(hueButtonData[i*2+0]));
      PyTuple_SetItem(py_tuple, 1, PyFloat_FromDouble(hueButtonData[i*2+1]));
      PyTuple_SetItem(py_tuple, 2, PyFloat_FromDouble(tmo));
      PyList_SetItem(py_list, i, py_tuple);
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

   return py_list;
}
