#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
   // These undefs necessary because microsoft
   #undef near
   #undef far
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#include <vector>
#include <math.h>

using namespace std;

GLfloat     *hueRingCoordBuffer  = NULL;  // Stores (X, Y) (float) for each vertex
GLfloat     *hueRingColorBuffer  = NULL;  // Stores (R, G, B) (float) for each vertex
GLushort    *hueRingIndices      = NULL;  // Stores index corresponding to each vertex
GLuint      hueRingVerts;                 // Total number of vertices
GLubyte     prevHueRingNumHues;           // Used for updating Granularity changes
GLfloat     prevHueDotScale      = 0.0;   // Used for animating granularity changes
GLfloat     prevHueDotDist       = 0.0;   // Used for animating granularity changes
GLfloat     prevHueRingAng       = 0.0;   // Used for animating selection ring
GLfloat     prevHueRingAni       = 0.0;   // Used to resolve edge-case bug for animating selection ring
GLfloat     prevHueRingSel       = 0.0;   // Used to resolve edge-case bug for animating selection ring
GLfloat     *hueButtonData       = NULL;  /* X, Y, hue per button */
Matrix      hueRingMVP;                   // Transformation matrix passed to shader
Params      hueRingPrevState;             // Stores transformations to avoid redundant recalculation
GLuint      hueRingVBO;                   // Vertex Buffer Object ID
GLboolean   hueRingFirstRun = GL_TRUE;    // Determines if function is running for the first time (for VBO initialization)

PyObject* drawHueRing_drawButtons(PyObject *self, PyObject *args) {
   PyObject *py_list;
   PyObject *py_tuple;
   float w2h, scale, tmo, currentHue, interactionCursor, tDiff, gx=0.0f, gy=0.0f;
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
   if (  prevHueRingNumHues   != numHues  ||
         hueRingCoordBuffer   == NULL     ||
         hueRingColorBuffer   == NULL     ||
         hueRingIndices       == NULL     ){

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
      if (hueRingCoordBuffer == NULL) {
         hueRingCoordBuffer = new GLfloat[hueRingVerts*2];
      } else {
         delete [] hueRingCoordBuffer;
         hueRingCoordBuffer = new GLfloat[hueRingVerts*2];
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
         hueRingCoordBuffer[i*2]    = verts[i*2];
         hueRingCoordBuffer[i*2+1]  = verts[i*2+1];
         hueRingIndices[i]          = i;
         hueRingColorBuffer[i*3+0]  = colrs[i*3+0];
         hueRingColorBuffer[i*3+1]  = colrs[i*3+1];
         hueRingColorBuffer[i*3+2]  = colrs[i*3+2];
      }

      // Calculate initial transformation matrix
      Matrix Ortho;
      Matrix ModelView;

      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, 1.0f*gx, 1.0f*gy, 0.0f );
      if (w2h <= 1.0f) {
         MatrixScale( &ModelView, scale, scale*w2h, 1.0f );
      } else {
         MatrixScale( &ModelView, scale/w2h, scale, 1.0f );
      }
      //MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &hueRingMVP, &ModelView, &Ortho );

      //hueRingPrevState.ao = ao;
      hueRingPrevState.dx = gx;
      hueRingPrevState.dy = gy;
      hueRingPrevState.sx = scale;
      hueRingPrevState.sy = scale;
      hueRingPrevState.w2h = w2h;

      // Update State Machine variables
      prevHueRingAni = currentHue;
      prevHueRingSel = currentHue;
      prevHueRingNumHues = numHues;
      prevHueRingAng = float(prevHueRingAni*360.0 + 90.0);

      // Create buffer if one does not exist, otherwise, delete and make a new one
      if (hueRingFirstRun == GL_TRUE) {
         hueRingFirstRun = GL_FALSE;
         glGenBuffers(1, &hueRingVBO);
      } else {
         glDeleteBuffers(1, &hueRingVBO);
         glGenBuffers(1, &hueRingVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, hueRingVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*hueRingVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLuint64 offset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*hueRingVerts, hueRingCoordBuffer);
      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLuint64*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*hueRingVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*hueRingVerts, hueRingColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (GLuint64*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   }

   // Resolve an edge case where the selection ring can sometimes get stuck
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
            hueRingCoordBuffer,
            hueRingColorBuffer);

      // Update Contents of VBO
      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, hueRingVBO);
      // Convenience variable
      GLuint64 offset = 0;
      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*hueRingVerts, hueRingCoordBuffer);
      offset += 2*sizeof(GLfloat)*hueRingVerts;
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*hueRingVerts, hueRingColorBuffer);
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
      float tmr = float(0.15f);
      int index = 0;
      float ringX = 100.0, ringY = 100.0;
      for (int i = 0; i < numHues; i++) {

         // Calculate distance between dots about the center
         azi   = 1.0f / float(numHues);

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
               hueRingCoordBuffer);

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
            hueRingCoordBuffer,
            hueRingColorBuffer);

      // Update Contents of VBO
      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, hueRingVBO);
      // Convenience variables
      GLuint64 offset = 0;
      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*hueRingVerts, hueRingCoordBuffer);

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

         // Update Contents of VBO
         // Set active VBO
         glBindBuffer(GL_ARRAY_BUFFER, hueRingVBO);
         // Convenience variable
         GLuint64 offset = 2*sizeof(GLfloat)*hueRingVerts;
         // Load Vertex coordinate data into VBO
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*hueRingVerts, hueRingColorBuffer);
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

   // Update Transfomation Matrix if any change in parameters
   if (  //hueRingPrevState.ao != ao     ||
         hueRingPrevState.dx != gx     ||
         hueRingPrevState.dy != gy     ||
         hueRingPrevState.sx != scale  ||
         hueRingPrevState.sy != scale  ||
         hueRingPrevState.w2h != w2h   ){
      
      Matrix Ortho;
      Matrix ModelView;

      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, 1.0f*gx, 1.0f*gy, 0.0f );
      if (w2h <= 1.0f) {
         MatrixScale( &ModelView, scale, scale*w2h, 1.0f );
      } else {
         MatrixScale( &ModelView, scale/w2h, scale, 1.0f );
      }
      //MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &hueRingMVP, &ModelView, &Ortho );

      //hueRingPrevState.ao = ao;
      hueRingPrevState.dx = gx;
      hueRingPrevState.dy = gy;
      hueRingPrevState.sx = scale;
      hueRingPrevState.sy = scale;
      hueRingPrevState.w2h = w2h;

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, hueRingVBO);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*hueRingVerts));
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &hueRingMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, hueRingVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*hueRingVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, hueRingVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   return py_list;
}
