using namespace std;

drawCall    hueRingButton;

GLuint      hueDotsVerts;                 // Used for updating only a portion of the color cache
GLubyte     prevHueRingNumHues;           // Used for updating Granularity changes
GLfloat     prevHueDotScale      = 0.0;   // Used for animating granularity changes
GLfloat     prevHueDotDist       = 0.0;   // Used for animating granularity changes
GLfloat     prevHueRingAng       = 0.0;   // Used for animating selection ring
GLfloat     prevHueRingAni       = 0.0;   // Used to resolve edge-case bug for animating selection ring
GLfloat     prevHueRingSel       = 0.0;   // Used to resolve edge-case bug for animating selection ring
GLfloat     *hueButtonData       = NULL;  /* X, Y, hue per button */

PyObject* drawHueRing_hliGLutils(PyObject *self, PyObject *args) {
   PyObject*      py_list;
   PyObject*      py_tuple;
   GLfloat        w2h, 
                  scale, 
                  tmo, 
                  currentHue, 
                  tDiff, 
                  gx=0.0f, 
                  gy=0.0f, 
                  ao=0.0f;
   GLfloat        ringColor[4];
   char           circleSegments = 45;
   unsigned char  numHues = 12;
   GLuint         hueRingVerts;                 // Total number of vertices

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "ffffbOff",
            &gx, &gy,
            &scale,
            &currentHue,
            &numHues,
            &py_tuple,
            &w2h,
            &tDiff
            ))
   {
      Py_RETURN_NONE;
   }

   ringColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 0)));
   ringColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 1)));
   ringColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 2)));
   ringColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 3)));

   // (Re)Allocate and Define Geometry/Color buffers
   if (  prevHueRingNumHues      != numHues  ||
         hueRingButton.numVerts  == 0        ){

      //printf("Initializing Geometry for Hue Ring\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float ang, tmx, tmy, azi;
      float colors[4] = {0.0f, 0.0f, 0.0f, 1.0f};
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
      float ringX = 100.0f, ringY = 100.0f;
      for (int i = 0; i < numHues; i++) {

         if (  prevHueDotScale   == 0.0f   )
            prevHueDotScale = float(tmr*(12.0f/numHues));

         if (  prevHueDotDist    == 0.0f   )
            prevHueDotDist = float(0.67f*pow(numHues/12.0f, 1.0f/4.0f));

         // Calculate distance between dots about the center
         azi = 1.0f / float(numHues);

         // Convert HSV to RGB
         hsv2rgb(float(azi*i), 1.0f, 1.0f, colors);

         // Calculate angle (from screen center) of dot
         ang = float(360.0f*float(azi*i) + 90.0f);

         // Define relative positions of hue button dots
         tmx = float(cos(degToRad(ang))*prevHueDotDist);
         tmy = float(sin(degToRad(ang))*prevHueDotDist);

         // Store position and related data of button dot
         hueButtonData[i*2+0] = tmx;
         hueButtonData[i*2+1] = tmy;
         
         // Draw dot
         //drawEllipse(tmx, tmy, prevHueDotScale, circleSegments, colors, verts, colrs);
         defineEllipse(tmx, tmy, prevHueDotScale, prevHueDotScale, circleSegments, colors, verts, colrs);

         // Determine which button dot represents the currently selected hue
         if (abs(currentHue - float(azi*i)) <= 1.0f / float(numHues*2)) {
            ringX = tmx;
            ringY = tmy;
         }
      }

      hueDotsVerts = verts.size()/2;

      // Draw a circle around the button dot corresponding to the currently selected hue
      defineArch(
            ringX, ringY,
            float(1.06f*tmr*(12.0f/numHues)), 
            float(1.06f*tmr*(12.0f/numHues)),
            0.0f,
            360.0f,
            0.03f,
            circleSegments,
            ringColor,
            verts,
            colrs);

      // Total number of Vertices, useful for array updating
      hueRingVerts = verts.size()/2;

      // Update State Machine variables
      prevHueRingAni = currentHue;
      prevHueRingSel = currentHue;
      prevHueRingNumHues = numHues;
      prevHueRingAng = float(prevHueRingAni*360.0f + 90.0f);

      hueRingButton.buildCache(hueRingVerts, verts, colrs);

   }

   // Resolve an edge case where the selection ring can sometimes get stuck
   if (  prevHueRingSel != currentHue  ) {
         prevHueRingAni = prevHueRingSel;
   }

   // Update selection cursor circle if hue selection has changed
   float curAng, deltaAng, ringX = 100.0f, ringY = 100.0f;
   float tmr = float(0.15f);
   curAng = float(currentHue*360.0f + 90.0f);
   deltaAng = curAng - prevHueRingAng;

   // choose shortest path to new target, avoid looping around 0/360 threshold
   if (deltaAng < -180.0f)
      deltaAng += 360.0f;
   if (deltaAng > 180.0f)
      deltaAng -= 360.0f;

   tDiff *= float(2.71828f);

   // Determine angle of selection ring from
   // current location (prevHueRingAng) and target location (curAng)
   if (abs(deltaAng) > tDiff) {
      if ( deltaAng < -0.0f) {
         prevHueRingAng -= float(tDiff*abs(deltaAng));
      }
      if ( deltaAng >  0.0f) {
         prevHueRingAng += float(tDiff*abs(deltaAng));
      }
   } else {
      prevHueRingAng = curAng;
      prevHueRingAni = currentHue;
   }

   // Update position of the selection ring if needed
   if (prevHueRingAni != currentHue){
      ringX = float(cos(degToRad(prevHueRingAng))*0.67f*pow(numHues/12.0f, 1.0f/4.0f));
      ringY = float(sin(degToRad(prevHueRingAng))*0.67f*pow(numHues/12.0f, 1.0f/4.0f));
      updateArchGeometry(
            ringX, ringY,
            float(1.06f*tmr*(12.0f/numHues)), 
            float(1.06f*tmr*(12.0f/numHues)),
            0.0f,
            360.0f,
            0.03f,
            circleSegments,
            hueDotsVerts,
            hueRingButton.coordCache);

      hueRingButton.updateCoordCache();
   }

   prevHueRingSel = currentHue;

   // Animate Granularity changes
   float hueDotScale, hueDotDist, deltaScale, deltaDist;
   hueDotScale = float(tmr*(12.0f/numHues));
   hueDotDist  = float(0.67f*pow(numHues/12.0f, 1.0f/4.0f));
   deltaScale  = hueDotScale - prevHueDotScale;
   deltaDist   = hueDotDist - prevHueDotDist;

   if (  abs(deltaScale) > tDiff*0.01f ||
         abs(deltaDist)  > tDiff*0.01f ){
      if (deltaScale < -0.0f) {
         prevHueDotScale -= float(0.8f*tDiff*abs(deltaScale));
      }
      if (deltaScale > -0.0f) {
         prevHueDotScale += float(0.8f*tDiff*abs(deltaScale));
      }
      if (deltaDist < -0.0f) {
         prevHueDotDist -= float(0.8f*tDiff*abs(deltaDist));
      }
      if (deltaDist > -0.0f) {
         prevHueDotDist += float(0.8f*tDiff*abs(deltaDist));
      }

      // Actual meat of drawing hue ring
      float ang, tmx, tmy, azi;
      float tmr = float(0.15f);
      int index = 0;
      float ringX = 100.0f, ringY = 100.0f;
      for (int i = 0; i < numHues; i++) {

         // Calculate distance between dots about the center
         azi   = 1.0f / float(numHues);

         // Calculate angle (from screen center) of dot
         ang = float(360.0f*float(azi*i) + 90.0f);

         // Define relative positions of hue button dots
         tmx = float(cos(degToRad(ang))*prevHueDotDist);
         tmy = float(sin(degToRad(ang))*prevHueDotDist);

         // Draw dot
         index = updateEllipseGeometry(
               tmx, tmy, 
               prevHueDotScale, 
               prevHueDotScale, 
               circleSegments, 
               index, 
               hueRingButton.coordCache);

         // Determine which button dot represents the currently selected hue
         if (abs(currentHue - float(azi*i)) <= 1.0f / float(numHues*2)) {
            ringX = tmx;
            ringY = tmy;
         }
      }

      // Draw a circle around the button dot corresponding to the currently selected hue
      index = updateArchGeometry(
            ringX, ringY,
            float(1.06f*tmr*(12.0f/numHues)), 
            float(1.06f*tmr*(12.0f/numHues)),
            0.0f,
            360.0f,
            0.03f,
            circleSegments,
            hueDotsVerts,
            hueRingButton.coordCache);

      hueRingButton.updateCoordCache();
   
   } else {
      prevHueDotScale   = hueDotScale;
      prevHueDotDist    = hueDotDist;
   }

   GLboolean updateCache = GL_FALSE;
   // Check if selection Ring Color needs to be updated
   for (int i = 0; i < 4; i++) {
      if (hueRingButton.colorCache[hueDotsVerts*4+i] != ringColor[i]) {
         for (unsigned int k = hueDotsVerts; k < hueRingButton.numVerts; k++) {
            hueRingButton.colorCache[k*4+i] = ringColor[i];
         }

         updateCache = GL_TRUE;
      }
   }

   // Update colors, if needed
   if ( updateCache ){
      hueRingButton.updateColorCache();
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

   hueRingButton.updateMVP(gx, gy, scale, scale, ao, w2h);
   hueRingButton.draw();

   return py_list;
}
