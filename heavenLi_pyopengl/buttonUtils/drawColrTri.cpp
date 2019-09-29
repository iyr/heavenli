using namespace std;

drawCall    colrTriButton;
GLfloat     *triButtonData       = NULL;  // Stores data (X, Y, sat, val) for each button dot
GLfloat     prevTriX             = 0.0;   // Used for animating granularity changes
GLfloat     prevTriY             = 0.0;   // Used for animating granularity changes
GLfloat     prevTriDotScale      = 1.0;   // Used for animating granularity changes
GLfloat     prevRingX            = 0.0;   // Used for animating selection ring
GLfloat     prevRingY            = 0.0;   // Used for animating selection ring
GLfloat     prevTriHue;                   // Used for animating selection ring
GLfloat     prevTriSat;                   // Used for animating selection ring
GLfloat     prevTriVal;                   // Used for animating selection ring
GLfloat     prevTriSatSel        = 0.0;   // Used to resolve edge-case bug for animating selection ring
GLfloat     prevTriValSel        = 0.0;   // Used to resolve edge-case bug for animating selection ring
GLint       prevColrTriNumLevels;         // Used for updating Granularity changes
GLuint      colrTriDotsVerts;

PyObject* drawColrTri_hliGLutils(PyObject *self, PyObject *args) {
   PyObject*   py_list;
   PyObject*   py_tuple;
   float       w2h, 
               scale, 
               currentTriHue, 
               currentTriSat, 
               currentTriVal, 
               tDiff, 
               gx=0.0f, 
               gy=0.0f, 
               ao=0.0f;
   float       ringColor[4];
   char        circleSegments = 24;
   long        numLevels = 6;
   GLuint      colrTriVerts;                 // Total number of vertices

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
   ringColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 3)));

   /*
    * Granularity Levels:
    * 5: Low
    * 6: Medium
    * 7: High
    */

   // Sanity Check
   if (numLevels < 5)
      numLevels = 5;
   if (numLevels > 7)
      numLevels = 7;

   long numButtons = (numLevels * (numLevels + 1)) / 2;

   // (Re)Allocate and Define Geometry/Color buffers
   if (  prevColrTriNumLevels    != numLevels   ||
         colrTriButton.numVerts  == 0           ){

      printf("Initializing Geometry for Color Triangle\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      // Allocate buffer for storing relative positions of each button
      // (Used for processing user input)
      if (triButtonData == NULL) {
         triButtonData = new float[4*numButtons];
      } else {
         delete [] triButtonData;
         triButtonData = new float[4*numButtons];
      }

      // Resolve edge-case bug
      if (  prevTriX == 0.0   )
         prevTriX = float(-0.0383*numLevels);

      if (  prevTriY == 0.0   )
         prevTriY = float(+0.0616*numLevels);

      // Actual meat of drawing saturation/value triangle
      int index = 0;
      float tmx, tmy, tmr, saturation, value, ringX = 0.0f, ringY = 0.0f;
      float colors[4] = {0.0, 0.0, 0.0, 1.0};
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
            defineEllipse(tmx, tmy, tmr, tmr, circleSegments, colors, verts, colrs);

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

      colrTriDotsVerts = verts.size()/2;

      // Draw a circle around the button dot corresponding to the selected saturation/value
      defineArch(
            ringX, ringY,
            float(1.06*tmr), 
            float(1.06*tmr),
            0.0f,
            360.0f,
            0.03f,
            circleSegments,
            ringColor,
            verts,
            colrs);

      // Total number of Vertices, useful for array updating
      colrTriVerts = verts.size()/2;

      // Build coordinate and color caches
      colrTriButton.buildCache(colrTriVerts, verts, colrs);

      // Update State Machine variables
      prevTriHue     = currentTriHue;
      prevTriSat     = currentTriSat;
      prevTriVal     = currentTriVal;
      prevTriSatSel  = currentTriSat;
      prevTriValSel  = currentTriVal;
      prevColrTriNumLevels = numLevels;
   }

   // Animate granularity changes
   // Update position of hue/sat dot triangle
   GLfloat triX = float(-0.0383f*numLevels); 
   GLfloat triY = float(+0.0616f*numLevels);
   GLfloat deltaX, deltaY;
   deltaX = triX - prevTriX;
   deltaY = triY - prevTriY;

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
      GLint index = 0;
      GLfloat tmx, tmy, tmr, saturation, value, ringX=0.0f, ringY=0.0f;
      GLfloat colors[4] = {0.0, 0.0, 0.0, 0.0};

      // Bogus comparison to get rid of compiler warnings (-_-)
      if (ringX != ringX)
         ringX = 0.0f;
      if (ringY != ringY)
         ringY = 0.0f;

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
            index = updatePrimEllipseGeometry(
                  tmx, tmy, 
                  tmr, tmr,
                  circleSegments, 
                  index, 
                  colrTriButton.coordCache);

            // Determine which button dot represents the currently selected saturation and value
            if (  abs(currentTriSat - saturation) <= 1.0f / float(numLevels) &&
                  abs(currentTriVal - value     ) <= 1.0f / float(numLevels) ){
               ringX = tmx;
               ringY = tmy;
            }
         }
      }

      // Draw a circle around the button dot corresponding to the selected saturation/value
      index = updateArchGeometry(
            ringX, ringY,
            float(1.06*tmr), 
            float(1.06*tmr),
            0.0f,
            360.0f,
            0.03f,
            circleSegments,
            index,
            colrTriButton.coordCache);

      colrTriButton.updateCoordCache();

   } else {
      prevTriX = triX;
      prevTriY = triY;
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
   GLfloat tmr, saturation, value, ringX = 0.0f, ringY = 0.0f;
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

      updateArchGeometry(
            prevRingX, prevRingY,
            float(1.06*tmr), 
            float(1.06*tmr),
            0.0f,
            360.0f,
            0.03f,
            circleSegments,
            colrTriDotsVerts,
            colrTriButton.coordCache);

      colrTriButton.updateCoordCache();
   }

   // Selection Ring in place, stop updating position
   if (  abs(deltaX) <= tDiff*0.01 &&
         abs(deltaY) <= tDiff*0.01 ){
      prevTriSat = currentTriSat;
      prevTriVal = currentTriVal;
   }
   //prevTriSatSel = currentTriSat;
   //prevTriValSel = currentTriVal;

   GLboolean updateCache = GL_FALSE;

   // Update colors if current Hue has changed
   if ( (prevTriHue != currentTriHue)        &&
        (prevColrTriNumLevels == numLevels)  ){
      GLfloat saturation, value;
      GLfloat colors[4] = {0.0, 0.0, 0.0, 1.0};
      GLint colrIndex = 0;
      for (GLint i = 0; i < prevColrTriNumLevels; i++) {        /* Columns */
         for (GLint j = 0; j < prevColrTriNumLevels-i; j++) {   /* Rows */

            // Calculate Discrete Saturation and Value
            value = 1.0f - float(j) / float(prevColrTriNumLevels - 1);
            saturation  =  float(i) / float(prevColrTriNumLevels - 1 - j);

            // Resolve issues that occur when saturation or value are less than zero or NULL
            if (saturation != saturation || saturation <= 0.0 )
               saturation = 0.000001f;
            if (value != value || value <= 0.0 )
               value = 0.000001f;

            // Convert HSV to RGB 
	         hsv2rgb(currentTriHue, saturation, value, colors);
            colrIndex = updatePrimEllipseColor(
                  circleSegments, 
                  colors,
                  colrIndex, 
                  colrTriButton.colorCache);
         }
      }
      prevTriHue = currentTriHue;

      updateCache = GL_TRUE;
   }

   // Check if selection Ring Color needs to be updated
   for (int i = 0; i < 4; i++) {
      if ( (colrTriButton.colorCache[colrTriDotsVerts*4+i] != ringColor[i])   && 
           (prevColrTriNumLevels == numLevels)                                   ){
         for (unsigned int k = colrTriDotsVerts; k < colrTriButton.numVerts; k++) {
            colrTriButton.colorCache[k*4+i] = ringColor[i];
         }

         updateCache = GL_TRUE;
      }
   }

   // Update colors, if needed
   if ( updateCache ){
      colrTriButton.updateColorCache();
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

   colrTriButton.updateMVP(gx, gy, scale, scale, ao, w2h);
   colrTriButton.draw();

   return py_list;
}
