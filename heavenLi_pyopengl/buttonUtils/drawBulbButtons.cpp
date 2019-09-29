using namespace std;

drawCall bulbButton;
GLuint   vertsPerBulb;
GLfloat* buttonCoords       = NULL;
GLfloat  prevAngOffset;
GLfloat  prevBulbButtonScale;
GLfloat  prevBulbButtonW2H;
GLint    colorsStart;
GLint    colorsEnd;
GLint    detailEnd;
GLint    prevNumBulbs;
GLint    prevArn;

PyObject* drawBulbButton_hliGLutils(PyObject *self, PyObject *args)
{
   PyObject*   faceColorPyTup;
   PyObject*   detailColorPyTup;
   PyObject*   py_list;
   PyObject*   py_tuple;
   PyObject*   py_float;
   GLfloat     faceColor[4];
   GLfloat     detailColor[4]; 
   GLfloat     *bulbColors;
   GLfloat     angularOffset, 
               buttonScale, 
               w2h, 
               gx=0.0f, 
               gy=0.0f, 
               scale=1.0f, 
               ao=0.0f;
   GLint    arn, 
            numBulbs, 
            circleSegments;
   GLuint   bulbButtonVerts;

   // Parse input arguments
   if (!PyArg_ParseTuple(args, 
            "iiiffOOOf", 
            &arn,
            &numBulbs,
            &circleSegments,
            &angularOffset,
            &buttonScale,
            &faceColorPyTup,
            &detailColorPyTup,
            &py_list,
            &w2h))
   {
      Py_RETURN_NONE;
   }

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new float[numBulbs*3];
   for (int i = 0; i < numBulbs; i++){
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++){
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = float(PyFloat_AsDouble(py_float));
      }
   }

   // Parse RGB color tuples of face and detail colors
   for (int i = 0; i < 4; i++){
      faceColor[i] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, i)));
      detailColor[i] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, i)));
   }

   // Initialize / Update Vertex Geometry and Colors
   if (  prevNumBulbs         != numBulbs ||
         bulbButton.numVerts  == 0        ||
         buttonCoords         == NULL     ){

      if (numBulbs > 1) {
         printf("Initializing Geometry for Bulb Buttons\n");
      } else {
         printf("Initializing Geometry for Bulb Button\n");
      }

      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      // Set Number of edges on circles
      char degSegment = 360 / circleSegments;

      // Setup Transformations
      if (w2h <= 1.0)
      {
         buttonScale = w2h*buttonScale;
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

         defineEllipse(tmx, tmy, 0.4f*buttonScale, 0.4f*buttonScale, circleSegments, faceColor, verts, colrs);
         float tmbc[4];
         tmbc[0] = bulbColors[j*3+0];
         tmbc[1] = bulbColors[j*3+1];
         tmbc[2] = bulbColors[j*3+2];
         tmbc[3] = 1.0;
         defineBulb(tmx, tmy+0.035f, 0.15f*buttonScale, circleSegments, tmbc, detailColor, verts, colrs);

         if (j == 0) {
            vertsPerBulb = verts.size()/2;
            detailEnd = colrs.size();
         }
      }
      // Pack Vertices / Colors into global array buffers
      bulbButtonVerts = verts.size()/2;

      // Update Statemachine Variables
      prevNumBulbs         = numBulbs;
      prevAngOffset        = angularOffset;
      prevBulbButtonW2H    = w2h;
      prevArn              = arn;
      prevBulbButtonScale  = buttonScale;

      bulbButton.buildCache(bulbButtonVerts, verts, colrs);
   } 

   // Recalculate vertex geometry without expensive vertex/array reallocation
   else if( prevBulbButtonW2H    != w2h            ||
            prevArn              != arn            ||
            prevAngOffset        != angularOffset  ||
            prevBulbButtonScale  != buttonScale    ){
      // Set Number of edges on circles
      char degSegment = 360 / circleSegments;

      // Setup Transformations
      if (w2h <= 1.0)
      {
         buttonScale = w2h*buttonScale;
      }

      float tmx, tmy, ang;
      int index = 0;
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

         index = updatePrimEllipseGeometry(tmx, tmy, 0.4f*buttonScale, 0.4f*buttonScale, circleSegments, index, bulbButton.coordCache);
         index = updateBulbGeometry(tmx, tmy+0.035f, 0.15f*buttonScale, circleSegments, index, bulbButton.coordCache);
      }

      // Update Statemachine Variables
      prevNumBulbs = numBulbs;
      prevAngOffset = angularOffset;
      prevBulbButtonW2H = w2h;
      prevArn = arn;
      prevBulbButtonScale = buttonScale;

      bulbButton.updateCoordCache();
   }

   // Vertices / Geometry already calculated
   // Check if colors need to be updated
   int index = 0;
   for (int j = 0; j < numBulbs; j++) {
      index = updatePrimEllipseColor(circleSegments, faceColor, index, bulbButton.colorCache);
      float tmbc[4];
      tmbc[0] = bulbColors[j*3+0];
      tmbc[1] = bulbColors[j*3+1];
      tmbc[2] = bulbColors[j*3+2];
      tmbc[3] = 1.0;
      index = updateBulbColor(circleSegments, tmbc, detailColor, index, bulbButton.colorCache);
   }
   bulbButton.updateColorCache();

   //PyList_ClearFreeList();
   py_list = PyList_New(numBulbs);
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyTuple_New(2);
      PyTuple_SetItem(py_tuple, 0, PyFloat_FromDouble(buttonCoords[i*2+0]));
      PyTuple_SetItem(py_tuple, 1, PyFloat_FromDouble(buttonCoords[i*2+1]));
      PyList_SetItem(py_list, i, py_tuple);
   }

   // Cleanup
   delete [] bulbColors;
   
   bulbButton.updateMVP(gx, gy, scale, scale, ao, w2h);
   bulbButton.draw();

   return py_list;
}
