using namespace std;

drawCall granChangerButton;

PyObject* drawGranChanger_hliGLutils(PyObject *self, PyObject *args) {
   PyObject*   py_faceColor;
   PyObject*   py_detailColor;
   int         numHues;
   GLfloat     gx, 
               gy, 
               rotation, 
               w2h, 
               scale, 
               tDiff, 
               ao=0.0f;
   GLfloat     faceColor[4];
   GLfloat     detailColor[4];
   GLfloat     black[4] = {0.0, 0.0, 0.0, 1.0};
   GLfloat     white[4] = {1.0, 1.0, 1.0, 1.0};
   char        circleSegments = 45;
   GLuint      granChangerVerts;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "ffOOlffff",
            &gx,
            &gy,
            &py_faceColor,
            &py_detailColor,
            &numHues,
            &rotation,
            &w2h,
            &scale,
            &tDiff)) {
      Py_RETURN_NONE;
   }

   // Parse Colors
   faceColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 0)));
   faceColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 1)));
   faceColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 2)));
   faceColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 3)));
   detailColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 0)));
   detailColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 1)));
   detailColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 2)));
   detailColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 3)));

   granChangerButton.setNumColors(2);
   granChangerButton.setColorQuartet(0, faceColor);
   granChangerButton.setColorQuartet(1, detailColor);

   // Allocate and Define Geometry/Color
   if (  granChangerButton.numVerts == 0 ) {
      printf("Initializing Geometry for Granularity Rocker\n");
      float unit = float(1.0/36.0);
      float buttonSize = 0.8f;
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      // Upper Background Mask (quad)
      //defineQuad4pt(
            //-1.0, 1.0,
            //-1.0, 0.0,
            //+1.0, 1.0,
            //+1.0, 0.0,
            //black,
            //verts, 
            //colrs);

      // Lower Background Mask (Pill)
      definePill(
            -24.0f*unit, 0.0f,   // X, Y 
             24.0f*unit, 0.0f,   // X, Y 
             12.0f*unit,         // Radius 
             circleSegments,
             black,              // Color 
             verts, 
             colrs);

      // Left (Minus) Button
      defineCircle(
            -24.0f*unit, 0.0f,      // X, Y 
             12.0f*unit*buttonSize, // Radius 
             circleSegments,        // Number of Circle Triangles 
             faceColor,             // Colors 
             verts,
             colrs);

      // Right (Plus) Button
      defineCircle(
             24.0f*unit, 0.0f,      // X, Y 
             12.0f*unit*buttonSize, // Radius 
             circleSegments,        // Number of Circle Triangles 
             faceColor,             // Colors 
             verts,
             colrs);

      // Iconography
      defineCircle(-5.0f*unit*buttonSize,  6.0f*unit*buttonSize, 4.0f*unit*buttonSize, circleSegments, white, verts, colrs);
      defineCircle( 5.0f*unit*buttonSize,  0.0f*unit*buttonSize, 4.0f*unit*buttonSize, circleSegments, white, verts, colrs);
      defineCircle(-5.0f*unit*buttonSize, -6.0f*unit*buttonSize, 4.0f*unit*buttonSize, circleSegments, white, verts, colrs);

      // Minus Symbol
      float tmo = 18.0f;
      definePill(
            -32.0f*unit + tmo*unit*buttonSize, 0.0f,  // X, Y 
            -16.0f*unit - tmo*unit*buttonSize, 0.0f,  // X, Y 
            2.0f*unit*buttonSize,                     // Radius 
            circleSegments,
            detailColor,                              // Color 
            verts,
            colrs);

      // Plus Symbol
      definePill(
            32.0f*unit - tmo*unit*buttonSize, 0.0f,   // X, Y 
            16.0f*unit + tmo*unit*buttonSize, 0.0f,   // X, Y 
            2.0f*unit*buttonSize,                     // Radius 
            circleSegments,
            detailColor,                              // Color 
            verts,
            colrs);
      definePill(
            24.0f*unit,  8.0f*unit*buttonSize,  // X, Y 
            24.0f*unit, -8.0f*unit*buttonSize,  // X, Y 
            2.0f*unit*buttonSize,               // Radius 
            circleSegments,
            detailColor,                        // Color 
            verts,
            colrs);

      granChangerVerts = verts.size()/2;
      granChangerButton.buildCache(granChangerVerts, verts, colrs);
   }

   if (granChangerButton.colorsChanged) {
      unsigned int index = 0;

      // Upper Background Mask (quad)
      //index = updateQuadColor(
            //black,
            //index, 
            //granChangerButton.colorCache);

      // Lower Background Mask (Pill)
      index = updatePillColor(
             circleSegments,
             black,              // Color 
             index, 
             granChangerButton.colorCache);

      // Left (Minus) Button
      index = updateCircleColor(
             circleSegments,        // Number of Circle Triangles 
             faceColor,             // Colors 
             index,
             granChangerButton.colorCache);

      // Right (Plus) Button
      index = updateCircleColor(
             circleSegments,        // Number of Circle Triangles 
             faceColor,             // Colors 
             index,
             granChangerButton.colorCache);

      // Iconography
      index = updateCircleColor( circleSegments, white, index, granChangerButton.colorCache);
      index = updateCircleColor( circleSegments, white, index, granChangerButton.colorCache);
      index = updateCircleColor( circleSegments, white, index, granChangerButton.colorCache);

      // Minus Symbol
      index = updatePillColor(
            circleSegments,
            detailColor,                              // Color 
            index,
            granChangerButton.colorCache);

      // Plus Symbol
      index = updatePillColor(
            circleSegments,
            detailColor,                              // Color 
            index,
            granChangerButton.colorCache);
      index = updatePillColor(
            circleSegments,
            detailColor,                        // Color 
            index,
            granChangerButton.colorCache);

      granChangerButton.updateColorCache();
   }

   granChangerButton.updateMVP(gx, gy, scale, scale, ao, w2h);
   granChangerButton.draw();
   Py_RETURN_NONE;
}
