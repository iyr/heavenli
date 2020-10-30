using namespace std;

extern std::map<std::string, drawCall> drawCalls;

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
   const char  circleSegments = 45;
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

   if (drawCalls.count("granChangerButton") <= 0)
      drawCalls.insert(std::make_pair("granChangerButton", drawCall()));
   drawCall* granChangerButton = &drawCalls["granChangerButton"];
   granChangerButton->setShader("RGBAcolor_NoTexture");

   // Parse Colors
   faceColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 0)));
   faceColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 1)));
   faceColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 2)));
   faceColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 3)));
   detailColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 0)));
   detailColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 1)));
   detailColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 2)));
   detailColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 3)));

   granChangerButton->setNumColors(2);
   granChangerButton->setColorQuartet(0, faceColor);
   granChangerButton->setColorQuartet(1, detailColor);

   // Allocate and Define Geometry/Color
   if (  granChangerButton->numVerts == 0 ) {
      printf("Initializing Geometry for Granularity Rocker\n");
      float unit        = 1.0f/36.0f;
      float buttonSize  = 0.8f;
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

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
            16.0f*unit + tmo*unit*buttonSize, 0.0f,   // X, Y 
            32.0f*unit - tmo*unit*buttonSize, 0.0f,   // X, Y 
            2.0f*unit*buttonSize,                     // Radius 
            circleSegments,
            detailColor,                              // Color 
            verts,
            colrs);
      definePill(
            24.0f*unit, -8.0f*unit*buttonSize,  // X, Y 
            24.0f*unit,  8.0f*unit*buttonSize,  // X, Y 
            2.0f*unit*buttonSize,               // Radius 
            circleSegments,
            detailColor,                        // Color 
            verts,
            colrs);

      granChangerVerts = verts.size()/2;
      map<string, attribCache> attributeData;
      attributeData[VAS.coordData] = attribCache(VAS.coordData, 2, 0, 0);
      attributeData[VAS.colorData] = attribCache(VAS.colorData, 4, 2, 1);
      attributeData[VAS.coordData].writeCache(verts.data(), verts.size());
      attributeData[VAS.colorData].writeCache(colrs.data(), colrs.size());
      granChangerButton->buildCache(granChangerVerts, attributeData);
   }

   if (granChangerButton->colorsChanged) {
      unsigned int index = 0;

      // Lower Background Mask (Pill)
      index = updatePillColor(
             circleSegments,
             black,              // Color 
             index, 
             (GLfloat *)granChangerButton->getAttribCache(VAS.colorData));

      // Left (Minus) Button
      index = updateCircleColor(
             circleSegments,        // Number of Circle Triangles 
             faceColor,             // Colors 
             index,
             (GLfloat *)granChangerButton->getAttribCache(VAS.colorData));

      // Right (Plus) Button
      index = updateCircleColor(
             circleSegments,        // Number of Circle Triangles 
             faceColor,             // Colors 
             index,
             (GLfloat *)granChangerButton->getAttribCache(VAS.colorData));

      // Iconography
      index = updateCircleColor( circleSegments, white, index, (GLfloat *)granChangerButton->getAttribCache(VAS.colorData));
      index = updateCircleColor( circleSegments, white, index, (GLfloat *)granChangerButton->getAttribCache(VAS.colorData));
      index = updateCircleColor( circleSegments, white, index, (GLfloat *)granChangerButton->getAttribCache(VAS.colorData));

      // Minus Symbol
      index = updatePillColor(
            circleSegments,
            detailColor,
            index,
            (GLfloat *)granChangerButton->getAttribCache(VAS.colorData));

      // Plus Symbol
      index = updatePillColor(
            circleSegments,
            detailColor,
            index,
            (GLfloat *)granChangerButton->getAttribCache(VAS.colorData));
      index = updatePillColor(
            circleSegments,
            detailColor,
            index,
            (GLfloat *)granChangerButton->getAttribCache(VAS.colorData));

      granChangerButton->updateBuffer(VAS.colorData);
   }

   granChangerButton->updateMVP(gx, gy, scale, scale, ao, -w2h);
   granChangerButton->draw();
   Py_RETURN_NONE;
}
