using namespace std;

extern std::map<std::string, drawCall> drawCalls;

void drawPill(
      GLfloat     px, 
      GLfloat     py,
      GLfloat     qx,
      GLfloat     qy,
      GLfloat     radius,
      GLfloat     w2h,
      GLfloat*    pColor,
      GLfloat*    qColor,
      drawCall*   pillButton
      );

PyObject* drawPill_hliGLutils(PyObject* self, PyObject *args) {
   PyObject *pColorPyTup;
   PyObject *qColorPyTup;
   float px, py, qx, qy, w2h, radius;
   float pColor[4];
   float qColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "ffffffOO",
            &px, &py,
            &qx, &qy,
            &radius,
            &w2h,
            &pColorPyTup,
            &qColorPyTup
            ) )
   {
      Py_RETURN_NONE;
   }

   pColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(pColorPyTup, 0)));
   pColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(pColorPyTup, 1)));
   pColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(pColorPyTup, 2)));
   pColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(pColorPyTup, 3)));

   qColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(qColorPyTup, 0)));
   qColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(qColorPyTup, 1)));
   qColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(qColorPyTup, 2)));
   qColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(qColorPyTup, 3)));

   if (drawCalls.count("pillButton") <= 0)
      drawCalls.insert(std::make_pair("pillButton", drawCall()));
   drawCall* pillButton = &drawCalls["pillButton"];

   drawPill(
         px, 
         py,
         qx,
         qy,
         radius,
         w2h,
         pColor,
         qColor,
         pillButton
         );

   Py_RETURN_NONE;
}

void drawPill(
      GLfloat     px, 
      GLfloat     py,
      GLfloat     qx,
      GLfloat     qy,
      GLfloat     radius,
      GLfloat     w2h,
      GLfloat*    pColor,
      GLfloat*    qColor,
      drawCall*   pillButton
      ){
   static GLfloat prevPx,
                  prevPy,
                  prevQx,
                  prevQy,
                  prevRad;

   GLuint pillVerts;
   pillButton->setNumColors(2);
   pillButton->setColorQuartet(0, pColor);
   pillButton->setColorQuartet(1, qColor);

   int circleSegments = 60;
   if (pillButton->numVerts == 0){

      printf("Initializing Geometry for Pill Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      // Draw button face
      definePill(
            px, py,
            qx, qy,
            radius,
            circleSegments,
            pColor,
            qColor,
            verts, 
            colrs
            );

      pillVerts = verts.size()/2;

      prevPx   = px;
      prevPy   = py;
      prevQx   = qx;
      prevQy   = qy;
      prevRad  = radius;

      pillButton->buildCache(pillVerts, verts, colrs);
   }

   if (  prevPx   != px    ||
         prevPy   != py    ||
         prevQx   != qx    ||
         prevQy   != qy    ||
         prevRad  != radius){

      GLint index = 0;

      index = updatePillGeometry(
            px, py,
            qx, qy,
            radius,
            circleSegments,
            index,
            pillButton->coordCache
            );

      prevPx   = px;
      prevPy   = py;
      prevQx   = qx;
      prevQy   = qy;
      prevRad  = radius;

      pillButton->updateCoordCache();
   }

   if (pillButton->colorsChanged) {
      unsigned int index = 0;
      // Draw button face
      index = updatePillColor(
            circleSegments,
            pColor,
            qColor,
            index, 
            pillButton->colorCache);

      pillButton->updateColorCache();
   }

   float gx=0.0f, 
         gy=0.0f, 
         sx=1.0f, 
         sy=1.0f;

   pillButton->updateMVP(gx, gy, sx, sy, 0.0f, w2h);
   pillButton->draw();

   return;
}
