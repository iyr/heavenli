using namespace std;

extern std::map<std::string, drawCall> drawCalls;

PyObject* drawMenu_hliGLutils(PyObject* self, PyObject* args) {
   PyObject*   faceColorPyTup;
   //PyObject*   extraColorPyTup;
   PyObject*   detailColorPyTup;
   GLfloat     gx, gy, scale, w2h, deployed, direction;
   GLfloat     faceColor[4];
   //GLfloat     extraColor[4];
   GLfloat     detailColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "ffffffOO",//O",
            &gx, &gy,
            &scale,
            &direction,
            &deployed,
            &w2h,
            &faceColorPyTup,
            //&extraColorPyTup,
            &detailColorPyTup) )
   {
      Py_RETURN_NONE;
   }

   faceColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 0));
   faceColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 1));
   faceColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 2));
   faceColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 3));

   //extraColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 0));
   //extraColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 1));
   //extraColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 2));
   //extraColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 3));

   detailColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0));
   detailColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1));
   detailColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2));
   detailColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 3));

   if (drawCalls.count("MenuOpen") <= 0)
      drawCalls.insert(std::make_pair("MenuOpen", drawCall()));
   drawCall* MenuOpen = &drawCalls["MenuOpen"];
   if (drawCalls.count("MenuClosed") <= 0)
      drawCalls.insert(std::make_pair("MenuClosed", drawCall()));
   drawCall* MenuClosed = &drawCalls["MenuClosed"];

   drawMenu(
         gx,
         gy,
         scale,
         direction,
         deployed,
         w2h,
         faceColor,
         detailColor,
         MenuOpen,
         MenuClosed
         );

   Py_RETURN_NONE;
}

GLfloat  prevDep,
         prevDir;

void drawMenu(
      GLfloat     gx,
      GLfloat     gy,
      GLfloat     scale,
      GLfloat     direction,
      GLfloat     deployed,
      GLfloat     w2h,
      GLfloat*    faceColor,
      GLfloat*    detailColor,
      drawCall*   MenuOpen,
      drawCall*   MenuClosed
         ){

   GLuint circleSegments = 60;
   GLfloat ao = 0.0f;

   // Draw single circle when menu closed
   if (deployed <= 0.0001) {
      MenuClosed->setNumColors(2);
      MenuClosed->setColorQuartet(0, faceColor);
      MenuClosed->setColorQuartet(1, detailColor);

      if (  MenuClosed->numVerts == 0   ){

         vector<GLfloat> verts;
         vector<GLfloat> colrs;

         defineEllipse(
               0.0f, 0.0f,
               1.0f, 1.0f,
               circleSegments,
               faceColor,
               verts,
               colrs
               );

         prevDir = direction;
         prevDep = deployed;
         MenuClosed->buildCache(verts.size()/2, verts, colrs);
      }

      if (  MenuClosed->colorsChanged  ){
         GLuint index = 0;
         index = updateEllipseColor(
               circleSegments,
               faceColor,
               index,
               MenuClosed->colorCache
               );

         MenuClosed->updateColorCache();
      }

      MenuClosed->updateMVP(gx, gy, scale, scale, ao, w2h);
      MenuClosed->draw();
   } 
   else
   {
      MenuOpen->setNumColors(2);
      MenuOpen->setColorQuartet(0, faceColor);
      MenuOpen->setColorQuartet(1, detailColor);

      if (  MenuOpen->numVerts == 0   ){

         vector<GLfloat> verts;
         vector<GLfloat> colrs;

         GLfloat mx=0.0f,
                 my=0.0f,
                 dmx,
                 dmy;

         dmx = cos(degToRad(direction))*deployed*7.75f;
         dmy = sin(degToRad(direction))*deployed*7.75f;
         definePill(
               mx, 
               my,
               mx + dmx,
               my + dmy,
               1.0,
               circleSegments,
               faceColor,
               verts,
               colrs
               );

         prevDir = direction;
         prevDep = deployed;
         MenuOpen->buildCache(verts.size()/2, verts, colrs);
      }

      if (  MenuOpen->colorsChanged ){
         GLuint index = 0;

         index = updatePillColor(
               circleSegments,
               faceColor,
               index,
               MenuOpen->colorCache
               );

         MenuOpen->updateColorCache();
      }

      if (  prevDep  != deployed    ||
            prevDir  != direction   ){
         GLuint index = 0;

         GLfloat mx=0.0f,
                 my=0.0f,
                 dmx,
                 dmy;

         dmx = cos(degToRad(direction))*deployed*7.75f;
         dmy = sin(degToRad(direction))*deployed*7.75f;
         index = updatePillGeometry(
               mx, 
               my,
               mx + dmx,
               my + dmy,
               1.0,
               circleSegments,
               index,
               MenuOpen->coordCache
               );

         MenuOpen->updateCoordCache();
      }

      MenuOpen->updateMVP(gx, gy, scale, scale, ao, w2h);
      MenuOpen->draw();
   }

   return;
}
