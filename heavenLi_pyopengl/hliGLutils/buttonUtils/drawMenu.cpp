using namespace std;

extern std::map<std::string, drawCall> drawCalls;

PyObject* drawMenu_hliGLutils(PyObject* self, PyObject* args) {
   PyObject*   faceColorPyTup;
   //PyObject*   extraColorPyTup;
   PyObject*   detailColorPyTup;
   GLfloat     gx, gy, scale, w2h, deployed, direction, floatingIndex;
   GLuint      numElements, menuType;
   GLint       drawIndex;
   GLfloat     faceColor[4];
   GLfloat     detailColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "ffffffIIpfOO",//O",
            &gx, &gy,         // Menu Position
            &scale,           // Menu Size
            &direction,       // Direction, in degrees about the unit circle, the menu slides out to
            &deployed,        // 0.0=closed, 1.0=completely open
            &floatingIndex,   // index of the selected element, used for scroll bar
            &numElements,     // number of elements
            &menuType,        // 0=carousel w/ rollover, 1=linear strip w/ terminals, 2=value slider w/ min/max
            &drawIndex,       // whether or not to draw the index over the number of elements
            &w2h,             //
            &faceColorPyTup,
            &detailColorPyTup
            ) )
   {
      Py_RETURN_NONE;
   }

   faceColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 0));
   faceColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 1));
   faceColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 2));
   faceColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 3));

   detailColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0));
   detailColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1));
   detailColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2));
   detailColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 3))*deployed;

   if (drawCalls.count("MenuOpen") <= 0)
      drawCalls.insert(std::make_pair("MenuOpen", drawCall()));
   drawCall* MenuOpen = &drawCalls["MenuOpen"];
   if (drawCalls.count("MenuClosed") <= 0)
      drawCalls.insert(std::make_pair("MenuClosed", drawCall()));
   drawCall* MenuClosed = &drawCalls["MenuClosed"];

   drawMenu(
         gx, gy,        // Menu Position
         scale,         // Menu Size
         direction,     // Direction, in degrees about the unit circle, the menu slides out to
         deployed,      // 0.0=closed, 1.0=completely open
         floatingIndex, // index of the selected element, used for scroll bar
         numElements,   // number of elements
         menuType,      // 0=carousel w/ rollover, 1=linear strip w/ terminals, 2=value slider w/ min/max
         drawIndex,     // whether or not to draw the index over the number of elements
         w2h,           // width to height ratio
         faceColor,     // Main color for the body of the menu
         detailColor,   // scroll bar, 
         MenuOpen,      // drawCall object for drawing the menu open
         MenuClosed     // drawCall object for drawing the menu closed
         );

   Py_RETURN_NONE;
}

GLfloat  prevDep,
         prevDir;

void drawMenu(
      GLfloat     gx, 
      GLfloat     gy,            // Menu Position
      GLfloat     scale,         // Menu Size
      GLfloat     direction,     // Direction, in degrees about the unit circle, the menu slides out to
      GLfloat     deployed,      // 0.0=closed, 1.0=completely open
      GLfloat     floatingIndex, // index of the selected element, used for scroll bar
      GLuint      numElements,   // number of elements
      GLuint      menuType,      // 0=carousel w/ rollover, 1=linear strip w/ terminals, 2=value slider w/ min/max
      GLboolean   drawIndex,     // whether or not to draw the index over the number of elements
      GLfloat     w2h,           // width to height ratio
      GLfloat*    faceColor,     // Main color for the body of the menu
      GLfloat*    detailColor,   // scroll bar, 
      drawCall*   MenuOpen,      // drawCall object for drawing the menu open
      drawCall*   MenuClosed     // drawCall object for drawing the menu closed
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
      GLfloat arrowRad = 0.1f*pow(deployed, 2.0);
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

         dmx = cos(degToRad(direction))*deployed;
         dmy = sin(degToRad(direction))*deployed;

         // Menu Body
         definePill(
               mx, 
               my,
               mx + dmx*7.75f,
               my + dmy*7.75f,
               1.0,
               circleSegments,
               faceColor,
               verts,
               colrs
               );

         // Distil Arrow
         definePill(
               mx+dmx*8.5f,
               my+dmy*8.5f,
               mx+dmx*8.25f+dmy*0.25f,
               my+dmy*8.25f+dmx*0.25f,
               arrowRad,
               circleSegments/4,
               detailColor,
               verts,
               colrs
               );
         definePill(
               mx+dmx*8.5f,
               my+dmy*8.5f,
               mx+dmx*8.25f-dmy*0.25f,
               my+dmy*8.25f-dmx*0.25f,
               arrowRad,
               circleSegments/4,
               detailColor,
               verts,
               colrs
               );

         // Proximal Arrow
         definePill(
               mx+dmx*2.0f,
               my+dmy*2.0f,
               mx+dmx*2.25f+dmy*0.25f,
               my+dmy*2.25f+dmx*0.25f,
               arrowRad,
               circleSegments/4,
               detailColor,
               verts,
               colrs
               );
         definePill(
               mx+dmx*2.0f,
               my+dmy*2.0f,
               mx+dmx*2.25f-dmy*0.25f,
               my+dmy*2.25f-dmx*0.25f,
               arrowRad,
               circleSegments/4,
               detailColor,
               verts,
               colrs
               );


         prevDir = direction;
         prevDep = deployed;
         MenuOpen->buildCache(verts.size()/2, verts, colrs);
      }

      if (  MenuOpen->colorsChanged ){
         GLuint index = 0;

         // Menu Body
         index = updatePillColor(
               circleSegments,
               faceColor,
               index,
               MenuOpen->colorCache
               );

         // Distil Arrow
         index = updatePillColor(
               circleSegments/4,
               detailColor,
               index,
               MenuOpen->colorCache
               );
         index = updatePillColor(
               circleSegments/4,
               detailColor,
               index,
               MenuOpen->colorCache
               );

         // Proximal Arrow
         index = updatePillColor(
               circleSegments/4,
               detailColor,
               index,
               MenuOpen->colorCache
               );
         index = updatePillColor(
               circleSegments/4,
               detailColor,
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

         dmx = cos(degToRad(direction))*deployed;
         dmy = sin(degToRad(direction))*deployed;

         // Menu Body
         index = updatePillGeometry(
               mx, 
               my,
               mx + dmx*7.75f,
               my + dmy*7.75f,
               1.0,
               circleSegments,
               index,
               MenuOpen->coordCache
               );

         // Distil Arrow
         index = updatePillGeometry(
               mx+dmx*8.5f,
               my+dmy*8.5f,
               mx+dmx*8.25f+dmy*0.25f,
               my+dmy*8.25f+dmx*0.25f,
               arrowRad,
               circleSegments/4,
               index,
               MenuOpen->coordCache
               );
         index = updatePillGeometry(
               mx+dmx*8.5f,
               my+dmy*8.5f,
               mx+dmx*8.25f-dmy*0.25f,
               my+dmy*8.25f-dmx*0.25f,
               arrowRad,
               circleSegments/4,
               index,
               MenuOpen->coordCache
               );

         // Proximal Arrow
         index = updatePillGeometry(
               mx+dmx*2.0f,
               my+dmy*2.0f,
               mx+dmx*2.25f+dmy*0.25f,
               my+dmy*2.25f+dmx*0.25f,
               arrowRad,
               circleSegments/4,
               index,
               MenuOpen->coordCache
               );
         index = updatePillGeometry(
               mx+dmx*2.0f,
               my+dmy*2.0f,
               mx+dmx*2.25f-dmy*0.25f,
               my+dmy*2.25f-dmx*0.25f,
               arrowRad,
               circleSegments/4,
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
