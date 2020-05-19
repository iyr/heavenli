using namespace std;

extern std::map<std::string, drawCall> drawCalls;

PyObject* drawMenu_hliGLutils(PyObject* self, PyObject* args) {
   PyObject*   faceColorPyTup;
   //PyObject*   extraColorPyTup;
   PyObject*   detailColorPyTup;
   PyObject*   py_list;
   PyObject*   py_tuple;
   //PyObject*   py_float;
   GLfloat     gx, gy, scale, w2h, deployed, direction, floatingIndex, scrollCursor;
   GLuint      numElements, menuLayout, numListings;
   GLint       drawIndex;
   GLfloat     faceColor[4];
   GLfloat     detailColor[4];
   GLfloat*    elementCoords = NULL;

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "fffffffIIIpfOO",//O",
            &gx, &gy,         // Menu Position
            &scale,           // Menu Size
            &direction,       // Direction, in degrees about the unit circle, the menu slides out to
            &deployed,        // 0.0=closed, 1.0=completely open
            &floatingIndex,   // index of the selected element, used for scroll bar
            &scrollCursor,    // animation cursor for element motion during scrolling
            &numElements,     // number of elements
            &menuLayout,      // 0=carousel w/ rollover, 1=linear strip w/ terminals, 2=value slider w/ min/max
            &numListings,     // (experimental) number of listings to display
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

   //if (drawCalls.count("MenuNormal") <= 0)
      //drawCalls.insert(std::make_pair("MenuNormal", drawCall()));
   //drawCall* MenuNormal = &drawCalls["MenuNormal"];
   if (drawCalls.count("MenuOverflow") <= 0)
      drawCalls.insert(std::make_pair("MenuOverflow", drawCall()));
   drawCall* MenuOverflow = &drawCalls["MenuOverflow"];
   if (drawCalls.count("MenuClosed") <= 0)
      drawCalls.insert(std::make_pair("MenuClosed", drawCall()));
   drawCall* MenuClosed = &drawCalls["MenuClosed"];

   elementCoords = new GLfloat[3*(numListings+1)];

   drawMenu(
         gx, gy,        // Menu Position
         scale,         // Menu Size
         direction,     // Direction, in degrees about the unit circle, the menu slides out to
         deployed,      // 0.0=closed, 1.0=completely open
         floatingIndex, // index of the selected element, used for scroll bar
         scrollCursor,  // animation cursor for element motion during scrolling (-1.0 to 1.0)
         numElements,   // number of elements
         menuLayout,    // 0=carousel w/ rollover, 1=linear strip w/ terminals, 2=value slider w/ min/max
         numListings,   // number of listings to display at once
         drawIndex,     // whether or not to draw the index over the number of elements
         elementCoords, // Relative coordinate of menu elements
         w2h,           // width to height ratio
         faceColor,     // Main color for the body of the menu
         detailColor,   // scroll bar, 
         MenuOverflow,  // drawCall object for drawing the menu open
         MenuClosed     // drawCall object for drawing the menu closed
         );

   py_list = PyList_New(numListings+1);
   for (unsigned int i = 0; i < numListings+1; i++) {
      py_tuple = PyTuple_New(3);
      PyTuple_SetItem(py_tuple, 0, PyFloat_FromDouble(elementCoords[i*3+0]));
      PyTuple_SetItem(py_tuple, 1, PyFloat_FromDouble(elementCoords[i*3+1]));
      PyTuple_SetItem(py_tuple, 2, PyFloat_FromDouble(elementCoords[i*3+2]));
      PyList_SetItem(py_list, i, py_tuple);
   }

   delete [] elementCoords;

   return py_list;
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
      GLfloat     scrollCursor,  // animation cursor for element motion during scrolling (-1.0 to 1.0)
      GLuint      numElements,   // number of elements
      GLuint      menuLayout,    // 0=carousel w/ rollover, 1=linear strip w/ terminals, 2=value slider w/ min/max
      GLuint      numListings,   // number of elements to display at once
      GLboolean   drawIndex,     // whether or not to draw the index over the number of elements
      GLfloat*    elementCoords, // Relative coordinates of Menu elements
      GLfloat     w2h,           // width to height ratio
      GLfloat*    faceColor,     // Main color for the body of the menu
      GLfloat*    detailColor,   // scroll bar, 
      //drawCall*   MenuNormal,    //
      drawCall*   MenuOverflow,  // drawCall object for drawing the menu open
      drawCall*   MenuClosed     // drawCall object for drawing the menu closed
      ){

   GLuint circleSegments = 60;

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

      MenuClosed->updateMVP(gx, gy, scale, scale, -direction, w2h);
      MenuClosed->draw();
   } 
   else if (numElements > numListings) // Draw Menu Body w/ elements+scrollbar when open
   {
      //GLfloat arrowRad = 0.05f*pow(deployed, 2.0f);
      MenuOverflow->setNumColors(2);
      MenuOverflow->setColorQuartet(0, faceColor);
      MenuOverflow->setColorQuartet(1, detailColor);

      if (  MenuOverflow->numVerts == 0   ){

         vector<GLfloat> verts;
         vector<GLfloat> colrs;

         drawMenuOverflow(
            direction,     // Direction, in degrees, the menu slides out to
            deployed,      // 0.0=closed, 1.0=completely open
            floatingIndex, // index of the selected element, used for scroll bar
            scrollCursor,  // animation cursor for element motion during scrolling: -1.0 to 1.0
            numElements,   // number of elements
            menuLayout,      // 0=carousel w/ rollover, 1=terminated linear strip
            circleSegments,
            numListings,   // number of listings to display
            drawIndex,     // whether or not to draw the index over the number of elements
            elementCoords, // Relative coordinates of Menu elements
            w2h,           // width to height ratio
            faceColor,     // Main color for the body of the menu
            detailColor,   // scroll bar, 
            verts,       // Input Vector of x,y coordinates
            colrs        // Input Vector of r,g,b values
            );

         prevDir = direction;
         prevDep = deployed;
         MenuOverflow->buildCache(verts.size()/2, verts, colrs);
      }

      if (  prevDep  != deployed    ||
            prevDir  != direction   ){  
         GLuint index = 0;
         index = drawMenuOverflow(
               direction,     // Direction, in degrees, the menu slides out to
               deployed,      // 0.0=closed, 1.0=completely open
               floatingIndex, // index of the selected element, used for scroll bar
               scrollCursor,  // element animation cursor for scrolling: -1.0 to 1.0
               numElements,   // number of elements
               menuLayout,      // 0=carousel w/ rollover, 1=terminated linear strip
               circleSegments,// number of polygon segments
               numListings,   // number of listings to display
               drawIndex,     // whether or not to draw the index over the number of elements
               elementCoords, // Relative coordinates of Menu elements
               w2h,           // width to height ratio
               index,
               MenuOverflow->coordCache
         );
         MenuOverflow->updateCoordCache();
      }

      if (  MenuOverflow->colorsChanged ){
         GLuint index = 0;
         drawMenuOverflow(
            circleSegments,
            numListings,   // number of listings to display
            faceColor,     // Main color for the body of the menu
            detailColor,   // scroll bar, 
            index,
            MenuOverflow->colorCache
            );
         MenuOverflow->updateColorCache();
      }

      MenuOverflow->updateMVP(gx, gy, scale, scale, -direction, w2h);
      MenuOverflow->draw();
   }

   return;
}
