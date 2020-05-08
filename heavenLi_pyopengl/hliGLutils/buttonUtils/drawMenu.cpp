using namespace std;

extern std::map<std::string, drawCall> drawCalls;

PyObject* drawMenu_hliGLutils(PyObject* self, PyObject* args) {
   PyObject*   faceColorPyTup;
   //PyObject*   extraColorPyTup;
   PyObject*   detailColorPyTup;
   GLfloat     gx, gy, scale, w2h, deployed, direction, floatingIndex, scrollCursor;
   GLuint      numElements, menuType;
   GLint       drawIndex;
   GLfloat     faceColor[4];
   GLfloat     detailColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "fffffffIIpfOO",//O",
            &gx, &gy,         // Menu Position
            &scale,           // Menu Size
            &direction,       // Direction, in degrees about the unit circle, the menu slides out to
            &deployed,        // 0.0=closed, 1.0=completely open
            &floatingIndex,   // index of the selected element, used for scroll bar
            &scrollCursor,    // animation cursor for element motion during scrolling
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
         scrollCursor,  // animation cursor for element motion during scrolling (-1.0 to 1.0)
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
      GLfloat     scrollCursor,  // animation cursor for element motion during scrolling (-1.0 to 1.0)
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
   GLuint numListings = 3;

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
   else
   {
      GLfloat arrowRad = 0.05f*pow(deployed, 2.0f);
      MenuOpen->setNumColors(2);
      MenuOpen->setColorQuartet(0, faceColor);
      MenuOpen->setColorQuartet(1, detailColor);

      if (  MenuOpen->numVerts == 0   ){

         vector<GLfloat> verts;
         vector<GLfloat> colrs;

         GLfloat mx=0.0f,  // Origin of Menu
                 my=0.0f,  // Origin of Menu
                 tmo;     // local coordinate offset

         tmo = 5.75f+(GLfloat)drawIndex;

         // Menu Body
         definePill(
               mx, 
               my,
               mx + tmo*deployed,
               my,
               1.0,
               circleSegments,
               faceColor,
               verts,
               colrs
               );

         // Element Diamonds
         if (abs(scrollCursor) == 0.0f) {
            GLfloat tms = 1.0f;
            for (int i = 0; i < numListings+1; i++) {
               if (i == numListings)
                  tms = 0.0f;
               defineEllipse(
                     mx + (2.0f + (GLfloat)i*1.75f + (GLfloat)drawIndex)*deployed,
                     my,
                     0.1f*tms,
                     0.1f*tms,
                     2,
                     detailColor,
                     verts,
                     colrs
                     );
            }
         } else {
            GLfloat tms = 1.0f;
            GLfloat tmx, tma;
            for (int i = 0; i < numListings+1; i++) {

               if (i == 0) {
                  tma = -3.0f*pow(scrollCursor, 2.0f) + 4.0f*scrollCursor;
                  tms = scrollCursor;
                  tmx = mx + (
                        2.0f + 
                        ((GLfloat)i - tma + 1.0f)*1.75f + 
                        (GLfloat)drawIndex
                        )*deployed;
               } else if (i == numListings) {
                  tms -= abs(scrollCursor);
                  tma = -3.0f*pow(1.0f-scrollCursor, 2.0f) + 4.0f*(1.0f-scrollCursor);
                  tmx = mx + (
                        2.0f + 
                        ((GLfloat)i + tma - 2.0f)*1.75f + 
                        (GLfloat)drawIndex
                        )*deployed;
               } else {
                  tms = 1.0f;
                  tmx = mx + (
                        2.0f + 
                        ((GLfloat)i + scrollCursor - 1.0f)*1.75f + 
                        (GLfloat)drawIndex
                        )*deployed;
               }
               defineEllipse(
                     tmx,
                     my,
                     0.1f*tms,
                     0.1f*tms,
                     2,
                     detailColor,
                     verts,
                     colrs
                     );
            }
         }

         tmo = 3.75f;

         // Center Arrow
         definePill(
               mx+tmo*deployed,
               my+0.85f*pow(deployed, 3.0f),
               mx+tmo*deployed-0.20f*deployed,
               my+1.00f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               detailColor,
               verts,
               colrs
               );
         definePill(
               mx+tmo*deployed,
               my+0.85f*pow(deployed, 3.0f),
               mx+tmo*deployed+0.20f*deployed,
               my+1.00f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               detailColor,
               verts,
               colrs
               );
         tmo = 6.5f;

         // Distil Arrow
         definePill(
               mx+tmo*deployed,
               my,
               mx+tmo*deployed-0.25f*deployed,
               my+0.75f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               detailColor,
               verts,
               colrs
               );
         definePill(
               mx+tmo*deployed,
               my,
               mx+tmo*deployed-0.25f*deployed,
               my-0.75f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               detailColor,
               verts,
               colrs
               );

         tmo = 1.0f + arrowRad;

         // Proximal Arrow
         definePill(
               mx+tmo*deployed,
               my,
               mx+tmo*deployed+0.25f*deployed,
               my+0.75f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               detailColor,
               verts,
               colrs
               );
         definePill(
               mx+tmo*deployed,
               my,
               mx+tmo*deployed+0.25f*deployed,
               my-0.75f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               detailColor,
               verts,
               colrs
               );

         prevDir = direction;
         prevDep = deployed;
         MenuOpen->buildCache(verts.size()/2, verts, colrs);
      }

      if (  prevDep  != deployed    ||
            prevDir  != direction   ){
         GLuint index = 0;

         GLfloat mx=0.0f,  // Origin of Menu
                 my=0.0f,  // Origin of Menu
                 tmo;     // local coordinate offset

         tmo = 5.75f+(GLfloat)drawIndex;

         // Menu Body
         index = updatePillGeometry(
               mx, 
               my,
               mx + tmo*deployed,
               my,
               1.0,
               circleSegments,
               index,
               MenuOpen->coordCache
               );

         // Element Diamonds
         if (abs(scrollCursor) == 0.0f) {
            GLfloat tms = 1.0f;
            for (int i = 0; i < numListings+1; i++) {
               if (i == numListings)
                  tms = 0.0f;
               index = updateEllipseGeometry(
                     mx + (2.0f + (GLfloat)i*1.75f + (GLfloat)drawIndex)*deployed,
                     my,
                     0.1f*tms,
                     0.1f*tms,
                     2,
                     index,
                     MenuOpen->coordCache
                     );
            }
         } else {
            GLfloat tms = 1.0f;
            GLfloat tmx, tma;
            for (int i = 0; i < numListings+1; i++) {

               if (i == 0) {
                  tma = -3.0f*pow(scrollCursor, 2.0f) + 4.0f*scrollCursor;
                  tms = scrollCursor;
                  tmx = mx + (
                        2.0f + 
                        ((GLfloat)i - tma + 1.0f)*1.75f + 
                        (GLfloat)drawIndex
                        )*deployed;
               } else if (i == numListings) {
                  tms -= abs(scrollCursor);
                  tma = -3.0f*pow(1.0f-scrollCursor, 2.0f) + 4.0f*(1.0f-scrollCursor);
                  tmx = mx + (
                        2.0f + 
                        ((GLfloat)i + tma - 2.0f)*1.75f + 
                        (GLfloat)drawIndex
                        )*deployed;
               } else {
                  tms = 1.0f;
                  tmx = mx + (
                        2.0f + 
                        ((GLfloat)i + scrollCursor - 1.0f)*1.75f + 
                        (GLfloat)drawIndex
                        )*deployed;
               }
               index = updateEllipseGeometry(
                     tmx,
                     my,
                     0.1f*tms,
                     0.1f*tms,
                     2,
                     index,
                     MenuOpen->coordCache
                     );
            }
         }

         tmo = 3.75f;

         // Center Arrow
         index = updatePillGeometry(
               mx+tmo*deployed,
               my+0.85f*pow(deployed, 3.0f),
               mx+tmo*deployed-0.20f*deployed,
               my+1.00f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               index,
               MenuOpen->coordCache
               );

         index = updatePillGeometry(
               mx+tmo*deployed,
               my+0.85f*pow(deployed, 3.0f),
               mx+tmo*deployed+0.20f*deployed,
               my+1.00f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               index,
               MenuOpen->coordCache
               );

         tmo = 6.5f;

         // Distil Arrow
         index = updatePillGeometry(
               mx+tmo*deployed,
               my,
               mx+tmo*deployed-0.25f*deployed,
               my+0.75f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               index,
               MenuOpen->coordCache
               );
         index = updatePillGeometry(
               mx+tmo*deployed,
               my,
               mx+tmo*deployed-0.25f*deployed,
               my-0.75f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               index,
               MenuOpen->coordCache
               );

         tmo = 1.0f + arrowRad;

         // Proximal Arrow
         index = updatePillGeometry(
               mx+tmo*deployed,
               my,
               mx+tmo*deployed+0.25f*deployed,
               my+0.75f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               index,
               MenuOpen->coordCache
               );
         index = updatePillGeometry(
               mx+tmo*deployed,
               my,
               mx+tmo*deployed+0.25f*deployed,
               my-0.75f*pow(deployed, 3.0f),
               arrowRad,
               circleSegments/5,
               index,
               MenuOpen->coordCache
               );

         MenuOpen->updateCoordCache();
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

         // Element Diamonds
         if (abs(scrollCursor) == 0.0f) {
            for (int i = 0; i < numListings+1; i++) {
               index = updateEllipseColor(
                     2,
                     detailColor,
                     index,
                     MenuOpen->colorCache
                     );
            }
         } else {
            for (int i = 0; i < numListings+1; i++) {
               index = updateEllipseColor(
                     2,
                     detailColor,
                     index,
                     MenuOpen->colorCache
                     );
            }
         }

         // Center Arrow
         index = updatePillColor(
               circleSegments/5,
               detailColor,
               index,
               MenuOpen->colorCache
               );

         index = updatePillColor(
               circleSegments/5,
               detailColor,
               index,
               MenuOpen->colorCache
               );

         // Distil Arrow
         index = updatePillColor(
               circleSegments/5,
               detailColor,
               index,
               MenuOpen->colorCache
               );
         index = updatePillColor(
               circleSegments/5,
               detailColor,
               index,
               MenuOpen->colorCache
               );

         // Proximal Arrow
         index = updatePillColor(
               circleSegments/5,
               detailColor,
               index,
               MenuOpen->colorCache
               );
         index = updatePillColor(
               circleSegments/5,
               detailColor,
               index,
               MenuOpen->colorCache
               );

         MenuOpen->updateColorCache();
      }

      MenuOpen->updateMVP(gx, gy, scale, scale, -direction, w2h);
      MenuOpen->draw();
   }

   return;
}
