using namespace std;

extern std::map<std::string, drawCall> drawCalls;
extern std::map<std::string, textAtlas> textFonts;
extern std::string selectedAtlas;
extern VertexAttributeStrings VAS;

PyObject* drawMenu_hliGLutils(PyObject* self, PyObject* args) {
   PyObject*   faceColorPyTup;
   //PyObject*   extraColorPyTup;
   PyObject*   detailColorPyTup;
   PyObject*   py_list;
   PyObject*   py_tuple;
   PyObject*   Params;

   GLfloat     gx, gy, scale, w2h, deployed, direction, floatingIndex, scrollCursor;
   GLuint      numElements, menuLayout, numListings, selectedElement = 0;
   GLboolean   drawIndex, selectFromScroll;
   GLfloat     faceColor[4];
   GLfloat     detailColor[4];
   GLfloat*    elementCoords = NULL;

   // Parse Inputs
   if ( !PyArg_ParseTuple(args, "O", &Params ) )
   {
      Py_RETURN_NONE;
   }

   gx             = (GLfloat)PyFloat_AsDouble(PyDict_GetItem(Params, PyUnicode_FromString("gx")));
   gy             = (GLfloat)PyFloat_AsDouble(PyDict_GetItem(Params, PyUnicode_FromString("gy")));
   scale          = (GLfloat)PyFloat_AsDouble(PyDict_GetItem(Params, PyUnicode_FromString("scale")));
   direction      = (GLfloat)PyFloat_AsDouble(PyDict_GetItem(Params, PyUnicode_FromString("direction")));
   deployed       = (GLfloat)PyFloat_AsDouble(PyDict_GetItem(Params, PyUnicode_FromString("deployed")));
   floatingIndex  = (GLfloat)PyFloat_AsDouble(PyDict_GetItem(Params, PyUnicode_FromString("floatingIndex")));
   scrollCursor   = (GLfloat)PyFloat_AsDouble(PyDict_GetItem(Params, PyUnicode_FromString("scrollCursor")));
   selectedElement= (GLuint)PyLong_AsLong(PyDict_GetItem(Params, PyUnicode_FromString("selectedElement")));
   numElements    = (GLuint)PyLong_AsLong(PyDict_GetItem(Params, PyUnicode_FromString("numElements")));
   numListings    = (GLuint)PyLong_AsLong(PyDict_GetItem(Params, PyUnicode_FromString("numListings")));
   menuLayout     = (GLuint)PyLong_AsLong(PyDict_GetItem(Params, PyUnicode_FromString("menuLayout")));
   drawIndex      = (GLboolean)PyLong_AsLong(PyDict_GetItem(Params, PyUnicode_FromString("drawIndex")));
   w2h            = (GLfloat)PyFloat_AsDouble(PyDict_GetItem(Params, PyUnicode_FromString("w2h")));
   selectFromScroll  = (GLboolean)PyLong_AsLong(PyDict_GetItem(Params, PyUnicode_FromString("selectFromScroll")));
   faceColorPyTup    = PyDict_GetItem(Params, PyUnicode_FromString("faceColor"));
   detailColorPyTup  = PyDict_GetItem(Params, PyUnicode_FromString("detailColor"));

   faceColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 0));
   faceColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 1));
   faceColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 2));
   faceColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 3));

   detailColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0));
   detailColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1));
   detailColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2));
   detailColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 3))*deployed;

   if (drawCalls.count("MenuIndex") <= 0)
      drawCalls.insert(std::make_pair("MenuIndex", drawCall()));
   drawCall* MenuIndex     = &drawCalls["MenuIndex"];
   MenuIndex->setShader("RGBAcolor_Atexture");

   if (drawCalls.count("MenuOverflow") <= 0)
      drawCalls.insert(std::make_pair("MenuOverflow", drawCall()));
   drawCall* MenuOverflow  = &drawCalls["MenuOverflow"];
   MenuOverflow->setShader("RGBAcolor_NoTexture");

   if (drawCalls.count("MenuNormal") <= 0)
      drawCalls.insert(std::make_pair("MenuNormal", drawCall()));
   drawCall* MenuNormal    = &drawCalls["MenuNormal"];
   MenuNormal->setShader("RGBAcolor_NoTexture");

   if (drawCalls.count("MenuClosed") <= 0)
      drawCalls.insert(std::make_pair("MenuClosed", drawCall()));
   drawCall* MenuClosed    = &drawCalls["MenuClosed"];
   MenuClosed->setShader("RGBAcolor_NoTexture");

   elementCoords = new GLfloat[3*(numListings+1)];

   drawMenu(
         gx, gy,           // Menu Position
         scale,            // Menu Size
         direction,        // Direction, in degrees about the unit circle, the menu slides out to
         deployed,         // 0.0=closed, 1.0=completely open
         floatingIndex,    // index of the selected element, used for scroll bar
         scrollCursor,     // animation cursor for element motion during scrolling (-1.0 to 1.0)
         numElements,      // number of elements
         menuLayout,       // 0=carousel w/ rollover, 1=linear strip w/ terminals, 2=value slider w/ min/max
         numListings,      // number of listings to display at once
         selectedElement,  // Index of the current selected element
         drawIndex,        // whether or not to draw the index over the number of elements
         selectFromScroll, // whether or not elements are selected by scrolling to them
         elementCoords,    // Relative coordinate of menu elements
         w2h,              // width to height ratio
         faceColor,        // Main color for the body of the menu
         detailColor,      // scroll bar, 
         MenuIndex,        // drawCall object for drawing a numeric index of elements
         MenuOverflow,     // drawCall object for drawing the menu open
         MenuNormal,       // drawCall object for drawing the menu open
         MenuClosed        // drawCall object for drawing the menu closed
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

void drawMenu(
      GLfloat     gx, 
      GLfloat     gy,               // Menu Position
      GLfloat     scale,            // Menu Size
      GLfloat     direction,        // Direction, in degrees about the unit circle, the menu slides out to
      GLfloat     deployed,         // 0.0=closed, 1.0=completely open
      GLfloat     floatingIndex,    // index of the selected element, used for scroll bar
      GLfloat     scrollCursor,     // animation cursor for element motion during scrolling (-1.0 to 1.0)
      GLuint      numElements,      // number of elements
      GLuint      menuLayout,       // 0=carousel w/ rollover, 1=linear strip w/ terminals, 2=value slider w/ min/max
      GLuint      numListings,      // number of elements to display at once
      GLuint      selectedElement,  // Index of the current selected element
      GLboolean   drawIndex,        // whether or not to draw the index over the number of elements
      GLboolean   selectFromScroll, // whether or not elements are selected by scrolling to them
      GLfloat*    elementCoords,    // Relative coordinates of Menu elements
      GLfloat     w2h,              // width to height ratio
      GLfloat*    faceColor,        // Main color for the body of the menu
      GLfloat*    detailColor,      // scroll bar, 
      drawCall*   MenuIndex,        // drawCall object for drawing menu index
      drawCall*   MenuOverflow,     // drawCall object for drawing the menu open
      drawCall*   MenuNormal,       // drawCall object for drawing the menu open
      drawCall*   MenuClosed        // drawCall object for drawing the menu closed
      ){

   textAtlas* tmAt = &textFonts[selectedAtlas];

   static GLfloat prevDep,
                  prevDir,
                  prevScr,
                  prevFlc;

   static GLuint  prevNumListings   = 3,
                  prevNumElements   = -1,
                  prevMenuLayout    = -1;
   GLuint         circleSegments    = 60;
   static GLboolean prevIndexDraw   = false,
                    prevSelectFromScroll = false;
   bool  overFlow    = false;

   if (numListings < numElements) {
      overFlow = true;
   }

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
         map<string, attribCache> attributeData;
         attributeData[VAS.coordData] = attribCache(VAS.coordData, 2, 0, 0);
         attributeData[VAS.colorData] = attribCache(VAS.colorData, 4, 2, 1);
         attributeData[VAS.coordData].writeCache(verts.data(), verts.size());
         attributeData[VAS.colorData].writeCache(colrs.data(), colrs.size());
         MenuClosed->buildCache(verts.size()/2, attributeData);
      }

      if (  MenuClosed->colorsChanged  ){
         GLuint index = 0;
         index = updateEllipseColor(
               circleSegments,
               faceColor,
               index,
               (GLfloat *)MenuClosed->getAttribCache(VAS.colorData)
               );

         MenuClosed->updateBuffer(VAS.colorData);
      }

      // Set element diamond coordinates/sizes
      for (unsigned int i = 0; i < numListings+1; i++) {
         elementCoords[i*3+0] = 0.0f;
         elementCoords[i*3+1] = 0.0f;
         elementCoords[i*3+2] = 0.0f;
      }
      MenuClosed->updateMVP(gx, gy, scale, scale, direction, w2h);
      MenuClosed->draw();
   } 
   // Draw Menu Body w/ just elements
   else if (
         numElements < 4
         ){
      MenuNormal->setNumColors(2);
      MenuNormal->setColorQuartet(0, faceColor);
      MenuNormal->setColorQuartet(1, detailColor);

      if (  MenuNormal->numVerts == 0        ||
            prevNumElements != numElements   ){
         vector<GLfloat> verts;
         vector<GLfloat> colrs;

         defineMenuNormal(
            direction,        // Direction, in degrees, the menu slides out to
            deployed,         // 0.0=closed, 1.0=completely open
            floatingIndex,    // index of the selected element, used for scroll bar
            scrollCursor,     // animation cursor for element motion during scrolling: -1.0 to 1.0
            numElements,      // number of elements
            circleSegments,
            selectFromScroll,
            elementCoords,    // Relative coordinates of Menu elements
            w2h,              // width to height ratio
            faceColor,        // Main color for the body of the menu
            detailColor,      // scroll bar, 
            verts,            // Input Vector of x,y coordinates
            colrs             // Input Vector of r,g,b values
            );

         prevDir = direction;
         prevDep = deployed;
         prevScr = scrollCursor;
         prevFlc = floatingIndex;
         prevNumElements = numElements;
         map<string, attribCache> attributeData;
         attributeData[VAS.coordData] = attribCache(VAS.coordData, 2, 0, 0);
         attributeData[VAS.colorData] = attribCache(VAS.colorData, 4, 2, 1);
         attributeData[VAS.coordData].writeCache(verts.data(), verts.size());
         attributeData[VAS.colorData].writeCache(colrs.data(), colrs.size());
         MenuNormal->buildCache(verts.size()/2, attributeData);
      }

      if (  prevDep        != deployed       ||
            prevScr        != scrollCursor   ||
            prevFlc        != floatingIndex  ||
            prevDir        != direction      ||
            prevSelectFromScroll != selectFromScroll  ){
         GLuint index = 0;
         index = updateMenuNormalGeometry(
               direction,     // Direction, in degrees, the menu slides out to
               deployed,      // 0.0=closed, 1.0=completely open
               floatingIndex, // index of the selected element, used for scroll bar
               scrollCursor,  // element animation cursor for scrolling: -1.0 to 1.0
               numElements,   // number of elements
               circleSegments,// number of polygon segments
               selectFromScroll,
               elementCoords, // Relative coordinates of Menu elements
               w2h,           // width to height ratio
               index,
               (GLfloat *)MenuNormal->getAttribCache(VAS.coordData)
         );

         prevDep = deployed;
         prevDir = direction;
         prevScr = scrollCursor;
         prevFlc = floatingIndex;
         prevSelectFromScroll = selectFromScroll;
         MenuNormal->updateBuffer(VAS.coordData);
      } else {
		   float* glCoords = new float[(numElements+1)*3];
		   defineElementCoords(
		      direction,
		      deployed,
		      floatingIndex,
		      scrollCursor,
		      numElements,
		      1,
		      numElements,
		      selectFromScroll,
		      glCoords,
		      elementCoords
		      );
		      }

      if (  MenuNormal->colorsChanged ){
         GLuint index = 0;
         index = updateMenuNormalColors(
            numElements,   // number of listings to display
            circleSegments,
            faceColor,     // Main color for the body of the menu
            detailColor,   // scroll bar, 
            index,
            (GLfloat *)MenuNormal->getAttribCache(VAS.colorData)
            );
         MenuNormal->updateBuffer(VAS.colorData);
      }

      MenuNormal->updateMVP(gx, gy, scale, -scale, 180.0f+direction, w2h);
      MenuNormal->draw();
   }
   // Draw Menu Body w/ elements+scrollbar when open
   else if (   numElements > 3            ||
               numElements > numListings  ){

      MenuOverflow->setNumColors(2);
      MenuOverflow->setColorQuartet(0, faceColor);
      MenuOverflow->setColorQuartet(1, detailColor);

      if (  MenuOverflow->numVerts == 0      ||
            prevNumListings != numListings   ){

         vector<GLfloat> verts;
         vector<GLfloat> colrs;

         defineMenuOverflow(
            direction,        // Direction, in degrees, the menu slides out to
            deployed,         // 0.0=closed, 1.0=completely open
            floatingIndex,    // index of the selected element, used for scroll bar
            scrollCursor,     // animation cursor for element motion during scrolling: -1.0 to 1.0
            numElements,      // number of elements
            menuLayout,       // 0=carousel w/ rollover, 1=terminated linear strip
            circleSegments,
            numListings,      // number of listings to display
            drawIndex,        // whether or not to draw the index over the number of elements
            selectFromScroll,
            elementCoords,    // Relative coordinates of Menu elements
            w2h,              // width to height ratio
            faceColor,        // Main color for the body of the menu
            detailColor,      // scroll bar, 
            verts,            // Input Vector of x,y coordinates
            colrs             // Input Vector of r,g,b values
            );

         prevDir = direction;
         prevDep = deployed;
         prevScr = scrollCursor;
         prevFlc = floatingIndex;
         prevNumElements   = numElements;
         prevMenuLayout    = menuLayout;
         prevNumListings   = numListings;
         prevIndexDraw     = drawIndex;
         map<string, attribCache> attributeData;
         attributeData[VAS.coordData] = attribCache(VAS.coordData, 2, 0, 0);
         attributeData[VAS.colorData] = attribCache(VAS.colorData, 4, 2, 1);
         attributeData[VAS.coordData].writeCache(verts.data(), verts.size());
         attributeData[VAS.colorData].writeCache(colrs.data(), colrs.size());
         MenuOverflow->buildCache(verts.size()/2, attributeData);
      }

      if (  prevDep        != deployed       ||
            prevScr        != scrollCursor   ||
            prevFlc        != floatingIndex  ||
            prevDir        != direction      ||
            prevIndexDraw  != drawIndex      ||
            prevMenuLayout != menuLayout     ||
            prevSelectFromScroll != selectFromScroll  ){
         GLuint index = 0;
         index = updateMenuOverflowGeometry(
               direction,        // Direction, in degrees, the menu slides out to
               deployed,         // 0.0=closed, 1.0=completely open
               floatingIndex,    // index of the selected element, used for scroll bar
               scrollCursor,     // element animation cursor for scrolling: -1.0 to 1.0
               numElements,      // number of elements
               menuLayout,       // 0=carousel w/ rollover, 1=terminated linear strip
               circleSegments,   // number of polygon segments
               numListings,      // number of listings to display
               drawIndex,        // whether or not to draw the index over the number of elements
               selectFromScroll, // whether or not elements are selected by scrolling to them
               elementCoords,    // Relative coordinates of Menu elements
               w2h,              // width to height ratio
               index,
               (GLfloat *)MenuOverflow->getAttribCache(VAS.coordData)
         );

         prevDep = deployed;
         prevDir = direction;
         prevScr = scrollCursor;
         prevFlc = floatingIndex;
         prevMenuLayout       = menuLayout;
         prevSelectFromScroll = selectFromScroll;
         MenuOverflow->updateBuffer(VAS.coordData);
      } else {
         float* tml = new float[(numListings+1)*3];
         defineElementCoords(
            direction,
            deployed,
            floatingIndex,
            scrollCursor,
            numElements,
            menuLayout,
            numListings,
            selectFromScroll,
            tml,
            elementCoords
            );
      }

      if (  MenuOverflow->colorsChanged   ||
            prevIndexDraw != drawIndex    ){
         GLuint index = 0;
         index = updateMenuOverflowColors(
            drawIndex,
            circleSegments,
            numListings,   // number of listings to display
            faceColor,     // Main color for the body of the menu
            detailColor,   // scroll bar, 
            index,
            (GLfloat *)MenuOverflow->getAttribCache(VAS.colorData)
            );
         prevIndexDraw = drawIndex;
         MenuOverflow->updateBuffer(VAS.colorData);
      }

      MenuOverflow->updateMVP(gx, gy, scale, -scale, 180.0f+direction, w2h);
      MenuOverflow->draw();
   }

   // Draw text objects indicating floatingIndex, numElements
   if (  drawIndex         && 
         deployed > 0.0f   ){
      float mirror = -1.0f;
      if (  (direction <= 135.0f && direction >= -1.0f)
            ||
            (direction <= 361.0f && direction >= 315.0f)
         )
      {
         mirror = -1.0f;
      } else {
         mirror = 1.0f;
      }
      float tma = (float)degToRad(180.0f+direction),
            textDPIscalar = 32.0f/(float)tmAt->faceSize,
            tmx = scale*(6.75f*cos(tma)+0.5f*sin(-tma)*mirror)*deployed,                         // text location, X
            tmy = scale*(6.75f*sin(tma)+0.5f*cos(-tma)*mirror+0.25f*(float)overFlow)*deployed;   // text location, Y
      if (w2h < 1.0f)
         tmy *= w2h;
      else
         tmx /= w2h;
      int tme = constrain(numElements, 0, 999);

      std::string inputString = std::to_string(1+selectedElement);
      drawText(
            inputString,               // string to render
            float(-cosf(tma)/2.0f+0.5f),                      // Horizontal Alignment
            float(-sinf(tma)/2.0f+1.0f),                      // Vertical Alignment
            gx+tmx, gy+tmy,            // Text position
            3.0f*scale*textDPIscalar, 
            3.0f*scale*textDPIscalar,  // Text scale
            w2h,
            tmAt,                      // Text atlas
            detailColor,
            faceColor,
            MenuIndex                  // drawCall to write to
            );

      tmx = scale*(6.75f*cos(tma)-0.5f*sin(-tma)*mirror)*deployed;                         // text location, X
      tmy = scale*(6.75f*sin(tma)-0.5f*cos(-tma)*mirror+0.25f*(float)overFlow)*deployed;   // text location, Y
      if (w2h < 1.0f)
         tmy *= w2h;
      else
         tmx /= w2h;
      inputString = std::to_string(tme);
      drawText(
            inputString,               // string to render
            float(-cosf(tma)/2.0f+0.5f),                      // Horizontal Alignment
            float(-sinf(tma)/2.0f+1.0f),                      // Vertical Alignment
            gx+tmx, gy+tmy,            // Text position
            3.0f*scale*textDPIscalar, 
            3.0f*scale*textDPIscalar,  // Text scale
            w2h,
            tmAt,                      // Text atlas
            faceColor,
            detailColor,
            MenuIndex                  // drawCall to write to
            );
   }
   return;
}
