//#include <math.h>

using namespace std;

extern std::map<std::string, drawCall> drawCalls;
//extern std::map<std::string, textAtlas*> textFonts;

extern textAtlas* quack;

// Python 3 function for drawing text
PyObject* drawText_hliGLutils(PyObject* self, PyObject* args) {
   PyObject* textColorPyTup;
   PyObject* faceColorPyTup;
   PyObject* PyString;

   GLfloat gx, gy, sx, sy, w2h, horiAlignment, vertAlignment;
   GLfloat textColor[4];
   GLfloat faceColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "OfffffffOO",
            &PyString,
            &horiAlignment,
            &vertAlignment,
            &gx, &gy,
            &sx, &sy,
            &w2h,
            &textColorPyTup,
            &faceColorPyTup) )
   {
      Py_RETURN_NONE;
   }

   if (drawCalls.count("textLine") <= 0){
      drawCalls.insert(std::make_pair("textLine", drawCall()));
   }
   drawCall* textLine = &drawCalls["textLine"];

   if (drawCalls.count("textBackdrop") <= 0){
      drawCalls.insert(std::make_pair("textBackdrop", drawCall()));
   }
   drawCall* textBackdrop = &drawCalls["textBackdrop"];

   const char* inputChars  = PyUnicode_AsUTF8(PyString);
   std::string inputString = inputChars;
   textColor[0]   = float(PyFloat_AsDouble(PyTuple_GetItem(textColorPyTup, 0)));
   textColor[1]   = float(PyFloat_AsDouble(PyTuple_GetItem(textColorPyTup, 1)));
   textColor[2]   = float(PyFloat_AsDouble(PyTuple_GetItem(textColorPyTup, 2)));
   textColor[3]   = float(PyFloat_AsDouble(PyTuple_GetItem(textColorPyTup, 3)));

   faceColor[0]   = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 0)));
   faceColor[1]   = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 1)));
   faceColor[2]   = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 2)));
   faceColor[3]   = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 3)));

   drawText(
         inputString,
         horiAlignment,
         vertAlignment, // 0.0=bottom, 0.5=center, 1.0=top
         gx, gy,
         sx, sy,
         w2h,
         quack,
         textColor,
         faceColor,
         textLine,
         textBackdrop
         );

   Py_RETURN_NONE;
}

   // Draw whole texture atlas
   /*
   verts.push_back(-100.0f); verts.push_back(-10.0f); // Top-Left
   verts.push_back(-100.0f); verts.push_back(  5.0f); // Bottom-Left
   verts.push_back( 100.0f); verts.push_back(-10.0f); // Top-Right

   verts.push_back( 100.0f); verts.push_back(-10.0f); // Top-Right
   verts.push_back(-100.0f); verts.push_back(  5.0f); // Bottom-Left
   verts.push_back( 100.0f); verts.push_back(  5.0f); // Bottom-Right

   texuv.push_back(0.0f);  texuv.push_back(1.0f);  // Top-Left
   texuv.push_back(0.0f);  texuv.push_back(0.0f);  // Bottom-Left
   texuv.push_back(1.0f);  texuv.push_back(1.0f);  // Top-Right

   texuv.push_back(1.0f);  texuv.push_back(1.0f);  // Top-Right
   texuv.push_back(0.0f);  texuv.push_back(0.0f);  // Bottom-Left
   texuv.push_back(1.0f);  texuv.push_back(0.0f);  // Bottom-Right

   colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);
   colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);
   colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);

   colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);
   colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);
   colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);
   */

void drawText(
      std::string inputString,   // string of text draw
      GLfloat     horiAlignment, // 0.0=left, 0.5=center, 1.0=right
      GLfloat     vertAlignment, // 0.0=bottom, 0.5=center, 1.0=top
      GLfloat     gx,            // X position
      GLfloat     gy,            // Y position
      GLfloat     sx,            // X scale
      GLfloat     sy,            // Y scale
      GLfloat     w2h,           // width to height ration
      textAtlas*  atlas,         // texture atlas to draw characters from
      GLfloat*    textColor,     // color of text
      GLfloat*    faceColor,     // color of backdrop
      drawCall*   textLine,      // pointer to input drawCall to write text
      drawCall*   textBackdrop   // pointer to input drawCall to write text backdrop
      ){

   static GLfloat prevHoriAlignment,
                  prevVertAlignment;
   GLfloat ao=0.0f;
   textLine->setDrawType(GL_TRIANGLES);
   textLine->setNumColors(1);
   textBackdrop->setNumColors(1);

   GLuint stringLen = inputString.size();

   textLine->setColorQuartet(0, textColor);
   textBackdrop->setColorQuartet(0, faceColor);

   if (  textLine->numVerts      == 0        ||
         textBackdrop->numVerts  == 0        ||
         textLine->numVerts/6    < stringLen ){
      std::vector <GLfloat> verts;
      std::vector <GLfloat> colrs;
      std::vector <GLfloat> texuv;

      std::vector <GLfloat> bgverts;
      std::vector <GLfloat> bgcolrs;

      defineString(
            0.0f, 0.0f,
            inputString,
            horiAlignment,
            vertAlignment,
            quack,
            textColor,
            verts, texuv, colrs);

      GLfloat minX = (GLfloat)NULL, minY = (GLfloat)NULL, maxX = (GLfloat)NULL, maxY = (GLfloat)NULL;

      for (unsigned int i = 0; i < stringLen*6; i++){

         // only update extrema for characters with metrics
         if (inputString[i/6] > 32) {
            if (verts[i*2] < minX)
               minX = verts[i*2];
            if (verts[i*2] > maxX)
               maxX = verts[i*2];
            if (verts[i*2+1] < minY)
               minY = verts[i*2+1];
            if (verts[i*2+1] > maxY)
               maxY = verts[i*2+1];
         }
      }

      defineRoundRect(
            minX*1.05f, maxY*1.05f,
            maxX*1.05f, minY*1.05f,
            10.5f,
            15,
            faceColor,
            bgverts,
            bgcolrs);

      prevHoriAlignment = horiAlignment;
      prevVertAlignment = vertAlignment;
      textLine->text    = inputString;
      textLine->texID   = quack->tex;
      textLine->buildCache(verts.size()/2, verts, texuv, colrs);
      textBackdrop->buildCache(bgverts.size()/2, bgverts, bgcolrs);
   }

   if (  textLine->text.compare(inputString) != 0  ||
         prevVertAlignment != vertAlignment        ||
         prevHoriAlignment != horiAlignment        ){

      GLfloat minX = (GLfloat)NULL, minY = (GLfloat)NULL, maxX = (GLfloat)NULL, maxY = (GLfloat)NULL;
      const char* inputChars = inputString.c_str();

      for (unsigned int i = stringLen*6; i < textLine->text.size()*6; i++){
         textLine->coordCache[i*2+0] = 0.0f;
         textLine->coordCache[i*2+1] = 0.0f;
         textLine->texuvCache[i*2+0] = 0.0f;
         textLine->texuvCache[i*2+1] = 0.0f;
      }

      GLuint index = 0;

      index = updateString(
            0.0f, 0.0f,
            inputString,
            horiAlignment,
            vertAlignment,
            atlas,
            index,
            textLine->coordCache,
            textLine->texuvCache);

      index = 0;

      for (unsigned int i = 0; i < textLine->text.size(); i++)
         index = updateQuadColor(textColor, index, textLine->colorCache);
      textLine->updateColorCache();

      for (unsigned int i = 0; i < stringLen*6; i++){

         // only update extrema for characters with metrics
         if (inputChars[i/6] > 32) {
            if (textLine->coordCache[i*2] < minX)
               minX = textLine->coordCache[i*2];
            if (textLine->coordCache[i*2] > maxX)
               maxX = textLine->coordCache[i*2];
            if (textLine->coordCache[i*2+1] < minY)
               minY = textLine->coordCache[i*2+1];
            if (textLine->coordCache[i*2+1] > maxY)
               maxY = textLine->coordCache[i*2+1];
         }
      }

      index = 0;
      index = updateRoundRect(
            minX*1.05f, maxY*1.05f,
            maxX*1.05f, minY*1.05f,
            10.5f,
            15,
            index,
            textBackdrop->coordCache);

      prevHoriAlignment = horiAlignment;
      prevVertAlignment = vertAlignment;
      textLine->updateTexUVCache();
      textLine->updateCoordCache();
      textBackdrop->updateCoordCache();
      textLine->text = inputString;
   }

   if (  textLine->colorsChanged     ||
         textBackdrop->colorsChanged ){

      GLuint index = 0;
      for (unsigned int i = 0; i < textLine->text.size(); i++)
         index = updateQuadColor(textColor, index, textLine->colorCache);

      index = 0;
      index = updateRoundRect(15, faceColor, index, textBackdrop->colorCache);

      textLine->updateColorCache();
      textBackdrop->updateColorCache();
   }

   textLine->updateMVP(gx, gy, sx*0.007f, sy*0.007f, ao, w2h);
   textBackdrop->updateMVP(gx, gy, sx*0.007f, sy*0.007f, ao, w2h);

   textBackdrop->draw();
   textLine->draw();

   return;
}

void drawText(
      std::string inputString,   // string of text draw
      GLfloat     horiAlignment,     // 0.0=left, 0.5=center, 1.0=right
      GLfloat     vertAlignment, // 0.0=bottom, 0.5=center, 1.0=top
      GLfloat     gx,            // X position
      GLfloat     gy,            // Y position
      GLfloat     sx,            // X scale
      GLfloat     sy,            // Y scale
      GLfloat     w2h,           // width to height ration
      textAtlas*  atlas,         // texture atlas to draw characters from
      GLfloat*    textColor,     // color of text
      GLfloat*    faceColor,     // color of backdrop
      drawCall*   textLine       // pointer to input drawCall to write text
      ){

   GLfloat ao=0.0f;
   textLine->setDrawType(GL_TRIANGLES);
   textLine->setNumColors(1);

   static GLfloat prevHoriAlignment = -1.0f,
                  prevVertAlignment = -1.0f;
   static GLuint  prevStringLen  =  0;

   GLuint stringLen = inputString.size();

   textLine->setColorQuartet(0, textColor);

   if (  textLine->numVerts   == 0        ||
         textLine->numVerts/6 < stringLen ||
         prevStringLen        < stringLen ){

      std::vector <GLfloat> verts;
      std::vector <GLfloat> colrs;
      std::vector <GLfloat> texuv;

      //std::vector <GLfloat> bgverts;
      //std::vector <GLfloat> bgcolrs;

      defineString(
            0.0f, 0.0f,
            inputString,
            horiAlignment,
            vertAlignment,
            quack,
            textColor,
            verts, texuv, colrs);

      prevStringLen     = stringLen;
      prevHoriAlignment = horiAlignment;
      prevVertAlignment = vertAlignment;
      textLine->text    = inputString;
      textLine->texID   = quack->tex;
      textLine->buildCache(verts.size()/2, verts, texuv, colrs);
   }

   if (  textLine->text.compare(inputString) != 0  ||
         prevVertAlignment != vertAlignment        ||
         prevHoriAlignment != horiAlignment        ){

      for (unsigned int i = stringLen*6; i < textLine->text.size()*6; i++){
         textLine->coordCache[i*2+0] = 0.0f;
         textLine->coordCache[i*2+1] = 0.0f;
         textLine->texuvCache[i*2+0] = 0.0f;
         textLine->texuvCache[i*2+1] = 0.0f;
      }

      GLuint index = 0;

      index = updateString(
            0.0f, 0.0f,
            inputString,
            horiAlignment,
            vertAlignment,
            atlas,
            index,
            textLine->coordCache,
            textLine->texuvCache);

      index = 0;
      prevHoriAlignment = horiAlignment;
      prevVertAlignment = vertAlignment;

      textLine->updateTexUVCache();
      textLine->updateCoordCache();
      textLine->text = inputString;
      for (unsigned int i = 0; i < textLine->text.size(); i++)
         index = updateQuadColor(textColor, index, textLine->colorCache);

      textLine->updateColorCache();
   }

   if (textLine->colorsChanged){

      GLuint index = 0;
      for (unsigned int i = 0; i < textLine->text.size(); i++)
         index = updateQuadColor(textColor, index, textLine->colorCache);

      textLine->updateColorCache();
   }

   textLine->updateMVP(gx, gy, sx*0.007f, sy*0.007f, ao, w2h);
   textLine->draw();

   return;
}
