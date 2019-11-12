//#include <math.h>

using namespace std;
extern textAtlas* quack;

bool        firstRun = true;
std::string prevString;
drawCall    textLine;
drawCall    textBackdrop;
GLuint      prevStringLen;

// Python 3 function for drawing text
PyObject* drawText_hliGLutils(PyObject* self, PyObject* args) {
   PyObject* textColorPyTup;
   PyObject* faceColorPyTup;
   PyObject* PyString;

   GLfloat gx, gy, sx, sy, w2h, alignment, ao=0.0f;
   GLfloat textColor[4];
   GLfloat faceColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "OffffffOO",
            &PyString,
            &alignment,
            &gx, &gy,
            &sx, &sy,
            &w2h,
            &textColorPyTup,
            &faceColorPyTup) )
   {
      Py_RETURN_NONE;
   }

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
         alignment,
         gx, gy,
         sx, sy,
         w2h,
         quack,
         textColor,
         faceColor);

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
      GLfloat     alignment,     // 0.0=left, 0.5=center, 1.0=right
      GLfloat     gx,            // X position
      GLfloat     gy,            // Y position
      GLfloat     sx,            // X scale
      GLfloat     sy,            // Y scale
      GLfloat     w2h,           // width to height ration
      textAtlas*  atlas,         // texture atlas to draw characters from
      GLfloat*    textColor,     // color of text
      GLfloat*    faceColor      // color of backdrop
      ){

   if (firstRun) {
      firstRun = false;
      textLine.setDrawType(GL_TRIANGLES);
      textLine.setNumColors(1);
      textBackdrop.setNumColors(1);
      printf("DRAWTEXT FIRST RUN\n");
   }

   GLuint stringLen = inputString.size();

   textLine.setColorQuartet(0, textColor);
   textBackdrop.setColorQuartet(0, faceColor);

   if (  textLine.numVerts       == 0        ||
         textBackdrop.numVerts   == 0        ||
         prevStringLen           < stringLen ){
      std::vector <GLfloat> verts;
      std::vector <GLfloat> colrs;
      std::vector <GLfloat> texuv;

      std::vector <GLfloat> bgverts;
      std::vector <GLfloat> bgcolrs;

      defineString(
            0.0f, 0.0f,
            inputString,
            alignment,
            quack,
            textColor,
            verts, texuv, colrs);

      GLfloat minX = NULL, minY = NULL, maxX = NULL, maxY = NULL;

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

      prevString     = inputString;
      prevStringLen  = stringLen;
      textLine.texID = quack->tex;
      textLine.buildCache(verts.size()/2, verts, texuv, colrs);
      textBackdrop.buildCache(bgverts.size()/2, bgverts, bgcolrs);
   }

   if (  prevString.compare(inputString) != 0 ){

      //static GLfloat minX = NULL, minY = NULL, maxX = NULL, maxY = NULL;
      GLfloat minX = NULL, minY = NULL, maxX = NULL, maxY = NULL;
      const char* inputChars = inputString.c_str();

      for (unsigned int i = stringLen*6; i < prevStringLen*6; i++){
         textLine.coordCache[i*2+0] = 0.0f;
         textLine.coordCache[i*2+1] = 0.0f;
         textLine.texuvCache[i*2+0] = 0.0f;
         textLine.texuvCache[i*2+1] = 0.0f;
      }

      GLuint index = 0;

      index = updateString(
            0.0f, 0.0f,
            inputString,
            alignment,
            atlas,
            index,
            textLine.coordCache,
            textLine.texuvCache);

      index = 0;

      for (unsigned int i = 0; i < stringLen*6; i++){

         // only update extrema for characters with metrics
         if (inputChars[i/6] > 32) {
            if (textLine.coordCache[i*2] < minX)
               minX = textLine.coordCache[i*2];
            if (textLine.coordCache[i*2] > maxX)
               maxX = textLine.coordCache[i*2];
            if (textLine.coordCache[i*2+1] < minY)
               minY = textLine.coordCache[i*2+1];
            if (textLine.coordCache[i*2+1] > maxY)
               maxY = textLine.coordCache[i*2+1];
         }
      }

      index = updateRoundRect(
            minX*1.05f, maxY*1.05f,
            maxX*1.05f, minY*1.05f,
            10.5f,
            15,
            index,
            textBackdrop.coordCache);

      textLine.updateTexUVCache();
      textLine.updateCoordCache();
      textBackdrop.updateCoordCache();
      prevString = inputString;
   }

   if (  textLine.colorsChanged     ||
         textBackdrop.colorsChanged ){

      GLuint index = 0;
      for (unsigned int i = 0; i < prevStringLen; i++)
         index = updateQuadColor(textColor, index, textLine.colorCache);

      index = 0;
      index = updateRoundRect(15, faceColor, index, textBackdrop.colorCache);

      textLine.updateColorCache();
      textBackdrop.updateColorCache();
   }

   //if (w2h <= 1.0) {
      //textLine.updateMVP(gx, gy, sx*0.007f/w2h, sy*0.007f/w2h, 0.0f, w2h);
      //textBackdrop.updateMVP(gx, gy, sx*0.007f/w2h, sy*0.007f/w2h, 0.0f, w2h);
   //} else {
      //textLine.updateMVP(gx, gy, sx*0.007f*w2h, sy*0.007f*w2h, 0.0f, w2h);
      //textBackdrop.updateMVP(gx, gy, sx*0.007f*w2h, sy*0.007f*w2h, 0.0f, w2h);
   //}
   textLine.updateMVP(gx, gy, sx*0.007f, sy*0.007f, 0.0f, w2h);
   textBackdrop.updateMVP(gx, gy, sx*0.007f, sy*0.007f, 0.0f, w2h);

   textBackdrop.draw();
   textLine.draw();
}
