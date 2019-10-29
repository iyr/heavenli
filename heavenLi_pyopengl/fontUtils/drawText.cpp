#include <math.h>

using namespace std;
extern textAtlas* quack;

bool        firstRun = true;
std::string prevString;
drawCall    textLine;
GLuint      prevStringLen;

PyObject* drawText_hliGLutils(PyObject* self, PyObject *args) {
   PyObject *colourPyTup;
   PyObject *Pystring;

   GLfloat gx, gy, sx, sy, w2h, ao=0.0f;
   GLfloat textColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "OfffffO",
            &Pystring,
            &gx, &gy,
            &sx, &sy,
            &w2h,
            &colourPyTup) )
   {
      Py_RETURN_NONE;
   }

   const char* inputChars = PyUnicode_AsUTF8(Pystring);
   std::string inputString = inputChars;

   GLuint stringLen = inputString.size();

   textColor[0]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 0)));
   textColor[1]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 1)));
   textColor[2]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 2)));
   textColor[3]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 3)));

   if (firstRun) {
      firstRun = false;
      textLine.setDrawType(GL_TRIANGLES);
      /*
      for (unsigned int i = 0; i < 128; i++)
         //printf("glyph %c: width (bearingX): %12.0f, rows (bearingY): %12.0f, bearingLeft: %12.0f, bearingTop: %12.0f\n", 
         printf("glyph %c: width (bearingX): %12d, rows (bearingY): %12d, bearingLeft: %12d, bearingTop: %12d, texOffsetX: %0.5f, texOffsetY: %0.5f\n", 
               i, 
               (GLint)quack->glyphData[i].bearingX,
               (GLint)quack->glyphData[i].bearingY,
               (GLint)quack->glyphData[i].bearingLeft,
               (GLint)quack->glyphData[i].bearingTop,
               quack->glyphData[i].textureOffsetX,
               quack->glyphData[i].textureOffsetY
               );
      printf("DRAWTEXT FIRST RUN\n");
      */
   }

   if (  textLine.numVerts == 0              ||
         stringLen > prevStringLen           ){
      std::vector <GLfloat> verts;
      std::vector <GLfloat> colrs;
      std::vector <GLfloat> texuv;

      int c = 0;
      character* tmg;
      float x = 0.0f,
            y = 0.0f,
            ax;

      for (unsigned int i = 0; i < stringLen; i++) {
         c = inputChars[i];

         tmg = &quack->glyphData[c];

         // Only update non-control characters
         if (c >= 32) {
            ax =  (float)tmg->advanceX*0.015625f; // divide by 64
            x +=  ax;
         }

         // Shift downward and reset x position for line breaks
         if (c == (int)'\n') {
            y -= (float)quack->faceSize;
            x = 0.0f;
         }

         defineChar(
               x-ax, y, 
               c,
               quack,
               textColor, 
               verts, texuv, colrs);
      }

      prevString     = inputString;
      prevStringLen  = stringLen;
      textLine.texID = quack->tex;
      textLine.buildCache(verts.size()/2, verts, texuv, colrs);
   }

   if (  prevString.compare(inputString) != 0 ){
      int c = 0;
      character* tmg;
      float x = 0.0f,
            y = 0.0f,
            ax;

      for (unsigned int i = stringLen*6; i < prevStringLen*6; i++){
         textLine.coordCache[i*2+0] = 0.0f;
         textLine.coordCache[i*2+1] = 0.0f;
         textLine.texuvCache[i*2+0] = 0.0f;
         textLine.texuvCache[i*2+1] = 0.0f;
      }
      GLuint index = 0;
      for (unsigned int i = 0; i < prevStringLen; i++) {

         if (i < stringLen){
            c = inputChars[i];

            tmg = &quack->glyphData[c];

            // Only update non-control characters
            if (c >= 32) {
               ax =  (float)tmg->advanceX*0.015625f; // divide by 64
               x +=  ax;
            }

            // Shift downward and reset x position for line breaks
            if (c == (int)'\n') {
               y -= (float)quack->faceSize;
               x = 0.0f;
            }

            index = updateChar(
                  x-ax, y, 
                  c,
                  quack,
                  index,
                  textLine.coordCache, 
                  textLine.texuvCache);
         }
      }

      textLine.updateTexUVCache();
      textLine.updateCoordCache();
      prevString = inputString;
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

   if (w2h <= 1.0)
      textLine.updateMVP(gx, gy, sx*0.007f/w2h, sy*0.007f/w2h, 0.0f, w2h);
   else
      textLine.updateMVP(gx, gy, sx*0.007f*w2h, sy*0.007f*w2h, 0.0f, w2h);

   textLine.draw();

   Py_RETURN_NONE;
}
