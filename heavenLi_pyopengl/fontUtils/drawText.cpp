#include <math.h>

using namespace std;
extern textAtlas* quack;
extern GLint uniform_tex;

GLuint      vbo;
bool        firstRun = true;
std::string prevString;
drawCall    textLine;

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

   //printf(inputChars);
   //printf("\n");

   textColor[0]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 0)));
   textColor[1]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 1)));
   textColor[2]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 2)));
   textColor[3]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 3)));

   if (firstRun) {
      glGenBuffers(1, &vbo);
      firstRun = false;
      textLine.setDrawType(GL_TRIANGLES);
      /*
      for (unsigned int i = 0; i < 128-32; i++)
         //printf("glyph %c: width (bearingX): %12.0f, rows (bearingY): %12.0f, bearingLeft: %12.0f, bearingTop: %12.0f\n", 
         printf("glyph %c: width (bearingX): %12d, rows (bearingY): %12d, bearingLeft: %12d, bearingTop: %12d, texOffsetX: %0.5f, texOffsetY: %0.5f\n", 
               i+32, 
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

   glBindTexture(GL_TEXTURE_2D, quack->tex);
   glUniform1i(uniform_tex, 0);

   //glEnableVertexAttribArray(0);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);

   std::vector <GLfloat> verts;
   std::vector <GLfloat> colrs;
   std::vector <GLfloat> texuv;

   int c = 0;
   character tmg;
   float x = 0.0f,
         y = 0.0f,
         x2,
         y2,
         w,
         h;

   for (unsigned int i = 0; i < inputString.size(); i++) {
      c = inputChars[i]-32;

      tmg = quack->glyphData[c];

      x2 =   x + tmg.bearingLeft;
      y2 =  -y - tmg.bearingTop;
      w  =  tmg.bearingX;
      h  =  tmg.bearingY;
      x +=  tmg.advanceX*0.015625f;

      //x += ((float)tmg.advanceX / 64.0f);
      //y += ((float)tmg].advanceY / 64.0f);

      // Skip glyphs with no pixels
      if (!w || !h) {
         continue;
      }

      verts.push_back( x2);
      verts.push_back(-y2);

      verts.push_back( x2 + w);
      verts.push_back(-y2);

      verts.push_back( x2);
      verts.push_back(-y2 - h);

      verts.push_back( x2 + w);
      verts.push_back(-y2);

      verts.push_back( x2);
      verts.push_back(-y2 - h);

      verts.push_back( x2 + w);
      verts.push_back(-y2 - h);

      texuv.push_back(tmg.textureOffsetX);
      texuv.push_back(tmg.textureOffsetY);

      texuv.push_back(tmg.textureOffsetX + tmg.bearingX / quack->textureWidth);
      texuv.push_back(tmg.textureOffsetY);

      texuv.push_back(tmg.textureOffsetX);
      texuv.push_back(tmg.textureOffsetY + tmg.bearingY / quack->textureHeight);

      texuv.push_back(tmg.textureOffsetX + tmg.bearingX / quack->textureWidth);
      texuv.push_back(tmg.textureOffsetY);

      texuv.push_back(tmg.textureOffsetX);
      texuv.push_back(tmg.textureOffsetY + tmg.bearingY / quack->textureHeight);

      texuv.push_back(tmg.textureOffsetX + tmg.bearingX / quack->textureWidth);
      texuv.push_back(tmg.textureOffsetY + tmg.bearingY / quack->textureHeight);

      colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);
      colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);
      colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);
      colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);
      colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);
      colrs.push_back(textColor[0]);   colrs.push_back(textColor[1]);   colrs.push_back(textColor[2]);   colrs.push_back(textColor[3]);
      /*
      printf("glyph %c: width (bearingX): %4.0f, rows (bearingY): %4.0f, bearingLeft: %4.0f, bearingTop: %4.0f, texOffsetX: %1.5f, texOffsetY: %1.5f\n", 
            c, 
            (float)tmg.bearingX,
            (float)tmg.bearingY,
            (float)tmg.bearingLeft,
            (float)tmg.bearingTop,
            (float)tmg.textureOffsetX,
            (float)tmg.textureOffsetY
            );
            */

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

   textLine.texID = quack->tex;
   textLine.buildCache(verts.size()/2, verts, texuv, colrs);
   textLine.updateMVP(gx, gy, sx*0.01f, sy*0.01f, 0.0f, w2h);
   textLine.draw();

   Py_RETURN_NONE;
}
