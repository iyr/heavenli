#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
   // These undefs necessary because microsoft
   #undef near
   #undef far
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#include <math.h>
#include <vector>
#include <string>
using namespace std;

GLfloat*    stringVertexBuffer   =  NULL;
GLfloat*    stringColourBuffer   =  NULL;
GLuint*     stringIndices        =  NULL;
GLuint      stringVerts;
std::string      prevString;

PyObject* printText_drawButtons(PyObject* self, PyObject *args) {
   PyObject *colourPyTup;
   PyObject *Pystring;

   float posX, posY, scale, w2h;
   float textColor[3];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "ffffOO",
            &posX, &posY,
            &scale,
            &w2h,
            &Pystring,
            &colourPyTup) )
   {
      Py_RETURN_NONE;
   }

   const char* inputChars = PyUnicode_AsUTF8(Pystring);
   std::string inputString = inputChars;

   textColor[0]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 0)));
   textColor[1]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 1)));
   textColor[2]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 2)));

   // (Re)allocate and Define Geometry/Color buffers
   if (  stringVertexBuffer   == NULL  ||
         stringColourBuffer   == NULL  ||
         stringIndices        == NULL  ||
         prevString           != inputString){

      //printf("Input String: %s\n", inputString.c_str());

      //printf("Initializing Geometry for Confirm Button\n");
      vector<float> verts;
      vector<float> colrs;
      char circleSegments  = 20;
      float lineWidth      = 0.0f;
      float charSpacing    = 0.5f;
      //float charThickness  = 0.2f;
      //float charThickness  = 0.15f;
      //float charThickness  = 0.125f;
      float charThickness  = 0.04f;
      float charScale      = 1.0f;
      stringVerts          = 0;
      //printf("stringSize: %i\n", inputString.size());

      for (unsigned int i = 0; i < inputString.size(); i++) {
         drawChar(
               inputChars[i],
               lineWidth, 0.0f,  // X, Y coordinates 
               charScale,        // Scale 
               charThickness,    // Thickness/boldness 
               charSpacing,      // Amount of empty space after character 
               circleSegments, 
               textColor, 
               &lineWidth, 
               verts, 
               colrs);
      }

      stringVerts = verts.size()/2;
      //printf("stringVerts: %i\n", stringVerts);
      //printf("lineWidth: %f\n", lineWidth);

      // Pack Vertics and Colors into global array buffers
      if (stringVertexBuffer == NULL) {
         stringVertexBuffer = new GLfloat[stringVerts*2];
      } else {
         delete [] stringVertexBuffer;
         stringVertexBuffer = new GLfloat[stringVerts*2];
      }

      if (stringColourBuffer == NULL) {
         stringColourBuffer = new GLfloat[stringVerts*3];
      } else {
         delete [] stringColourBuffer;
         stringColourBuffer = new GLfloat[stringVerts*3];
      }

      if (stringIndices == NULL) {
         stringIndices = new GLuint[stringVerts];
      } else {
         delete [] stringIndices;
         stringIndices = new GLuint[stringVerts];
      }

      for (unsigned int i = 0; i < stringVerts; i++) {
         stringVertexBuffer[i*2+0]  = verts[i*2+0];
         stringVertexBuffer[i*2+1]  = verts[i*2+1];
         stringIndices[i]           = i;
         stringColourBuffer[i*3+0]  = colrs[i*3+0];
         stringColourBuffer[i*3+1]  = colrs[i*3+1];
         stringColourBuffer[i*3+2]  = colrs[i*3+2];
      }

      prevString = inputString;
   }
   glPushMatrix();
   glTranslatef(posX*w2h, posY, 0.0f);
   if (w2h <= 1.0) {
         scale = scale*w2h;
   }

   glScalef(scale, scale, 1);
   glColorPointer(3, GL_FLOAT, 0, stringColourBuffer);
   glVertexPointer(2, GL_FLOAT, 0, stringVertexBuffer);
   glDrawElements( GL_TRIANGLES, stringVerts, GL_UNSIGNED_INT, stringIndices);
   glPopMatrix();

   Py_RETURN_NONE;
}
