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
#include <string>
#include <map>

using namespace std;

PyObject* buildAtlas_hliGLutils(PyObject* self, PyObject *args) {

   PyObject*   faceName;      // Name of the face we're building
   PyObject*   GlyphData;     // List of Character python objects that contain glyph metrics and bitmap
   PyObject*   numCharacters; // Number of Characters to map to atlas
   character*  glyphData;     // Array of Character structs that contain glyph metrics and bitmap
   GLuint      numChars;      // Number of characters in arrays

   // Parse Inputs with error-checking
   if ( !PyArg_ParseTuple(args,
            "OOO",
            &faceName,
            &GlyphData,
            &numCharacters) )
   {
      printf("ERROR::loadChar: failed to parse arguments\n");
      Py_RETURN_NONE;
   }

   // Parse Font-Face name
   const char* inputChars = PyUnicode_AsUTF8(faceName);
   std::string inputString = inputChars;

   // Parse number of characters
   numChars = PyLong_AsLong(numCharacters);

   // Initialize character glyph array
   glyphData = new character[numChars];

   // Fill-in character array
   PyObject* PyChar;
   PyObject* PyAttr;
   PyObject* PyBitmap;
   for (unsigned int c = 0; c < numChars; c++) {
      PyChar = PyList_GetItem(GlyphData, c);

      PyAttr = PyObject_GetAttrString(PyChar, "advanceX");
      glyphData[c].advanceX = float(PyFloat_AsDouble(PyAttr));

      PyAttr = PyObject_GetAttrString(PyChar, "advanceY");
      glyphData[c].advanceY = float(PyFloat_AsDouble(PyAttr));

      PyAttr = PyObject_GetAttrString(PyChar, "bearingX");
      glyphData[c].bearingX = float(PyFloat_AsDouble(PyAttr));

      PyAttr = PyObject_GetAttrString(PyChar, "bearingY");
      glyphData[c].bearingY = float(PyFloat_AsDouble(PyAttr));

      PyAttr = PyObject_GetAttrString(PyChar, "bearingTop");
      glyphData[c].bearingTop = float(PyFloat_AsDouble(PyAttr));

      PyAttr = PyObject_GetAttrString(PyChar, "bearingLeft");
      glyphData[c].bearingLeft = float(PyFloat_AsDouble(PyAttr));

      PyBitmap = PyObject_GetAttrString(PyChar, "bitmap");
      unsigned int bufferLength;
      bufferLength = PyList_Size(PyBitmap);
      GLubyte* tmb = new GLubyte[bufferLength];
      for (unsigned int i = 0; i < bufferLength; i++) {
         PyAttr = PyList_GetItem(PyBitmap, i);
         tmb[i] = GLubyte(PyLong_AsLong(PyAttr));
      }
      
      glyphData[c].bitmap = tmb;      
      glyphData[c].binChar = c;

      // Cleanup
      delete [] tmb;
   }


   /*
   int w = 0;
   int h = 0;

   for (int c = 0; c < numChars; c++) {
      w += Characters[c].sizeX;
      h = std::max(h, Characters[c].sizeY);
   }

   int atlas_width = w;

   GLuint tex;
   glActiveTexture(GL_TEXTURE0);
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glTexImage2D(
         GL_TEXTURE_2D,    // Target, should just remain 'GL_TEXTURE_2D'
         0,                // Mipmap, should remain 0 for now
         GL_ALPHA,         // Internal Format, should remain just ALPHA for now
         w,                // Texture Width
         h,                // Texture Height
         0,                // Border, must remain 0 for ES 2.0 compliance
         GL_ALPHA,         // Must be the same as 'Internal Format' for ES 2.0 compliance
         GL_UNSIGNED_BYTE, // Texture Type
         0                 // Buffer containing texture data
         );

   int x = 0;
   for (int c = 0; c < numChars; c++) {
      glTexSubImage2D(
            GL_TEXTURE_2D,       // Target, should just remain 'GL_TEXTURE_2D'
            0,                   // Mipmap, should remain 0 for now
            x,                   // x offset, x index of the texel to start writing to
            0,                   // y offset, y index of the texel to start writing to
            Characters[c].sizeX, // width of subregion to update
            Characters[c].sizeY, // height of subregion to update
            GL_ALPHA,            // Internal Format, should remain just ALPHA for now
            GL_UNSIGNED_BYTE,    // Texture Type
            Characters[c].bitmap // buffer of pixels
            );

      x += Characters[c].sizeX;
   }
   */

   Py_RETURN_NONE;
}
