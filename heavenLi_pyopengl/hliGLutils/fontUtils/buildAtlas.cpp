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
//extern textAtlas* quack;
extern std::map<std::string, textAtlas> textFonts;
extern std::string selectedAtlas;

PyObject* buildAtlas_hliGLutils(PyObject* self, PyObject *args) {

   PyObject*   faceName;      // Name of the face we're building
   PyObject*   numCharacters; // Number of Characters to map to atlas
   PyObject*   faceSize;      // pt font size
   PyObject*   GlyphData;     // List of Character python objects that contain glyph metrics and bitmap
   character*  glyphData;     // Array of Character structs that contain glyph metrics and bitmap
   GLuint      numChars,      // Number of characters in arrays
               size;

   // Parse Inputs with error-checking
   if ( !PyArg_ParseTuple(args,
            "OOOO",
            &faceName,
            &numCharacters,
            &faceSize,
            &GlyphData
            ))
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

   // Parse font point size
   size = PyLong_AsLong(faceSize);

   // Fill-in character array
   PyObject* PyChar;
   PyObject* PyAttr;
   PyObject* PyBitmap;
   for (unsigned int c = 0; c < numChars; c++) {
      PyChar = PyList_GetItem(GlyphData, c);

      PyAttr = PyObject_GetAttrString(PyChar, "advanceX");
      glyphData[c].advanceX      = (GLfloat)PyFloat_AsDouble(PyAttr);

      PyAttr = PyObject_GetAttrString(PyChar, "advanceY");
      glyphData[c].advanceY      = (GLfloat)PyFloat_AsDouble(PyAttr);

      PyAttr = PyObject_GetAttrString(PyChar, "bearingX");
      glyphData[c].bearingX      = (GLfloat)PyFloat_AsDouble(PyAttr);//+2.0f;

      PyAttr = PyObject_GetAttrString(PyChar, "bearingY");
      glyphData[c].bearingY      = (GLfloat)PyFloat_AsDouble(PyAttr);//+2.0f;

      PyAttr = PyObject_GetAttrString(PyChar, "bearingTop");
      glyphData[c].bearingTop    = (GLfloat)PyFloat_AsDouble(PyAttr);

      PyAttr = PyObject_GetAttrString(PyChar, "bearingLeft");
      glyphData[c].bearingLeft   = (GLfloat)PyFloat_AsDouble(PyAttr);

      PyBitmap = PyObject_GetAttrString(PyChar, "bitmap");
      unsigned int bufferLength;
      //bufferLength = PyList_Size(PyBitmap);
      bufferLength = glyphData[c].bearingX*glyphData[c].bearingY;
      GLubyte* tmb = new GLubyte[bufferLength*4];

      for (unsigned int i = 0; i < bufferLength; i++) {
         tmb[i*4+0] = 255;
         tmb[i*4+1] = 255;
         tmb[i*4+2] = 255;
         PyAttr = PyList_GetItem(PyBitmap, i);
         tmb[i*4+3] = GLubyte(PyLong_AsLong(PyAttr));
      }
      
      glyphData[c].bitmap = tmb;      
   }

   if (textFonts.empty())
      selectedAtlas = inputString;
   if (textFonts.count(inputString) <= 0)
      textFonts.insert(std::make_pair(inputString, textAtlas()));
   else {
      textFonts.erase(inputString);
      textFonts.insert(std::make_pair(inputString, textAtlas()));
   }

   textAtlas* ta = &textFonts[inputString];
   ta->makeAtlas(inputString, numChars, size, glyphData);

   Py_RETURN_NONE;
}
