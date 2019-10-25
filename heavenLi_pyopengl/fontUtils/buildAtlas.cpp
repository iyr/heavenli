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
extern textAtlas* quack;

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
      glyphData[c].bearingX      = (GLfloat)PyFloat_AsDouble(PyAttr);

      PyAttr = PyObject_GetAttrString(PyChar, "bearingY");
      glyphData[c].bearingY      = (GLfloat)PyFloat_AsDouble(PyAttr);

      PyAttr = PyObject_GetAttrString(PyChar, "bearingTop");
      glyphData[c].bearingTop    = (GLfloat)PyFloat_AsDouble(PyAttr);

      PyAttr = PyObject_GetAttrString(PyChar, "bearingLeft");
      glyphData[c].bearingLeft   = (GLfloat)PyFloat_AsDouble(PyAttr);

      PyBitmap = PyObject_GetAttrString(PyChar, "bitmap");
      unsigned int bufferLength;
      bufferLength = PyList_Size(PyBitmap);
      GLubyte* tmb = new GLubyte[bufferLength];

      for (unsigned int i = 0; i < bufferLength; i++) {
         PyAttr = PyList_GetItem(PyBitmap, i);
         tmb[i] = GLubyte(PyLong_AsLong(PyAttr));

         //printf("%.3d", tmb[i]);
         //if ((i+1) % glyphData[c].bearingX == 0 )
            //printf("\n");
      }
      //printf("\n");

      /*
      printf("%c: bufferLength: %.4d, width: %.3d, rows: %.3d\n",
            c+32, 
            bufferLength,
            glyphData[c].bearingX,
            glyphData[c].bearingY
            );
            */
      
      glyphData[c].bitmap = tmb;      
   }

   quack = new textAtlas("Barlow-Regular", numChars, size, glyphData);

   Py_RETURN_NONE;
}
