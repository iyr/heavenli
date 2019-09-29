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
#include <map>
//#include "characterClass.h"

using namespace std;
//extern std::map<uint8_t, Character> Characters;
//extern Character* Characters;// = new Character[128];

PyObject* loadChar_hliGLutils(PyObject* self, PyObject *args) {
   PyObject *Pylist;
   PyObject *Pystring;
   GLubyte* buffer;
   GLuint   sizeX, sizeY, bearingX, bearingY, advanceX, advanceY;

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "OiiiiiiO",
            &Pystring,
            &sizeX, &sizeY,
            &bearingX, &bearingY,
            &advanceX, &advanceY,
            &Pylist ) )
   {
      printf("ERROR::loadChar: failed to parse arguments\n");
      Py_RETURN_NONE;
   }

   // Determine which character we're building
   const char* inputChars = PyUnicode_AsUTF8(Pystring);
   uint8_t character = inputChars[0];

   // Get length of texture buffer
   GLuint bufferLength;
   bufferLength = PyList_Size(Pylist);
   
   // Allocate temporary buffer
   buffer = new GLubyte[bufferLength];

   // Copy contents of Pylist to temporary buffer
   PyObject* Pylong;
   for (unsigned int i = 0; i < bufferLength; i++){
      Pylong      = PyList_GetItem(Pylist, i);
      buffer[i]   = byte(PyLong_AsLong(Pylong));
   }

   // Create Character Object
   //Character ch;

   // Assign object parameters
   /*
   ch.setSizeX(sizeX);
   ch.setSizeY(sizeY);
   ch.setBearingX(bearingX);
   ch.setBearingY(bearingY);
   ch.setAdvanceX(advanceX);
   ch.setAdvanceY(advanceY);
   ch.setOffset();
   ch.setBitmap(buffer, bufferLength);
   */
   
   //Characters[character] = ch;
   //Characters.insert(std::pair<uint8_t, Character>(character, ch));

   // Cleanup
   delete [] buffer;
   //delete ch;

   Py_RETURN_NONE;
}
