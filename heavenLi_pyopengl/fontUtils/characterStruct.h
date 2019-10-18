#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
   // These undefs necessary because microsoft
   #undef near
   #undef far
#endif
#include <GL/gl.h>
#include <GL/glext.h>

#ifndef characterStruct
#define characterStruct

using namespace std;
struct character {
   GLfloat  advanceX;      // ax ~ linearHoriAdvance
   GLfloat  advanceY;      // ay ~ linearVertAdvance

   GLfloat  bearingX;      // bw ~ bitmap Width
   GLfloat  bearingY;      // bh ~ bitmap Height

   GLfloat  bearingTop;    // bt ~ bitmap_top
   GLfloat  bearingLeft;   // bl ~ bitmap_left

   GLfloat  textureOffsetX;// tx
   GLfloat  textureOffsetY;// ty

   GLubyte* bitmap = NULL;
};

#endif
