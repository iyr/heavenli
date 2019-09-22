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
   float advanceX;
   float advanceY;

   float bearingX;
   float bearingY;

   float bearingTop;
   float bearingLeft;

   float textureOffsetX;
   float textureOffsetY;
   GLubyte* bitmap = NULL;
   GLuint   binChar;
};

/*
class Character {
   public: 
      GLfloat  TextureID;  // Texture ID of glyph
      GLfloat  sizeX;      // Size of glyph (width)
      GLfloat  sizeY;      // Size of glyph (rows)
      GLfloat  bearingX;   // Offset from baseline to left of glyph
      GLfloat  bearingY;   // Offset form baseline to top of glyph
      GLfloat  advanceX;   // Spatial offset for next glyph
      GLfloat  advanceY;   // Spatial offset for next glyph
      GLfloat  texOffset;  // Horizontal offset in texture coords
      GLubyte* bitmap;
      Character(void);
      ~Character(void);
      void setSizeX(GLfloat size);
      void setSizeY(GLfloat size);
      void setBearingX(GLfloat bearing);
      void setBearingY(GLfloat bearing);
      void setAdvanceX(GLfloat advance);
      void setAdvanceY(GLfloat advance);
      void setBitmap(GLubyte* bitmap, GLuint bufferLength);
      void setOffset(GLfloat offset);
};

Character::Character(void) {
   //printf("Creating Character...\n");
}

Character::~Character(void) {
   delete [] bitmap;
}

void Character::setSizeX(GLfloat size) {
   this->sizeX = size;
   return;
}

void Character::setSizeY(GLfloat size) {
   this->sizeY = size;
   return;
}

void Character::setBearingX(GLfloat bearing) {
   this->bearingX = bearing;
   return;
}

void Character::setBearingY(GLfloat bearing) {
   this->bearingY = bearing;
   return;
}

void Character::setAdvanceX(GLfloat advance) {
   this->advanceX = advance;
   return;
}

void Character::setAdvanceY(GLfloat advance) {
   this->advanceY = advance;
   return;
}

void Character::setOffset(GLfloat offset) {
   this->texOffset = offset;
   return;
}

void Character::setBitmap(GLubyte* bitmap, GLuint bufferLength) {
   this->bitmap = new GLubyte[bufferLength];
   for (unsigned int i = 0; i < bufferLength; i++) {
      this->bitmap[i] = bitmap[i];
   }
   return;
}

*/
#endif
