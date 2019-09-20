#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
   // These undefs necessary because microsoft
   #undef near
   #undef far
#endif
#include <GL/gl.h>
#include <GL/glext.h>

#ifndef characterClass 
#define characterClass

using namespace std;

class Character {
   public: 
      GLuint   TextureID;  // Texture ID of glyph
      GLuint   sizeX;      // Size of glyph (width)
      GLuint   sizeY;      // Size of glyph (rows)
      GLuint   bearingX;   // Offset from baseline to left of glyph
      GLuint   bearingY;   // Offset form baseline to top of glyph
      GLuint   advance;    // Spatial offset for next glyph
      GLubyte* bitmap;
      Character(void);
      ~Character(void);
      void setSizeX(GLuint size);
      void setSizeY(GLuint size);
      void setBearingX(GLuint bearing);
      void setBearingY(GLuint bearing);
      void setAdvance(GLuint advance);
      void setBitmap(GLubyte* bitmap, GLuint bufferLength);
};

Character::Character(void) {
   //printf("Creating Character...\n");
}

Character::~Character(void) {
   delete [] bitmap;
}

void Character::setSizeX(GLuint size) {
   this->sizeX = size;
   return;
}

void Character::setSizeY(GLuint size) {
   this->sizeY = size;
   return;
}

void Character::setBearingX(GLuint bearing) {
   this->bearingX = bearing;
   return;
}

void Character::setBearingY(GLuint bearing) {
   this->bearingY = bearing;
   return;
}

void Character::setAdvance(GLuint advance) {
   this->advance = advance;
   return;
}

void Character::setBitmap(GLubyte* bitmap, GLuint bufferLength) {
   this->bitmap = new GLubyte[bufferLength];
   for (unsigned int i = 0; i < bufferLength; i++) {
      this->bitmap[i] = bitmap[i];
   }
   return;
}

#endif
