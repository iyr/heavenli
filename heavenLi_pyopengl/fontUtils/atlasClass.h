#include <algorithm>
#include "characterStruct.h"

#ifndef atlasClass
#define atlasClass 
#define MAXWIDTH 1024

using namespace std;
GLint uniform_tex;
GLint uniform_color;

class textAtlas {
   public:
      std::string faceName;
      GLuint tex;

      unsigned int textureWidth;
      unsigned int textureHeight;
      
      character* glyphData = NULL;

      textAtlas(std::string faceName, GLuint numChars, character* glyphData);
      ~textAtlas(void);
};

textAtlas::textAtlas(std::string faceName, GLuint numChars, character* glyphData) {
   this->faceName = faceName;
   this->glyphData = glyphData;

   unsigned int roww = 0;
   unsigned int rowh = 0;

   this->textureWidth = 0;
   this->textureHeight = 0;

   for (unsigned int i = 0; i < numChars; i++) {
      if (roww + this->glyphData[i].bearingX + i >= MAXWIDTH) {
         this->textureWidth = roww > (unsigned int)this->textureWidth ? roww : (unsigned int)this-textureWidth;
         this->textureHeight += rowh;
         roww = 0;
         rowh = 0;
      }

      roww += int(this->glyphData[i].bearingX + i);
      rowh = rowh > (unsigned int)this->glyphData[i].bearingY ? rowh : (unsigned int)this->glyphData[i].bearingY;
   }

   this->textureWidth = roww > (unsigned int)this->textureWidth ? roww : (unsigned int)this-textureWidth;
   this->textureHeight += rowh;

   glActiveTexture(GL_TEXTURE0);
   glGenTextures(1, &this->tex);
   glBindTexture(GL_TEXTURE_2D, this->tex);
   glUniform1i(uniform_tex, 0);

   glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, this->textureWidth, this->textureHeight, 0, GL_ALPHA, GL_UNSIGNED_BYTE, 0);

   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

   int ox = 0;
   int oy = 0;

   rowh = 0;

   for (unsigned int i = 0; i < numChars; i++) {
      if (ox + this->glyphData[i].bearingX + 1 >= MAXWIDTH) {
         oy += rowh;
         rowh = 0;
         ox = 0;
      }

      glTexSubImage2D(
            GL_TEXTURE_2D, 
            0, 
            ox, 
            oy, 
            (unsigned int)this->glyphData[i].bearingX, 
            (unsigned int)this->glyphData[i].bearingY, 
            GL_ALPHA, 
            GL_UNSIGNED_BYTE, 
            this->glyphData[i].bitmap);

      this->glyphData[i].textureOffsetX = ox / float(this->textureWidth);
      this->glyphData[i].textureOffsetY = oy / float(this->textureHeight);

      rowh = rowh > (unsigned int)this->glyphData[i].bearingY ? rowh : (unsigned int)this->glyphData[i].bearingY;
      ox += int(this->glyphData[i].bearingX + 1);
   }

   fprintf(stderr, "Generated a %d x %d (%d kb) texture atlas\n", this->textureWidth, this->textureHeight, this->textureWidth + this->textureHeight /1024);
};

textAtlas::~textAtlas() {
   delete [] glyphData;
   glDeleteTextures(1, &this->tex);
};
#endif
