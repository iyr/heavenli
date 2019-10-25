#include <algorithm>
#include "characterStruct.h"

#ifndef atlasClass
#define atlasClass 
#define MAXWIDTH 1024

using namespace std;
extern GLint uniform_tex;
GLint uniform_color;

class textAtlas {
   public:
      std::string faceName;
      GLuint tex;

      GLuint textureWidth;
      GLuint textureHeight;

      GLuint faceSize;
      
      character* glyphData = NULL;

      textAtlas(std::string faceName, GLuint numChars, GLuint size, character* glyphData);
      ~textAtlas(void);
};

textAtlas::textAtlas(std::string faceName, GLuint numChars, GLuint size, character* glyphData) {
   this->faceName       = faceName;
   this->glyphData      = glyphData;
   this->faceSize       = size;

   unsigned int roww    = 0;
   unsigned int rowh    = 0;

   this->textureWidth   = 0;
   this->textureHeight  = 0;

   printf("Determining Atlas dimensions...\n");
   for (unsigned int i = 0; i < numChars; i++) {
      if (roww + this->glyphData[i].bearingX + 1 >= MAXWIDTH) {
         this->textureWidth   = roww > this->textureWidth ? roww : this->textureWidth;
         this->textureHeight += rowh;
         roww = 0;
         rowh = 0;
      }

      roww += (GLint)this->glyphData[i].bearingX + 1;
      rowh = rowh >= (GLuint)this->glyphData[i].bearingY 
         ? rowh 
         : (GLint)this->glyphData[i].bearingY;
   }

   this->textureWidth   = roww > this->textureWidth ? roww : this->textureWidth;
   this->textureHeight += rowh;

   printf("Generating Atlas Texture...\n");
   glActiveTexture(GL_TEXTURE0);
   glGenTextures(1, &this->tex);
   printf("Atlas texture id: %x\n", this->tex);
   glBindTexture(GL_TEXTURE_2D, this->tex);
   glUniform1i(uniform_tex, 0);

   glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, this->textureWidth, this->textureHeight, 0, GL_ALPHA, GL_UNSIGNED_BYTE, NULL);

   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

   std::string prevString;
   GLint ox = 0;
   GLint oy = 0;

   rowh = 0;

   for (unsigned int i = 0; i < numChars; i++) {
      if (ox + (GLsizei)this->glyphData[i].bearingX + 1 >= MAXWIDTH) {
         oy += rowh;
         rowh = 0;
         ox = 0;
      }

      /*
      printf("%c:\n", i);
      for (unsigned int j = 0; j < (unsigned int)(this->glyphData[i].bearingX*this->glyphData[i].bearingY); j++) {
         printf("%.3d", this->glyphData[i].bitmap[j]);
         if ((j+1) % this->glyphData[i].bearingX == 0 )
            printf("\n");
      }
      printf("\n");
      */

      glTexSubImage2D(
            GL_TEXTURE_2D, 
            0, 
            ox, 
            oy, 
            (GLsizei)this->glyphData[i].bearingX, 
            (GLsizei)this->glyphData[i].bearingY, 
            GL_ALPHA, 
            GL_UNSIGNED_BYTE, 
            this->glyphData[i].bitmap);

      this->glyphData[i].textureOffsetX = (float)ox / (float)this->textureWidth;
      this->glyphData[i].textureOffsetY = (float)oy / (float)this->textureHeight;

      ox    += (GLint)this->glyphData[i].bearingX + 1;
      rowh   = rowh >= (GLuint)this->glyphData[i].bearingY 
                     ? rowh 
                     : (GLint)this->glyphData[i].bearingY;

   }

   fprintf(stderr, "Generated a %d x %d (%d kb) texture atlas\n", this->textureWidth, this->textureHeight, (this->textureWidth * this->textureHeight) /1024);

   glBindTexture(GL_TEXTURE_2D, 0);
   return;
};

textAtlas::~textAtlas() {
   delete [] glyphData;
   glDeleteTextures(1, &this->tex);

   return;
};
#endif
