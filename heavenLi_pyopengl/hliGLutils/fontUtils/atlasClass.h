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

      //textAtlas(std::string faceName, GLuint numChars, GLuint size, character* glyphData);
      textAtlas(void);
      ~textAtlas(void);

      void makeAtlas(std::string faceName, GLuint numChars, GLuint size, character* glyphData);
};

void textAtlas::makeAtlas(std::string faceName, GLuint numChars, GLuint size, character* glyphData){
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

   //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, this->textureWidth, this->textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, this->textureWidth, this->textureHeight, 0, GL_ALPHA, GL_UNSIGNED_BYTE, NULL);

   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
   //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

   std::string prevString;
   GLint xOffset = 0;
   GLint yOffset = 0;

   rowh = 0;

   for (unsigned int i = 0; i < numChars; i++) {
      if (xOffset + (GLsizei)this->glyphData[i].bearingX + 1 >= MAXWIDTH) {
         yOffset += rowh;
         rowh = 0;
         xOffset = 0;
      }

      // Convoluted print statement
      /*
      printf("%c:\n", i);
      for (unsigned int j = 0; j < (unsigned int)(this->glyphData[i].bearingX*this->glyphData[i].bearingY); j++) {
         printf("%.3d", this->glyphData[i].bitmap[j]);
         if ((j+1) % this->glyphData[i].bearingX == 0 )
            printf("\n");
      }
      printf("\n");
      glTexSubImage2D(
            GL_TEXTURE_2D, 
            0, 
            xOffset, 
            yOffset, 
            (GLsizei)this->glyphData[i].bearingX, 
            (GLsizei)this->glyphData[i].bearingY, 
            GL_RGBA, 
            GL_UNSIGNED_BYTE, 
            this->glyphData[i].bitmap);
      */
      glTexSubImage2D(
            GL_TEXTURE_2D, 
            0, 
            xOffset, 
            yOffset, 
            (GLsizei)this->glyphData[i].bearingX, 
            (GLsizei)this->glyphData[i].bearingY, 
            GL_ALPHA, 
            GL_UNSIGNED_BYTE, 
            this->glyphData[i].bitmap);

      this->glyphData[i].textureOffsetX = (float)xOffset / (float)this->textureWidth;
      this->glyphData[i].textureOffsetY = (float)yOffset / (float)this->textureHeight;

      xOffset    += (GLint)this->glyphData[i].bearingX + 1;
      rowh   = rowh >= (GLuint)this->glyphData[i].bearingY 
                     ? rowh 
                     : (GLint)this->glyphData[i].bearingY;

   }

   // Convoluted print statement
      /*
   for (unsigned int i = 0; i < numChars; i++)
      printf("glyph %c (%4d): advanceX: %12.5f -+- width (bearingX): %3d -+- rows (bearingY): %3d -+- bearingLeft: %3d -+- bearingTop: %3d -+- texOffsetX: %0.5f -+- texOffsetY: %0.5f\n", 
            i,
            i, 
            this->glyphData[i].advanceX,
            (GLint)this->glyphData[i].bearingX,
            (GLint)this->glyphData[i].bearingY,
            (GLint)this->glyphData[i].bearingLeft,
            (GLint)this->glyphData[i].bearingTop,
            this->glyphData[i].textureOffsetX,
            this->glyphData[i].textureOffsetY
            );
            */

   fprintf(stderr, "Generated a %d x %d (%d kb) texture atlas\n", this->textureWidth, this->textureHeight, (this->textureWidth * this->textureHeight) /1024);

   glBindTexture(GL_TEXTURE_2D, 0);
   return;
};

textAtlas::textAtlas(void) {
   this->faceName       = "none";
   this->glyphData      = NULL;
   this->faceSize       = 0;
   this->textureWidth   = 0;
   this->textureHeight  = 0;
   this->tex            = (unsigned int)NULL;

   return;
};

textAtlas::~textAtlas() {
   if (  (this->faceSize || this->textureWidth || this->textureHeight) )
      printf("Deleting texture atlas \"%s\" (gl tex id: %x) with glyph size: %i, resolution: %i x %i.\n", this->faceName.c_str(), this->tex, this->faceSize, this->textureWidth, this->textureHeight);
   delete [] glyphData;

   //printf("Deleting Texture Atlas id: %x\n", this->tex);
   glDeleteTextures(1, &this->tex);

   return;
};
#endif
