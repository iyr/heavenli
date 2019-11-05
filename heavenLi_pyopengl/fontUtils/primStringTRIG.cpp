/*
 * Uses character primitive to draw strings with textured quads
 */

using namespace std;
extern textAtlas* quack;

unsigned int defineString(
      float       x,             // x-postion
      float       y,             // y-postion
      std::string inputString,   // Input string to draw
      textAtlas*  atlas,         // texture atlas to draw characters from
      float*      textColor,     // Polygon Color                                         
      std::vector<float> &verts, // Input Vector of x,y coordinates                       
      std::vector<float> &texuv, // Input Vector of u,y texture coordinates
      std::vector<float> &colrs  // Input Vector of r,g,b values                          
      ) {

   int c = 0;
   character* tmg;
   float ax;

   GLuint stringLen = inputString.size();
   const char* inputChars = inputString.c_str();

   for (unsigned int i = 0; i < stringLen; i++) {
      c = inputChars[i];

      tmg = &quack->glyphData[c];

      // Only update non-control characters
      if (c >= 32) {
         ax =  (float)tmg->advanceX*0.015625f; // divide by 64
         x +=  ax;
      }

      // Shift downward and reset x position for line breaks
      if (c == (int)'\n') {
         y -= (float)quack->faceSize;
         x = 0.0f;
      }

      defineChar(
            x-ax, y, 
            c,
            quack,
            textColor, 
            verts, texuv, colrs);
   }

   return verts.size()/2;
}

unsigned int updateString(
      float       x,             // x-postion
      float       y,             // y-postion
      std::string inputString,   // Input string to draw
      textAtlas*  atlas,         // texture atlas to draw characters from
      GLuint      index,         // Index of where to start writing to input arrays
      float*      verts,         // Input Vector of x,y coordinates                       
      float*      texuv          // Input Vector of u,y texture coordinates
      ){

   int c = 0;
   float ax;
   GLuint stringLen = inputString.size();
   const char* inputChars = inputString.c_str();
   character* tmg;

   for (unsigned int i = 0; i < stringLen; i++) {

      //if (i < stringLen){
         c = inputChars[i];

         tmg = &quack->glyphData[c];

         // Only update non-control characters
         if (c >= 32) {
            ax =  (float)tmg->advanceX*0.015625f; // divide by 64
            x +=  ax;
         }

         // Shift downward and reset x position for line breaks
         if (c == (int)'\n') {
            y -= (float)quack->faceSize;
            x = 0.0f;
         }

         index = updateChar(
               x-ax, y, 
               c,
               quack,
               index,
               verts, 
               texuv);
      //}
   }

   return index;
}
