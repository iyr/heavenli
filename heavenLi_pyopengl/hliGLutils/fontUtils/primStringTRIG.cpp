/*
 * Uses character primitive to draw strings with textured quads
 */

using namespace std;
extern textAtlas* quack;

unsigned int defineString(
      float       x,             // x-postion
      float       y,             // y-postion
      std::string inputString,   // Input string to draw
      float       horiAlignment, // 0.0=left, 0.5=center, 1.0=right
      float       vertAlignment, // 0.0=bottom, 0.5=center, 1.0=top
      textAtlas*  atlas,         // texture atlas to draw characters from
      float*      textColor,     // Polygon Color                                         
      std::vector<float> &verts, // Input Vector of x,y coordinates                       
      std::vector<float> &texuv, // Input Vector of u,y texture coordinates
      std::vector<float> &colrs  // Input Vector of r,g,b values                          
      ) {

   int   c           = 0, 
         lastLineVert= 0;
   float ax          = 0.0f, 
         lineLength  = 0.0f,
         lineHeight  = 0.0f;
   GLuint stringLen  = inputString.size();
   const char* inputChars = inputString.c_str();
   character* tmg;

   for (unsigned int i = 0; i < stringLen; i++) {
      c = inputChars[i];

      tmg = &atlas->glyphData[c];

      // Only update non-control characters
      if (c >= 32) {
         ax =  (float)tmg->advanceX*0.015625f; // divide by 64
         x +=  ax;
      }

      defineChar(
            x-ax, y, 
            c,
            atlas,
            textColor, 
            verts, texuv, colrs);

      // Shift downward and reset x position for line breaks
      if (  c == (int)'\n'    || 
            i == stringLen-1  ){

         if (i != stringLen-1){
            y -= (float)atlas->faceSize;
            x = 0.0f;
         }

         // Get the length of the line (furthest quad vert on left to furthest on right)
         // Also get the height of the line
         for (unsigned int j = lastLineVert; j < (i+1)*6; j++) {
            if(verts[j*2+0] > lineLength)
               lineLength = verts[j*2+0];
            if(verts[j*2+1] > lineHeight)
               lineHeight = verts[j*2+1];
         }

         // Shift the line left by horizontal alignment coefficient
         // 0.0 = left-aligned
         // 0.5 = center-aligned
         // 1.0 = right-aligned
         for (unsigned int j = lastLineVert; j < (i+1)*6; j++) {
            verts[j*2+0] -= horiAlignment*lineLength;
            verts[j*2+1] -= vertAlignment*lineHeight;
         }

         // Reset linelength
         lineLength = 0.0f;

         // set the index of the last line
         lastLineVert = i*6;
      }
   }

   return verts.size()/2;
}

unsigned int updateString(
      float       x,             // x-postion
      float       y,             // y-postion
      std::string inputString,   // Input string to draw
      float       horiAlignment, // 0.0=left, 0.5=center, 1.0=right
      float       vertAlignment, // 0.0=bottom, 0.5=center, 1.0=top
      textAtlas*  atlas,         // texture atlas to draw characters from
      GLuint      index,         // Index of where to start writing to input arrays
      float*      verts,         // Input Vector of x,y coordinates                       
      float*      texuv          // Input Vector of u,y texture coordinates
      ){

   int   c           = 0, 
         lastLineVert= 0;
   float ax          = 0.0f, 
         lineLength  = 0.0f,
         lineHeight  = 0.0f;
   GLuint stringLen  = inputString.size();
   const char* inputChars = inputString.c_str();
   character* tmg;

   for (unsigned int i = 0; i < stringLen; i++) {
      c = inputChars[i];
      tmg = &atlas->glyphData[c];

      // Only update non-control characters
      if (c >= 32) {
         ax =  (float)tmg->advanceX*0.015625f; // divide by 64
         x +=  ax;
      }

      index = updateChar(
            x-ax, y, 
            c,
            atlas,
            index,
            verts, 
            texuv);

      // Shift downward and reset x position for line breaks
      if (  c == (int)'\n'    || 
            i == stringLen-1  ){

         if (i != stringLen-1){
            y -= (float)atlas->faceSize;
            x = 0.0f;
         }

         // Get the length of the line (furthest quad vert on left to furthest on right)
         // Also get the height of the line
         for (unsigned int j = lastLineVert; j < (i+1)*6; j++) {
            if(verts[j*2+0] > lineLength)
               lineLength = verts[j*2+0];
            if(verts[j*2+1] > lineHeight)
               lineHeight = verts[j*2+1];
         }

         // Shift the line left by horizontal alignment coefficient
         // 0.0 = left-aligned
         // 0.5 = center-aligned
         // 1.0 = right-aligned
         for (unsigned int j = lastLineVert; j < (i+1)*6; j++) {
            verts[j*2+0] -= horiAlignment*lineLength;
            verts[j*2+1] -= vertAlignment*lineHeight;
         }

         // Reset linelength
         lineLength = 0.0f;

         // set the index of the last line
         lastLineVert = i*6;
      }
   }

   return index;
}
