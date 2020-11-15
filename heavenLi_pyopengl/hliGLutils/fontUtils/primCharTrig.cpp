/*
 * Defines a textured quad of a character for drawing text from a text atlas
 */

using namespace std;

unsigned int defineChar(
      float       x,             // x-postion
      float       y,             // y-postion
      int         c,             // index of the character
      float       hCrop,         // (-1 to +1) 0.0: no crop, >0: from right, <0: from left
      float       vCrop,         // (-1 to +1) 0.0: no crop, >0: from bottom, <0: from top
      textAtlas*  atlas,         // texture atlas to draw characters from
      float*      color,         // Polygon Color                                         
      std::vector<float> &verts, // Input Vector of x,y coordinates                       
      std::vector<float> &texuv, // Input Vector of u,y texture coordinates
      std::vector<float> &colrs  // Input Vector of r,g,b values                          
      ) {

   float x2, y2, w, h, texWidth, texHeight, texOffsetX, texOffsetY;
   character* charGlyph = &atlas->glyphData[c];

   if ( c > 32 ) {
      x2          =  x + charGlyph->bearingLeft;
      y2          = -y - charGlyph->bearingTop;
      w           = charGlyph->bearingX;
      h           = charGlyph->bearingY;
      texOffsetX  = charGlyph->textureOffsetX;
      texOffsetY  = charGlyph->textureOffsetY;
   } else {
      x2          = 0.0f;
      y2          = 0.0f;
      w           = 0.0f;
      h           = 0.0f;
      texOffsetX  = 0.0f;
      texOffsetY  = 0.0f;
   }
   texWidth    = (float)atlas->textureWidth;
   texHeight   = (float)atlas->textureHeight;

   // Convenience Variables for vertical cropping of vertex height
   float v1    = (vCrop >= 0.0f ?  h*vCrop : 0.0f),
         v2    = (vCrop <  0.0f ?  h*vCrop : 0.0f);
   // Convenience Variables for horizontal cropping of vertex height
   float h1    = (hCrop >= 0.0f ? -w*hCrop : 0.0f),
         h2    = (hCrop <  0.0f ? -w*hCrop : 0.0f);
   texWidth    = (float)atlas->textureWidth;
   texHeight   = (float)atlas->textureHeight;

   // R
   verts.push_back( x2 + h2);
   verts.push_back(-y2 - h + v1);
   // Q
   verts.push_back( x2 + w + h1);
   verts.push_back(-y2 + v2);
   // P
   verts.push_back( x2 + h2);
   verts.push_back(-y2 + v2);

   // Q
   verts.push_back( x2 + w + h1);
   verts.push_back(-y2 + v2);
   // R
   verts.push_back( x2 + h2);
   verts.push_back(-y2 - h + v1);
   // S
   verts.push_back( x2 + w + h1);
   verts.push_back(-y2 - h + v1);

   float xTexOffset  = 1.0f/(2.0f*texWidth),
         yTexOffset  = 1.0f/(2.0f*texHeight),
         wRatio      = w/texWidth,
         hRatio      = h/texHeight;

   // Convenience Variables for vertical cropping of texture coordinate
   v1 = (vCrop >= 0.0f ? vCrop*(hRatio+yTexOffset) : 0.0f);
   v2 = (vCrop <  0.0f ? vCrop*(hRatio+yTexOffset) : 0.0f);
   // Convenience Variables for horizontal cropping of textured coordinate
   h1 = (hCrop >= 0.0f ? hCrop*(wRatio+xTexOffset) : 0.0f);
   h2 = (hCrop <  0.0f ? hCrop*(wRatio+xTexOffset) : 0.0f);

   // R
   texuv.push_back(texOffsetX + xTexOffset - h2);
   texuv.push_back(texOffsetY - yTexOffset + hRatio - v1);
   // Q
   texuv.push_back(texOffsetX - xTexOffset + wRatio - h1);
   texuv.push_back(texOffsetY + yTexOffset - v2);
   // P
   texuv.push_back(texOffsetX + xTexOffset - h2);
   texuv.push_back(texOffsetY + yTexOffset - v2);

   // Q
   texuv.push_back(texOffsetX - xTexOffset + wRatio - h1);
   texuv.push_back(texOffsetY + yTexOffset - v2);
   // R
   texuv.push_back(texOffsetX + xTexOffset - h2);
   texuv.push_back(texOffsetY - yTexOffset + hRatio - v1);
   // S
   texuv.push_back(texOffsetX - xTexOffset + wRatio - h1);
   texuv.push_back(texOffsetY - yTexOffset + hRatio - v1);

   colrs.push_back(color[0]);   colrs.push_back(color[1]);   colrs.push_back(color[2]);   colrs.push_back(color[3]);
   colrs.push_back(color[0]);   colrs.push_back(color[1]);   colrs.push_back(color[2]);   colrs.push_back(color[3]);
   colrs.push_back(color[0]);   colrs.push_back(color[1]);   colrs.push_back(color[2]);   colrs.push_back(color[3]);
   colrs.push_back(color[0]);   colrs.push_back(color[1]);   colrs.push_back(color[2]);   colrs.push_back(color[3]);
   colrs.push_back(color[0]);   colrs.push_back(color[1]);   colrs.push_back(color[2]);   colrs.push_back(color[3]);
   colrs.push_back(color[0]);   colrs.push_back(color[1]);   colrs.push_back(color[2]);   colrs.push_back(color[3]);

   return verts.size()/2;
}

unsigned int updateChar(
      float       x,       // x-postion
      float       y,       // y-postion
      GLuint      c,       // index of the character
      float       hCrop,   // (-1 to +1) 0.0: no crop, >0: from right, <0: from left
      float       vCrop,   // (-1 to +1) 0.0: no crop, >0: from bottom, <0: from top
      textAtlas*  atlas,   // texture atlas to draw characters from
      GLuint      index,   // Index of where to start writing to input arrays
      float*      verts,   // Input Array of x,y coordinates
      float*      texuv    // Input Array of u,v texture coordinates
      ) {

   unsigned int vertIndex = index*2;

   float x2, y2, w, h, texWidth, texHeight, texOffsetX, texOffsetY;
   character* charGlyph = &atlas->glyphData[c];

   if ( c > 32 ) {
      x2          =  x + charGlyph->bearingLeft;
      y2          = -y - charGlyph->bearingTop;
      w           = charGlyph->bearingX;
      h           = charGlyph->bearingY;
      texOffsetX  = charGlyph->textureOffsetX;
      texOffsetY  = charGlyph->textureOffsetY;
   } else {
      x2          = 0.0f;
      y2          = 0.0f;
      w           = 0.0f;
      h           = 0.0f;
      texOffsetX  = 0.0f;
      texOffsetY  = 0.0f;
   }

   // Convenience Variables for vertical cropping of vertex height
   float v1    = (vCrop >= 0.0f ?  h*vCrop : 0.0f),
         v2    = (vCrop <  0.0f ?  h*vCrop : 0.0f);
   // Convenience Variables for horizontal cropping of vertex height
   float h1    = (hCrop >= 0.0f ? -w*hCrop : 0.0f),
         h2    = (hCrop <  0.0f ? -w*hCrop : 0.0f);
   texWidth    = (float)atlas->textureWidth;
   texHeight   = (float)atlas->textureHeight;

   // R
   verts[vertIndex+0]   =  x2 + h2;
   verts[vertIndex+1]   = -y2 - h + v1;
   // Q
   verts[vertIndex+2]   =  x2 + w + h1;
   verts[vertIndex+3]   = -y2 + v2;
   // P
   verts[vertIndex+4]   =  x2 + h2;
   verts[vertIndex+5]   = -y2 + v2;
   
   // Q
   verts[vertIndex+6]   =  x2 + w + h1;
   verts[vertIndex+7]   = -y2 + v2;
   // R
   verts[vertIndex+8]   =  x2 + h2;
   verts[vertIndex+9]   = -y2 - h + v1;
   // S
   verts[vertIndex+10]  =  x2 + w + h1;
   verts[vertIndex+11]  = -y2 - h + v1;

   float xTexOffset  = 1.0f/(2.0f*texWidth),
         yTexOffset  = 1.0f/(2.0f*texHeight),
         wRatio      = w/texWidth,
         hRatio      = h/texHeight;


   // Convenience Variables for vertical cropping of texture coordinate
   v1 = (vCrop >= 0.0f ? vCrop*(hRatio+yTexOffset) : 0.0f);
   v2 = (vCrop <  0.0f ? vCrop*(hRatio+yTexOffset) : 0.0f);
   // Convenience Variables for horizontal cropping of textured coordinate
   h1 = (hCrop >= 0.0f ? hCrop*(wRatio+xTexOffset) : 0.0f);
   h2 = (hCrop <  0.0f ? hCrop*(wRatio+xTexOffset) : 0.0f);

   // R
   texuv[vertIndex+0]   = texOffsetX + xTexOffset - h2;
   texuv[vertIndex+1]   = texOffsetY - yTexOffset + hRatio - v1;
   // Q
   texuv[vertIndex+2]   = texOffsetX - xTexOffset + wRatio - h1;
   texuv[vertIndex+3]   = texOffsetY + yTexOffset - v2;
   // P
   texuv[vertIndex+4]   = texOffsetX + xTexOffset - h2;
   texuv[vertIndex+5]   = texOffsetY + yTexOffset - v2;

   // Q
   texuv[vertIndex+6]   = texOffsetX - xTexOffset + wRatio - h1;
   texuv[vertIndex+7]   = texOffsetY + yTexOffset - v2;
   // R
   texuv[vertIndex+8]   = texOffsetX + xTexOffset - h2;
   texuv[vertIndex+9]   = texOffsetY - yTexOffset + hRatio - v1;
   // S
   texuv[vertIndex+10]  = texOffsetX - xTexOffset + wRatio - h1;
   texuv[vertIndex+11]  = texOffsetY - yTexOffset + hRatio - v1;

   vertIndex += 12;

   return vertIndex/2;
}

unsigned int updateChar(
      float*   color,   // Polygon Color                                         
      GLuint   index,   // Index of where to start writing to input arrays
      float*   colrs    // Input Array of R, G, B, A, color values 
      ){

   GLuint colrIndex = index*4;

   colrs[colrIndex+0] = color[0];   
   colrs[colrIndex+1] = color[1];   
   colrs[colrIndex+2] = color[2];   
   colrs[colrIndex+3] = color[3];
   
   colrs[colrIndex+4] = color[0];   
   colrs[colrIndex+5] = color[1];   
   colrs[colrIndex+6] = color[2];   
   colrs[colrIndex+7] = color[3];
   
   colrs[colrIndex+8] = color[0];   
   colrs[colrIndex+9] = color[1];   
   colrs[colrIndex+10] = color[2];   
   colrs[colrIndex+11] = color[3];
   
   colrs[colrIndex+12] = color[0];   
   colrs[colrIndex+13] = color[1];   
   colrs[colrIndex+14] = color[2];   
   colrs[colrIndex+15] = color[3];
   
   colrs[colrIndex+16] = color[0];   
   colrs[colrIndex+17] = color[1];   
   colrs[colrIndex+18] = color[2];   
   colrs[colrIndex+19] = color[3];
   
   colrs[colrIndex+20] = color[0];   
   colrs[colrIndex+21] = color[1];   
   colrs[colrIndex+22] = color[2];   
   colrs[colrIndex+23] = color[3];

   colrIndex += 24;

   return colrIndex/4;
}

