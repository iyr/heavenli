/*
 * Forward Declarations
 */

/*
 * Text draw code + helper functions
 */
#include "fontUtils/characterStruct.h"    // Provides a simple struct for caching character glyph data
#include "fontUtils/atlasClass.h"         // Provides a class for building a Text Atlas + OpenGL texture mapping, etc.
#include "fontUtils/primCharTrig.cpp"     // Provides a primitive for drawing characters
#include "fontUtils/primStringTRIG.cpp"   // Provides a high-order primitive for drawing strings

//std::vector<textAtlas> fontAtlases;            // Used to store all generated fonts
textAtlas* quack;

// C/C++ function for drawing text
void drawText(
      std::string inputString,   // string of text draw
      GLfloat     horiAlignment, // 0.0=left, 0.5=center, 1.0=right
      GLfloat     vertAlignment, // 0.0=bottom, 0.5=center, 1.0=top
      GLfloat     gx,            // X position
      GLfloat     gy,            // Y position
      GLfloat     sx,            // X scale
      GLfloat     sy,            // Y scale
      GLfloat     w2h,           // width to height ration
      textAtlas*  atlas,         // texture atlas to draw characters from
      GLfloat*    textColor,     // color of text
      GLfloat*    faceColor,     // color of backdrop
      drawCall*   textLine,      // pointer to input drawCall to write text
      drawCall*   textBackdrop   // pointer to input drawCall to write text backdrop
      );

// C/C++ overload for drawing text, no background
void drawText(
      std::string inputString,   // string of text draw
      GLfloat     horiAlignment, // 0.0=left, 0.5=center, 1.0=right
      GLfloat     vertAlignment, // 0.0=bottom, 0.5=center, 1.0=top
      GLfloat     gx,            // X position
      GLfloat     gy,            // Y position
      GLfloat     sx,            // X scale
      GLfloat     sy,            // Y scale
      GLfloat     w2h,           // width to height ration
      textAtlas*  atlas,         // texture atlas to draw characters from
      GLfloat*    textColor,     // color of text
      GLfloat*    faceColor,     // color of backdrop
      drawCall*   textLine      // pointer to input drawCall to write text
      );
#include "fontUtils/drawText.cpp"         // Draws an input string with a given font

#include "fontUtils/buildAtlas.cpp"       // Builds a text Atlas with data ferried from Python, stores in global vector

