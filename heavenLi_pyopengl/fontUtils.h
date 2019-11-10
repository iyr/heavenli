/*
 * Text draw code + helper functions
 */
#include "fontUtils/characterStruct.h"    // Provides a simple struct for caching character glyph data
#include "fontUtils/atlasClass.h"         // Provides a class for building a Text Atlas + OpenGL texture mapping, etc.
#include "fontUtils/primCharTrig.cpp"     // Provides a primitive for drawing characters
#include "fontUtils/primStringTRIG.cpp"   // Provides a high-order primitive for drawing strings

//std::vector<textAtlas> fontAtlases;            // Used to store all generated fonts
textAtlas* quack;

//#include "fontUtils/loadChar.cpp"         // Will likely get depricated
#include "fontUtils/drawText.cpp"         // Draws an input string with a given font
#include "fontUtils/buildAtlas.cpp"       // Builds a text Atlas with data ferried from Python, stores in global vector

