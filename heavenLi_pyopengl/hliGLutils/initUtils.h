
/*
 * Initialize globals
 */
vector<string> drawQue;
map<string, drawCall>     drawCalls;     // Contains all draw call objects
map<string, textAtlas>    textFonts;     // Contains all text atlases
//map<string, GLuint>       shaders;       // Contains all shader ids
map<string, shaderProg>   shaderPrograms;// Contains shader programs
string selectedAtlas;
GLuint   whiteTex;
#include "initUtils/initShaders.cpp"      // Code that builds a shaderProgram (vert+frag) from source
