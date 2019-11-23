//#include "initUtils/initHLIobjects.cpp"   // initializes hli related global objects

GLuint buildShader(const GLchar* vertexSource, const GLchar* fragmentSource);
std::vector<std::string> drawQue;
std::map<std::string, drawCall>     drawCalls;
std::map<std::string, textAtlas*>   textFonts;
std::map<std::string, GLuint>       shaders;
std::string selectedAtlas;
GLuint   whiteTex;
#include "initUtils/initShaders.cpp"      // Code that builds a shaderProgram (vert+frag) from source
