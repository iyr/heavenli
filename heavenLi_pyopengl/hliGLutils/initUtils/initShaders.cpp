#include <fstream>

GLint    uniform_tex;
extern GLuint     whiteTex;
extern map<string, GLuint> shaders;
extern map<string, shaderProg> shaderPrograms;

PyObject* initShaders_hliGLutils(PyObject* self, PyObject *args) {

   //Generate plain white texture for drawing solid color objects
   GLubyte* blankTexture;
   blankTexture = new GLubyte[4];
   memset(blankTexture, 255, 4);
   GLint maxTexSize;
   glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTexSize);
   printf("Platform Maximum supported texture size: %d\n", maxTexSize);

   glActiveTexture(GL_TEXTURE0);
   glGenTextures(1, &whiteTex);
   printf("blank texture id: %x\n", whiteTex);
   glBindTexture(GL_TEXTURE_2D, whiteTex);
   glUniform1i(uniform_tex, 0);

   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, blankTexture);

   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

   delete [] blankTexture;

   glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &maxTexSize);
   printf("Platform Maximum number of shader attributes: %d\n", maxTexSize);

   if (shaderPrograms.count("RGBAcolor_RGBAtexture") <= 0)
      shaderPrograms.insert(
            make_pair(
               "RGBAcolor_RGBAtexture",
               shaderProg(
                  "hliGLutils/shaders/RGBAcolor_UVtexture.vert",
                  "hliGLutils/shaders/RGBAcolor_RGBAtexture.frag"
                  )
               )
            );
   shaderPrograms["RGBAcolor_RGBAtexture"].buildShader();

   if (shaderPrograms.count("Default") <= 0)
      shaderPrograms.insert(
            make_pair(
               "Default", 
               shaderPrograms["RGBAcolor_RGBAtexture"]
               )
            );

   if (shaderPrograms.count("RGBAcolor_Atexture") <= 0)
      shaderPrograms.insert(
            make_pair(
               "RGBAcolor_Atexture", 
               shaderProg(
                  "hliGLutils/shaders/RGBAcolor_UVtexture.vert",
                  "hliGLutils/shaders/RGBAcolor_Atexture.frag"
                  )
               )
            );
   shaderPrograms["RGBAcolor_Atexture"].buildShader();

   if (shaderPrograms.count("RGBAcolor_NoTexture") <= 0)
      shaderPrograms.insert(
            make_pair(
               "RGBAcolor_NoTexture", 
               shaderProg(
                  "hliGLutils/shaders/RGBAcolor_NoTexture.vert",
                  "hliGLutils/shaders/RGBAcolor_NoTexture.frag"
                  )
               )
            );
   shaderPrograms["RGBAcolor_NoTexture"].buildShader();

   Py_RETURN_NONE;
}
