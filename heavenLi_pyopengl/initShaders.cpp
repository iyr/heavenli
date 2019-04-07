#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
GLuint shaderProgram;

GLuint LoadShader(const char *shadersrc, GLenum type) {
   GLuint shader;
   GLint compiled;

   shader = glCreateShader(type);

   // Sanity Check
   if (shader == 0)
      return 0;

   glShaderSource(shader, 1, &shadersrc, NULL);
   glCompileShader(shader);
   glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

   if (!compiled)
   {
      GLint infoLen = 0;

      glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
      
      if (infoLen > 1) {
         char *infoLog = new char[infoLen];

         glGetShaderInfoLog(shader, infoLen, NULL, infoLog);
         printf("Shader Compilation Failed :(  \n%s\n", infoLog);
         delete [] infoLog;
      }

      glDeleteShader(shader);
      return 0;
   }

   return shader;
}

PyObject* initShaders_shaderUtils(PyObject* self, PyObject *args) {
   GLbyte vertShaderSource[] = 
      "attribute vec4 vPostion;  \n"
      "void main() {             \n"
      "gl_Position = vPostion;   \n"
      "}                         \n";
   GLbyte fragShaderSource[] = 
      "precision mediump float;  \n"
      "void main() {             \n"
      "gl_FragColor = vec4(0.33, 0.05, 0.90, 1.0); \n"
      "}                         \n";

   GLuint linked;
   GLuint vertShader;
   GLuint fragShader;

   vertShader = LoadShader(vertShaderSource, GL_VERTEX_SHADER);
   fragShader = LoadShader(fragShaderSource, GL_FRAGMENT_SHADER);

   shaderProgram = glCreateProgram();

   if (shaderProgram == 0)
      return 0;

   glAttachShader(shaderProgram, vertShader);
   glAttachShader(shaderProgram, fragShader);

   glBindAttribLocation(shaderProgram, "vertCoord");

   glLinkProgram(shaderProgram);

   glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linked);
   
   if (!linked) 
   {
      GLint infoLen = 0;

      glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &infoLen);
      
      if (infoLen > 1) {
         char *infoLog = new char[infoLen];

         glGetShaderInfoLog(shader, infoLen, NULL, infoLog);
         printf("Shader Program Linking Failed :(  \n%s\n", infoLog);
         delete [] infoLog;
      }

      glDeleteProgram(shaderProgram);
      return 0;
   }

   Py_RETURN_NONE;
}
