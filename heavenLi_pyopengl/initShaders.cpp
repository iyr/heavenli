#include <Python.h>
#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glext.h>
GLuint shaderProgram;

GLuint LoadShader(const GLchar *shadersrc, GLenum type) {
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

   printf("Shader Compiled Successfully\n");
   return shader;
}

PyObject* initShaders_shaderUtils(PyObject* self, PyObject *args) {
   const GLchar vertShaderSource[] = 
      "#version 100			               \n"
      "attribute  vec4 vertCoord;         \n"
      "attribute  vec4 vertColor;         \n"
      "uniform    mat4 MVP;               \n"
      "varying    vec4 color;             \n"
      "varying    vec2 texCoord;          \n"
      "void main() {                      \n"
         "color = vertColor;              \n"
         "gl_Position = MVP * vertCoord;  \n"
         "texCoord = vertCoord.zw;        \n"
      "}                                  \n";

   const GLchar fragShaderSource[] = 
      "#version 100			               \n"
      "precision  mediump float;		      \n"
      "varying    vec2 texCoord;          \n"
      "varying    vec4 color;             \n"
      "uniform    sampler2D tex;          \n"
      "void main() {                      \n"
         //"gl_FragColor = vec4(1, 1, 1, texture2D(tex, texCoord).r)*color;           \n"
         "gl_FragColor = color;           \n"
      "}                                  \n";

   GLint linked;
   GLuint vertShader;
   GLuint fragShader;

   vertShader = LoadShader(vertShaderSource, GL_VERTEX_SHADER);
   fragShader = LoadShader(fragShaderSource, GL_FRAGMENT_SHADER);

   shaderProgram = glCreateProgram();

   if (shaderProgram == 0)
      return 0;

   glAttachShader(shaderProgram, vertShader);
   glAttachShader(shaderProgram, fragShader);

   glBindAttribLocation(shaderProgram, 0, "vertCoord");
   glBindAttribLocation(shaderProgram, 1, "vertColor");

   glLinkProgram(shaderProgram);

   glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linked);
   
   if (!linked) 
   {
      GLint infoLen = 0;

      glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &infoLen);
      
      if (infoLen > 1) {
         char *infoLog = new char[infoLen];

         glGetShaderInfoLog(shaderProgram, infoLen, NULL, infoLog);
         printf("Shader Program Linking Failed :(  \n%s\n", infoLog);
         delete [] infoLog;
      }

      glDeleteProgram(shaderProgram);
      return 0;
   }

   printf("shaderProgram ID: %i\n", shaderProgram);

   glUseProgram(shaderProgram);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);

   Py_RETURN_NONE;
}
