#include <Python.h>
#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glext.h>

GLuint   shaderProgram;
GLint    uniform_tex;
extern GLuint     whiteTex;
//extern GLubyte*   blankTexture;

/*
typedef struct {
   GLuint   shaderProgram;

   GLint    vertCoord;
   GLint    vertColor;

   GLint    vertSampler;

   GLuint   textureID;
}
*/

// Shader halper function
GLuint LoadShader(const GLchar *shadersrc, GLenum type) {
   GLuint shader;    // Output variable to store the compiled shader
   GLint compiled;   // Used to determine if compilation was successful

   // Set Shader type (EG vert, frac)
   shader = glCreateShader(type);

   // Sanity Check
   if (shader == 0)
      return 0;

   // Load shader glsl
   glShaderSource(shader, 1, &shadersrc, NULL);

   // Compile shader
   glCompileShader(shader);

   // Get compilation status
   glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

   // Print Error Message if compilation failed
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

PyObject* initShaders_hliGLutils(PyObject* self, PyObject *args) {

   //Generate plain white texture for drawing solid color objects
   //OpenGL ES 2.0 requires a minimum texture size of 64*64, apparently
   GLubyte* blankTexture;
   blankTexture = new GLubyte[64*64*4];
   memset(blankTexture, 255, 64*64*4);

   glActiveTexture(GL_TEXTURE0);
   glGenTextures(1, &whiteTex);
   printf("blank texture id: %x\n", whiteTex);
   glBindTexture(GL_TEXTURE_2D, whiteTex);
   glUniform1i(uniform_tex, 0);

   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, blankTexture);

   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   delete [] blankTexture;

   const GLchar vertShaderSource[] = 
      "#version 100			               \n"
      "attribute  vec2 vertCoord;         \n"
      "attribute  vec2 vertTexUV;         \n"
      "attribute  vec4 vertColor;         \n"
      "uniform    mat4 MVP;               \n"

      "varying    vec4 color;             \n"
      "varying    vec2 texCoord;          \n"
      "void main() {                      \n"
         "color = vertColor;              \n"
         "gl_Position = MVP * vec4(vertCoord, 1, 1);  \n"
         "texCoord = vertTexUV;           \n"
      "}                                  \n";

   const GLchar fragShaderSource[] = 
      "#version 100			               \n"
      "precision  mediump float;		      \n"
      "varying    vec2 texCoord;          \n"
      "varying    vec4 color;             \n"
      "uniform    sampler2D tex;          \n"
      "void main() {                      \n"
         //"gl_FragColor = vec4(color.r, color.g, color.b*texture2D(tex, texCoord).a, color.a);           \n"
         "gl_FragColor = vec4(color.r, color.g, color.b, color.a*texture2D(tex, texCoord).a);           \n"
         //"gl_FragColor = color;           \n"
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
   glBindAttribLocation(shaderProgram, 1, "vertTexUV");
   glBindAttribLocation(shaderProgram, 2, "vertColor");

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
   glEnableVertexAttribArray(2);

   Py_RETURN_NONE;
}
