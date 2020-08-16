#include <fstream>
GLuint   shaderProgram;
GLint    uniform_tex;
extern GLuint     whiteTex;
extern std::map<std::string, GLuint> shaders;

// Shader helper function
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
         if (type == GL_VERTEX_SHADER) printf("Vertex Shader Compilation Failed :(  \n%s\n", infoLog);
         if (type == GL_FRAGMENT_SHADER) printf("Fragment Shader Compilation Failed :(  \n%s\n", infoLog);
         delete [] infoLog;
      }

      glDeleteShader(shader);
      return 0;
   }

   if (type == GL_VERTEX_SHADER) printf("Vertex Shader Compiled Successfully\n");
   if (type == GL_FRAGMENT_SHADER) printf("Fragment Shader Compiled Successfully\n");
   return shader;
}

GLuint buildShader(
      const GLchar* vertexSourceFile,
      const GLchar* fragmentSourceFile
      ){
   GLuint shaderID;

   std::ifstream vertShaderFile(vertexSourceFile);
   std::string vertShaderFileContents(
         (std::istreambuf_iterator<char>(vertShaderFile)), 
         std::istreambuf_iterator<char>()
         );

   const GLchar *vertShaderSource = vertShaderFileContents.c_str();
   vertShaderFile.close();

   std::ifstream fragShaderFile(fragmentSourceFile);
   std::string fragShaderFileContents(
         (std::istreambuf_iterator<char>(fragShaderFile)), 
         std::istreambuf_iterator<char>()
         );

   const GLchar *fragShaderSource = fragShaderFileContents.c_str();
   fragShaderFile.close();

   GLint linked;
   GLuint vertShader;
   GLuint fragShader;

   vertShader = LoadShader(vertShaderSource, GL_VERTEX_SHADER);
   fragShader = LoadShader(fragShaderSource, GL_FRAGMENT_SHADER);

   shaderID = glCreateProgram();

   if (shaderID == 0) return 0;

   glAttachShader(shaderID, vertShader);
   glAttachShader(shaderID, fragShader);

   glBindAttribLocation(shaderID, 0, "vertCoord");
   glBindAttribLocation(shaderID, 1, "vertTexUV");
   glBindAttribLocation(shaderID, 2, "vertColor");

   glLinkProgram(shaderID);

   glGetProgramiv(shaderID, GL_LINK_STATUS, &linked);
   
   if (!linked) 
   {
      GLint infoLen = 0;

      glGetProgramiv(shaderID, GL_INFO_LOG_LENGTH, &infoLen);
      
      if (infoLen > 1) {
         char *infoLog = new char[infoLen];

         glGetShaderInfoLog(shaderID, infoLen, NULL, infoLog);
         printf("Shader Program Linking Failed :(  \n%s\n", infoLog);
         delete [] infoLog;
      }

      glDeleteProgram(shaderID);
      return 0;
   }

   printf("shaderProgram ID: %i\n", shaderID);

   return shaderID;
}

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

   if (shaders.count("RGBAcolor_RGBAtexture") <= 0)
      shaders.insert(
            std::make_pair(
               "RGBAcolor_RGBAtexture", 
               buildShader(
                  "hliGLutils/shaders/RGBAcolor_UVtexture.vert",
                  "hliGLutils/shaders/RGBAcolor_RGBAtexture.frag"
                  )
               )
            );
   if (shaders.count("Default") <= 0)
      shaders.insert(
            std::make_pair(
               "Default", 
               shaders["RGBAcolor_RGBAtexture"]
               )
            );
   if (shaders.count("RGBAcolor_Atexture") <= 0)
      shaders.insert(
            std::make_pair(
               "RGBAcolor_Atexture", 
               buildShader(
                  "hliGLutils/shaders/RGBAcolor_UVtexture.vert",
                  "hliGLutils/shaders/RGBAcolor_Atexture.frag"
                  )
               )
            );
   if (shaders.count("RGBAcolor_NoTexture") <= 0)
      shaders.insert(
            std::make_pair(
               "RGBAcolor_NoTexture", 
               buildShader(
                  "hliGLutils/shaders/RGBAcolor_NoTexture.vert",
                  "hliGLutils/shaders/RGBAcolor_NoTexture.frag"
                  )
               )
            );

   //shaderProgram = shaders["Default"];
   //glUseProgram(shaderProgram);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);
   glEnableVertexAttribArray(2);

   Py_RETURN_NONE;
}
