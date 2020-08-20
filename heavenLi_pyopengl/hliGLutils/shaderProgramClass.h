#ifndef shaderProgClass
#define shaderProgClass

#include <fstream>

/*
 * Implements a helper class that wraps shader related operations
 * such as building, querying shader metadata, etc...
 * WIP
 */
class shaderProg {
   public:
      string vertShaderSourcePath;  // File path for vertex shader source code
      string fragShaderSourcePath;  // File path for fragment shader source code
      vector<vertAttribMetadata>  vertexAttribs;   // structs contain attrib metadata

      shaderProg(void);
      shaderProg(const GLchar* vertSourceFilePath, const GLchar* fragSourceFilePath);
      ~shaderProg(void);

      GLbyte   getAttribInd(const char* attribName);

      GLubyte  getNumAttribs(void);
      GLubyte  getNumAttributes(void);

      GLuint   getID(void);
      GLuint   getProgID(void);
      GLuint   getProgramID(void);

      GLuint   getVertexVectorWidth(void);

      GLuint   compileShader(const GLchar *shadersrc, GLenum type);
      GLuint   buildShader(void);//const GLchar* vertSourceFilePath, const GLchar* fragSourceFilePath);

   private:
      GLuint programID;
      GLuint numAttributes;
      GLuint vertexVectorWidth;

      GLuint parseVertAttribs(void);
};

shaderProg::shaderProg(void){
   return;
};

shaderProg::shaderProg(const GLchar* vertSourceFilePath, const GLchar* fragSourceFilePath){
   this->vertShaderSourcePath = string(vertSourceFilePath);
   this->fragShaderSourcePath = string(fragSourceFilePath);

   this->parseVertAttribs();

   return;
};

shaderProg::~shaderProg(void){
   return;
};

/*
 * Get total number of vector components of all attributes
 */
GLuint  shaderProg::getVertexVectorWidth(void) { return this->vertexVectorWidth;};

/*
 * Get number of attributes
 */
GLubyte shaderProg::getNumAttribs(void)   {return this->numAttributes;};
GLubyte shaderProg::getNumAttributes(void){return this->numAttributes;};

/*
 * Get ID of shader program for rendering
 */
GLuint shaderProg::getID(void)   {return this->programID;};
GLuint shaderProg::getProgID(void)  {return this->programID;};
GLuint shaderProg::getProgramID(void)  {return this->programID;};

/*
 * Get the index of the attribute with specified name
 * returns -1 if no such attribute exists
 */
GLbyte shaderProg::getAttribInd(const char* attribName){
   GLbyte tmn = this->numAttributes;

   for (GLbyte i = 0; i < tmn; i++)
      if (this->vertexAttribs[i].locationString.compare(attribName) == 0)
         return i;

   return -1;
}

/*
 * Parses vertex shader source to get info about its attributes
 */
GLuint shaderProg::parseVertAttribs(void){
   this->numAttributes = 0;
   this->vertexAttribs.clear();

   // Get vertex shader source code from file path
   ifstream vertShaderFile(this->vertShaderSourcePath);
   string vert_src(
         (istreambuf_iterator<char>(vertShaderFile)), 
         istreambuf_iterator<char>()
         );
   vertShaderFile.close();

   // Look for attributes, record metadata
   size_t attIndex = vert_src.find("attribute");
   while (attIndex != string::npos) {
      //printf("Found Attribute: ");

      vertAttribMetadata attributeInfo;
      attributeInfo.attribIndex = this->numAttributes;

      // Determine number of vector components
      attIndex = vert_src.find("vec", attIndex);
      GLubyte vectorSize = vert_src[attIndex+3]-'0'; 
      attributeInfo.vectorSize = vectorSize;
      attIndex += 4;

      // Skip over spaces to start of variable name
      while (vert_src[attIndex] == ' ')
         attIndex++;

      // Record index of start of variable name
      size_t tms = attIndex;

      // Find end of variable declaration
      attIndex = vert_src.find(";", attIndex);

      // Skip over spaces to end of variable name
      while (vert_src[attIndex-1] == ' ')
         attIndex--;
      
      // Get Variable name
      string tmn = vert_src.substr(tms, attIndex-tms);      
      attributeInfo.locationString = tmn;

      // Look for next attribute, if any
      attIndex = vert_src.find("attribute", attIndex);

      // Record attribute metadata
      this->vertexAttribs.push_back(attributeInfo);

      this->numAttributes += 1;
   }

   // Get total number of vector components
   GLuint totalVecLength = 0;
   for (GLuint i = 0; i < this->numAttributes; i++){
      this->vertexAttribs[i].vectorOffset = totalVecLength;
      totalVecLength += this->vertexAttribs[i].vectorSize;
   }

   this->vertexVectorWidth = totalVecLength;

   return this->numAttributes;
};

/*
 * Compiles shader source code (vert/frag) into GL shader objects
 */
GLuint shaderProg::compileShader(const GLchar *shadersrc, GLenum type) {
   GLuint shader;    // Output variable to store the compiled shader
   GLint compiled;   // Used to determine if compilation was successful

   // Set Shader type (EG vert, frac)
   shader = glCreateShader(type);

   // Sanity Check
   if (shader == 0) return 0;

   // Load shader glsl
   glShaderSource(shader, 1, &shadersrc, NULL);

   // Compile shader
   glCompileShader(shader);

   // Get compilation status
   glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

   // Print Error Message if compilation failed
   if (!compiled) {
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
};

/*
 * Compiles, links vert/frag GL shader objects into a usable GL program object
 */
GLuint shaderProg::buildShader(void){
   GLuint shaderID;

   // Get Vertex Shader source code from file path
   const GLchar* vertSourceFilePath = this->vertShaderSourcePath.c_str();
   ifstream vertShaderFile(vertSourceFilePath);
   string vertShaderFileContents(
         (istreambuf_iterator<char>(vertShaderFile)), 
         istreambuf_iterator<char>()
         );
   const GLchar* vertShaderSource = vertShaderFileContents.c_str();
   vertShaderFile.close();

   // Get Fragment Shader source code from file path
   const GLchar* fragSourceFilePath = this->fragShaderSourcePath.c_str();
   ifstream fragShaderFile(fragSourceFilePath);
   string fragShaderFileContents(
         (istreambuf_iterator<char>(fragShaderFile)), 
         istreambuf_iterator<char>()
         );
   const GLchar* fragShaderSource = fragShaderFileContents.c_str();
   fragShaderFile.close();

   GLuint vertShader = this->compileShader(vertShaderSource, GL_VERTEX_SHADER);
   GLuint fragShader = this->compileShader(fragShaderSource, GL_FRAGMENT_SHADER);

   shaderID = glCreateProgram();

   // Sanity Check
   if (shaderID == 0) return 0;

   glAttachShader(shaderID, vertShader);
   glAttachShader(shaderID, fragShader);

   // Dynamically allocate attributes according shader, heck ye
   for (GLubyte i = 0; i < this->numAttributes; i++){
      glBindAttribLocation(
            shaderID,
            i,
            this->vertexAttribs[i].locationString.c_str()
            );
      glEnableVertexAttribArray(i);
   }

   glLinkProgram(shaderID);

   // Fail gracefully-ish if link unsuccessful
   GLint linked;
   glGetProgramiv(shaderID, GL_LINK_STATUS, &linked);
   if (!linked) {
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

   this->programID = shaderID;
   printf("shaderProgram ID: %i\n", this->programID);
   return shaderID;
};

#endif
