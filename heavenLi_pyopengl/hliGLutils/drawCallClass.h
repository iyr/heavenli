#ifndef drawCallClass
#define drawCallClass
#define GET_VARIABLE_NAME(Variable) (#Variable)
#define GET_CLASS_NAME(ClassName) (#ClassName)

extern GLuint whiteTex;
//extern map<string, GLuint> shaders;
extern map<string, shaderProg> shaderPrograms;
extern VertexAttributeStrings VAS;

struct cmp_str
{
   bool operator()(char const *a, char const *b) const
   {
      return strcmp(a, b) < 0;
   }
};

/*
 * Implements a helper class that wraps buffer object and transformation matrix functions
 * necessary for each and every draw-call
 */
class drawCall {
   public:
      //map<const GLchar*, attribCache, cmp_str>* vertCaches;
      map<string, attribCache>* vertCaches;

      GLuint      texID;         // OpenGL texture ID, defaults to plain white texture for drawing solid objects
      GLuint      numVerts;      // number of Vertices (not length of cache arrays)
      GLuint      VBO;           // Buffer Object for OpenGL to store data in graphics memory
      Matrix      MVP;           // Model-View-Projection Matrix for storing transformations
      GLboolean   colorsChanged; // Determines whether color-buffer should be updated
      string text;          // Object text

      drawCall(void);
      ~drawCall(void);
      void buildCache(GLuint numVerts, map<string, attribCache> &attributeData);
      void updateMVP(GLfloat gx, GLfloat gy, GLfloat sx, GLfloat sy, GLfloat rot, GLfloat w2h);
      void setMVP(Matrix newMVP);
      void setMVP(GLfloat* newMVP);
      void setMVP(GLdouble* newMVP);
      void draw(void);
      void updateBuffer(string vertAttrib);
      void setNumColors(unsigned int numColors);
      void setColorQuartet(unsigned int setIndex, GLfloat* quartet);
      void setDrawType(GLenum type);
      void setShader(string shader);
      void setTex(GLuint TexID);

      void* getAttribCache(string CacheName);

   private:
      void*       junkData;         // Block of data for gracefully handling unknown attribute strings

      Params      prevTransforms;   // Useful for know if MVP should be recomputed

      GLfloat*    colorQuartets;    // Continuous array to store colorSet quartets

      GLboolean   firstRun;         // Determines if function is running for the first time (for VBO initialization)
      GLenum      drawType;         // GL_TRIANGLE_STRIP / GL_LINE_STRIP
      GLuint      numColors;        // Number of colorSet quartets (R, G, B, A) to manage (min: 1, typ: 2)

      string shader;        // Object shader
};

drawCall::drawCall(void) {
   printf("Building drawCall Object...\n");

   // Used to setup GL objects 
   this->firstRun       = GL_TRUE;   
   //this->shader         = "Default";
   //this->vertCaches     = new vector<attribCache>(shaderPrograms["Default"].getNumAttribs());
   this->vertCaches     = new map<string, attribCache>;

   this->setShader("Default");

   // Caches are empty
   this->numVerts       = 0;

   this->numColors      = 1;

   this->colorQuartets  = new GLfloat[this->numColors*4];
   memset(this->colorQuartets, 0.0f, this->numColors*4*sizeof(GLfloat));

   this->colorsChanged  = GL_FALSE;

   //this->drawType       = GL_LINE_STRIP;
   this->drawType       = GL_TRIANGLE_STRIP;

   this->texID          = 0;
   return;
};

drawCall::~drawCall(void){
   // Deallocate caches
   //glDeleteBuffers(1, &this->VBO);
   return;
};

/*
 * Returns a pointer to the buffer where vertex data
 * for the corresponding attribute is stored
 */
void* drawCall::getAttribCache(string CacheName){
   if (this->vertCaches->count(CacheName) != 0)
      return (* this->vertCaches)[CacheName].getCachePtr();
   else
      return this->junkData;
}

void drawCall::setDrawType(GLenum type) {
   this->drawType = type;
};

void drawCall::setNumColors(unsigned int numColors) {
   if (this->numColors != numColors) {
      this->numColors = numColors;
      //printf("Updating number of colors: %d, array length: %d\n", numColors, numColors*4);
      
      // Safely (re)allocate array that contains color quartets
      if (  this->colorQuartets == NULL) {
         this->colorQuartets = new GLfloat[this->numColors*4];
         for (GLuint i = 0; i < numColors*4; i++) this->colorQuartets[i] = 0.0f;
      } else {
         delete [] this->colorQuartets;
         this->colorQuartets = new GLfloat[this->numColors*4];
         for (GLuint i = 0; i < numColors*4; i++) this->colorQuartets[i] = 0.0f;
      }
   }

   return;
}

/*
 * choose material shader for the draw call
 */
void drawCall::setShader(string shader){

   // Safety Check
   if (shaderPrograms.count(shader) <= 0) return;

   if (this->shader.compare(shader) != 0) {
      // Assign Shader 
      this->shader = shader;

      // Erase Caches and reset verts
      this->vertCaches->clear();
      this->numVerts = 0;
   }

   return;
}

/*
 * Set RGBA values at Index
 */
void drawCall::setColorQuartet(unsigned int setIndex, GLfloat* quartet) {

   // Saftey check
   if (  setIndex+3  >  this->numColors*4 ||
         setIndex+1  >  this->numColors   ){
      return;
   } else {

      // Check for change in color, only updating if so
      // and setting flag
      for (unsigned int i = 0; i < 4; i++) {
         if ( this->colorQuartets[4*setIndex+i] != quartet[i] ){
            this->colorQuartets[4*setIndex+i] = quartet[i];
            this->colorsChanged = GL_TRUE;
         }
      }
   }

   return;
}

/*
 * Builds buffer from input map of attribCache instances
 */
void drawCall::buildCache(const GLuint numVerts, map<string, attribCache> &attributeData){

   this->numVerts = numVerts;
   this->vertCaches->clear();

   printf("building cache from map of attribCaches...\n");

   map<string, attribCache>::iterator iter;

   GLubyte totalVectorSize = 0;
   attribCache tmpAttCache;
   for (iter = attributeData.begin(); iter != attributeData.end(); iter++) {
      (* this->vertCaches)[iter->first].copy(iter->second);
      totalVectorSize += (GLubyte)  (iter->second.getVectorSize());
   }

   printf("Total Vector Size of incoming attribute data: %d\n", totalVectorSize);

   if (this->texID == 0)
      this->texID = whiteTex;
   glGenBuffers(1, &this->VBO);

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

   GLintptr offset = 0;

   glBufferData(GL_ARRAY_BUFFER, totalVectorSize*sizeof(GLfloat)*this->numVerts, NULL, GL_STATIC_DRAW);

   GLuint vertsBytes = 0;
   for (iter = this->vertCaches->begin(); iter != this->vertCaches->end(); iter++){
      // Number of bytes per vert times number of verts
      // NOT total number of bytes (mult by vec size)
      // Convenience variable
      vertsBytes = GLsizeof(iter->second.getGLtype())*numVerts;
      glBufferSubData(GL_ARRAY_BUFFER,
         iter->second.getVectorOffset()*vertsBytes,
         iter->second.getVectorSize()*vertsBytes,
         iter->second.getCachePtr()
         );
      glVertexAttribPointer(
            iter->second.getAttribIndex(),
            iter->second.getVectorSize(),
            iter->second.getGLtype(),
            GL_FALSE,
            iter->second.getVectorSize()*GLsizeof(iter->second.getGLtype()),
            (GLintptr*)offset
            );

      glEnableVertexAttribArray(iter->second.getAttribIndex());

      offset += iter->second.getVectorSize()*GLsizeof(iter->second.getGLtype())*numVerts;
   }

   return;
};

/*
 * Method for manually assigning a precomputed matrix
 */
void drawCall::setMVP(Matrix newMVP){
   this->MVP = newMVP;
   return;
};

/*
 * Method for manually assigning a precomputed matrix from a contiguous C-Array
 */
void drawCall::setMVP(GLfloat* newMVP){
   Matrix* tmp = &this->MVP;
   tmp->mat[0][0] = newMVP[0*4+0];
   tmp->mat[0][1] = newMVP[0*4+1];
   tmp->mat[0][2] = newMVP[0*4+2];
   tmp->mat[0][3] = newMVP[0*4+3];

   tmp->mat[1][0] = newMVP[1*4+0];
   tmp->mat[1][1] = newMVP[1*4+1];
   tmp->mat[1][2] = newMVP[1*4+2];
   tmp->mat[1][3] = newMVP[1*4+3];

   tmp->mat[2][0] = newMVP[2*4+0];
   tmp->mat[2][1] = newMVP[2*4+1];
   tmp->mat[2][2] = newMVP[2*4+2];
   tmp->mat[2][3] = newMVP[2*4+3];

   tmp->mat[3][0] = newMVP[3*4+0];
   tmp->mat[3][1] = newMVP[3*4+1];
   tmp->mat[3][2] = newMVP[3*4+2];
   tmp->mat[3][3] = newMVP[3*4+3];

   return;
};

/*
 * Method for manually assigning a precomputed matrix from a contiguous C-Array
 */
void drawCall::setMVP(GLdouble* newMVP){
   this->MVP.mat[0][0] = (GLfloat)newMVP[0*4+0];
   this->MVP.mat[0][1] = (GLfloat)newMVP[0*4+1];
   this->MVP.mat[0][2] = (GLfloat)newMVP[0*4+2];
   this->MVP.mat[0][3] = (GLfloat)newMVP[0*4+3];

   this->MVP.mat[1][0] = (GLfloat)newMVP[1*4+0];
   this->MVP.mat[1][1] = (GLfloat)newMVP[1*4+1];
   this->MVP.mat[1][2] = (GLfloat)newMVP[1*4+2];
   this->MVP.mat[1][3] = (GLfloat)newMVP[1*4+3];

   this->MVP.mat[2][0] = (GLfloat)newMVP[2*4+0];
   this->MVP.mat[2][1] = (GLfloat)newMVP[2*4+1];
   this->MVP.mat[2][2] = (GLfloat)newMVP[2*4+2];
   this->MVP.mat[2][3] = (GLfloat)newMVP[2*4+3];

   this->MVP.mat[3][0] = (GLfloat)newMVP[3*4+0];
   this->MVP.mat[3][1] = (GLfloat)newMVP[3*4+1];
   this->MVP.mat[3][2] = (GLfloat)newMVP[3*4+2];
   this->MVP.mat[3][3] = (GLfloat)newMVP[3*4+3];

   return;
};

/*
 * Method for computing/cache transform matrix for 2D elements
 */
void drawCall::updateMVP(GLfloat gx, GLfloat gy, GLfloat sx, GLfloat sy, GLfloat rot, GLfloat w2h) {
   Matrix Ortho;
   Matrix ModelView;

   // Only recompute MVP matrix if there was an actual change in transformations
   if (  this->prevTransforms.ao    != rot   ||
         this->prevTransforms.dx    != gx    ||
         this->prevTransforms.dy    != gy    ||
         this->prevTransforms.sx    != sx    ||
         this->prevTransforms.sy    != sy    ||
         this->prevTransforms.w2h   != w2h   ){
      float left = -w2h, right = w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, gx, gy, 0.0f );
      if (w2h <= 1.0f) {
         MatrixScale( &ModelView, sx, sy*w2h, 1.0f );
      } else {
         MatrixScale( &ModelView, sx/w2h, sy, 1.0f );
      }
      MatrixRotate( &ModelView, rot, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &this->MVP, &ModelView, &Ortho );

      this->prevTransforms.ao = rot;
      this->prevTransforms.dx = gx;
      this->prevTransforms.dy = gy;
      this->prevTransforms.sx = sx;
      this->prevTransforms.sy = sy;
      this->prevTransforms.w2h = w2h;
   }

   return;
};

/*
 * This method provides the actual OpenGL draw call:
 * The transformation matrix is passed to the shader and
 * the vertex buffer is selected and mapped out
 */
void drawCall::draw(void) {

   // Set desired shader program
   glUseProgram(shaderPrograms[this->shader].getID());

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &this->MVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

   // Bind Object texture
   glBindTexture(GL_TEXTURE_2D, this->texID);

   GLubyte  tmNumAttribs = shaderPrograms[this->shader].getNumAttribs();
   GLuint   vecSize  = 0, 
            offset   = 0;
   for (GLubyte i = 0; i < tmNumAttribs; i++) {
      vecSize = shaderPrograms[this->shader].vertexAttribs[i].getVectorSize();
      glVertexAttribPointer(
            shaderPrograms[this->shader].vertexAttribs[i].getAttribIndex(),
            vecSize,
            GL_FLOAT,
            GL_FALSE,
            vecSize*sizeof(GLfloat),
            (void*)(offset*sizeof(GLfloat)*this->numVerts)
            );
      offset += vecSize;
   }

   glDrawArrays(this->drawType, 0, this->numVerts);

   // Unbind Buffer Object
   //glBindBuffer(GL_ARRAY_BUFFER, 0);

   //glUseProgram(0);
   return;
}

/*
 * Updates Buffer object with cache
 */
void drawCall::updateBuffer(string vertAttrib){
   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

   // Initialize offset to begin storing data in latter part of the buffer
   if (this->vertCaches->count(vertAttrib) != 0) {
      attribCache* tmAttrib = &(* this->vertCaches)[vertAttrib];
      GLuint vertsBytes = this->numVerts*sizeof(GLfloat);

      // Load Vertex coordinate data into VBO
      glBufferSubData(
            GL_ARRAY_BUFFER,                                   // Buffer type
            (GLintptr )tmAttrib->getVectorOffset()*vertsBytes, // Where to start writing
            (GLsizei  )tmAttrib->getVectorSize()*vertsBytes,   // How much to write
            (GLfloat *)tmAttrib->getCachePtr()                 // What we're writing
            );
   } else {
      printf("WARNING: trying to update buffer of unknown vertex attribute (ain't nuthin' gonna happen)\n");
   }

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   if (vertAttrib.compare("colorData") == 0) 
      this->colorsChanged = GL_FALSE;

   return;
};

void drawCall::setTex(GLuint TexID){
   this->texID = TexID;
   return;
};

#endif
