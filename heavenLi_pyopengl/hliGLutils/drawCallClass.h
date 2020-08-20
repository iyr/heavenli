#ifndef drawCallClass
#define drawCallClass

extern GLuint whiteTex;
extern map<string, GLuint> shaders;
extern map<string, shaderProg> shaderPrograms;

/*
 * Implements a helper class that wraps buffer object and transformation matrix functions
 * necessary for each and every draw-call
 */
class drawCall {
   public:
      GLfloat*    coordCache;    // Array of Vertex coordinate data (X, Y).
      GLfloat*    texuvCache;    // Array of Vertex Texture coordinates data (U, V).
      GLfloat*    colorCache;    // Array of Vertex color data (R, G, B, A)
      GLfloat*    normCache;     // Array of Vertex normals, for 3D use (X, Y, Z)

      GLuint      texID;         // OpenGL texture ID, defaults to plain white texture for drawing solid objects
      GLuint      numVerts;      // number of Vertices (not length of cache arrays)
      GLuint      VBO;           // Buffer Object for OpenGL to store data in graphics memory
      Matrix      MVP;           // Model-View-Projection Matrix for storing transformations
      GLboolean   colorsChanged; // Determines whether color-buffer should be updated
      string text;          // Object text

      drawCall(void);
      ~drawCall(void);
      void buildCache(GLuint numVerts, vector<GLfloat> &verts, vector<GLfloat> &colrs);
      void buildCache(GLuint numVerts, vector<GLfloat> &verts, vector<GLfloat> &texuv, vector<GLfloat> &colrs);
      void updateMVP(GLfloat gx, GLfloat gy, GLfloat sx, GLfloat sy, GLfloat rot, GLfloat w2h);
      void setMVP(Matrix newMVP);
      void draw(void);
      void updateCoordCache(void);
      void updateTexUVCache(void);
      void updateColorCache(void);
      void updateNormVCache(void);
      void setNumColors(unsigned int numColors);
      void setColorQuartet(unsigned int setIndex, GLfloat* quartet);
      void setDrawType(GLenum type);
      void setShader(string shader);
      void setTex(GLuint TexID);

   private:
      Params      prevTransforms;   // Useful for know if MVP should be recomputed

      GLfloat*    colorQuartets;    // Continuous array to store colorSet quartets

      GLboolean   firstRun;         // Determines if function is running for the first time (for VBO initialization)
      GLenum      drawType;         // GL_TRIANGLE_STRIP / GL_LINE_STRIP
      GLuint      numColors;        // Number of colorSet quartets (R, G, B, A) to manage (min: 1, typ: 2)

      string shader;        // Object shader
};

drawCall::drawCall(void) {
   printf("Building drawCall Object...\n");
   // Initialize Caches
   this->coordCache     = NULL;
   this->texuvCache     = NULL;
   this->colorCache     = NULL;
   this->normCache      = NULL;

   // Used to setup GL objects 
   this->firstRun       = GL_TRUE;   
   this->shader         = "Default";

   // Caches are empty
   this->numVerts       = 0;

   this->numColors      = 1;

   this->colorQuartets  = new GLfloat[this->numColors*4];

   this->colorsChanged  = GL_FALSE;

   //this->drawType       = GL_LINE_STRIP;
   this->drawType       = GL_TRIANGLE_STRIP;

   this->texID          = 0;
   return;
};

drawCall::~drawCall(void){
   // Deallocate caches
   delete [] this->coordCache;
   delete [] this->texuvCache;
   delete [] this->colorCache;
   delete [] this->normCache;
   //glDeleteBuffers(1, &this->VBO);
   return;
};

void drawCall::setDrawType(GLenum type) {
   this->drawType = type;
}

void drawCall::setNumColors(unsigned int numColors) {
   if (this->numColors != numColors) {
      this->numColors = numColors;
      printf("Updating number of colors: %d, array length: %d\n", numColors, numColors*4);
      
      // Safely (re)allocate array that contains color quartets
      if (  this->colorQuartets == NULL) {
         this->colorQuartets = new GLfloat[this->numColors*4];
      } else {
         delete [] this->colorQuartets;
         this->colorQuartets = new GLfloat[this->numColors*4];
      }
   }

   return;
}

/*
 * choose material shader for the draw call
 */
void drawCall::setShader(string shader){
   //if (shaders.count(shader) <= 0) return;
   if (shaderPrograms.count(shader) <= 0) return;
   this->shader = shader;
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
 * Builds cache from input vectors and writes to buffer object
 */
void drawCall::buildCache(GLuint numVerts, vector<GLfloat> &verts, vector<GLfloat> &texuv, vector<GLfloat> &colrs) {
   this->numVerts = numVerts;

   GLuint congruentVertices = 0;

   // Safely allocate cache arrays
   if (this->coordCache == NULL) {
      this->coordCache = new GLfloat[this->numVerts*2];
   } else {
      delete [] this->coordCache;
      this->coordCache = new GLfloat[this->numVerts*2];
   }

   if (this->texuvCache == NULL) {
      this->texuvCache = new GLfloat[this->numVerts*2];
   } else {
      delete [] this->texuvCache;
      this->texuvCache = new GLfloat[this->numVerts*2];
   }

   if (this->colorCache == NULL) {
      this->colorCache = new GLfloat[this->numVerts*4];
   } else {
      delete [] this->colorCache;
      this->colorCache = new GLfloat[this->numVerts*4];
   }

   // Copy contents of input vectors to cache arrays
   for (unsigned int i = 0; i < this->numVerts; i++) {
      this->coordCache[i*2]   = verts[i*2];
      this->coordCache[i*2+1] = verts[i*2+1];

      this->texuvCache[i*2]   = texuv[i*2];
      this->texuvCache[i*2+1] = texuv[i*2+1];

      this->colorCache[i*4+0] = colrs[i*4+0];
      this->colorCache[i*4+1] = colrs[i*4+1];
      this->colorCache[i*4+2] = colrs[i*4+2];
      this->colorCache[i*4+3] = colrs[i*4+3];

      // Count congruent vertices
      if ( i < this->numVerts-2)
         if (  verts[i*2+0] == verts[(i+1)*2+0]  ||
               verts[i*2+1] == verts[(i+1)*2+1]  )
            congruentVertices++;
   }

   // Create buffer object if one does not exist, otherwise, delete and make a new one
   // This cannot be done in the constructor for global object instances because OpenGL
   // must be initialized before any GL functions are called. 
   if (this->firstRun == GL_TRUE) {
      this->firstRun = GL_FALSE;
      if (this->texID == 0)
         this->texID = whiteTex;
      glGenBuffers(1, &this->VBO);
   } else {
      glDeleteBuffers(1, &this->VBO);
      glGenBuffers(1, &this->VBO);
   }

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

   // Allocate space to hold all vertex coordinate and color data
   printf("Allocating Buffer, size: %d bytes, %d Total Vertices, %d Congruent Vertices.\n", int(8*sizeof(GLfloat)*this->numVerts), this->numVerts, congruentVertices);

   GLuint   totalVecWidth  = shaderPrograms[this->shader].getVertexVectorWidth(),
            vecSize        = 0;
   GLintptr offset         = 0;
   GLubyte  numAttribs     = shaderPrograms[this->shader].getNumAttribs();

   glBufferData(GL_ARRAY_BUFFER, totalVecWidth*sizeof(GLfloat)*this->numVerts, NULL, GL_STATIC_DRAW);

   for (GLubyte i = 0; i < numAttribs; i++){
      vecSize = shaderPrograms[this->shader].vertexAttribs[i].vectorSize;

      if (shaderPrograms[this->shader].vertexAttribs[i].locationString.compare("vertCoord") == 0)
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*vecSize*this->numVerts, this->coordCache);

      if (shaderPrograms[this->shader].vertexAttribs[i].locationString.compare("vertTexUV") == 0)
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*vecSize*this->numVerts, this->texuvCache);

      if (shaderPrograms[this->shader].vertexAttribs[i].locationString.compare("vertColor") == 0)
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*vecSize*this->numVerts, this->colorCache);

      glVertexAttribPointer(
            i,
            vecSize,
            GL_FLOAT,
            GL_FALSE,
            vecSize*sizeof(GLfloat),
            (GLintptr*)offset
            );
      glEnableVertexAttribArray(i);

      offset += vecSize*sizeof(GLfloat)*this->numVerts;
   }

   /*
   glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat)*this->numVerts, NULL, GL_STATIC_DRAW);
   //glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat)*this->numVerts, NULL, GL_DYNAMIC_DRAW);
   //glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat)*this->numVerts, NULL, GL_STREAM_DRAW);

   // Convenience variables
   GLintptr offset = 0;
   GLuint vertAttribCoord = glGetAttribLocation(shaders[this->shader], "vertCoord");
   GLuint vertAttribTexUV = glGetAttribLocation(shaders[this->shader], "vertTexUV");
   GLuint vertAttribColor = glGetAttribLocation(shaders[this->shader], "vertColor");

   // Load Vertex coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*this->numVerts, this->coordCache);
   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
   // Enable the vertex attribute
   glEnableVertexAttribArray(vertAttribCoord);

   // Update offset to begin storing data in latter part of the buffer
   offset += 2*sizeof(GLfloat)*this->numVerts;

   // Load Vertex Texture UV coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*this->numVerts, this->texuvCache);
   // Define how the data is layed out in the buffer
   glVertexAttribPointer(vertAttribTexUV, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
   // Enable the vertex attribute
   glEnableVertexAttribArray(vertAttribTexUV);

   // Update offset to begin storing data in latter part of the buffer
   offset += 2*sizeof(GLfloat)*this->numVerts;

   // Load Vertex Color data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*4*this->numVerts, this->colorCache);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(vertAttribColor, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (GLintptr*)offset);
   // Enable the vertex attribute
   glEnableVertexAttribArray(vertAttribColor);
   */

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   printf("Finished building buffers\n");
   return;
};

/*
 * Builds cache from input vectors and writes to buffer object
 */
void drawCall::buildCache(GLuint numVerts, vector<GLfloat> &verts, vector<GLfloat> &colrs) {
   vector<GLfloat> texuv(numVerts*2, 0.0f);
   this->buildCache(numVerts, verts, texuv, colrs);
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
   //glUseProgram(shaders[this->shader]);
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
      vecSize = shaderPrograms[this->shader].vertexAttribs[i].vectorSize;
      glVertexAttribPointer(
            i,
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
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   //glUseProgram(0);
   return;
}

/*
 * Updates Buffer object with cache
 */
void drawCall::updateColorCache(void) {

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

   shaderProg* tmsh = &shaderPrograms[this->shader];

   // Initialize offset to begin storing data in latter part of the buffer
   GLubyte  attInd   = (GLubyte)tmsh->getAttribInd("vertColor");
   GLubyte  vecSize  = tmsh->vertexAttribs[attInd].vectorSize;
   GLubyte  vecOffset= tmsh->vertexAttribs[attInd].vectorOffset;
   GLintptr offset   = vecOffset*sizeof(GLfloat)*this->numVerts;

   // Load Vertex coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*vecSize*this->numVerts, this->colorCache);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   this->colorsChanged = GL_FALSE;
   return;
}

/*
 * Updates Buffer object with cache
 */
void drawCall::updateCoordCache(void) {

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

   shaderProg* tmsh = &shaderPrograms[this->shader];

   // Initialize offset to begin storing data in latter part of the buffer
   GLubyte  attInd   = (GLubyte)tmsh->getAttribInd("vertCoord");
   GLubyte  vecSize  = tmsh->vertexAttribs[attInd].vectorSize;
   GLubyte  vecOffset= tmsh->vertexAttribs[attInd].vectorOffset;
   GLintptr offset   = vecOffset*sizeof(GLfloat)*this->numVerts;

   // Load Vertex coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*vecSize*this->numVerts, this->coordCache);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   return;
}

/*
 * Updates Buffer object with cache
 */
void drawCall::updateTexUVCache(void) {

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

   shaderProg* tmsh = &shaderPrograms[this->shader];

   // Initialize offset to begin storing data in latter part of the buffer
   GLubyte  attInd   = (GLubyte)tmsh->getAttribInd("vertTexUV");
   GLubyte  vecSize  = tmsh->vertexAttribs[attInd].vectorSize;
   GLubyte  vecOffset= tmsh->vertexAttribs[attInd].vectorOffset;
   GLintptr offset   = vecOffset*sizeof(GLfloat)*this->numVerts;

   // Load Vertex coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*vecSize*this->numVerts, this->texuvCache);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   return;
}

void drawCall::setTex(GLuint TexID){
   this->texID = TexID;
   return;
}
#endif
