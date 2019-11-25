#ifndef drawCallClass
#define drawCallClass

using namespace std;
extern GLuint shaderProgram;
extern GLuint whiteTex;

/*
 * Implements a helper class that wraps buffer object and transformation matrix functions
 * necessary for each and every draw-call
 */
class drawCall {
   public:
      GLfloat*    coordCache;    // Array of Vertex coordinate data (X, Y).
      GLfloat*    texuvCache;    // Array of Vertex Texture coordinates data (U, V).
      GLfloat*    colorCache;    // Array of Vertex color data (R, G, B, A)

      GLuint      texID;         // OpenGL texture ID, defaults to plain white texture for drawing solid objects
      GLuint      numVerts;      // number of Vertices (not length of cache arrays)
      GLuint      VBO;           // Buffer Object for OpenGL to store data in graphics memory
      Matrix      MVP;           // Model-View-Projection Matrix for storing transformations
      GLboolean   colorsChanged; // Determines whether color-buffer should be updated
      std::string text;          // Object text

      drawCall(void);
      drawCall(unsigned int numColors);
      ~drawCall(void);
      void buildCache(GLuint numVerts, std::vector<GLfloat> &verts, std::vector<GLfloat> &colrs);
      void buildCache(GLuint numVerts, std::vector<GLfloat> &verts, std::vector<GLfloat> &texuv, std::vector<GLfloat> &colrs);
      void updateMVP(GLfloat gx, GLfloat gy, GLfloat sx, GLfloat sy, GLfloat rot, GLfloat w2h);
      void draw(void);
      void updateCoordCache(void);
      void updateTexUVCache(void);
      void updateColorCache(void);
      void setNumColors(unsigned int numColors);
      void setColorQuartet(unsigned int setIndex, GLfloat* quartet);
      void setDrawType(GLenum type);
      void setTex(GLuint TexID);

   private:
      Params      prevTransforms;   // Useful for know if MVP should be recomputed

      GLfloat*    colorQuartets;    // Continuous array to store colorSet quartets

      GLboolean   firstRun;         // Determines if function is running for the first time (for VBO initialization)
      GLboolean   usesTex;          // Determines whether or not geometry uses a texture
      GLenum      drawType;         // GL_TRIANGLE_STRIP / GL_LINE_STRIP
      GLuint      numColors;        // Number of colorSet quartets (R, G, B, A) to manage (min: 1, typ: 2)
};

drawCall::drawCall(void) {
   printf("Building drawCall Object...\n");
   // Initialize Caches
   this->coordCache     = NULL;
   this->texuvCache     = NULL;
   this->colorCache     = NULL;

   // Used to setup GL objects 
   this->firstRun       = GL_TRUE;   

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

drawCall::drawCall(unsigned int numColors) {
   printf("Building drawCall Object...\n");
   // Initialize Caches
   this->coordCache     = NULL;
   this->texuvCache     = NULL;
   this->colorCache     = NULL;

   // Used to setup GL objects 
   this->firstRun       = GL_TRUE;   

   // Caches are empty
   this->numVerts       = 0;

   this->numColors      = numColors;

   this->colorQuartets  = new GLfloat[this->numColors*4];

   this->colorsChanged  = GL_FALSE;

   this->drawType       = GL_TRIANGLE_STRIP;

   this->texID          = 0;
   return;
};

drawCall::~drawCall(void){
   // Deallocate caches
   delete [] this->coordCache;
   delete [] this->texuvCache;
   delete [] this->colorCache;
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
void drawCall::buildCache(GLuint numVerts, std::vector<GLfloat> &verts, std::vector<GLfloat> &texuv, std::vector<GLfloat> &colrs) {
   this->usesTex  = true;
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
   glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat)*this->numVerts, NULL, GL_STATIC_DRAW);
   //glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat)*this->numVerts, NULL, GL_DYNAMIC_DRAW);
   //glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat)*this->numVerts, NULL, GL_STREAM_DRAW);

   // Convenience variables
   GLintptr offset = 0;
   GLuint vertAttribCoord = glGetAttribLocation(shaderProgram, "vertCoord");
   GLuint vertAttribTexUV = glGetAttribLocation(shaderProgram, "vertTexUV");
   GLuint vertAttribColor = glGetAttribLocation(shaderProgram, "vertColor");

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

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   return;
};
/*
 * Builds cache from input vectors and writes to buffer object
 */
void drawCall::buildCache(GLuint numVerts, std::vector<GLfloat> &verts, std::vector<GLfloat> &colrs) {
   this->usesTex  = false;
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
      this->texuvCache[i*2]   = 0.0f;
      this->texuvCache[i*2+1] = 0.0f;

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
   //printf("Creating Buffer Object, size: %d bytes, %d Total Vertices.\n", 6*sizeof(GLfloat)*this->numVerts, this->numVerts);
   glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat)*this->numVerts, NULL, GL_STATIC_DRAW);

   // Convenience variables
   GLintptr offset = 0;
   GLuint vertAttribCoord = glGetAttribLocation(shaderProgram, "vertCoord");
   GLuint vertAttribTexUV = glGetAttribLocation(shaderProgram, "vertTexUV");
   GLuint vertAttribColor = glGetAttribLocation(shaderProgram, "vertColor");

   // Load Vertex coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*this->numVerts, this->coordCache);
   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
   // Enable the vertex attribute
   glEnableVertexAttribArray(vertAttribCoord);

   // Update offset to begin storing data in latter part of the buffer
   offset += 2*sizeof(GLfloat)*this->numVerts;

   // Load Vertex coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*this->numVerts, this->texuvCache);
   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(vertAttribTexUV, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
   // Enable the vertex attribute
   glEnableVertexAttribArray(vertAttribTexUV);

   // Update offset to begin storing data in latter part of the buffer
   offset += 2*sizeof(GLfloat)*this->numVerts;

   // Load Vertex coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*4*this->numVerts, this->colorCache);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(vertAttribColor, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (GLintptr*)offset);
   // Enable the vertex attribute
   glEnableVertexAttribArray(vertAttribColor);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   return;
};

void drawCall::updateMVP(GLfloat gx, GLfloat gy, GLfloat sx, GLfloat sy, GLfloat rot, GLfloat w2h) {
   Matrix Ortho;
   Matrix ModelView;

   // Only recompute MVP matrix if there was an actual change in transformations
   if (  this->prevTransforms.ao != rot   ||
         this->prevTransforms.dx != gx    ||
         this->prevTransforms.dy != gy    ||
         this->prevTransforms.sx != sx    ||
         this->prevTransforms.sy != sy    ||
         this->prevTransforms.w2h != w2h  ){
      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, 1.0f*gx, 1.0f*gy, 0.0f );
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
   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &this->MVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

   // Bind Object texture
   glBindTexture(GL_TEXTURE_2D, this->texID);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex Texture UV data is layed out in the buffer
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*this->numVerts));
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (void*)(4*sizeof(GLfloat)*this->numVerts));

   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);

   glDrawArrays(this->drawType, 0, this->numVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   return;
}

/*
 * Updates Buffer object with cache
 */
void drawCall::updateColorCache(void) {

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

   // Initialize offset to begin storing data in latter part of the buffer
   GLintptr offset = 4*sizeof(GLfloat)*this->numVerts;

   // Load Vertex coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*4*this->numVerts, this->colorCache);

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

   // Convenience variables
   GLintptr offset = 0;

   // Load Vertex coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*this->numVerts, this->coordCache);

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

   // Initialize offset to begin storing data in latter part of the buffer
   GLintptr offset = 2*sizeof(GLfloat)*this->numVerts;

   // Load Vertex coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*this->numVerts, this->texuvCache);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   return;
}

void drawCall::setTex(GLuint TexID){
   this->texID = TexID;
   return;
}
#endif
