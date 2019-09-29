#ifndef drawCallClass
#define drawCallClass

using namespace std;

/*
 * Implements a helper class that wraps buffer object and transformation matrix functions
 * necessary for each and every draw-call
 */
class drawCall {
   public:
      GLfloat*    coordCache;    // Array of Vertex coordinate data (X, Y). May also contain texture coordinates (X, Y, tX, tY)
      GLfloat*    colorCache;    // Array of Vertex color data (R, G, B, A)

      GLuint      numVerts;      // number of Vertices (not length of cache arrays)
      GLuint      VBO;           // Buffer Object for OpenGL to store data in graphics memory
      Matrix      MVP;           // Model-View-Projection Matrix for storing transformations
      GLboolean   colorsChanged; // Determines whether color-buffer should be updated

      drawCall(void);
      drawCall(unsigned int numColors);
      ~drawCall(void);
      void buildCache(GLuint numVerts, std::vector<GLfloat> &verts, std::vector<GLfloat> &colrs);
      void updateMVP(GLfloat gx, GLfloat gy, GLfloat sx, GLfloat sy, GLfloat rot, GLfloat w2h);
      void draw(void);
      void updateCoordCache(void);
      void updateColorCache(void);
      void setNumColors(unsigned int numColors);
      void setColorQuartet(unsigned int setIndex, GLfloat* quartet);

   private:
      Params      prevTransforms;   // Useful for know if MVP should be recomputed

      GLuint      numColors;        // Number of colorSet quartets (R, G, B, A) to manage (min: 1, typ: 2)
      GLfloat*    colorQuartets;    // Continuous array to store colorSet quartets

      GLboolean   firstRun;         // Determines if function is running for the first time (for VBO initialization)
};

drawCall::drawCall(void) {
   printf("Building drawCall Object...\n");
   // Initialize Caches
   this->coordCache     = NULL;
   this->colorCache     = NULL;

   this->firstRun       = GL_TRUE;   
   this->numVerts       = 0;
   this->numColors      = 1;

   this->colorQuartets  = new GLfloat[this->numColors*4];

   this->colorsChanged  = GL_FALSE;
   return;
};

drawCall::drawCall(unsigned int numColors) {
   printf("Building drawCall Object...\n");
   // Initialize Caches
   this->coordCache     = NULL;
   this->colorCache     = NULL;

   this->firstRun       = GL_TRUE;   
   this->numVerts       = 0;
   this->numColors      = numColors;

   this->colorQuartets  = new GLfloat[this->numColors*4];

   this->colorsChanged  = GL_FALSE;
   return;
};


drawCall::~drawCall(void){
   // Deallocate caches
   delete [] this->coordCache;
   delete [] this->colorCache;
   //glDeleteBuffers(1, &this->VBO);
   return;
};

void drawCall::setNumColors(unsigned int numColors) {
   if (this->numColors != numColors) {
      this->numColors = numColors;
      
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
void drawCall::buildCache(GLuint numVerts, std::vector<GLfloat> &verts, std::vector<GLfloat> &colrs) {
   this->numVerts = numVerts;

   // Safely allocate cache arrays
   if (this->coordCache == NULL) {
      this->coordCache = new GLfloat[this->numVerts*2];
   } else {
      delete [] this->coordCache;
      this->coordCache = new GLfloat[this->numVerts*2];
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

      this->colorCache[i*4+0] = colrs[i*4+0];
      this->colorCache[i*4+1] = colrs[i*4+1];
      this->colorCache[i*4+2] = colrs[i*4+2];
      this->colorCache[i*4+3] = colrs[i*4+3];
   }

   // Create buffer object if one does not exist, otherwise, delete and make a new one
   // This cannot be done in the constructor for global object instances because OpenGL
   // must be initialized before any GL functions are called. 
   if (this->firstRun == GL_TRUE) {
      this->firstRun = GL_FALSE;
      glGenBuffers(1, &this->VBO);
   } else {
      glDeleteBuffers(1, &this->VBO);
      glGenBuffers(1, &this->VBO);
   }

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

   // Allocate space to hold all vertex coordinate and color data
   printf("Creating Buffer Object, size: %d bytes, %d Total Vertices.\n", 6*sizeof(GLfloat)*this->numVerts, this->numVerts);
   glBufferData(GL_ARRAY_BUFFER, 6*sizeof(GLfloat)*this->numVerts, NULL, GL_STATIC_DRAW);

   // Convenience variables
   GLintptr offset = 0;
   GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
   GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

   // Load Vertex coordinate data into VBO
   glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*this->numVerts, this->coordCache);
   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
   // Enable the vertex attribute
   glEnableVertexAttribArray(vertAttribCoord);

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

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*this->numVerts));

   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);

   glDrawArrays(GL_TRIANGLE_STRIP, 0, this->numVerts);

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
   GLuint offset = 2*sizeof(GLfloat)*this->numVerts;

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

#endif
