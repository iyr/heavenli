#ifndef attribCacheClass
#define attribCacheClass

/*
 * Inherits from vertAttribClass to include functionality 
 * for caching vertex data for a given attribute (WIP)
 */
class attribCache: public vertAttribMetadata {
   public:
      attribCache(void);
      attribCache(vertAttribMetadata* VAMD);
      attribCache(string LocationString, GLubyte VectorSize, GLubyte VectorOffset, GLuint attribIndex);
      attribCache(const GLvoid* buffer, GLuint numVerts, string LocationString, GLubyte VectorSize, GLubyte VectorOffset, GLuint attribIndex);
      virtual ~attribCache(void);

      // Valid vertex datatypes:
      // GL_HALF_FLOAT_OES
      // GL_FLOAT
      // GL_BYTE
      // GL_UNSIGNED_BYTE
      // GL_SHORT
      // GL_UNSIGNED_SHORT
      // GL_FIXED
      void setGLtype(GLenum type);  // Change Attrib type, erase cache
      GLenum getGLtype(void);       // Get datatype of the cache

      // This quantity should be the same for
      // all caches belonging to the same drawcall
      GLuint getNumVerts(void);

      // Free allocated memory, mark cache pointer to null
      void eraseCache(void);

      // update cache data, resizing if necessary
      void writeCache(const GLvoid* buffer, size_t numVerts);

      // get pointer to cache
      void* getCachePtr(void);

      void copy(attribCache &VAC);

   private:
      GLenum   datatype = GL_FLOAT;
      GLuint   numVerts = 0;
      GLvoid*  cache    = NULL;
};

void attribCache::copy(attribCache &VAC){
   printf("copying attribute...\n");
   this->setAttribIndex    (VAC.getAttribIndex());
   this->setLocationString (VAC.getLocationString());
   this->setVectorSize     (VAC.getVectorSize());
   this->setVectorOffset   (VAC.getVectorOffset());
   this->setGLtype         (VAC.getGLtype());
   this->writeCache        (VAC.getCachePtr(), VAC.getNumVerts()*(GLuint)this->vectorSize);
   return;
};

attribCache::attribCache(void){return;};

/*
 * Initialize cache based on attrib metadata
 * Retrieved from shader class
 */
attribCache::attribCache(vertAttribMetadata* VAMD){
   this->vectorSize     = VAMD->getVectorSize();
   this->vectorOffset   = VAMD->getVectorOffset();
   this->setLocationString(VAMD->getLocationString());
   return;
};

/*
 * Initialize cache from meta-data parameters
 */
attribCache::attribCache(string LocationString, GLubyte VectorSize, GLubyte VectorOffset, GLuint attribIndex){
   this->setLocationString(LocationString);
   this->setVectorSize(VectorSize);
   this->setVectorOffset(VectorOffset);
   this->setAttribIndex(attribIndex);
   return;
};

attribCache::attribCache(const GLvoid* buffer, GLuint numVerts, string LocationString, GLubyte VectorSize, GLubyte VectorOffset, GLuint attribIndex){
   this->setLocationString (LocationString);
   this->setVectorSize     (VectorSize);
   this->setVectorOffset   (VectorOffset);
   this->setAttribIndex    (attribIndex);
   this->writeCache        (buffer, numVerts*(GLuint)VectorSize);
   return;
};

// !!Virtual deconstructor frees allocated memory
attribCache::~attribCache(void){
   this->eraseCache();
   return;
};

// Unnecessary comment to explain one-liner
GLuint   attribCache::getNumVerts(void){return this->numVerts;};
void*    attribCache::getCachePtr(void){return this->cache;};

/*
 * Set datatype, erase old cache
 */
void attribCache::setGLtype(GLenum type){
   if (this->datatype != type) {
      printf("updating datatype, erasing old cache\n");
      this->eraseCache();
      this->datatype = type;
   }
   return;
};

GLenum attribCache::getGLtype(void){return this->datatype;};

/*
 * Clear cache, mark cache pointer null
 */
void attribCache::eraseCache(void){
   // Check cache isn't already set
   if (this->cache != NULL) free(this->cache);

   this->cache    = NULL;
   this->numVerts = 0;
   return;
};

/*
 * NOT TYPE-SAFE, FEED BUFFERS OF SAME TYPE AS CACHE
 * Update Cache, resize if necessary
 */
void attribCache::writeCache(const GLvoid* buffer, size_t numElements) {
   GLuint vectorSize = 4;
   if ((this->vectorSize < 1) || (this->vectorSize > 4)) {
      printf("Invalid vector size for writing buffer elements, (must be 1-4)\n");
      printf("vector size: %d\n", this->vectorSize);
      return;
   } else {
      vectorSize = this->vectorSize;
   }

   // Erase old cache if diff in number of elements (safety check)
   if (  (GLuint)numElements  != this->numVerts*vectorSize  &&
         this->cache          != NULL                             ){
      printf("Reallocating cache size: \n\
            prev (numVerts*VecSize): %d\n\
            new  (numElements): %d\n",
            this->numVerts*vectorSize,
            (GLuint)numElements);
      this->cache = realloc(this->cache, numElements*GLsizeof(this->datatype));
   }

   this->numVerts = (GLuint)numElements/this->vectorSize;

   // Allocate memory if not allocated
   if (this->cache == NULL) {
      this->cache = malloc(numElements*GLsizeof(this->datatype));
      /*
      printf("malloc'ing %d bytes for %s attrib (%dD): (%d verts, %d elements)\n",
            (unsigned int)numElements*GLsizeof(this->datatype),
            this->locationString.c_str(),
            vectorSize,
            this->numVerts,
            (GLuint)numElements
            );
            */
   }

   // copy data from buffer to cache
   memcpy(this->cache, buffer, numElements*GLsizeof(this->datatype));

   return;
};

#endif
