#ifndef vertAttribMetadataClass
#define vertAttribMetadataClass

class vertAttribMetadata {
   public:
      string locationString = "UNNAMED VERTEX ATTRIBUTE";   // Name of the attribute in the shader

      // (De)constructors
      vertAttribMetadata(void);
      virtual ~vertAttribMetadata(void);

      // Setters
      void setLocationString  (string newLocString);
      void setVectorSize      (GLubyte newVecSize);
      void setVectorOffset    (GLubyte newOffset);
      void setAttribIndex     (GLuint  newIndex);

      // Getters
      string   getLocationString (void);
      GLubyte  getVectorSize     (void);
      GLubyte  getVectorOffset   (void);
      GLuint   getAttribIndex    (void);
     
   //private:
   protected:
      GLubyte  vectorSize  = 0,  // Number of vector components (1-4)
               vectorOffset= 0;  // May get deprecated
      GLuint   attribIndex = 0;  // May get deprecated
};

vertAttribMetadata::vertAttribMetadata(void){
   return;
};

vertAttribMetadata::~vertAttribMetadata(void){
   return;
};

void vertAttribMetadata::setLocationString(string newLocString){
   this->locationString = newLocString;
   return;
};

void vertAttribMetadata::setVectorSize(GLubyte newVecSize){
   //printf("new vector size: %d\n", (GLuint)newVecSize);
   this->vectorSize = newVecSize;
   return;
};

void vertAttribMetadata::setVectorOffset(GLubyte newOffset){
   this->vectorOffset = newOffset;
   return;
};

void vertAttribMetadata::setAttribIndex(GLuint newIndex){
   this->attribIndex = newIndex;
   return;
};

string vertAttribMetadata::getLocationString(void){
   return this->locationString;
};

GLubyte vertAttribMetadata::getVectorSize(void){
   return this->vectorSize;
};

GLubyte vertAttribMetadata::getVectorOffset(void){
   return this->vectorOffset;
};

GLuint vertAttribMetadata::getAttribIndex(void){
   return this->attribIndex;
};

#endif
