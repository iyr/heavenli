#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
   std::string locationString;   // Name of the attribute in the shader

   GLubyte  vectorSize  = 0,     // Number of vector components (1-4)
            vectorOffset= 0;     // 
   GLuint   attribIndex = 0;     // May get deprecated

   //GLfloat* clientCache = NULL;

} vertAttribMetadata;

#ifdef __cplusplus
}
#endif
