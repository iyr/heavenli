#ifndef ESUTIL_H
#define ESUTIL_H

#ifdef __cplusplus

extern "C" {
#endif

///
//  Macros
//
#define ESUTIL_API  __cdecl
#define ESCALLBACK  __cdecl

///
// Types
//

typedef struct
{
   float mat[4][4];
} Matrix;

//
///
/// \brief Load a shader, check for compile errors, print error messages to output log
/// \param type Type of shader (GL_VERTEX_SHADER or GL_FRAGMENT_SHADER)
/// \param shaderSrc Shader source string
/// \return A new shader object on success, 0 on failure
//
//GLuint ESUTIL_API esLoadShader ( GLenum type, const char *shaderSrc );

//
///
/// \brief Load a vertex and fragment shader, create a program object, link program.
///        Errors output to log.
/// \param vertShaderSrc Vertex shader source code
/// \param fragShaderSrc Fragment shader source code
/// \return A new program object linked with the vertex/fragment shader pair, 0 on failure
//
//GLuint ESUTIL_API esLoadProgram ( const char *vertShaderSrc, const char *fragShaderSrc );

//
/// \brief Loads a 24-bit TGA image from a file
/// \param fileName Name of the file on disk
/// \param width Width of loaded image in pixels
/// \param height Height of loaded image in pixels
///  \return Pointer to loaded image.  NULL on failure. 
//
//char* ESUTIL_API esLoadTGA ( char *fileName, int *width, int *height );

//
/// \brief multiply matrix specified by result with a scaling matrix and return new matrix in result
/// \param result Specifies the input matrix.  Scaled matrix is returned in result.
/// \param sx, sy, sz Scale factors along the x, y and z axes respectively
//
//void ESUTIL_API esScale(Matrix *result, float sx, float sy, float sz);
void MatrixScale(Matrix *result, float sx, float sy, float sz);

//
/// \brief multiply matrix specified by result with a translation matrix and return new matrix in result
/// \param result Specifies the input matrix.  Translated matrix is returned in result.
/// \param tx, ty, tz Scale factors along the x, y and z axes respectively
//
//void ESUTIL_API esTranslate(Matrix *result, float tx, float ty, float tz);
void MatrixTranslate(Matrix *result, float tx, float ty, float tz);

//
/// \brief multiply matrix specified by result with a rotation matrix and return new matrix in result
/// \param result Specifies the input matrix.  Rotated matrix is returned in result.
/// \param angle Specifies the angle of rotation, in degrees.
/// \param x, y, z Specify the x, y and z coordinates of a vector, respectively
//
//void ESUTIL_API esRotate(Matrix *result, float angle, float x, float y, float z);
void MatrixRotate(Matrix *result, float angle, float x, float y, float z);

//
// \brief multiply matrix specified by result with a perspective matrix and return new matrix in result
/// \param result Specifies the input matrix.  new matrix is returned in result.
/// \param left, right Coordinates for the left and right vertical clipping planes
/// \param bottom, top Coordinates for the bottom and top horizontal clipping planes
/// \param nearZ, farZ Distances to the near and far depth clipping planes.  Both distances must be positive.
//
//void ESUTIL_API esFrustum(Matrix *result, float left, float right, float bottom, float top, float nearZ, float farZ);
void MatrixFrustum(Matrix *result, float left, float right, float bottom, float top, float nearZ, float farZ);

//
/// \brief multiply matrix specified by result with a perspective matrix and return new matrix in result
/// \param result Specifies the input matrix.  new matrix is returned in result.
/// \param fovy Field of view y angle in degrees
/// \param aspect Aspect ratio of screen
/// \param nearZ Near plane distance
/// \param farZ Far plane distance
//
//void ESUTIL_API esPerspective(Matrix *result, float fovy, float aspect, float nearZ, float farZ);
void MatrixPerspective(Matrix *result, float fovy, float aspect, float nearZ, float farZ);

//
/// \brief multiply matrix specified by result with a perspective matrix and return new matrix in result
/// \param result Specifies the input matrix.  new matrix is returned in result.
/// \param left, right Coordinates for the left and right vertical clipping planes
/// \param bottom, top Coordinates for the bottom and top horizontal clipping planes
/// \param nearZ, farZ Distances to the near and far depth clipping planes.  These values are negative if plane is behind the viewer
//
//void ESUTIL_API esOrtho(Matrix *result, float left, float right, float bottom, float top, float nearZ, float farZ);
void MatrixOrtho(Matrix *result, float left, float right, float bottom, float top, float nearZ, float farZ);

//
/// \brief perform the following operation - result matrix = srcA matrix * srcB matrix
/// \param result Returns multiplied matrix
/// \param srcA, srcB Input matrices to be multiplied
//
//void ESUTIL_API esMatrixMultiply(Matrix *result, Matrix *srcA, Matrix *srcB);
void MatrixMultiply(Matrix *result, Matrix *srcA, Matrix *srcB);

//
//// \brief return an indentity matrix 
//// \param result returns identity matrix
//
//void ESUTIL_API esMatrixLoadIdentity(Matrix *result);
void MatrixLoadIdentity(Matrix *result);

#ifdef __cplusplus
}
#endif

#endif // ESUTIL_H
