#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
   float mat[4][4];
} Matrix;

// Perform linear algebra matrix operations for scaling
void MatrixScale(Matrix *Output, float ScaleX, float ScaleY, float ScaleZ);

// Perform linear algebra matrix operations for translation
void MatrixTranslate(Matrix *Output, float tx, float ty, float tz);

// Perform linear algebra matrix operations for rotation
void MatrixRotate(Matrix *Output, float angle, float x, float y, float z);

// Helper function for constructing perspective matrix
void MatrixFrustum(Matrix *Output, float left, float right, float bottom, float top, float nearZ, float farZ);

// Construct perspective projection matrix
void MatrixPerspective(Matrix *Output, float fovy, float aspect, float nearZ, float farZ);

// Construct orthogonal projection matrix
void MatrixOrtho(Matrix *Output, float left, float right, float bottom, float top, float nearZ, float farZ);

// Multiply Matrices
void MatrixMultiply(Matrix *Output, Matrix *InputA, Matrix *InputB);

// Construct Identity Matrix
void MatrixLoadIdentity(Matrix *Output);

#ifdef __cplusplus
}
#endif
