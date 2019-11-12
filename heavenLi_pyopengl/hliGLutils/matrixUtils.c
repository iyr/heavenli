///
//  Includes
//
#include "matrixUtils.h"
#include <math.h>

#define PI 3.14159265358979f

void MatrixScale(Matrix *Output, float ScaleX, float ScaleY, float ScaleZ)
{
   // Peform Linear Algebra for Scaling
   Output->mat[0][0] *= ScaleX;
   Output->mat[0][1] *= ScaleX;
   Output->mat[0][2] *= ScaleX;
   Output->mat[0][3] *= ScaleX;

   Output->mat[1][0] *= ScaleY;
   Output->mat[1][1] *= ScaleY;
   Output->mat[1][2] *= ScaleY;
   Output->mat[1][3] *= ScaleY;

   Output->mat[2][0] *= ScaleZ;
   Output->mat[2][1] *= ScaleZ;
   Output->mat[2][2] *= ScaleZ;
   Output->mat[2][3] *= ScaleZ;
}

void MatrixTranslate(Matrix *Output, float tx, float ty, float tz)
{
   Output->mat[3][0] += (Output->mat[0][0] * tx + Output->mat[1][0] * ty + Output->mat[2][0] * tz);
   Output->mat[3][1] += (Output->mat[0][1] * tx + Output->mat[1][1] * ty + Output->mat[2][1] * tz);
   Output->mat[3][2] += (Output->mat[0][2] * tx + Output->mat[1][2] * ty + Output->mat[2][2] * tz);
   Output->mat[3][3] += (Output->mat[0][3] * tx + Output->mat[1][3] * ty + Output->mat[2][3] * tz);
}

void MatrixRotate(Matrix *Output, float angle, float x, float y, float z)
{
   float mag = sqrtf(x * x + y * y + z * z);
      
   // Only Rotate if Axis of rotation is defined
   if ( mag > 0.0f )
   {
      float sinAngle, cosAngle;
      float xx, yy, zz, xy, yz, zx, xs, ys, zs;
      float oneMinusCos;
      Matrix rotMat;
   
      sinAngle = sinf ( angle * PI / 180.0f );
      cosAngle = cosf ( angle * PI / 180.0f );

      // Normalize Values
      x /= mag;
      y /= mag;
      z /= mag;

      // Perform Linear Algebra for Rotation
      xx = x * x;
      yy = y * y;
      zz = z * z;
      xy = x * y;
      yz = y * z;
      zx = z * x;
      xs = x * sinAngle;
      ys = y * sinAngle;
      zs = z * sinAngle;
      oneMinusCos = 1.0f - cosAngle;

      rotMat.mat[0][0] = (oneMinusCos * xx) + cosAngle;
      rotMat.mat[0][1] = (oneMinusCos * xy) - zs;
      rotMat.mat[0][2] = (oneMinusCos * zx) + ys;
      rotMat.mat[0][3] = 0.0f; 

      rotMat.mat[1][0] = (oneMinusCos * xy) + zs;
      rotMat.mat[1][1] = (oneMinusCos * yy) + cosAngle;
      rotMat.mat[1][2] = (oneMinusCos * yz) - xs;
      rotMat.mat[1][3] = 0.0f;

      rotMat.mat[2][0] = (oneMinusCos * zx) - ys;
      rotMat.mat[2][1] = (oneMinusCos * yz) + xs;
      rotMat.mat[2][2] = (oneMinusCos * zz) + cosAngle;
      rotMat.mat[2][3] = 0.0f; 

      rotMat.mat[3][0] = 0.0f;
      rotMat.mat[3][1] = 0.0f;
      rotMat.mat[3][2] = 0.0f;
      rotMat.mat[3][3] = 1.0f;

      MatrixMultiply( Output, &rotMat, Output );
   }
}

// Perspective Matrix Helper
void MatrixFrustum(Matrix *Output, float left, float right, float bottom, float top, float nearZ, float farZ)
{
   float   deltaX = right - left;
   float   deltaY = top - bottom;
   float   deltaZ = farZ - nearZ;
   Matrix  frust;

   if ( (nearZ <= 0.0f) || (farZ <= 0.0f) ||
      (deltaX <= 0.0f) || (deltaY <= 0.0f) || (deltaZ <= 0.0f) )
      return;

   frust.mat[0][0] = 2.0f * nearZ / deltaX;
   frust.mat[0][1] = frust.mat[0][2] = frust.mat[0][3] = 0.0f;

   frust.mat[1][1] = 2.0f * nearZ / deltaY;
   frust.mat[1][0] = frust.mat[1][2] = frust.mat[1][3] = 0.0f;

   frust.mat[2][0] = (right + left) / deltaX;
   frust.mat[2][1] = (top + bottom) / deltaY;
   frust.mat[2][2] = -(nearZ + farZ) / deltaZ;
   frust.mat[2][3] = -1.0f;

   frust.mat[3][2] = -2.0f * nearZ * farZ / deltaZ;
   frust.mat[3][0] = frust.mat[3][1] = frust.mat[3][3] = 0.0f;

   MatrixMultiply(Output, &frust, Output);
}

// Construct Perspective Matrix
void MatrixPerspective(Matrix *Output, float fovy, float aspect, float nearZ, float farZ)
{
   float frustumW, frustumH;
   
   frustumH = tanf( fovy / 360.0f * PI ) * nearZ;
   frustumW = frustumH * aspect;

   MatrixFrustum( Output, -frustumW, frustumW, -frustumH, frustumH, nearZ, farZ );
}

// Construct Orthogonal Projection Matrix
void MatrixOrtho(Matrix *Output, float left, float right, float bottom, float top, float nearZ, float farZ)
{
    float   deltaX = right - left;
    float   deltaY = top - bottom;
    float   deltaZ = farZ - nearZ;
    Matrix  ortho;

    // Sanity Check
    if ( (deltaX == 0.0f) || (deltaY == 0.0f) || (deltaZ == 0.0f) )
        return;

    MatrixLoadIdentity(&ortho);

    ortho.mat[0][0] =  2.0f / deltaX;
    ortho.mat[1][1] =  2.0f / deltaY;
    ortho.mat[2][2] = -2.0f / deltaZ;
    ortho.mat[3][0] = -(right + left) / deltaX;
    ortho.mat[3][1] = -(top + bottom) / deltaY;
    ortho.mat[3][2] = -(nearZ + farZ) / deltaZ;

    MatrixMultiply(Output, &ortho, Output);
}

void MatrixMultiply(Matrix *Output, Matrix *InputA, Matrix *InputB)
{
   Matrix  tmp;

   // Perform Row Dot Products
	for (int i = 0; i < 4; i++)
	{
		tmp.mat[i][0] =	
         (InputA->mat[i][0] * InputB->mat[0][0]) +
			(InputA->mat[i][1] * InputB->mat[1][0]) +
			(InputA->mat[i][2] * InputB->mat[2][0]) +
			(InputA->mat[i][3] * InputB->mat[3][0]) ;

		tmp.mat[i][1] =	
         (InputA->mat[i][0] * InputB->mat[0][1]) + 
			(InputA->mat[i][1] * InputB->mat[1][1]) +
			(InputA->mat[i][2] * InputB->mat[2][1]) +
			(InputA->mat[i][3] * InputB->mat[3][1]) ;

		tmp.mat[i][2] =	
         (InputA->mat[i][0] * InputB->mat[0][2]) + 
			(InputA->mat[i][1] * InputB->mat[1][2]) +
			(InputA->mat[i][2] * InputB->mat[2][2]) +
			(InputA->mat[i][3] * InputB->mat[3][2]) ;

		tmp.mat[i][3] =	
         (InputA->mat[i][0] * InputB->mat[0][3]) + 
			(InputA->mat[i][1] * InputB->mat[1][3]) +
			(InputA->mat[i][2] * InputB->mat[2][3]) +
			(InputA->mat[i][3] * InputB->mat[3][3]) ;
	}

   memcpy(Output, &tmp, sizeof(Matrix));
}

void MatrixLoadIdentity(Matrix *Output)
{
   // Initialize Empty 4x4 Matrix of zeroes
   memset(Output, 0x0, sizeof(Matrix));

   // Set Diagonals to 1.0f to make identitf matrix
   Output->mat[0][0] = 1.0f;
   Output->mat[1][1] = 1.0f;
   Output->mat[2][2] = 1.0f;
   Output->mat[3][3] = 1.0f;
}

