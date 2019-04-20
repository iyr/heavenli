///
//  Includes
//
#include "matrixUtils.h"
#include <math.h>

#define PI 3.1415926535897932384626433832795f

void MatrixScale(Matrix *OutputMatrix, float sx, float sy, float sz)
{
    OutputMatrix->m[0][0] *= sx;
    OutputMatrix->m[0][1] *= sx;
    OutputMatrix->m[0][2] *= sx;
    OutputMatrix->m[0][3] *= sx;

    OutputMatrix->m[1][0] *= sy;
    OutputMatrix->m[1][1] *= sy;
    OutputMatrix->m[1][2] *= sy;
    OutputMatrix->m[1][3] *= sy;

    OutputMatrix->m[2][0] *= sz;
    OutputMatrix->m[2][1] *= sz;
    OutputMatrix->m[2][2] *= sz;
    OutputMatrix->m[2][3] *= sz;
}

void MatrixTranslate(Matrix *OutputMatrix, float tx, float ty, float tz)
{
    OutputMatrix->m[3][0] += (OutputMatrix->m[0][0] * tx + OutputMatrix->m[1][0] * ty + OutputMatrix->m[2][0] * tz);
    OutputMatrix->m[3][1] += (OutputMatrix->m[0][1] * tx + OutputMatrix->m[1][1] * ty + OutputMatrix->m[2][1] * tz);
    OutputMatrix->m[3][2] += (OutputMatrix->m[0][2] * tx + OutputMatrix->m[1][2] * ty + OutputMatrix->m[2][2] * tz);
    OutputMatrix->m[3][3] += (OutputMatrix->m[0][3] * tx + OutputMatrix->m[1][3] * ty + OutputMatrix->m[2][3] * tz);
}

void MatrixRotate(Matrix *OutputMatrix, float angle, float x, float y, float z)
{
   float sinAngle, cosAngle;
   float mag = sqrtf(x * x + y * y + z * z);
      
   sinAngle = sinf ( angle * PI / 180.0f );
   cosAngle = cosf ( angle * PI / 180.0f );
   if ( mag > 0.0f )
   {
      float xx, yy, zz, xy, yz, zx, xs, ys, zs;
      float oneMinusCos;
      Matrix rotMat;
   
      x /= mag;
      y /= mag;
      z /= mag;

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

      rotMat.m[0][0] = (oneMinusCos * xx) + cosAngle;
      rotMat.m[0][1] = (oneMinusCos * xy) - zs;
      rotMat.m[0][2] = (oneMinusCos * zx) + ys;
      rotMat.m[0][3] = 0.0F; 

      rotMat.m[1][0] = (oneMinusCos * xy) + zs;
      rotMat.m[1][1] = (oneMinusCos * yy) + cosAngle;
      rotMat.m[1][2] = (oneMinusCos * yz) - xs;
      rotMat.m[1][3] = 0.0F;

      rotMat.m[2][0] = (oneMinusCos * zx) - ys;
      rotMat.m[2][1] = (oneMinusCos * yz) + xs;
      rotMat.m[2][2] = (oneMinusCos * zz) + cosAngle;
      rotMat.m[2][3] = 0.0F; 

      rotMat.m[3][0] = 0.0F;
      rotMat.m[3][1] = 0.0F;
      rotMat.m[3][2] = 0.0F;
      rotMat.m[3][3] = 1.0F;

      MatrixMultiply( OutputMatrix, &rotMat, OutputMatrix );
   }
}

void MatrixFrustum(Matrix *OutputMatrix, float left, float right, float bottom, float top, float nearZ, float farZ)
{
    float       deltaX = right - left;
    float       deltaY = top - bottom;
    float       deltaZ = farZ - nearZ;
    Matrix    frust;

    if ( (nearZ <= 0.0f) || (farZ <= 0.0f) ||
         (deltaX <= 0.0f) || (deltaY <= 0.0f) || (deltaZ <= 0.0f) )
         return;

    frust.m[0][0] = 2.0f * nearZ / deltaX;
    frust.m[0][1] = frust.m[0][2] = frust.m[0][3] = 0.0f;

    frust.m[1][1] = 2.0f * nearZ / deltaY;
    frust.m[1][0] = frust.m[1][2] = frust.m[1][3] = 0.0f;

    frust.m[2][0] = (right + left) / deltaX;
    frust.m[2][1] = (top + bottom) / deltaY;
    frust.m[2][2] = -(nearZ + farZ) / deltaZ;
    frust.m[2][3] = -1.0f;

    frust.m[3][2] = -2.0f * nearZ * farZ / deltaZ;
    frust.m[3][0] = frust.m[3][1] = frust.m[3][3] = 0.0f;

    MatrixMultiply(OutputMatrix, &frust, OutputMatrix);
}

void MatrixPerspective(Matrix *OutputMatrix, float fovy, float aspect, float nearZ, float farZ)
{
   float frustumW, frustumH;
   
   frustumH = tanf( fovy / 360.0f * PI ) * nearZ;
   frustumW = frustumH * aspect;

   MatrixFrustum( OutputMatrix, -frustumW, frustumW, -frustumH, frustumH, nearZ, farZ );
}

void MatrixOrtho(Matrix *OutputMatrix, float left, float right, float bottom, float top, float nearZ, float farZ)
{
    float       deltaX = right - left;
    float       deltaY = top - bottom;
    float       deltaZ = farZ - nearZ;
    Matrix    ortho;

    if ( (deltaX == 0.0f) || (deltaY == 0.0f) || (deltaZ == 0.0f) )
        return;

    MatrixLoadIdentity(&ortho);
    ortho.m[0][0] = 2.0f / deltaX;
    ortho.m[3][0] = -(right + left) / deltaX;
    ortho.m[1][1] = 2.0f / deltaY;
    ortho.m[3][1] = -(top + bottom) / deltaY;
    ortho.m[2][2] = -2.0f / deltaZ;
    ortho.m[3][2] = -(nearZ + farZ) / deltaZ;

    MatrixMultiply(OutputMatrix, &ortho, OutputMatrix);
}

void MatrixMultiply(Matrix *OutputMatrix, Matrix *srcA, Matrix *srcB)
{
    Matrix    tmp;
    int         i;

	for (i=0; i<4; i++)
	{
		tmp.m[i][0] =	(srcA->m[i][0] * srcB->m[0][0]) +
						(srcA->m[i][1] * srcB->m[1][0]) +
						(srcA->m[i][2] * srcB->m[2][0]) +
						(srcA->m[i][3] * srcB->m[3][0]) ;

		tmp.m[i][1] =	(srcA->m[i][0] * srcB->m[0][1]) + 
						(srcA->m[i][1] * srcB->m[1][1]) +
						(srcA->m[i][2] * srcB->m[2][1]) +
						(srcA->m[i][3] * srcB->m[3][1]) ;

		tmp.m[i][2] =	(srcA->m[i][0] * srcB->m[0][2]) + 
						(srcA->m[i][1] * srcB->m[1][2]) +
						(srcA->m[i][2] * srcB->m[2][2]) +
						(srcA->m[i][3] * srcB->m[3][2]) ;

		tmp.m[i][3] =	(srcA->m[i][0] * srcB->m[0][3]) + 
						(srcA->m[i][1] * srcB->m[1][3]) +
						(srcA->m[i][2] * srcB->m[2][3]) +
						(srcA->m[i][3] * srcB->m[3][3]) ;
	}
    memcpy(OutputMatrix, &tmp, sizeof(Matrix));
}

void MatrixLoadIdentity(Matrix *OutputMatrix)
{
    memset(OutputMatrix, 0x0, sizeof(Matrix));
    OutputMatrix->m[0][0] = 1.0f;
    OutputMatrix->m[1][1] = 1.0f;
    OutputMatrix->m[2][2] = 1.0f;
    OutputMatrix->m[3][3] = 1.0f;
}

