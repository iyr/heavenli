using namespace std;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#define PI 3.14159265358979f

// Multiply A*B and store product in output.
// OUTPUT MUST BE PRE-ALLOCATED
void multiplyMatrix(double *output, double *InputA, double *InputB){

   double tmp[16];
   // Perform Row Dot Products
	for (int i = 0; i < 4; i++)
	{
		tmp[i*4+0] =	
         (InputA[i*4+0] * InputB[0*4+0]) +
			(InputA[i*4+1] * InputB[1*4+0]) +
			(InputA[i*4+2] * InputB[2*4+0]) +
			(InputA[i*4+3] * InputB[3*4+0]) ;

		tmp[i*4+1] =	
         (InputA[i*4+0] * InputB[0*4+1]) + 
			(InputA[i*4+1] * InputB[1*4+1]) +
			(InputA[i*4+2] * InputB[2*4+1]) +
			(InputA[i*4+3] * InputB[3*4+1]) ;

		tmp[i*4+2] =	
         (InputA[i*4+0] * InputB[0*4+2]) + 
			(InputA[i*4+1] * InputB[1*4+2]) +
			(InputA[i*4+2] * InputB[2*4+2]) +
			(InputA[i*4+3] * InputB[3*4+2]) ;

		tmp[i*4+3] =	
         (InputA[i*4+0] * InputB[0*4+3]) + 
			(InputA[i*4+1] * InputB[1*4+3]) +
			(InputA[i*4+2] * InputB[2*4+3]) +
			(InputA[i*4+3] * InputB[3*4+3]) ;
	}

   for (unsigned int i = 0; i < 16; i++) output[i] = tmp[i];
   return;
}

// Returns 1 if matrix is valid, 0 otherwise.
unsigned int checkTransformMatrixValidity(PyArrayObject* arr){

   // Get numpy typenum for input array
   int numDims = PyArray_NDIM(arr);
   int numElts = (int)PyArray_Size((PyObject *)arr);

   // Safety check for matrix validity
   if (  numDims != 2  ||
         numElts != 16 ){
      printf("invalid input matrix: \n");
      if (numDims != 2) printf("matrix isn't 2-dimensional: %d\n", numDims);
      if (numElts != 16) printf("matrix has invalid number of elements (must be 16 for 4x4 transform matrix): %d\n", numElts);

      return 0;
   } else 
      return 1;
}

// Returns a pointer to a 4x4 identity matrix
void buildIdentity(double* matrix){
   // Initialize Empty 4x4 Matrix of zeroes
   memset(matrix, 0x0, sizeof(double)*16);

   // Set Diagonals to 1.0 to make identity matrix
   matrix[0*4+0] = 1.0;
   matrix[1*4+1] = 1.0;
   matrix[2*4+2] = 1.0;
   matrix[3*4+3] = 1.0;

   return;
}

#include "ndTransformUtils/ndScale.c"
#include "ndTransformUtils/ndRotate.c"
#include "ndTransformUtils/ndTranslate.c"
#include "ndTransformUtils/ndPerspective.c"
#include "ndTransformUtils/ndOrtho.c"

#include "ndTransformUtils/ndPrintMatrix.c"
