/*
 * Returns a transformation matrix for scaling points in 3d space
 */
PyArrayObject* ndScale_hliGLutils(PyObject* self, PyObject* args){
   PyArrayObject* ndArray;
   double scaleX, scaleY, scaleZ;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "Oddd",
            &ndArray,
            &scaleX,
            &scaleY,
            &scaleZ
            ))
   {
      printf("Error parsing inputs\n");
      // Build and return 4x4 matrix of zeros of type double
      npy_intp dims[2] = {4, 4};
      ndArray = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
      return ndArray;
   }

   // Safety check, reject and return matrix if invalid
   if (checkTransformMatrixValidity(ndArray) == 0){
      printf("Returning input matrix unchanged\n");
      Py_INCREF(ndArray);
      return ndArray;
   }

   // Cast Array to double if not already double
   if (PyArray_TYPE(ndArray) != NPY_DOUBLE) {
      ndArray = (PyArrayObject *)PyArray_FROM_OT((PyObject *)ndArray, NPY_DOUBLE);
   }

   // Get Contiguous C-array of input matrix
   double*  matrix = (double *)PyArray_DATA(ndArray);
   
   // Peform Linear Algebra for Scaling
   matrix[0*4+0] *= scaleX;
   matrix[0*4+1] *= scaleX;
   matrix[0*4+2] *= scaleX;
   matrix[0*4+3] *= scaleX;

   matrix[1*4+0] *= scaleY;
   matrix[1*4+1] *= scaleY;
   matrix[1*4+2] *= scaleY;
   matrix[1*4+3] *= scaleY;

   matrix[2*4+0] *= scaleZ;
   matrix[2*4+1] *= scaleZ;
   matrix[2*4+2] *= scaleZ;
   matrix[2*4+3] *= scaleZ;

   // Update reference count for output matrix
   Py_INCREF(ndArray);

   return ndArray;
}
