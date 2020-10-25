/*
 * Returns a transformation matrix for linearly moving points in 3d space
 */
PyArrayObject* ndTranslate_hliGLutils(PyObject* self, PyObject* args){
   PyArrayObject* ndArray;
   double tx, ty, tz;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "Oddd",
            &ndArray,
            &tx,
            &ty,
            &tz
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
   matrix[3*4+0] += (matrix[0*4+0] * tx + matrix[1*4+0] * ty + matrix[2*4+0] * tz);
   matrix[3*4+1] += (matrix[0*4+1] * tx + matrix[1*4+1] * ty + matrix[2*4+1] * tz);
   matrix[3*4+2] += (matrix[0*4+2] * tx + matrix[1*4+2] * ty + matrix[2*4+2] * tz);
   matrix[3*4+3] += (matrix[0*4+3] * tx + matrix[1*4+3] * ty + matrix[2*4+3] * tz);

   // Update reference count for output matrix
   Py_INCREF(ndArray);

   return ndArray;
}
