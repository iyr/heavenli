/*
 * Generate and return a 4x4 Orthogonal projection matrix
 */
PyArrayObject* ndOrtho_hliGLutils(PyObject* self, PyObject* args){
   double left, right, bottom, top, nearZ, farZ, deltaX, deltaY, deltaZ;
   PyArrayObject* ndArray;    // Output Matrix
   npy_intp dims[2] = {4, 4}; // Used to define output matrix dimensions

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "dddddd",
            //"Odddddd",
            //&ndArray,
            &left,
            &right,
            &bottom,
            &top,
            &nearZ,
            &farZ
            ))
   {
      printf("Error parsing inputs\n");
      // Build and return 4x4 matrix of zeros of type double
      ndArray = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
      return ndArray;
   }

   // Convenience variables
   deltaX = right - left;
   deltaY = top - bottom;
   deltaZ = farZ - nearZ;

   // Sanity Check, prevent division by zero
   if ( (deltaX == 0.0) || (deltaY == 0.0) || (deltaZ == 0.0) ) {
      printf("invalid dimensions, (zero delta on an axis), returning matrix unchanged\n");
      ndArray = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
      return ndArray;
   }

   // Allocate memory and set it to identity matrix
   double *ortho = (double *)malloc(sizeof(double)*16);;
   buildIdentity(ortho);
   
   // Set orthogonal matrix parameters
   ortho[0*4+0] =  2.0 / deltaX;
   ortho[1*4+1] =  2.0 / deltaY;
   ortho[2*4+2] = -2.0 / deltaZ;
   ortho[3*4+0] = -(right + left) / deltaX;
   ortho[3*4+1] = -(top + bottom) / deltaY;
   ortho[3*4+2] = -(nearZ + farZ) / deltaZ;

   // Build output PyArrayObject from c-array
   ndArray = (PyArrayObject *)PyArray_SimpleNewFromData(
         2,             // Number of dimensions
         dims,          // Array of size of each dimension
         NPY_DOUBLE,    // ndArray type
         (void*)ortho   // pointer to >>contiguous<< input data
         );
   return ndArray;
}
