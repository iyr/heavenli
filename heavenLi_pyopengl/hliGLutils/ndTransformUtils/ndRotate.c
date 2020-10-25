/*
 * Returns a transformation matrix for rotating points 
 * about an axis in 3d space
 */
PyArrayObject* ndRotate_hliGLutils(PyObject* self, PyObject* args){
   PyArrayObject* ndArray;
   double angle, x, y, z, mag;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "Odddd",
            &ndArray,
            &angle,
            &x,
            &y,
            &z
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

   mag = sqrt(x*x + y*y + z*z);

   // Only Rotate if Axis of rotation is defined
   if (mag > 0.0) {
      // Get Contiguous C-array of input matrix
      double*  matrix = (double *)PyArray_DATA(ndArray);
   
      // Peform Linear Algebra for Scaling
      double   sinAngle, 
               cosAngle,
               xx, yy, zz, xy, yz, zx, xs, ys, zs,
               oneMinusCos;

      double   rotMat[16];
   
      sinAngle = sin( angle * PI / 180.0 );
      cosAngle = cos( angle * PI / 180.0 );

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
      oneMinusCos = 1.0 - cosAngle;

      rotMat[0*4+0] = (oneMinusCos * xx) + cosAngle;
      rotMat[0*4+1] = (oneMinusCos * xy) - zs;
      rotMat[0*4+2] = (oneMinusCos * zx) + ys;
      rotMat[0*4+3] = 0.0; 

      rotMat[1*4+0] = (oneMinusCos * xy) + zs;
      rotMat[1*4+1] = (oneMinusCos * yy) + cosAngle;
      rotMat[1*4+2] = (oneMinusCos * yz) - xs;
      rotMat[1*4+3] = 0.0;

      rotMat[2*4+0] = (oneMinusCos * zx) - ys;
      rotMat[2*4+1] = (oneMinusCos * yz) + xs;
      rotMat[2*4+2] = (oneMinusCos * zz) + cosAngle;
      rotMat[2*4+3] = 0.0; 

      rotMat[3*4+0] = 0.0;
      rotMat[3*4+1] = 0.0;
      rotMat[3*4+2] = 0.0;
      rotMat[3*4+3] = 1.0;

      multiplyMatrix( matrix, rotMat, matrix );
   }
   // Update reference count for output matrix
   Py_INCREF(ndArray);

   return ndArray;
}
