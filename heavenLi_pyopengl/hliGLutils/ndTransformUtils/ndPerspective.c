/*
 * Generate and return a 4x4 perspective projection matrix
 */
PyArrayObject* ndPerspective_hliGLutils(PyObject* self, PyObject* args){
   double   fovy,       // Horizonal field-of-view
            aspect,     // Vertical field-of-view
            nearZ,      // how close tris can get before being clipped
            farZ,       // how far tris can get before being clipped
            left,       // convenience variable
            right,      // convenience variable
            top,        // convenience variable
            bottom,     // convenience variable
            deltaX,     // convenience variable
            deltaY,     // convenience variable
            deltaZ,     // convenience variable
            frustrumW, 
            frustrumH;
   PyArrayObject* ndArray;    // Output Matrix
   npy_intp dims[2] = {4, 4}; // Used to define output matrix dimensions

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "dddd",
            &fovy,
            &aspect,
            &nearZ,
            &farZ
            ))
   {
      printf("Error parsing inputs\n");
      // Build and return 4x4 matrix of zeros of type double
      ndArray = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
      return ndArray;
   }

   frustrumH   = tan( fovy / 360.0 * 3.14159265358979 ) * nearZ;
   frustrumW   = frustrumH * aspect;
   left        = -frustrumW;
   right       =  frustrumW;
   bottom      = -frustrumH;
   top         =  frustrumH;

   // Convenience variables
   deltaX = right - left;
   deltaY = top - bottom;
   deltaZ = farZ - nearZ;

   // Sanity Check, prevent division by zero
   if (  (nearZ  <= 0.0) ||
         (farZ   <= 0.0) ||
         (deltaX == 0.0) || 
         (deltaY == 0.0) || 
         (deltaZ == 0.0) ){
      printf("invalid dimensions, (zero delta on an axis), returning matrix unchanged\n");
      ndArray = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
      return ndArray;
   }

   // Allocate memory and set it to identity matrix
   double *persp = (double *)malloc(sizeof(double)*16);;
   buildIdentity(persp);
   
   // Set perspective matrix parameters
   persp[0*4+0] = 2.0 * nearZ / deltaX;
   persp[0*4+1] = persp[0*4+2] = persp[0*4+3] = 0.0;

   persp[1*4+1] = 2.0 * nearZ / deltaY;
   persp[1*4+0] = persp[1*4+2] = persp[1*4+3] = 0.0;

   persp[2*4+0] =  (right + left) / deltaX;
   persp[2*4+1] =  (top + bottom) / deltaY;
   persp[2*4+2] = -(nearZ + farZ) / deltaZ;
   persp[2*4+3] = -1.0;

   persp[3*4+2] = -2.0 * nearZ * farZ / deltaZ;
   persp[3*4+0] = persp[3*4+1] = persp[3*4+3] = 0.0;

   // Build output PyArrayObject from c-array
   ndArray = (PyArrayObject *)PyArray_SimpleNewFromData(
         2,             // Number of dimensions
         dims,          // Array of size of each dimension
         NPY_DOUBLE,    // ndArray type
         (void*)persp   // pointer to >>contiguous<< input data
         );
   return ndArray;
}
