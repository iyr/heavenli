/*
 * Reference function to read-in, modify/cast, and return a numpy ndarray.
 */
PyArrayObject* ndPrintMatrix_hliGLutils(PyObject* self, PyObject* args){
   PyArrayObject* ndArray;
   double adder = 0.0;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "O",//d",
            &ndArray//,
            //&adder
            ))
   {
      printf("Error parsing inputs\n");
      // Build and return 4x4 matrix of zeros of type double
      npy_intp dims[2] = {4, 4};
      ndArray = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
      return ndArray;
   }

   // Get numpy typenum for input array
   int inType  = PyArray_TYPE(ndArray);
   int numElts = (int)PyArray_Size((PyObject *)ndArray);

   // Safety check for matrix validity
   if (checkTransformMatrixValidity(ndArray) == 0){
      printf("Returning input matrix unchanged\n");

      // Matrix is invalid, reject and return
      Py_INCREF(ndArray);
      return ndArray;
   }

   // Cast Array to double if not already double
   if (inType != NPY_DOUBLE) {
      ndArray = (PyArrayObject *)PyArray_FROM_OT((PyObject *)ndArray, NPY_DOUBLE);
   }

   // Get Contiguous C-array of input matrix
   double*  ndStuffs = (double *)PyArray_DATA(ndArray);
   printf("matrix: \n");
   for(unsigned int i = 0; i < 4; i++){
      printf("%2.4f %2.4f %2.4f %2.4f\n",
            (GLfloat)ndStuffs[i*4+0],
            (GLfloat)ndStuffs[i*4+1],
            (GLfloat)ndStuffs[i*4+2],
            (GLfloat)ndStuffs[i*4+3]
            );
   }
   
   // Operate on the matrix elements
   for (int i = 0; i < numElts; i++)
      ndStuffs[i] += adder;

   // Define dimensions of output array and build from array
   //npy_intp* dims = PyArray_DIMS(ndArray);
   //ndArray = (PyArrayObject *)PyArray_SimpleNewFromData(
         //2,                // Number of dimensions
         //dims,             // Array of size of each dimension
         //NPY_DOUBLE,       // ndArray type
         //(void*)ndStuffs   // pointer to >>contiguous<< input data
         //);

   // Or just incref ndArray if you're not building a new object:
   Py_INCREF(ndArray);

   // Print Updated array
   //ndStuffs = (double *)PyArray_DATA(ndArray);
   //for (int i = 0; i < numElts; i++) {
      //if ( !(i%4) ) printf("\n");
      //printf("%f ", ndStuffs[i]);
   //}
   //printf("\n");

   return ndArray;
}
