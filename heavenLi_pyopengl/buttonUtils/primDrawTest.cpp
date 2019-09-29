using namespace std;

drawCall    primTest;

PyObject* primTest_hliGLutils(PyObject* self, PyObject *args) {
   PyObject*   faceColorPyTup;
   PyObject*   extraColorPyTup;
   PyObject*   detailColorPyTup;
   GLfloat     gx, 
               gy, 
               scale, 
               w2h, 
               ao=0.0f;
   GLfloat     faceColor[4];
   GLfloat     extraColor[4];
   GLfloat     detailColor[4];
   GLuint      primTestVerts;                 // Total number of vertices

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "ffffOOO",
            &gx, &gy,
            &scale,
            &w2h,
            &faceColorPyTup,
            &extraColorPyTup,
            &detailColorPyTup) )
   {
      Py_RETURN_NONE;
   }

   faceColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 0)));
   faceColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 1)));
   faceColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 2)));
   faceColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 3)));

   extraColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 0)));
   extraColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 1)));
   extraColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 2)));
   extraColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 3)));

   detailColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0)));
   detailColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1)));
   detailColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2)));
   detailColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 3)));

   // Allocate and Define Geometry/Color buffers
   if (  primTest.numVerts == 0 ) {

      printf("Initializing Geometry for PrimTest Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      float colors[9] = {
         1.0f, 1.0f, 0.0f,
         0.0f, 1.0f, 1.0f,
         1.0f, 0.0f, 1.0f};

      float* tmc = new float[4];
      tmc[0] = 1.0f;
      tmc[1] = 1.0f;
      tmc[2] = 0.0f;
      tmc[3] = 1.0f;
      defineQuad2pt(
            -0.5f, -0.5f,
            0.5f, 0.5f,
            tmc,
            verts,
            colrs);

      /*
      defineColorWheel(
            0.0f, 0.0f, 
            1.0f, 
            60, 
            3, 
            1.0f, 
            colors, 
            verts, 
            colrs);
            */

      primTestVerts = verts.size()/2;
      primTest.buildCache(primTestVerts, verts, colrs);
      printf("%d, %d\n", primTestVerts, primTest.numVerts);
      
      delete [] tmc;
   }

   primTest.updateMVP(gx, gy, scale, scale, ao, w2h);
   primTest.draw();

   Py_RETURN_NONE;
}
