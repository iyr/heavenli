using namespace std;

drawCall    primButton;
GLuint      extraPrimVerts;

PyObject* drawPrim_hliGLutils(PyObject* self, PyObject *args) {
   PyObject *faceColorPyTup;
   PyObject *extraColorPyTup;
   PyObject *detailColorPyTup;
   float gx, gy, ao, scale, w2h;
   float faceColor[4];
   float extraColor[4];
   float detailColor[4];
   GLuint primVerts;

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "fffffOOO",
            &gx, &gy,
            &scale,
            &ao,
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

   int circleSegments = 60;
   if (primButton.numVerts == 0){

      printf("Initializing Geometry for Prim Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      float px, py, qx, qy, radius;

      float colors[18] = {
         1.0f, 1.0f, 0.0f,
         1.0f, 0.0f, 1.0f,
         0.0f, 1.0f, 1.0f,
         1.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f,
         0.0f, 0.0f, 1.0f};
      
      defineQuad2pt(
            -0.25f, -0.25f,
            -0.5f, -0.5f,
            faceColor,
            verts, colrs);

      defineQuad2pt(
            0.25f, -0.25f,
            0.5f, -0.5f,
            faceColor,
            verts, colrs);

      defineQuad2pt(
            -0.25f, 0.25f,
            -0.5f, 0.5f,
            faceColor,
            verts, colrs);

      defineQuad2pt(
            0.25f, 0.25f,
            0.5f, 0.5f,
            faceColor,
            verts, colrs);

      primVerts = verts.size()/2;

      primButton.buildCache(primVerts, verts, colrs);

      printf("primVerts: %d, primButton.numVerts: %d\n", primVerts, primButton.numVerts);
   }

   /*
   int index = 0;
   GLboolean updateCache = GL_FALSE;
   // Update Color, if needed
   for (int i = 0; i < 4; i++) {
      if ( primButton.colorCache[extraPrimVerts*4+i] != extraColor[i] ) {
         for (unsigned int k = extraPrimVerts; k < primButton.numVerts; k++) {
            primButton.colorCache[k*4 + i] = extraColor[i];
         }

         updateCache = GL_TRUE;
      }
   }

   // Update colors, if needed
   if ( updateCache ){
      primButton.updateColorCache();
   }
   */

   primButton.updateMVP(gx, gy, scale, scale, ao, w2h);

   primButton.draw();

   Py_RETURN_NONE;
}
