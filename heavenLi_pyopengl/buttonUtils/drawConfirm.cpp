using namespace std;

drawCall    confirmButton;       // drawCall object
GLuint      extraConfirmVerts;   // Used for determining where to write to update cache

PyObject* drawConfirm_drawUtils(PyObject* self, PyObject *args) {
   PyObject*   faceColorPyTup;
   PyObject*   extraColorPyTup;
   PyObject*   detailColorPyTup;
   GLfloat     gx, gy, scale, w2h, ao=0.0f;
   GLfloat     faceColor[4];
   GLfloat     extraColor[4];
   GLfloat     detailColor[4];
   GLuint      confirmVerts;                 // Total number of vertices

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

   if (  confirmButton.numVerts == 0   ){

      printf("Initializing Geometry for Confirm Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      float px, py, qx, qy, radius;
      int circleSegments = 60;
      defineEllipse(
            0.0f, 0.0f,
            1.0f, 1.0f,
            circleSegments,
            faceColor,
            verts, colrs);

      px = -0.75f, py =  0.0f;
      qx = -0.25f, qy = -0.5f;
      radius = float(sqrt(2.0)*0.125f);
      definePill(
            px, py,
            qx, qy,
            radius,
            circleSegments/2,
            detailColor,
            verts, colrs);


      px = 0.625f, py = 0.375f;
      extraConfirmVerts = definePill(
            px, py, 
            qx, qy, 
            radius, 
            circleSegments/2,
            detailColor, 
            verts, colrs);

      px = -0.75f, py =  0.0f;
      radius = 0.125f;
      definePill(
            px, py,
            qx, qy,
            radius,
            circleSegments/2,
            extraColor,
            verts, colrs);

      px = 0.625f, py = 0.375f;
      definePill(
            px, py,
            qx, qy,
            radius,
            circleSegments/2,
            extraColor,
            verts, colrs);

      confirmVerts = verts.size()/2;

      confirmButton.buildCache(confirmVerts, verts, colrs);
   }

   GLboolean updateCache = GL_FALSE;
   // Geometry allocated, check if color needs to be updated
   for (int i = 0; i < 4; i++) {
      if ( confirmButton.colorCache[extraConfirmVerts*4+i] != extraColor[i] ) {
         for (unsigned int k = extraConfirmVerts; k < confirmButton.numVerts; k++) {
            confirmButton.colorCache[k*4 + i] = extraColor[i];
         }
         updateCache = GL_TRUE;
      }
   }

   if ( updateCache ){
      confirmButton.updateColorCache();
   }

   confirmButton.updateMVP(gx, gy, scale, scale, ao, w2h);
   confirmButton.draw();

   Py_RETURN_NONE;
}
