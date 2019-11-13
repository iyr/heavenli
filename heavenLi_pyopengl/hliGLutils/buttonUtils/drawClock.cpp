using namespace std;

drawCall clockButton;
GLfloat  prevClockHour;    // Used for animated hour hand
GLfloat  prevClockMinute;  // Used for animated minute hand
GLuint   clockVerts;       // Total number of vertices
GLuint   faceVerts;        // Number of vertices of face (makes animating hands easier)
extern GLuint whiteTex;

PyObject* drawClock_hliGLutils(PyObject *self, PyObject *args)
{
   PyObject* faceColorPyTup;
   PyObject* detailColorPyTup;
   GLfloat gx, gy, px, py, qx, qy, radius, ao=0.0f;
   GLfloat scale, w2h, hour, minute;
   GLfloat detailColor[4];
   GLfloat faceColor[4];
   // Set Number of edges on circles
   char circleSegments = 60;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "ffffffOO",
            &gx, &gy,
            &hour,
            &minute,
            &scale,
            &w2h,
            &faceColorPyTup,
            &detailColorPyTup))
   {
      Py_RETURN_NONE;
   }

   // Parse RGB color tuples of face and detail colors
   for (unsigned int i = 0; i < 4; i++){
      faceColor[i] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, i));
      detailColor[i] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, i));
   }

   clockButton.setNumColors(2);
   clockButton.setColorQuartet(0, faceColor);
   clockButton.setColorQuartet(1, detailColor);
   
   if (  clockButton.numVerts == 0  ){
      printf("Initializing Geometry for Clock Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      defineEllipse(
            0.0f, 0.0f, 
            0.5f, 0.5f,
            circleSegments,
            faceColor,
            verts,
            colrs);

      faceVerts = verts.size()/2;

      px = 0.0;
      py = 0.0;
      qx = float(0.2*cos(degToRad(90-360*(hour/12.0))));
      qy = float(0.2*sin(degToRad(90-360*(hour/12.0))));
      radius = float(0.02);
      definePill(px, py, qx, qy, radius, circleSegments/4, detailColor, verts, colrs);

      qx = float(0.4*cos(degToRad(90-360*(minute/60.0))));
      qy = float(0.4*sin(degToRad(90-360*(minute/60.0))));
      radius = float(0.01);
      definePill(px, py, qx, qy, radius, circleSegments/4, detailColor, verts, colrs);

      clockVerts = verts.size()/2;

      clockButton.buildCache(clockVerts, verts, colrs);
   } 

   // Animate Clock Hands
   if (prevClockHour     != hour    ||
       prevClockMinute   != minute  ){
      px = 0.0;
      py = 0.0;
      qx = float(0.2*cos(degToRad(90-360*(hour/12.0))));
      qy = float(0.2*sin(degToRad(90-360*(hour/12.0))));
      radius = float(0.02);

      int tmp;
      tmp = updatePillGeometry(
            px, py,
            qx, qy,
            radius,
            circleSegments/4,
            faceVerts,
            clockButton.coordCache);

      qx = float(0.4*cos(degToRad(90-360*(minute/60.0))));
      qy = float(0.4*sin(degToRad(90-360*(minute/60.0))));
      radius = float(0.01);
      tmp = updatePillGeometry(
            px, py,
            qx, qy,
            radius,
            circleSegments/4,
            tmp,
            clockButton.coordCache);

      prevClockHour     = hour;
      prevClockMinute   = minute;

      clockButton.updateCoordCache();
   }

   if (clockButton.colorsChanged) {
      unsigned int index = 0;

      index = updateEllipseColor(
            circleSegments,
            faceColor,
            index,
            clockButton.colorCache);

      index = updatePillColor(circleSegments/4, detailColor, index, clockButton.colorCache);

      index = updatePillColor(circleSegments/4, detailColor, index, clockButton.colorCache);

      clockButton.updateColorCache();
   }

   clockButton.updateMVP(gx, gy, scale, scale, ao, w2h);
   clockButton.draw();

   Py_RETURN_NONE;
}
