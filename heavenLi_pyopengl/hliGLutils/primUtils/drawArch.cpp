using namespace std;

extern std::map<std::string, drawCall> drawCalls;

void drawArch(
         GLfloat     gx, 
         GLfloat     gy,
         GLfloat     sx,
         GLfloat     sy,
         GLfloat     start,
         GLfloat     end,
         GLfloat     rs,
         GLfloat     w2h,
         GLfloat*    faceColor,
         drawCall*   archButton
      );

PyObject* drawArch_hliGLutils(PyObject* self, PyObject *args) {
   PyObject *faceColorPyTup;
   float gx, gy, sx, sy, start, end, rs, w2h;
   float faceColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "ffffffffO",
            &gx, &gy,
            &sx, &sy,
            &start,
            &end,
            &rs,
            &w2h,
            &faceColorPyTup
            ) )
   {
      Py_RETURN_NONE;
   }

   faceColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 0)));
   faceColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 1)));
   faceColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 2)));
   faceColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 3)));

   if (drawCalls.count("archButton") <= 0)
      drawCalls.insert(std::make_pair("archButton", drawCall()));
   drawCall* archButton = &drawCalls["archButton"];

   drawArch(
         gx, 
         gy,
         sx,
         sy,
         start,
         end,
         rs,
         w2h,
         faceColor,
         archButton
         );

   Py_RETURN_NONE;
}

void drawArch(
         GLfloat     gx, 
         GLfloat     gy,
         GLfloat     sx,
         GLfloat     sy,
         GLfloat     start,
         GLfloat     end,
         GLfloat     rs,
         GLfloat     w2h,
         GLfloat*    faceColor,
         drawCall*   archButton
         ){
   static GLfloat prevGx,
                  prevGy,
                  prevSx,
                  prevSy,
                  prevStart,
                  prevEnd,
                  prevRs;

   GLuint archVerts;
   archButton->setNumColors(1);
   archButton->setColorQuartet(0, faceColor);
   int circleSegments = 60;

   if (archButton->numVerts == 0){

      printf("Initializing Geometry for Arch Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      // Draw button face
      defineArch(
            gx, gy,
            sx, sy,
            start,
            end,
            rs,
            circleSegments,
            faceColor,
            verts, 
            colrs);

      archVerts = verts.size()/2;

      prevGx      = gx;
      prevGy      = gy;
      prevSx      = sx;
      prevSy      = sy;
      prevStart   = start;
      prevEnd     = end;
      prevRs      = rs;

      archButton->buildCache(archVerts, verts, colrs);
   }

   if (  prevGx      != gx    ||
         prevGy      != gy    ||
         prevSx      != sx    ||
         prevSy      != sy    ||
         prevStart   != start ||
         prevEnd     != end   ||
         prevRs      != rs    ){

      GLint index = 0;

      index = updateArchGeometry(
            gx, gy,
            sx, sy,
            start,
            end,
            rs,
            circleSegments,
            index,
            archButton->coordCache
            );

      prevGx      = gx;
      prevGy      = gy;
      prevSx      = sx;
      prevSy      = sy;
      prevStart   = start;
      prevEnd     = end;
      prevRs      = rs;

      archButton->updateCoordCache();
   }

   if (archButton->colorsChanged) {
      unsigned int index = 0;
      // Draw button face
      index = updateArchColor(
            circleSegments,
            faceColor,
            index, 
            archButton->colorCache);

      archButton->updateColorCache();
   }

   //archButton->updateMVP(gx, gy, sx, sy, 0.0f, w2h);
   archButton->updateMVP(0.0f, 0.0f, 1.0f, 1.0f, 0.0f, w2h);
   archButton->draw();
   return;
}
