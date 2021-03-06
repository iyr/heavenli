using namespace std;

extern std::map<std::string, drawCall> drawCalls;
extern VertexAttributeStrings VAS;

PyObject* drawArrow_hliGLutils(PyObject* self, PyObject *args) {
   PyObject *faceColorPyTup;
   PyObject *extraColorPyTup;
   PyObject *detailColorPyTup;
   float gx, gy, ao, scale, w2h;
   float faceColor[4];
   float extraColor[4];
   float detailColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "fffffOOO",
            &gx, &gy,
            &ao,
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

   if (drawCalls.count("arrowButton") <= 0)
      drawCalls.insert(std::make_pair("arrowButton", drawCall()));
   drawCall* arrowButton = &drawCalls["arrowButton"];

   drawArrow(
         gx, gy,
         ao,
         scale,
         w2h,
         faceColor,
         extraColor,
         detailColor,
         arrowButton
         );

   Py_RETURN_NONE;
}

void drawArrow(
      GLfloat     gx, 
      GLfloat     gy,
      GLfloat     ao,
      GLfloat     scale,
      GLfloat     w2h,
      GLfloat*    faceColor,
      GLfloat*    extraColor,
      GLfloat*    detailColor,
      drawCall*   arrowButton
      ){

   GLuint arrowVerts;
   arrowButton->setNumColors(3);
   arrowButton->setColorQuartet(0, faceColor);
   arrowButton->setColorQuartet(1, extraColor);
   arrowButton->setColorQuartet(2, detailColor);
   arrowButton->setShader("RGBAcolor_NoTexture");

   int circleSegments = 60;
   if (arrowButton->numVerts == 0){

      printf("Initializing Geometry for Arrow Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      float px, py, qx, qy, radius;
      
      // Draw button face
      defineEllipse(
            0.0f, 0.0f,
            1.0f, 1.0f,
            circleSegments,
            faceColor,
            verts, colrs);

      // Draw check-mark base
      px = -0.125f, py = 0.625f;
      qx =  0.500f, qy =  0.00f;
      radius = float(sqrt(2.0)*0.125f);
      definePill(
            px, py,
            qx, qy,
            radius,
            circleSegments/2,
            detailColor,
            verts, colrs);

      px = -0.125f, py = -0.625f;
      definePill(
            px, py, 
            qx, qy, 
            radius, 
            circleSegments/2,
            detailColor, 
            verts, colrs);

      // Draw check-mark infill
      px = -0.125f, py =  0.625f;
      radius = 0.125f;
      definePill(
            px, py, 
            qx, qy, 
            radius, 
            circleSegments/2,
            extraColor,
            verts, colrs);

      px = -0.125f, py = -0.625f;
      definePill(
            px, py, 
            qx, qy, 
            radius, 
            circleSegments/2,
            extraColor,
            verts, colrs);

      arrowVerts = verts.size()/2;

      map<string, attribCache> attributeData;
      attributeData[VAS.coordData] = attribCache(VAS.coordData, 2, 0, 0);
      attributeData[VAS.colorData] = attribCache(VAS.colorData, 4, 2, 1);
      attributeData[VAS.coordData].writeCache(verts.data(), verts.size());
      attributeData[VAS.colorData].writeCache(colrs.data(), colrs.size());
      arrowButton->buildCache(arrowVerts, attributeData);
   }

   if (arrowButton->colorsChanged) {
      unsigned int index = 0;
      // Draw button face
      index = updateEllipseColor(
            circleSegments,
            faceColor,
            index, 
            (GLfloat *)arrowButton->getAttribCache(VAS.colorData));

      // Draw check-mark base
      index = updatePillColor(
            circleSegments/2,
            detailColor,
            index, 
            (GLfloat *)arrowButton->getAttribCache(VAS.colorData));

      index = updatePillColor(
            circleSegments/2,
            detailColor, 
            index, 
            (GLfloat *)arrowButton->getAttribCache(VAS.colorData));

      // Draw check-mark infill
      index = updatePillColor(
            circleSegments/2,
            extraColor,
            index, 
            (GLfloat *)arrowButton->getAttribCache(VAS.colorData));

      index = updatePillColor(
            circleSegments/2,
            extraColor,
            index, 
            (GLfloat *)arrowButton->getAttribCache(VAS.colorData));

      arrowButton->updateBuffer(VAS.colorData);
   }

   arrowButton->updateMVP(gx, gy, scale, scale, ao, w2h);
   arrowButton->draw();

   return;
}

