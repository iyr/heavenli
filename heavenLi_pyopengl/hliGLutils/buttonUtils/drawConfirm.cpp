using namespace std;

extern std::map<std::string, drawCall> drawCalls;

PyObject* drawConfirm_hliGLutils(PyObject* self, PyObject* args) {
   PyObject*   faceColorPyTup;
   PyObject*   extraColorPyTup;
   PyObject*   detailColorPyTup;
   GLfloat     gx, gy, scale, w2h;
   GLfloat     faceColor[4];
   GLfloat     extraColor[4];
   GLfloat     detailColor[4];

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

   faceColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 0));
   faceColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 1));
   faceColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 2));
   faceColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 3));

   extraColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 0));
   extraColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 1));
   extraColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 2));
   extraColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 3));

   detailColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0));
   detailColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1));
   detailColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2));
   detailColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 3));

   if (drawCalls.count("confirmButton") <= 0)
      drawCalls.insert(std::make_pair("confirmButton", drawCall()));
   drawCall* confirmButton = &drawCalls["confirmButton"];

   drawConfirm(
         gx,
         gy,
         scale,
         w2h,
         faceColor,
         extraColor,
         detailColor,
         confirmButton
         );

   Py_RETURN_NONE;
}

void drawConfirm(
      GLfloat     gx,
      GLfloat     gy,
      GLfloat     scale,
      GLfloat     w2h,
      GLfloat*    faceColor,
      GLfloat*    extraColor,
      GLfloat*    detailColor,
      drawCall*   confirmButton
      ){

   GLfloat  ao=0.0f;
   GLuint   confirmVerts;                 // Total number of vertices

   confirmButton->setNumColors(3);
   confirmButton->setColorQuartet(0, faceColor);
   confirmButton->setColorQuartet(1, extraColor);
   confirmButton->setColorQuartet(2, detailColor);
   confirmButton->setShader("RGBAcolor_NoTexture");

   if (  confirmButton->numVerts == 0   ){

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
      definePill(
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
            verts, 
            colrs
            );

      confirmVerts = verts.size()/2;

      map<string, attribCache> attributeData;
      attributeData[VAS.coordData] = attribCache(VAS.coordData, 2, 0, 0);
      attributeData[VAS.colorData] = attribCache(VAS.colorData, 4, 2, 1);
      attributeData[VAS.coordData].writeCache(verts.data(), verts.size());
      attributeData[VAS.colorData].writeCache(colrs.data(), colrs.size());
      confirmButton->buildCache(confirmVerts, attributeData);
   }

   if ( confirmButton->colorsChanged ) {
      unsigned int index = 0;

      int circleSegments = 60;
      index = updateEllipseColor(
            circleSegments,
            faceColor,
            index, 
            (GLfloat *)confirmButton->getAttribCache(VAS.colorData)
            );

      index = updatePillColor(
            circleSegments/2,
            detailColor,
            index, 
            (GLfloat *)confirmButton->getAttribCache(VAS.colorData));

      index = updatePillColor(
            circleSegments/2,
            detailColor, 
            index, 
            (GLfloat *)confirmButton->getAttribCache(VAS.colorData));

      index = updatePillColor(
            circleSegments/2,
            extraColor,
            index, 
            (GLfloat *)confirmButton->getAttribCache(VAS.colorData));

      index = updatePillColor(
            circleSegments/2,
            extraColor,
            index, 
            (GLfloat *)confirmButton->getAttribCache(VAS.colorData));

      confirmButton->updateBuffer(VAS.colorData);
   }

   confirmButton->updateMVP(gx, gy, scale, scale, ao, w2h);
   confirmButton->draw();

   return;
}

