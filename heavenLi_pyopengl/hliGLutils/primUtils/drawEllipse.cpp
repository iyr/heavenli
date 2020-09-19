using namespace std;

extern std::map<std::string, drawCall> drawCalls;
extern VertexAttributeStrings VAS;

void drawEllipse(
      GLfloat     gx, 
      GLfloat     gy,
      GLfloat     sx,
      GLfloat     sy,
      GLfloat     w2h,
      GLfloat*    faceColor,
      drawCall*   ellipseButton
      );

PyObject* drawEllipse_hliGLutils(PyObject* self, PyObject *args) {
   PyObject *faceColorPyTup;
   float gx, gy, sx, sy, w2h;
   float faceColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "fffffO",
            &gx, &gy,
            &sx, &sy,
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

   if (drawCalls.count("ellipseButton") <= 0)
      drawCalls.insert(std::make_pair("ellipseButton", drawCall()));
   drawCall* ellipseButton = &drawCalls["ellipseButton"];

   drawEllipse(
         gx, 
         gy,
         sx,
         sy,
         w2h,
         faceColor,
         ellipseButton
         );

   Py_RETURN_NONE;
}

void drawEllipse(
      GLfloat     gx, 
      GLfloat     gy,
      GLfloat     sx,
      GLfloat     sy,
      GLfloat     w2h,
      GLfloat*    faceColor,
      drawCall*   ellipseButton
      ){

   static GLfloat prevGx,
                  prevGy,
                  prevSx,
                  prevSy;

   GLuint ellipseVerts;
   ellipseButton->setNumColors(1);
   ellipseButton->setColorQuartet(0, faceColor);

   int circleSegments = 60;
   if (ellipseButton->numVerts == 0){

      printf("Initializing Geometry for Ellipse Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      // Draw button face
      defineEllipse(
            //0.0f, 0.0f,
            //1.0f, 1.0f,
            gx, gy,
            sx, sy,
            circleSegments,
            faceColor,
            verts, colrs);

      ellipseVerts = verts.size()/2;
      map<string, attribCache> attributeData;
      attributeData[VAS.coordData] = attribCache(VAS.coordData, 2, 0, 0);
      attributeData[VAS.colorData] = attribCache(VAS.colorData, 4, 2, 1);
      attributeData[VAS.coordData].writeCache(verts.data(), verts.size());
      attributeData[VAS.colorData].writeCache(colrs.data(), colrs.size());

      ellipseButton->buildCache(ellipseVerts, attributeData);
   }

   if (  prevGx   != gx ||
         prevGy   != gy ||
         prevSx   != sx ||
         prevSy   != sy ){

      GLint index = 0;

      index = updateEllipseGeometry(
            gx, gy,
            sx, sy,
            circleSegments,
            index,
            (GLfloat *)ellipseButton->getAttribCache(VAS.coordData)
            );

      prevGx   = gx;
      prevGy   = gy;
      prevSx   = sx;
      prevSy   = sy;

      ellipseButton->updateBuffer(VAS.coordData);
   }

   if (ellipseButton->colorsChanged) {
      unsigned int index = 0;
      // Draw button face
      index = updateEllipseColor(
            circleSegments,
            faceColor,
            index, 
            (GLfloat *)ellipseButton->getAttribCache(VAS.colorData));

      ellipseButton->updateBuffer(VAS.colorData);
   }

   //ellipseButton->updateMVP(gx, gy, sx, sy, 0.0f, w2h);
   ellipseButton->updateMVP(0.0f, 0.0f, 1.0f, 1.0f, 0.0f, w2h);
   ellipseButton->draw();

   return;
}
