#include "primIconCircle.cpp"

/*
 * Heavenli opengl drawcode for circular arrangments (backgroun+iconography)
 */
using namespace std;
extern std::map<std::string, drawCall> drawCalls;

GLuint   prevHomeCircleNumBulbs;

void drawIconCircle(
      GLfloat  gx,            // icon position X
      GLfloat  gy,            // icon position Y
      GLfloat  scale,         // icon size
      GLuint   features,      // iconography features
      GLfloat* detailColor,   // feature colors
      GLuint   numBulbs,      // number of elements
      GLfloat  ao,            // icon rotation angle
      GLfloat  w2h,           // width to hight ration
      GLfloat* bulbColors     // colors of the elements (bulbs)
      );

PyObject* drawHomeCircle_hliGLutils(PyObject *self, PyObject *args) {
   PyObject*   py_list;
   PyObject*   py_tuple;
   PyObject*   py_float;
   GLfloat*    bulbColors;
   GLfloat     gx, 
               gy, 
               wx, 
               wy, 
               ao, 
               w2h, 
               alpha=1.0f;
   GLfloat     tmc[4];
   GLuint      numBulbs;

   if (!PyArg_ParseTuple(args,
            "fffflffO",
            &gx, &gy,   // background position (X, Y)
            &wx, &wy,   // background scale (X, Y)
            &numBulbs,  // number of elements
            &ao,        // background rotation angle
            &w2h,       // width to height ratio
            &py_list    // colors of the background segments
            ))
   {
      Py_RETURN_NONE;
   }

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new float[numBulbs*3];
   for (unsigned int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (unsigned int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = float(PyFloat_AsDouble(py_float));
      }
   }
   if (drawCalls.count("homeCircle") <= 0)
      drawCalls.insert(std::make_pair("homeCircle", drawCall()));
   drawCall* homeCircle = &drawCalls["homeCircle"];

   homeCircle->setNumColors(numBulbs);
   homeCircle->setShader("RGBAcolor_NoTexture");
   for (unsigned int i = 0; i < numBulbs; i++ ) {
      tmc[0] = bulbColors[i*3+0];
      tmc[1] = bulbColors[i*3+1];
      tmc[2] = bulbColors[i*3+2];
      tmc[3] = alpha;
      homeCircle->setColorQuartet(i, tmc);
   }


   if (  homeCircle->numVerts     == 0        ){

      printf("Initializing Geometry for Circular Background\n");
      GLuint circleSegments = 60;
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      homeCircle->setNumColors(numBulbs);
      defineColorWheel(0.0f, 0.0f, 10.0f, circleSegments, 180.0f, numBulbs, 1.0f, bulbColors, verts, colrs);
      printf("homeCircle vertexBuffer length: %d, Number of vertices: %d, tris: %d\n", (int)verts.size()*8, (int)verts.size(), (int)verts.size()/6);

      prevHomeCircleNumBulbs = numBulbs;
      map<string, attribCache> attributeData;
      attributeData[VAS.coordData] = attribCache(VAS.coordData, 2, 0, 0);
      attributeData[VAS.colorData] = attribCache(VAS.colorData, 4, 2, 1);
      attributeData[VAS.coordData].writeCache(verts.data(), verts.size());
      attributeData[VAS.colorData].writeCache(colrs.data(), colrs.size());
      homeCircle->buildCache(verts.size()/2, attributeData);
   }

   // Update Colors, if necessary
   if (  homeCircle->colorsChanged               ||
         prevHomeCircleNumBulbs     != numBulbs ){

      unsigned int index = 0;
      GLuint circleSegments = 60;

      // Changes in bulb quantity necessitate color update
      if (  prevHomeCircleNumBulbs  != numBulbs ){
         homeCircle->setNumColors(numBulbs);
         float tmc[4];
         for (unsigned int i = 0; i < numBulbs; i++) {
            tmc[0] = bulbColors[i*3+0];
            tmc[1] = bulbColors[i*3+1];
            tmc[2] = bulbColors[i*3+2];
            tmc[3] = alpha;
            homeCircle->setColorQuartet(i, tmc);
         }

         updateColorWheelColor(circleSegments, numBulbs, 1.0f, bulbColors, index, (GLfloat *)homeCircle->getAttribCache(VAS.colorData));
         homeCircle->updateBuffer(VAS.colorData);
         index = 0;
         prevHomeCircleNumBulbs = numBulbs;
      }

      updateColorWheelColor(circleSegments, numBulbs, 1.0f, bulbColors, index, (GLfloat *)homeCircle->getAttribCache(VAS.colorData));
      homeCircle->updateBuffer(VAS.colorData);
   }

   homeCircle->updateMVP(gx, gy, 1.0f, 1.0f, -ao, 1.0f);
   homeCircle->draw();

   delete [] bulbColors;
   Py_RETURN_NONE;
}


/*
 * Explanation of features:
 * <= 0: just the color representation
 * <= 1: color representation + outline
 * <= 2: color representation + outline + bulb markers
 * <= 3: color representation + outline + bulb markers + bulb marker halos
 * <= 4: color representation + outline + bulb markers + bulb marker halos + grand halo
 */
GLuint   prevIconCircleNumBulbs;
GLuint   prevIconCircleFeatures;

PyObject* drawIconCircle_hliGLutils(PyObject *self, PyObject *args) {
   PyObject*   detailColorPyTup;
   PyObject*   py_list;
   PyObject*   py_tuple;
   PyObject*   py_float;
   GLfloat*    bulbColors;
   GLfloat     detailColor[4];
   GLfloat     gx, gy, scale, ao, w2h;
   GLuint      numBulbs, features;

   if (!PyArg_ParseTuple(args,
            "ffflOlffO",
            &gx, &gy,            // icon positon (X, Y)
            &scale,              // icon size
            &features,           // iconography features
            &detailColorPyTup,   // feature colors
            &numBulbs,           // number of elements
            &ao,                 // icon rotation angle
            &w2h,                // width to hight ration
            &py_list             // colors of the elements (bulbs)
            ))
   {
      Py_RETURN_NONE;
   }

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new float[numBulbs*3];
   for (unsigned int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (unsigned char j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = float(PyFloat_AsDouble(py_float));
      }
   }

   // Parse RGB detail colors
   detailColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0)));
   detailColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1)));
   detailColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2)));
   detailColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 3)));

   drawIconCircle(
         gx,
         gy,
         scale,
         features,
         detailColor,
         numBulbs,
         ao,
         w2h,
         bulbColors);

   delete [] bulbColors;

   Py_RETURN_NONE;
}

void drawIconCircle(
      GLfloat  gx,            // icon position X
      GLfloat  gy,            // icon position Y
      GLfloat  scale,         // icon size
      GLuint   features,      // iconography features
      GLfloat* detailColor,   // feature colors
      GLuint   numBulbs,      // number of elements
      GLfloat  ao,            // icon rotation angle
      GLfloat  w2h,           // width to hight ration
      GLfloat* bulbColors     // colors of the elements (bulbs)
      ){

   unsigned int circleSegments = 60;

   if (drawCalls.count("iconCircle") <= 0)
      drawCalls.insert(std::make_pair("iconCircle", drawCall()));
   drawCall* iconCircle = &drawCalls["iconCircle"];

   iconCircle->setNumColors(numBulbs+1);
   iconCircle->setShader("RGBAcolor_NoTexture");

   float tmc[4];
   for (unsigned int i = 0; i < numBulbs; i++) {
      tmc[0] = bulbColors[i*3+0];
      tmc[1] = bulbColors[i*3+1];
      tmc[2] = bulbColors[i*3+2];
      tmc[3] = detailColor[3];
      iconCircle->setColorQuartet(i, tmc);
   }
   iconCircle->setColorQuartet(numBulbs, detailColor);

   // Allocate and Define Geometry/Color buffers
   if (  iconCircle->numVerts     == 0  ){

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      iconCircle->setNumColors(numBulbs+1);

      defineIconCircle(
            0.0f, 0.0f,       // pos (X, Y)
            1.0f,             // size
            features,         // feature level of the icon
            circleSegments,   // number of polygons
            numBulbs,         // number of colors to represent
            detailColor[3],   // Alpha transparency
            bulbColors,       // bulb colors
            detailColor,      // color of the accent details
            verts, colrs);

      prevIconCircleNumBulbs = numBulbs;
      map<string, attribCache> attributeData;
      attributeData[VAS.coordData] = attribCache(VAS.coordData, 2, 0, 0);
      attributeData[VAS.colorData] = attribCache(VAS.colorData, 4, 2, 1);
      attributeData[VAS.coordData].writeCache(verts.data(), verts.size());
      attributeData[VAS.colorData].writeCache(colrs.data(), colrs.size());
      iconCircle->buildCache(verts.size()/2, attributeData);
   }

   // Update Geometry, if necessary
   if (  prevIconCircleFeatures  != features ||
         prevIconCircleNumBulbs  != numBulbs ){

      unsigned int index = 0;

      // Changes in bulb quantity necessitate color update
      if (  prevIconCircleNumBulbs  != numBulbs ){
         iconCircle->setNumColors(numBulbs+1);
         iconCircle->setColorQuartet(numBulbs, detailColor);
         float tmc[4];
         for (unsigned int i = 0; i < numBulbs; i++) {
            tmc[0] = bulbColors[i*3+0];
            tmc[1] = bulbColors[i*3+1];
            tmc[2] = bulbColors[i*3+2];
            tmc[3] = detailColor[3];
            iconCircle->setColorQuartet(i, tmc);
         }

         updateIconCircleColor(circleSegments, numBulbs, 1.0f, bulbColors, detailColor, index, (GLfloat *)iconCircle->getAttribCache(VAS.colorData));
         iconCircle->updateBuffer(VAS.colorData);
         index = 0;
      }

      updateIconCircleGeometry(0.0f, 0.0f, 1.0f, features, circleSegments, numBulbs, index, (GLfloat *)iconCircle->getAttribCache(VAS.coordData));
      iconCircle->updateBuffer(VAS.coordData);
      prevIconCircleNumBulbs = numBulbs;
      prevIconCircleFeatures = features;
   }

   // Update Colors, if necessary
   if (  iconCircle->colorsChanged   ){
      unsigned int index = 0;
      updateIconCircleColor(circleSegments, numBulbs, 1.0f, bulbColors, detailColor, index, (GLfloat *)iconCircle->getAttribCache(VAS.colorData));
      iconCircle->updateBuffer(VAS.colorData);
   }

   iconCircle->updateMVP(gx, gy, scale, scale, -ao, w2h);
   iconCircle->draw();
   return;
}
