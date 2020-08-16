#include "primIconLinear.cpp"

/*
 * Heavenli opengl drawcode for linear arrangments (backgroun+iconography)
 */
using namespace std;
extern float offScreen;

drawCall homeLinear;
GLuint   prevHomeLinearNumBulbs;
GLfloat  prevHomeLinearAO,
         sxCorrected;

void drawIconLinear(
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

PyObject* drawHomeLinear_hliGLutils(PyObject *self, PyObject *args) {
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

   homeLinear.setNumColors(numBulbs);
   homeLinear.setShader("RGBAcolor_NoTexture");
   for (unsigned int i = 0; i < numBulbs; i++ ) {
      tmc[0] = bulbColors[i*3+0];
      tmc[1] = bulbColors[i*3+1];
      tmc[2] = bulbColors[i*3+2];
      tmc[3] = alpha;
      homeLinear.setColorQuartet(i, tmc);
   }

   // Allocate and Define Geometry/Color buffers
   if (  homeLinear.numVerts     == 0        ||
         prevHomeLinearNumBulbs  != numBulbs ){

      //printf("Generating geometry for homeLinear\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy;

      homeLinear.setNumColors(numBulbs);
      for (unsigned int j = 0; j < numBulbs; j++) {
         tmc[0] = float(bulbColors[j*3+0]);
         tmc[1] = float(bulbColors[j*3+1]);
         tmc[2] = float(bulbColors[j*3+2]);
         tmc[3] = alpha;

         // Avoid integer division
         GLfloat i0 = (float)j / (float)numBulbs;
         GLfloat i1 = (float)(j+1) / (float)numBulbs;

         TLx = -2.0f + 4.0f*i0;
         TLy = +3.0f;

         TRx = -2.0f + 4.0f*i1;
         TRy = +3.0f;

         BLx = -2.0f + 4.0f*i0;
         BLy = -3.0f;

         BRx = -2.0f + 4.0f*i1;
         BRy = -3.0f;

         if (j == 0)
            TLx *= 2.0f, BLx *=2.0f;

         if (j == numBulbs-1)
            TRx *= 2.0f, BRx *=2.0f;

         defineQuad4pt(
               TLx, TLy,
               BLx, BLy,
               TRx, TRy,
               BRx, BRy,
               tmc,
               verts, colrs);
      }

      //printf("homeLinear vertexBuffer length: %d, Number of vertices: %d, tris: %d\n", verts.size()*8, verts.size(), verts.size()/6);
      prevHomeLinearNumBulbs = numBulbs;
      homeLinear.buildCache(verts.size()/2, verts, colrs);
   } 

   // Geometry already calculated, check if any colors need to be updated.
   if (  homeLinear.colorsChanged               ){

      unsigned int index = 0;

      for (unsigned int j = 0; j < numBulbs; j++) {
         tmc[0] = float(bulbColors[j*3+0]);
         tmc[1] = float(bulbColors[j*3+1]);
         tmc[2] = float(bulbColors[j*3+2]);
         tmc[3] = alpha;

         index = updateQuadColor(
               tmc,
               index,
               homeLinear.colorCache
               );
         }

      homeLinear.updateColorCache();
   }

   if (prevHomeLinearAO != ao) {
      sxCorrected = -0.50f-pow(sin((float)degToRad(ao)), 2.0f);
      prevHomeLinearAO = ao;
   }

   homeLinear.updateMVP(gx, gy, sxCorrected, 0.50f, ao, 1.0f);
   homeLinear.draw();

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

drawCall iconLinear;
GLuint   prevIconLinearNumBulbs;
GLuint   prevIconLinearFeatures;

PyObject* drawIconLinear_hliGLutils(PyObject *self, PyObject *args) {
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
            &gx, &gy,
            &scale, 
            &features,
            &detailColorPyTup,
            &numBulbs,
            &ao,
            &w2h,
            &py_list
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

   // Parse RGB detail colors
   detailColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0)));
   detailColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1)));
   detailColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2)));
   detailColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 3)));

   drawIconLinear(
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

void drawIconLinear(
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

   GLuint iconLinearVerts, circleSegments = 20;

   float tmc[4];
   for (unsigned int i = 0; i < numBulbs; i++) {
      tmc[0] = bulbColors[i*3+0];
      tmc[1] = bulbColors[i*3+1];
      tmc[2] = bulbColors[i*3+2];
      tmc[3] = detailColor[3];
      iconLinear.setColorQuartet(i, tmc);
   }
   iconLinear.setColorQuartet(numBulbs, detailColor);
   iconLinear.setShader("RGBAcolor_NoTexture");

   // Allocate and Define Geometry/Color buffers
   if (  iconLinear.numVerts     == 0        ){

      printf("Generating geometry for iconLinear\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      iconLinear.setNumColors(numBulbs+1);

      // Define Square of Stripes with Rounded Corners
      defineIconLinear(
            0.0f, 0.0f,       // pos (X, Y)
            1.0f,             // size
            features,         // feature level of the icon
            circleSegments,   // number of polygons
            numBulbs,         // number of colors to represent
            detailColor[3],   // Alpha transparency
            bulbColors,       // bulb colors
            detailColor,      // color of the accent details
            verts, colrs);

      iconLinearVerts = verts.size()/2;
      printf("iconLinear vertexBuffer length: %.i, Number of vertices: %.i, tris: %.i\n", iconLinearVerts*2, iconLinearVerts, iconLinearVerts/3);

      // Update State machine variables
      prevIconLinearNumBulbs = numBulbs;
      prevIconLinearFeatures = features;

      iconLinear.buildCache(iconLinearVerts, verts, colrs);
   } 

   // Update features
   if (prevIconLinearFeatures != features ||
       prevIconLinearNumBulbs != numBulbs ){

      unsigned int index = 0;

      // Changes in bulb quantity necessitate color update
      if (  prevIconLinearNumBulbs  != numBulbs ){
         iconLinear.setNumColors(numBulbs+1);
         iconLinear.setColorQuartet(numBulbs, detailColor);
         float tmc[4];
         for (unsigned int i = 0; i < numBulbs; i++) {
            tmc[0] = bulbColors[i*3+0];
            tmc[1] = bulbColors[i*3+1];
            tmc[2] = bulbColors[i*3+2];
            tmc[3] = detailColor[3];
	    iconLinear.setColorQuartet(i, tmc);
         }

         //updateIconLinearColor();
         updateIconLinearColor(circleSegments, numBulbs, 1.0f, bulbColors, detailColor, index, iconLinear.colorCache);
         iconLinear.updateColorCache();
         index = 0;
      }

      updateIconLinearGeometry(0.0f, 0.0f, 1.0f, features, circleSegments, numBulbs, index, iconLinear.coordCache);
      iconLinear.updateCoordCache();
      prevIconLinearNumBulbs = numBulbs;
      prevIconLinearFeatures = features;
   }

   // Geometry allocated/calculated, check if colors need to be updated
   if (  iconLinear.colorsChanged   ){
      unsigned int index = 0;
      updateIconLinearColor(circleSegments, numBulbs, 1.0f, bulbColors, detailColor, index, iconLinear.colorCache);
      iconLinear.updateColorCache();
   }

   iconLinear.updateMVP(gx, gy, -scale, scale, ao, w2h);
   iconLinear.draw();

   return;
}

