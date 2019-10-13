#include <Python.h>
#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
   // These undefs necessary because microsoft
   #undef near
   #undef far
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>
#include "primIconLinear.cpp"
using namespace std;
extern float offScreen;

GLuint      prevHomeLinearNumBulbs;
drawCall    homeLinear;

PyObject* drawHomeLinear_hliGLutils(PyObject *self, PyObject *args) {
   PyObject*   py_list;
   PyObject*   py_tuple;
   PyObject*   py_float;
   GLfloat*    bulbColors;
   GLfloat     gx, gy, wx, wy, ao, w2h, alpha=1.0f;
   GLfloat     tmc[4];
   GLuint      numBulbs;
   GLuint      homeLinearVerts;
   if (!PyArg_ParseTuple(args,
            "fffflffO",
            &gx, &gy,
            &wx, &wy,
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

   homeLinear.setNumColors(numBulbs);
   for (unsigned int i = 0; i < numBulbs; i++ ) {
      tmc[0] = bulbColors[i*numBulbs+0];
      tmc[1] = bulbColors[i*numBulbs+1];
      tmc[2] = bulbColors[i*numBulbs+2];
      tmc[3] = alpha;
      homeLinear.setColorQuartet(i, tmc);
   }

   GLuint segments = 60;
   // Allocate and Define Geometry/Color buffers
   if (  homeLinear.numVerts == 0   ||
         prevHomeLinearNumBulbs != numBulbs ){

      printf("Generating geometry for homeLinear\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy;
      float offset = 4.0f / 60.0f;
      unsigned int limit = segments/numBulbs;

      for (unsigned int j = 0; j < numBulbs; j++) {
         tmc[0] = float(bulbColors[j*3+0]);
         tmc[1] = float(bulbColors[j*3+1]);
         tmc[2] = float(bulbColors[j*3+2]);
         tmc[3] = alpha;

         for (unsigned int i = 0; i < limit; i++) {

            if (  i == 0   &&
                  j == 0   ){
               TLx = -4.0f;
               TLy =  4.0f;

               BLx = -4.0f;
               BLy = -4.0f;
            } else {
               TLx = float(-2.0f + i*offset + j*offset*limit);
               TLy =  4.0f;

               BLx = float(-2.0f + i*offset + j*offset*limit);
               BLy = -4.0f;
            }

            if (  i == numBulbs-1   &&
                  j == limit-1      ){
               TRx =  4.0f;
               TRy =  4.0f;

               BRx =  4.0f;
               BRy = -4.0f;
            } else {
               TRx = float(-2.0f + (i+1)*offset + (j+1)*offset*limit);
               TRy =  4.0f;

               BRx = float(-2.0f + (i+1)*offset + (j+1)*offset*limit);
               BRy = -4.0f;
            }

            defineQuad4pt(
                  TLx, TLy,
                  BLx, BLy,
                  TRx, TRy,
                  BRx, BRy,
                  tmc,
                  verts, colrs);

         }
      }

      homeLinearVerts = verts.size()/2;
      printf("homeLinear vertexBuffer length: %.i, Number of vertices: %.i, tris: %.i\n", homeLinearVerts*2, homeLinearVerts, homeLinearVerts/3);
      prevHomeLinearNumBulbs = numBulbs;

      homeLinear.buildCache(verts.size()/2, verts, colrs);
   } 

   // Geometry already calculated, check if any colors need to be updated.
   if (  homeLinear.colorsChanged               ||
         prevHomeLinearNumBulbs     != numBulbs ){

      unsigned int index = 0;
      unsigned int limit = segments/numBulbs;

      for (unsigned int j = 0; j < numBulbs; j++) {
         tmc[0] = float(bulbColors[j*3+0]);
         tmc[1] = float(bulbColors[j*3+1]);
         tmc[2] = float(bulbColors[j*3+2]);
         tmc[3] = alpha;

         for (unsigned int i = 0; i < limit; i++) {
            index = updateQuadColor(
                  tmc,
                  index,
                  homeLinear.colorCache
                  );
         }
      }

      homeLinear.updateColorCache();

      prevHomeLinearNumBulbs = numBulbs;
   }

   prevHomeLinearNumBulbs = numBulbs;
   delete [] bulbColors;

   homeLinear.updateMVP(gx, gy, -0.50f-pow(sin((float)degToRad(ao)), 2.0f), 0.50f, ao, 1.0f);
   homeLinear.draw();

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
   GLfloat     gx, gy, scale, ao, w2h, alpha=1.0f;
   GLfloat     R, G, B;
   GLuint      numBulbs, features, iconLinearVerts;
   GLuint      vertIndex = 0;
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

   unsigned int circleSegments = 20;

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

   float tmc[4];
   for (unsigned int i = 0; i < numBulbs; i++) {
      tmc[0] = bulbColors[i*3+0];
      tmc[1] = bulbColors[i*3+1];
      tmc[2] = bulbColors[i*3+2];
      tmc[3] = detailColor[3];
      iconLinear.setColorQuartet(i, tmc);
   }
   iconLinear.setColorQuartet(numBulbs, detailColor);

   // Allocate and Define Geometry/Color buffers
   if (  iconLinear.numVerts     == 0        ){

      printf("Generating geometry for iconLinear\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      iconLinear.setNumColors(numBulbs+1);
      float offset = float(2.0/60.0);
      float degSegment = float(360.0/float(circleSegments));
      float delta = float(degSegment/4.0);

      // Define Square of Stripes with Rounded Corners
      int tmb = 0;
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

   delete [] bulbColors;

   iconLinear.updateMVP(gx, gy, -scale, scale, ao, w2h);
   iconLinear.draw();

   Py_RETURN_NONE;
}
