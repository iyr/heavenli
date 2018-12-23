#include <Python.h>
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>
#define degToRad(angleInDegrees) ((angleInDegrees) * 3.1415926535 / 180.0)
using namespace std;

GLfloat  *vertexBuffer = NULL;
GLfloat  *colahbuffah  = NULL;
GLfloat  *prevColors   = NULL;
GLfloat  *curnColors   = NULL;
GLushort *indices      = NULL;
GLuint   numVerts      = NULL;
GLuint   bulbVerts     = NULL;
int      colorsStart   = NULL;
int      colorsEnd     = NULL;
int      prevNumBulbs  = NULL;
float    prevAngOffset = NULL;
float    prevW2H       = NULL;
PyObject* drawBulbButton_drawButtons(PyObject *self, PyObject *args)
{
   PyObject* faceColorPyTup;
   PyObject* lineColorPyTup;
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   double faceColor[3];
   double lineColor[3]; 
   double *bulbColors;
   //double bulbColor[3];
   float angularOffset, scale, w2h;
   int arn, numBulbs;

   // Parse input arguments
   if (!PyArg_ParseTuple(args, 
            "iiffOOOf", 
            &arn,
            &numBulbs,
            &angularOffset,
            &scale,
            &faceColorPyTup,
            &lineColorPyTup,
            &py_list,
            &w2h))
   {
      Py_RETURN_NONE;
   }

   //py_tuple = PyList_GetItem(py_list, 0)
   bulbColors = new double[numBulbs*3];
   for (int i = 0; i < numBulbs; i++){
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++){
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = double(PyFloat_AsDouble(py_float));
      }
   }

   // Parse RGB tuples
   for (int i = 0; i < 3; i++){
      faceColor[i] = PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, i));
      lineColor[i] = PyFloat_AsDouble(PyTuple_GetItem(lineColorPyTup, i));
   }

   // Initialize / Update Vertex Geometry and Colors
   if (  vertexBuffer == NULL || 
         colahbuffah == NULL || 
         indices == NULL || 
         prevNumBulbs != numBulbs ||
         prevAngOffset != angularOffset ||
         prevW2H != w2h){
      printf("Recalculating bulb button geometry\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      prevNumBulbs = numBulbs;
      prevAngOffset = angularOffset;
      prevW2H = w2h;

      // Set Number of edges on circles
      char circleSegments = 60;
      char degSegment = 360 / circleSegments;

      // Setup Transformations
      if (w2h <= 1.0)
      {
         scale = w2h*scale;
      }

      float tmx, tmy;
      // Define verts / colors for each bulb button
#     pragma omp parallel for
      for (int j = 0; j < numBulbs; j++) {
         tmx = float(0.75*cos(degToRad(j*360/numBulbs - 90 + angularOffset + 180/numBulbs)));
         tmy = float(0.75*sin(degToRad(j*360/numBulbs - 90 + angularOffset + 180/numBulbs)));
         if (w2h >= 1.0) {
            tmx *= float(pow(w2h, 0.5));
         } else {
            tmx *= float(w2h);
            tmy *= float(pow(w2h, 0.5));
         }
         // Define Vertices / Colors for Button Face
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ verts.push_back(float(tmx+0.0));
            /* Y */ verts.push_back(float(tmy+0.0));
            /* R */ colrs.push_back(float(faceColor[0]));
            /* G */ colrs.push_back(float(faceColor[1]));
            /* B */ colrs.push_back(float(faceColor[2]));

            /* X */ verts.push_back(float(tmx+0.4*cos(degToRad(i*degSegment))*scale));
            /* Y */ verts.push_back(float(tmy+0.4*sin(degToRad(i*degSegment))*scale));
            /* R */ colrs.push_back(float(faceColor[0]));
            /* G */ colrs.push_back(float(faceColor[1]));
            /* B */ colrs.push_back(float(faceColor[2]));

            /* X */ verts.push_back(float(tmx+0.4*cos(degToRad((i+1)*degSegment))*scale));
            /* Y */ verts.push_back(float(tmy+0.4*sin(degToRad((i+1)*degSegment))*scale));
            /* R */ colrs.push_back(float(faceColor[0]));
            /* G */ colrs.push_back(float(faceColor[1]));
            /* B */ colrs.push_back(float(faceColor[2]));
         }

         if (j == 0) {
            colorsStart = colrs.size();
         }
         // Define Vertices for Bulb Icon
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ verts.push_back(float(tmx+0.0*scale));
            /* Y */ verts.push_back(float(tmy+0.1*scale));
            /* R */ colrs.push_back(float(bulbColors[j*numBulbs+0]));
            /* G */ colrs.push_back(float(bulbColors[j*numBulbs+1]));
            /* B */ colrs.push_back(float(bulbColors[j*numBulbs+2]));

            /* X */ verts.push_back(float(tmx+0.2*cos(degToRad(i*degSegment))*scale));
            /* Y */ verts.push_back(float(tmy+(0.1+0.2*sin(degToRad(i*degSegment)))*scale));
            /* R */ colrs.push_back(float(bulbColors[j*numBulbs+0]));
            /* G */ colrs.push_back(float(bulbColors[j*numBulbs+1]));
            /* B */ colrs.push_back(float(bulbColors[j*numBulbs+2]));

            /* X */ verts.push_back(float(tmx+0.2*cos(degToRad((i+1)*degSegment))*scale));
            /* Y */ verts.push_back(float(tmy+(0.1+0.2*sin(degToRad((i+1)*degSegment)))*scale));
            /* R */ colrs.push_back(float(bulbColors[j*numBulbs+0]));
            /* G */ colrs.push_back(float(bulbColors[j*numBulbs+1]));
            /* B */ colrs.push_back(float(bulbColors[j*numBulbs+2]));
         }
         if (j == 0) {
            colorsEnd = colrs.size();
         }

         // Define Verts for bulb screw base
         GLfloat tmp[54] = {
            float(tmx-0.085*scale), float(tmy-0.085*scale),
            float(tmx+0.085*scale), float(tmy-0.085*scale),
            float(tmx+0.085*scale), float(tmy-0.119*scale),
            float(tmx-0.085*scale), float(tmy-0.085*scale),
            float(tmx+0.085*scale), float(tmy-0.119*scale),
            float(tmx-0.085*scale), float(tmy-0.119*scale),
   
            float(tmx+0.085*scale), float(tmy-0.119*scale),
            float(tmx-0.085*scale), float(tmy-0.119*scale),
            float(tmx-0.085*scale), float(tmy-0.153*scale),
   
            float(tmx+0.085*scale), float(tmy-0.136*scale),
            float(tmx-0.085*scale), float(tmy-0.170*scale),
            float(tmx-0.085*scale), float(tmy-0.204*scale),
            float(tmx+0.085*scale), float(tmy-0.136*scale),
            float(tmx+0.085*scale), float(tmy-0.170*scale),
            float(tmx-0.085*scale), float(tmy-0.204*scale),
   
            float(tmx+0.085*scale), float(tmy-0.187*scale),
            float(tmx-0.085*scale), float(tmy-0.221*scale),
            float(tmx-0.085*scale), float(tmy-0.255*scale),
            float(tmx+0.085*scale), float(tmy-0.187*scale),
            float(tmx+0.085*scale), float(tmy-0.221*scale),
            float(tmx-0.085*scale), float(tmy-0.255*scale),
   
            float(tmx+0.085*scale), float(tmy-0.238*scale),
            float(tmx-0.085*scale), float(tmy-0.272*scale),
            float(tmx-0.051*scale), float(tmy-0.306*scale),
            float(tmx+0.085*scale), float(tmy-0.238*scale),
            float(tmx+0.051*scale), float(tmy-0.306*scale),
            float(tmx-0.051*scale), float(tmy-0.306*scale),
         };
   
         for (int i = 0; i < 27; i++) {
            verts.push_back(float(tmp[i*2+0]));
            verts.push_back(float(tmp[i*2+1]));
            colrs.push_back(float(lineColor[0]));
            colrs.push_back(float(lineColor[1]));
            colrs.push_back(float(lineColor[2]));
         }

         if (j == 0) {
            bulbVerts = verts.size()/2;
         }
      }
      // Pack Vertices / Colors into global array buffers
      numVerts = verts.size()/2;

      // (Re)allocate vertex buffer
      if (vertexBuffer == NULL) {
         vertexBuffer = new GLfloat[numVerts*2];
      } else {
         delete [] vertexBuffer;
         vertexBuffer = new GLfloat[numVerts*2];
      }

      // (Re)allocate color buffer
      if (colahbuffah == NULL) {
         colahbuffah  = new GLfloat[verts.size()*3];
      } else {
         delete [] colahbuffah;
         colahbuffah = new GLfloat[verts.size()*3];
      }

      // (Re)allocate index array
      if (indices == NULL) {
         indices = new GLushort[numVerts];
      } else {
         delete [] indices;
         indices = new GLushort[numVerts];
      }

      // Pack indices, vertex and color bufferes
      for (unsigned int i = 0; i < numVerts; i++){
         vertexBuffer[i*2] = verts[i*2];
         vertexBuffer[i*2+1] = verts[i*2+1];
         indices[i] = i;
         colahbuffah[i*3+0] = colrs[i*3+0];
         colahbuffah[i*3+1] = colrs[i*3+1];
         colahbuffah[i*3+2] = colrs[i*3+2];
      }

      // (Re)allocate button colors
      if (prevColors == NULL) {
         prevColors   = new GLfloat[3*numBulbs+6];
      } else {
         delete [] prevColors;
         prevColors   = new GLfloat[3*numBulbs+6];
      }

      if (curnColors == NULL) {
         curnColors   = new GLfloat[3*numBulbs+6];
      } else {
         delete [] curnColors;
         curnColors   = new GLfloat[3*numBulbs+6];
      }

      for (int i = 0; i < 3; i++) {
         curnColors[0+i] = float(faceColor[i]);
         curnColors[3+i] = float(lineColor[i]);

         prevColors[0+i] = float(faceColor[i]);
         prevColors[3+i] = float(lineColor[i]);
      }
      for (int i = 0; i < 3*numBulbs; i++){
         curnColors[6+i] = float(bulbColors[i]);

         prevColors[6+i] = float(bulbColors[i]);
      }
      
   } 
   // Vertices / Geometry already calculated
   // Check if colors need to be updated
   else
   {
      /* Update current palate of colors
       * First 3 elements are face color
       * Next 3 elements are line color
       * Remaining elements are bulb colors
       */
      curnColors[0] = float(faceColor[0]);
      curnColors[1] = float(faceColor[1]);
      curnColors[2] = float(faceColor[2]);
      curnColors[3] = float(lineColor[0]);
      curnColors[4] = float(lineColor[1]);
      curnColors[5] = float(lineColor[2]);

      for (int i = 0; i < 3*numBulbs; i++){
         curnColors[6+i] = float(bulbColors[i]);
      }

      // See if any bulb colors need to be updated
#     pragma omp parallel for
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < numBulbs; j++) {
            if (curnColors[6+i+j*3] != prevColors[6+i+j*3]) {
               for (int k = 0; k < (colorsEnd-colorsStart)/3; k++) {
                  colahbuffah[ j*bulbVerts*3 + colorsStart + i + k*3 ] = curnColors[6+i+j*3];
               }
            }
         }
      }

      prevColors[0] = curnColors[0];
      prevColors[1] = curnColors[1];
      prevColors[2] = curnColors[2];
      prevColors[3] = curnColors[3];
      prevColors[4] = curnColors[4];
      prevColors[5] = curnColors[5];
      /*
      prevColors[0] = float(faceColor[0]);
      prevColors[1] = float(faceColor[1]);
      prevColors[2] = float(faceColor[2]);

      prevColors[3] = float(lineColor[0]);
      prevColors[4] = float(lineColor[1]);
      prevColors[5] = float(lineColor[2]);
      */

      for (int i = 0; i < 3*numBulbs; i++){
         prevColors[6+i] = curnColors[6+i];
         //prevColors[6+i] = float(bulbColors[i]);
      }
   } 

   // Cleanup
   delete [] bulbColors;
   
   // Copy Vertex / Color Array Bufferes to GPU, draw
   glColorPointer(3, GL_FLOAT, 0, colahbuffah);
   glVertexPointer(2, GL_FLOAT, 0, vertexBuffer);
   glDrawElements( GL_TRIANGLES, numVerts, GL_UNSIGNED_SHORT, indices);

   Py_RETURN_NONE;
}

static PyMethodDef drawButtons_methods[] = {
   { "drawBulbButton", (PyCFunction)drawBulbButton_drawButtons, METH_VARARGS },
   { NULL, NULL, 0, NULL}
};

static PyModuleDef drawButtons_module = {
   PyModuleDef_HEAD_INIT,
   "drawButtons",
   "Draws buttons",
   0,
   drawButtons_methods
};

/*
void initdrawBulbButton(void)
{
   Py_InitModule3("drawButtons", drawButtons_methods, "quack");
}
*/

PyMODINIT_FUNC PyInit_drawButtons() {
   //return PyModule_Create(&drawButtons_module);
   PyObject* m = PyModule_Create(&drawButtons_module);
   if (m == NULL) {
      return NULL;
   }
   return m;
}
