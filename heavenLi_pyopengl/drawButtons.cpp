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
float*   buttonCoords  = NULL;
int      colorsStart   = NULL;
int      colorsEnd     = NULL;
int      lineEnd       = NULL;
int      prevNumBulbs  = NULL;
int      prevArn       = NULL;
float    prevAngOffset = NULL;
float    prevScale     = NULL;
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
         buttonCoords == NULL ||
         prevNumBulbs != numBulbs
         ) {

      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      // Set Number of edges on circles
      char circleSegments = 60;
      char degSegment = 360 / circleSegments;

      // Setup Transformations
      if (w2h <= 1.0)
      {
         scale = w2h*scale;
      }

      if (buttonCoords == NULL) {
         buttonCoords = new float[2*numBulbs];
      } else {
         delete [] buttonCoords;
         buttonCoords = new float[2*numBulbs];
      }

      float tmx, tmy, ang;
      // Define verts / colors for each bulb button
#     pragma omp parallel for
      for (int j = 0; j < numBulbs; j++) {
         if (arn == 0) {
            ang = float(degToRad(j*360/numBulbs - 90 + angularOffset + 180/numBulbs));
         } else if (arn == 1) {
            ang = float(degToRad(
                  (j*180)/(numBulbs-1 < 1 ? 1 : numBulbs-1) + 
                  angularOffset + 
                  (numBulbs == 1 ? -90 : 0)
                  ));
         }

         // Relative coordinates of each button (from the center of the circle)
         tmx = float(0.75*cos(ang));
         tmy = float(0.75*sin(ang));
         if (w2h >= 1.0) {
            tmx *= float(pow(w2h, 0.5));
         } else {
            tmx *= float(w2h);
            tmy *= float(pow(w2h, 0.5));
         }

         buttonCoords[j*2+0] = tmx;
         buttonCoords[j*2+1] = tmy;

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
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            /* X */ verts.push_back(float(tmx+0.2*cos(degToRad(i*degSegment))*scale));
            /* Y */ verts.push_back(float(tmy+(0.1+0.2*sin(degToRad(i*degSegment)))*scale));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            /* X */ verts.push_back(float(tmx+0.2*cos(degToRad((i+1)*degSegment))*scale));
            /* Y */ verts.push_back(float(tmy+(0.1+0.2*sin(degToRad((i+1)*degSegment)))*scale));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));
         }
         if (j == 0) {
            colorsEnd = colrs.size();
         }

         // Define Verts for bulb screw base
         GLfloat tmp[54] = {
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.119*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.153*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.136*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.170*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.204*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.136*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.170*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.204*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.187*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.221*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.255*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.187*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.221*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.255*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.238*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.272*scale),
            /* X, Y */ float(tmx-0.051*scale), float(tmy-0.306*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.238*scale),
            /* X, Y */ float(tmx+0.051*scale), float(tmy-0.306*scale),
            /* X, Y */ float(tmx-0.051*scale), float(tmy-0.306*scale),
         };
   
         for (int i = 0; i < 27; i++) {
            /* X */ verts.push_back(float(tmp[i*2+0]));
            /* Y */ verts.push_back(float(tmp[i*2+1]));
            /* R */ colrs.push_back(float(lineColor[0]));
            /* G */ colrs.push_back(float(lineColor[1]));
            /* B */ colrs.push_back(float(lineColor[2]));
         }

         if (j == 0) {
            bulbVerts = verts.size()/2;
            lineEnd = colrs.size();
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
         colahbuffah = new GLfloat[numVerts*3];
      } else {
         delete [] colahbuffah;
         colahbuffah = new GLfloat[numVerts*3];
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

      prevNumBulbs = numBulbs;
      prevAngOffset = angularOffset;
      prevW2H = w2h;
      prevArn = arn;
      prevScale = scale;

   } 
   // Recalculate vertex geometry without expensieve reallocation
   else if (
         prevW2H != w2h ||
         prevArn != arn ||
         prevAngOffset != angularOffset ||
         prevScale != scale
         ) {
      // Set Number of edges on circles
      char circleSegments = 60;
      char degSegment = 360 / circleSegments;

      // Setup Transformations
      if (w2h <= 1.0)
      {
         scale = w2h*scale;
      }

      float tmx, tmy, ang;
      // Define verts / colors for each bulb button
#     pragma omp parallel for
      for (int j = 0; j < numBulbs; j++) {
         if (arn == 0) {
            ang = float(degToRad(j*360/numBulbs - 90 + angularOffset + 180/numBulbs));
         } else if (arn == 1) {
            ang = float(degToRad(
                  (j*180)/(numBulbs-1 < 1 ? 1 : numBulbs-1) + 
                  angularOffset + 
                  (numBulbs == 1 ? -90 : 0)
                  ));
         }

         // Relative coordinates of each button (from the center of the circle)
         tmx = float(0.75*cos(ang));
         tmy = float(0.75*sin(ang));
         if (w2h >= 1.0) {
            tmx *= float(pow(w2h, 0.5));
         } else {
            tmx *= float(w2h);
            tmy *= float(pow(w2h, 0.5));
         }

         buttonCoords[j*2+0] = tmx;
         buttonCoords[j*2+1] = tmy;

         // Define Vertices / Colors for Button Face
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ vertexBuffer[j*bulbVerts*2+i*6+0] = (float(tmx+0.0));
            /* Y */ vertexBuffer[j*bulbVerts*2+i*6+1] = (float(tmy+0.0));

            /* X */ vertexBuffer[j*bulbVerts*2+i*6+2] = (float(tmx+0.4*cos(degToRad(i*degSegment))*scale));
            /* Y */ vertexBuffer[j*bulbVerts*2+i*6+3] = (float(tmy+0.4*sin(degToRad(i*degSegment))*scale));

            /* X */ vertexBuffer[j*bulbVerts*2+i*6+4] = (float(tmx+0.4*cos(degToRad((i+1)*degSegment))*scale));
            /* Y */ vertexBuffer[j*bulbVerts*2+i*6+5] = (float(tmy+0.4*sin(degToRad((i+1)*degSegment))*scale));
         }

         // Define Vertices for Bulb Icon
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ vertexBuffer[j*bulbVerts*2+(circleSegments+1)*6+i*6+0] = (float(tmx+0.0*scale));
            /* Y */ vertexBuffer[j*bulbVerts*2+(circleSegments+1)*6+i*6+1] = (float(tmy+0.1*scale));

            /* X */ vertexBuffer[j*bulbVerts*2+(circleSegments+1)*6+i*6+2] = (float(tmx+0.2*cos(degToRad(i*degSegment))*scale));
            /* Y */ vertexBuffer[j*bulbVerts*2+(circleSegments+1)*6+i*6+3] = (float(tmy+(0.1+0.2*sin(degToRad(i*degSegment)))*scale));

            /* X */ vertexBuffer[j*bulbVerts*2+(circleSegments+1)*6+i*6+4] = (float(tmx+0.2*cos(degToRad((i+1)*degSegment))*scale));
            /* Y */ vertexBuffer[j*bulbVerts*2+(circleSegments+1)*6+i*6+5] = (float(tmy+(0.1+0.2*sin(degToRad((i+1)*degSegment)))*scale));
         }

         // Define Verts for bulb screw base
         GLfloat tmp[54] = {
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.085*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.119*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.119*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.153*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.136*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.170*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.204*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.136*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.170*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.204*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.187*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.221*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.255*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.187*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.221*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.255*scale),
   
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.238*scale),
            /* X, Y */ float(tmx-0.085*scale), float(tmy-0.272*scale),
            /* X, Y */ float(tmx-0.051*scale), float(tmy-0.306*scale),
            /* X, Y */ float(tmx+0.085*scale), float(tmy-0.238*scale),
            /* X, Y */ float(tmx+0.051*scale), float(tmy-0.306*scale),
            /* X, Y */ float(tmx-0.051*scale), float(tmy-0.306*scale),
         };
   
         for (int i = 0; i < 27; i++) {
            /* X */ vertexBuffer[j*bulbVerts*2+i*2+(circleSegments+1)*12+0] = (float(tmp[i*2+0]));
            /* Y */ vertexBuffer[j*bulbVerts*2+i*2+(circleSegments+1)*12+1] = (float(tmp[i*2+1]));
         }
      }

      prevAngOffset = angularOffset;
      prevW2H = w2h;
      prevArn = arn;
      prevScale = scale;
   }
   // Vertices / Geometry already calculated
   // Check if colors need to be updated
   else
   {
      for (int i = 0; i < 3; i++) {
         // Update face color, if needed
         if (float(faceColor[i]) != colahbuffah[i]) {
            //printf("Updating Face color\n");
            for (int j = 0; j < numBulbs; j++) {
               for (int k = 0; k < colorsStart/3; k++) {
                  colahbuffah[ j*bulbVerts*3 + k*3 + i ] = float(faceColor[i]);
               }
            }
         }

         // Update Line Color, if needed
         if (float(lineColor[i]) != colahbuffah[colorsEnd+i]) {
            //printf("Updating Line Color\n");
            for (int j = 0; j < numBulbs; j++) {
               for (int k = 0; k < (lineEnd - colorsEnd)/3; k++) {
                  colahbuffah[ colorsEnd + j*bulbVerts*3 + k*3 + i ] = float(lineColor[i]);
               }
            }
         }
      }
      
      // Update any bulb colors, if needed
#     pragma omp parallel for
      // Iterate through colors (R0, G1, B2)
      for (int i = 0; i < 3; i++) {

         // Iterate though bulbs
         for (int j = 0; j < numBulbs; j++) {

            // Iterate through color buffer to update colors
            if (float(bulbColors[i+j*3]) != colahbuffah[colorsStart + i + j*bulbVerts*3]) {
               //printf("Updating Color %.i on Bulb %.i\n", i, j);
               for (int k = 0; k < (colorsEnd-colorsStart)/3; k++) {
                  colahbuffah[ j*bulbVerts*3 + colorsStart + i + k*3 ] = float(bulbColors[i+j*3]);
               }
            }
         }
      }
   } 

   PyList_ClearFreeList();
   py_list = PyList_New(numBulbs);
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyTuple_New(2);
      PyTuple_SetItem(py_tuple, 0, PyFloat_FromDouble(buttonCoords[i*2+0]));
      PyTuple_SetItem(py_tuple, 1, PyFloat_FromDouble(buttonCoords[i*2+1]));
      PyList_SetItem(py_list, i, py_tuple);
      //PyList_Append(py_list, py_tuple);
   }

   // Cleanup
   delete [] bulbColors;
   
   // Copy Vertex / Color Array Bufferes to GPU, draw
   glColorPointer(3, GL_FLOAT, 0, colahbuffah);
   glVertexPointer(2, GL_FLOAT, 0, vertexBuffer);
   glDrawElements( GL_TRIANGLES, numVerts, GL_UNSIGNED_SHORT, indices);

   return py_list;
   //Py_RETURN_NONE;
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
