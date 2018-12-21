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
int      colorsStart   = NULL;
int      colorsEnd     = NULL;
PyObject* drawBulbButton_drawButtons(PyObject *self, PyObject *args)
{
   PyObject* faceColorPyTup;
   PyObject* lineColorPyTup;
   PyObject* bulbColorPyTup;
   double faceColor[3];
   double lineColor[3]; 
   double bulbColor[3];
   float gx, gy, scale, w2h, squash;

   // Parse input arguments
   if (!PyArg_ParseTuple(args, 
            "fffOOOf", 
            &gx,
            &gy,
            &scale,
            &faceColorPyTup,
            &lineColorPyTup,
            &bulbColorPyTup,
            &w2h))
   {
      Py_RETURN_NONE;
   }

   // Parse RGB tuples
   for (int i = 0; i < 3; i++){
      faceColor[i] = PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, i));
      lineColor[i] = PyFloat_AsDouble(PyTuple_GetItem(lineColorPyTup, i));
      bulbColor[i] = PyFloat_AsDouble(PyTuple_GetItem(bulbColorPyTup, i));
   }

   // Setup Transformations
   glPushMatrix();
   glTranslatef(gx, gy, 0);
   if (w2h <= 1.0)
   {
      glScalef(w2h*scale, w2h*scale, 1);
      //glLineWidth(w2h*3.0);
      //squash = w2h*scale;
   }
   else
   {
      glScalef(scale, scale, 1);
      //glLineWidth(pow(1.0/w2h, 0.5)*3.0);
      //squash = scale;
   }

   //glEnableClientState(GL_VERTEX_ARRAY);
   //glEnableClientState(GL_COLOR_ARRAY );

   // Set Number of edges on circles
   char circleSegments = 60;

   // Initialize / Update Vertex Geometry and Colors
   if (vertexBuffer == NULL || colahbuffah == NULL || indices == NULL){
      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      char degSegment = 360 / circleSegments;

      // Define Vertices / Colors for Button Face
      for (int i = 0; i < circleSegments+1; i++){
         /* X */ verts.push_back(float(0.0));
         /* Y */ verts.push_back(float(0.0));
         /* R */ colrs.push_back(float(faceColor[0]));
         /* G */ colrs.push_back(float(faceColor[1]));
         /* B */ colrs.push_back(float(faceColor[2]));

         /* X */ verts.push_back(float(0.4*cos(degToRad(i*degSegment))));
         /* Y */ verts.push_back(float(0.4*sin(degToRad(i*degSegment))));
         /* R */ colrs.push_back(float(faceColor[0]));
         /* G */ colrs.push_back(float(faceColor[1]));
         /* B */ colrs.push_back(float(faceColor[2]));

         /* X */ verts.push_back(float(0.4*cos(degToRad((i+1)*degSegment))));
         /* Y */ verts.push_back(float(0.4*sin(degToRad((i+1)*degSegment))));
         /* R */ colrs.push_back(float(faceColor[0]));
         /* G */ colrs.push_back(float(faceColor[1]));
         /* B */ colrs.push_back(float(faceColor[2]));
      }

      // Define Vertices for Bulb Icon
      colorsStart = colrs.size();
      printf("%.1i\n", colorsStart);
      for (int i = 0; i < circleSegments+1; i++){
         /* X */ verts.push_back(float(0.0));
         /* Y */ verts.push_back(float(0.1));
         /* R */ colrs.push_back(float(bulbColor[0]));
         /* G */ colrs.push_back(float(bulbColor[1]));
         /* B */ colrs.push_back(float(bulbColor[2]));

         /* X */ verts.push_back(float(0.2*cos(degToRad(i*degSegment))));
         /* Y */ verts.push_back(float(0.1+0.2*sin(degToRad(i*degSegment))));
         /* R */ colrs.push_back(float(bulbColor[0]));
         /* G */ colrs.push_back(float(bulbColor[1]));
         /* B */ colrs.push_back(float(bulbColor[2]));

         /* X */ verts.push_back(float(0.2*cos(degToRad((i+1)*degSegment))));
         /* Y */ verts.push_back(float(0.1+0.2*sin(degToRad((i+1)*degSegment))));
         /* R */ colrs.push_back(float(bulbColor[0]));
         /* G */ colrs.push_back(float(bulbColor[1]));
         /* B */ colrs.push_back(float(bulbColor[2]));
      }
      colorsEnd = colrs.size();
      printf("%.1i\n", colorsEnd);

      // Define Verts for bulb screw base
      GLfloat tmp[54] = {
         -0.085, -0.085,
         +0.085, -0.085,
         +0.085, -0.119,
         -0.085, -0.085,
         +0.085, -0.119,
         -0.085, -0.119,

         +0.085, -0.119,
         -0.085, -0.119,
         -0.085, -0.153,

         +0.085, -0.136,
         -0.085, -0.170,
         -0.085, -0.204,
         +0.085, -0.136,
         +0.085, -0.170,
         -0.085, -0.204,

         +0.085, -0.187,
         -0.085, -0.221,
         -0.085, -0.255,
         +0.085, -0.187,
         +0.085, -0.221,
         -0.085, -0.255,

         +0.085, -0.238,
         -0.085, -0.272,
         -0.051, -0.306,
         +0.085, -0.238,
         +0.051, -0.306,
         -0.051, -0.306,
      };

      for (int i = 0; i < 27; i++) {
         verts.push_back(float(tmp[i*2+0]));
         verts.push_back(float(tmp[i*2+1]));
         colrs.push_back(float(lineColor[0]));
         colrs.push_back(float(lineColor[1]));
         colrs.push_back(float(lineColor[2]));
      }

      // Pack Vertices / Colors into global array buffers
      numVerts = verts.size()/2;
      vertexBuffer = new GLfloat[verts.size()];
      colahbuffah  = new GLfloat[verts.size()*3];
      indices = new GLushort[numVerts];
      for (int i = 0; i < numVerts; i++){
         vertexBuffer[i*2] = verts[i*2];
         vertexBuffer[i*2+1] = verts[i*2+1];
         indices[i] = i;
         colahbuffah[i*3+0] = colrs[i*3+0];
         colahbuffah[i*3+1] = colrs[i*3+1];
         colahbuffah[i*3+2] = colrs[i*3+2];
      }
      prevColors   = new GLfloat[9];
      curnColors   = new GLfloat[9];
      for (int i = 0; i < 3; i++){
         prevColors[0+i] = float(faceColor[i]);
         prevColors[3+i] = float(bulbColor[i]);
         prevColors[6+i] = float(lineColor[i]);

         curnColors[0+i] = float(faceColor[i]);
         curnColors[3+i] = float(bulbColor[i]);
         curnColors[6+i] = float(lineColor[i]);
      }
   } 
   else
   {
      for (int i = 0; i < 3; i++){
         curnColors[0+i] = float(faceColor[i]);
         curnColors[3+i] = float(bulbColor[i]);
         curnColors[6+i] = float(lineColor[i]);
      }

      // Update Color(s) of Button Face 
      if (curnColors[0] != prevColors[0] &&
          curnColors[1] != prevColors[1] &&
          curnColors[2] != prevColors[2] ){

         for (int i = 0; i < colorsStart/3; i++){
            colahbuffah[i*3+0] = float(bulbColor[0]);
            colahbuffah[i*3+1] = float(bulbColor[1]);
            colahbuffah[i*3+2] = float(bulbColor[2]);
         }

         prevColors[0] = float(bulbColor[0]);
         prevColors[1] = float(bulbColor[1]);
         prevColors[2] = float(bulbColor[2]);
      }

      // Update Color(s) of Bulb Icon
      //if ((curnColors[3] != prevColors[3]) &&
          //(curnColors[4] != prevColors[4]) &&
          //(curnColors[5] != prevColors[5]) ){

         for (int i = colorsStart/3; i < colorsEnd/3; i++){
            colahbuffah[i*3+0] = float(bulbColor[0]);
            colahbuffah[i*3+1] = float(bulbColor[1]);
            colahbuffah[i*3+2] = float(bulbColor[2]);
         }

         prevColors[3] = curnColors[3];
         prevColors[4] = curnColors[4];
         prevColors[5] = curnColors[5];
         //printf("%.2f, %.2f, %.2f\n", float(bulbColor[0]), float(bulbColor[1]), float(bulbColor[2]));
      //}
   } 
   
   // Copy Vertex / Color Array Bufferes to GPU, draw
   glColorPointer(3, GL_FLOAT, 0, colahbuffah);
   glVertexPointer(2, GL_FLOAT, 0, vertexBuffer);
   glDrawElements( GL_TRIANGLES, numVerts, GL_UNSIGNED_SHORT, indices);
   glPopMatrix();

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
