#include <Python.h>
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <gl/GL.h>
#include <vector>
#include <math.h>
#define degToRad(angleInDegrees) ((angleInDegrees) * 3.1415926535 / 180.0)
using namespace std;

PyObject* drawBulbButton_drawButtons(PyObject *self, PyObject *args)
//void drawBulbButton_drawButtons(PyObject *self, PyObject *args)
{
   PyObject* faceColorPyTup;
   PyObject* lineColorPyTup;
   PyObject* bulbColorPyTup;
   double faceColor[3];
   double lineColor[3]; 
   double bulbColor[3];
   float gx, gy, scale, w2h, squash;
   //float bulbVerts[606];
   //GLfloat bulbVerts[270][2];
   //vector< tuple<GLfloat, GLfloat> verts;
   GLfloat bulbVerts[141][2];
   GLfloat bulbColrs[141][3];

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
   else
   {
      //printf("%.6f\n", w2h);
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
      glLineWidth(w2h*3.0);
      //squash = w2h*scale;
   }
   else
   {
      glScalef(scale, scale, 1);
      glLineWidth(pow(1.0/w2h, 0.5)*3.0);
      //squash = scale;
   }

   /*
   glBegin(GL_TRIANGLES);
   glColor3f(faceColor[0], faceColor[1], faceColor[2]);
   for (int i = 0; i < 46; i++){
      	//glVertex2f(gx + 0.0, gy + 0.0);
      	//glVertex2f(gx + squash*0.4*cos(degToRad(i*8)), gy + squash*0.4*sin(degToRad(i*8)));
      	//glVertex2f(gx + squash*0.4*cos(degToRad((i+1)*8)), gy + squash*0.4*sin(degToRad((i+1)*8)));
      	glVertex2f(0.0, 0.0);
      	glVertex2f(0.4*cos(degToRad(i*8)), 0.4*sin(degToRad(i*8)));
      	glVertex2f(0.4*cos(degToRad((i+1)*8)), 0.4*sin(degToRad((i+1)*8)));
   }
   glEnd();
   */
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_COLOR_ARRAY );
   glColor3f(faceColor[0], faceColor[1], faceColor[2]);
   for (int i = 0; i < 46; i++){
      /*
      verts.push_back({0.0, 0.0});
      verts.push_back({0.4*cos(degToRad(i*8)), 0.4*sin(degToRad(i*8))});
      verts.push_back({0.4*cos(degToRad((i+1)*8)), 0.4*sin(degToRad((i+1)*8))});
      */

      /*
      verts.push_back(0.0);
      verts.push_back(0.0);
      verts.push_back(0.4*cos(degToRad(i*8)));
      verts.push_back(0.4*sin(degToRad(i*8)));
      verts.push_back(0.4*cos(degToRad((i+1)*8)));
      verts.push_back(0.4*sin(degToRad((i+1)*8)));
      */

      bulbVerts[i*3][0] = 0.0;
      bulbVerts[i*3][1] = 0.0;
      bulbVerts[i*3+1][0] = 0.4*cos(degToRad(i*8));
      bulbVerts[i*3+1][1] = 0.4*sin(degToRad(i*8));
      bulbVerts[i*3+2][0] = 0.4*cos(degToRad((i+1)*8)); 
      bulbVerts[i*3+2][1] = 0.4*sin(degToRad((i+1)*8));
   }
   //GLfloat *bulbVerts = &verts[0];
   /*
   float** bulbVerts = new float*[verts.size()/2];
   for (int i = 0; i < verts.size()/2; i++){
      bulbVerts[i] = new float[2];
      bulbVerts[i][0] = verts[i*2];
      bulbVerts[i][1] = verts[i*2+1];
   }
   */
   
   //int * indices = new int [verts.size()/2];
   GLubyte indices[141];
   for (int i = 0; i < 141; i++){
      indices[i] = i;
      bulbColrs[i][0] = faceColor[0];
      bulbColrs[i][1] = faceColor[1];
      bulbColrs[i][2] = faceColor[2];
   }

   glColorPointer(3, GL_FLOAT, 0, bulbColrs);
   glVertexPointer(2, GL_FLOAT, 0, bulbVerts);
   glDrawElements( GL_TRIANGLES, 135, GL_UNSIGNED_BYTE, indices);
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
