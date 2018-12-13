#include <Python.h>
#include <windows.h>
#include <gl/GL.h>
#include <gl/glu.h>
#include <math.h>

inline double to_degrees(double radians) {
       return radians * (180.0 / 3.1415926535);
}

static PyObject* drawBulbButton_drawButtons(PyObject *self, PyObject *args)
//void drawBulbButton_drawButtons(PyObject *self, PyObject *args)
{
   PyObject* faceColorPyTup;
   PyObject* lineColorPyTup;
   PyObject* bulbColorPyTup;
   float gx, gy, scale, w2h;
   float faceColor[3];
   float lineColor[3]; 
   float bulbColor[3];
   //float bulbVerts[606];
   GLfloat bulbVerts[270];

   // Parse input arguments
   if (!PyArg_ParseTuple(args, 
            "dddOOOd", 
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
      glLineWidth(w2h*3.0);
      float squash = w2h;
   }
   else
   {
      glScalef(scale, scale, 1);
      glLineWidth(pow(1.0/w2h, 0.5)*3.0);
      float squash = 1.0;
   }

   /*
   for (int i = 0; i < 46; i += 6){
      bulbVerts[i]   = 0.0;
      bulbVerts[i+1] = 0.0;
      bulbVerts[i+2] = 0.4*cos(to_degrees(i*8));
      bulbVerts[i+3] = 0.4*sin(to_degrees(i*8));
      bulbVerts[i+4] = 0.4*cos(to_degrees((i+1)*8));
      bulbVerts[i+5] = 0.4*sin(to_degrees((i+1)*8));

      bulbVerts[50+i]   = 0.0;
      bulbVerts[50+i+1] = 0.05;
      bulbVerts[50+i+2] = 0.2*cos(to_degrees(i*8));
      bulbVerts[50+i+3] = 0.2*sin(to_degrees(i*8)+0.1);
      bulbVerts[50+i+4] = 0.2*cos(to_degrees((i+1)*8));
      bulbVerts[50+i+5] = 0.2*sin(to_degrees((i+1)*8)+0.1);
   }
   GLfloat indices[270];
   for (int i = 0; i < 46; i++)
   {
      indices[i+0] = i;
      indices[i+1] = i;
      indices[i+2] = i+2;
      indices[i+3] = i+2;
      indices[i+4] = i+3;
      indices[i+5] = i+3;
   }
   for (int i = 0; i < 46; i += 6){
      bulbVerts[i]   = 0.0;
      bulbVerts[i+1] = 0.0;
      bulbVerts[i+2] = 0.4*cos(to_degrees(i*8));
      bulbVerts[i+3] = 0.4*sin(to_degrees(i*8));
      bulbVerts[i+4] = 0.4*cos(to_degrees((i+1)*8));
      bulbVerts[i+5] = 0.4*sin(to_degrees((i+1)*8));
   }
   */
   glColor3f(faceColor[0], faceColor[1], faceColor[2]);

   glBegin(GL_TRIANGLES);
   for (int i = 0; i < 46; i += 6){
      glVertex2f(0.0, 0.0);
      glVertex2f(0.4*cos(to_degrees(i*8)), 0.4*sin(to_degrees(i*8)));
      glVertex2f(0.4*cos(to_degrees((i+1)*8)), 0.4*sin(to_degrees((i+1)*8)));
   }
   glEnd();

   //glEnableClientState(GL_VERTEX_ARRAY);
   //glDrawElements( GL_TRIANGLES, 45, GLfloat, indices);

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
