#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>

using namespace std;

GLfloat*    arrowVertexBuffer = NULL;
GLfloat*    arrowColorBuffer   = NULL;
GLushort*   arrowIndices       = NULL;
GLuint      arrowVerts;
GLuint      extraArrowVerts;

PyObject* drawArrow_drawButtons(PyObject* self, PyObject *args) {
   PyObject *faceColorPyTup;
   PyObject *extraColorPyTup;
   PyObject *detailColorPyTup;
   float posX, posY, ang, scale, w2h;
   float faceColor[3];
   float extraColor[3];
   float detailColor[3];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "fffffOOO",
            &posX, &posY,
            &ang,
            &scale,
            &w2h,
            &faceColorPyTup,
            &extraColorPyTup,
            &detailColorPyTup) )
   {
      Py_RETURN_NONE;
   }

   faceColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 0)));
   faceColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 1)));
   faceColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 2)));

   extraColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 0)));
   extraColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 1)));
   extraColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(extraColorPyTup, 2)));

   detailColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0)));
   detailColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1)));
   detailColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2)));

   // Allocate and Define Geometry/Color buffers
   if (  arrowVertexBuffer == NULL  ||
         arrowColorBuffer  == NULL  ||
         arrowIndices      == NULL  ){

      printf("Initializing Geometry for Arrow Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      float px, py, qx, qy, radius;
      int circleSegments = 60;
      drawEllipse(0.0f, 0.0f, 1.0f, circleSegments, faceColor, verts, colrs);

      px = -0.125f, py = 0.625f;
      qx =  0.500f, qy =  0.00f;
      radius = float(sqrt(2.0)*0.125f);
      drawPill(px, py, qx, qy, radius, detailColor, verts, colrs);

      px = -0.125f, py = -0.625f;
      extraArrowVerts = drawPill(px, py, qx, qy, radius, detailColor, verts, colrs);

      px = -0.125f, py =  0.625f;
      radius = 0.125f;
      drawPill(px, py, qx, qy, radius, extraColor, verts, colrs);

      px = -0.125f, py = -0.625f;
      drawPill(px, py, qx, qy, radius, extraColor, verts, colrs);

      arrowVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (arrowVertexBuffer == NULL) {
         arrowVertexBuffer = new GLfloat[arrowVerts*2];
      } else {
         delete [] arrowVertexBuffer;
         arrowVertexBuffer = new GLfloat[arrowVerts*2];
      }

      if (arrowColorBuffer == NULL) {
         arrowColorBuffer = new GLfloat[arrowVerts*3];
      } else {
         delete [] arrowColorBuffer;
         arrowColorBuffer = new GLfloat[arrowVerts*3];
      }

      if (arrowIndices == NULL) {
         arrowIndices = new GLushort[arrowVerts];
      } else {
         delete [] arrowIndices;
         arrowIndices = new GLushort[arrowVerts];
      }

      for (unsigned int i = 0; i < arrowVerts; i++) {
         arrowVertexBuffer[i*2]   = verts[i*2];
         arrowVertexBuffer[i*2+1] = verts[i*2+1];
         arrowIndices[i]          = i;
         arrowColorBuffer[i*3+0]  = colrs[i*3+0];
         arrowColorBuffer[i*3+1]  = colrs[i*3+1];
         arrowColorBuffer[i*3+2]  = colrs[i*3+2];
      }
   }

   for (int i = 0; i < 3; i++) {
      if ( arrowColorBuffer[extraArrowVerts*3+i] != extraColor[i] ) {
         for (unsigned int k = extraArrowVerts; k < arrowVerts; k++) {
            arrowColorBuffer[k*3 + i] = extraColor[i];
         }
      }
   }

   glPushMatrix();
   glTranslatef(posX*w2h, posY, 0.0f);
   if (w2h <= 1.0) {
         scale = scale*w2h;
   }

   glScalef(scale, scale, 1);
   glRotatef(ang, 0.0, 0.0, 1.0);
   glColorPointer(3, GL_FLOAT, 0, arrowColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, arrowVertexBuffer);
   glDrawElements( GL_TRIANGLES, arrowVerts, GL_UNSIGNED_SHORT, arrowIndices);
   glPopMatrix();

   Py_RETURN_NONE;
}
