#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>

using namespace std;

GLfloat*    confirmVertexBuffer  = NULL;
GLfloat*    confirmColorBuffer   = NULL;
GLushort*   confirmIndices       = NULL;
GLuint      confirmVerts;
GLuint      extraConfirmVerts;

PyObject* drawConfirm_drawButtons(PyObject* self, PyObject *args) {
   PyObject *faceColorPyTup;
   PyObject *extraColorPyTup;
   PyObject *detailColorPyTup;
   float posX, posY, scale, w2h;
   float faceColor[3];
   float extraColor[3];
   float detailColor[3];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "ffffOOO",
            &posX, &posY,
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
   if (  confirmVertexBuffer  == NULL  ||
         confirmColorBuffer   == NULL  ||
         confirmIndices       == NULL  ){

      printf("Initializing Geometry for Confirm Button\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      float px, py, qx, qy, radius;
      int circleSegments = 60;
      drawEllipse(0.0f, 0.0f, 1.0f, circleSegments, faceColor, verts, colrs);

      px = -0.75f, py =  0.0f;
      qx = -0.25f, qy = -0.5f;
      radius = float(sqrt(2.0)*0.125f);
      drawPill(px, py, qx, qy, radius, detailColor, verts, colrs);

      px = 0.625f, py = 0.375f;
      extraConfirmVerts = drawPill(px, py, qx, qy, radius, detailColor, verts, colrs);

      px = -0.75f, py =  0.0f;
      radius = 0.125f;
      drawPill(px, py, qx, qy, radius, extraColor, verts, colrs);

      px = 0.625f, py = 0.375f;
      drawPill(px, py, qx, qy, radius, extraColor, verts, colrs);

      confirmVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (confirmVertexBuffer == NULL) {
         confirmVertexBuffer = new GLfloat[confirmVerts*2];
      } else {
         delete [] confirmVertexBuffer;
         confirmVertexBuffer = new GLfloat[confirmVerts*2];
      }

      if (confirmColorBuffer == NULL) {
         confirmColorBuffer = new GLfloat[confirmVerts*3];
      } else {
         delete [] confirmColorBuffer;
         confirmColorBuffer = new GLfloat[confirmVerts*3];
      }

      if (confirmIndices == NULL) {
         confirmIndices = new GLushort[confirmVerts];
      } else {
         delete [] confirmIndices;
         confirmIndices = new GLushort[confirmVerts];
      }

      for (unsigned int i = 0; i < confirmVerts; i++) {
         confirmVertexBuffer[i*2]   = verts[i*2];
         confirmVertexBuffer[i*2+1] = verts[i*2+1];
         confirmIndices[i]          = i;
         confirmColorBuffer[i*3+0]  = colrs[i*3+0];
         confirmColorBuffer[i*3+1]  = colrs[i*3+1];
         confirmColorBuffer[i*3+2]  = colrs[i*3+2];
      }
   }

   for (int i = 0; i < 3; i++) {
      if ( confirmColorBuffer[extraConfirmVerts*3+i] != extraColor[i] ) {
         for (unsigned int k = extraConfirmVerts; k < confirmVerts; k++) {
            confirmColorBuffer[k*3 + i] = extraColor[i];
         }
      }
   }

   glPushMatrix();
   glTranslatef(posX*w2h, posY, 0.0f);
   if (w2h <= 1.0) {
         scale = scale*w2h;
   }

   glScalef(scale, scale, 1);
   glColorPointer(3, GL_FLOAT, 0, confirmColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, confirmVertexBuffer);
   glDrawElements( GL_TRIANGLES, confirmVerts, GL_UNSIGNED_SHORT, confirmIndices);
   glPopMatrix();

   Py_RETURN_NONE;
}
