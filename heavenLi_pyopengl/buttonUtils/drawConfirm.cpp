#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
   // These undefs necessary because microsoft
   #undef near
   #undef far
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#include <vector>
#include <math.h>

using namespace std;

GLfloat*    confirmVertexBuffer  = NULL;
GLfloat*    confirmColorBuffer   = NULL;
GLushort*   confirmIndices       = NULL;
GLuint      confirmVerts;
GLuint      extraConfirmVerts;
Matrix      confirmMVP;
Params      confirmPrevState;

PyObject* drawConfirm_drawButtons(PyObject* self, PyObject *args) {
   PyObject *faceColorPyTup;
   PyObject *extraColorPyTup;
   PyObject *detailColorPyTup;
   float gx, gy, scale, w2h;
   float faceColor[3];
   float extraColor[3];
   float detailColor[3];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "ffffOOO",
            &gx, &gy,
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

      // Calculate Initial Transformation Matrix
      Matrix Ortho;
      Matrix ModelView;

      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, 1.0f*gx, 1.0f*gy, 0.0f );
      if (w2h <= 1.0f) {
         MatrixScale( &ModelView, scale, scale*w2h, 1.0f );
      } else {
         MatrixScale( &ModelView, scale/w2h, scale, 1.0f );
      }
      //MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &confirmMVP, &ModelView, &Ortho );

      //confirmPrevState.ao = ao;
      confirmPrevState.dx = gx;
      confirmPrevState.dy = gy;
      confirmPrevState.sx = scale;
      confirmPrevState.sy = scale;
      confirmPrevState.w2h = w2h;
   }

   for (int i = 0; i < 3; i++) {
      if ( confirmColorBuffer[extraConfirmVerts*3+i] != extraColor[i] ) {
         for (unsigned int k = extraConfirmVerts; k < confirmVerts; k++) {
            confirmColorBuffer[k*3 + i] = extraColor[i];
         }
      }
   }

   // Old, Fixed-Function ES 1.1 code
   /*
   glPushMatrix();
   glTranslatef(gx*w2h, gy, 0.0f);
   if (w2h <= 1.0) {
         scale = scale*w2h;
   }

   glScalef(scale, scale, 1);
   glColorPointer(3, GL_FLOAT, 0, confirmColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, confirmVertexBuffer);
   glDrawElements( GL_TRIANGLES, confirmVerts, GL_UNSIGNED_SHORT, confirmIndices);
   glPopMatrix();
   */

   // Update Transfomation Matrix if any change in parameters
   if (  //confirmPrevState.ao != ao     ||
         confirmPrevState.dx != gx     ||
         confirmPrevState.dy != gy     ||
         confirmPrevState.sx != scale  ||
         confirmPrevState.sy != scale  ||
         confirmPrevState.w2h != w2h   ){
      
      Matrix Ortho;
      Matrix ModelView;

      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, 1.0f*gx, 1.0f*gy, 0.0f );
      if (w2h <= 1.0f) {
         MatrixScale( &ModelView, scale, scale*w2h, 1.0f );
      } else {
         MatrixScale( &ModelView, scale/w2h, scale, 1.0f );
      }
      //MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &confirmMVP, &ModelView, &Ortho );

      //confirmPrevState.ao = ao;
      confirmPrevState.dx = gx;
      confirmPrevState.dy = gy;
      confirmPrevState.sx = scale;
      confirmPrevState.sy = scale;
      confirmPrevState.w2h = w2h;
   }

   //GLint mvpLoc;
   //mvpLoc = glGetUniformLocation( 3, "MVP" );
   //glUniformMatrix4fv( mvpLoc, 1, GL_FALSE, &confirmMVP.mat[0][0] );
   glUniformMatrix4fv( 0, 1, GL_FALSE, &confirmMVP.mat[0][0] );
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, confirmVertexBuffer);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, confirmColorBuffer);
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, confirmVerts);

   Py_RETURN_NONE;
}
