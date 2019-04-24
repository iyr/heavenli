#include <Python.h>
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

GLfloat*    clockVertexBuffer = NULL;
GLfloat*    clockColorBuffer  = NULL;
GLushort*   clockIndices      = NULL;
GLuint      clockVerts;
GLuint      faceVerts;
float       prevClockHour;
float       prevClockMinute;
Matrix      clockMVP;
Params      clockPrevState;

PyObject* drawClock_drawButtons(PyObject *self, PyObject *args)
{
   PyObject* faceColorPyTup;
   PyObject* detailColorPyTup;
   GLfloat gx, gy, px, py, qx, qy, radius;
   GLfloat scale, w2h, hour, minute;
   GLfloat detailColor[3];
   GLfloat faceColor[3];
   // Set Number of edges on circles
   char circleSegments = 60;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "ffffffOO",
            &gx, &gy,
            &hour,
            &minute,
            &scale,
            &w2h,
            &faceColorPyTup,
            &detailColorPyTup))
   {
      Py_RETURN_NONE;
   }

   // Parse RGB color tuples of face and detail colors
   for (int i = 0; i < 3; i++){
      faceColor[i] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, i)));
      detailColor[i] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, i)));
   }
   
   if (  clockVertexBuffer == NULL  ||
         clockColorBuffer  == NULL  ||
         clockIndices      == NULL  ){

      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      float R, G, B;
      R = float(faceColor[0]);
      G = float(faceColor[1]);
      B = float(faceColor[2]);

      char degSegment = 360 / circleSegments;

      for (int i = 0; i < circleSegments; i++) {
         /* X */ verts.push_back(float(0.0));
         /* Y */ verts.push_back(float(0.0));
         /* X */ verts.push_back(float(0.5*cos(degToRad(i*degSegment))));
         /* Y */ verts.push_back(float(0.5*sin(degToRad(i*degSegment))));
         /* X */ verts.push_back(float(0.5*cos(degToRad((i+1)*degSegment))));
         /* Y */ verts.push_back(float(0.5*sin(degToRad((i+1)*degSegment))));

         /* R */ colrs.push_back(R);
         /* G */ colrs.push_back(G);
         /* B */ colrs.push_back(B);
         /* R */ colrs.push_back(R);
         /* G */ colrs.push_back(G);
         /* B */ colrs.push_back(B);
         /* R */ colrs.push_back(R);
         /* G */ colrs.push_back(G);
         /* B */ colrs.push_back(B);
      }

      faceVerts = verts.size()/2;

      px = 0.0;
      py = 0.0;
      qx = float(0.2*cos(degToRad(90-360*(hour/12.0))));
      qy = float(0.2*sin(degToRad(90-360*(hour/12.0))));
      radius = float(0.02);
      drawPill(px, py, qx, qy, radius, detailColor, verts, colrs);

      qx = float(0.4*cos(degToRad(90-360*(minute/60.0))));
      qy = float(0.4*sin(degToRad(90-360*(minute/60.0))));
      radius = float(0.01);
      drawPill(px, py, qx, qy, radius, detailColor, verts, colrs);

      clockVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (clockVertexBuffer == NULL) {
         clockVertexBuffer = new GLfloat[clockVerts*2];
      } else {
         delete [] clockVertexBuffer;
         clockVertexBuffer = new GLfloat[clockVerts*2];
      }

      if (clockColorBuffer == NULL) {
         clockColorBuffer = new GLfloat[clockVerts*3];
      } else {
         delete [] clockColorBuffer;
         clockColorBuffer = new GLfloat[clockVerts*3];
      }

      if (clockIndices == NULL) {
         clockIndices = new GLushort[clockVerts];
      } else {
         delete [] clockIndices;
         clockIndices = new GLushort[clockVerts];
      }

      for (unsigned int i = 0; i < clockVerts; i++) {
         clockVertexBuffer[i*2]   = verts[i*2];
         clockVertexBuffer[i*2+1] = verts[i*2+1];
         clockIndices[i]          = i;
         clockColorBuffer[i*3+0]  = colrs[i*3+0];
         clockColorBuffer[i*3+1]  = colrs[i*3+1];
         clockColorBuffer[i*3+2]  = colrs[i*3+2];
      }

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
      MatrixMultiply( &clockMVP, &ModelView, &Ortho );

      //clockPrevState.ao = ao;
      clockPrevState.dx = gx;
      clockPrevState.dy = gy;
      clockPrevState.sx = scale;
      clockPrevState.sy = scale;
      clockPrevState.w2h = w2h;

      prevClockHour     = hour;
      prevClockMinute   = minute;
   } 

   if (prevClockHour     != hour    ||
       prevClockMinute   != minute  ){
      px = 0.0;
      py = 0.0;
      qx = float(0.2*cos(degToRad(90-360*(hour/12.0))));
      qy = float(0.2*sin(degToRad(90-360*(hour/12.0))));
      radius = float(0.02);

      int tmp;
      tmp = drawPill(
            px, py, 
            qx, qy, 
            radius, 
            faceVerts, 
            detailColor, 
            clockVertexBuffer, 
            clockColorBuffer);

      qx = float(0.4*cos(degToRad(90-360*(minute/60.0))));
      qy = float(0.4*sin(degToRad(90-360*(minute/60.0))));
      radius = float(0.01);
      tmp = drawPill(
            px, py, 
            qx, qy, 
            radius, 
            tmp, 
            detailColor, 
            clockVertexBuffer, 
            clockColorBuffer);

      prevClockHour     = hour;
      prevClockMinute   = minute;
   }

   // Geometry up to date, check if colors need to be updated
   for (int i = 0; i < 3; i++) {
      if (faceColor[i] != clockColorBuffer[i]) {
         for (int k = 0; k < circleSegments*3; k++) {
            clockColorBuffer[i + k*3] = faceColor[i];
         }
      }
   }
   
   // Old, Fixed-Function ES 1.1 code
   /*
   glPushMatrix();
   if (w2h <= 1.0) {
         scale = scale*w2h;
   }
   glScalef(scale, scale, 1);
   glColorPointer(3, GL_FLOAT, 0, clockColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, clockVertexBuffer);
   glDrawElements( GL_TRIANGLES, clockVerts, GL_UNSIGNED_SHORT, clockIndices);
   glPopMatrix();
   */

   // Update Transfomation Matrix if any change in parameters
   if (  //clockPrevState.ao != ao     ||
         clockPrevState.dx != gx     ||
         clockPrevState.dy != gy     ||
         clockPrevState.sx != scale  ||
         clockPrevState.sy != scale  ||
         clockPrevState.w2h != w2h   ){
      
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
      MatrixMultiply( &clockMVP, &ModelView, &Ortho );

      //clockPrevState.ao = ao;
      clockPrevState.dx = gx;
      clockPrevState.dy = gy;
      clockPrevState.sx = scale;
      clockPrevState.sy = scale;
      clockPrevState.w2h = w2h;
   }

   //GLint mvpLoc;
   //mvpLoc = glGetUniformLocation( 3, "MVP" );
   //glUniformMatrix4fv( mvpLoc, 1, GL_FALSE, &clockMVP.mat[0][0] );
   glUniformMatrix4fv( 0, 1, GL_FALSE, &clockMVP.mat[0][0] );
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, clockVertexBuffer);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, clockColorBuffer);
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, clockVerts);

   Py_RETURN_NONE;
}
