#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
#endif
#include <GL/gl.h>
#include <vector>
#include <math.h>

using namespace std;

GLfloat  *clockVertexBuffer = NULL;
GLfloat  *clockColorBuffer  = NULL;
GLushort *clockIndices      = NULL;
GLuint    clockVerts;
GLuint    faceVerts;
float     prevClockScale;
float     prevClockw2h;
float     prevClockHour;
float     prevClockMinute;
PyObject* drawClock_drawButtons(PyObject *self, PyObject *args)
{
   PyObject* faceColorPyTup;
   PyObject* detailColorPyTup;
   GLfloat px, py, qx, qy, radius;
   float scale, w2h, hour, minute;
   double detailColor[3];
   double faceColor[3];

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "ffffOO",
            &hour,
            &minute,
            &scale,
            &w2h,
            &detailColorPyTup,
            &faceColorPyTup)) 
   {
      Py_RETURN_NONE;
   }

   // Parse RGB color tuples of face and detail colors
   for (int i = 0; i < 3; i++){
      faceColor[i] = PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, i));
      detailColor[i] = PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, i));
   }
   
   if (  clockVertexBuffer == NULL  ||
         clockColorBuffer  == NULL  ||
         clockIndices      == NULL  ||
         prevClockScale    != scale ||
         prevClockw2h      != w2h   
         ){

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      // Set Number of edges on circles
      char circleSegments = 60;
      char degSegment = 360 / circleSegments;

      if (w2h <= 1.0)
      {
         scale = scale*w2h;
      }

      //float tmx, tmy, ang;
      for (int i = 0; i < circleSegments+1; i++) {
         /* X */ verts.push_back(float(0.0));
         /* Y */ verts.push_back(float(0.0));
         /* R */ colrs.push_back(float(faceColor[0]));
         /* G */ colrs.push_back(float(faceColor[1]));
         /* B */ colrs.push_back(float(faceColor[2]));

         /* X */ verts.push_back(float(0.5*cos(degToRad(i*degSegment))*scale));
         /* Y */ verts.push_back(float(0.5*sin(degToRad(i*degSegment))*scale));
         /* R */ colrs.push_back(float(faceColor[0]));
         /* G */ colrs.push_back(float(faceColor[1]));
         /* B */ colrs.push_back(float(faceColor[2]));

         /* X */ verts.push_back(float(0.5*cos(degToRad((i+1)*degSegment))*scale));
         /* Y */ verts.push_back(float(0.5*sin(degToRad((i+1)*degSegment))*scale));
         /* R */ colrs.push_back(float(faceColor[0]));
         /* G */ colrs.push_back(float(faceColor[1]));
         /* B */ colrs.push_back(float(faceColor[2]));
      }

      faceVerts = verts.size()/2;

      px = 0.0;
      py = 0.0;
      qx = float(0.2*cos(degToRad(90-360*(hour/12.0)))*scale);
      qy = float(0.2*sin(degToRad(90-360*(hour/12.0)))*scale);
      radius = float(0.02*scale);
      drawPill(px, py, qx, qy, radius, detailColor, verts, colrs);

      qx = float(0.4*cos(degToRad(90-360*(minute/60.0)))*scale);
      qy = float(0.4*sin(degToRad(90-360*(minute/60.0)))*scale);
      radius = float(0.01*scale);
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

      prevClockScale    = scale;
      prevClockw2h      = w2h;
      prevClockHour     = hour;
      prevClockMinute   = minute;
   } else if (
         prevClockHour     != hour  ||
         prevClockMinute   != minute
         ){
      px = 0.0;
      py = 0.0;
      qx = float(0.2*cos(degToRad(90-360*(hour/12.0)))*scale);
      qy = float(0.2*sin(degToRad(90-360*(hour/12.0)))*scale);
      radius = float(0.02*scale);

      int tmp;
      tmp = drawPill(
            px, py, 
            qx, qy, 
            radius, 
            faceVerts, 
            //faceColor, 
            detailColor, 
            clockVertexBuffer, 
            clockColorBuffer);

      qx = float(0.4*cos(degToRad(90-360*(minute/60.0)))*scale);
      qy = float(0.4*sin(degToRad(90-360*(minute/60.0)))*scale);
      radius = float(0.01*scale);

      tmp = drawPill(
            px, py, 
            qx, qy, 
            radius, 
            tmp, 
            //faceColor, 
            detailColor, 
            clockVertexBuffer, 
            clockColorBuffer);

      prevClockHour     = hour;
      prevClockMinute   = minute;
   }

   glColorPointer(3, GL_FLOAT, 0, clockColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, clockVertexBuffer);
   glDrawElements( GL_TRIANGLES, clockVerts, GL_UNSIGNED_SHORT, clockIndices);

   Py_RETURN_NONE;
}
