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

GLfloat*    granRockerVertexBuffer  = NULL;
GLfloat*    granRockerColorBuffer   = NULL;
GLushort*   granRockerIndices       = NULL;
GLuint      granRockerVerts;

PyObject* drawGranRocker_drawButtons(PyObject *self, PyObject *args) {
   PyObject *py_faceColor;
   PyObject *py_detailColor;
   //PyObject *py_tuple;
   int   numHues;
   float posX, posY, rotation, w2h, scale, tDiff;
   float faceColor[3];
   float detailColor[3];
   float black[3] = {0.0, 0.0, 0.0};
   float white[3] = {1.0, 1.0, 1.0};
   char circleSegments = 45;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "ffOOlffff",
            &posX,
            &posY,
            &py_faceColor,
            &py_detailColor,
            &numHues,
            &rotation,
            &w2h,
            &scale,
            &tDiff)) {
      Py_RETURN_NONE;
   }

   // Parse Colors
   faceColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 0)));
   faceColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 1)));
   faceColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_faceColor, 2)));
   detailColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 0)));
   detailColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 1)));
   detailColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_detailColor, 2)));

   // Allocate and Define Geometry/Color buffers
   if (  granRockerVertexBuffer  == NULL  ||
         granRockerColorBuffer   == NULL  ||
         granRockerIndices       == NULL  ){
      printf("Initializing Geometry for Granularity Rocker\n");
      float unit = float(1.0/36.0);
      float R, G, B, buttonSize = 0.8f;
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      R = black[0];
      G = black[1];
      B = black[2];

      // Upper Background Mask (quad)
      /* X */ verts.push_back(-1.0);
      /* Y */ verts.push_back( 1.0);
      /* X */ verts.push_back(-1.0);
      /* Y */ verts.push_back( 0.0);
      /* X */ verts.push_back( 1.0);
      /* Y */ verts.push_back( 1.0);

      /* X */ verts.push_back( 1.0);
      /* Y */ verts.push_back( 1.0);
      /* X */ verts.push_back(-1.0);
      /* Y */ verts.push_back( 0.0);
      /* X */ verts.push_back( 1.0);
      /* Y */ verts.push_back( 0.0);

      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);

      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);

      // Lower Background Mask (Pill)
      drawPill(
            -24.0f*unit, 0.0f,   /* X, Y */
             24.0f*unit, 0.0f,   /* X, Y */
             12.0f*unit,         /* Radius */
             black,              /* Color */
             verts,
             colrs);

      // Left (Minus) Button
      drawCircle(
            -24.0f*unit, 0.0f, /* X, Y */
             12.0f*unit*buttonSize,       /* Radius */
             circleSegments,              /* Number of Circle Triangles */
             faceColor,                   /* Colors */
             verts,
             colrs);

      // Right (Plus) Button
      drawCircle(
             24.0f*unit, 0.0f, /* X, Y */
             12.0f*unit*buttonSize,       /* Radius */
             circleSegments,              /* Number of Circle Triangles */
             faceColor,                   /* Colors */
             verts,
             colrs);

      // Iconography
      drawCircle(-5.0f*unit*buttonSize,  6.0f*unit*buttonSize, 4.0f*unit*buttonSize, circleSegments, white, verts, colrs);
      drawCircle( 5.0f*unit*buttonSize,  0.0f*unit*buttonSize, 4.0f*unit*buttonSize, circleSegments, white, verts, colrs);
      drawCircle(-5.0f*unit*buttonSize, -6.0f*unit*buttonSize, 4.0f*unit*buttonSize, circleSegments, white, verts, colrs);

      // Minus Symbol
      float tmo = 18.0f;
      drawPill(
            -32.0f*unit + tmo*unit*buttonSize, 0.0f,  /* X, Y */
            -16.0f*unit - tmo*unit*buttonSize, 0.0f,  /* X, Y */
            2.0f*unit*buttonSize,                     /* Radius */
            detailColor,                              /* Color */
            verts,
            colrs);

      // Plus Symbol
      drawPill(
            32.0f*unit - tmo*unit*buttonSize, 0.0f,   /* X, Y */
            16.0f*unit + tmo*unit*buttonSize, 0.0f,   /* X, Y */
            2.0f*unit*buttonSize,                     /* Radius */
            detailColor,                              /* Color */
            verts,
            colrs);
      drawPill(
            24.0f*unit,  8.0f*unit*buttonSize,  /* X, Y */
            24.0f*unit, -8.0f*unit*buttonSize,  /* X, Y */
            2.0f*unit*buttonSize,               /* Radius */
            detailColor,                        /* Color */
            verts,
            colrs);

      granRockerVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (granRockerVertexBuffer == NULL) {
         granRockerVertexBuffer = new GLfloat[granRockerVerts*2];
      } else {
         delete [] granRockerVertexBuffer;
         granRockerVertexBuffer = new GLfloat[granRockerVerts*2];
      }

      if (granRockerColorBuffer == NULL) {
         granRockerColorBuffer = new GLfloat[granRockerVerts*3];
      } else {
         delete [] granRockerColorBuffer;
         granRockerColorBuffer = new GLfloat[granRockerVerts*3];
      }

      if (granRockerIndices == NULL) {
         granRockerIndices = new GLushort[granRockerVerts];
      } else {
         delete [] granRockerIndices;
         granRockerIndices = new GLushort[granRockerVerts];
      }

      for (unsigned int i = 0; i < granRockerVerts; i++) {
         granRockerVertexBuffer[i*2]   = verts[i*2];
         granRockerVertexBuffer[i*2+1] = verts[i*2+1];
         granRockerIndices[i]          = i;
         granRockerColorBuffer[i*3+0]  = colrs[i*3+0];
         granRockerColorBuffer[i*3+1]  = colrs[i*3+1];
         granRockerColorBuffer[i*3+2]  = colrs[i*3+2];
      }
   }

   glPushMatrix();
   if (w2h >= 1) {
      glTranslatef(posX*w2h, posY, 0);
      glScalef(scale, scale, 1);
   } else {
      glTranslatef(posX*w2h, posY*w2h, 0);
      glScalef(scale*w2h, scale*w2h, 1);
   }
   glRotatef(rotation, 0, 0, 1);
   glColorPointer(3, GL_FLOAT, 0, granRockerColorBuffer);
   glVertexPointer(2, GL_FLOAT, 0, granRockerVertexBuffer);
   glDrawElements( GL_TRIANGLES, granRockerVerts, GL_UNSIGNED_SHORT, granRockerIndices);
   glPopMatrix();
   
   Py_RETURN_NONE;
}
