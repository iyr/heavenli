#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
/*
 * Draw a WaveFront Object file from filepath
 */
using namespace std;
extern std::map<std::string, drawCall> drawCalls;

PyObject* drawWFobject_hliGLutils(PyObject* self, PyObject* args) {
   PyObject* objectName;   // Relative Path + filename for object
   PyObject* colorPyTup;   // RGBA tuple
   PyArrayObject* MVP;     // model-view-projection matrix as a contiguous C-array

   float    w2h;
   float    color[4];
   double*   cMVP;
   //extern GLint uniform_tex;

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "OOfO",
            &objectName,
            &MVP,
            &w2h,
            &colorPyTup
            ) )
   {
      Py_RETURN_NONE;
   }

   color[0] = float(PyFloat_AsDouble(PyTuple_GetItem(colorPyTup, 0)));
   color[1] = float(PyFloat_AsDouble(PyTuple_GetItem(colorPyTup, 1)));
   color[2] = float(PyFloat_AsDouble(PyTuple_GetItem(colorPyTup, 2)));
   color[3] = float(PyFloat_AsDouble(PyTuple_GetItem(colorPyTup, 3)));

   // Parse object name
   const char* objNameChars = PyUnicode_AsUTF8(objectName);
   std::string objNameString = objNameChars;

   cMVP = (double *)PyArray_DATA(MVP);

   if (drawCalls.count(objNameString) <= 0){
      drawCalls.insert(std::make_pair(objNameString, drawCall()));
   }
   drawCall* object = &drawCalls[objNameString];
   object->setShader("3DRGBA_color_texture");
   //object->setDrawType(GL_LINE_STRIP);
   object->setDrawType(GL_TRIANGLES);

   drawWFobject(
         objNameString,
         cMVP,
         w2h,
         color,
         object
         );

   Py_RETURN_NONE;
}

void drawWFobject(
      string      filepath,
      GLdouble*   MVP,
      GLfloat     w2h,
      GLfloat*    color,
      drawCall*   object
      ){
   GLuint objectVerts;
   object->setNumColors(1);
   object->setColorQuartet(0, color);

   if (  object->numVerts  == 0
      ){

      std::vector <GLfloat> verts;
      std::vector <GLfloat> colrs;
      std::vector <GLfloat> texuv;
      std::vector <GLfloat> nrmls;

      objectVerts = defineObjTrig(
            filepath,
            color,
            verts,
            texuv,
            colrs,
            nrmls
            );

      map<string, attribCache> attributeData;
      // Define vertex attributes, initialize caches
      attributeData[VAS.coordData] = attribCache(VAS.coordData, 4, 0, 0);
      attributeData[VAS.colorData] = attribCache(VAS.colorData, 4, 4, 1);
      attributeData[VAS.normlData] = attribCache(VAS.normlData, 4, 8, 2);
      attributeData[VAS.texuvData] = attribCache(VAS.texuvData, 2, 12, 3);
      attributeData[VAS.coordData].writeCache(verts.data(), verts.size());
      attributeData[VAS.colorData].writeCache(colrs.data(), colrs.size());
      attributeData[VAS.normlData].writeCache(nrmls.data(), nrmls.size());
      attributeData[VAS.texuvData].writeCache(texuv.data(), texuv.size());
      objectVerts = verts.size()/4;
      object->buildCache(objectVerts, attributeData);

      // Elaborate print statement for debugging transform matrices
      /*
      float x, y, z, w, tx, ty, tz, tw;
      float* tmp = (float *)object->getAttribCache(VAS.coordData);
      for (unsigned int i = 0; i < objectVerts; i++ ) {
         x  = tmp[i*4+0];
         y  = tmp[i*4+1];
         z  = tmp[i*4+2];
         w  = tmp[i*4+3];
         tx = (GLfloat)MVP[0*4+0]*x + (GLfloat)MVP[1*4+0]*y + (GLfloat)MVP[2*4+0]*z + (GLfloat)MVP[3*4+0]*w;
         ty = (GLfloat)MVP[0*4+1]*x + (GLfloat)MVP[1*4+1]*y + (GLfloat)MVP[2*4+1]*z + (GLfloat)MVP[3*4+1]*w;
         tz = (GLfloat)MVP[0*4+2]*x + (GLfloat)MVP[1*4+2]*y + (GLfloat)MVP[2*4+2]*z + (GLfloat)MVP[3*4+2]*w;
         tw = (GLfloat)MVP[0*4+3]*x + (GLfloat)MVP[1*4+3]*y + (GLfloat)MVP[2*4+3]*z + (GLfloat)MVP[3*4+3]*w;
         printf("%d: raw coordinate, transformed coordinate:\n", i);
         printf("X: %3.5f, %3.5f\n", x, tx);
         printf("Y: %3.5f, %3.5f\n", y, ty);
         printf("Z: %3.5f, %3.5f\n", z, tz);
         printf("W: %3.5f, %3.5f\n", w, tw);
      }
      printf("matrix: \n");
      for(unsigned int i = 0; i < 4; i++){
         printf("%2.4f %2.4f %2.4f %2.4f\n",
               (GLfloat)MVP[i*4+0],
               (GLfloat)MVP[i*4+1],
               (GLfloat)MVP[i*4+2],
               (GLfloat)MVP[i*4+3]
               );
      }
      */
   }

   object->setMVP(MVP);
   object->draw();
   return;
}
