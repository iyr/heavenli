//#include "primIconCircle.cpp"
//#include "primIconLinear.cpp"

using namespace std;
extern float offScreen;
//extern textAtlas* quack;
extern std::map<std::string, drawCall> drawCalls;
extern std::map<std::string, textAtlas> textFonts;
extern std::string selectedAtlas;
//extern std::map<std::string, textAtlas*> textFonts;

/*
 * Explanation of features:
 * <= 0: just the color representation
 * <= 1: color representation + outline
 * <= 2: color representation + outline + bulb markers
 * <= 3: color representation + outline + bulb markers + bulb marker halos
 * <= 4: color representation + outline + bulb markers + bulb marker halos + grand halo
 */

PyObject* drawIcon_hliGLutils(PyObject* self, PyObject* args){
   PyObject*   PyString;
   PyObject*   detailColorPyTup;
   PyObject*   faceColorPyTup;
   PyObject*   py_list;
   PyObject*   py_tuple;
   PyObject*   py_float;
   GLfloat*    bulbColors;
   GLfloat     detailColor[4];
   GLfloat     faceColor[4];
   GLfloat     gx, gy, scale, ao, w2h;
   GLuint      numBulbs, features, arn;
   //textAtlas* tmAt = &textFonts[selectedAtlas];

   if (!PyArg_ParseTuple(args,
            "fffIOIOOIffO",
            &gx, &gy,            // icon positon (X, Y)
            &scale,              // icon size
            &arn,                // arrangement
            &PyString,           // lamp alias text
            &features,           // iconography features
            &faceColorPyTup,     // feature colors
            &detailColorPyTup,   // feature colors
            &numBulbs,           // number of elements
            &ao,                 // icon rotation angle
            &w2h,                // width to hight ration
            &py_list             // colors of the elements (bulbs)
            ))
   {
      printf("failed to parse arguments :(\n");
      Py_RETURN_NONE;
   }

   //if (drawCalls.count("arnIconLinear") <= 0)
      //drawCalls.insert(std::make_pair("arnIconLinear", drawCall()));
   //drawCall* arnIconLinear = &drawCalls["arnIconLinear"];
   //if (drawCalls.count("arnIconCircle") <= 0)
      //drawCalls.insert(std::make_pair("arnIconCircle", drawCall()));
   //drawCall* arnIconCircle = &drawCalls["arnIconCircle"];
   if (drawCalls.count("arnIconText") <= 0)
      drawCalls.insert(std::make_pair("arnIconText", drawCall()));
   drawCall* arnIconText = &drawCalls["arnIconText"];
   if (drawCalls.count("arnIconTextbg") <= 0)
      drawCalls.insert(std::make_pair("arnIconTextbg", drawCall()));
   drawCall* arnIconTextbg = &drawCalls["arnIconTextbg"];

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new GLfloat[numBulbs*3];
   for (unsigned int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (unsigned char j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = (GLfloat)PyFloat_AsDouble(py_float);
      }
   }

   // Parse RGBA detail colors
   detailColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0));
   detailColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1));
   detailColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2));
   detailColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 3));

   // Parse RGBA face colors
   faceColor[0] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 0));
   faceColor[1] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 1));
   faceColor[2] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 2));
   faceColor[3] = (GLfloat)PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, 3));

   // Parse alias text
   const char* inputChars  = PyUnicode_AsUTF8(PyString);
   std::string inputString = inputChars;

   switch(arn){
      case 0:
         drawIconCircle(
               gx,
               gy,
               scale,
               features,
               detailColor,
               numBulbs,
               ao,
               w2h,
               bulbColors);
         break;

      case 1:
         drawIconLinear(
               gx,
               gy,
               scale,
               features,
               detailColor,
               numBulbs,
               ao,
               w2h,
               bulbColors);
         break;
   }

   drawText(
         inputString,
         0.5f,          // horizontal text alignment
         0.0f,          // vertical text alignment
         gx, gy,
         scale*2.0f, scale*2.0f,
         w2h,
         //tmAt,         // texture atlas to draw characters from
         &textFonts[selectedAtlas],
         detailColor,
         faceColor,
         arnIconText,
         arnIconTextbg);

   delete [] bulbColors;

   Py_RETURN_NONE;
}
