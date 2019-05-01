#define GL_GLEXT_PROTOTYPES
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
   #include <windows.h>
   // These undefs necessary because microsoft
   #undef near
   #undef far
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#include <math.h>
#include <vector>
#include <string>
using namespace std;

GLfloat     *stringCoordBuffer  = NULL;  // Stores (X, Y) (float) for each vertex
GLfloat     *stringColorBuffer  = NULL;  // Stores (R, G, B) (float) for each vertex
GLushort    *stringIndices      = NULL;  // Stores index corresponding to each vertex
GLuint      stringVerts;
std::string prevString;
Matrix      stringMVP;                   // Transformation matrix passed to shader
Params      stringPrevState;             // Stores transformations to avoid redundant recalculation
GLuint      stringVBO;                   // Vertex Buffer Object ID
GLboolean   stringFirstRun = GL_TRUE;    // Determines if function is running for the first time (for VBO initialization)

PyObject* printText_drawButtons(PyObject* self, PyObject *args) {
   PyObject *colourPyTup;
   PyObject *Pystring;

   GLfloat gx, gy, scale, w2h, ao=0.0f;
   GLfloat textColor[3];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "ffffOO",
            &gx, &gy,
            &scale,
            &w2h,
            &Pystring,
            &colourPyTup) )
   {
      Py_RETURN_NONE;
   }

   const char* inputChars = PyUnicode_AsUTF8(Pystring);
   std::string inputString = inputChars;

   textColor[0]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 0)));
   textColor[1]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 1)));
   textColor[2]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 2)));

   // (Re)allocate and Define Geometry/Color buffers
   if (  stringCoordBuffer == NULL        ||
         stringColorBuffer == NULL        ||
         stringIndices     == NULL        ||
         prevString        != inputString ){

      //printf("Input String: %s\n", inputString.c_str());

      //printf("Initializing Geometry for Confirm Button\n");
      vector<float> verts;
      vector<float> colrs;
      char circleSegments  = 20;
      float lineWidth      = 0.0f;
      float charSpacing    = 0.5f;
      //float charThickness  = 0.2f;
      //float charThickness  = 0.15f;
      //float charThickness  = 0.125f;
      float charThickness  = 0.04f;
      float charScale      = 1.0f;
      stringVerts          = 0;
      //printf("stringSize: %i\n", inputString.size());

      for (unsigned int i = 0; i < inputString.size(); i++) {
         drawChar(
               inputChars[i],
               lineWidth, 0.0f,  // X, Y coordinates 
               charScale,        // Scale 
               charThickness,    // Thickness/boldness 
               charSpacing,      // Amount of empty space after character 
               circleSegments, 
               textColor, 
               &lineWidth, 
               verts, 
               colrs);
      }

      stringVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (stringCoordBuffer == NULL) {
         stringCoordBuffer = new GLfloat[stringVerts*2];
      } else {
         delete [] stringCoordBuffer;
         stringCoordBuffer = new GLfloat[stringVerts*2];
      }

      if (stringColorBuffer == NULL) {
         stringColorBuffer = new GLfloat[stringVerts*3];
      } else {
         delete [] stringColorBuffer;
         stringColorBuffer = new GLfloat[stringVerts*3];
      }

      if (stringIndices == NULL) {
         stringIndices = new GLushort[stringVerts];
      } else {
         delete [] stringIndices;
         stringIndices = new GLushort[stringVerts];
      }

      for (unsigned int i = 0; i < stringVerts; i++) {
         stringCoordBuffer[i*2+0]  = verts[i*2+0];
         stringCoordBuffer[i*2+1]  = verts[i*2+1];
         stringIndices[i]          = i;
         stringColorBuffer[i*3+0]  = colrs[i*3+0];
         stringColorBuffer[i*3+1]  = colrs[i*3+1];
         stringColorBuffer[i*3+2]  = colrs[i*3+2];
      }

      prevString = inputString;
      
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
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &stringMVP, &ModelView, &Ortho );

      stringPrevState.ao = ao;
      stringPrevState.dx = gx;
      stringPrevState.dy = gy;
      stringPrevState.sx = scale;
      stringPrevState.sy = scale;
      stringPrevState.w2h = w2h;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (stringFirstRun == GL_TRUE) {
         stringFirstRun = GL_FALSE;
         glGenBuffers(1, &stringVBO);
      } else {
         glDeleteBuffers(1, &stringVBO);
         glGenBuffers(1, &stringVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, stringVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*stringVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLintptr offset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*stringVerts, stringCoordBuffer);
      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*stringVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*stringVerts, stringColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   }

   // Geometry already calculated, update colors
   /*
    * Iterate through each color channel 
    * 0 - RED
    * 1 - GREEN
    * 2 - BLUE
    */
   for (GLuint i = 0; i < 3; i++) {
      if (float(textColor[i]) != stringColorBuffer[i] ){
         for (GLuint k = 0; k < stringVerts; k++) {
            if (float(textColor[i]) != stringColorBuffer[i + k*3]){
               stringColorBuffer[k*3 + i] = float(textColor[i]);
            }
         }
         // Update Contents of VBO
         // Set active VBO
         glBindBuffer(GL_ARRAY_BUFFER, stringVBO);
         // Convenience variable
         GLintptr offset = 2*sizeof(GLfloat)*stringVerts;
         // Load Vertex Color data into VBO
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*stringVerts, stringColorBuffer);
      }
   }

   // Update Transfomation Matrix if any change in parameters
   if (  stringPrevState.ao != ao     ||
         stringPrevState.dx != gx     ||
         stringPrevState.dy != gy     ||
         stringPrevState.sx != scale  ||
         stringPrevState.sy != scale  ||
         stringPrevState.w2h != w2h   ){
      
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
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &stringMVP, &ModelView, &Ortho );

      stringPrevState.ao = ao;
      stringPrevState.dx = gx;
      stringPrevState.dy = gy;
      stringPrevState.sx = scale;
      stringPrevState.sy = scale;
      stringPrevState.w2h = w2h;

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, stringVBO);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*stringVerts));
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &stringMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, stringVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*stringVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, stringVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);


   Py_RETURN_NONE;
}
