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

GLfloat     *colrTriCoordBuffer  = NULL;  // Stores (X, Y) (float) for each vertex
GLfloat     *colrTriColorBuffer  = NULL;  // Stores (R, G, B) (float) for each vertex
GLushort    *colrTriIndices      = NULL;  // Stores index corresponding to each vertex
GLuint      colrTriVerts;                 // Total number of vertices
GLint       prevColrTriNumLevels;         // Used for updating Granularity changes
GLfloat     *triButtonData       = NULL;  // Stores data (X, Y, sat, val) for each button dot
GLfloat     prevTriX             = 0.0;   // Used for animating granularity changes
GLfloat     prevTriY             = 0.0;   // Used for animating granularity changes
GLfloat     prevTriDotScale      = 1.0;   // Used for animating granularity changes
GLfloat     prevRingX            = 0.0;   // Used for animating selection ring
GLfloat     prevRingY            = 0.0;   // Used for animating selection ring
GLfloat     prevTriHue;                   // Used for animating selection ring
GLfloat     prevTriSat;                   // Used for animating selection ring
GLfloat     prevTriVal;                   // Used for animating selection ring
GLfloat     prevTriSatSel        = 0.0;   // Used to resolve edge-case bug for animating selection ring
GLfloat     prevTriValSel        = 0.0;   // Used to resolve edge-case bug for animating selection ring
Matrix      colrTriMVP;                   // Transformation matrix passed to shader
Params      colrTriPrevState;             // Stores transformations to avoid redundant recalculation
GLuint      colrTriVBO;                   // Vertex Buffer Object ID
GLboolean   colrTriFirstRun   = GL_TRUE;  // Determines if function is running for the first time (for VBO initialization)
GLuint      colrTriDotsVerts;

PyObject* drawColrTri_drawUtils(PyObject *self, PyObject *args) {
   PyObject *py_list;
   PyObject *py_tuple;
   float w2h, scale, currentTriHue, currentTriSat, currentTriVal, tDiff, gx=0.0f, gy=0.0f, ao=0.0f;
   float ringColor[4];
   char circleSegments = 24;
   long numLevels = 6;

   // Parse Inputs
   if (!PyArg_ParseTuple(args,
            "ffflOfff",
            &currentTriHue,
            &currentTriSat,
            &currentTriVal,
            &numLevels,
            &py_tuple,
            &w2h,
            &scale,
            &tDiff))
   {
      Py_RETURN_NONE;
   }

   ringColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 0)));
   ringColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 1)));
   ringColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 2)));
   ringColor[3] = float(PyFloat_AsDouble(PyTuple_GetItem(py_tuple, 3)));

   /*
    * Granularity Levels:
    * 5: Low
    * 6: Medium
    * 7: High
    */

   if (numLevels < 5)
      numLevels = 5;
   if (numLevels > 7)
      numLevels = 7;

   long numButtons = (numLevels * (numLevels + 1)) / 2;

   // (Re)Allocate and Define Geometry/Color buffers
   if (  prevColrTriNumLevels != numLevels   ||
         colrTriCoordBuffer   == NULL        ||
         colrTriColorBuffer   == NULL        ||
         colrTriIndices       == NULL        ){

      //printf("Initializing Geometry for Color Triangle\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      // Allocate buffer for storing relative positions of each button
      // (Used for processing user input)
      if (triButtonData == NULL) {
         triButtonData = new float[4*numButtons];
      } else {
         delete [] triButtonData;
         triButtonData = new float[4*numButtons];
      }

      if (  prevTriX == 0.0   )
         prevTriX = float(-0.0383*numLevels);

      if (  prevTriY == 0.0   )
         prevTriY = float(+0.0616*numLevels);

      // Actual meat of drawing saturation/value triangle
      int index = 0;
      float tmx, tmy, tmr, saturation, value, ringX = 0.0f, ringY = 0.0f;
      float colors[4] = {0.0, 0.0, 0.0, 1.0};
      tmr = 0.05f*prevTriDotScale;
      for (int i = 0; i < numLevels; i++) {        /* Columns */
         for (int j = 0; j < numLevels-i; j++) {   /* Rows */

            // Calculate Discrete Saturation and Value
            value = 1.0f - float(j) / float(numLevels - 1);
            saturation  =  float(i) / float(numLevels - 1 - j);

            // Resolve issues that occur when saturation or value are less than zero or NULL
            if (saturation != saturation || saturation <= 0.0)
               saturation = 0.000001f;
            if (value != value || value <= 0.0)
               value = 0.000001f;

            // Convert HSV to RGB
            hsv2rgb(currentTriHue, saturation, value, colors);

            // Define relative positions of sat/val button dots
            tmx = float(prevTriX + (i*0.13f));
            tmy = float(prevTriY - (i*0.075f + j*0.145f));

            // Draw dot
            defineEllipse(tmx, tmy, tmr, tmr, circleSegments, colors, verts, colrs);

            // Store position and related data of button dot
            triButtonData[index*4 + 0] = float(-0.0383*numLevels + (i*0.13f));
            triButtonData[index*4 + 1] = float(+0.0616*numLevels - (i*0.075f + j*0.145f));
            triButtonData[index*4 + 2] = saturation;
            triButtonData[index*4 + 3] = value;
            index++;

            // Determine which button dot represents the currently selected saturation and value
            if (  abs(currentTriSat - saturation) <= 1.0f / float(numLevels*2) &&
                  abs(currentTriVal - value     ) <= 1.0f / float(numLevels*2) ){
               ringX = tmx;
               ringY = tmy;
            }
         }
      }

      colrTriDotsVerts = verts.size()/2;

      // Draw a circle around the button dot corresponding to the selected saturation/value
      defineArch(
            ringX, ringY,
            float(1.06*tmr), 
            float(1.06*tmr),
            0.0f,
            360.0f,
            0.03f,
            circleSegments,
            ringColor,
            verts,
            colrs);

      // Total number of Vertices, useful for array updating
      colrTriVerts = verts.size()/2;

      // (Re)Allocate buffer for vertex data
      if (colrTriCoordBuffer == NULL) {
         colrTriCoordBuffer = new GLfloat[colrTriVerts*2];
      } else {
         delete [] colrTriCoordBuffer;
         colrTriCoordBuffer = new GLfloat[colrTriVerts*2];
      }

      // (Re)Allocate buffer for color data
      if (colrTriColorBuffer == NULL) {
         colrTriColorBuffer = new GLfloat[colrTriVerts*4];
      } else {
         delete [] colrTriColorBuffer;
         colrTriColorBuffer = new GLfloat[colrTriVerts*4];
      }

      // (Re)Allocate buffer for indices
      if (colrTriIndices == NULL) {
         colrTriIndices = new GLushort[colrTriVerts];
      } else {
         delete [] colrTriIndices;
         colrTriIndices = new GLushort[colrTriVerts];
      }

      // Pack Vertices and Colors into global array buffers
      for (unsigned int i = 0; i < colrTriVerts; i++) {
         colrTriCoordBuffer[i*2+0]  = verts[i*2];
         colrTriCoordBuffer[i*2+1]  = verts[i*2+1];
         colrTriIndices[i]          = i;
         colrTriColorBuffer[i*4+0]  = colrs[i*4+0];
         colrTriColorBuffer[i*4+1]  = colrs[i*4+1];
         colrTriColorBuffer[i*4+2]  = colrs[i*4+2];
         colrTriColorBuffer[i*4+3]  = colrs[i*4+3];
      }

      // Calculate initial transformation matrix
      Matrix Ortho;
      Matrix ModelView;

      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, 1.0f*gx, 1.0f*gy, 0.0f );
      MatrixScale( &ModelView, scale/w2h, scale, 1.0f );
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &colrTriMVP, &ModelView, &Ortho );

      colrTriPrevState.ao = ao;
      colrTriPrevState.dx = gx;
      colrTriPrevState.dy = gy;
      colrTriPrevState.sx = scale;
      colrTriPrevState.sy = scale;
      colrTriPrevState.w2h = w2h;

      // Update State Machine variables
      prevTriHue = currentTriHue;
      prevTriSat = currentTriSat;
      prevTriVal = currentTriVal;
      prevTriSatSel = currentTriSat;
      prevTriValSel = currentTriVal;
      prevColrTriNumLevels = numLevels;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (colrTriFirstRun == GL_TRUE) {
         colrTriFirstRun = GL_FALSE;
         glGenBuffers(1, &colrTriVBO);
      } else {
         glDeleteBuffers(1, &colrTriVBO);
         glGenBuffers(1, &colrTriVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, colrTriVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 6*sizeof(GLfloat)*colrTriVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLintptr offset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*colrTriVerts, colrTriCoordBuffer);
      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*colrTriVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*4*colrTriVerts, colrTriColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   }

   // Animate granularity changes
   // Update position of hue/sat dot triangle
   GLfloat triX = float(-0.0383f*numLevels); 
   GLfloat triY = float(+0.0616f*numLevels);
   GLfloat deltaX, deltaY;
   deltaX = triX - prevTriX;
   deltaY = triY - prevTriY;

   if (  abs(deltaX) > tDiff*0.01   ||
         abs(deltaY) > tDiff*0.01   ){
      if (deltaX < -0.0) {
         prevTriX -= float(3.0*tDiff*abs(deltaX));
      }
      if (deltaX > -0.0) {
         prevTriX += float(3.0*tDiff*abs(deltaX));
      }
      if (deltaY < -0.0) {
         prevTriY -= float(3.0*tDiff*abs(deltaY));
      }
      if (deltaY > -0.0) {
         prevTriY += float(3.0*tDiff*abs(deltaY));
      }

      // Actual meat of drawing saturation/value triangle
      GLint index = 0;
      GLfloat tmx, tmy, tmr, saturation, value, ringX=0.0f, ringY=0.0f;
      GLfloat colors[4] = {0.0, 0.0, 0.0, 0.0};

      // Bogus comparison to get rid of compiler warnings (-_-)
      if (ringX != ringX)
         ringX = 0.0f;
      if (ringY != ringY)
         ringY = 0.0f;

      tmr = 0.05f*prevTriDotScale;
      for (int i = 0; i < numLevels; i++) {        /* Columns */
         for (int j = 0; j < numLevels-i; j++) {   /* Rows */

            // Calculate Discrete Saturation and Value
            value = 1.0f - float(j) / float(numLevels - 1);
            saturation  =  float(i) / float(numLevels - 1 - j);

            // Resolve issues that occur when saturation or value are less than zero or NULL
            if (saturation != saturation || saturation <= 0.0)
               saturation = 0.000001f;
            if (value != value || value <= 0.0)
               value = 0.000001f;

            // Convert HSV to RGB
            hsv2rgb(currentTriHue, saturation, value, colors);

            // Define relative positions of sat/val button dots
            tmx = float(prevTriX + (i*0.13f));
            tmy = float(prevTriY - (i*0.075f + j*0.145f));

            // Draw dot
            index = updatePrimEllipseGeometry(
                  tmx, tmy, 
                  tmr, tmr,
                  circleSegments, 
                  index, 
                  colrTriCoordBuffer);

            // Determine which button dot represents the currently selected saturation and value
            if (  abs(currentTriSat - saturation) <= 1.0f / float(numLevels) &&
                  abs(currentTriVal - value     ) <= 1.0f / float(numLevels) ){
               ringX = tmx;
               ringY = tmy;
            }
         }
      }

      // Draw a circle around the button dot corresponding to the selected saturation/value
      index = updateArchGeometry(
            ringX, ringY,
            float(1.06*tmr), 
            float(1.06*tmr),
            0.0f,
            360.0f,
            0.03f,
            circleSegments,
            index,
            colrTriCoordBuffer);

      // Update Contents of VBO
      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, colrTriVBO);
      // Convenience variables
      GLintptr offset = 0;
      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*colrTriVerts, colrTriCoordBuffer);

   } else {
      prevTriX = triX;
      prevTriY = triY;
   }

   // Resolves edge case bug animating selection ring
   if (  prevTriSatSel  != currentTriSat  ){
      prevTriSat = prevTriSatSel;
   } 
   if (  prevTriValSel  != currentTriVal  ){
      prevTriVal = prevTriValSel;
   } 

   // Determine distance of selection ring from 
   // current location (prevTri) to target location (ring)
   GLfloat tmr, saturation, value, ringX = 0.0f, ringY = 0.0f;
   tmr = 0.05f;
   for (int i = 0; i < numLevels; i++) {        // Columns
      for (int j = 0; j < numLevels-i; j++) {   // Rows

         // Calculate Discrete Saturation and Value
         value = 1.0f - float(j) / float(numLevels - 1);
         saturation  =  float(i) / float(numLevels - 1 - j);

         // Resolve issues that occur when saturation or value are less than zero or NULL
         if (saturation != saturation || saturation <= 0.0)
            saturation = 0.000001f;
         if (value != value || value <= 0.0)
            value = 0.000001f;

         // Define relative positions of sat/val button dots
         if (  abs(currentTriSat - saturation) <= 1.0f / float(numLevels*2) &&
               abs(currentTriVal - value     ) <= 1.0f / float(numLevels*2) ){
            ringX = float(-0.0383*numLevels + (i*0.13f));
            ringY = float(+0.0616*numLevels - (i*0.075f + j*0.145f));
         }
      }
   }

   deltaX = ringX-prevRingX;
   deltaY = ringY-prevRingY;

   // Update x-position of selection ring if needed
   if (abs(deltaX) > tDiff*0.01) {
      if (deltaX < -0.0) {
         prevRingX -= float(3.0*tDiff*abs(deltaX));
      }
      if (deltaX > -0.0) {
         prevRingX += float(3.0*tDiff*abs(deltaX));
      }
   } else {
      prevRingX = ringX;
   }

   // Update y-position of selection ring if needed
   if (abs(deltaY) > tDiff*0.01) {
      if (deltaY < -0.0) {
         prevRingY -= float(3.0*tDiff*abs(deltaY));
      }
      if (deltaY > -0.0) {
         prevRingY += float(3.0*tDiff*abs(deltaY));
      }
   } else {
      prevRingY = ringY;
   }

   // Update position of the selection ring if needed
   if ( (prevTriSat  != currentTriSat  ||
         prevTriVal  != currentTriVal) ){

      updateArchGeometry(
            prevRingX, prevRingY,
            float(1.06*tmr), 
            float(1.06*tmr),
            0.0f,
            360.0f,
            0.03f,
            circleSegments,
            colrTriDotsVerts,
            colrTriCoordBuffer);

      // Update Contents of VBO
      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, colrTriVBO);
      // Convenience variable
      GLintptr offset = 0;
      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*colrTriVerts, colrTriCoordBuffer);
      offset += 2*sizeof(GLfloat)*colrTriVerts;
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*4*colrTriVerts, colrTriColorBuffer);
   }

   // Selection Ring in place, stop updating position
   if (  abs(deltaX) <= tDiff*0.01 &&
         abs(deltaY) <= tDiff*0.01 ){
      prevTriSat = currentTriSat;
      prevTriVal = currentTriVal;
   }

   prevTriSatSel = currentTriSat;
   prevTriValSel = currentTriVal;

   // Update colors if current Hue has changed
   if ( (prevTriHue != currentTriHue)        &&
        (prevColrTriNumLevels == numLevels)  ){
      GLfloat saturation, value;
      GLfloat colors[4] = {0.0, 0.0, 0.0, 1.0};
      GLint colrIndex = 0;
      for (GLint i = 0; i < prevColrTriNumLevels; i++) {        /* Columns */
         for (GLint j = 0; j < prevColrTriNumLevels-i; j++) {   /* Rows */

            // Calculate Discrete Saturation and Value
            value = 1.0f - float(j) / float(prevColrTriNumLevels - 1);
            saturation  =  float(i) / float(prevColrTriNumLevels - 1 - j);

            // Resolve issues that occur when saturation or value are less than zero or NULL
            if (saturation != saturation || saturation <= 0.0 )
               saturation = 0.000001f;
            if (value != value || value <= 0.0 )
               value = 0.000001f;

            // Convert HSV to RGB 
	         hsv2rgb(currentTriHue, saturation, value, colors);
            colrIndex = updatePrimEllipseColor(
                  circleSegments, 
                  colors,
                  colrIndex, 
                  colrTriColorBuffer);
         }
      }
      prevTriHue = currentTriHue;

      // Update Contents of VBO
      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, colrTriVBO);
      // Convenience variable
      GLintptr offset = 2*sizeof(GLfloat)*colrTriVerts;
      // Load Vertex Color data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*4*colrTriVerts, colrTriColorBuffer);
   }

   // Check if selection Ring Color needs to be updated
   for (int i = 0; i < 4; i++) {
      if ( (colrTriColorBuffer[colrTriDotsVerts*4+i] != ringColor[i])   && 
           (prevColrTriNumLevels == numLevels)                                   ){
         for (unsigned int k = colrTriDotsVerts; k < colrTriVerts; k++) {
            colrTriColorBuffer[k*4+i] = ringColor[i];
         }
         // Update Contents of VBO
         // Set active VBO
         glBindBuffer(GL_ARRAY_BUFFER, colrTriVBO);
         // Convenience variable
         GLintptr offset = 2*sizeof(GLfloat)*colrTriVerts;
         // Load Vertex Color data into VBO
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*4*colrTriVerts, colrTriColorBuffer);
      }
   }

   // Create a Python List of tuples containing data of each button dot
   py_list = PyList_New(numButtons);
   for (int i = 0; i < numButtons; i++) {
      py_tuple = PyTuple_New(4);
      PyTuple_SetItem(py_tuple, 0, PyFloat_FromDouble(triButtonData[i*4+0]));
      PyTuple_SetItem(py_tuple, 1, PyFloat_FromDouble(triButtonData[i*4+1]));
      PyTuple_SetItem(py_tuple, 2, PyFloat_FromDouble(triButtonData[i*4+2]));
      PyTuple_SetItem(py_tuple, 3, PyFloat_FromDouble(triButtonData[i*4+3]));
      PyList_SetItem(py_list, i, py_tuple);
   }

   glPushMatrix();
   if (w2h <= 1.0) {
         scale = scale*w2h;
   }

   // Update Transfomation Matrix if any change in parameters
   if (  colrTriPrevState.ao != ao     ||
         colrTriPrevState.dx != gx     ||
         colrTriPrevState.dy != gy     ||
         colrTriPrevState.sx != scale  ||
         colrTriPrevState.sy != scale  ||
         colrTriPrevState.w2h != w2h   ){
      
      Matrix Ortho;
      Matrix ModelView;

      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, 1.0f*gx, 1.0f*gy, 0.0f );
      MatrixScale( &ModelView, scale/w2h, scale, 1.0f );
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &colrTriMVP, &ModelView, &Ortho );

      colrTriPrevState.ao = ao;
      colrTriPrevState.dx = gx;
      colrTriPrevState.dy = gy;
      colrTriPrevState.sx = scale;
      colrTriPrevState.sy = scale;
      colrTriPrevState.w2h = w2h;

      glBindBuffer(GL_ARRAY_BUFFER, colrTriVBO);
      glBufferSubData(GL_ARRAY_BUFFER, 2*sizeof(GLfloat)*colrTriVerts, sizeof(GLfloat)*3*colrTriVerts, colrTriColorBuffer);
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &colrTriMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, colrTriVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*colrTriVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, colrTriVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   return py_list;
}
