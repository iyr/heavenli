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

GLfloat     *bulbButtonCoordBuffer  = NULL; // Stores (X, Y) (float) for each vertex
GLfloat     *bulbButtonColorBuffer  = NULL; // Stores (R, G, B) (float) for each vertex
GLushort    *bulbButtonIndices      = NULL; // Stores index corresponding to each vertex
GLuint      bulbButtonVerts;
GLuint      vertsPerBulb;
float*      buttonCoords       = NULL;
GLint       colorsStart;
GLint       colorsEnd;
GLint       detailEnd;
GLint       prevNumBulbs;
GLint       prevArn;
GLfloat     prevAngOffset;
GLfloat     prevBulbButtonScale;
GLfloat     prevBulbButtonW2H;
Matrix      bulbButtonMVP;                  // Transformation matrix passed to shader
Params      bulbButtonPrevState;            // Stores transformations to avoid redundant recalculation
GLuint      bulbButtonVBO;                  // Vertex Buffer Object ID
GLboolean   bulbButtonFirstRun  = GL_TRUE;  // Determines if function is running for the first time (for VBO initialization)

PyObject* drawBulbButton_drawButtons(PyObject *self, PyObject *args)
{
   PyObject* faceColorPyTup;
   PyObject* detailColorPyTup;
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   GLdouble faceColor[3];
   GLdouble detailColor[3]; 
   GLdouble *bulbColors;
   //double bulbColor[3];
   GLfloat angularOffset, buttonScale, w2h, gx=0.0f, gy=0.0f, scale=1.0f, ao=0.0f;
   GLint arn, numBulbs;

   // Parse input arguments
   if (!PyArg_ParseTuple(args, 
            "iiffOOOf", 
            &arn,
            &numBulbs,
            &angularOffset,
            &buttonScale,
            &faceColorPyTup,
            &detailColorPyTup,
            &py_list,
            &w2h))
   {
      Py_RETURN_NONE;
   }

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new double[numBulbs*3];
   for (int i = 0; i < numBulbs; i++){
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++){
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = double(PyFloat_AsDouble(py_float));
      }
   }

   // Parse RGB color tuples of face and detail colors
   for (int i = 0; i < 3; i++){
      faceColor[i] = PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, i));
      detailColor[i] = PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, i));
   }

   // Initialize / Update Vertex Geometry and Colors
   if (  prevNumBulbs            != numBulbs ||
         bulbButtonCoordBuffer   == NULL     || 
         bulbButtonColorBuffer   == NULL     || 
         bulbButtonIndices       == NULL     || 
         buttonCoords            == NULL     ){

      if (numBulbs > 1) {
         printf("Initializing Geometry for Bulb Buttons\n");
      } else {
         printf("Initializing Geometry for Bulb Button\n");
      }

      vector<GLfloat> verts;
      vector<GLfloat> colrs;
      // Set Number of edges on circles
      char circleSegments = 60;
      char degSegment = 360 / circleSegments;

      // Setup Transformations
      if (w2h <= 1.0)
      {
         buttonScale = w2h*buttonScale;
      }

      if (buttonCoords == NULL) {
         buttonCoords = new float[2*numBulbs];
      } else {
         delete [] buttonCoords;
         buttonCoords = new float[2*numBulbs];
      }

      float tmx, tmy, ang;
      // Define verts / colors for each bulb button
      for (int j = 0; j < numBulbs; j++) {
         if (arn == 0) {
            ang = float(degToRad(j*360/numBulbs - 90 + angularOffset + 180/numBulbs));
         } else if (arn == 1) {
            ang = float(degToRad(
                  (j*180)/(numBulbs-1 < 1 ? 1 : numBulbs-1) + 
                  angularOffset + 
                  (numBulbs == 1 ? -90 : 0)
                  ));
         } else {
            ang = float(0.0);
         }

         // Relative coordinates of each button (from the center of the circle)
         tmx = float(0.75*cos(ang));
         tmy = float(0.75*sin(ang));
         if (w2h >= 1.0) {
            tmx *= float(pow(w2h, 0.5));
         } else {
            tmx *= float(w2h);
            tmy *= float(pow(w2h, 0.5));
         }

         buttonCoords[j*2+0] = tmx;
         buttonCoords[j*2+1] = tmy;

         // Define Vertices / Colors for Button Face
//#        pragma omp parallel for
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ verts.push_back(float(tmx+0.0));
            /* Y */ verts.push_back(float(tmy+0.0));
            /* R */ colrs.push_back(float(faceColor[0]));
            /* G */ colrs.push_back(float(faceColor[1]));
            /* B */ colrs.push_back(float(faceColor[2]));

            /* X */ verts.push_back(float(tmx+0.4*cos(degToRad(i*degSegment))*buttonScale));
            /* Y */ verts.push_back(float(tmy+0.4*sin(degToRad(i*degSegment))*buttonScale));
            /* R */ colrs.push_back(float(faceColor[0]));
            /* G */ colrs.push_back(float(faceColor[1]));
            /* B */ colrs.push_back(float(faceColor[2]));

            /* X */ verts.push_back(float(tmx+0.4*cos(degToRad((i+1)*degSegment))*buttonScale));
            /* Y */ verts.push_back(float(tmy+0.4*sin(degToRad((i+1)*degSegment))*buttonScale));
            /* R */ colrs.push_back(float(faceColor[0]));
            /* G */ colrs.push_back(float(faceColor[1]));
            /* B */ colrs.push_back(float(faceColor[2]));
         }

         if (j == 0) {
            colorsStart = colrs.size();
         }
         // Define Vertices for Bulb Icon
//#        pragma omp parallel for
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ verts.push_back(float(tmx+0.0*buttonScale));
            /* Y */ verts.push_back(float(tmy+0.1*buttonScale));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            /* X */ verts.push_back(float(tmx+0.2*cos(degToRad(i*degSegment))*buttonScale));
            /* Y */ verts.push_back(float(tmy+(0.1+0.2*sin(degToRad(i*degSegment)))*buttonScale));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));

            /* X */ verts.push_back(float(tmx+0.2*cos(degToRad((i+1)*degSegment))*buttonScale));
            /* Y */ verts.push_back(float(tmy+(0.1+0.2*sin(degToRad((i+1)*degSegment)))*buttonScale));
            /* R */ colrs.push_back(float(bulbColors[j*3+0]));
            /* G */ colrs.push_back(float(bulbColors[j*3+1]));
            /* B */ colrs.push_back(float(bulbColors[j*3+2]));
         }
         if (j == 0) {
            colorsEnd = colrs.size();
         }

         // Define Verts for bulb screw base
         GLfloat tmp[54] = {
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.085*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.085*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.119*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.085*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.119*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.119*buttonScale),
   
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.119*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.119*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.153*buttonScale),
   
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.136*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.170*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.204*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.136*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.170*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.204*buttonScale),
   
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.187*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.221*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.255*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.187*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.221*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.255*buttonScale),
   
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.238*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.272*buttonScale),
            /* X, Y */ float(tmx-0.051*buttonScale), float(tmy-0.306*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.238*buttonScale),
            /* X, Y */ float(tmx+0.051*buttonScale), float(tmy-0.306*buttonScale),
            /* X, Y */ float(tmx-0.051*buttonScale), float(tmy-0.306*buttonScale),
         };
   
         for (int i = 0; i < 27; i++) {
            /* X */ verts.push_back(float(tmp[i*2+0]));
            /* Y */ verts.push_back(float(tmp[i*2+1]));
            /* R */ colrs.push_back(float(detailColor[0]));
            /* G */ colrs.push_back(float(detailColor[1]));
            /* B */ colrs.push_back(float(detailColor[2]));
         }

         if (j == 0) {
            vertsPerBulb = verts.size()/2;
            detailEnd = colrs.size();
         }
      }
      // Pack Vertices / Colors into global array buffers
      bulbButtonVerts = verts.size()/2;

      // (Re)allocate vertex buffer
      if (bulbButtonCoordBuffer == NULL) {
         bulbButtonCoordBuffer = new GLfloat[bulbButtonVerts*2];
      } else {
         delete [] bulbButtonCoordBuffer;
         bulbButtonCoordBuffer = new GLfloat[bulbButtonVerts*2];
      }

      // (Re)allocate color buffer
      if (bulbButtonColorBuffer == NULL) {
         bulbButtonColorBuffer = new GLfloat[bulbButtonVerts*3];
      } else {
         delete [] bulbButtonColorBuffer;
         bulbButtonColorBuffer = new GLfloat[bulbButtonVerts*3];
      }

      // (Re)allocate index array
      if (bulbButtonIndices == NULL) {
         bulbButtonIndices = new GLushort[bulbButtonVerts];
      } else {
         delete [] bulbButtonIndices;
         bulbButtonIndices = new GLushort[bulbButtonVerts];
      }

      // Pack bulbButtonIndices, vertex and color bufferes
//#     pragma omp parallel for
      for (unsigned int i = 0; i < bulbButtonVerts; i++){
         bulbButtonCoordBuffer[i*2]   = verts[i*2];
         bulbButtonCoordBuffer[i*2+1] = verts[i*2+1];
         bulbButtonIndices[i]         = i;
         bulbButtonColorBuffer[i*3+0] = colrs[i*3+0];
         bulbButtonColorBuffer[i*3+1] = colrs[i*3+1];
         bulbButtonColorBuffer[i*3+2] = colrs[i*3+2];
      }

      // Calculate initial Transformation matrix
      Matrix Ortho;
      Matrix ModelView;

      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, 1.0f*gx, 1.0f*gy, 0.0f );
      MatrixScale( &ModelView, scale/w2h, scale, 1.0f );
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &bulbButtonMVP, &ModelView, &Ortho );

      bulbButtonPrevState.ao = ao;
      bulbButtonPrevState.dx = gx;
      bulbButtonPrevState.dy = gy;
      bulbButtonPrevState.sx = scale;
      bulbButtonPrevState.sy = scale;
      bulbButtonPrevState.w2h = w2h;

      // Update Statemachine Variables
      prevNumBulbs = numBulbs;
      prevAngOffset = angularOffset;
      prevBulbButtonW2H = w2h;
      prevArn = arn;
      prevBulbButtonScale = buttonScale;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (bulbButtonFirstRun == GL_TRUE) {
         bulbButtonFirstRun = GL_FALSE;
         glGenBuffers(1, &bulbButtonVBO);
      } else {
         glDeleteBuffers(1, &bulbButtonVBO);
         glGenBuffers(1, &bulbButtonVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, bulbButtonVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*bulbButtonVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLintptr offset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*bulbButtonVerts, bulbButtonCoordBuffer);
      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*bulbButtonVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*bulbButtonVerts, bulbButtonColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   } 

   // Recalculate vertex geometry without expensive vertex/array reallocation
   else if( prevBulbButtonW2H    != w2h            ||
            prevArn              != arn            ||
            prevAngOffset        != angularOffset  ||
            prevBulbButtonScale  != buttonScale    ){
      // Set Number of edges on circles
      char circleSegments = 60;
      char degSegment = 360 / circleSegments;

      // Setup Transformations
      if (w2h <= 1.0)
      {
         buttonScale = w2h*buttonScale;
      }

      float tmx, tmy, ang;
      // Define verts / colors for each bulb button
      for (int j = 0; j < numBulbs; j++) {
         if (arn == 0) {
            ang = float(degToRad(j*360/numBulbs - 90 + angularOffset + 180/numBulbs));
         } else if (arn == 1) {
            ang = float(degToRad(
                  (j*180)/(numBulbs-1 < 1 ? 1 : numBulbs-1) + 
                  angularOffset + 
                  (numBulbs == 1 ? -90 : 0)
                  ));
         } else {
            ang = float(0.0);
         }

         // Relative coordinates of each button (from the center of the circle)
         tmx = float(0.75*cos(ang));
         tmy = float(0.75*sin(ang));
         if (w2h >= 1.0) {
            tmx *= float(pow(w2h, 0.5));
         } else {
            tmx *= float(w2h);
            tmy *= float(pow(w2h, 0.5));
         }

         buttonCoords[j*2+0] = tmx;
         buttonCoords[j*2+1] = tmy;

         // Define Vertices / Colors for Button Face
//#        pragma omp parallel for
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ bulbButtonCoordBuffer[j*vertsPerBulb*2+i*6+0] = (float(tmx+0.0));
            /* Y */ bulbButtonCoordBuffer[j*vertsPerBulb*2+i*6+1] = (float(tmy+0.0));

            /* X */ bulbButtonCoordBuffer[j*vertsPerBulb*2+i*6+2] = (float(tmx+0.4*cos(degToRad(i*degSegment))*buttonScale));
            /* Y */ bulbButtonCoordBuffer[j*vertsPerBulb*2+i*6+3] = (float(tmy+0.4*sin(degToRad(i*degSegment))*buttonScale));

            /* X */ bulbButtonCoordBuffer[j*vertsPerBulb*2+i*6+4] = (float(tmx+0.4*cos(degToRad((i+1)*degSegment))*buttonScale));
            /* Y */ bulbButtonCoordBuffer[j*vertsPerBulb*2+i*6+5] = (float(tmy+0.4*sin(degToRad((i+1)*degSegment))*buttonScale));
         }

         // Define Vertices for Bulb Icon
//#        pragma omp parallel for
         for (int i = 0; i < circleSegments+1; i++){
            /* X */ bulbButtonCoordBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+0] = (float(tmx+0.0*buttonScale));
            /* Y */ bulbButtonCoordBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+1] = (float(tmy+0.1*buttonScale));

            /* X */ bulbButtonCoordBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+2] = (float(tmx+0.2*cos(degToRad(i*degSegment))*buttonScale));
            /* Y */ bulbButtonCoordBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+3] = (float(tmy+(0.1+0.2*sin(degToRad(i*degSegment)))*buttonScale));

            /* X */ bulbButtonCoordBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+4] = (float(tmx+0.2*cos(degToRad((i+1)*degSegment))*buttonScale));
            /* Y */ bulbButtonCoordBuffer[j*vertsPerBulb*2+(circleSegments+1)*6+i*6+5] = (float(tmy+(0.1+0.2*sin(degToRad((i+1)*degSegment)))*buttonScale));
         }

         // Define Verts for bulb screw base
         GLfloat tmp[54] = {
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.085*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.085*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.119*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.085*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.119*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.119*buttonScale),
   
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.119*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.119*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.153*buttonScale),
   
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.136*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.170*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.204*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.136*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.170*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.204*buttonScale),
   
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.187*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.221*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.255*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.187*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.221*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.255*buttonScale),
   
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.238*buttonScale),
            /* X, Y */ float(tmx-0.085*buttonScale), float(tmy-0.272*buttonScale),
            /* X, Y */ float(tmx-0.051*buttonScale), float(tmy-0.306*buttonScale),
            /* X, Y */ float(tmx+0.085*buttonScale), float(tmy-0.238*buttonScale),
            /* X, Y */ float(tmx+0.051*buttonScale), float(tmy-0.306*buttonScale),
            /* X, Y */ float(tmx-0.051*buttonScale), float(tmy-0.306*buttonScale),
         };
   
//#        pragma omp parallel for
         for (int i = 0; i < 27; i++) {
            /* X */ bulbButtonCoordBuffer[j*vertsPerBulb*2+i*2+(circleSegments+1)*12+0] = (float(tmp[i*2+0]));
            /* Y */ bulbButtonCoordBuffer[j*vertsPerBulb*2+i*2+(circleSegments+1)*12+1] = (float(tmp[i*2+1]));
         }
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, bulbButtonVBO);
      // Convenience variable
      GLintptr offset = 0;
      // Load Vertex Coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*bulbButtonVerts, bulbButtonCoordBuffer);

      offset = 2*sizeof(GLfloat)*bulbButtonVerts;
      // Load Vertex Color data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*bulbButtonVerts, bulbButtonColorBuffer);
   }
   // Vertices / Geometry already calculated
   // Check if colors need to be updated
   //else
   {
      /*
       * Iterate through each color channel 
       * 0 - RED
       * 1 - GREEN
       * 2 - BLUE
       */
      for (int i = 0; i < 3; i++) {

         // Update face color, if needed
         if (float(faceColor[i]) != bulbButtonColorBuffer[i]) {
            for (int j = 0; j < numBulbs; j++) {
//#              pragma omp parallel for
               for (int k = 0; k < colorsStart/3; k++) {
                  bulbButtonColorBuffer[ j*vertsPerBulb*3 + k*3 + i ] = float(faceColor[i]);
               }
            }
            // Update Contents of VBO
            // Set active VBO
            glBindBuffer(GL_ARRAY_BUFFER, bulbButtonVBO);
            // Convenience variable
            GLintptr offset = 2*sizeof(GLfloat)*bulbButtonVerts;
            // Load Vertex Color data into VBO
            glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*bulbButtonVerts, bulbButtonColorBuffer);
         }

         // Update Detail Color, if needed
         if (float(detailColor[i]) != bulbButtonColorBuffer[colorsEnd+i]) {
            for (int j = 0; j < numBulbs; j++) {
               for (int k = 0; k < (detailEnd - colorsEnd)/3; k++) {
                  bulbButtonColorBuffer[ colorsEnd + j*vertsPerBulb*3 + k*3 + i ] = float(detailColor[i]);
               }
            }
            // Update Contents of VBO
            // Set active VBO
            glBindBuffer(GL_ARRAY_BUFFER, bulbButtonVBO);
            // Convenience variable
            GLintptr offset = 2*sizeof(GLfloat)*bulbButtonVerts;
            // Load Vertex Color data into VBO
            glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*bulbButtonVerts, bulbButtonColorBuffer);
         }
      }
      
      // Update any bulb colors, if needed
      // Iterate through colors (R0, G1, B2)
      for (int i = 0; i < 3; i++) {

         // Iterate though bulbs
         for (int j = 0; j < numBulbs; j++) {

            // Iterate through color buffer to update colors
            if (float(bulbColors[i+j*3]) != bulbButtonColorBuffer[colorsStart + i + j*vertsPerBulb*3]) {
               for (int k = 0; k < (colorsEnd-colorsStart)/3; k++) {
                  bulbButtonColorBuffer[ j*vertsPerBulb*3 + colorsStart + i + k*3 ] = float(bulbColors[i+j*3]);
               }
               // Update Contents of VBO
               // Set active VBO
               glBindBuffer(GL_ARRAY_BUFFER, bulbButtonVBO);
               // Convenience variable
               GLintptr offset = 2*sizeof(GLfloat)*bulbButtonVerts;
               // Load Vertex Color data into VBO
               glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*bulbButtonVerts, bulbButtonColorBuffer);
            }
         }
      }
   } 

   //PyList_ClearFreeList();
   py_list = PyList_New(numBulbs);
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyTuple_New(2);
      PyTuple_SetItem(py_tuple, 0, PyFloat_FromDouble(buttonCoords[i*2+0]));
      PyTuple_SetItem(py_tuple, 1, PyFloat_FromDouble(buttonCoords[i*2+1]));
      PyList_SetItem(py_list, i, py_tuple);
   }

   // Cleanup
   delete [] bulbColors;
   
   // Update Transfomation Matrix if any change in parameters
   if (  bulbButtonPrevState.ao != ao     ||
         bulbButtonPrevState.dx != gx     ||
         bulbButtonPrevState.dy != gy     ||
         bulbButtonPrevState.sx != scale  ||
         bulbButtonPrevState.sy != scale  ||
         bulbButtonPrevState.w2h != w2h   ){
      
      Matrix Ortho;
      Matrix ModelView;

      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixLoadIdentity( &ModelView );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixTranslate( &ModelView, 1.0f*gx, 1.0f*gy, 0.0f );
      MatrixScale( &ModelView, scale/w2h, scale, 1.0f );
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &bulbButtonMVP, &ModelView, &Ortho );

      bulbButtonPrevState.ao = ao;
      bulbButtonPrevState.dx = gx;
      bulbButtonPrevState.dy = gy;
      bulbButtonPrevState.sx = scale;
      bulbButtonPrevState.sy = scale;
      bulbButtonPrevState.w2h = w2h;
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &bulbButtonMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, bulbButtonVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*bulbButtonVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, bulbButtonVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   return py_list;
}

