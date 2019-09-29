/*
 * Heavenli opengl drawcode for circular arrangments (backgroun+iconography)
 */

using namespace std;
extern float offScreen;

drawCall homeCircle;
GLint    prevHomeCircleNumBulbs;

PyObject* drawHomeCircle_hliGLutils(PyObject *self, PyObject *args) {
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   GLfloat *bulbColors;
   GLfloat gx, gy, wx, wy, ao, w2h, alpha=1.0f;
   //GLfloat R, G, B;
   GLint numBulbs;

   if (!PyArg_ParseTuple(args,
            "fffflffO",
            &gx, &gy,      // background position (X, Y)
            &wx, &wy,      // background scale (X, Y)
            &numBulbs,     // number of elements
            &ao,           // background rotation angle
            &w2h,          // width to height ratio
            &py_list       // colors of the background segments
            ))
   {
      Py_RETURN_NONE;
   }

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new float[numBulbs*3];
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = float(PyFloat_AsDouble(py_float));
      }
   }

   for (int i = 0; i < numBulbs; i++ ) {
      GLfloat tmc[4];
      tmc[0] = bulbColors[i*3+0];
      tmc[1] = bulbColors[i*3+1];
      tmc[2] = bulbColors[i*3+2];
      tmc[3] = alpha;
      homeCircle.setColorQuartet(i, tmc);
   }

   unsigned int circleSegments = 60;

   if (  homeCircle.numVerts     == 0        ||
         prevHomeCircleNumBulbs  != numBulbs ){

      homeCircle.setNumColors(numBulbs);

      printf("Initializing Geometry for Circular Background\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      defineColorWheel(0.0f, 0.0f, 10.0f, circleSegments, 180.0f, numBulbs, 1.0f, bulbColors, verts, colrs);

      prevHomeCircleNumBulbs = numBulbs;
      homeCircle.buildCache(verts.size()/2, verts, colrs);
   }

   if (homeCircle.colorsChanged) {
      unsigned int index = 0;
      updateColorWheelColor(circleSegments, numBulbs, 1.0f, bulbColors, index, homeCircle.colorCache);
      homeCircle.updateColorCache();
   }

   homeCircle.updateMVP(gx, gy, 1.0f, 1.0f, -ao, 1.0f);
   homeCircle.draw();

   Py_RETURN_NONE;
}

/*
GLfloat     *homeCircleCoordBuffer  = NULL; // Stores (X, Y) (float) for each vertex
GLfloat     *homeCircleColorBuffer  = NULL; // Stores (R, G, B) (float) for each vertex
GLushort    *homeCircleIndices      = NULL; // Stores index corresponding to each vertex
GLuint      homeCircleVerts;
GLuint      homeCircleBuffer;
GLuint      homeCircleIBO;
GLint       attribVertexPosition;
GLint       attribVertexColor;
Matrix      homeCircleMVP;                  // Transformation matrix passed to shader
Params      homeCirclePrevState;            // Stores transformations to avoid redundant recalculation
GLuint      homeCircleVBO;                  // Vertex Buffer Object ID
GLboolean   homeCircleFirstRun = GL_TRUE;   // Determines if function is running for the first time (for VBO initialization)

PyObject* drawHomeCircle_hliGLutils(PyObject *self, PyObject *args) {
   PyObject* py_list;
   PyObject* py_tuple;
   PyObject* py_float;
   GLfloat *bulbColors;
   GLfloat gx, gy, wx, wy, ao, w2h; 
   GLfloat R, G, B;
   GLint numBulbs;

   if (!PyArg_ParseTuple(args,
            "fffflffO",
            &gx, &gy,      // background position (X, Y)
            &wx, &wy,      // background scale (X, Y)
            &numBulbs,     // number of elements
            &ao,           // background rotation angle
            &w2h,          // width to height ratio
            &py_list       // colors of the background segments
            ))
   {
      Py_RETURN_NONE;
   }
   //char circleSegments = 60/numBulbs;
   char circleSegments = 60;

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new float[numBulbs*3];
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = float(PyFloat_AsDouble(py_float));
      }
   }

   // Allocate and Define Geometry/Color buffers
   if (homeCircleCoordBuffer  == NULL  ||
       homeCircleColorBuffer  == NULL  ||
       homeCircleIndices      == NULL  ){

      printf("Generating geometry for homeCircle\n");
      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma;
      
      for (int j = 0; j < numBulbs; j++) {
         R = float(bulbColors[j*3+0]);
         G = float(bulbColors[j*3+1]);
         B = float(bulbColors[j*3+2]);
         for (int i = 0; i < circleSegments/numBulbs; i++) {
            verts.push_back(float(0.0));
            verts.push_back(float(0.0));

            tma = float(degToRad(i*float(degSegment) + j*angOffset - 90.0));
            verts.push_back(float(cos(tma)));
            verts.push_back(float(sin(tma)));

            tma = float(degToRad((i+1)*float(degSegment) + j*angOffset - 90.0));
            verts.push_back(float(cos(tma)));
            verts.push_back(float(sin(tma)));

            colrs.push_back(R);   colrs.push_back(G);   colrs.push_back(B);
            colrs.push_back(R);   colrs.push_back(G);   colrs.push_back(B);
            colrs.push_back(R);   colrs.push_back(G);   colrs.push_back(B);
         }
      }

      homeCircleVerts = verts.size()/2;
      printf("homeCircle vertexBuffer length: %.i, Number of vertices: %.i, tris: %.i\n", homeCircleVerts*2, homeCircleVerts, homeCircleVerts/3);

      if (homeCircleCoordBuffer == NULL) {
         homeCircleCoordBuffer = new GLfloat[homeCircleVerts*2];
      } else {
         delete [] homeCircleCoordBuffer;
         homeCircleCoordBuffer = new GLfloat[homeCircleVerts*2];
      }

      if (homeCircleColorBuffer == NULL) {
         homeCircleColorBuffer = new GLfloat[homeCircleVerts*3];
      } else {
         delete [] homeCircleColorBuffer;
         homeCircleColorBuffer = new GLfloat[homeCircleVerts*3];
      }

      if (homeCircleIndices == NULL) {
         homeCircleIndices = new GLushort[homeCircleVerts];
      } else {
         delete [] homeCircleIndices;
         homeCircleIndices = new GLushort[homeCircleVerts];
      }

      for (unsigned int i = 0; i < homeCircleVerts; i++) {
         homeCircleCoordBuffer[i*2+0] = verts[i*2+0];
         homeCircleCoordBuffer[i*2+1] = verts[i*2+1];
         homeCircleColorBuffer[i*3+0]  = colrs[i*3+0];
         homeCircleColorBuffer[i*3+1]  = colrs[i*3+1];
         homeCircleColorBuffer[i*3+2]  = colrs[i*3+2];
         homeCircleIndices[i]          = i;
      }

      // Calculate initial transformation matrix
      Matrix Ortho;
      Matrix ModelView;
      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixLoadIdentity( &ModelView );
      MatrixScale( &ModelView, 2.0f, 2.0f , 1.0f );
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &homeCircleMVP, &ModelView, &Ortho );
      homeCirclePrevState.ao = ao;

      prevHomeCircleNumBulbs = numBulbs;
      homeCirclePrevState.ao = ao;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (homeCircleFirstRun == GL_TRUE) {
         homeCircleFirstRun = GL_FALSE;
         glGenBuffers(1, &homeCircleVBO);
      } else {
         glDeleteBuffers(1, &homeCircleVBO);
         glGenBuffers(1, &homeCircleVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, homeCircleVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*homeCircleVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLintptr offset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*homeCircleVerts, homeCircleCoordBuffer);
      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*homeCircleVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*homeCircleVerts, homeCircleColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   } 

   // Geometry already calculated, update colors
   // Iterate through each color channel 
   // 0 - RED
   // 1 - GREEN
   // 2 - BLUE
   for (int i = 0; i < 3; i++) {
         
      // Update color, if needed
      for (int j = 0; j < numBulbs; j++) {
         if (float(bulbColors[ i + j*3 ]) != homeCircleColorBuffer[ i + j*(60/numBulbs)*9 ] ||
               prevHomeCircleNumBulbs != numBulbs) {
            for (int k = 0; k < (60/numBulbs)*3; k++) {
               if (float(bulbColors[ i + j*3 ]) != homeCircleColorBuffer[ i + k*3 + j*(60/numBulbs)*9 ]) {
                  homeCircleColorBuffer[ j*(60/numBulbs)*9 + k*3 + i ] = float(bulbColors[i+j*3]);
               }
            }
            // Update Contents of VBO
            // Set active VBO
            glBindBuffer(GL_ARRAY_BUFFER, homeCircleVBO);
            // Convenience variable
            GLintptr offset = 2*sizeof(GLfloat)*homeCircleVerts;
            // Load Vertex Color data into VBO
            glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*homeCircleVerts, homeCircleColorBuffer);
         }
      }
   }

   prevHomeCircleNumBulbs = numBulbs;
   delete [] bulbColors;

   // Update Transfomation Matrix if any change in parameters
   if (  homeCirclePrevState.ao != ao                       ||
         homeCirclePrevState.ao != homeCirclePrevState.ao   ){
      Matrix Ortho;
      Matrix ModelView;
      float left = -1.0f*w2h, right = 1.0f*w2h, bottom = 1.0f, top = 1.0f, near = 1.0f, far = 1.0f;
      MatrixLoadIdentity( &Ortho );
      MatrixOrtho( &Ortho, left, right, bottom, top, near, far );
      MatrixLoadIdentity( &ModelView );
      MatrixScale( &ModelView, 2.0f, 2.0f , 1.0f );
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &homeCircleMVP, &ModelView, &Ortho );
      homeCirclePrevState.ao = ao;
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &homeCircleMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, homeCircleVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*homeCircleVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, homeCircleVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   Py_RETURN_NONE;
}
*/

/*
 * Explanation of features:
 * <= 0: just the color representation
 * <= 1: color representation + outline
 * <= 2: color representation + outline + bulb markers
 * <= 3: color representation + outline + bulb markers + bulb marker halos
 * <= 4: color representation + outline + bulb markers + bulb marker halos + grand halo
 */
GLfloat     *iconCircleCoordBuffer  = NULL; // Stores (X, Y) (float) for each vertex
GLfloat     *iconCircleColorBuffer  = NULL; // Stores (R, G, B) (float) for each vertex
GLushort    *iconCircleIndices      = NULL; // Stores index corresponding to each vertex
GLfloat     *iconCircleBulbVertices = NULL;
GLuint      iconCircleVerts;
int         prevIconCircleNumBulbs;
int         prevIconCircleFeatures;
Matrix      iconCircleMVP;                  // Transformation matrix passed to shader
Params      iconCirclePrevState;            // Stores transformations to avoid redundant recalculation
GLuint      iconCircleVBO;                  // Vertex Buffer Object ID
GLboolean   iconCircleFirstRun = GL_TRUE;   // Determines if function is running for the first time (for VBO initialization)

PyObject* drawIconCircle_hliGLutils(PyObject *self, PyObject *args) {
   PyObject*   detailColorPyTup;
   PyObject*   py_list;
   PyObject*   py_tuple;
   PyObject*   py_float;
   GLfloat*    bulbColors;
   GLfloat     detailColor[3];
   GLfloat     gx, gy, scale, ao, w2h;
   long        numBulbs, features;
   GLint       vertIndex = 0;
   if (!PyArg_ParseTuple(args,
            "ffflOlffO",
            &gx, &gy,            // icon positon (X, Y)
            &scale,              // icon size
            &features,           // iconography features
            &detailColorPyTup,   // feature colors
            &numBulbs,           // number of elements
            &ao,                 // icon rotation angle
            &w2h,                // width to hight ration
            &py_list             // colors of the elements (bulbs)
            ))
   {
      Py_RETURN_NONE;
   }

   char circleSegments = 60;

   // Parse array of tuples containing RGB Colors of bulbs
   bulbColors = new float[numBulbs*3];
   for (int i = 0; i < numBulbs; i++) {
      py_tuple = PyList_GetItem(py_list, i);

      for (int j = 0; j < 3; j++) {
         py_float = PyTuple_GetItem(py_tuple, j);
         bulbColors[i*3+j] = float(PyFloat_AsDouble(py_float));
      }
   }

   // Parse RGB detail colors
   detailColor[0] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 0)));
   detailColor[1] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 1)));
   detailColor[2] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, 2)));

   // Allocate and Define Geometry/Color buffers
   if (iconCircleCoordBuffer  == NULL  ||
       iconCircleColorBuffer  == NULL  ||
       iconCircleIndices      == NULL  ||
       iconCircleBulbVertices == NULL  ){

      printf("Generating geometry for iconCircle\n");
      vector<GLfloat> markerVerts;
      vector<GLfloat> markerColrs;

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      char degSegment = 360 / circleSegments;
      float angOffset = float(360.0 / float(numBulbs));
      float tma, tmx, tmy, delta, R, G, B;

      drawEllipse(float(0.0), float(0.0), float(0.16), circleSegments/3, detailColor, markerVerts, markerColrs);
      drawHalo(float(0.0), float(0.0), float(0.22), float(0.22), float(0.07), circleSegments/3, detailColor, markerVerts, markerColrs);

      // Draw Only the color wheel if 'features' <= 0
      delta = degSegment;
      for (int j = 0; j < numBulbs; j++) {
         R = bulbColors[j*3+0];
         G = bulbColors[j*3+1];
         B = bulbColors[j*3+2];
         for (int i = 0; i < circleSegments/numBulbs; i++) {
            /* X */ verts.push_back(float(0.0));
            /* Y */ verts.push_back(float(0.0));

            tma = float(degToRad(i*delta + j*angOffset - 90.0));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));

            tma = float(degToRad((i+1)*delta + j*angOffset - 90.0));
            /* X */ verts.push_back(float(cos(tma)));
            /* Y */ verts.push_back(float(sin(tma)));

            /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
            /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
            /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
         }
      }

      // Draw Color Wheel + Outline if 'features' >= 1
      R = detailColor[0];
      G = detailColor[1];
      B = detailColor[2];
      delta = float(degSegment);
      if (features >= 1) {
         tmx = 0.0;
         tmy = 0.0;
      } else {
         tmx = offScreen;
         tmy = offScreen;
      }
      drawHalo(
            tmx, tmy,
            float(1.0), float(1.0),
            float(0.1),
            circleSegments,
            detailColor,
            verts, colrs);

      // Draw Color Wheel + Outline + BulbMarkers if 'features' >= 2
      int iUlim = circleSegments/3;
      int tmo = 180/numBulbs;
      degSegment = 360/iUlim;
      for (int j = 0; j < 6; j++) {
         if ( (j < numBulbs) && (features >= 2) ) {
            tmx = float(cos(degToRad(-90 - j*(angOffset) + tmo))*1.05);
            tmy = float(sin(degToRad(-90 - j*(angOffset) + tmo))*1.05);
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
         drawEllipse(tmx, tmy, float(0.16), circleSegments/3, detailColor, verts, colrs);
      }

      // Draw Halos for bulb Markers
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos if 'features' == 3
      for (int j = 0; j < 6; j++) {
         if (j < numBulbs && features >= 3) {
            tmx = float(cos(degToRad(-90 - j*(angOffset) + tmo))*1.05);
            tmy = float(sin(degToRad(-90 - j*(angOffset) + tmo))*1.05);
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
         drawHalo(
               tmx, tmy, 
               float(0.22), float(0.22), 
               float(0.07), 
               circleSegments/3, 
               detailColor, 
               verts, colrs);
      }
       
      // Draw Grand (Room) Halo
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos + Grand Halo if 'features' == 4
      if (features >= 4) {
         tmx = 0.0;
         tmy = 0.0;
      } else {
         tmx = offScreen;
         tmy = offScreen;
      }
      drawHalo(
            tmx, tmy,
            float(1.28), float(1.28),
            float(0.08),
            circleSegments,
            detailColor,
            verts, colrs);

      iconCircleVerts = verts.size()/2;
      printf("iconCircle vertexBuffer length: %.i, Number of vertices: %.i, tris: %.i\n", iconCircleVerts*2, iconCircleVerts, iconCircleVerts/3);

      // Safely (Re)allocate memory for bulb marker vertices
      if (iconCircleBulbVertices == NULL) {
         iconCircleBulbVertices = new GLfloat[markerVerts.size()];
      } else {
         delete [] iconCircleBulbVertices;
         iconCircleBulbVertices = new GLfloat[markerVerts.size()];
      }

      // Safely (Re)allocate memory for icon Vertex Buffer
      if (iconCircleCoordBuffer == NULL) {
         iconCircleCoordBuffer = new GLfloat[iconCircleVerts*2];
      } else {
         delete [] iconCircleCoordBuffer;
         iconCircleCoordBuffer = new GLfloat[iconCircleVerts*2];
      }

      // Safely (Re)allocate memory for icon Color Buffer
      if (iconCircleColorBuffer == NULL) {
         iconCircleColorBuffer = new GLfloat[iconCircleVerts*3];
      } else {
         delete [] iconCircleColorBuffer;
         iconCircleColorBuffer = new GLfloat[iconCircleVerts*3];
      }

      // Safely (Re)allocate memory for icon indices
      if (iconCircleIndices == NULL) {
         iconCircleIndices = new GLushort[iconCircleVerts];
      } else {
         delete [] iconCircleIndices;
         iconCircleIndices = new GLushort[iconCircleVerts];
      }

      for (unsigned int i = 0; i < markerVerts.size()/2; i++) {
         iconCircleBulbVertices[i*2+0] = markerVerts[i*2+0];
         iconCircleBulbVertices[i*2+1] = markerVerts[i*2+1];
      }

      for (unsigned int i = 0; i < iconCircleVerts; i++) {
         iconCircleCoordBuffer[i*2+0] = verts[i*2+0];
         iconCircleCoordBuffer[i*2+1] = verts[i*2+1];
         iconCircleColorBuffer[i*3+0]  = colrs[i*3+0];
         iconCircleColorBuffer[i*3+1]  = colrs[i*3+1];
         iconCircleColorBuffer[i*3+2]  = colrs[i*3+2];
         iconCircleIndices[i]          = i;
      }

      // Calculate initial transformation matrix
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
      MatrixMultiply( &iconCircleMVP, &ModelView, &Ortho );

      iconCirclePrevState.ao = ao;
      iconCirclePrevState.dx = gx;
      iconCirclePrevState.dy = gy;
      iconCirclePrevState.sx = scale;
      iconCirclePrevState.sy = scale;
      iconCirclePrevState.w2h = w2h;

      // Update State machine variables
      prevIconCircleNumBulbs = numBulbs;
      prevIconCircleFeatures = features;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (iconCircleFirstRun == GL_TRUE) {
         iconCircleFirstRun = GL_FALSE;
         glGenBuffers(1, &iconCircleVBO);
      } else {
         glDeleteBuffers(1, &iconCircleVBO);
         glGenBuffers(1, &iconCircleVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, iconCircleVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*iconCircleVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLintptr offset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*iconCircleVerts, iconCircleCoordBuffer);
      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*iconCircleVerts;

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*iconCircleVerts, iconCircleColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
   } 
   // Update Geometry, if alreay allocated
   if (prevIconCircleFeatures != features ||
       prevIconCircleNumBulbs != numBulbs ){

      float angOffset = float(360.0 / float(numBulbs));
      float tmx, tmy;
      vertIndex = 0;
      
      /*
       * Explanation of features:
       * <= 0: just the color representation
       * <= 1: color representation + outline
       * <= 2: color representation + outline + bulb markers
       * <= 3: color representation + outline + bulb markers + bulb marker halos
       * <= 4: color representation + outline + bulb markers + bulb marker halos + grand halo
       */
      // Draw Only the color wheel if 'features' <= 0
      // Update Color Wheel
      for (int j = 0; j < numBulbs; j++) {
         for (int i = 0; i < circleSegments/numBulbs; i++) {
            vertIndex += 6;
         }
      }

      // Draw Color Wheel + Outline if 'features' == 1
      // Update Outline
      if (features >= 1) {
         // Move Outline on-screen if off-screen
         if (iconCircleCoordBuffer[vertIndex] > offScreen/2) {
            tmx = -offScreen;
            tmy = -offScreen;
         } else {
            tmx = 0.0;
            tmy = 0.0;
         }
      } else {
         // Move Outline of screen if on screen
         if (iconCircleCoordBuffer[vertIndex] > offScreen/2) {
            tmx = 0.0;
            tmy = 0.0;
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
      }

      for (int i = 0; i < circleSegments; i++) {
         /* X */ iconCircleCoordBuffer[vertIndex +  0] = iconCircleCoordBuffer[vertIndex +  0] + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex +  1] = iconCircleCoordBuffer[vertIndex +  1] + tmy;
         /* X */ iconCircleCoordBuffer[vertIndex +  2] = iconCircleCoordBuffer[vertIndex +  2] + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex +  3] = iconCircleCoordBuffer[vertIndex +  3] + tmy;
         /* X */ iconCircleCoordBuffer[vertIndex +  4] = iconCircleCoordBuffer[vertIndex +  4] + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex +  5] = iconCircleCoordBuffer[vertIndex +  5] + tmy;

         /* X */ iconCircleCoordBuffer[vertIndex +  6] = iconCircleCoordBuffer[vertIndex +  6] + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex +  7] = iconCircleCoordBuffer[vertIndex +  7] + tmy;
         /* X */ iconCircleCoordBuffer[vertIndex +  8] = iconCircleCoordBuffer[vertIndex +  8] + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex +  9] = iconCircleCoordBuffer[vertIndex +  9] + tmy;
         /* X */ iconCircleCoordBuffer[vertIndex + 10] = iconCircleCoordBuffer[vertIndex + 10] + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex + 11] = iconCircleCoordBuffer[vertIndex + 11] + tmy;
         vertIndex += 12;
      }

      // Update Bulb Markers
      // Draw Color Wheel + Outline + BulbMarkers if 'features' >= 2
      int iUlim = circleSegments/3;
      for (int j = 0; j < 6; j++) {
         if (j < numBulbs && features >= 2) {
            tmx = float(cos(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
            tmy = float(sin(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
         for (int i = 0; i < iUlim; i++) {
            /* X */ iconCircleCoordBuffer[vertIndex++] = tmx + iconCircleBulbVertices[i*6+0];
            /* Y */ iconCircleCoordBuffer[vertIndex++] = tmy + iconCircleBulbVertices[i*6+1];

            /* X */ iconCircleCoordBuffer[vertIndex++] = tmx + iconCircleBulbVertices[i*6+2];
            /* Y */ iconCircleCoordBuffer[vertIndex++] = tmy + iconCircleBulbVertices[i*6+3];

            /* X */ iconCircleCoordBuffer[vertIndex++] = tmx + iconCircleBulbVertices[i*6+4];
            /* Y */ iconCircleCoordBuffer[vertIndex++] = tmy + iconCircleBulbVertices[i*6+5];
         }
      }

      // Draw Halos for bulb Markers
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos if 'features' == 3
      for (int j = 0; j < 6; j++) {
         if (j < numBulbs && features >= 3) {
            tmx = float(cos(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
            tmy = float(sin(degToRad(-90 - j*(angOffset) + 180/numBulbs))*1.05);
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }

         for (int i = 0; i < iUlim; i++) {
            /* X */ iconCircleCoordBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 +  0];
            /* Y */ iconCircleCoordBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 +  1];
            /* X */ iconCircleCoordBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 +  2];
            /* Y */ iconCircleCoordBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 +  3];
            /* X */ iconCircleCoordBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 +  4];
            /* Y */ iconCircleCoordBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 +  5];
            /* X */ iconCircleCoordBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 +  6];
            /* Y */ iconCircleCoordBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 +  7];
            /* X */ iconCircleCoordBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 +  8];
            /* Y */ iconCircleCoordBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 +  9];
            /* X */ iconCircleCoordBuffer[vertIndex++] = tmx + iconCircleBulbVertices[iUlim*6 + i*12 + 10];
            /* Y */ iconCircleCoordBuffer[vertIndex++] = tmy + iconCircleBulbVertices[iUlim*6 + i*12 + 11];
         }
      }

      // Update Grand (Room) Outline
      // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos + Grand Halo if 'features' == 4
      if (features >= 4) {
         // Move Outline on-screen if off-screen
         if (iconCircleCoordBuffer[vertIndex] > offScreen/2) {
            tmx = -offScreen;
            tmy = -offScreen;
         } else {
            tmx = 0.0;
            tmy = 0.0;
         }
      } else {
         // Move Outline of screen if on screen
         if (iconCircleCoordBuffer[vertIndex] > offScreen/2) {
            tmx = 0.0;
            tmy = 0.0;
         } else {
            tmx = offScreen;
            tmy = offScreen;
         }
      }

      for (int i = 0; i < circleSegments; i++) {
         /* X */ iconCircleCoordBuffer[vertIndex +  0] = iconCircleCoordBuffer[vertIndex +  0]  + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex +  1] = iconCircleCoordBuffer[vertIndex +  1]  + tmy;
         /* X */ iconCircleCoordBuffer[vertIndex +  2] = iconCircleCoordBuffer[vertIndex +  2]  + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex +  3] = iconCircleCoordBuffer[vertIndex +  3]  + tmy;
         /* X */ iconCircleCoordBuffer[vertIndex +  4] = iconCircleCoordBuffer[vertIndex +  4]  + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex +  5] = iconCircleCoordBuffer[vertIndex +  5]  + tmy;

         /* X */ iconCircleCoordBuffer[vertIndex +  6] = iconCircleCoordBuffer[vertIndex +  6]  + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex +  7] = iconCircleCoordBuffer[vertIndex +  7]  + tmy;
         /* X */ iconCircleCoordBuffer[vertIndex +  8] = iconCircleCoordBuffer[vertIndex +  8]  + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex +  9] = iconCircleCoordBuffer[vertIndex +  9]  + tmy;
         /* X */ iconCircleCoordBuffer[vertIndex + 10] = iconCircleCoordBuffer[vertIndex + 10]  + tmx;
         /* Y */ iconCircleCoordBuffer[vertIndex + 11] = iconCircleCoordBuffer[vertIndex + 11]  + tmy;
         vertIndex += 12;
      }

      prevIconCircleFeatures = features;
      // Update Contents of VBO
      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, iconCircleVBO);
      // Convenience variable
      GLintptr offset = 0;
      // Load Vertex Color data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*iconCircleVerts, iconCircleCoordBuffer);
   }
   // Geometry already calculated, update colors
   /*
    * Iterate through each color channel 
    * 0 - RED
    * 1 - GREEN
    * 2 - BLUE
    */
   for (int i = 0; i < 3; i++) {
         
      // Update color wheel, if needed
      for (int j = 0; j < numBulbs; j++) {
         int tmo = (60/numBulbs)*9;
         if (float(bulbColors[i+j*3]) != iconCircleColorBuffer[i + j*tmo]
               || prevIconCircleNumBulbs != numBulbs){
            for (int k = 0; k < tmo/3; k++) {
               if (float(bulbColors[i+j*3]) != iconCircleColorBuffer[i + k*3 + j*tmo]){
                  iconCircleColorBuffer[j*tmo + k*3 + i] = float(bulbColors[i+j*3]);
               }
            }
            // Update Contents of VBO
            // Set active VBO
            glBindBuffer(GL_ARRAY_BUFFER, iconCircleVBO);
            // Convenience variable
            GLintptr offset = 2*sizeof(GLfloat)*iconCircleVerts;
            // Load Vertex Color data into VBO
            glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*iconCircleVerts, iconCircleColorBuffer);
         }
      }

      // Update outline, bulb markers, if needed
      if (float(detailColor[i]) != float(iconCircleColorBuffer[i + circleSegments*9])) {
         for (unsigned int k = circleSegments*3; k < iconCircleVerts; k++) {
            iconCircleColorBuffer[ k*3 + i ] = float(detailColor[i]);
         }
         // Update Contents of VBO
         // Set active VBO
         glBindBuffer(GL_ARRAY_BUFFER, iconCircleVBO);
         // Convenience variable
         GLintptr offset = 2*sizeof(GLfloat)*iconCircleVerts;
         // Load Vertex Color data into VBO
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*3*iconCircleVerts, iconCircleColorBuffer);
      }
   }

   prevIconCircleNumBulbs = numBulbs;
   delete [] bulbColors;

   // Update Transfomation Matrix if any change in parameters
   if (  iconCirclePrevState.ao != ao     ||
         iconCirclePrevState.dx != gx     ||
         iconCirclePrevState.dy != gy     ||
         iconCirclePrevState.sx != scale  ||
         iconCirclePrevState.sy != scale  ||
         iconCirclePrevState.w2h != w2h   ){
      
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
      MatrixMultiply( &iconCircleMVP, &ModelView, &Ortho );

      iconCirclePrevState.ao = ao;
      iconCirclePrevState.dx = gx;
      iconCirclePrevState.dy = gy;
      iconCirclePrevState.sx = scale;
      iconCirclePrevState.sy = scale;
      iconCirclePrevState.w2h = w2h;
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &iconCircleMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, iconCircleVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*iconCircleVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, iconCircleVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   Py_RETURN_NONE;
}
