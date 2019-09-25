using namespace std;

/*
GLfloat     *clockCoordBuffer  = NULL; // Stores (X, Y) (float) for each vertex
GLfloat     *clockColorBuffer  = NULL; // Stores (R, G, B) (float) for each vertex
GLushort    *clockIndices      = NULL; // Stores index corresponding to each vertex
Matrix      clockMVP;                  // Transformation matrix passed to shader
Params      clockPrevState;            // Stores transformations to avoid redundant recalculation
GLuint      clockVBO;                  // Vertex Buffer Object ID
GLboolean   clockFirstRun = GL_TRUE;   // Determines if function is running for the first time (for VBO initialization)
*/

drawCall clockButton;
GLfloat  prevClockHour;    // Used for animated hour hand
GLfloat  prevClockMinute;  // Used for animated minute hand
GLuint   clockVerts;       // Total number of vertices
GLuint   faceVerts;        // Number of vertices of face (makes animating hands easier)

PyObject* drawClock_drawUtils(PyObject *self, PyObject *args)
{
   PyObject* faceColorPyTup;
   PyObject* detailColorPyTup;
   GLfloat gx, gy, px, py, qx, qy, radius, ao=0.0f;
   GLfloat scale, w2h, hour, minute;
   GLfloat detailColor[4];
   GLfloat faceColor[4];
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
   for (int i = 0; i < 4; i++){
      faceColor[i] = float(PyFloat_AsDouble(PyTuple_GetItem(faceColorPyTup, i)));
      detailColor[i] = float(PyFloat_AsDouble(PyTuple_GetItem(detailColorPyTup, i)));
   }
   
   /*
   if (  clockCoordBuffer  == NULL  ||
         clockColorBuffer  == NULL  ||
         clockIndices      == NULL  ){
   */
   if (  clockButton.numVerts == 0  ){

      vector<GLfloat> verts;
      vector<GLfloat> colrs;

      defineEllipse(
            0.0f, 0.0f, 
            0.5f, 0.5f,
            circleSegments,
            faceColor,
            verts,
            colrs);

      faceVerts = verts.size()/2;

      px = 0.0;
      py = 0.0;
      qx = float(0.2*cos(degToRad(90-360*(hour/12.0))));
      qy = float(0.2*sin(degToRad(90-360*(hour/12.0))));
      radius = float(0.02);
      definePill(px, py, qx, qy, radius, circleSegments/4, detailColor, verts, colrs);

      qx = float(0.4*cos(degToRad(90-360*(minute/60.0))));
      qy = float(0.4*sin(degToRad(90-360*(minute/60.0))));
      radius = float(0.01);
      definePill(px, py, qx, qy, radius, circleSegments/4, detailColor, verts, colrs);

      clockVerts = verts.size()/2;

      clockButton.buildCache(clockVerts, verts, colrs);

      /*
      // Pack Vertics and Colors into global array buffers
      if (clockCoordBuffer == NULL) {
         clockCoordBuffer = new GLfloat[clockVerts*2];
      } else {
         delete [] clockCoordBuffer;
         clockCoordBuffer = new GLfloat[clockVerts*2];
      }

      if (clockColorBuffer == NULL) {
         clockColorBuffer = new GLfloat[clockVerts*4];
      } else {
         delete [] clockColorBuffer;
         clockColorBuffer = new GLfloat[clockVerts*4];
      }

      if (clockIndices == NULL) {
         clockIndices = new GLushort[clockVerts];
      } else {
         delete [] clockIndices;
         clockIndices = new GLushort[clockVerts];
      }

      for (unsigned int i = 0; i < clockVerts; i++) {
         clockCoordBuffer[i*2]   = verts[i*2];
         clockCoordBuffer[i*2+1] = verts[i*2+1];
         clockIndices[i]         = i;
         clockColorBuffer[i*4+0] = colrs[i*4+0];
         clockColorBuffer[i*4+1] = colrs[i*4+1];
         clockColorBuffer[i*4+2] = colrs[i*4+2];
         clockColorBuffer[i*4+3] = colrs[i*4+3];
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
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &clockMVP, &ModelView, &Ortho );

      clockPrevState.ao = ao;
      clockPrevState.dx = gx;
      clockPrevState.dy = gy;
      clockPrevState.sx = scale;
      clockPrevState.sy = scale;
      clockPrevState.w2h = w2h;

      // Update State Machine variables
      prevClockHour     = hour;
      prevClockMinute   = minute;

      // Create buffer object if one does not exist, otherwise, delete and make a new one
      if (clockFirstRun == GL_TRUE) {
         clockFirstRun = GL_FALSE;
         glGenBuffers(1, &clockVBO);
      } else {
         glDeleteBuffers(1, &clockVBO);
         glGenBuffers(1, &clockVBO);
      }

      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, clockVBO);

      // Allocate space to hold all vertex coordinate and color data
      glBufferData(GL_ARRAY_BUFFER, 6*sizeof(GLfloat)*clockVerts, NULL, GL_STATIC_DRAW);

      // Convenience variables
      GLintptr offset = 0;
      GLuint vertAttribCoord = glGetAttribLocation(3, "vertCoord");
      GLuint vertAttribColor = glGetAttribLocation(3, "vertColor");

      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*clockVerts, clockCoordBuffer);
      // Define how the Vertex coordinate data is layed out in the buffer
      glVertexAttribPointer(vertAttribCoord, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribCoord);

      // Update offset to begin storing data in latter part of the buffer
      offset += 2*sizeof(GLfloat)*clockVerts;

      // Load Vertex Color data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*4*clockVerts, clockColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
      */
   } 

   // Animate Clock Hands
   if (prevClockHour     != hour    ||
       prevClockMinute   != minute  ){
      px = 0.0;
      py = 0.0;
      qx = float(0.2*cos(degToRad(90-360*(hour/12.0))));
      qy = float(0.2*sin(degToRad(90-360*(hour/12.0))));
      radius = float(0.02);

      int tmp;
      tmp = updatePillGeometry(
            px, py,
            qx, qy,
            radius,
            circleSegments/4,
            faceVerts,
            clockButton.coordCache);

      qx = float(0.4*cos(degToRad(90-360*(minute/60.0))));
      qy = float(0.4*sin(degToRad(90-360*(minute/60.0))));
      radius = float(0.01);
      tmp = updatePillGeometry(
            px, py,
            qx, qy,
            radius,
            circleSegments/4,
            tmp,
            clockButton.coordCache);

      prevClockHour     = hour;
      prevClockMinute   = minute;

      clockButton.updateCoordCache();
      /*
      // Update Contents of VBO
      // Set active VBO
      glBindBuffer(GL_ARRAY_BUFFER, clockVBO);
      // Convenience variables
      GLintptr offset = 0;
      // Load Vertex coordinate data into VBO
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*2*clockVerts, clockCoordBuffer);
      */
   }

   GLboolean updateCache = GL_FALSE;
   // Geometry up to date, check if colors need to be updated
   for (int i = 0; i < 4; i++) {
      // Update Clock Face Color
      if (faceColor[i] != clockButton.colorCache[i]) {
         for (unsigned int k = 0; k < faceVerts; k++) {
            clockButton.colorCache[i + k*4] = faceColor[i];
         }
         updateCache = GL_TRUE;
         /*
         // Update Contents of VBO
         // Set active VBO
         glBindBuffer(GL_ARRAY_BUFFER, clockVBO);
         // Convenience variable
         GLintptr offset = 2*sizeof(GLfloat)*clockVerts;
         // Load Vertex Color data into VBO
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*4*clockVerts, clockColorBuffer);
         */

      }

      // Update Hand Colors
      if (detailColor[i] != clockButton.colorCache[i + faceVerts]) {
         for (unsigned int k = faceVerts; k < clockVerts; k++) {
            clockButton.colorCache[i + k*4] = detailColor[i];
         }
         updateCache = GL_TRUE;
         /*
         // Update Contents of VBO
         // Set active VBO
         glBindBuffer(GL_ARRAY_BUFFER, clockVBO);
         // Convenience variable
         GLintptr offset = 2*sizeof(GLfloat)*clockVerts;
         // Load Vertex Color data into VBO
         glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*4*clockVerts, clockColorBuffer);
         */
      }
   }

   if ( updateCache ){
      clockButton.updateColorCache();
   }
   
   /*
   // Update Transfomation Matrix if any change in parameters
   if (  clockPrevState.ao != ao     ||
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
      MatrixRotate( &ModelView, -ao, 0.0f, 0.0f, 1.0f);
      MatrixMultiply( &clockMVP, &ModelView, &Ortho );

      clockPrevState.ao = ao;
      clockPrevState.dx = gx;
      clockPrevState.dy = gy;
      clockPrevState.sx = scale;
      clockPrevState.sy = scale;
      clockPrevState.w2h = w2h;
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &clockMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, clockVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*clockVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, clockVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   */
   
   clockButton.updateMVP(gx, gy, scale, scale, ao, w2h);
   clockButton.draw();

   Py_RETURN_NONE;
}
