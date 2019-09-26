#include <math.h>

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

PyObject* drawText_drawUtils(PyObject* self, PyObject *args) {
   PyObject *colourPyTup;
   PyObject *Pystring;

   GLfloat gx, gy, scale, w2h, ao=0.0f;
   GLfloat textColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "OffffO",
            &Pystring,
            &gx, &gy,
            &scale,
            &w2h,
            &colourPyTup) )
   {
      Py_RETURN_NONE;
   }

   const char* inputChars = PyUnicode_AsUTF8(Pystring);
   std::string inputString = inputChars;

   textColor[0]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 0)));
   textColor[1]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 1)));
   textColor[2]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 2)));
   textColor[3]   = float(PyFloat_AsDouble(PyTuple_GetItem(colourPyTup, 3)));

   // (Re)allocate and Define Geometry/Color buffers
   if (  stringCoordBuffer == NULL        ||
         stringColorBuffer == NULL        ||
         stringIndices     == NULL        ||
         prevString        != inputString ){

      vector<float> verts;
      vector<float> colrs;

      /*
      //void render_text(const char *text, float x, float y, float sx, float sy) {
      struct point {
            GLfloat x;
            GLfloat y;
            GLfloat s;
            GLfloat t;
            } coords[6 * strlen(text)];

      int n = 0;
      float x = 0.0;
      float y = 0.0;
      float sx = 10.0;
      float sy = 10.0;

      for(const char *p = inputChars; *p; p++) { 
         float x2 =  x + Characters[*p].bl * sx;
         float y2 = -y - Characters[*p].bt * sy;
         float w = Characters[*p].bw * sx;
         float h = Characters[*p].bh * sy;
   
         // Advance the cursor to the start of the next character 
         x += Characters[*p].ax * sx;
         y += Characters[*p].ay * sy;
   
         // Skip glyphs that have no pixels 
         if(!w || !h)
         continue;
   
         coords[n++] = (point){x2,     -y2    , Characters[*p].tx,                                    0};
         coords[n++] = (point){x2 + w, -y2    , Characters[*p].tx + Characters[*p].bw / atlas_width,  0};
         coords[n++] = (point){x2,     -y2 - h, Characters[*p].tx,                                    Characters[*p].bh / atlas_height}; //remember: each glyph occupies a different amount of vertical space
         coords[n++] = (point){x2 + w, -y2    , Characters[*p].tx + Characters[*p].bw / atlas_width,  0};
         coords[n++] = (point){x2,     -y2 - h, Characters[*p].tx,                                    Characters[*p].bh / atlas_height};
         coords[n++] = (point){x2 + w, -y2 - h, Characters[*p].tx + c[*p].bw / atlas_width,           Characters[*p].bh / atlas_height};
      }

      glBufferData(GL_ARRAY_BUFFER, sizeof coords, coords, GL_DYNAMIC_DRAW);
      glDrawArrays(GL_TRIANGLES, 0, n);
      //}
      */


      stringVerts = verts.size()/2;

      // Pack Vertics and Colors into global array buffers
      if (stringCoordBuffer == NULL) {
         stringCoordBuffer = new GLfloat[stringVerts*2];
      } else {
         delete [] stringCoordBuffer;
         stringCoordBuffer = new GLfloat[stringVerts*2];
      }

      if (stringColorBuffer == NULL) {
         stringColorBuffer = new GLfloat[stringVerts*4];
      } else {
         delete [] stringColorBuffer;
         stringColorBuffer = new GLfloat[stringVerts*4];
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
         stringColorBuffer[i*4+0]  = colrs[i*4+0];
         stringColorBuffer[i*4+1]  = colrs[i*4+1];
         stringColorBuffer[i*4+2]  = colrs[i*4+2];
         stringColorBuffer[i*4+3]  = colrs[i*4+3];
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
      glBufferData(GL_ARRAY_BUFFER, 6*sizeof(GLfloat)*stringVerts, NULL, GL_STATIC_DRAW);

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
      glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(GLfloat)*4*stringVerts, stringColorBuffer);
      // Define how the Vertex color data is layed out in the buffer
      glVertexAttribPointer(vertAttribColor, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (GLintptr*)offset);
      // Enable the vertex attribute
      glEnableVertexAttribArray(vertAttribColor);
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
      glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*stringVerts));
   }

   // Pass Transformation Matrix to shader
   glUniformMatrix4fv( 0, 1, GL_FALSE, &stringMVP.mat[0][0] );

   // Set active VBO
   glBindBuffer(GL_ARRAY_BUFFER, stringVBO);

   // Define how the Vertex coordinate data is layed out in the buffer
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);
   // Define how the Vertex color data is layed out in the buffer
   glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)*stringVerts));
   //glEnableVertexAttribArray(0);
   //glEnableVertexAttribArray(1);
   glDrawArrays(GL_TRIANGLES, 0, stringVerts);

   // Unbind Buffer Object
   glBindBuffer(GL_ARRAY_BUFFER, 0);


   Py_RETURN_NONE;
}