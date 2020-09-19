using namespace std;

extern std::map<std::string, drawCall> drawCalls;
extern GLint uniform_tex;

PyObject* drawImageSquare_hliGLutils(PyObject* self, PyObject* args) {
   PyObject* imageName;          // relative path + filename for image
   PyObject* imageBuffer;        // buffer containing pixel data
   PyObject* colorPyTup;         // RGBA tuple
   float gx, gy, ao, scale, xRes, yRes, w2h;
   bool  copyBuffer = false;
   float color[4];
   //float detailColor[4];

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "OOfffffffO",
            &imageName,
            &imageBuffer,
            &gx, 
            &gy,
            &ao,
            &scale,
            &xRes, 
            &yRes,
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

   // Parse image name
   const char* imgNameChars = PyUnicode_AsUTF8(imageName);
   std::string imgNameString = imgNameChars;
   //printf("quack\n");

   if (drawCalls.count(imgNameString) <= 0){
      drawCalls.insert(std::make_pair(imgNameString, drawCall()));
      copyBuffer = true;
   }
   drawCall* image = &drawCalls[imgNameString];

   if (copyBuffer) {
      printf("building image texture...\n");
      unsigned int bufferLength;
      bufferLength = (unsigned int)(xRes*yRes);
      GLubyte* tmb = new GLubyte[bufferLength*4];
      PyObject* rpx;
      PyObject* gpx;
      PyObject* bpx;
      PyObject* apx;
      for (unsigned int i = 0; i < bufferLength; i++) {
         rpx = PyList_GetItem(imageBuffer, i*4+0);
         tmb[i*4+0] = GLubyte(PyLong_AsLong(rpx));

         gpx = PyList_GetItem(imageBuffer, i*4+1);
         tmb[i*4+1] = GLubyte(PyLong_AsLong(gpx));

         bpx = PyList_GetItem(imageBuffer, i*4+2);
         tmb[i*4+2] = GLubyte(PyLong_AsLong(bpx));

         apx = PyList_GetItem(imageBuffer, i*4+3);
         tmb[i*4+3] = GLubyte(PyLong_AsLong(apx));
      }

      glActiveTexture(GL_TEXTURE0);
      glGenTextures(1, &image->texID);
      printf("image texture id: %x\n", image->texID);
      glBindTexture(GL_TEXTURE_2D, image->texID);
      glUniform1i(uniform_tex, 0);

      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (unsigned int)xRes, (unsigned int)yRes, 0, GL_RGBA, GL_UNSIGNED_BYTE, tmb);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

      //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
      //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

      //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      glBindTexture(GL_TEXTURE_2D, 0);
      printf("finished building texture\n");
   }

   drawImageSquare(
         gx,
         gy,
         ao,
         scale,
         w2h,
         color,
         image
         );

   Py_RETURN_NONE;
}

void drawImageSquare(
      GLfloat     gx, 
      GLfloat     gy,
      GLfloat     ao,
      GLfloat     scale,
      GLfloat     w2h,
      GLfloat*    color,
      drawCall*   image
      ){

   GLuint imageVerts;
   //image->setDrawType(GL_TRIANGLES);
   image->setNumColors(1);
   image->setColorQuartet(0, color);

   if (  image->numVerts   == 0
      ){

      std::vector <GLfloat> verts;
      std::vector <GLfloat> colrs;
      std::vector <GLfloat> texuv;
      imageVerts = defineTexQuad(
            0.0f, 0.0f,
            1.0f, 1.0f,
            color,
            verts, texuv, colrs
            );
      map<string, attribCache> attributeData;
      // Define vertex attributes, initialize caches
      attributeData[VAS.coordData] = attribCache(VAS.coordData, 2, 0, 0);
      attributeData[VAS.texuvData] = attribCache(VAS.texuvData, 2, 2, 1);
      attributeData[VAS.colorData] = attribCache(VAS.colorData, 4, 4, 2);
      attributeData[VAS.coordData].writeCache(verts.data(), verts.size());
      attributeData[VAS.texuvData].writeCache(texuv.data(), texuv.size());
      attributeData[VAS.colorData].writeCache(colrs.data(), colrs.size());
      image->buildCache(imageVerts, attributeData);
   }

   if (image->colorsChanged){
      GLuint index = 0;
      index = updateQuadColor(color, index, (GLfloat *)image->getAttribCache(VAS.colorData));
      image->updateBuffer(VAS.colorData);
   }

   image->updateMVP(gx, gy, scale, -scale, ao, w2h);
   image->draw();
   return;
}
