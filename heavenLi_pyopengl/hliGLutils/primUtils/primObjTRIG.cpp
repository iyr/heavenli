/*
 * Read Wavefront Object file from disk, parse, build vectors
 */
unsigned int defineObjTrig(
      string         filepath,
      float*         color,
      vector<float>  &verts,
      vector<float>  &texuv,
      vector<float>  &colrs,
      vector<float>  &nrmls
      ){
   vector<float>  coords, uvcoords, normals;
   FILE*          obj      = fopen(filepath.c_str(), "r");
   unsigned short buffsize = 128;
   char*          lineBuff = (char *)malloc(buffsize);
   
   // Fail-safe
   if (obj == NULL){
      printf("error reading file\n");
      free(lineBuff);
      return 0;
   }

   printf("reading obj...\n");
   while (fscanf(obj, "%s", lineBuff) == 1) {
      if (strcmp(lineBuff, "v") == 0) {
         float x, y, z, w = 1.0f;
         fscanf(obj, "%f %f %f\n", &x, &y, &z);
         coords.push_back(x);
         coords.push_back(y);
         coords.push_back(z);
         coords.push_back(w);
         //printf("vertex: %3.5f, %3.5f, %3.5f, %3.5f\n",x,y,z,w);
      } else if (strcmp(lineBuff, "vt") == 0) {
         float u, v = 0.0f;
         fscanf(obj, "%f %f\n", &u, &v);
         uvcoords.push_back(u);
         uvcoords.push_back(v);
      } else if (strcmp(lineBuff, "vn") == 0) {
         float x, y, z, w = 1.0f;
         fscanf(obj, "%f %f %f\n", &x, &y, &z);
         normals.push_back(x);
         normals.push_back(y);
         normals.push_back(z);
         normals.push_back(w);
      } else if (strcmp(lineBuff, "f") == 0) {
         //static bool flipTri = false;
         static bool flipTri = true;
         int vertInd[3], uvInd[3], nrmInd[3];
         int matches = fscanf(obj, "%d/%d/%d %d/%d/%d %d/%d/%d\n",
               &vertInd[0],
               &uvInd [0],
               &nrmInd[0],
               &vertInd[1],
               &uvInd [1],
               &nrmInd[1],
               &vertInd[2],
               &uvInd [2],
               &nrmInd[2]
               );
         if (matches != 9) {
            printf("incompatible file\n");
            free(lineBuff);
            fclose(obj);
            return 0;
         }

         for (unsigned int i = 0; i < 3; i++){
            vertInd[i]--;
            nrmInd[i]--;
            uvInd[i]--;
         }

         // Used to adjust vertex order
         int triInd[3];

         float U[3];
         float V[3];
         float trigNorm[3];
         float meanNorm[3];

         // p1 - p0
         U[0] = coords[vertInd[1]*4+0] - coords[vertInd[0]*4+0];
         U[1] = coords[vertInd[1]*4+1] - coords[vertInd[0]*4+1];
         U[2] = coords[vertInd[1]*4+2] - coords[vertInd[0]*4+2];

         // p2 - p0
         V[0] = coords[vertInd[2]*4+0] - coords[vertInd[0]*4+0];
         V[1] = coords[vertInd[2]*4+1] - coords[vertInd[0]*4+1];
         V[2] = coords[vertInd[2]*4+2] - coords[vertInd[0]*4+2];

         // Calculate Normal vector of the three points
         trigNorm[0] = U[1]*V[2] - U[2]*V[1];
         trigNorm[1] = U[2]*V[0] - U[0]*V[2];
         trigNorm[2] = U[0]*V[1] - U[1]*V[0];

         // Average the normal vectors of the vertices read from file
         meanNorm[0] = (
               normals[nrmInd[0]*4+0] + 
               normals[nrmInd[1]*4+0] +
               normals[nrmInd[2]*4+0])/3.0f;
         meanNorm[1] = (
               normals[nrmInd[0]*4+1] + 
               normals[nrmInd[1]*4+1] +
               normals[nrmInd[2]*4+1])/3.0f;
         meanNorm[2] = (
               normals[nrmInd[0]*4+2] + 
               normals[nrmInd[1]*4+2] +
               normals[nrmInd[2]*4+2])/3.0f;

         // Compute dot product of normals
         float dot = 
            trigNorm[0]*meanNorm[0] +
            trigNorm[1]*meanNorm[1] +
            trigNorm[2]*meanNorm[2];

         //printf("trigNorm x: %.3f trigNorm y: %.3f trigNorm z: %.3f \nmeanNorm x: %.3f meanNorm y: %.3f meanNorm z: %.3f \ndot product: %1.5f\n", 
               //trigNorm[0],
               //trigNorm[1],
               //trigNorm[2],
               //meanNorm[0],
               //meanNorm[1],
               //meanNorm[2],
               //dot
               //);
         if (dot > 0.0f) {
            triInd[0] = 0;
            triInd[1] = 1;
            triInd[2] = 2;
         } else {
            triInd[0] = 2;
            triInd[1] = 1;
            triInd[2] = 0;
         }

         verts.push_back(coords[vertInd[triInd[0]]*4+0]);
         verts.push_back(coords[vertInd[triInd[0]]*4+1]);
         verts.push_back(coords[vertInd[triInd[0]]*4+2]);
         verts.push_back(coords[vertInd[triInd[0]]*4+3]);
         nrmls.push_back(normals[nrmInd[triInd[0]]*4+0]);
         nrmls.push_back(normals[nrmInd[triInd[0]]*4+1]);
         nrmls.push_back(normals[nrmInd[triInd[0]]*4+2]);
         nrmls.push_back(normals[nrmInd[triInd[0]]*4+3]);
         texuv.push_back(uvcoords[uvInd[triInd[0]]*2+0]);
         texuv.push_back(uvcoords[uvInd[triInd[0]]*2+1]);
         //colrs.push_back(1.0f);
         //colrs.push_back(1.0f);
         //colrs.push_back(0.0f);
         //colrs.push_back(1.0f);
         colrs.push_back(color[0]);
         colrs.push_back(color[1]);
         colrs.push_back(color[2]);
         colrs.push_back(color[3]);
         verts.push_back(coords[vertInd[triInd[1]]*4+0]);
         verts.push_back(coords[vertInd[triInd[1]]*4+1]);
         verts.push_back(coords[vertInd[triInd[1]]*4+2]);
         verts.push_back(coords[vertInd[triInd[1]]*4+3]);
         nrmls.push_back(normals[nrmInd[triInd[1]]*4+0]);
         nrmls.push_back(normals[nrmInd[triInd[1]]*4+1]);
         nrmls.push_back(normals[nrmInd[triInd[1]]*4+2]);
         nrmls.push_back(normals[nrmInd[triInd[1]]*4+3]);
         texuv.push_back(uvcoords[uvInd[triInd[1]]*2+0]);
         texuv.push_back(uvcoords[uvInd[triInd[1]]*2+1]);
         //colrs.push_back(1.0f);
         //colrs.push_back(0.0f);
         //colrs.push_back(1.0f);
         //colrs.push_back(1.0f);
         colrs.push_back(color[0]);
         colrs.push_back(color[1]);
         colrs.push_back(color[2]);
         colrs.push_back(color[3]);
         verts.push_back(coords[vertInd[triInd[2]]*4+0]);
         verts.push_back(coords[vertInd[triInd[2]]*4+1]);
         verts.push_back(coords[vertInd[triInd[2]]*4+2]);
         verts.push_back(coords[vertInd[triInd[2]]*4+3]);
         nrmls.push_back(normals[nrmInd[triInd[2]]*4+0]);
         nrmls.push_back(normals[nrmInd[triInd[2]]*4+1]);
         nrmls.push_back(normals[nrmInd[triInd[2]]*4+2]);
         nrmls.push_back(normals[nrmInd[triInd[2]]*4+3]);
         texuv.push_back(uvcoords[uvInd[triInd[2]]*2+0]);
         texuv.push_back(uvcoords[uvInd[triInd[2]]*2+1]);
         //colrs.push_back(0.0f);
         //colrs.push_back(1.0f);
         //colrs.push_back(1.0f);
         //colrs.push_back(1.0f);
         colrs.push_back(color[0]);
         colrs.push_back(color[1]);
         colrs.push_back(color[2]);
         colrs.push_back(color[3]);
      } else {
			char junk[1000];
			fgets(junk, 1000, obj);
      }
   } 

   fclose(obj);
   return verts.size()/4;
}
