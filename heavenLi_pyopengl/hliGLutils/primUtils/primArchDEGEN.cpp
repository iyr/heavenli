/*
 *  Defines a circular Arch for TRIANGLE_STRIP with degenerate vertices
 */

using namespace std;

// Append vertices to input vectors
unsigned int defineArch(
      float bx,                     // X-Coordinate                                          
      float by,                     // Y-Coordinate                                          
      float rx,                     // x inner radius, set to 0.0 for drawing pizzas/slices
      float ry,                     // y inner radius, set to 0.0 for drawing pizzas/slices
      float start,                  // Where, in degrees on the unit circle, the arch begins 
      float end,                    // Where, in degrees on the unit circle, the arch endgs  
      float rs,                     // outer radius
      unsigned int circleSegments,  // Number of sides                                       
      float* color,                 // Polygon Color                                         
      std::vector<float> &verts,    // Input Vector of x,y coordinates                       
      std::vector<float> &colrs     // Input Vector of r,g,b values                          
      ){
   float tma, R, G, B, A;
   float degSegment;
   R = color[0];
   G = color[1];
   B = color[2];
   A = color[3];
   float begin;
   if (start <= end) {
      degSegment = abs(end - start) / float(circleSegments);
      begin = start;
   } else {
      degSegment = ((360.0f-start) + end) / float(circleSegments);
      begin = start;
   }

   // Prepend Degenerate Vertex
   tma = float(degToRad(begin));

   verts.push_back(float(bx+cos(tma)*(rx+rs))); // X
   verts.push_back(float(by+sin(tma)*(ry+rs))); // Y
   colrs.push_back(R);   
   colrs.push_back(G);   
   colrs.push_back(B);
   colrs.push_back(A);

   for (unsigned int i = 0; i < circleSegments+1; i ++ ) {
      tma = float(degToRad(begin+i*degSegment));

      verts.push_back(float(bx+cos(tma)*(rx+rs))); // X
      verts.push_back(float(by+sin(tma)*(ry+rs))); // Y
      colrs.push_back(R);   
      colrs.push_back(G);   
      colrs.push_back(B);
      colrs.push_back(A);

      verts.push_back(float(bx+cos(tma)*rx));   // X
      verts.push_back(float(by+sin(tma)*ry));   // Y
      colrs.push_back(R);   
      colrs.push_back(G);   
      colrs.push_back(B);
      colrs.push_back(A);
   }

   tma = float(degToRad(begin+(circleSegments)*degSegment));

   verts.push_back(float(bx+cos(tma)*rx));   // X
   verts.push_back(float(by+sin(tma)*ry));   // Y
   colrs.push_back(R);   
   colrs.push_back(G);   
   colrs.push_back(B);
   colrs.push_back(A);

   return verts.size()/2;
}

/*
 * Update Vertex position coordinates in pre-existing array passed via pointer starting 
 */
unsigned int updateArchGeometry(
      float bx,                     // X-Coordinate
      float by,                     // Y-Coordinate
      float rx,                    // x-Scale 2.0=spans display before GL scaling
      float ry,                    // y-Scale 2.0=spans display before GL scaling
      float start,                  // Where, in degrees on the unit circle, the arch begins
      float end,                    // Where, in degrees on the unit circle, the arch endgs
      float rs,                     // Halo thickness
      unsigned int circleSegments,  // Number of sides
      unsigned int index,           // Index of where to start writing to input arrays
      float *verts                  // Input Array of x,y coordinates
      ){
   float tma, degSegment, begin;
   unsigned int vertIndex = index*2;
   if (start <= end) {
      degSegment = abs(end - start) / float(circleSegments);
      begin = start;
   } else {
      degSegment = ((360.0f-start) + end) / float(circleSegments);
      begin = start;
   }

   // Prepend Degenerate Vertex
   tma = float(degToRad(begin));
   verts[vertIndex++] = (float)(bx+cos(tma)*(rx+rs));   // X
   verts[vertIndex++] = (float)(by+sin(tma)*(ry+rs));   // Y

   for (unsigned int i = 0; i < circleSegments+1; i ++ ) {
      tma = float(degToRad(begin+i*degSegment));
      verts[vertIndex++] = (float)(bx+cos(tma)*(rx+rs));   // X
      verts[vertIndex++] = (float)(by+sin(tma)*(ry+rs));   // Y
      verts[vertIndex++] = (float)(bx+cos(tma)*rx);  // X
      verts[vertIndex++] = (float)(by+sin(tma)*ry);  // Y
   }

   tma = float(degToRad(begin+(circleSegments)*degSegment));
   verts[vertIndex++] = (float)(bx+cos(tma)*rx);  // X
   verts[vertIndex++] = (float)(by+sin(tma)*ry);  // Y

   return vertIndex/2;
}

// Update vertices to allocated arrays
unsigned int updateArchColor(
      unsigned int   circleSegments,   // Number of sides 
      float*         color,            // Polygon Color 
      unsigned int   index,            // Index of where to start writing to input arrays 
      float*         colrs             // Input Vector of r,g,b values 
      ){
   unsigned int colrIndex = index*4;
   float R, G, B, A;
   R = color[0];
   G = color[1];
   B = color[2];
   A = color[3];

   // Prepend Degenerate Vertex
   colrs[colrIndex++] = R;   
   colrs[colrIndex++] = G;   
   colrs[colrIndex++] = B;
   colrs[colrIndex++] = A;

   for (unsigned int i = 0; i < circleSegments+1; i ++ ) {
      colrs[colrIndex++] = R;   
      colrs[colrIndex++] = G;   
      colrs[colrIndex++] = B;
      colrs[colrIndex++] = A;

      colrs[colrIndex++] = R;   
      colrs[colrIndex++] = G;   
      colrs[colrIndex++] = B;
      colrs[colrIndex++] = A;
   }
   colrs[colrIndex++] = R;   
   colrs[colrIndex++] = G;   
   colrs[colrIndex++] = B;
   colrs[colrIndex++] = A;

   return colrIndex/4;
}
