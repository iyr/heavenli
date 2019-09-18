/*
 *  Defines a Quad for TRIANGLE_STRIP with degenerate vertices
 */

#include <math.h>
#include <vector>
using namespace std;

// Append vertices to input vectors
// Draw Quad via four pairs of X-Y coordinates that define the corners of a rectangle
int defineQuad4pt(
      float px,                  /* X-Coordinate of first point      */
      float py,                  /* Y-Coordinate of first point      */
      float qx,                  /* X-Coordinate of second point     */
      float qy,                  /* Y-Coordinate of second point     */
      float rx,                  /* X-Coordinate of third point      */
      float ry,                  /* Y-Coordinate of third point      */
      float sx,                  /* X-Coordinate of fourth point     */
      float sy,                  /* Y-Coordinate of fourth point     */
      float *color,              /* Polygon Color                    */
      std::vector<float> &verts, /* Input Vector of x,y coordinates  */
      std::vector<float> &colrs  /* Input Vector of r,g,b values     */
      ){
   float R, G, B, A;
   R = color[0];
   G = color[1];
   B = color[2];
   A = color[3];

   // Prepend Degenerate Vertex
   if (verts.size() == 0) {
      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);
   } else {
      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);

      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);
   }
   
   /* X */ verts.push_back(px);
   /* Y */ verts.push_back(py);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   /* X */ verts.push_back(qx);
   /* Y */ verts.push_back(qy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   /* X */ verts.push_back(rx);
   /* Y */ verts.push_back(ry);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   /* X */ verts.push_back(sx);
   /* Y */ verts.push_back(sy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   // Append Degenerate Vertex
   /* X */ verts.push_back(sx);
   /* Y */ verts.push_back(sy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);
   return verts.size()/2;
}
// Draw Quad via Two pairs of X-Y coordinates that define the corners of a rectangle
int defineQuad2pt(
      float px,                  /* X-Coordinate of first corner                          */
      float py,                  /* Y-Coordinate of first corner                          */
      float qx,                  /* X-Coordinate of second corner                         */
      float qy,                  /* Y-Coordinate of second corner                         */
      float *color,              /* Polygon Color                                         */
      std::vector<float> &verts, /* Input Vector of x,y coordinates                       */
      std::vector<float> &colrs  /* Input Vector of r,g,b values                          */
      ){
   float R, G, B, A;
   R = color[0];
   G = color[1];
   B = color[2];
   A = color[3];

   // Prepend Degenerate Vertex
   if (verts.size() == 0) {
      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);
   } else {
      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);

      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);
   }
   
   /* X */ verts.push_back(px);
   /* Y */ verts.push_back(py);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   /* X */ verts.push_back(px);
   /* Y */ verts.push_back(qy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   /* X */ verts.push_back(qx);
   /* Y */ verts.push_back(py);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   /* X */ verts.push_back(qx);
   /* Y */ verts.push_back(qy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   // Append Degenerate Vertex
   /* X */ verts.push_back(qx);
   /* Y */ verts.push_back(qy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);
   return verts.size()/2;
}

// Append vertices to input vectors
// Draw Quad with its center at px, py, with radius rx, ry
int defineQuadRad(
      float gx,                  /* X-Coordinate of center of quad   */
      float gy,                  /* Y-Coordinate of center of quad   */
      float rx,                  /* X-radius of rectangle            */
      float ry,                  /* Y-radius of rectangle            */
      float *color,              /* Polygon Color                    */
      std::vector<float> &verts, /* Input Vector of x,y coordinates  */
      std::vector<float> &colrs  /* Input Vector of r,g,b values     */
      ){
   float R, G, B, A, px, py, qx, qy;
   R  = color[0];
   G  = color[1];
   B  = color[2];
   A  = color[3];

   px = gx-rx;
   py = gy-ry;
   qx = gx+rx;
   qy = gy+ry;

   // Prepend Degenerate Vertex
   if (verts.size() == 0) {
      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);
   } else {
      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);

      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);
   }
   
   /* X */ verts.push_back(px);
   /* Y */ verts.push_back(py);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   /* X */ verts.push_back(px);
   /* Y */ verts.push_back(qy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   /* X */ verts.push_back(qx);
   /* Y */ verts.push_back(py);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   /* X */ verts.push_back(qx);
   /* Y */ verts.push_back(qy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   // Append Degenerate Vertex
   /* X */ verts.push_back(qx);
   /* Y */ verts.push_back(qy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);
   return verts.size()/2;
}

// Update preallocated arrays starting at index
// Draw Quad via Two pairs of X-Y coordinates that define the corners of a rectangle
int updateQuad2ptGeometry(
      float px,      /* X-Coordinate of first corner                    */
      float py,      /* Y-Coordinate of first corner                    */
      float qx,      /* X-Coordinate of second corner                   */
      float qy,      /* Y-Coordinate of second corner                   */
      int   index,   /* Index of where to start writing in input array  */
      float *verts   /* Input Vector of x,y coordinates                 */
      ){
   int vertIndex = index*2;   /* index (X, Y)   */

   // Prepend Degenerate Vertex
   if (vertIndex == 0) {
      /* X */ verts[vertIndex++] = px;
      /* Y */ verts[vertIndex++] = py;
   } else {
      /* X */ verts[vertIndex++] = px;
      /* Y */ verts[vertIndex++] = py;
      /* X */ verts[vertIndex++] = px;
      /* Y */ verts[vertIndex++] = py;
   }
   
   /* X */ verts[vertIndex++] = px;
   /* Y */ verts[vertIndex++] = py;

   /* X */ verts[vertIndex++] = px;
   /* Y */ verts[vertIndex++] = qy;

   /* X */ verts[vertIndex++] = qx;
   /* Y */ verts[vertIndex++] = py;

   /* X */ verts[vertIndex++] = qx;
   /* Y */ verts[vertIndex++] = qy;

   // Append Degenerate Vertex
   /* X */ verts[vertIndex++] = qx;
   /* Y */ verts[vertIndex++] = qy;
   return vertIndex/2;
}

// Update preallocated arrays starting at index
// Draw Quad with its center at px, py, with radius rx, ry
int updateQuadRadGeometry(
      float gx,      /* X-Coordinate of center of quad   */
      float gy,      /* Y-Coordinate of center of quad   */
      float rx,      /* X-radius of rectangle            */
      float ry,      /* Y-radius of rectangle            */
      int   index,   /* Index of where to start writing in input array  */
      float *verts   /* Input Vector of x,y coordinates                 */
      ){
   int vertIndex = index*2;   /* index (X, Y)   */
   float px, py, qx, qy;
   px = gx-rx;
   py = gy-ry;
   qx = gx+rx;
   qy = gy+ry;

   // Prepend Degenerate Vertex
   if (vertIndex == 0) {
      /* X */ verts[vertIndex++] = px;
      /* Y */ verts[vertIndex++] = py;
   } else {
      /* X */ verts[vertIndex++] = px;
      /* Y */ verts[vertIndex++] = py;
      /* X */ verts[vertIndex++] = px;
      /* Y */ verts[vertIndex++] = py;
   }
   
   /* X */ verts[vertIndex++] = px;
   /* Y */ verts[vertIndex++] = py;

   /* X */ verts[vertIndex++] = px;
   /* Y */ verts[vertIndex++] = qy;

   /* X */ verts[vertIndex++] = qx;
   /* Y */ verts[vertIndex++] = py;

   /* X */ verts[vertIndex++] = qx;
   /* Y */ verts[vertIndex++] = qy;

   // Append Degenerate Vertex
   /* X */ verts[vertIndex++] = qx;
   /* Y */ verts[vertIndex++] = qy;
   return vertIndex/2;
}

// Append vertices to input vectors
// Draw Quad with its center at px, py, with radius rx, ry
int updateQuadColor(
      float *color,  /* Polygon Color                                   */
      int   index,   /* Index of where to start writing in input array  */
      float *colrs   /* Input Vector of r,g,b values                    */
      ){
   int colrIndex = index*4;   /* index (r, g, b) */
   float R, G, B, A;
   R  = color[0];
   G  = color[1];
   B  = color[2];
   A  = color[3];

   // Prepend Degenerate Vertex
   if (colrIndex == 0) {
      /* R */ colrs[colrIndex++] = R;
      /* G */ colrs[colrIndex++] = G;
      /* B */ colrs[colrIndex++] = B;
      /* A */ colrs[colrIndex++] = A;
   } else {
      /* R */ colrs[colrIndex++] = R;
      /* G */ colrs[colrIndex++] = G;
      /* B */ colrs[colrIndex++] = B;
      /* A */ colrs[colrIndex++] = A;

      /* R */ colrs[colrIndex++] = R;
      /* G */ colrs[colrIndex++] = G;
      /* B */ colrs[colrIndex++] = B;
      /* A */ colrs[colrIndex++] = A;
   }
   
   /* R */ colrs[colrIndex++] = R;
   /* G */ colrs[colrIndex++] = G;
   /* B */ colrs[colrIndex++] = B;
   /* A */ colrs[colrIndex++] = A;

   /* R */ colrs[colrIndex++] = R;
   /* G */ colrs[colrIndex++] = G;
   /* B */ colrs[colrIndex++] = B;
   /* A */ colrs[colrIndex++] = A;

   /* R */ colrs[colrIndex++] = R;
   /* G */ colrs[colrIndex++] = G;
   /* B */ colrs[colrIndex++] = B;
   /* A */ colrs[colrIndex++] = A;

   /* R */ colrs[colrIndex++] = R;
   /* G */ colrs[colrIndex++] = G;
   /* B */ colrs[colrIndex++] = B;
   /* A */ colrs[colrIndex++] = A;

   // Append Degenerate Vertex
   /* R */ colrs[colrIndex++] = R;
   /* G */ colrs[colrIndex++] = G;
   /* B */ colrs[colrIndex++] = B;
   /* A */ colrs[colrIndex++] = A;
   return colrIndex/4;
}
