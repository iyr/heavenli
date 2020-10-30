/*
 *  Defines a Quad for TRIANGLE_STRIP with degenerate vertices
 */

//#include <math.h>
//#include <vector>
using namespace std;

// Append vertices to input vectors
// Draw Quad via four pairs of X-Y coordinates that define the corners of a rectangle
unsigned int defineQuad4pt(
      float px,                  // X-Coordinate of first point
      float py,                  // Y-Coordinate of first point
      float qx,                  // X-Coordinate of second point
      float qy,                  // Y-Coordinate of second point
      float rx,                  // X-Coordinate of third point
      float ry,                  // Y-Coordinate of third point
      float sx,                  // X-Coordinate of fourth point
      float sy,                  // Y-Coordinate of fourth point
      float *color,              // Polygon Color
      std::vector<float> &verts, // Input Vector of x,y coordinates
      std::vector<float> &colrs  // Input Vector of r,g,b values
      ){
   float R, G, B, A;
   R = color[0];
   G = color[1];
   B = color[2];
   A = color[3];

   // Prepend Degenerate Vertex
   verts.push_back(px);
   verts.push_back(py);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);
   
   // Draw Rectangle
   verts.push_back(px);
   verts.push_back(py);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   verts.push_back(qx);
   verts.push_back(qy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   verts.push_back(rx);
   verts.push_back(ry);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   verts.push_back(sx);
   verts.push_back(sy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   // Append Degenerate Vertex
   verts.push_back(sx);
   verts.push_back(sy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);
   return verts.size()/2;
}
// Draw Quad via Two pairs of X-Y coordinates that define the corners of a rectangle
unsigned int defineQuad2pt(
      float px,                  // X-Coordinate of first corner
      float py,                  // Y-Coordinate of first corner
      float qx,                  // X-Coordinate of second corner
      float qy,                  // Y-Coordinate of second corner
      float *color,              // Polygon Color
      std::vector<float> &verts, // Input Vector of x,y coordinates
      std::vector<float> &colrs  // Input Vector of r,g,b values
      ){
   float R, G, B, A;
   R = color[0];
   G = color[1];
   B = color[2];
   A = color[3];

   // Prepend Degenerate Vertex
   verts.push_back(px);
   verts.push_back(py);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);
   
   // Draw Rectangle
   verts.push_back(px);
   verts.push_back(py);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   verts.push_back(qx);
   verts.push_back(py);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   verts.push_back(px);
   verts.push_back(qy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   verts.push_back(qx);
   verts.push_back(qy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   // Append Degenerate Vertex
   verts.push_back(qx);
   verts.push_back(qy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   return verts.size()/2;
}

// Append vertices to input vectors
// Draw Quad with its center at px, py, with radius rx, ry
unsigned int defineQuadRad(
      float gx,                  // X-Coordinate of center of quad
      float gy,                  // Y-Coordinate of center of quad
      float rx,                  // X-radius of rectangle
      float ry,                  // Y-radius of rectangle
      float *color,              // Polygon Color
      std::vector<float> &verts, // Input Vector of x,y coordinates
      std::vector<float> &colrs  // Input Vector of r,g,b values
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
   verts.push_back(px);
   verts.push_back(py);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);
   
   // Defines rectangle
   verts.push_back(px);
   verts.push_back(py);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   verts.push_back(px);
   verts.push_back(qy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   verts.push_back(qx);
   verts.push_back(py);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   verts.push_back(qx);
   verts.push_back(qy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);

   // Append Degenerate Vertex
   verts.push_back(qx);
   verts.push_back(qy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);
   return verts.size()/2;
}

unsigned int updateQuad4ptGeometry(
      float px,      // X-Coordinate of first point
      float py,      // Y-Coordinate of first point
      float qx,      // X-Coordinate of second point
      float qy,      // Y-Coordinate of second point
      float rx,      // X-Coordinate of third point
      float ry,      // Y-Coordinate of third point
      float sx,      // X-Coordinate of fourth point
      float sy,      // Y-Coordinate of fourth point
      int   index,   // Index of where to start writing in input array
      float *verts   // Input Vector of x,y coordinates
      ){
   unsigned int vertIndex = index*2;   // index (X, Y)

   // Prepend Degenerate Vertex
   verts[vertIndex++] = px;
   verts[vertIndex++] = py;
   
   // Define Rectangle
   verts[vertIndex++] = px;
   verts[vertIndex++] = py;

   verts[vertIndex++] = qx;
   verts[vertIndex++] = qy;

   verts[vertIndex++] = rx;
   verts[vertIndex++] = ry;

   verts[vertIndex++] = sx;
   verts[vertIndex++] = sy;

   // Append Degenerate Vertex
   verts[vertIndex++] = sx;
   verts[vertIndex++] = sy;

   return vertIndex/2;
}

// Update preallocated arrays starting at index
// Draw Quad via Two pairs of X-Y coordinates that define the corners of a rectangle
unsigned int updateQuad2ptGeometry(
      float px,      // X-Coordinate of first corner
      float py,      // Y-Coordinate of first corner
      float qx,      // X-Coordinate of second corner
      float qy,      // Y-Coordinate of second corner
      int   index,   // Index of where to start writing in input array
      float *verts   // Input Vector of x,y coordinates
      ){
   unsigned int vertIndex = index*2;   // index (X, Y)

   // Prepend Degenerate Vertex
   verts[vertIndex++] = px;
   verts[vertIndex++] = py;
   
   // Define Rectangle
   verts[vertIndex++] = px;
   verts[vertIndex++] = py;

   verts[vertIndex++] = qx;
   verts[vertIndex++] = py;

   verts[vertIndex++] = px;
   verts[vertIndex++] = qy;

   verts[vertIndex++] = qx;
   verts[vertIndex++] = qy;

   // Append Degenerate Vertex
   verts[vertIndex++] = qx;
   verts[vertIndex++] = qy;

   return vertIndex/2;
}

// Update preallocated arrays starting at index
// Draw Quad with its center at px, py, with radius rx, ry
unsigned int updateQuadRadGeometry(
      float gx,      // X-Coordinate of center of quad
      float gy,      // Y-Coordinate of center of quad
      float rx,      // X-radius of rectangle
      float ry,      // Y-radius of rectangle
      int   index,   // Index of where to start writing in input array
      float *verts   // Input Vector of x,y coordinates
      ){
   unsigned int vertIndex = index*2;   // index (X, Y)
   float px, py, qx, qy;
   px = gx-rx;
   py = gy-ry;
   qx = gx+rx;
   qy = gy+ry;

   // Prepend Degenerate Vertex
   verts[vertIndex++] = px;
   verts[vertIndex++] = py;
   
   // Define quad
   verts[vertIndex++] = px;
   verts[vertIndex++] = py;

   verts[vertIndex++] = px;
   verts[vertIndex++] = qy;

   verts[vertIndex++] = qx;
   verts[vertIndex++] = py;

   verts[vertIndex++] = qx;
   verts[vertIndex++] = qy;

   // Append Degenerate Vertex
   verts[vertIndex++] = qx;
   verts[vertIndex++] = qy;
   return vertIndex/2;
}

// Append vertices to input vectors
// Draw Quad with its center at px, py, with radius rx, ry
unsigned int updateQuadColor(
      float *color,  // Polygon Color
      int   index,   // Index of where to start writing in input array
      float *colrs   // Input Vector of r,g,b values
      ){
   unsigned int colrIndex = index*4;   // index (r, g, b)
   float R, G, B, A;
   R  = color[0];
   G  = color[1];
   B  = color[2];
   A  = color[3];

   // Prepend Degenerate Vertex
   colrs[colrIndex++] = R;
   colrs[colrIndex++] = G;
   colrs[colrIndex++] = B;
   colrs[colrIndex++] = A;
   
   // Update quad
   colrs[colrIndex++] = R;
   colrs[colrIndex++] = G;
   colrs[colrIndex++] = B;
   colrs[colrIndex++] = A;

   colrs[colrIndex++] = R;
   colrs[colrIndex++] = G;
   colrs[colrIndex++] = B;
   colrs[colrIndex++] = A;

   colrs[colrIndex++] = R;
   colrs[colrIndex++] = G;
   colrs[colrIndex++] = B;
   colrs[colrIndex++] = A;

   colrs[colrIndex++] = R;
   colrs[colrIndex++] = G;
   colrs[colrIndex++] = B;
   colrs[colrIndex++] = A;

   // Append Degenerate Vertex
   colrs[colrIndex++] = R;
   colrs[colrIndex++] = G;
   colrs[colrIndex++] = B;
   colrs[colrIndex++] = A;
   return colrIndex/4;
}
