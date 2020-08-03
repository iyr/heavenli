/*
 *  Defines a Textured Quad for TRIANGLE_STRIP with degenerate vertices
 */

using namespace std;

unsigned int defineTexQuad(
      float gx,                  // X-Coordinate of center of quad
      float gy,                  // Y-Coordinate of center of quad
      float rx,                  // X-radius of rectangle
      float ry,                  // Y-radius of rectangle
      float *color,              // Polygon Color
      std::vector<float> &verts, // Input Vector of x,y coordinates
      std::vector<float> &texuv, // Input Vector of u,v texture coordinates
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

   // Bottom left:   px py
   // Bottom right:  qx py
   // Top left:      px qy
   // Top right:     qx qy

   // Prepend Degenerate Vertex
   verts.push_back(px);
   verts.push_back(py);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);
   texuv.push_back(0.0f);
   texuv.push_back(0.0f);
   
   // Defines rectangle
   verts.push_back(px);
   verts.push_back(py);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);
   texuv.push_back(0.0f);
   texuv.push_back(0.0f);

   verts.push_back(px);
   verts.push_back(qy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);
   texuv.push_back(0.0f);
   texuv.push_back(1.0f);

   verts.push_back(qx);
   verts.push_back(py);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);
   texuv.push_back(1.0f);
   texuv.push_back(0.0f);

   verts.push_back(qx);
   verts.push_back(qy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);
   texuv.push_back(1.0f);
   texuv.push_back(1.0f);

   // Append Degenerate Vertex
   verts.push_back(qx);
   verts.push_back(qy);
   colrs.push_back(R);
   colrs.push_back(G);
   colrs.push_back(B);
   colrs.push_back(A);
   texuv.push_back(1.0f);
   texuv.push_back(1.0f);

   return verts.size()/2;
}
