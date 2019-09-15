/*
 *  Defines a Quad for TRIANGLE_STRIP with degenerate vertices
 */

#include <math.h>
#include <vector>
using namespace std;

// Append vertices to input vectors
int defineQuad(
      float px,                  /* X-Coordinate of first corner                          */
      float py,                  /* Y-Coordinate of first corner                          */
      float qx,                  /* X-Coordinate of second corner                         */
      float qy,                  /* Y-Coordinate of second corner                         */
      float *color,              /* Polygon Color                                         */
      std::vector<float> &verts, /* Input Vector of x,y coordinates                       */
      std::vector<float> &colrs  /* Input Vector of r,g,b values                          */
      ){
   float R, G, B;
   R = color[0];
   G = color[1];
   B = color[2];

   // Prepend Degenerate Vertex
   if (verts.size() == 0) {
      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
   } else {
      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
      /* X */ verts.push_back(px);
      /* Y */ verts.push_back(py);
      /* R */ colrs.push_back(R);
      /* G */ colrs.push_back(G);
      /* B */ colrs.push_back(B);
   }
   
   /* X */ verts.push_back(px);
   /* Y */ verts.push_back(py);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);

   /* X */ verts.push_back(px);
   /* Y */ verts.push_back(qy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);

   /* X */ verts.push_back(qx);
   /* Y */ verts.push_back(py);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);

   /* X */ verts.push_back(qx);
   /* Y */ verts.push_back(qy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);

   // Append Degenerate Vertex
   /* X */ verts.push_back(qx);
   /* Y */ verts.push_back(qy);
   /* R */ colrs.push_back(R);
   /* G */ colrs.push_back(G);
   /* B */ colrs.push_back(B);
   return verts.size()/2;
}
