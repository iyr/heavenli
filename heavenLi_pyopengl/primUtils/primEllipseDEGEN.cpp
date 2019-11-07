/*
 *  Defines an efficient circle for TRIANGLE_STRIP with degenerate vertices
 */

#include <math.h>
#include <vector>
using namespace std;

// Append to input vectors
unsigned int defineEllipse(
      float bx,                     /* X-Coordinate */
      float by,                     /* Y-Coordinate */
      float bsx,                    /* x-Scale 2.0=spans display before  scaling */
      float bsy,                    /* y-Scale 2.0=spans display before  scaling */
      unsigned int circleSegments,  /* Number of sides */
      float *color,                 /* Polygon Color */
      std::vector<float> &verts,    /* Input Vector of x,y coordinates */
      std::vector<float> &colrs     /* Input Vector of r,g,b values */
      ){
   float R, G, B, A, tma;
   float degSegment = 360.0f / float(circleSegments);
   degSegment /= 2.0f;
   R = color[0];
   G = color[1];
   B = color[2];
   A = color[3];

   // Prepend degenerate vertex iff not the first primitive in the vector
   if (verts.size() == 0) {
      verts.push_back(float(bx + bsx*1.0f)); // X
      verts.push_back(float(by + bsy*0.0f)); // Y
      colrs.push_back(R);                    // R
      colrs.push_back(G);                    // G
      colrs.push_back(B);                    // B
      colrs.push_back(A);                    // A
   } else {
      verts.push_back(float(bx + bsx*1.0f)); // X
      verts.push_back(float(by + bsy*0.0f)); // Y
      colrs.push_back(R);                    // R
      colrs.push_back(G);                    // G
      colrs.push_back(B);                    // B
      colrs.push_back(A);                    // A

      verts.push_back(float(bx + bsx*1.0f)); // X
      verts.push_back(float(by + bsy*0.0f)); // Y
      colrs.push_back(R);                    // R
      colrs.push_back(G);                    // G
      colrs.push_back(B);                    // B
      colrs.push_back(A);                    // A
   }

   for (unsigned int i = 0; i < circleSegments; i++ ) {
      tma = (float)degToRad((float)i*degSegment);

      verts.push_back(float(bx + bsx*cos(tma)));   // X
      verts.push_back(float(by + bsy*sin(tma)));   // Y
      colrs.push_back(R);                          // R
      colrs.push_back(G);                          // G
      colrs.push_back(B);                          // B
      colrs.push_back(A);                          // A

      verts.push_back(float(bx + bsx*cos(tma)));   // X
      verts.push_back(float(by - bsy*sin(tma)));   // Y
      colrs.push_back(R);                          // R
      colrs.push_back(G);                          // G
      colrs.push_back(B);                          // B
      colrs.push_back(A);                          // A
   }
   verts.push_back(float(bx - bsx*1.0f)); // X
   verts.push_back(float(by + bsy*0.0f)); // Y
   colrs.push_back(R);                    // R
   colrs.push_back(G);                    // G
   colrs.push_back(B);                    // B
   colrs.push_back(A);                    // A

   verts.push_back(float(bx - bsx*1.0f)); // X
   verts.push_back(float(by + bsy*0.0f)); // Y
   colrs.push_back(R);                    // R
   colrs.push_back(G);                    // G
   colrs.push_back(B);                    // B
   colrs.push_back(A);                    // A

   //printf("verts size before return: %d\n", verts.size());
   return verts.size()/2;
}

// Useful overload
unsigned int defineEllipse(
      float bx,                     /* X-Coordinate */
      float by,                     /* Y-Coordinate */
      float bs,                     /* Scale 2.0=spans display before MVP scaling */
      unsigned int circleSegments,  /* Number of sides */
      float *color,                 /* Polygon Color */
      std::vector<float> &verts,    /* Input Vector of x,y coordinates */
      std::vector<float> &colrs     /* Input Vector of r,g,b values */
      ){
   return defineEllipse(bx, by, bs, bs, circleSegments, color, verts, colrs);
}

// Useful pseudo-overload
unsigned int defineCircle(
      float bx,                     /* X-Coordinate */
      float by,                     /* Y-Coordinate */
      float bs,                     /* Scale 2.0=spans display before MVP scaling */
      unsigned int circleSegments,  /* Number of sides */
      float *color,                 /* Polygon Color */
      std::vector<float> &verts,    /* Input Vector of x,y coordinates */
      std::vector<float> &colrs     /* Input Vector of r,g,b values */
      ){
   return defineEllipse(bx, by, bs, bs, circleSegments, color, verts, colrs);
}
 
// Write to pre-allocated input array, updating vertices only 
unsigned int updateEllipseGeometry(
      float bx,                     /* X-Coordinate */
      float by,                     /* Y-Coordinate */
      float bsx,                    /* x-Scale 2.0=spans display before GL scaling */
      float bsy,                    /* y-Scale 2.0=spans display before GL scaling */
      unsigned int circleSegments,  /* Number of sides */
      unsigned int index,           /* Index of where to start writing to input arrays */
      float *verts                  /* Input Vector of x,y values */
      ){
   unsigned int vertIndex = index*2;   /* index (x, y) */
   float tma, degSegment = 360.0f / float(circleSegments);
   degSegment /= 2.0f;

   if (vertIndex == 0) {
      verts[vertIndex++] = (float(bx + bsx*1.0f)); // X
      verts[vertIndex++] = (float(by + bsy*0.0f)); // Y
   } else {
      verts[vertIndex++] = (float(bx + bsx*1.0f)); // X
      verts[vertIndex++] = (float(by + bsy*0.0f)); // Y

      verts[vertIndex++] = (float(bx + bsx*1.0f)); // X
      verts[vertIndex++] = (float(by + bsy*0.0f)); // Y
   }

   for (unsigned int i = 0; i < circleSegments; i++ ) {
      tma = (float)degToRad((float)i*degSegment);

      verts[vertIndex++] = (float(bx + bsx*cos(tma)));   // X
      verts[vertIndex++] = (float(by + bsy*sin(tma)));   // Y

      verts[vertIndex++] = (float(bx + bsx*cos(tma)));   // X
      verts[vertIndex++] = (float(by - bsy*sin(tma)));   // Y
   }
   verts[vertIndex++] = (float(bx - bsx*1.0f)); // X
   verts[vertIndex++] = (float(by + bsy*0.0f)); // Y

   verts[vertIndex++] = (float(bx - bsx*1.0f)); // X
   verts[vertIndex++] = (float(by + bsy*0.0f)); // Y

   return vertIndex/2;
}

// Write to pre-allocated input array, updating color only 
unsigned int updateEllipseColor(
      unsigned int circleSegments,  /* Number of sides */
      float *color,                 /* Polygon Color */
      unsigned int   index,         /* Index of where to start writing to input arrays */
      float *colrs                  /* Input Vector of r,g,b values */
      ){
   unsigned int colrIndex = index*4;   /* index (r, g, b, a) */
   float R, G, B, A;
   R = color[0];
   G = color[1];
   B = color[2];
   A = color[3];

   if (colrIndex == 0) {
      colrs[colrIndex++] = R; // R
      colrs[colrIndex++] = G; // G
      colrs[colrIndex++] = B; // B
      colrs[colrIndex++] = A; // A
   } else {
      colrs[colrIndex++] = R; // R
      colrs[colrIndex++] = G; // G
      colrs[colrIndex++] = B; // B
      colrs[colrIndex++] = A; // A
      
      colrs[colrIndex++] = R; // R
      colrs[colrIndex++] = G; // G
      colrs[colrIndex++] = B; // B
      colrs[colrIndex++] = A; // A
   }

   for (unsigned int i = 0; i < circleSegments; i++ ) {
      colrs[colrIndex++] = R; // R
      colrs[colrIndex++] = G; // G
      colrs[colrIndex++] = B; // B
      colrs[colrIndex++] = A; // A
      
      colrs[colrIndex++] = R; // R
      colrs[colrIndex++] = G; // G
      colrs[colrIndex++] = B; // B
      colrs[colrIndex++] = A; // A
   }
   colrs[colrIndex++] = R; // R
   colrs[colrIndex++] = G; // G
   colrs[colrIndex++] = B; // B
   colrs[colrIndex++] = A; // A
   
   colrs[colrIndex++] = R; // R
   colrs[colrIndex++] = G; // G
   colrs[colrIndex++] = B; // B
   colrs[colrIndex++] = A; // A

   return colrIndex/4;
}
