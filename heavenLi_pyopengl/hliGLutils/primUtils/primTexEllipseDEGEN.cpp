/*
 *  Defines an efficient circle for TRIANGLE_STRIP with degenerate vertices
 */

#include <math.h>
#include <vector>
using namespace std;

// Append to input vectors
unsigned int defineTexEllipse(
      float bx,                     // X-Coordinate
      float by,                     // Y-Coordinate
      float bsx,                    // x-Scale 2.0=spans display before  scaling
      float bsy,                    // y-Scale 2.0=spans display before  scaling
      unsigned int circleSegments,  // Number of sides
      float *color,                 // Polygon Color
      std::vector<float> &verts,    // Input Vector of x,y coordinates
      std::vector<float> &texuv,    // Input Vector of u,v texture coordinates
      std::vector<float> &colrs     // Input Vector of r,g,b values
      ){
   float R, G, B, A, tma, tmx, tmy;
   float degSegment = 360.0f / float(circleSegments);
   degSegment /= 2.0f;
   R = color[0];
   G = color[1];
   B = color[2];
   A = color[3];

   // Prepend degenerate vertex
   verts.push_back(float(bx + bsx*1.0f)); // X
   verts.push_back(float(by + bsy*0.0f)); // Y
   colrs.push_back(R);                    // R
   colrs.push_back(G);                    // G
   colrs.push_back(B);                    // B
   colrs.push_back(A);                    // A
   texuv.push_back(1.0f);                 // U
   texuv.push_back(0.0f);                 // V

   for (unsigned int i = 0; i < circleSegments+1; i++ ) {
      tma = (float)degToRad((float)i*degSegment);
      tmx = cos(tma);
      tmy = sin(tma);

      verts.push_back(float(bx + bsx*tmx));        // X
      verts.push_back(float(by + bsy*tmy));        // Y
      colrs.push_back(R);                          // R
      colrs.push_back(G);                          // G
      colrs.push_back(B);                          // B
      colrs.push_back(A);                          // A
      texuv.push_back(float((1.0f + tmx)*0.5f));   // U
      texuv.push_back(float((1.0f + -tmy)*0.5f));  // V

      verts.push_back(float(bx + bsx*tmx));     // X
      verts.push_back(float(by - bsy*tmy));     // Y
      colrs.push_back(R);                       // R
      colrs.push_back(G);                       // G
      colrs.push_back(B);                       // B
      colrs.push_back(A);                       // A
      texuv.push_back(float((1.0f + tmx)*0.5f));// U
      texuv.push_back(float((1.0f + tmy)*0.5f));// V
   }

   // Append degenerate vertex
   tma = (float)degToRad((float)circleSegments*degSegment);
   tmx = cos(tma);
   tmy = sin(tma);
   verts.push_back(float(bx + bsx*tmx));  // X
   verts.push_back(float(by + bsy*tmy));  // Y
   colrs.push_back(R);                    // R
   colrs.push_back(G);                    // G
   colrs.push_back(B);                    // B
   colrs.push_back(A);                    // A
   texuv.push_back(1.0f);                 // U
   texuv.push_back(0.0f);                 // V

   return verts.size()/2;
}

// Useful overload
unsigned int defineTexEllipse(
      float bx,                     // X-Coordinate
      float by,                     // Y-Coordinate
      float bs,                     // Scale 2.0=spans display before MVP scaling
      unsigned int circleSegments,  // Number of sides
      float *color,                 // Polygon Color
      std::vector<float> &verts,    // Input Vector of x,y coordinates
      std::vector<float> &texuv,    // Input Vector of u,v texture coordina
      std::vector<float> &colrs     // Input Vector of r,g,b values
      ){
   return defineEllipse(bx, by, bs, bs, circleSegments, color, verts, colrs);
}

// Useful pseudo-overload
unsigned int defineTexCircle(
      float bx,                     // X-Coordinate
      float by,                     // Y-Coordinate
      float bs,                     // Scale 2.0=spans display before MVP scaling
      unsigned int circleSegments,  // Number of sides
      float *color,                 // Polygon Color
      std::vector<float> &verts,    // Input Vector of x,y coordinates
      std::vector<float> &colrs     // Input Vector of r,g,b values
      ){
   return defineEllipse(bx, by, bs, bs, circleSegments, color, verts, colrs);
}

/*
// Write to pre-allocated input array, updating vertices only 
unsigned int updateEllipseGeometry(
      float bx,                     // X-Coordinate
      float by,                     // Y-Coordinate
      float bsx,                    // x-Scale 2.0=spans display before GL scaling
      float bsy,                    // y-Scale 2.0=spans display before GL scaling
      unsigned int circleSegments,  // Number of sides
      unsigned int index,           // Index of where to start writing to input arrays
      float *verts                  // Input Vector of x,y values
      ){
   unsigned int vertIndex = index*2;   // index (x, y)
   float tma, degSegment = 360.0f / float(circleSegments);
   degSegment /= 2.0f;

   // Prepend degenerate vertex
   verts[vertIndex++] = (float(bx + bsx*1.0f)); // X
   verts[vertIndex++] = (float(by + bsy*0.0f)); // Y

   for (unsigned int i = 0; i < circleSegments+1; i++ ) {
      tma = (float)degToRad((float)i*degSegment);

      verts[vertIndex++] = (float(bx + bsx*cos(tma)));   // X
      verts[vertIndex++] = (float(by + bsy*sin(tma)));   // Y

      verts[vertIndex++] = (float(bx + bsx*cos(tma)));   // X
      verts[vertIndex++] = (float(by - bsy*sin(tma)));   // Y
   }

   // Append degenerate vertex
   tma = (float)degToRad((float)circleSegments*degSegment);
   verts[vertIndex++] = (float(bx + bsx*cos(tma)));   // X
   verts[vertIndex++] = (float(by + bsy*sin(tma)));   // Y

   return vertIndex/2;
}

// Write to pre-allocated input array, updating color only 
unsigned int updateEllipseColor(
      unsigned int circleSegments,  // Number of sides
      float *color,                 // Polygon Color
      unsigned int   index,         // Index of where to start writing to input arrays
      float *colrs                  // Input Vector of r,g,b values
      ){
   unsigned int colrIndex = index*4;   // index (r, g, b, a)
   float R, G, B, A;
   R = color[0];
   G = color[1];
   B = color[2];
   A = color[3];

   // Prepend degenerate vertex
   colrs[colrIndex++] = R; // R
   colrs[colrIndex++] = G; // G
   colrs[colrIndex++] = B; // B
   colrs[colrIndex++] = A; // A

   for (unsigned int i = 0; i < circleSegments+1; i++ ) {
      colrs[colrIndex++] = R; // R
      colrs[colrIndex++] = G; // G
      colrs[colrIndex++] = B; // B
      colrs[colrIndex++] = A; // A
      
      colrs[colrIndex++] = R; // R
      colrs[colrIndex++] = G; // G
      colrs[colrIndex++] = B; // B
      colrs[colrIndex++] = A; // A
   }

   // Append degenerate vertex
   colrs[colrIndex++] = R; // R
   colrs[colrIndex++] = G; // G
   colrs[colrIndex++] = B; // B
   colrs[colrIndex++] = A; // A
   
   return colrIndex/4;
}

unsigned int updateCircleColor(
      unsigned int   circleSegments,
      float*         color,
      unsigned int   index,
      float*         colrs
      ){
   return updateEllipseColor(circleSegments, color, index, colrs);
}
*/
