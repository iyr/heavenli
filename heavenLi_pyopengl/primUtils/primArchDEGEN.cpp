/*
 *  Defines a circular Arch for TRIANGLE_STRIP with degenerate vertices
 */

#include <math.h>
#include <vector>
using namespace std;

// Append vertices to input vectors
int defineArch(
      float bx,                  /* X-Coordinate */
      float by,                  /* Y-Coordinate */
      float bsx,                 /* x-Scale 2.0=spans display before GL scaling */
      float bsy,                 /* y-Scale 2.0=spans display before GL scaling */
      float start,               /* Where, in degrees on the unit circle, the arch begins */
      float end,                 /* Where, in degrees on the unit circle, the arch endgs */
      float rs,                  /* Halo thickness */
      char  circleSegments,      /* Number of sides */
      float *color,              /* Polygon Color */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   float tma, R, G, B;
   float degSegment;
   R = color[0];
   G = color[1];
   B = color[2];
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
   /* X */ verts.push_back(float(bx+cos(tma)*bsx));
   /* Y */ verts.push_back(float(by+sin(tma)*bsy));
   /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);

   for (int i = 0; i < circleSegments+1; i ++ ) {
      tma = float(degToRad(begin+i*degSegment));
      /* X */ verts.push_back(float(bx+cos(tma)*bsx));
      /* Y */ verts.push_back(float(by+sin(tma)*bsy));
      /* X */ verts.push_back(float(bx+cos(tma)*(bsx+rs)));
      /* Y */ verts.push_back(float(by+sin(tma)*(bsy+rs)));

      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
   }
   tma = float(degToRad(begin+(circleSegments)*degSegment));
   /* X */ verts.push_back(float(bx+cos(tma)*(bsx+rs)));
   /* Y */ verts.push_back(float(by+sin(tma)*(bsy+rs)));

   /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);

   return verts.size()/2;
}

// Update vertices to allocated arrays
int updateArchGeometry(
      float bx,                  /* X-Coordinate */
      float by,                  /* Y-Coordinate */
      float bsx,                 /* x-Scale 2.0=spans display before GL scaling */
      float bsy,                 /* y-Scale 2.0=spans display before GL scaling */
      float start,               /* Where, in degrees on the unit circle, the arch begins */
      float end,                 /* Where, in degrees on the unit circle, the arch endgs */
      float rs,                  /* Halo thickness */
      char  circleSegments,      /* Number of sides */
      int   index,               /* Index of where to start writing to input arrays */
      float *verts               /* Input Array of x,y coordinates */
      ){
   float tma, degSegment, begin;
   int vertIndex = index*2;
   if (start <= end) {
      degSegment = abs(end - start) / float(circleSegments);
      begin = start;
   } else {
      degSegment = ((360.0f-start) + end) / float(circleSegments);
      begin = start;
   }

   // Prepend Degenerate Vertex
   tma = float(degToRad(begin));
   /* X */ verts[vertIndex++] = (float(bx+cos(tma)*bsx));
   /* Y */ verts[vertIndex++] = (float(by+sin(tma)*bsy));

   for (int i = 0; i < circleSegments+1; i ++ ) {
      tma = float(degToRad(begin+i*degSegment));
      /* X */ verts[vertIndex++] = (float(bx+cos(tma)*bsx));
      /* Y */ verts[vertIndex++] = (float(by+sin(tma)*bsy));
      /* X */ verts[vertIndex++] = (float(bx+cos(tma)*(bsx+rs)));
      /* Y */ verts[vertIndex++] = (float(by+sin(tma)*(bsy+rs)));
   }
   tma = float(degToRad(begin+(circleSegments)*degSegment));
   /* X */ verts[vertIndex++] = (float(bx+cos(tma)*(bsx+rs)));
   /* Y */ verts[vertIndex++] = (float(by+sin(tma)*(bsy+rs)));

   return vertIndex/2;
}

// Update vertices to allocated arrays
int updateArchColor(
      float start,               /* Where, in degrees on the unit circle, the arch begins */
      float end,                 /* Where, in degrees on the unit circle, the arch endgs */
      float rs,                  /* Halo thickness */
      char  circleSegments,      /* Number of sides */
      float *color,              /* Polygon Color */
      int   index,               /* Index of where to start writing to input arrays */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   int colrIndex = index*3;
   float R, G, B;
   R = color[0];
   G = color[1];
   B = color[2];

   // Prepend Degenerate Vertex
   /* R */ colrs[colrIndex++] = R;   /* G */ colrs[colrIndex++] = G;   /* B */ colrs[colrIndex++] = B;

   for (int i = 0; i < circleSegments+1; i ++ ) {
      /* R */ colrs[colrIndex++] = R;   /* G */ colrs[colrIndex++] = G;   /* B */ colrs[colrIndex++] = B;
      /* R */ colrs[colrIndex++] = R;   /* G */ colrs[colrIndex++] = G;   /* B */ colrs[colrIndex++] = B;
   }
   /* R */ colrs[colrIndex++] = R;   /* G */ colrs[colrIndex++] = G;   /* B */ colrs[colrIndex++] = B;

   return colrIndex/3;
}
