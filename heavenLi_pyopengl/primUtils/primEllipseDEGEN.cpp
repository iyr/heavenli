/*
 *  Defines an efficient circle for TRIANGLE_STRIP with degenerate vertices
 */

#include <math.h>
#include <vector>
using namespace std;

// Append to input vectors
int defineEllipse(
      float bx,                  /* X-Coordinate */
      float by,                  /* Y-Coordinate */
      float bsx,                 /* x-Scale 2.0=spans display before  scaling */
      float bsy,                 /* y-Scale 2.0=spans display before  scaling */
      char circleSegments,       /* Number of sides */
      float *color,              /* Polygon Color */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   float R, G, B;
   char degSegment = 360 / circleSegments;
   R = color[0];
   G = color[1];
   B = color[2];

   // Prepend degenerate vertex iff not the first primitive in the vector
   if (verts.size() == 0) {
      /* X */ verts.push_back(float(bx + bsx*1.0f));
      /* Y */ verts.push_back(float(by + bsy*0.0f));
      /* R */ colrs.push_back(R);  /* G */ colrs.push_back(G);  /* B */ colrs.push_back(B);
   } else {
      /* X */ verts.push_back(float(bx + bsx*1.0f));
      /* Y */ verts.push_back(float(by + bsy*0.0f));
      /* R */ colrs.push_back(R);  /* G */ colrs.push_back(G);  /* B */ colrs.push_back(B);
      /* X */ verts.push_back(float(bx + bsx*1.0f));
      /* Y */ verts.push_back(float(by + bsy*0.0f));
      /* R */ colrs.push_back(R);  /* G */ colrs.push_back(G);  /* B */ colrs.push_back(B);
   }

   for (char i = 1; i < circleSegments; i++ ) {
      /* X */ verts.push_back(float(bx + bsx*cos(degToRad(i*degSegment))));
      /* Y */ verts.push_back(float(by + bsy*sin(degToRad(i*degSegment))));
      /* R */ colrs.push_back(R);  /* G */ colrs.push_back(G);  /* B */ colrs.push_back(B);
      /* X */ verts.push_back(float(bx + bsx*cos(degToRad(i*degSegment))));
      /* Y */ verts.push_back(float(by - bsy*sin(degToRad(i*degSegment))));
      /* R */ colrs.push_back(R);  /* G */ colrs.push_back(G);  /* B */ colrs.push_back(B);
   }
   /* X */ verts.push_back(float(bx - bsx*1.0f));
   /* Y */ verts.push_back(float(by + bsy*0.0f));
   /* R */ colrs.push_back(R);  /* G */ colrs.push_back(G);  /* B */ colrs.push_back(B);
   /* X */ verts.push_back(float(bx - bsx*1.0f));
   /* Y */ verts.push_back(float(by + bsy*0.0f));
   /* R */ colrs.push_back(R);  /* G */ colrs.push_back(G);  /* B */ colrs.push_back(B);

   return verts.size()/2;
}

// Write to pre-allocated input array, updating vertices only 
int updatePrimEllipseGeometry(
      float bx,               /* X-Coordinate */
      float by,               /* Y-Coordinate */
      float bsx,              /* x-Scale 2.0=spans display before GL scaling */
      float bsy,              /* y-Scale 2.0=spans display before GL scaling */
      char  circleSegments,   /* Number of sides */
      int   index,            /* Index of where to start writing to input arrays */
      float *verts            /* Input Vector of x,y values */
      ){
   int vertIndex = index*2;   /* index (x, y) */
   char degSegment = 360 / circleSegments;

   if (vertIndex == 0) {
      /* X */ verts[vertIndex++] = (float(bx + bsx*1.0f));
      /* Y */ verts[vertIndex++] = (float(by + bsy*0.0f));
   } else {
      /* X */ verts[vertIndex++] = (float(bx + bsx*1.0f));
      /* Y */ verts[vertIndex++] = (float(by + bsy*0.0f));
      /* X */ verts[vertIndex++] = (float(bx + bsx*1.0f));
      /* Y */ verts[vertIndex++] = (float(by + bsy*0.0f));
   }

   for (char i = 1; i < circleSegments; i++ ) {
      /* X */ verts[vertIndex++] = (float(bx + bsx*cos(degToRad(i*degSegment))));
      /* Y */ verts[vertIndex++] = (float(by + bsy*sin(degToRad(i*degSegment))));
      /* X */ verts[vertIndex++] = (float(bx + bsx*cos(degToRad(i*degSegment))));
      /* Y */ verts[vertIndex++] = (float(by - bsy*sin(degToRad(i*degSegment))));
   }
   /* X */ verts[vertIndex++] = (float(bx - bsx*1.0f));
   /* Y */ verts[vertIndex++] = (float(by + bsy*0.0f));
   /* X */ verts[vertIndex++] = (float(bx - bsx*1.0f));
   /* Y */ verts[vertIndex++] = (float(by + bsy*0.0f));

   return vertIndex/2;
}

// Write to pre-allocated input array, updating color only 
int updatePrimEllipseColor(
      char circleSegments, /* Number of sides */
      int   index,         /* Index of where to start writing to input arrays */
      float *color,        /* Polygon Color */
      float *colrs         /* Input Vector of r,g,b values */
      ){
   int colrIndex = index*3;   /* index (r, g, b) */
   float R, G, B;
   R = color[0];
   G = color[1];
   B = color[2];

   if (colrIndex == 0) {
      /* R */ colrs[colrIndex++] = R;  /* G */ colrs[colrIndex++] = G;  /* B */ colrs[colrIndex++] = B;
   } else {
      /* R */ colrs[colrIndex++] = R;  /* G */ colrs[colrIndex++] = G;  /* B */ colrs[colrIndex++] = B;
      /* R */ colrs[colrIndex++] = R;  /* G */ colrs[colrIndex++] = G;  /* B */ colrs[colrIndex++] = B;
   }

   for (char i = 1; i < circleSegments; i++ ) {
      /* R */ colrs[colrIndex++] = R;  /* G */ colrs[colrIndex++] = G;  /* B */ colrs[colrIndex++] = B;
      /* R */ colrs[colrIndex++] = R;  /* G */ colrs[colrIndex++] = G;  /* B */ colrs[colrIndex++] = B;
   }
   /* R */ colrs[colrIndex++] = R;  /* G */ colrs[colrIndex++] = G;  /* B */ colrs[colrIndex++] = B;
   /* R */ colrs[colrIndex++] = R;  /* G */ colrs[colrIndex++] = G;  /* B */ colrs[colrIndex++] = B;

   return colrIndex/3;
}
