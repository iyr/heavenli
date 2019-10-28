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
   float R, G, B, A;
   float degSegment = 360.0f / float(circleSegments);
   degSegment /= 2.0f;
   R = color[0];
   G = color[1];
   B = color[2];
   A = color[3];

   // Prepend degenerate vertex iff not the first primitive in the vector
   if (verts.size() == 0) {
      /* X */ verts.push_back(float(bx + bsx*1.0f));
      /* Y */ verts.push_back(float(by + bsy*0.0f));
      /* R */ colrs.push_back(R);  
      /* G */ colrs.push_back(G);  
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);
   } else {
      /* X */ verts.push_back(float(bx + bsx*1.0f));
      /* Y */ verts.push_back(float(by + bsy*0.0f));
      /* R */ colrs.push_back(R);  
      /* G */ colrs.push_back(G);  
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);

      /* X */ verts.push_back(float(bx + bsx*1.0f));
      /* Y */ verts.push_back(float(by + bsy*0.0f));
      /* R */ colrs.push_back(R);  
      /* G */ colrs.push_back(G);  
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);
   }

   for (unsigned int i = 0; i < circleSegments; i++ ) {
      //printf("circleSegments: %d, i: %d, verts size: %d\n", circleSegments, i, verts.size());
      /* X */ verts.push_back(float(bx + bsx*cos(degToRad((float)i*degSegment))));
      /* Y */ verts.push_back(float(by + bsy*sin(degToRad((float)i*degSegment))));
      /* R */ colrs.push_back(R);  
      /* G */ colrs.push_back(G);  
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);

      /* X */ verts.push_back(float(bx + bsx*cos(degToRad((float)i*degSegment))));
      /* Y */ verts.push_back(float(by - bsy*sin(degToRad((float)i*degSegment))));
      /* R */ colrs.push_back(R);  
      /* G */ colrs.push_back(G);  
      /* B */ colrs.push_back(B);
      /* A */ colrs.push_back(A);
   }
   /* X */ verts.push_back(float(bx - bsx*1.0f));
   /* Y */ verts.push_back(float(by + bsy*0.0f));
   /* R */ colrs.push_back(R);  
   /* G */ colrs.push_back(G);  
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

   /* X */ verts.push_back(float(bx - bsx*1.0f));
   /* Y */ verts.push_back(float(by + bsy*0.0f));
   /* R */ colrs.push_back(R);  
   /* G */ colrs.push_back(G);  
   /* B */ colrs.push_back(B);
   /* A */ colrs.push_back(A);

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
   float degSegment = 360.0f / float(circleSegments);
   degSegment /= 2.0f;

   if (vertIndex == 0) {
      /* X */ verts[vertIndex++] = (float(bx + bsx*1.0f));
      /* Y */ verts[vertIndex++] = (float(by + bsy*0.0f));
   } else {
      /* X */ verts[vertIndex++] = (float(bx + bsx*1.0f));
      /* Y */ verts[vertIndex++] = (float(by + bsy*0.0f));

      /* X */ verts[vertIndex++] = (float(bx + bsx*1.0f));
      /* Y */ verts[vertIndex++] = (float(by + bsy*0.0f));
   }

   for (unsigned int i = 0; i < circleSegments; i++ ) {
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

   for (unsigned int i = 0; i < circleSegments; i++ ) {
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

   return colrIndex/4;
}
