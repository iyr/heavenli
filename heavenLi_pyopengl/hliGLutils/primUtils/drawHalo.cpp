#include <math.h>
#include <vector>
using namespace std;

// Update to pre-existing input arrays staring at index
unsigned int drawHalo(
      float bx,               /* X-Coordinate */
      float by,               /* Y-Coordinate */
      float bsx,              /* x-Scale 2.0=spans display before GL scaling */
      float bsy,              /* y-Scale 2.0=spans display before GL scaling */
      float rs,               /* Halo thickness */
      char  circleSegments,   /* Number of sides */
      int   index,            /* index of where to start writing to input arrays */
      float *color,           /* Polygon Color */
      float *verts,           /* Input Array of x,y coordinates */
      float *colrs            /* Input Array of r,g,b values */
      ){
   int vertIndex = index*2;
   int colrIndex = index*3;
   float tma, R, G, B;
   char degSegment = 360 / circleSegments;
   R = color[0];
   G = color[1];
   B = color[2];

//#  pragma omp parallel for
   for (int i = 0; i < circleSegments; i ++ ) {
      tma = float(degToRad((i+0)*float(degSegment)));
      /* X */ verts[vertIndex++] = float(bx+cos(tma)*bsx);
      /* Y */ verts[vertIndex++] = float(by+sin(tma)*bsy);
      /* X */ verts[vertIndex++] = float(bx+cos(tma)*(bsx+rs));
      /* Y */ verts[vertIndex++] = float(by+sin(tma)*(bsy+rs));
      tma = float(degToRad((i+1)*float(degSegment)));
      /* X */ verts[vertIndex++] = float(bx+cos(tma)*bsx);
      /* Y */ verts[vertIndex++] = float(by+sin(tma)*bsy);

      /* X */ verts[vertIndex++] = float(bx+cos(tma)*(bsx+rs));
      /* Y */ verts[vertIndex++] = float(by+sin(tma)*(bsy+rs));
      /* X */ verts[vertIndex++] = float(bx+cos(tma)*bsx);
      /* Y */ verts[vertIndex++] = float(by+sin(tma)*bsy);
      tma = float(degToRad((i+0)*float(degSegment)));
      /* X */ verts[vertIndex++] = float(bx+cos(tma)*(bsx+rs));
      /* Y */ verts[vertIndex++] = float(by+sin(tma)*(bsy+rs));

      /* R */ colrs[colrIndex++] = R;   /* G */ colrs[colrIndex++] = G;   /* B */ colrs[colrIndex++] = B;
      /* R */ colrs[colrIndex++] = R;   /* G */ colrs[colrIndex++] = G;   /* B */ colrs[colrIndex++] = B;
      /* R */ colrs[colrIndex++] = R;   /* G */ colrs[colrIndex++] = G;   /* B */ colrs[colrIndex++] = B;
      /* R */ colrs[colrIndex++] = R;   /* G */ colrs[colrIndex++] = G;   /* B */ colrs[colrIndex++] = B;
      /* R */ colrs[colrIndex++] = R;   /* G */ colrs[colrIndex++] = G;   /* B */ colrs[colrIndex++] = B;
      /* R */ colrs[colrIndex++] = R;   /* G */ colrs[colrIndex++] = G;   /* B */ colrs[colrIndex++] = B;
   }

   return vertIndex/2;
}

// Update to pre-existing input arrays staring at index
unsigned int drawHalo(
      float bx,               /* X-Coordinate */
      float by,               /* Y-Coordinate */
      float bs,               /* Scale 2.0=spans display before GL scaling */
      float rs,               /* Halo thickness */
      char  circleSegments,   /* Number of sides */
      int   index,            /* index of where to start writing to input arrays */
      float *color,           /* Polygon Color */
      float *verts,           /* Input Array of x,y coordinates */
      float *colrs            /* Input Array of r,g,b values */
      ){
   return drawHalo(bx, by, bs, bs, rs, circleSegments, index, color, verts, colrs);
}

// Append Halo vertices to input vectors
unsigned int drawHalo(
      float bx,                  /* X-Coordinate */
      float by,                  /* Y-Coordinate */
      float bsx,                 /* x-Scale 2.0=spans display before GL scaling */
      float bsy,                 /* y-Scale 2.0=spans display before GL scaling */
      float rs,                  /* Halo thickness */
      char circleSegments,       /* Number of sides */
      float *color,              /* Polygon Color */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   float tma, R, G, B;
   char degSegment = 360 / circleSegments;
   R = color[0];
   G = color[1];
   B = color[2];

//#  pragma omp parallel for
   for (int i = 0; i < circleSegments; i ++ ) {
      tma = float(degToRad((i+0)*float(degSegment)));
      /* X */ verts.push_back(float(bx+cos(tma)*bsx));
      /* Y */ verts.push_back(float(by+sin(tma)*bsy));
      /* X */ verts.push_back(float(bx+cos(tma)*(bsx+rs)));
      /* Y */ verts.push_back(float(by+sin(tma)*(bsy+rs)));
      tma = float(degToRad((i+1)*float(degSegment)));
      /* X */ verts.push_back(float(bx+cos(tma)*bsx));
      /* Y */ verts.push_back(float(by+sin(tma)*bsy));

      /* X */ verts.push_back(float(bx+cos(tma)*(bsx+rs)));
      /* Y */ verts.push_back(float(by+sin(tma)*(bsy+rs)));
      /* X */ verts.push_back(float(bx+cos(tma)*bsx));
      /* Y */ verts.push_back(float(by+sin(tma)*bsy));
      tma = float(degToRad((i+0)*float(degSegment)));
      /* X */ verts.push_back(float(bx+cos(tma)*(bsx+rs)));
      /* Y */ verts.push_back(float(by+sin(tma)*(bsy+rs)));

      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
   }

   return verts.size()/2;
}
