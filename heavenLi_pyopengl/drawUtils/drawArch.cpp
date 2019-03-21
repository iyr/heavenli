#include <math.h>
#include <vector>
using namespace std;

// Append Halo vertices to input vectors
void drawArch(
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
   float degSegment = float(end - start) / float(circleSegments);
   R = color[0];
   G = color[1];
   B = color[2];
   float begin;
   if (start <= end)
      begin = start;
   else
      begin = -start;

//#  pragma omp parallel for
   for (int i = 0; i < circleSegments; i ++ ) {
      tma = float(degToRad(begin+(i+0)*degSegment));
      /* X */ verts.push_back(float(bx+cos(tma)*bsx));
      /* Y */ verts.push_back(float(by+sin(tma)*bsy));
      /* X */ verts.push_back(float(bx+cos(tma)*(bsx+rs)));
      /* Y */ verts.push_back(float(by+sin(tma)*(bsy+rs)));
      tma = float(degToRad(begin+(i+1)*degSegment));
      /* X */ verts.push_back(float(bx+cos(tma)*bsx));
      /* Y */ verts.push_back(float(by+sin(tma)*bsy));

      /* X */ verts.push_back(float(bx+cos(tma)*(bsx+rs)));
      /* Y */ verts.push_back(float(by+sin(tma)*(bsy+rs)));
      /* X */ verts.push_back(float(bx+cos(tma)*bsx));
      /* Y */ verts.push_back(float(by+sin(tma)*bsy));
      tma = float(degToRad(begin+(i+0)*degSegment));
      /* X */ verts.push_back(float(bx+cos(tma)*(bsx+rs)));
      /* Y */ verts.push_back(float(by+sin(tma)*(bsy+rs)));

      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);   /* G */ colrs.push_back(G);   /* B */ colrs.push_back(B);
   }

   return;
}
