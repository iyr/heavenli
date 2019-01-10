#include <math.h>
#include <vector>
using namespace std;

// Append Halo vertices to input vectors
int drawHalo(
      float bx,                  /* X-Coordinate */
      float by,                  /* Y-Coordinate */
      float bsx,                 /* x-Scale 2.0=spans display before GL scaling */
      float bsy,                 /* y-Scale 2.0=spans display before GL scaling */
      float rs,                  /* Halo thickness */
      char circleSegments,       /* Number of sides */
      double *color,             /* Polygon Color */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   float tma, R, G, B;
   char degSegment = 360 / circleSegments;
   R = float(color[0]);
   G = float(color[1]);
   B = float(color[2]);

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
