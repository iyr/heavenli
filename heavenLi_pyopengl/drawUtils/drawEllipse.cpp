#include <math.h>
#include <vector>
using namespace std;

// Append Ellipse vertices to input vectors
int drawEllipse(
      float bx,                  /* X-Coordinate */
      float by,                  /* Y-Coordinate */
      float bsx,                 /* x-Scale 2.0=spans display before GL scaling */
      float bsy,                 /* y-Scale 2.0=spans display before GL scaling */
      char circleSegments,       /* Number of sides */
      float *color,             /* Polygon Color */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   float R, G, B;
   char degSegment = 360 / circleSegments;
   R = color[0];
   G = color[1];
   B = color[2];

//#  pragma omp parallel for
   for (char i = 0; i < circleSegments; i++) {
      /* X */ verts.push_back(float(bx));
      /* Y */ verts.push_back(float(by));

      /* X */ verts.push_back(float(bx + bsx*cos(degToRad(90+i*degSegment))));
      /* Y */ verts.push_back(float(by + bsy*sin(degToRad(90+i*degSegment))));

      /* X */ verts.push_back(float(bx + bsx*cos(degToRad(90+(i+1)*degSegment))));
      /* Y */ verts.push_back(float(by + bsy*sin(degToRad(90+(i+1)*degSegment))));

      /* R */ colrs.push_back(R);  /* G */ colrs.push_back(G);  /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);  /* G */ colrs.push_back(G);  /* B */ colrs.push_back(B);
      /* R */ colrs.push_back(R);  /* G */ colrs.push_back(G);  /* B */ colrs.push_back(B);
   }
   return verts.size()/2;
}

// Append Circle vertices to input vectors
int drawEllipse(
      float bx,                  /* X-Coordinate */
      float by,                  /* Y-Coordinate */
      float bs,                  /* Scale~ 2.0=spans display before GL Transformations */
      char circleSegments,       /* Number of sides */
      float *color,             /* Polygon Color */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   return drawEllipse(bx, by, bs, bs, circleSegments, color, verts, colrs);
}

// Append Circle vertices to input vectors
int drawCircle(
      float bx,                  /* X-Coordinate */
      float by,                  /* Y-Coordinate */
      float bs,                  /* Scale~ 2.0=spans display before GL Transformations */
      char circleSegments,       /* Number of sides */
      float *color,             /* Polygon Color */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   return drawEllipse(bx, by, bs, bs, circleSegments, color, verts, colrs);
}

