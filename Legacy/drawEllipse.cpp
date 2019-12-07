#include <math.h>
#include <vector>
using namespace std;

// Write to pre-allocated input array, updating color only 
int updateEllipseColor(
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

//#  pragma omp parallel for
   for (char i = 0; i < circleSegments; i++) {
      /* R */ colrs[colrIndex++] = R;  
      /* G */ colrs[colrIndex++] = G;  
      /* B */ colrs[colrIndex++] = B;
     
      /* R */ colrs[colrIndex++] = R;  
      /* G */ colrs[colrIndex++] = G;  
      /* B */ colrs[colrIndex++] = B;
      
      /* R */ colrs[colrIndex++] = R;  
      /* G */ colrs[colrIndex++] = G;  
      /* B */ colrs[colrIndex++] = B;
   }
   return colrIndex/3;
}

// Write to pre-allocated input array, updating vertices only 
int updateEllipseGeometry(
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

//#  pragma omp parallel for
   for (char i = 0; i < circleSegments; i++) {
      /* X */ verts[vertIndex++] = float(bx);
      /* Y */ verts[vertIndex++] = float(by);

      /* X */ verts[vertIndex++] = float(bx + bsx*cos(degToRad(90+i*degSegment)));
      /* Y */ verts[vertIndex++] = float(by + bsy*sin(degToRad(90+i*degSegment)));

      /* X */ verts[vertIndex++] = float(bx + bsx*cos(degToRad(90+(i+1)*degSegment)));
      /* Y */ verts[vertIndex++] = float(by + bsy*sin(degToRad(90+(i+1)*degSegment)));
   }
   return vertIndex/2;
}

// Write to pre-allocated input array, updating vertices only 
int updateEllipseGeometry(
      float bx,               /* X-Coordinate */
      float by,               /* Y-Coordinate */
      float bs,               /* Scale 2.0=spans display before GL scaling */
      char  circleSegments,   /* Number of sides */
      int   index,            /* Index of where to start writing to input arrays */
      float *verts            /* Input Vector of x,y values */
      ){
   return updateEllipseGeometry(bx, by, bs, bs, circleSegments, index, verts);
}

// Append Ellipse vertices to input vectors
int drawEllipse(
      float bx,                  /* X-Coordinate */
      float by,                  /* Y-Coordinate */
      float bsx,                 /* x-Scale 2.0=spans display before GL scaling */
      float bsy,                 /* y-Scale 2.0=spans display before GL scaling */
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
      float *color,              /* Polygon Color */
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
      float *color,              /* Polygon Color */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   return drawEllipse(bx, by, bs, bs, circleSegments, color, verts, colrs);
}

