#include <math.h>
#include <vector>
using namespace std;

void draw_e(
      float bx,                  /* X-Coordinate */
      float by,                  /* Y-Coordinate */
      float bs,                  /* Scale */
      float bt,                  /* Thickness */
      float b_,                  /* Amount of empty space after character */
      char  bc,                  /* Polygon Detail Count */
      float *color,              /* Array containing RGB values */
      float *lineWidth,          /* Stores the width of all of the characters on a line */
      std::vector<float> &verts, /* Input Vector of x,y coordinate */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){

   drawArch(
         bx, by, 
         bs, bs, 
         0.0f, 315.0f,
         2.0f*bt, 
         bc, 
         color, 
         verts, 
         colrs);
   drawPill(
         bx-bs, by,
         bx+bs*1.1f, by,
         bt,
         color,
         verts,
         colrs);
   *lineWidth += 2.2f*(bs+bt)+b_;

   return;
}
