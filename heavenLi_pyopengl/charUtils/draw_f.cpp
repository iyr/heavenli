#include <math.h>
#include <vector>
using namespace std;

void draw_f(
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
         bx, by+1.125f, 
         bs, bs, 
         65.0f, 180.0f,
         2.0f*bt, 
         bc, color, 
         verts, colrs);
   drawPill(
         bx -1.0f-bt, by +1.125f+bt, 
         bx -1.0f-bt, by -1.0f-bt, 
         bt, color, 
         verts, colrs);
   drawPill(
         bx+0.3f-bt, by+0.66f, 
         bx-1.5f+bt, by+0.66f, 
         bt, color, 
         verts, colrs);
   *lineWidth += ((bx+bs)-(bx-0.5f+bt))+b_;

   return;
}

