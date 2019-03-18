#include <math.h>
#include <vector>
using namespace std;

void draw_char(
      char  character,           /* Character to draw */
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

   switch (character) {
      case 'a':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         drawPill(
               bx+1.0f+bt, by+ 1.0f+bt, 
               bx+1.0f+bt, by+-1.0f-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 'b':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         drawPill(
               bx+-1.0f-bt, by+ 2.0f+bt, 
               bx+-1.0f-bt, by+-1.0f-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 'c':
         drawArch(
               bx, by, 
               bs, bs, 
               45.0f, 315.0f,
               2.0f*bt, 
               bc, 
               color, 
               verts, 
               colrs);
         *lineWidth += 1.6f*(bs+bt)+b_;
         break;

      case 'd':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         drawPill(
               bx +1.0f+bt, by+ 2.0f+bt, 
               bx +1.0f+bt, by+-1.0f-bt, 
               bt, 
               color, 
               verts, 
               colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 'e':
         drawArch(
               bx, by, 
               bs, bs, 
               0.0f, 315.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         drawPill(
               bx-bs, by,
               bx+bs*1.1f, by,
               bt, color,
               verts, colrs);
         *lineWidth += 2.2f*(bs+bt)+b_;
         break;

      case 'f':
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
         break;

      case 'g':
         break;

      case 'h':
         break;

      case 'i':
         break;

      case 'j':
         break;

      case 'k':
         break;

      case 'l':
         break;

      case 'm':
         break;

      case 'n':
         break;

      case 'o':
         break;

      case 'p':
         break;

      case 'q':
         break;

      case 'r':
         break;

      case 's':
         break;

      case 't':
         break;

      case 'u':
         break;

      case 'v':
         break;

      case 'w':
         drawPill(
               bx+1.2f, by+1.0f, 
               bx+0.6f, by-1.0f, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+0.0f, by+1.0f, 
               bx+0.6f, by-1.0f, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx-1.2f, by+1.0f, 
               bx-0.6f, by-1.0f, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+0.0f, by+1.0f, 
               bx-0.6f, by-1.0f, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 'x':
         drawPill(
               bx+1.0f, by-1.0f, 
               bx-1.0f, by+1.0f, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+1.0f, by+1.0f, 
               bx-1.0f, by-1.0f, 
               bt, color, 
               verts, colrs);
         *lineWidth += 1.8f*(bs+bt)+b_;
         break;

      case 'y':
         drawPill(
               bx+0.0f, by-0.5f, 
               bx-1.0f, by+1.0f, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+1.0f, by+1.0f, 
               bx-1.0f, by-2.0f, 
               bt, color, 
               verts, colrs);
         *lineWidth += 1.8f*(bs+bt)+b_;
         break;

      case 'z':
         drawPill(
               bx+1.0f, by+1.0f, 
               bx-1.0f, by+1.0f, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+1.0f, by-1.0f, 
               bx-1.0f, by-1.0f, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+1.0f, by+1.0f, 
               bx-1.0f, by-1.0f, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

   }

   return;
}

