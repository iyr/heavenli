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

   bt *= bs;

   switch (character) {
      case 'a':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         drawPill(
               bx+bs+bt, by+ bs+bt, 
               bx+bs+bt, by+-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 'b':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         drawPill(
               bx+-bs-bt, by+ bs*2.0f+bt, 
               bx+-bs-bt, by+-bs-bt, 
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
               bc, color, 
               verts, colrs);
         *lineWidth += 1.6f*(bs+bt)+b_;
         break;

      case 'd':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         drawPill(
               bx +bs+bt, by+ bs*2.0f+bt, 
               bx +bs+bt, by+-bs-bt, 
               bt, color, 
               verts, colrs);
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
               bx, by+bs*1.125f, 
               bs, bs, 
               65.0f, 180.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         drawPill(
               bx-bs-bt, by+bs*1.125f+bt, 
               bx-bs-bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs*0.3f-bt, by+bs*0.5f, 
               bx-bs*1.5f+bt, by+bs*0.5f, 
               bt, color, 
               verts, colrs);
         //*lineWidth += ((bx+bs)-(bx-bs*0.5f+bt))+b_;
         *lineWidth += 1.7f*(bs+bt)+b_;
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
         drawArch(
               bx, by+bs/1.7f, 
               bs/2.0f, bs/2.2f, 
               40.0f, 280.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         drawArch(
               bx-bs/4.0f, by-bs/1.7f, 
               bs/2.0f, bs/2.2f, 
               270.0f, 40.0f, 
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 't':
         drawArch(
               bx+bs/2.0f+bt, by-bs, 
               bs/2.0f, bs/2.0f, 
               180.0f, 270.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         drawPill(
               bx, by+bs*2.0f+bt, 
               bx, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs-bt, by+bs, 
               bx-bs+bt, by+bs, 
               bt, color, 
               verts, colrs);
         *lineWidth += 1.9f*(bs+bt)+b_;
         break;

      case 'u':
         drawArch(
               bx, by, 
               bs, bs, 
               180.0f, 360.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         drawPill(
               bx-bs-bt, by+bs, 
               bx-bs-bt, by+bs*0.125f-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs+bt, by+ bs, 
               bx+bs+bt, by+-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 'v':
         drawPill(
               bx+bs, by+bs, 
               bx, by-bs, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx-bs, by+bs, 
               bx, by-bs, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 'w':
         drawPill(
               bx+bs*1.2f, by+bs, 
               bx+bs*0.6f, by-bs, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx, by+bs, 
               bx+bs*0.6f, by-bs, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx-bs*1.2f, by+bs, 
               bx-bs*0.6f, by-bs, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx, by+bs, 
               bx-bs*0.6f, by-bs, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 'x':
         drawPill(
               bx+bs, by-bs, 
               bx-bs, by+bs, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs, by+bs, 
               bx-bs, by-bs, 
               bt, color, 
               verts, colrs);
         *lineWidth += 1.8f*(bs+bt)+b_;
         break;

      case 'y':
         drawPill(
               bx, by-bs*0.5f, 
               bx-bs, by+bs, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs, by+bs, 
               bx-bs, by-bs*2.0f, 
               bt, color, 
               verts, colrs);
         *lineWidth += 1.8f*(bs+bt)+b_;
         break;

      case 'z':
         drawPill(
               bx+bs, by+bs, 
               bx-bs, by+bs, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs, by-bs, 
               bx-bs, by-bs, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs, by+bs, 
               bx-bs, by-bs, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;
   }
   return;
}

