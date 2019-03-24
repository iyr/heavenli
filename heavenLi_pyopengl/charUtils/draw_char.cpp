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
               bx+bs+bt, by+bs+bt, 
               bx+bs+bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
         break;

      case 'b':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         drawPill(
               bx-bs-bt, by+bs*2.0f+bt, 
               bx-bs-bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
         break;

      case 'c':
         drawArch(
               bx, by, 
               bs, bs, 
               45.0f, 315.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         *lineWidth += 1.7f*(bs+bt)+b_;
         break;

      case 'd':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         drawPill(
               bx+bs+bt, by+bs*2.0f+bt, 
               bx+bs+bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
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
               bx+bs+bt, by,
               bt, color,
               verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
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
         *lineWidth += 1.2f*(bs+bt)+b_;
         break;

      case 'g':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         drawPill(
               bx+bs+bt, by+bs+bt, 
               bx+bs+bt, by-bs*1.125f, 
               bt, color, 
               verts, colrs);
         drawArch(
               bx, by-bs*1.125f, 
               bs, bs, 
               210.0f, 360.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
         break;

      case 'h':
         drawArch(
               bx, by, 
               bs, bs, 
               360.0f, 180.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         drawPill(
               bx+bs+bt, by, 
               bx+bs+bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx-bs-bt, by+bs*2.0f+bt, 
               bx-bs-bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
         break;

      case 'i':
         bx -= 1.1f*bs;
         *lineWidth -= 1.1f*bs;
         drawCircle(
               bx, by+bs+bt*2.2f,
               bt*2.0f, bc, color,
               verts, colrs);
         drawPill(
               bx, by+bs-bt*2.2f, 
               bx, by-bs*1.125f, 
               bt, color, 
               verts, colrs);
         *lineWidth += 1.1f*(bs+bt)+b_;
         break;

      case 'j':
         bx -= 0.3f*bs;
         *lineWidth -= 0.3f*bs;
         drawCircle(
               bx+bs+bt, by+bs+bt*2.2f,
               bt*2.0f, bc, color,
               verts, colrs);
         drawPill(
               bx+bs+bt, by+bs-bt*2.2f, 
               bx+bs+bt, by-bs*1.125f, 
               bt, color, 
               verts, colrs);
         drawArch(
               bx, by-bs*1.125f, 
               bs, bs, 
               225.0f, 360.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         *lineWidth += 2.2f*(bs+bt)+b_;
         break;

      case 'k':
         drawPill(
               bx+-bs-bt, by+bs*2.0f+bt, 
               bx+-bs-bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs, by-bs-bt, 
               bx, by, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs, by+bs+bt, 
               bx-bs-bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
         break;

      case 'l':
         bx -= 1.2f*bs;
         *lineWidth -= 1.2f*bs;
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
         *lineWidth += 1.3f*(bs+bt)+b_;
         break;

      case 'm':
         float archY; 
         archY = by+bs/(2.0f+bt);
         drawArch(
               bx-bs/(1.89f-bt), archY, 
               bs/(2.19f+bt), bs/(2.0f+bt), 
               360.0f, 180.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         drawArch(
               bx+bs/(1.89f-bt), archY, 
               bs/(2.19f+bt), bs/(2.0f+bt), 
               360.0f, 180.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         drawPill(
               bx+bs+bt, archY, 
               bx+bs+bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx, archY, 
               bx, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx-bs-bt, by+bs+bt, 
               bx-bs-bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
         break;

      case 'n':
         drawArch(
               bx, by, 
               bs, bs, 
               360.0f, 180.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         drawPill(
               bx+bs+bt, by, 
               bx+bs+bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx-bs-bt, by+bs+bt, 
               bx-bs-bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
         break;

      case 'o':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
         break;

      case 'p':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         drawPill(
               bx-bs-bt, by+bs+bt, 
               bx-bs-bt, by-bs*2.0f-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
         break;

      case 'q':
         drawHalo(bx, by, bs, bs, 2.0f*bt, bc, color, verts, colrs);
         drawPill(
               bx+bs+bt, by+bs+bt, 
               bx+bs+bt, by-bs*2.0f-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.1f*(bs+bt)+b_;
         break;

      case 'r':
         drawArch(
               bx, by, 
               bs, bs, 
               40.0f, 180.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         drawPill(
               bx-bs-bt, by-bs-bt,
               bx-bs-bt, by+bs+bt,
               bt, color,
               verts, colrs);
         *lineWidth += 1.8f*(bs+bt)+b_;
         break;

      case 's':
         bx -= 0.2f*bs;
         *lineWidth -= 0.2f*bs;
         drawArch(
               bx, by+bs/1.85f, 
               bs/1.2f, bs/2.5f, 
               20.0f, 280.0f,
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         drawArch(
               bx, by-bs/1.8f, 
               bs/1.2f, bs/2.5f, 
               200.0f, 90.0f, 
               2.0f*bt, 
               bc, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 't':
         bx -= 0.4f*bs;
         *lineWidth -= 0.4f*bs;
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
               bx+bs-bt, by+bs*0.5f, 
               bx-bs+bt, by+bs*0.5f, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
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
               bx-bs-bt, by+bs+bt, 
               bx-bs-bt, by+bs*0.125f-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs+bt, by+bs+bt, 
               bx+bs+bt, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 'v':
         drawPill(
               bx+bs, by+bs+bt, 
               bx, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx-bs, by+bs+bt, 
               bx, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 'w':
         drawPill(
               bx+bs*1.2f, by+bs+bt, 
               bx+bs*0.6f, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx, by+bs+bt, 
               bx+bs*0.6f, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx-bs*1.2f, by+bs+bt, 
               bx-bs*0.6f, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx, by+bs+bt, 
               bx-bs*0.6f, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      case 'x':
         drawPill(
               bx+bs, by-bs-bt, 
               bx-bs, by+bs+bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs, by+bs+bt, 
               bx-bs, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 1.9f*(bs+bt)+b_;
         break;

      case 'y':
         drawPill(
               bx, by-bs*0.42f, 
               bx-bs, by+bs+bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs, by+bs+bt, 
               bx-bs, by-bs*2.0f, 
               bt, color, 
               verts, colrs);
         *lineWidth += 1.8f*(bs+bt)+b_;
         break;

      case 'z':
         drawPill(
               bx+bs, by+bs+bt, 
               bx-bs, by+bs+bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs, by-bs-bt, 
               bx-bs, by-bs-bt, 
               bt, color, 
               verts, colrs);
         drawPill(
               bx+bs, by+bs+bt, 
               bx-bs, by-bs-bt, 
               bt, color, 
               verts, colrs);
         *lineWidth += 2.0f*(bs+bt)+b_;
         break;

      default:
         *lineWidth += b_;
         break;
   }
   return;
}

