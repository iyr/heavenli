/*
 *  Defines a circular Arch for TRIANGLE_STRIP with degenerate vertices
 */

#include <math.h>
#include <vector>
using namespace std;

// Append vertices to input vectors
int defineBulb(
      float bx,                  /* X-Coordinate                                          */
      float by,                  /* Y-Coordinate                                          */
      float bs,                  /* Scale 2.0=spans display before GL scaling             */
      char  circleSegments,      /* Number of sides                                       */
      float *color,              /* Polygon Color                                         */
      float *detailColor,        /* Accent Color                                          */
      std::vector<float> &verts, /* Input Vector of x,y coordinates                       */
      std::vector<float> &colrs  /* Input Vector of r,g,b values                          */
      ){
   float thickness = 0.3f;
   circleSegments /= 3;
   defineEllipse(bx, by, bs*1.005f, bs*1.005f, circleSegments, color, verts, colrs);
   defineArch(bx, by, bs, bs, 305.0f, 270.0f, bs*thickness, circleSegments, detailColor, verts, colrs);
   circleSegments /= 4;
   defineQuad2pt(bx-bs*thickness, by-bs*2.25f, bx+bs*thickness, by-bs*2.25f-bs*thickness, detailColor, verts, colrs);
   defineArch(bx+bs*thickness, by-bs*2.25f, 0.0f, 0.0f, 270.0f, 0.0f, bs*thickness, circleSegments, detailColor, verts, colrs);
   defineArch(bx-bs*thickness, by-bs*2.25f, 0.0f, 0.0f, 180.0f, 270.0f, bs*thickness, circleSegments, detailColor, verts, colrs);
   defineArch(bx+(bs+bs*thickness*0.5f)*float(cos(degToRad(305.0f))), by+(bs+bs*thickness*0.5f)*float(sin(degToRad(305.0f))), 0.0f, 0.0f, 125.0f, 305.0f, bs*thickness*0.5f, circleSegments, detailColor, verts, colrs);
   defineQuad2pt(bx-bs*thickness, by-bs*2.25f, bx-bs*thickness*2.0f, by-bs*1.6f, detailColor, verts, colrs);
   defineQuad2pt(bx+bs*thickness, by-bs*2.25f, bx+bs*thickness*2.0f, by-bs*1.6f, detailColor, verts, colrs);
   defineArch(bx, by-bs*1.6f, bs*thickness, bs*thickness, 0.0f, 90.0f, bs*thickness, circleSegments, detailColor, verts, colrs);
   defineArch(bx-bs*thickness*1.5f, by-bs*1.6f, 0.0f, 0.0f, 0.0f, 180.0f, bs*thickness*0.5f, circleSegments, detailColor, verts, colrs);

   return verts.size()/2;
}

int updateBulbGeometry(
      float bx,               /* X-Coordinate                                    */
      float by,               /* Y-Coordinate                                    */
      float bs,               /* Scale 2.0=spans display before GL scaling       */
      char  circleSegments,   /* Number of sides                                 */
      int   index,            /* Index of where to start writing to input arrays */
      float *verts            /* Input Array of x,y coordinates                  */
      ){
   int vertIndex = index*2;   /* index (X, Y)   */
   float thickness = 0.3f;
   circleSegments /= 3;
   index = updatePrimEllipseGeometry(bx, by, bs*1.005f, bs*1.005f, circleSegments, index, verts);
   index = updateArchGeometry(bx, by, bs, bs, 305.0f, 270.0f, bs*thickness, circleSegments, index, verts);
   circleSegments /= 4;
   index = updateQuad2ptGeometry(bx-bs*thickness, by-bs*2.25f, bx+bs*thickness, by-bs*2.25f-bs*thickness, index, verts);
   index = updateArchGeometry(bx+bs*thickness, by-bs*2.25f, 0.0f, 0.0f, 270.0f, 0.0f, bs*thickness, circleSegments, index, verts);
   index = updateArchGeometry(bx-bs*thickness, by-bs*2.25f, 0.0f, 0.0f, 180.0f, 270.0f, bs*thickness, circleSegments, index, verts);
   index = updateArchGeometry(bx+(bs+bs*thickness*0.5f)*float(cos(degToRad(305.0f))), by+(bs+bs*thickness*0.5f)*float(sin(degToRad(305.0f))), 0.0f, 0.0f, 125.0f, 305.0f, bs*thickness*0.5f, circleSegments, index, verts);
   index = updateQuad2ptGeometry(bx-bs*thickness, by-bs*2.25f, bx-bs*thickness*2.0f, by-bs*1.6f, index, verts);
   index = updateQuad2ptGeometry(bx+bs*thickness, by-bs*2.25f, bx+bs*thickness*2.0f, by-bs*1.6f, index, verts);
   index = updateArchGeometry(bx, by-bs*1.6f, bs*thickness, bs*thickness, 0.0f, 90.0f, bs*thickness, circleSegments, index, verts);
   index = updateArchGeometry(bx-bs*thickness*1.5f, by-bs*1.6f, 0.0f, 0.0f, 0.0f, 180.0f, bs*thickness*0.5f, circleSegments, index, verts);
   return index;
}

int updateBulbColor(
      char  circleSegments,   /* Number of sides                                 */
      float *color,           /* Polygon Color                                   */
      float *detailColor,     /* Accent Color                                    */
      int   index,            /* Index of where to start writing to input arrays */
      float *colrs            /* Input Vector of r,g,b values                    */
      ){
   int colrIndex = index*4;   /* index (r, g, b) */
   circleSegments /= 3;
   index = updatePrimEllipseColor(circleSegments, color, index, colrs);
   index = updateArchColor(circleSegments, detailColor, index, colrs);
   circleSegments /= 4;
   index = updateQuadColor(detailColor, index, colrs);
   index = updateArchColor(circleSegments, detailColor, index, colrs);
   index = updateArchColor(circleSegments, detailColor, index, colrs);
   index = updateArchColor(circleSegments, detailColor, index, colrs);
   index = updateQuadColor(detailColor, index, colrs);
   index = updateQuadColor(detailColor, index, colrs);
   index = updateArchColor(circleSegments, detailColor, index, colrs);
   index = updateArchColor(circleSegments, detailColor, index, colrs);
   return index;
}
