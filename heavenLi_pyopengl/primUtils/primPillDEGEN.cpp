/*
 *  Defines an efficient pill for TRIANGLE_STRIP with degenerate vertices
 */

#include <math.h>
#include <vector>
using namespace std;

// Append to input vectors
int definePill(
      float px,                  /* x-coordinate of Point P */
      float py,                  /* y-coordinate of Point P */
      float qx,                  /* x-coordinate of Point Q */
      float qy,                  /* y-coordinate of Point Q */
      float radius,              /* Radius/Thickness of pill */
      char circleSegments,       /* Number of sides */
      float *pColor,             /* RGB values of P */
      float *qColor,             /* RGB values of Q */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   float slope, pR, pG, pB, qR, qG, qB, ang;

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
      ang = float(degToRad(180)+atan(slope));
   } else {
      slope = (py-qy)/(px-qx);
      ang = float(atan(slope));
   }

   pR = float(pColor[0]);
   pG = float(pColor[1]);
   pB = float(pColor[2]);
   qR = float(qColor[0]);
   qG = float(qColor[1]);
   qB = float(qColor[2]);

   char degSegment = 360 / circleSegments;
   degSegment /= 4.0f;

   // Prepend degenerate vertex iff not the first primitive in the vector
   if (verts.size() == 0) {
      /* pX */ verts.push_back(float(px + radius*cos(ang+0)));
      /* pY */ verts.push_back(float(py + radius*sin(ang+0)));
      /* pR */ colrs.push_back(pR);  /* G */ colrs.push_back(pG);  /* B */ colrs.push_back(pB);
   } else {
      /* pX */ verts.push_back(float(px + radius*cos(ang+0)));
      /* pY */ verts.push_back(float(py + radius*sin(ang+0)));
      /* pX */ verts.push_back(float(px + radius*cos(ang+0)));
      /* pY */ verts.push_back(float(py + radius*sin(ang+0)));
      /* pR */ colrs.push_back(pR);  /* G */ colrs.push_back(pG);  /* B */ colrs.push_back(pB);
      /* pR */ colrs.push_back(pR);  /* G */ colrs.push_back(pG);  /* B */ colrs.push_back(pB);
   }

   for (char i = 1; i < circleSegments; i++ ) {
      /* pX */ verts.push_back(float(px + radius*cos(ang+degToRad(i*degSegment))));
      /* pY */ verts.push_back(float(py + radius*sin(ang+degToRad(i*degSegment))));
      /* pX */ verts.push_back(float(px + radius*cos(ang+degToRad(-i*degSegment))));
      /* pY */ verts.push_back(float(py + radius*sin(ang+degToRad(-i*degSegment))));
      /* pR */ colrs.push_back(pR);  /* G */ colrs.push_back(pG);  /* B */ colrs.push_back(pB);
      /* pR */ colrs.push_back(pR);  /* G */ colrs.push_back(pG);  /* B */ colrs.push_back(pB);
   }

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
      ang = float(atan(slope));
   } else {
      slope = (py-qy)/(px-qx);
      ang = float(degToRad(180)+atan(slope));
   }
   for (char i = 1; i < circleSegments; i++ ) {
      /* qX */ verts.push_back(float(qx + radius*cos(ang+degToRad(-(circleSegments-i)*degSegment))));
      /* qY */ verts.push_back(float(qy + radius*sin(ang+degToRad(-(circleSegments-i)*degSegment))));
      /* qX */ verts.push_back(float(qx + radius*cos(ang+degToRad((circleSegments-i)*degSegment))));
      /* qY */ verts.push_back(float(qy + radius*sin(ang+degToRad((circleSegments-i)*degSegment))));
      /* qR */ colrs.push_back(qR);  /* G */ colrs.push_back(qG);  /* B */ colrs.push_back(qB);
      /* qR */ colrs.push_back(qR);  /* G */ colrs.push_back(qG);  /* B */ colrs.push_back(qB);
   }

   /* qX */ verts.push_back(float(qx + radius*cos(ang+degToRad(-0.0f))));
   /* qY */ verts.push_back(float(qy + radius*sin(ang+degToRad(-0.0f))));
   /* qX */ verts.push_back(float(qx + radius*cos(ang+degToRad(0.0f))));
   /* qY */ verts.push_back(float(qy + radius*sin(ang+degToRad(0.0f))));
   /* qR */ colrs.push_back(qR);  /* G */ colrs.push_back(qG);  /* B */ colrs.push_back(qB);
   /* qR */ colrs.push_back(qR);  /* G */ colrs.push_back(qG);  /* B */ colrs.push_back(qB);

   return verts.size()/2;
}

// Write to pre-allocated input array, updating vertices only 
