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
   float slope, pR, pG, pB, pA, qR, qG, qB, qA, ang;

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
   pA = float(pColor[3]);

   qR = float(qColor[0]);
   qG = float(qColor[1]);
   qB = float(qColor[2]);
   qA = float(pColor[3]);

   char degSegment = 360 / circleSegments;
   degSegment /= 2;

   // Prepend degenerate vertex iff not the first primitive in the vector
   if (verts.size() == 0) {
      /* pX */ verts.push_back(float(px + radius*cos(ang)));
      /* pY */ verts.push_back(float(py + radius*sin(ang)));
      /* pR */ colrs.push_back(pR);  
      /* pG */ colrs.push_back(pG);  
      /* pB */ colrs.push_back(pB);
      /* pA */ colrs.push_back(pA);
   } else {
      /* pX */ verts.push_back(float(px + radius*cos(ang)));
      /* pY */ verts.push_back(float(py + radius*sin(ang)));
      /* pR */ colrs.push_back(pR);  
      /* pG */ colrs.push_back(pG);  
      /* pB */ colrs.push_back(pB);
      /* pA */ colrs.push_back(pA);

      /* pX */ verts.push_back(float(px + radius*cos(ang)));
      /* pY */ verts.push_back(float(py + radius*sin(ang)));
      /* pR */ colrs.push_back(pR);  
      /* pG */ colrs.push_back(pG);  
      /* pB */ colrs.push_back(pB);
      /* pA */ colrs.push_back(pA);
   }

   for (char i = 1; i < circleSegments/2; i++ ) {
      /* pX */ verts.push_back(float(px + radius*cos(ang+degToRad(i*degSegment))));
      /* pY */ verts.push_back(float(py + radius*sin(ang+degToRad(i*degSegment))));
      /* pR */ colrs.push_back(pR);  
      /* pG */ colrs.push_back(pG);  
      /* pB */ colrs.push_back(pB);
      /* pA */ colrs.push_back(pA);

      /* pX */ verts.push_back(float(px + radius*cos(ang+degToRad(-i*degSegment))));
      /* pY */ verts.push_back(float(py + radius*sin(ang+degToRad(-i*degSegment))));
      /* pR */ colrs.push_back(pR);  
      /* pG */ colrs.push_back(pG);  
      /* pB */ colrs.push_back(pB);
      /* pA */ colrs.push_back(pA);
   }

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
      ang = float(atan(slope));
   } else {
      slope = (py-qy)/(px-qx);
      ang = float(degToRad(180)+atan(slope));
   }
   for (char i = 1; i < circleSegments/2; i++ ) {
      /* qX */ verts.push_back(float(qx + radius*cos(ang+degToRad(-(circleSegments/2-i)*degSegment))));
      /* qY */ verts.push_back(float(qy + radius*sin(ang+degToRad(-(circleSegments/2-i)*degSegment))));
      /* qR */ colrs.push_back(qR);  
      /* qG */ colrs.push_back(qG);  
      /* qB */ colrs.push_back(qB);
      /* qA */ colrs.push_back(qA);

      /* qX */ verts.push_back(float(qx + radius*cos(ang+degToRad((circleSegments/2-i)*degSegment))));
      /* qY */ verts.push_back(float(qy + radius*sin(ang+degToRad((circleSegments/2-i)*degSegment))));
      /* qR */ colrs.push_back(qR);  
      /* qG */ colrs.push_back(qG);  
      /* qB */ colrs.push_back(qB);
      /* qA */ colrs.push_back(qA);
   }

   /* qX */ verts.push_back(float(qx + radius*cos(ang+degToRad(0.0f))));
   /* qY */ verts.push_back(float(qy + radius*sin(ang+degToRad(0.0f))));
   /* qR */ colrs.push_back(qR);  
   /* qG */ colrs.push_back(qG);  
   /* qB */ colrs.push_back(qB);
   /* qA */ colrs.push_back(qA);

   /* qX */ verts.push_back(float(qx + radius*cos(ang+degToRad(0.0f))));
   /* qY */ verts.push_back(float(qy + radius*sin(ang+degToRad(0.0f))));
   /* qR */ colrs.push_back(qR);  
   /* qG */ colrs.push_back(qG);  
   /* qB */ colrs.push_back(qB);
   /* qA */ colrs.push_back(qA);

   return verts.size()/2;
}

/* useful overload */
int definePill(
      float px,                  /* x-coordinate of Point P */
      float py,                  /* y-coordinate of Point P */
      float qx,                  /* x-coordinate of Point Q */
      float qy,                  /* y-coordinate of Point Q */
      float radius,              /* Radius/Thickness of pill */
      char circleSegments,       /* Number of sides */
      float *Color,              /* RGB values of Pill */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   return definePill(px, py, qx, qy, radius, circleSegments, Color, Color, verts, colrs);
}

// Write to pre-allocated input array, updating vertices only 
int updatePillGeometry(
      float px,            /* x-coordinate of Point P */
      float py,            /* y-coordinate of Point P */
      float qx,            /* x-coordinate of Point Q */
      float qy,            /* y-coordinate of Point Q */
      float radius,        /* Radius/Thickness of pill */
      char circleSegments, /* Number of sides */
      int index,           /* Index of where to start writing */
      float *verts         /* Input Vector of x,y coordinates */
      ){
   float slope, ang;
   int vertIndex = index*2;

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
      ang = float(degToRad(180)+atan(slope));
   } else {
      slope = (py-qy)/(px-qx);
      ang = float(atan(slope));
   }

   char degSegment = 360 / circleSegments;
   degSegment /= 2;

   // Prepend degenerate vertex iff not the first primitive in the vector
   if (vertIndex == 0) {
      /* pX */ verts[vertIndex++] = (float(px + radius*cos(ang)));
      /* pY */ verts[vertIndex++] = (float(py + radius*sin(ang)));
   } else {
      /* pX */ verts[vertIndex++] = (float(px + radius*cos(ang)));
      /* pY */ verts[vertIndex++] = (float(py + radius*sin(ang)));
      /* pX */ verts[vertIndex++] = (float(px + radius*cos(ang)));
      /* pY */ verts[vertIndex++] = (float(py + radius*sin(ang)));
   }

   for (char i = 1; i < circleSegments/2; i++ ) {
      /* pX */ verts[vertIndex++] = (float(px + radius*cos(ang+degToRad(i*degSegment))));
      /* pY */ verts[vertIndex++] = (float(py + radius*sin(ang+degToRad(i*degSegment))));
      /* pX */ verts[vertIndex++] = (float(px + radius*cos(ang+degToRad(-i*degSegment))));
      /* pY */ verts[vertIndex++] = (float(py + radius*sin(ang+degToRad(-i*degSegment))));
   }

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
      ang = float(atan(slope));
   } else {
      slope = (py-qy)/(px-qx);
      ang = float(degToRad(180)+atan(slope));
   }
   for (char i = 1; i < circleSegments/2; i++ ) {
      /* qX */ verts[vertIndex++] = (float(qx + radius*cos(ang+degToRad(-(circleSegments/2-i)*degSegment))));
      /* qY */ verts[vertIndex++] = (float(qy + radius*sin(ang+degToRad(-(circleSegments/2-i)*degSegment))));
      /* qX */ verts[vertIndex++] = (float(qx + radius*cos(ang+degToRad((circleSegments/2-i)*degSegment))));
      /* qY */ verts[vertIndex++] = (float(qy + radius*sin(ang+degToRad((circleSegments/2-i)*degSegment))));
   }

   /* qX */ verts[vertIndex++] = (float(qx + radius*cos(ang+degToRad(0.0f))));
   /* qY */ verts[vertIndex++] = (float(qy + radius*sin(ang+degToRad(0.0f))));
   /* qX */ verts[vertIndex++] = (float(qx + radius*cos(ang+degToRad(0.0f))));
   /* qY */ verts[vertIndex++] = (float(qy + radius*sin(ang+degToRad(0.0f))));

   return vertIndex/2;
}

int updatePillColor(
      char circleSegments, /* Number of sides */
      float *pColor,       /* RGB values of P */
      float *qColor,       /* RGB values of Q */
      int index,           /* Index of where to start writing */
      float *colrs         /* Input Vector of r,g,b values */
      ){
   float pR, pG, pB, pA, qR, qG, qB, qA;
   int colrIndex = index*4;

   pR = float(pColor[0]);
   pG = float(pColor[1]);
   pB = float(pColor[2]);
   pA = float(pColor[3]);

   qR = float(qColor[0]);
   qG = float(qColor[1]);
   qB = float(qColor[2]);
   qA = float(pColor[3]);

   // Prepend degenerate vertex iff not the first primitive in the vector
   if (colrIndex == 0) {
      /* pR */ colrs[colrIndex++] = (pR);  
      /* pG */ colrs[colrIndex++] = (pG);  
      /* pB */ colrs[colrIndex++] = (pB);
      /* pA */ colrs[colrIndex++] = (pA);
   } else {
      /* pR */ colrs[colrIndex++] = (pR);  
      /* pG */ colrs[colrIndex++] = (pG);  
      /* pB */ colrs[colrIndex++] = (pB);
      /* pA */ colrs[colrIndex++] = (pA);

      /* pR */ colrs[colrIndex++] = (pR);  
      /* pG */ colrs[colrIndex++] = (pG);  
      /* pB */ colrs[colrIndex++] = (pB);
      /* pA */ colrs[colrIndex++] = (pA);
   }

   for (char i = 1; i < circleSegments/2; i++ ) {
      /* pR */ colrs[colrIndex++] = (pR);  
      /* pG */ colrs[colrIndex++] = (pG);  
      /* pB */ colrs[colrIndex++] = (pB);
      /* pA */ colrs[colrIndex++] = (pA);

      /* pR */ colrs[colrIndex++] = (pR);  
      /* pG */ colrs[colrIndex++] = (pG);  
      /* pB */ colrs[colrIndex++] = (pB);
      /* pA */ colrs[colrIndex++] = (pA);
   }

   for (char i = 1; i < circleSegments/2; i++ ) {
      /* qR */ colrs[colrIndex++] = (qR);  
      /* qG */ colrs[colrIndex++] = (qG);  
      /* qB */ colrs[colrIndex++] = (qB);
      /* qA */ colrs[colrIndex++] = (qA);

      /* qR */ colrs[colrIndex++] = (qR);  
      /* qG */ colrs[colrIndex++] = (qG);  
      /* qB */ colrs[colrIndex++] = (qB);
      /* qA */ colrs[colrIndex++] = (qA);
   }

   /* qR */ colrs[colrIndex++] = (qR);  
   /* qG */ colrs[colrIndex++] = (qG);  
   /* qB */ colrs[colrIndex++] = (qB);
   /* qA */ colrs[colrIndex++] = (qA);

   /* qR */ colrs[colrIndex++] = (qR);  
   /* qG */ colrs[colrIndex++] = (qG);  
   /* qB */ colrs[colrIndex++] = (qB);
   /* qA */ colrs[colrIndex++] = (qA);

   return colrIndex/4;
}

/* useful overload */
int updatePillColor(
      char circleSegments, /* Number of sides */
      float *Color,        /* RGB values of Pill */
      int index,           /* Index of where to start writing */
      float *colrs         /* Input Vector of r,g,b values */
      ){
   return updatePillColor(circleSegments, Color, Color, index, colrs);
}
