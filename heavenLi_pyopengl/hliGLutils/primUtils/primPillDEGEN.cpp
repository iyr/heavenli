/*
 *  Defines an efficient pill for TRIANGLE_STRIP with degenerate vertices
 */

#include <math.h>
#include <vector>
using namespace std;

// Append to input vectors
GLuint definePill(
      float px,                  /* x-coordinate of Point P */
      float py,                  /* y-coordinate of Point P */
      float qx,                  /* x-coordinate of Point Q */
      float qy,                  /* y-coordinate of Point Q */
      float radius,              /* Radius/Thickness of pill */
      GLuint circleSegments,       /* Number of sides */
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

   if ( slope != slope ) slope = 0.0f;

   pR = pColor[0];
   pG = pColor[1];
   pB = pColor[2];
   pA = pColor[3];

   qR = qColor[0];
   qG = qColor[1];
   qB = qColor[2];
   qA = pColor[3];

   GLint degSegment = 360 / circleSegments;
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

   for (GLuint i = 1; i < circleSegments/2; i++ ) {
      /* pX */ verts.push_back(float(px + radius*cos(ang+degToRad(i*degSegment))));
      /* pY */ verts.push_back(float(py + radius*sin(ang+degToRad(i*degSegment))));
      /* pR */ colrs.push_back(pR);  
      /* pG */ colrs.push_back(pG);  
      /* pB */ colrs.push_back(pB);
      /* pA */ colrs.push_back(pA);

      /* pX */ verts.push_back(float(px + radius*cos(ang+degToRad(-(GLint)i*degSegment))));
      /* pY */ verts.push_back(float(py + radius*sin(ang+degToRad(-(GLint)i*degSegment))));
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
   if ( slope != slope ) slope = 0.0f;
   for (GLuint i = 1; i < circleSegments/2; i++ ) {
      /* qX */ verts.push_back(float(qx + radius*cos(ang+degToRad(-(GLint)(circleSegments/2-i)*degSegment))));
      /* qY */ verts.push_back(float(qy + radius*sin(ang+degToRad(-(GLint)(circleSegments/2-i)*degSegment))));
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
GLuint definePill(
      float px,                  // x-coordinate of Point P
      float py,                  // y-coordinate of Point P
      float qx,                  // x-coordinate of Point Q
      float qy,                  // y-coordinate of Point Q
      float radius,              // Radius/Thickness of pill
      GLuint circleSegments,     // Number of sides
      float *Color,              // RGB values of Pill
      std::vector<float> &verts, // Input Vector of x,y coordinates
      std::vector<float> &colrs  // Input Vector of r,g,b values
      ){
   return definePill(px, py, qx, qy, radius, circleSegments, Color, Color, verts, colrs);
}

/*
 * Used for updating a position index without recalculation or needless memory reads/writes
 */
GLuint updatePillIndex(
      GLuint circleSegments, /* Number of sides */
      GLuint numElements,     /* Number of elements per vertex */
      GLuint index            /* Index of where to start writing */
      ){
   GLuint subIndex = index*numElements;
   GLint lim = circleSegments/2;

   // Prepend degenerate vertex iff not the first primitive in the vector
   if (subIndex == 0) {
      subIndex += numElements;
   } else {
      subIndex += numElements*2;
   }

   for (GLint i = 1; i < lim; i++ ) {
      subIndex += numElements*2;
   }

   for (GLint i = 1; i < lim; i++ ) {
      subIndex += numElements*2;
   }

   subIndex += numElements*2;

   return subIndex/numElements;
}

/*
 * Write to pre-allocated input array, updating vertices only 
 */
GLuint updatePillGeometry(
      float px,            /* x-coordinate of Point P */
      float py,            /* y-coordinate of Point P */
      float qx,            /* x-coordinate of Point Q */
      float qy,            /* y-coordinate of Point Q */
      float radius,        /* Radius/Thickness of pill */
      GLuint circleSegments, /* Number of sides */
      GLuint index,           /* Index of where to start writing */
      float *verts         /* Input Vector of x,y coordinates */
      ){
   float slope, ang;
   GLuint vertIndex = index*2;

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
      ang = float(degToRad(180)+atan(slope));
   } else {
      slope = (py-qy)/(px-qx);
      ang = float(atan(slope));
   }

   GLint degSegment = 360 / circleSegments;
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

   for (GLuint i = 1; i < circleSegments/2; i++ ) {
      /* pX */ verts[vertIndex++] = (float(px + radius*cos(ang+degToRad(i*degSegment))));
      /* pY */ verts[vertIndex++] = (float(py + radius*sin(ang+degToRad(i*degSegment))));
      /* pX */ verts[vertIndex++] = (float(px + radius*cos(ang+degToRad(-(GLint)i*degSegment))));
      /* pY */ verts[vertIndex++] = (float(py + radius*sin(ang+degToRad(-(GLint)i*degSegment))));
   }

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
      ang = float(atan(slope));
   } else {
      slope = (py-qy)/(px-qx);
      ang = float(degToRad(180)+atan(slope));
   }
   for (GLuint i = 1; i < circleSegments/2; i++ ) {
      /* qX */ verts[vertIndex++] = (float(qx + radius*cos(ang+degToRad(-(GLint)(circleSegments/2-(GLint)i)*degSegment))));
      /* qY */ verts[vertIndex++] = (float(qy + radius*sin(ang+degToRad(-(GLint)(circleSegments/2-(GLint)i)*degSegment))));
      /* qX */ verts[vertIndex++] = (float(qx + radius*cos(ang+degToRad((circleSegments/2-i)*degSegment))));
      /* qY */ verts[vertIndex++] = (float(qy + radius*sin(ang+degToRad((circleSegments/2-i)*degSegment))));
   }

   /* qX */ verts[vertIndex++] = (float(qx + radius*cos(ang+degToRad(0.0f))));
   /* qY */ verts[vertIndex++] = (float(qy + radius*sin(ang+degToRad(0.0f))));
   /* qX */ verts[vertIndex++] = (float(qx + radius*cos(ang+degToRad(0.0f))));
   /* qY */ verts[vertIndex++] = (float(qy + radius*sin(ang+degToRad(0.0f))));

   return vertIndex/2;
}

GLuint updatePillColor(
      GLuint circleSegments, /* Number of sides */
      float *pColor,       /* RGB values of P */
      float *qColor,       /* RGB values of Q */
      GLuint index,           /* Index of where to start writing */
      float *colrs         /* Input Vector of r,g,b values */
      ){
   float pR, pG, pB, pA, qR, qG, qB, qA;
   GLuint colrIndex = index*4;

   pR = float(pColor[0]);
   pG = float(pColor[1]);
   pB = float(pColor[2]);
   pA = float(pColor[3]);

   qR = float(qColor[0]);
   qG = float(qColor[1]);
   qB = float(qColor[2]);
   qA = float(qColor[3]);

   // Prepend degenerate vertex iff not the first primitive in the vector
   if (colrIndex == 0) {
      /* pR */ colrs[colrIndex+0] = pR;
      /* pG */ colrs[colrIndex+1] = pG;
      /* pB */ colrs[colrIndex+2] = pB;
      /* pA */ colrs[colrIndex+3] = pA;

      colrIndex += 4;
   } else {
      /* pR */ colrs[colrIndex+4] = pR;
      /* pG */ colrs[colrIndex+5] = pG;
      /* pB */ colrs[colrIndex+6] = pB;
      /* pA */ colrs[colrIndex+7] = pA;

      /* pR */ colrs[colrIndex+8] = pR;
      /* pG */ colrs[colrIndex+9] = pG;
      /* pB */ colrs[colrIndex+10] = pB;
      /* pA */ colrs[colrIndex+11] = pA;
      colrIndex += 8;
   }

   for (GLuint i = 1; i < circleSegments/2; i++ ) {
      /* pR */ colrs[colrIndex+0] = pR;
      /* pG */ colrs[colrIndex+1] = pG;
      /* pB */ colrs[colrIndex+2] = pB;
      /* pA */ colrs[colrIndex+3] = pA;

      /* pR */ colrs[colrIndex+4] = pR;
      /* pG */ colrs[colrIndex+5] = pG;
      /* pB */ colrs[colrIndex+6] = pB;
      /* pA */ colrs[colrIndex+7] = pA;
      colrIndex += 8;
   }

   for (GLuint i = 1; i < circleSegments/2; i++ ) {
      /* qR */ colrs[colrIndex+0] = qR;
      /* qG */ colrs[colrIndex+1] = qG;
      /* qB */ colrs[colrIndex+2] = qB;
      /* qA */ colrs[colrIndex+3] = qA;

      /* qR */ colrs[colrIndex+4] = qR;
      /* qG */ colrs[colrIndex+5] = qG;
      /* qB */ colrs[colrIndex+6] = qB;
      /* qA */ colrs[colrIndex+7] = qA;
      colrIndex += 8;
   }

   /* qR */ colrs[colrIndex+0] = qR;
   /* qG */ colrs[colrIndex+1] = qG;
   /* qB */ colrs[colrIndex+2] = qB;
   /* qA */ colrs[colrIndex+3] = qA;

   /* qR */ colrs[colrIndex+4] = qR;
   /* qG */ colrs[colrIndex+5] = qG;
   /* qB */ colrs[colrIndex+6] = qB;
   /* qA */ colrs[colrIndex+7] = qA;
   colrIndex += 8;

   return colrIndex/4;
}

/* useful overload */
GLuint updatePillColor(
      GLint circleSegments, /* Number of sides */
      float *Color,        /* RGB values of Pill */
      GLuint index,           /* Index of where to start writing */
      float *colrs         /* Input Vector of r,g,b values */
      ){
   return updatePillColor(circleSegments, Color, Color, index, colrs);
}
