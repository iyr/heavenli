/*
 *  Defines an efficient pill for TRIANGLE_STRIP with degenerate vertices
 */

//#include <math.h>
//#include <vector>
//using namespace std;

// Append to input vectors
GLuint definePill(
      float px,                  // x-coordinate of Point P
      float py,                  // y-coordinate of Point P
      float qx,                  // x-coordinate of Point Q
      float qy,                  // y-coordinate of Point Q
      float radius,              // Radius/Thickness of pill
      GLuint circleSegments,     // Number of sides
      float *pColor,             // RGB values of P
      float *qColor,             // RGB values of Q
      std::vector<float> &verts, // Input Vector of x,y coordinates
      std::vector<float> &colrs  // Input Vector of r,g,b values
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
      verts.push_back(float(px + radius*cosf(ang)));   // pX
      verts.push_back(float(py + radius*sinf(ang)));   // pY
      colrs.push_back( pR );
      colrs.push_back( pG );
      colrs.push_back( pB );
      colrs.push_back( pA );
   } else {
      verts.push_back(float(px + radius*cosf(ang)));   // pX
      verts.push_back(float(py + radius*sinf(ang)));   // pY
      colrs.push_back( pR );
      colrs.push_back( pG );
      colrs.push_back( pB );
      colrs.push_back( pA );

      verts.push_back(float(px + radius*cosf(ang)));   // pX
      verts.push_back(float(py + radius*sinf(ang)));   // pY
      colrs.push_back( pR );
      colrs.push_back( pG );
      colrs.push_back( pB );
      colrs.push_back( pA );
   }

   for (GLuint i = 1; i < circleSegments/2; i++ ) {
      verts.push_back(float(px + radius*cosf(ang+degToRad(i*degSegment)))); // pX
      verts.push_back(float(py + radius*sinf(ang+degToRad(i*degSegment)))); // pY
      colrs.push_back( pR );
      colrs.push_back( pG );
      colrs.push_back( pB );
      colrs.push_back( pA );

      verts.push_back(float(px + radius*cosf(ang+degToRad(-(GLint)i*degSegment))));  // pX
      verts.push_back(float(py + radius*sinf(ang+degToRad(-(GLint)i*degSegment))));  // pY
      colrs.push_back( pR );
      colrs.push_back( pG );
      colrs.push_back( pB );
      colrs.push_back( pA );
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
      verts.push_back(float(qx + radius*cosf(ang+degToRad(-(GLint)(circleSegments/2-i)*degSegment)))); // qX
      verts.push_back(float(qy + radius*sinf(ang+degToRad(-(GLint)(circleSegments/2-i)*degSegment)))); // qY
      colrs.push_back( qR );
      colrs.push_back( qG );
      colrs.push_back( qB );
      colrs.push_back( qA );

      verts.push_back(float(qx + radius*cosf(ang+degToRad((circleSegments/2-i)*degSegment))));   // qX
      verts.push_back(float(qy + radius*sinf(ang+degToRad((circleSegments/2-i)*degSegment))));   // qY
      colrs.push_back( qR );
      colrs.push_back( qG );
      colrs.push_back( qB );
      colrs.push_back( qA );
   }

   verts.push_back(float(qx + radius*cosf(ang+degToRad(0.0f))));   // qX
   verts.push_back(float(qy + radius*sinf(ang+degToRad(0.0f))));   // qY
   colrs.push_back( qR );
   colrs.push_back( qG );
   colrs.push_back( qB );
   colrs.push_back( qA );

   verts.push_back(float(qx + radius*cosf(ang+degToRad(0.0f))));   // qX
   verts.push_back(float(qy + radius*sinf(ang+degToRad(0.0f))));   // qY
   colrs.push_back( qR );
   colrs.push_back( qG );
   colrs.push_back( qB );
   colrs.push_back( qA );

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
      verts[vertIndex++] = px + radius*cosf(ang);   // pX
      verts[vertIndex++] = py + radius*sinf(ang);   // pY
   } else {
      verts[vertIndex++] = px + radius*cosf(ang);   // pX
      verts[vertIndex++] = py + radius*sinf(ang);   // pY
      verts[vertIndex++] = px + radius*cosf(ang);   // pX
      verts[vertIndex++] = py + radius*sinf(ang);   // pY
   }

   for (GLuint i = 1; i < circleSegments/2; i++ ) {
      verts[vertIndex++] = px + radius*cosf(ang+degToRad(i*degSegment)); // pX
      verts[vertIndex++] = py + radius*sinf(ang+degToRad(i*degSegment)); // pY
      verts[vertIndex++] = px + radius*cosf(ang+degToRad(-(GLint)i*degSegment));  // pX
      verts[vertIndex++] = py + radius*sinf(ang+degToRad(-(GLint)i*degSegment));  // pY
   }

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
      ang = float(atan(slope));
   } else {
      slope = (py-qy)/(px-qx);
      ang = float(degToRad(180)+atan(slope));
   }
   for (GLuint i = 1; i < circleSegments/2; i++ ) {
      verts[vertIndex++] = qx + radius*cosf(ang+degToRad(-(GLint)(circleSegments/2-(GLint)i)*degSegment));   // qX
      verts[vertIndex++] = qy + radius*sinf(ang+degToRad(-(GLint)(circleSegments/2-(GLint)i)*degSegment));   // qY
      verts[vertIndex++] = qx + radius*cosf(ang+degToRad((circleSegments/2-i)*degSegment));   // qX
      verts[vertIndex++] = qy + radius*sinf(ang+degToRad((circleSegments/2-i)*degSegment));   // qY
   }

   verts[vertIndex++] = qx + radius*cosf(ang+degToRad(0.0f));   // qX
   verts[vertIndex++] = qy + radius*sinf(ang+degToRad(0.0f));   // qY
   verts[vertIndex++] = qx + radius*cosf(ang+degToRad(0.0f));   // qX
   verts[vertIndex++] = qy + radius*sinf(ang+degToRad(0.0f));   // qY

   return vertIndex/2;
}

GLuint updatePillColor(
      GLuint circleSegments,  // Number of sides
      float *pColor,          // RGB values of P
      float *qColor,          // RGB values of Q
      GLuint index,           // Index of where to start writing
      float *colrs            // Input Vector of r,g,b values
      ){
   float pR, pG, pB, pA, qR, qG, qB, qA;
   GLuint colrIndex = index*4;

   pR = pColor[0];
   pG = pColor[1];
   pB = pColor[2];
   pA = pColor[3];

   qR = qColor[0];
   qG = qColor[1];
   qB = qColor[2];
   qA = qColor[3];

   // Prepend degenerate vertex iff not the first primitive in the vector
   if (colrIndex == 0) {
      colrs[colrIndex+0] = pR;
      colrs[colrIndex+1] = pG;
      colrs[colrIndex+2] = pB;
      colrs[colrIndex+3] = pA;

      colrIndex += 4;
   } else {
      colrs[colrIndex+4] = pR;
      colrs[colrIndex+5] = pG;
      colrs[colrIndex+6] = pB;
      colrs[colrIndex+7] = pA;

      colrs[colrIndex+8] = pR;
      colrs[colrIndex+9] = pG;
      colrs[colrIndex+10] = pB;
      colrs[colrIndex+11] = pA;
      colrIndex += 8;
   }

   for (GLuint i = 1; i < circleSegments/2; i++ ) {
      colrs[colrIndex+0] = pR;
      colrs[colrIndex+1] = pG;
      colrs[colrIndex+2] = pB;
      colrs[colrIndex+3] = pA;

      colrs[colrIndex+4] = pR;
      colrs[colrIndex+5] = pG;
      colrs[colrIndex+6] = pB;
      colrs[colrIndex+7] = pA;
      colrIndex += 8;
   }

   for (GLuint i = 1; i < circleSegments/2; i++ ) {
      colrs[colrIndex+0] = qR;
      colrs[colrIndex+1] = qG;
      colrs[colrIndex+2] = qB;
      colrs[colrIndex+3] = qA;

      colrs[colrIndex+4] = qR;
      colrs[colrIndex+5] = qG;
      colrs[colrIndex+6] = qB;
      colrs[colrIndex+7] = qA;
      colrIndex += 8;
   }

   colrs[colrIndex+0] = qR;
   colrs[colrIndex+1] = qG;
   colrs[colrIndex+2] = qB;
   colrs[colrIndex+3] = qA;

   colrs[colrIndex+4] = qR;
   colrs[colrIndex+5] = qG;
   colrs[colrIndex+6] = qB;
   colrs[colrIndex+7] = qA;
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
