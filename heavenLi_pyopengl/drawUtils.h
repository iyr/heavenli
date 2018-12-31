#include <math.h>
#define degToRad(angleInDegrees) ((angleInDegrees) * 3.1415926535 / 180.0)

// Writing to pre-allocated input arrays, update pill shape between two points
int drawPill(
      float px, 
      float py, 
      float qx, 
      float qy, 
      float radius, 
      int index,
      double *pColor, 
      double *qColor,
      float *verts,
      float *colrs
      ){
   int vertIndex = index*2;   /* index (x, y) */
   int colrIndex = index*3;   /* index (r, g, b) */
   float rx, ry, slope, tma;

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
   } else {
      slope = (py-qy)/(px-qx);
   }
   float ang = float(degToRad(90)+atan(slope));

   rx = radius*cos(ang);
   ry = radius*sin(ang);

   // Draw Pill Body, (Rectangle)
   verts[vertIndex++] = float(px+rx);
   verts[vertIndex++] = float(py+ry);
   colrs[colrIndex++] = float(pColor[0]);
   colrs[colrIndex++] = float(pColor[1]);
   colrs[colrIndex++] = float(pColor[2]);

   verts[vertIndex++] = float(px-rx);
   verts[vertIndex++] = float(py-ry);
   colrs[colrIndex++] = float(pColor[0]);
   colrs[colrIndex++] = float(pColor[1]);
   colrs[colrIndex++] = float(pColor[2]);

   verts[vertIndex++] = float(qx-rx);
   verts[vertIndex++] = float(qy-ry);
   colrs[colrIndex++] = float(qColor[0]);
   colrs[colrIndex++] = float(qColor[1]);
   colrs[colrIndex++] = float(qColor[2]);

   verts[vertIndex++] = float(qx-rx);
   verts[vertIndex++] = float(qy-ry);
   colrs[colrIndex++] = float(qColor[0]);
   colrs[colrIndex++] = float(qColor[1]);
   colrs[colrIndex++] = float(qColor[2]);

   verts[vertIndex++] = float(qx+rx);
   verts[vertIndex++] = float(qy+ry);
   colrs[colrIndex++] = float(qColor[0]);
   colrs[colrIndex++] = float(qColor[1]);
   colrs[colrIndex++] = float(qColor[2]);

   verts[vertIndex++] = float(px+rx);
   verts[vertIndex++] = float(py+ry);
   colrs[colrIndex++] = float(pColor[0]);
   colrs[colrIndex++] = float(pColor[1]);
   colrs[colrIndex++] = float(pColor[2]);


#     pragma omp parallel for
   for (int i = 0; i < 15; i++) {
      // Draw endcap for point P
      if (qx >= px)
         tma = ang + float(degToRad(+i*12.0));
      else
         tma = ang + float(degToRad(-i*12.0));
      rx = radius*cos(tma);
      ry = radius*sin(tma);
         
      verts[vertIndex++] = float(px);
      verts[vertIndex++] = float(py);
      colrs[colrIndex++] = float(pColor[0]);
      colrs[colrIndex++] = float(pColor[1]);
      colrs[colrIndex++] = float(pColor[2]);

      verts[vertIndex++] = float(px+rx);
      verts[vertIndex++] = float(py+ry);
      colrs[colrIndex++] = float(pColor[0]);
      colrs[colrIndex++] = float(pColor[1]);
      colrs[colrIndex++] = float(pColor[2]);

      if (qx >= px)
         tma = ang + float(degToRad(+(i+1)*12.0));
      else 
         tma = ang + float(degToRad(-(i+1)*12.0));
      rx = radius*cos(tma);
      ry = radius*sin(tma);
         
      verts[vertIndex++] = float(px+rx);
      verts[vertIndex++] = float(py+ry);
      colrs[colrIndex++] = float(pColor[0]);
      colrs[colrIndex++] = float(pColor[1]);
      colrs[colrIndex++] = float(pColor[2]);

      // Draw endcap for point Q
      if (qx >= px)
         tma = ang + float(degToRad(+i*12.0));
      else
         tma = ang + float(degToRad(-i*12.0));
      rx = radius*cos(tma);
      ry = radius*sin(tma);
         
      verts[vertIndex++] = float(qx);
      verts[vertIndex++] = float(qy);
      colrs[colrIndex++] = float(qColor[0]);
      colrs[colrIndex++] = float(qColor[1]);
      colrs[colrIndex++] = float(qColor[2]);

      verts[vertIndex++] = float(qx-rx);
      verts[vertIndex++] = float(qy-ry);
      colrs[colrIndex++] = float(qColor[0]);
      colrs[colrIndex++] = float(qColor[1]);
      colrs[colrIndex++] = float(qColor[2]);

      if (qx >= px)
         tma = ang + float(degToRad(+(i+1)*12.0));
      else
         tma = ang + float(degToRad(-(i+1)*12.0));
      rx = radius*cos(tma);
      ry = radius*sin(tma);
         
      verts[vertIndex++] = float(qx-rx);
      verts[vertIndex++] = float(qy-ry);
      colrs[colrIndex++] = float(qColor[0]);
      colrs[colrIndex++] = float(qColor[1]);
      colrs[colrIndex++] = float(qColor[2]);
   }
   return (vertIndex)/2;
}

// Appending to input arrays, define vertices for a pill shape between two points
int drawPill(
      float px, 
      float py, 
      float qx, 
      float qy, 
      float radius, 
      double *pColor, 
      double *qColor,
      std::vector<float> &verts,
      std::vector<float> &colrs
      ){
   float slope;

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
   } else {
      slope = (py-qy)/(px-qx);
   }
   float rx;
   float ry;
   float ang = float(degToRad(90)+atan(slope));
   float tma;

   rx = radius*cos(ang);
   ry = radius*sin(ang);

   // Draw Pill Body, (Rectangle)
   verts.push_back(float(px+rx));
   verts.push_back(float(py+ry));
   colrs.push_back(float(pColor[0]));
   colrs.push_back(float(pColor[1]));
   colrs.push_back(float(pColor[2]));
   verts.push_back(float(px-rx));
   verts.push_back(float(py-ry));
   colrs.push_back(float(pColor[0]));
   colrs.push_back(float(pColor[1]));
   colrs.push_back(float(pColor[2]));
   verts.push_back(float(qx-rx));
   verts.push_back(float(qy-ry));
   colrs.push_back(float(qColor[0]));
   colrs.push_back(float(qColor[1]));
   colrs.push_back(float(qColor[2]));

   verts.push_back(float(qx-rx));
   verts.push_back(float(qy-ry));
   colrs.push_back(float(qColor[0]));
   colrs.push_back(float(qColor[1]));
   colrs.push_back(float(qColor[2]));
   verts.push_back(float(qx+rx));
   verts.push_back(float(qy+ry));
   colrs.push_back(float(qColor[0]));
   colrs.push_back(float(qColor[1]));
   colrs.push_back(float(qColor[2]));
   verts.push_back(float(px+rx));
   verts.push_back(float(py+ry));
   colrs.push_back(float(pColor[0]));
   colrs.push_back(float(pColor[1]));
   colrs.push_back(float(pColor[2]));


#     pragma omp parallel for
   for (int i = 0; i < 15; i++) {
      // Draw endcap for point P
      if (qx >= px)
         tma = ang + float(degToRad(+i*12.0));
      else
         tma = ang + float(degToRad(-i*12.0));
      rx = radius*cos(tma);
      ry = radius*sin(tma);
         
      verts.push_back(float(px));
      verts.push_back(float(py));
      colrs.push_back(float(pColor[0]));
      colrs.push_back(float(pColor[1]));
      colrs.push_back(float(pColor[2]));

      verts.push_back(float(px+rx));
      verts.push_back(float(py+ry));
      colrs.push_back(float(pColor[0]));
      colrs.push_back(float(pColor[1]));
      colrs.push_back(float(pColor[2]));

      if (qx >= px)
         tma = ang + float(degToRad(+(i+1)*12.0));
      else 
         tma = ang + float(degToRad(-(i+1)*12.0));
      rx = radius*cos(tma);
      ry = radius*sin(tma);
         
      verts.push_back(float(px+rx));
      verts.push_back(float(py+ry));
      colrs.push_back(float(pColor[0]));
      colrs.push_back(float(pColor[1]));
      colrs.push_back(float(pColor[2]));

      // Draw endcap for point Q
      if (qx >= px)
         tma = ang + float(degToRad(+i*12.0));
      else
         tma = ang + float(degToRad(-i*12.0));
      rx = radius*cos(tma);
      ry = radius*sin(tma);
         
      verts.push_back(float(qx));
      verts.push_back(float(qy));
      colrs.push_back(float(qColor[0]));
      colrs.push_back(float(qColor[1]));
      colrs.push_back(float(qColor[2]));

      verts.push_back(float(qx-rx));
      verts.push_back(float(qy-ry));
      colrs.push_back(float(qColor[0]));
      colrs.push_back(float(qColor[1]));
      colrs.push_back(float(qColor[2]));

      if (qx >= px)
         tma = ang + float(degToRad(+(i+1)*12.0));
      else
         tma = ang + float(degToRad(-(i+1)*12.0));
      rx = radius*cos(tma);
      ry = radius*sin(tma);
         
      verts.push_back(float(qx-rx));
      verts.push_back(float(qy-ry));
      colrs.push_back(float(qColor[0]));
      colrs.push_back(float(qColor[1]));
      colrs.push_back(float(qColor[2]));
   }
   return verts.size()/2;
}

// Writing to pre-allocated input arrays, update pill shape between two points
int drawPill(
      float px, 
      float py, 
      float qx, 
      float qy, 
      float radius, 
      int   index,
      double *color, 
      float *verts,
      float *colrs
      ){
   return drawPill(px, py, qx, qy, radius, index, color, color, verts, colrs);
}

// Appending to input arrays, define vertices for a pill shape between two points
int drawPill(
      float px, 
      float py, 
      float qx, 
      float qy, 
      float radius, 
      double *color, 
      std::vector<float> &verts,
      std::vector<float> &colrs
      ){
   return drawPill(px, py, qx, qy, radius, color, color, verts, colrs);
}
