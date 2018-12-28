#include <math.h>

int drawPill(
      float px, 
      float py, 
      float qx, 
      float qy, 
      float radius, 
      double &pColor, 
      double &qColor
      std::vector<float> &verts,
      std::vector<float> &colrs
      ){

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
   } else {
      slope = (py-qy)/(px-qx);
   }
   float rx;
   float ry;
   float ang = degToRad(90)+atan(slope);
   float tma;
   float tmp;

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
         tma = ang + degToRad(+i*12.0);
      else
         tma = ang + degToRad(-i*12.0);
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
         tma = ang + degToRad(+(i+1)*12.0);
      else 
         tma = ang + degToRad(-(i+1)*12.0);
      rx = radius*cos(tma);
      ry = radius*sin(tma);
         
      verts.push_back(float(px+rx));
      verts.push_back(float(py+ry));
      colrs.push_back(float(pColor[0]));
      colrs.push_back(float(pColor[1]));
      colrs.push_back(float(pColor[2]));

      // Draw endcap for point Q
      if (qx >= px)
         tma = ang + degToRad(+i*12.0);
      else
         tma = ang + degToRad(-i*12.0);
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
         tma = ang + degToRad(+(i+1)*12.0);
      else
         tma = ang + degToRad(-(i+1)*12.0);
      rx = radius*cos(tma);
      ry = radius*sin(tma);
         
      verts.push_back(float(qx-rx));
      verts.push_back(float(qy-ry));
      colrs.push_back(float(qColor[0]));
      colrs.push_back(float(qColor[1]));
      colrs.push_back(float(qColor[2]));
   }
   return 0;
}
