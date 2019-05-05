#include <math.h>
#include <vector>
using namespace std;

// Write to pre-allocated input arrays, update pill shape between two points
unsigned int drawPill(
      float px,      /* x-coordinate of Point P */
      float py,      /* y-coordinate of Point P */
      float qx,      /* x-coordinate of Point Q */
      float qy,      /* y-coordinate of Point Q */
      float radius,  /* Radius/Thickness of pill */
      int   index,   /* index of where to start writing to input arrays */
      float *pColor, /* RGB values of P */
      float *qColor, /* RGB values of Q */
      float *verts,  /* Input Array of x,y coordinates */
      float *colrs   /* Input Array of r,g,b values */
      ){
   int vertIndex = index*2;   /* index (x, y) */
   int colrIndex = index*3;   /* index (r, g, b) */
   float rx, ry, slope, tma, pR, pG, pB, qR, qG, qB;

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
   } else {
      slope = (py-qy)/(px-qx);
   }
   float ang = float(degToRad(90)+atan(slope));

   rx = float(radius*cos(ang));
   ry = float(radius*sin(ang));

   pR = float(pColor[0]);
   pG = float(pColor[1]);
   pB = float(pColor[2]);
   qR = float(qColor[0]);
   qG = float(qColor[1]);
   qB = float(qColor[2]);
   // Draw Pill Body, (Rectangle)
   /* pX */ verts[vertIndex++] = float(px+rx);
   /* pY */ verts[vertIndex++] = float(py+ry);
   /* pX */ verts[vertIndex++] = float(px-rx);
   /* pY */ verts[vertIndex++] = float(py-ry);
   /* qX */ verts[vertIndex++] = float(qx-rx);
   /* qY */ verts[vertIndex++] = float(qy-ry);

   /* qX */ verts[vertIndex++] = float(qx-rx);
   /* qY */ verts[vertIndex++] = float(qy-ry);
   /* qX */ verts[vertIndex++] = float(qx+rx);
   /* qY */ verts[vertIndex++] = float(qy+ry);
   /* pX */ verts[vertIndex++] = float(px+rx);
   /* pY */ verts[vertIndex++] = float(py+ry);

   /* pR */ colrs[colrIndex++] = pR;   /* pG */ colrs[colrIndex++] = pG;   /* pB */ colrs[colrIndex++] = pB;
   /* pR */ colrs[colrIndex++] = pR;   /* pG */ colrs[colrIndex++] = pG;   /* pB */ colrs[colrIndex++] = pB;
   /* qR */ colrs[colrIndex++] = qR;   /* qG */ colrs[colrIndex++] = qG;   /* qB */ colrs[colrIndex++] = qB;

   /* qR */ colrs[colrIndex++] = qR;   /* qG */ colrs[colrIndex++] = qG;   /* qB */ colrs[colrIndex++] = qB;
   /* qR */ colrs[colrIndex++] = qR;   /* qG */ colrs[colrIndex++] = qG;   /* qB */ colrs[colrIndex++] = qB;
   /* pR */ colrs[colrIndex++] = pR;   /* pG */ colrs[colrIndex++] = pG;   /* pB */ colrs[colrIndex++] = pB;

//#  pragma omp parallel for
   for (int i = 0; i < 15; i++) {
      // Draw endcap for point P
      if (qx >= px)
         tma = ang + float(degToRad(+i*12.0));
      else
         tma = ang + float(degToRad(-i*12.0));
      rx = float(radius*cos(tma));
      ry = float(radius*sin(tma));
         
      /* pX */ verts[vertIndex++] = float(px);
      /* pY */ verts[vertIndex++] = float(py);

      /* pX */ verts[vertIndex++] = float(px+rx);
      /* pY */ verts[vertIndex++] = float(py+ry);

      if (qx >= px)
         tma = ang + float(degToRad(+(i+1)*12.0));
      else 
         tma = ang + float(degToRad(-(i+1)*12.0));
      rx = float(radius*cos(tma));
      ry = float(radius*sin(tma));
         
      /* pX */ verts[vertIndex++] = float(px+rx);
      /* pY */ verts[vertIndex++] = float(py+ry);

      /* pR */ colrs[colrIndex++] = pR;  /* pG */ colrs[colrIndex++] = pG;  /* pB */ colrs[colrIndex++] = pB;
      /* pR */ colrs[colrIndex++] = pR;  /* pG */ colrs[colrIndex++] = pG;  /* pB */ colrs[colrIndex++] = pB;
      /* pR */ colrs[colrIndex++] = pR;  /* pG */ colrs[colrIndex++] = pG;  /* pB */ colrs[colrIndex++] = pB;


      // Draw endcap for point Q
      if (qx >= px)
         tma = ang + float(degToRad(+i*12.0));
      else
         tma = ang + float(degToRad(-i*12.0));
      rx = float(radius*cos(tma));
      ry = float(radius*sin(tma));
         
      /* qX */ verts[vertIndex++] = float(qx);
      /* qY */ verts[vertIndex++] = float(qy);

      /* qX */ verts[vertIndex++] = float(qx-rx);
      /* qY */ verts[vertIndex++] = float(qy-ry);

      if (qx >= px)
         tma = ang + float(degToRad(+(i+1)*12.0));
      else
         tma = ang + float(degToRad(-(i+1)*12.0));
      rx = float(radius*cos(tma));
      ry = float(radius*sin(tma));
         
      /* qX */ verts[vertIndex++] = float(qx-rx);
      /* qY */ verts[vertIndex++] = float(qy-ry);

      /* qR */ colrs[colrIndex++] = qR;  /* qG */ colrs[colrIndex++] = qG;  /* qB */ colrs[colrIndex++] = qB;
      /* qR */ colrs[colrIndex++] = qR;  /* qG */ colrs[colrIndex++] = qG;  /* qB */ colrs[colrIndex++] = qB;
      /* qR */ colrs[colrIndex++] = qR;  /* qG */ colrs[colrIndex++] = qG;  /* qB */ colrs[colrIndex++] = qB;
   }
   return (vertIndex)/2;
}

// Appending to input arrays, define vertices for a pill shape between two points
unsigned int drawPill(
      float px,                  /* x-coordinate of Point P */
      float py,                  /* y-coordinate of Point P */
      float qx,                  /* x-coordinate of Point Q */
      float qy,                  /* y-coordinate of Point Q */
      float radius,              /* Radius/Thickness of pill */
      float *pColor,            /* RGB values of P */
      float *qColor,            /* RGB values of Q */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   float rx, ry, slope, tma, pR, pG, pB, qR, qG, qB;

   if (qx >= px) {
      slope = (qy-py)/(qx-px);
   } else {
      slope = (py-qy)/(px-qx);
   }
   float ang = float(degToRad(90)+atan(slope));

   pR = float(pColor[0]);
   pG = float(pColor[1]);
   pB = float(pColor[2]);
   qR = float(qColor[0]);
   qG = float(qColor[1]);
   qB = float(qColor[2]);

   rx = float(radius*cos(ang));
   ry = float(radius*sin(ang));

   // Draw Pill Body, (Rectangle)
   /* pX */ verts.push_back(float(px+rx));
   /* pY */ verts.push_back(float(py+ry));
   /* pX */ verts.push_back(float(px-rx));
   /* pY */ verts.push_back(float(py-ry));
   /* qX */ verts.push_back(float(qx-rx));
   /* qY */ verts.push_back(float(qy-ry));

   /* qX */ verts.push_back(float(qx-rx));
   /* qY */ verts.push_back(float(qy-ry));
   /* qX */ verts.push_back(float(qx+rx));
   /* qY */ verts.push_back(float(qy+ry));
   /* pX */ verts.push_back(float(px+rx));
   /* pY */ verts.push_back(float(py+ry));

   /* pR */ colrs.push_back(pR);  /* pG */ colrs.push_back(pG);  /* pB */ colrs.push_back(pB);
   /* pR */ colrs.push_back(pR);  /* pG */ colrs.push_back(pG);  /* pB */ colrs.push_back(pB);
   /* qR */ colrs.push_back(pR);  /* qG */ colrs.push_back(pG);  /* qB */ colrs.push_back(pB);

   /* qR */ colrs.push_back(qR);  /* qG */ colrs.push_back(qG);  /* qB */ colrs.push_back(qB);
   /* qR */ colrs.push_back(qR);  /* qG */ colrs.push_back(qG);  /* qB */ colrs.push_back(qB);
   /* pR */ colrs.push_back(qR);  /* pG */ colrs.push_back(qG);  /* pB */ colrs.push_back(qB);


//#  pragma omp parallel for
   for (int i = 0; i < 15; i++) {
      // Draw endcap for point P
      if (qx >= px)
         tma = ang + float(degToRad(+i*12.0));
      else
         tma = ang + float(degToRad(-i*12.0));
      rx = float(radius*cos(tma));
      ry = float(radius*sin(tma));
         
      /* pX */ verts.push_back(float(px));
      /* pY */ verts.push_back(float(py));

      /* pX */ verts.push_back(float(px+rx));
      /* pY */ verts.push_back(float(py+ry));

      if (qx >= px)
         tma = ang + float(degToRad(+(i+1)*12.0));
      else 
         tma = ang + float(degToRad(-(i+1)*12.0));
      rx = float(radius*cos(tma));
      ry = float(radius*sin(tma));
         
      /* pX */ verts.push_back(float(px+rx));
      /* pY */ verts.push_back(float(py+ry));

      /* pR */ colrs.push_back(pR);  /* pG */ colrs.push_back(pG);  /* pB */ colrs.push_back(pB);
      /* pR */ colrs.push_back(pR);  /* pG */ colrs.push_back(pG);  /* pB */ colrs.push_back(pB);
      /* pR */ colrs.push_back(pR);  /* pG */ colrs.push_back(pG);  /* pB */ colrs.push_back(pB);


      // Draw endcap for point Q
      if (qx >= px)
         tma = ang + float(degToRad(+i*12.0));
      else
         tma = ang + float(degToRad(-i*12.0));
      rx = float(radius*cos(tma));
      ry = float(radius*sin(tma));
         
      /* qX */ verts.push_back(float(qx));
      /* qY */ verts.push_back(float(qy));

      /* qX */ verts.push_back(float(qx-rx));
      /* qY */ verts.push_back(float(qy-ry));

      if (qx >= px)
         tma = ang + float(degToRad(+(i+1)*12.0));
      else
         tma = ang + float(degToRad(-(i+1)*12.0));
      rx = float(radius*cos(tma));
      ry = float(radius*sin(tma));
         
      /* qX */ verts.push_back(float(qx-rx));
      /* qY */ verts.push_back(float(qy-ry));

      /* qR */ colrs.push_back(qR);  /* qG */ colrs.push_back(qG);  /* qB */ colrs.push_back(qB);
      /* qR */ colrs.push_back(qR);  /* qG */ colrs.push_back(qG);  /* qB */ colrs.push_back(qB);
      /* qR */ colrs.push_back(qR);  /* qG */ colrs.push_back(qG);  /* qB */ colrs.push_back(qB);
   }
   //printf("Pill verts (v, c): %i, %i\n", verts.size()/2, colrs.size()/3);
   return verts.size()/2;
}

// Writing to pre-allocated input arrays, update pill shape between two points
unsigned int drawPill(
      float px,         /* x-coordinate of Point P */
      float py,         /* y-coordinate of Point P */
      float qx,         /* x-coordinate of Point Q */
      float qy,         /* y-coordinate of Point Q */
      float radius,     /* Radius/Thickness of pill */
      int index,        /* index of where to start writing to input arrays */
      float *color,    /* RGB values of pill */
      float *verts,     /* Input Array of x,y coordinates */
      float *colrs      /* Input Array of r,g,b values */
      ){
   return drawPill(px, py, qx, qy, radius, index, color, color, verts, colrs);
}

// Appending to input vectors, define vertices for a pill shape between two points
unsigned int drawPill(
      float px,                  /* x-coordinate of Point P */
      float py,                  /* y-coordinate of Point P */
      float qx,                  /* x-coordinate of Point Q */
      float qy,                  /* y-coordinate of Point Q */
      float radius,              /* Radius/Thickness of pill */
      float *color,             /* RGB values of pill */
      std::vector<float> &verts, /* Input Vector of x,y coordinates */
      std::vector<float> &colrs  /* Input Vector of r,g,b values */
      ){
   return drawPill(px, py, qx, qy, radius, color, color, verts, colrs);
}
