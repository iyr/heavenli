float cx;
float cy;
float wx  = 140.0;   //width modifier
float wy  = 200.0;   //height modifier
float ao  = 0.0;     //angular offset
float dBias;
int   nb  = 3;       //number of zones
int mode  = 1;       //lighting arrangement: 1 = circular, -1 = linear

void setup() {
  size(800, 800);
  rectMode(CORNER);
  cx = float(width/2);
  cy = float(height/2);

  randomSeed(628);

  if (wy > wx)
    dBias  = wx;
  else if (wx > wy)
    dBias  = wy;
  else if (wx == wy)
    dBias  = wx;
}

void draw() {
  if (wy > wx)
    dBias  = wx;
  else if (wx > wy)
    dBias  = wy;
  else if (wx == wy)
    dBias  = wx;
  background(255, 255, 0);
  //noStroke();
  //fill(255);
  //rect(gx-dx, gy-dy, 2*dx, 2*dy);

  // Ellipse Visual Marker
  for (float i = 0; i < 360; i += 1) {
    //println(str(i) + " quack");
    stroke(255, 0, 0);
    ellipse(
      cx + cos(radians(i))*sqrt(pow(wx, 2) + pow(wy, 2))*(wx/wy), 
      cy + sin(radians(i))*sqrt(pow(wx, 2) + pow(wy, 2))*(wy/wx), 
      1.0, 
      1.0);
    stroke(0, 0, 255);
    ellipse(
      cx + cos(radians(i))*sqrt(pow(wx, 2) + pow(wy, 2)), 
      cy + sin(radians(i))*sqrt(pow(wx, 2) + pow(wy, 2)), 
      1.0, 
      1.0);
  }

  if (mode > 0) 
    drawHomeCircle(cx, cy, wx, wy, nb);

  else if (mode < 0) {
    drawHomeTri(cx, cy, wx, wy, nb);

    // Bulb Button visual markers
    for (int i = 0; i < nb; i++) {
      stroke(0, 0, 255);
      fill(i*(255/nb));
      float quack = radians(i*(180/constrain(nb-1, 1, nb))+(ao-ao%45));
      ellipse(
        cx + cos(quack)*0.78*dBias, 
        cy + sin(quack)*0.78*dBias, 
        dBias*0.3, 
        dBias*0.3);
    }
  }

  // Center Clock button visual marker
  fill(0, 64, 48, 192);
  ellipse(cx, cy, dBias, dBias);
}

void keyPressed() {
  switch(keyCode) {
  case LEFT:
    ao -= 5.0;
    if (ao < 0)
      ao += 360;

    println(ao);
    break;

  case RIGHT:
    ao += 5.0;
    if (ao > 360)
      ao -= 360;

    println(ao);
    break;

  case UP:
    if (mode > 0)
      nb = constrain(nb+1, 1, 6);
    if (mode < 0)
      nb = constrain(nb+1, 1, 5);
    break;

  case DOWN:
    if (mode > 0)
      nb = constrain(nb-1, 1, 6);
    if (mode < 0)
      nb = constrain(nb-1, 1, 5);
    break;

  case 98:   //Numpad 2 (down)
    wy -= 5;
    break;

  case 104:  //Numpad 8 (up)
    wy += 5;
    break;

  case 100:  //Numpad 4 (left)
    wx -= 5;
    break;

  case 102:  //Numpad 6 (right)
    wx += 5;
    break;
  }

  switch(key) {
  case ' ':
    mode *= -1;
    ao = 0;
    if (mode < 0)
      nb = constrain(nb, 1, 5);
    println(mode);
    break;
  }
}

void drawHomeCircle(float gx, 
  float gy, 
  float dx, 
  float dy, 
  int nz) {
  {
    //stroke(0, 0, 255);
    for (int i = 0; i < nz; i++) {
      // Green Lines
      float px = constrain(gx + cos(radians(i*(360/nz)+ao+45))*sqrt(pow(dx, 2) + pow(dy, 2)), gx-dx, gx+dx);
      float py = constrain(gy + sin(radians(i*(360/nz)+ao+45))*sqrt(pow(dx, 2) + pow(dy, 2)), gy-dy, gy+dy);

      // Blue Lines
      float qx = constrain(gx + cos(radians(i*(360/nz)+ao))*sqrt(pow(dx, 2) + pow(dy, 2)), gx-dx, gx+dx);
      float qy = constrain(gy + sin(radians(i*(360/nz)+ao))*sqrt(pow(dx, 2) + pow(dy, 2)), gy-dy, gy+dy);
      // Next Line in iteration for zoning
      float rx = constrain(gx + cos(radians((i+1)*(360/nz)+ao))*sqrt(pow(dx, 2) + pow(dy, 2)), gx-dx, gx+dx);
      float ry = constrain(gy + sin(radians((i+1)*(360/nz)+ao))*sqrt(pow(dx, 2) + pow(dy, 2)), gy-dy, gy+dy);

      fill(i*(255/nz));
      stroke(i*(255/nz));
      line(px, py, qx, qy);
      line(px, py, rx, ry);

      switch(nz) {
      case 1:
        fill(128, 64, 192);
        rect(gx-dx, gx-dy, 2*dx, 2*dy);
        break;

      case 2:
        if (qx == gx-dx && rx == gx+dx) {
          quad(rx, ry, qx, qy, gx-dx, gy-dy, gx+dx, gy-dy);
        }
        if (qy == gy-dy && ry == gy+dy) {
          quad(rx, ry, qx, qy, gx+dx, gy-dy, gx+dx, gy+dy);
        }
        if (qx == gx+dx && rx == gx-dx) {
          quad(qx, qy, rx, ry, gx-dx, gy+dy, gx+dx, gy+dy);
        }
        if (qy == gy+dy && ry == gy-dy) {
          quad(qx, qy, rx, ry, gx-dx, gy-dy, gx-dx, gy+dy);
        }
        break;

      default:
        /* General Purpose Background Geometry Rendering for Circularly Arranged Lights
         * As of writing, all edge cases (the oodles of conditionals) were accounted for
         * 1 - 6 lights 
         */
        if (px == qx || py == qy)
          triangle(px, py, qx, qy, 
            nz == 3 ? gx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gx, 
            nz == 3 ? gy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gy);
        if (qx == rx || qy == ry)
          triangle(rx, ry, qx, qy, 
            nz == 3 ? gx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gx, 
            nz == 3 ? gy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gy);
        if (rx == px || ry == py)
          triangle(px, py, rx, ry, 
            nz == 3 ? gx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gx, 
            nz == 3 ? gy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gy);

        if (px != qx && py != qy)
        {
          triangle(px, py, qx, qy, 
            nz == 3 ? gx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gx, 
            nz == 3 ? gy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gy);

          triangle(px, py, qx, qy, 
            (px == gx-dx) || (px == gx+dx) ? px : qx, 
            (py == gy-dy) || (py == gy+dy) ? py : qy
            );

          if (
            (px > gx-dx) && 
            (px < gx+dx) &&
            (qx > gx-dx) &&
            (qx < gx+dx)
            ) {
            if ((px > gx-dx) && (px < gx))
              quad(px, py, qx, qy, gx-dx, gy-dy, gx-dx, gy+dy);

            if ((px < gx+dx) && (px > gx))
              quad(px, py, qx, qy, gx+dx, gy+dy, gx+dx, gy-dy);

            if ((qx > gx-dx) && (qx < gx))
              quad(qx, qy, px, py, gx-dx, gy-dy, gx-dx, gy+dy);

            if ((qx < gx+dx) && (qx > gx))
              quad(qx, qy, px, py, gx+dx, gy+dy, gx+dx, gy-dy);
          }
          if (
            (py > gy-dy) && 
            (py < gy+dy) &&
            (qy > gy-dy) &&
            (qy < gy+dy)
            ) {
            if ((py > gy-dy) && (py < gy))
              quad(px, py, qx, qy, gx-dx, gy-dy, gx+dx, gy-dy);

            if ((py < gy+dy) && (py > gy))
              quad(px, py, qx, qy, gx+dx, gy+dy, gx-dx, gy+dy);

            if ((qy > gy-dy) && (qy < gy))
              quad(qx, qy, px, py, gx-dx, gy-dy, gx-dx, gy+dy);

            if ((qy < gy+dy) && (qx > gy))
              quad(qx, qy, px, py, gx+dx, gy+dy, gx+dx, gy-dy);
          }
        }
        if (rx != px && ry != py)
        {
          triangle(px, py, rx, ry, 
            nz == 3 ? gx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gx, 
            nz == 3 ? gy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gy);

          triangle(px, py, rx, ry, 
            (px == gx-dx) || (px == gx+dx) ? px : rx, 
            (py == gy-dy) || (py == gy+dy) ? py : ry
            );

          if (
            (px > gx-dx) && 
            (px < gx+dx) &&
            (rx > gx-dx) &&
            (rx < gx+dx)
            ) {
            if ((px > gx-dx) && (px < gx))
              quad(px, py, rx, ry, gx-dx, gy-dy, gx-dx, gy+dy);

            if ((px < gx+dx) && (px > gx))
              quad(px, py, rx, ry, gx+dx, gy+dy, gx+dx, gy-dy);

            if ((rx > gx-dx) && (rx < gx))
              quad(gx-dx, gy-dy, gx-dx, gy+dy, rx, ry, px, py);

            if ((rx < gx+dx) && (rx > gx))
              quad(gx+dx, gy+dy, gx+dx, gy-dy, rx, ry, px, py);
          }
          if (
            (py > gy-dy) && 
            (py < gy+dy) &&
            (ry > gy-dy) &&
            (ry < gy+dy)
            ) {
            if ((py > gy-dy) && (py < gy))
              quad(px, py, rx, ry, gx-dx, gy-dy, gx+dx, gy-dy);

            if ((py < gy+dy) && (py > gy))
              quad(px, py, rx, ry, gx-dx, gy+dy, gx+dx, gy+dy);

            if ((ry > gy-dy) && (ry < gy))
              quad(rx, ry, px, py, gx-dx, gy-dy, gx+dx, gy-dy);

            if ((ry < gy+dy) && (ry > gy))
              quad(px, py, rx, ry, gx-dx, gy+dy, gx+dx, gy+dy);
          }
        }

        // Red Line Visual Marker
        //  if (i == 1) {
        //stroke(255,0,0);
        //    line(
        //      cx, 
        //      cy, 
        //      cx + cos(radians(i*(360/nz)+(180/nz)+ao))*sqrt(pow(dx, 2) + pow(dy, 2)), 
        //      cy + sin(radians(i*(360/nz)+(180/nz)+ao))*sqrt(pow(dx, 2) + pow(dy, 2)));
        //  }
        break;
      }

      // Bulb Button Visual Markers
      stroke(0, 0, 255);
      ellipse(
        cx + cos(radians(i*(360/nz)+(180/nz)+ao))*dBias*0.75, 
        cy + sin(radians(i*(360/nz)+(180/nz)+ao))*dBias*0.75, 
        dBias*0.3, 
        dBias*0.3);

      // Zone Line division Visual Marker
      //if (nz > 1)
      //  line(
      //    nz == 3 ? cx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : cx, 
      //    nz == 3 ? cy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : cy, 
      //    qx, 
      //    qy
      //    );
      //noStroke();
      // Green Line Visual Marker
      stroke(i*(255/nz));
      line(
        nz == 3 ? gx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gx, 
        nz == 3 ? gy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gy, 
        px, 
        py
        );
    }
  }
}

void drawHomeTri (float gx, 
  float gy, 
  float dx, 
  float dy, 
  int nz) {
  switch(nz) {    
  case 3:
    {
      float[] ptsX = new float[7];
      float[] ptsY = new float[7];

      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5) {
        float[] tmX = {gx, gx+dx, gx+dx, gx+dx-dBias*0.333, gx-dx+dBias*0.333, gx-dx, gx-dx};
        float[] tmY = {gy-dy, gy-dy, gy+dy, gy+dy, gy+dy, gy+dy, gy-dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 67.5  && ao <= 112.5) {          
        float[] tmX = {gx+dx, gx+dx, gx-dx, gx-dx, gx-dx, gx-dx, gx+dx};
        float[] tmY = {gy, gy+dy, gy+dy, gy+dy-dBias*0.333, gy-dy+dBias*0.333, gy-dy, gy-dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 157.5 && ao <= 202.5) {
        float[] tmX = {gx, gx-dx, gx-dx, gx-dx+dBias*0.333, gx+dx-dBias*0.333, gx+dx, gx+dx};
        float[] tmY = {gy+dy, gy+dy, gy-dy, gy-dy, gy-dy, gy-dy, gy+dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 247.5 && ao <= 292.5) {          
        float[] tmX = {gx-dx, gx-dx, gx+dx, gx+dx, gx+dx, gx+dx, gx-dx};
        float[] tmY = {gy, gy-dy, gy-dy, gy-dy+dBias*0.333, gy+dy-dBias*0.333, gy+dy, gy+dy};
        ptsX = tmX;
        ptsY = tmY;
      }

      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5) {
        float[] tmX = {gx+dx, gx+dx, gx-dx/4, gx-dx, gx-dx, gx-dx, 0};
        float[] tmY = {gy-dy, gy+dy, gy+dy, gy+dy, gy+dy/4, gy-dy, 0};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 112.5 && ao <= 157.5) {
        float[] tmX = {gx+dx, gx-dx, gx-dx, gx-dx, gx-dx/4, gx+dx, 0};
        float[] tmY = {gy+dy, gy+dy, gy-dy/4, gy-dy, gy-dy, gy-dy, 0};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 202.5 && ao <= 247.5) {
        float[] tmX = {gx-dx, gx-dx, gx+dx/4, gx+dx, gx+dx, gx+dx, 0};
        float[] tmY = {gy+dy, gy-dy, gy-dy, gy-dy, gy-dy/4, gy+dy, 0};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 292.5 && ao <= 337.5) {
        float[] tmX = {gx-dx, gx+dx, gx+dx, gx+dx, gx+dx/4, gx-dx, 0};
        float[] tmY = {gy-dy, gy-dy, gy+dy/4, gy+dy, gy+dy, gy+dy, 0};
        ptsX = tmX;
        ptsY = tmY;
      }

      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5  || 
        ao > 67.5  && ao <= 112.5 ||
        ao > 157.5 && ao <= 202.5 ||
        ao > 247.5 && ao <= 292.5) {
        fill(0*(255/nz));
        quad(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2], ptsX[3], ptsY[3]);
        fill(1*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[3], ptsY[3], ptsX[4], ptsY[4]);
        fill(2*(255/nz));
        quad(ptsX[0], ptsY[0], ptsX[4], ptsY[4], ptsX[5], ptsY[5], ptsX[6], ptsY[6]);
      } else 
      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5  ||
        ao > 112.5 && ao <= 157.5 ||
        ao > 202.5 && ao <= 247.5 ||
        ao > 292.5 && ao <= 337.5) {
        fill(0*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2]);
        fill(1*(255/nz));
        quad(ptsX[0], ptsY[0], ptsX[2], ptsY[2], ptsX[3], ptsY[3], ptsX[4], ptsY[4]);
        fill(2*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[4], ptsY[4], ptsX[5], ptsY[5]);
      }
    }
    break;

  case 4: 
    {
      float[] ptsX = new float[6];
      float[] ptsY = new float[6];

      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5) {
        float[] tmX = {gx,    gx+dx, gx+dx, gx,    gx-dx, gx-dx};
        float[] tmY = {gy-dy, gy-dy, gy+dy, gy+dy, gy+dy, gy-dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 67.5  && ao <= 112.5) {          
        float[] tmX = {gx+dx, gx+dx, gx-dx, gx-dx, gx-dx, gx+dx};
        float[] tmY = {gy,    gy+dy, gy+dy, gy,    gy-dy, gy-dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 157.5 && ao <= 202.5) {
        float[] tmX = {gx,    gx-dx, gx-dx, gx,    gx+dx, gx+dx};
        float[] tmY = {gy+dy, gy+dy, gy-dy, gy-dy, gy-dy, gy+dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 247.5 && ao <= 292.5) {          
        float[] tmX = {gx-dx, gx-dx, gx+dx, gx+dx, gx+dx, gx-dx};
        float[] tmY = {gy,    gy-dy, gy-dy, gy,    gy+dy, gy+dy};
        ptsX = tmX;
        ptsY = tmY;
      }

      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5) {
        float[] tmX = {gx+dx, gx+dx, gx, gx-dx, gx-dx, gx-dx};
        float[] tmY = {gy-dy, gy+dy, gy+dy, gy+dy, gy, gy-dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 112.5 && ao <= 157.5) {
        float[] tmX = {gx+dx, gx-dx, gx-dx, gx-dx, gx, gx+dx};
        float[] tmY = {gy+dy, gy+dy, gy, gy-dy, gy-dy, gy-dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 202.5 && ao <= 247.5) {
        float[] tmX = {gx-dx, gx-dx, gx, gx+dx, gx+dx, gx+dx};
        float[] tmY = {gy+dy, gy-dy, gy-dy, gy-dy, gy, gy+dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 292.5 && ao <= 337.5) {
        float[] tmX = {gx-dx, gx+dx, gx+dx, gx+dx, gx, gx-dx};
        float[] tmY = {gy-dy, gy-dy, gy, gy+dy, gy+dy, gy+dy};
        ptsX = tmX;
        ptsY = tmY;
      }

      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5  || 
        ao > 67.5  && ao <= 112.5 ||
        ao > 157.5 && ao <= 202.5 ||
        ao > 247.5 && ao <= 292.5) {
        fill(0*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2]);
        fill(1*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[2], ptsY[2], ptsX[3], ptsY[3]);
        fill(2*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[3], ptsY[3], ptsX[4], ptsY[4]);
        fill(3*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[4], ptsY[4], ptsX[5], ptsY[5]);
      } else 
      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5  ||
        ao > 112.5 && ao <= 157.5 ||
        ao > 202.5 && ao <= 247.5 ||
        ao > 292.5 && ao <= 337.5) {
        fill(0*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2]);
        fill(1*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[2], ptsY[2], ptsX[3], ptsY[3]);
        fill(2*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[3], ptsY[3], ptsX[4], ptsY[4]);
        fill(3*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[4], ptsY[4], ptsX[5], ptsY[5]);
      }
    }
    break;

  case 5:
    {
      float[] ptsX = new float[9];
      float[] ptsY = new float[9];

      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5) {
        //             0      1      2        3        4        5      6      7        8
        float[] tmX = {gx,    gx+dx, gx+dx,   gx+dx, gx+dx/2, gx-dx/2, gx-dx, gx-dx,   gx-dx};
        float[] tmY = {gy-dy, gy-dy, gy+dy/2, gy+dy, gy+dy,   gy+dy,   gy+dy, gy+dy/2, gy-dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 67.5  && ao <= 112.5) {
        //             0      1      2        3        4        5      6      7        8
        float[] tmX = {gx+dx, gx+dx, gx-dx/2, gx-dx, gx-dx,   gx-dx,   gx-dx, gx-dx/2, gx+dx};
        float[] tmY = {gy,    gy+dy, gy+dy,   gy+dy, gy+dy/2, gy-dy/2, gy-dy, gy-dy,   gy-dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 157.5 && ao <= 202.5) {
        //             0      1      2        3        4        5      6      7        8
        float[] tmX = {gx,    gx-dx, gx-dx,   gx-dx, gx-dx/2, gx+dx/2, gx+dx, gx+dx,   gx+dx};
        float[] tmY = {gy+dy, gy+dy, gy-dy/2, gy-dy, gy-dy,   gy-dy,   gy-dy, gy-dy/2, gy+dy};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 247.5 && ao <= 292.5) {
        //             0      1      2        3        4        5      6      7        8
        float[] tmX = {gx-dx, gx-dx, gx+dx/2, gx+dx, gx+dx,   gx+dx,   gx+dx, gx+dx/2, gx-dx};
        float[] tmY = {gy,    gy-dy, gy-dy,   gy-dy, gy-dy/2, gy+dy/2, gy+dy, gy+dy,   gy+dy};
        ptsX = tmX;
        ptsY = tmY;
      }

      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5) {
        //             0      1      2        3        4      5        6        7       8
        float[] tmX = {gx+dx, gx+dx, gx+dx/3, gx-dx/2, gx-dx, gx-dx,   gx-dx,   gx-dx,  0};
        float[] tmY = {gy-dy, gy+dy, gy+dy,   gy+dy,   gy+dy, gy+dy/2, gy-dy/3, gy-dy,  0};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 112.5 && ao <= 157.5) {
        //             0      1      2        3        4      5        6        7       8
        float[] tmX = {gx+dx, gx-dx, gx-dx,   gx-dx,   gx-dx, gx-dx/2, gx+dx/3, gx+dx,  0};
        float[] tmY = {gy+dy, gy+dy, gy+dy/3, gy-dy/2, gy-dy, gy-dy,   gy-dy,   gy-dy,  0};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 202.5 && ao <= 247.5) {
        //             0      1      2        3        4      5        6        7       8
        float[] tmX = {gx-dx, gx-dx, gx-dx/3, gx+dx/2, gx+dx, gx+dx,   gx+dx,   gx+dx,  0};
        float[] tmY = {gy+dy, gy-dy, gy-dy,   gy-dy,   gy-dy, gy-dy/2, gy+dy/3, gy+dy,  0};
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 292.5 && ao <= 337.5) {
        //             0      1      2        3        4      5        6        7       8
        float[] tmX = {gx-dx, gx+dx, gx+dx,   gx+dx,   gx+dx, gx+dx/2, gx-dx/3, gx-dx,  0};
        float[] tmY = {gy-dy, gy-dy, gy-dy/3, gy+dy/2, gy+dy, gy+dy,   gy+dy,   gy+dy,  0};
        ptsX = tmX;
        ptsY = tmY;
      }

      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5  || 
        ao > 67.5  && ao <= 112.5 ||
        ao > 157.5 && ao <= 202.5 ||
        ao > 247.5 && ao <= 292.5) {
        fill(0*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2]);
        fill(1*(255/nz));
        quad(ptsX[0], ptsY[0], ptsX[2], ptsY[2], ptsX[3], ptsY[3], ptsX[4], ptsY[4]);
        fill(2*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[4], ptsY[4], ptsX[5], ptsY[5]);
        fill(3*(255/nz));
        quad(ptsX[0], ptsY[0], ptsX[5], ptsY[5], ptsX[6], ptsY[6], ptsX[7], ptsY[7]);
        fill(4*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[7], ptsY[7], ptsX[8], ptsY[8]);
      } else 
      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5  ||
        ao > 112.5 && ao <= 157.5 ||
        ao > 202.5 && ao <= 247.5 ||
        ao > 292.5 && ao <= 337.5) {
        fill(0*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2]);
        fill(1*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[2], ptsY[2], ptsX[3], ptsY[3]);
        fill(2*(255/nz));
        quad(ptsX[0], ptsY[0], ptsX[3], ptsY[3], ptsX[4], ptsY[4], ptsX[5], ptsY[5]);
        fill(3*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[5], ptsY[5], ptsX[6], ptsY[6]);
        fill(4*(255/nz));
        triangle(ptsX[0], ptsY[0], ptsX[6], ptsY[6], ptsX[7], ptsY[7]);
      }
    }
    break;

  default:
    drawHomeCircle(gx, gy, dx, dy, nz);
    break;
  }
}
