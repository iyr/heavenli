/*
 * Defines icon for lamps with circular arrangements
 */
unsigned int defineIconCircle(
      float          bx,               // X-Coordinate
      float          by,               // Y-Coordinate
      float          bs,               // Scale 2.0=spans display before GL scaling
      unsigned int   featureLevel,     // level of featureLevel
      unsigned int   circleSegments,   // Number of sides
      //float          angOffset,        // How much, in degrees, to rotate
      unsigned char  numBulbs,         // number of bulbs
      float          alpha,            // alpha transparency value
      float*         bulbColors,       // colors of the bulbs
      float*         detailColor,      // color of the accent details
      std::vector<float> &verts,       // Input Vector of x,y coordinates
      std::vector<float> &colrs        // Input Vector of r,g,b values
      ){

   float angOffset = 360.0f / float(numBulbs);
   float tmx, tmy, featScale;

   // Draw Only the color wheel if 'featureLevel' <= 0
   defineColorWheel(
         0.0f, 0.0f, 
         1.0f, 
         circleSegments, 
         180.0f, 
         (unsigned char)numBulbs, 
         detailColor[3], 
         bulbColors, 
         verts, colrs);

   // Draw Color Wheel + Outline if 'featureLevel' >= 1
   if (featureLevel >= 1) {
      featScale = 0.1f;    // Arch inner radius < outer radius
   } else {
      featScale = 0.0f;    // Arch inner radius == outer radius
   }

   defineArch(
         0.0f, 0.0f,       // (X, Y) spacial coordinates
         1.0f, 1.0f,       // (X, Y) arch radii
         0.0f, 360.0f,     // 0-360 arch draws a full circle
         featScale,        // width of circle (not radius)
         circleSegments,   // Number of polygon sides
         detailColor,      // arch color
         verts,
         colrs);

   // Draw Color Wheel + Outline + BulbMarkers if 'featureLevel' >= 2
   int tmo = 180/numBulbs;
   for (int j = 0; j < 6; j++) {
      if ( (j < numBulbs) && (featureLevel >= 2) ) {
         featScale = 0.16f;
      } else {
         featScale = 0.0f;
      }

      tmx = float(cos(degToRad(-90.0f - j*(angOffset) + tmo))*1.05f);
      tmy = float(sin(degToRad(-90.0f - j*(angOffset) + tmo))*1.05f);

      defineEllipse(
            tmx, tmy,               
            featScale, featScale,
            circleSegments/3, 
            detailColor, 
            verts, colrs);
   }

   // Draw Halos for bulb Markers
   // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos if 'featureLevel' == 3
   for (int j = 0; j < 6; j++) {
      if (j < numBulbs && featureLevel >= 3) {
         featScale = 0.07f;
      } else {
         featScale = 0.0f;
      }

      tmx = float(cos(degToRad(-90.0f - j*(angOffset) + tmo))*1.05f);
      tmy = float(sin(degToRad(-90.0f - j*(angOffset) + tmo))*1.05f);

      defineArch(
            tmx, tmy,         // (X, Y) spacial coordinates
            0.22f, 0.22f,     // (X, Y) arch radii
            0.0f, 360.0f,     // 0-360 arch draws a full circle
            featScale,        // width of circle (not radius)
            circleSegments/2, // Number of polygon sides
            detailColor,      // arch color
            verts, colrs);
   }

   // Draw Grand Halo
   // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos + Grand Halo if 'featureLevel' == 4
   if (featureLevel >= 4) {
      featScale = 0.08f;
   } else {
      featScale = 0.0f;
   }

   tmx = 0.0f;
   tmy = 0.0f;
   defineArch(
         tmx, tmy,         // (X, Y) spacial coordinates
         1.28f, 1.28f,     // (X, Y) arch radii
         0.0f, 360.0f,     // 0-360 arch draws a full circle
         featScale,        // width of circle (not radius)
         circleSegments,   // Number of polygon sides
         detailColor,      // arch color
         verts, colrs);

   return verts.size()/2;
}


/*
 *
 */
unsigned int updateIconCircleGeometry(
      float          bx,               // X-Coordinate
      float          by,               // Y-Coordinate
      float          bs,               // Scale 2.0=spans display before GL scaling
      unsigned int   featureLevel,     // level of featureLevel
      unsigned int   circleSegments,   // Number of sides
      unsigned char  numBulbs,         // number of bulbs
      unsigned int   index,            // Index of where to start writing in input array
      float*         verts             // Input Vector of x,y coordinates
      ){

   float angOffset = 360.0f / float(numBulbs);
   float tmx, tmy, featScale;
   unsigned int subIndex = index;

   // Draw Only the color wheel if 'featureLevel' <= 0
   subIndex = updateColorWheelGeometry(
         0.0f, 0.0f, 
         1.0f, 
         circleSegments, 
         180.0f, 
         (unsigned char)numBulbs, 
         subIndex,         // Index of where to start writing to input arrays
         verts);

   // Draw Color Wheel + Outline if 'featureLevel' >= 1
   if (featureLevel >= 1) {
      featScale = 0.1f;    // Arch inner radius < outer radius
   } else {
      featScale = 0.0f;    // Arch inner radius == outer radius
   }

   subIndex = updateArchGeometry(
         0.0f, 0.0f,       // (X, Y) spacial coordinates
         1.0f, 1.0f,       // (X, Y) arch radii
         0.0f, 360.0f,     // 0-360 arch draws a full circle
         featScale,        // width of circle (not radius)
         circleSegments,   // Number of polygon sides
         subIndex,         // Index of where to start writing to input arrays
         verts);

   // Draw Color Wheel + Outline + BulbMarkers if 'featureLevel' >= 2
   int tmo = 180/numBulbs;
   for (int j = 0; j < 6; j++) {
      if ( (j < numBulbs) && (featureLevel >= 2) ) {
         featScale = 0.16f;
      } else {
         featScale = 0.0f;
      }

      tmx = float(cos(degToRad(-90.0f - j*(angOffset) + tmo))*1.05f);
      tmy = float(sin(degToRad(-90.0f - j*(angOffset) + tmo))*1.05f);

      subIndex = updateEllipseGeometry(
            tmx, tmy,               
            featScale, featScale,
            circleSegments/3, 
            subIndex,               // Index of where to start writing to input arrays
            verts);
   }

   // Draw Halos for bulb Markers
   // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos if 'featureLevel' == 3
   for (int j = 0; j < 6; j++) {
      if (j < numBulbs && featureLevel >= 3) {
         featScale = 0.07f;
      } else {
         featScale = 0.0f;
      }

      tmx = float(cos(degToRad(-90.0f - j*(angOffset) + tmo))*1.05f);
      tmy = float(sin(degToRad(-90.0f - j*(angOffset) + tmo))*1.05f);

      subIndex = updateArchGeometry(
            tmx, tmy,         // (X, Y) spacial coordinates
            0.22f, 0.22f,     // (X, Y) arch radii
            0.0f, 360.0f,     // 0-360 arch draws a full circle
            featScale,        // width of circle (not radius)
            circleSegments/2, // Number of polygon sides
            subIndex,         // Index of where to start writing to input arrays
            verts);
   }

   // Draw Grand Halo
   // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos + Grand Halo if 'featureLevel' == 4
   if (featureLevel >= 4) {
      featScale = 0.08f;
   } else {
      featScale = 0.0f;
   }

   tmx = 0.0f;
   tmy = 0.0f;
   subIndex = updateArchGeometry(
         tmx, tmy,         // (X, Y) spacial coordinates
         1.28f, 1.28f,     // (X, Y) arch radii
         0.0f, 360.0f,     // 0-360 arch draws a full circle
         featScale,        // width of circle (not radius)
         circleSegments,   // Number of polygon sides
         subIndex,         // Index of where to start writing to input arrays
         verts);

   return subIndex;
}

unsigned int updateIconCircleColor(
      //unsigned int   featureLevel,     // level of featureLevel
      unsigned int   circleSegments,   // Number of sides
      unsigned char  numBulbs,         // number of bulbs
      float          alpha,            // alpha transparency value
      float*         bulbColors,       // colors of the bulbs
      float*         detailColor,      // color of the accent details
      unsigned int   index,            // index of where to start writing to input array
      float*         colrs             // Input Vector of r,g,b values
      ){

   float angOffset = 360.0f / float(numBulbs);
   unsigned int subIndex = index;

   // Draw Only the color wheel if 'featureLevel' <= 0
   subIndex = updateColorWheelColor(
         circleSegments, 
         (unsigned char)numBulbs, 
         detailColor[3], 
         bulbColors, 
         subIndex,         // Index of where to start writing to input arrays
         colrs);

   // Draw Color Wheel + Outline if 'featureLevel' >= 1
   subIndex = updateArchColor(
         circleSegments,   // Number of polygon sides
         detailColor,      // arch color
         subIndex,         // Index of where to start writing to input arrays
         colrs);

   // Draw Color Wheel + Outline + BulbMarkers if 'featureLevel' >= 2
   for (int j = 0; j < 6; j++) {
      subIndex = updateEllipseColor(
            circleSegments/3, 
            detailColor, 
            subIndex,         // Index of where to start writing to input arrays
            colrs);
   }

   // Draw Halos for bulb Markers
   // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos if 'featureLevel' == 3
   for (int j = 0; j < 6; j++) {
      subIndex = updateArchColor(
            circleSegments/2, // Number of polygon sides
            detailColor,      // arch color
            subIndex,         // Index of where to start writing to input arrays
            colrs);
   }

   // Draw Grand Halo
   // Draw Color Wheel + Outline + Bulb Markers + Bulb Halos + Grand Halo if 'featureLevel' == 4
   subIndex = updateArchColor(
         circleSegments,   // Number of polygon sides
         detailColor,      // arch color
         subIndex,         // Index of where to start writing to input arrays
         colrs);

   return subIndex;
}

