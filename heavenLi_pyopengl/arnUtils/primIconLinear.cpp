/*
 * Defines icon for lamps with linear arrangements
 */
unsigned int defineIconLinear(
      float          bx,               // X-Coordinate
      float          by,               // Y-Coordinate
      float          bs,               // Scale 2.0=spans display before GL scaling
      unsigned int   featureLevel,     // level of featureLevel
      unsigned int   circleSegments,   // Number of sides
      unsigned char  numBulbs,         // number of bulbs
      float          alpha,            // alpha transparency value
      float*         bulbColors,       // colors of the bulbs
      float*         detailColor,      // color of the accent details
      std::vector<float> &verts,       // Input Vector of x,y coordinates
      std::vector<float> &colrs        // Input Vector of r,g,b values
      ){

      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy, featScale;
      float offset = 2.0f / 60.0f;
      float tmc[4];
      unsigned int segments = 60;
      unsigned int limit = segments/numBulbs;

      // Draw Block of stripes
      for (unsigned int j = 0; j < numBulbs; j++) {
         tmc[0] = float(bulbColors[j*3+0]);
         tmc[1] = float(bulbColors[j*3+1]);
         tmc[2] = float(bulbColors[j*3+2]);
         tmc[3] = alpha;

         for (unsigned int i = 0; i < limit; i++) {

            TLx = constrain(float(-1.0f + i*offset + j*offset*limit), -0.75f, 0.75f);
            TLy =  1.00f;

            BLx = constrain(float(-1.0f + i*offset + j*offset*limit), -0.75f, 0.75f);
            BLy = -1.00f;

            TRx = constrain(float(-1.0f + (i+1)*offset + j*offset*limit), -0.75f, 0.75f);
            TRy =  1.00f;

            BRx = constrain(float(-1.0f + (i+1)*offset + j*offset*limit), -0.75f, 0.75f);
            BRy = -1.00f;

            defineQuad4pt(
                  TLx, TLy,
                  BLx, BLy,
                  TRx, TRy,
                  BRx, BRy,
                  tmc,           // Quad color
                  verts, colrs); // Input Vectors
         }
      }

      // Draw rounded corners for end slices
      tmc[0] = float(bulbColors[0]);
      tmc[1] = float(bulbColors[1]);
      tmc[2] = float(bulbColors[2]);
      tmc[3] = alpha;
      defineArch(
            -0.75, 0.75f,  // X, Y positional coordinates
            0.0f, 0.0f,    // X, Y inner radii
            90.0f, 180.0f, // Start/end arch angles (degrees, unit circle)
            0.25f,         // Width of arch from inner radii
            segments/3,    // Number of polygons
            tmc,           // Arch Color
            verts, colrs); // Input Vectors

      defineArch(
            -0.75, -0.75f,    // X, Y positional coordinates
            0.0f, 0.0f,       // X, Y inner radii
            180.0f, 270.0f,   // Start/end arch angles (degrees, unit circle)
            0.25f,            // Width of arch from inner radii
            segments/3,       // Number of polygons
            tmc,              // Arch Color
            verts, colrs);    // Input Vectors
      defineQuad4pt(
            -0.75f, +0.75f,
            -0.75f, -0.75f,
            -1.00f, +0.75f,
            -1.00f, -0.75f,
            tmc,              // Quad color
            verts, colrs);    // Input Vectors

      tmc[0] = float(bulbColors[(numBulbs-1)*3+0]);
      tmc[1] = float(bulbColors[(numBulbs-1)*3+1]);
      tmc[2] = float(bulbColors[(numBulbs-1)*3+2]);
      tmc[3] = alpha;
      defineArch(
            0.75, -0.75f,     // X, Y positional coordinates
            0.0f, 0.0f,       // X, Y inner radii
            270.0f, 360.0f,   // Start/end arch angles (degrees, unit circle)
            0.25f,            // Width of arch from inner radii
            segments/3,       // Number of polygons
            tmc,              // Arch Color
            verts, colrs);    // Input Vectors

      defineArch(
            0.75, +0.75f,  // X, Y positional coordinates
            0.0f, 0.0f,    // X, Y inner radii
            360.0f, 90.0f, // Start/end arch angles (degrees, unit circle)
            0.25f,         // Width of arch from inner radii
            segments/3,    // Number of polygons
            tmc,           // Arch Color
            verts, colrs); // Input Vectors
      defineQuad4pt(
            0.75f, +0.75f,
            0.75f, -0.75f,
            1.00f, +0.75f,
            1.00f, -0.75f,
            tmc,           // Quad color
            verts, colrs); // Input Vectors
            

      // Define OutLine
      if (featureLevel >= 1) {
         featScale = 9.0f / 8.0f;
      } else {
         featScale = 1.0f;
      }

      // Draw Outer Straights
      defineQuad4pt(
            featScale, 0.75f,
            featScale, -0.75f,
            1.0f, 0.75f,
            1.0f, -0.75f,
            detailColor,      // Quad Color
            verts, colrs);    // Input Vectors
      defineQuad4pt(
            -featScale, 0.75f,
            -featScale, -0.75f,
            -1.0f, 0.75f,
            -1.0f, -0.75f,
            detailColor,      // Quad Color
            verts, colrs);    // Input Vectors
      defineQuad4pt(
            -0.75f, featScale,
            0.75f, featScale, 
            -0.75f, 1.0f,
            0.75f, 1.0f, 
            detailColor,      // Quad Color
            verts, colrs);    // Input Vectors
      defineQuad4pt(
            -0.75f, -featScale,
            0.75f, -featScale, 
            -0.75f, -1.0f,
            0.75f, -1.0f, 
            detailColor,      // Quad Color
            verts, colrs);    // Input Vectors

      if (featureLevel >= 1) {
         featScale = 1.0f / 8.0f;
      } else {
         featScale = 0.0f;
      }

      // Draw Rounded Corners
      defineArch(
            0.75f, +0.75f, // X, Y positional coordinates
            0.25f, 0.25f,  // X, Y inner radii
            360.0f, 90.0f, // Start/end arch angles (degrees, unit circle)
            featScale,     // Width of arch from inner radii
            segments/3,    // Number of polygons
            detailColor,   // Arch Color
            verts, colrs); // Input Vectors
      defineArch(
            -0.75f, +0.75f,// X, Y positional coordinates
            0.25f, 0.25f,  // X, Y inner radii
            90.0f, 180.0f, // Start/end arch angles (degrees, unit circle)
            featScale,     // Width of arch from inner radii
            segments/3,    // Number of polygons
            detailColor,   // Arch Color
            verts, colrs); // Input Vectors
      defineArch(
            -0.75f, -0.75f,   // X, Y positional coordinates
            0.25f, 0.25f,     // X, Y inner radii
            180.0f, 270.0f,   // Start/end arch angles (degrees, unit circle)
            featScale,        // Width of arch from inner radii
            segments/3,       // Number of polygons
            detailColor,      // Arch Color
            verts, colrs);    // Input Vectors
      defineArch(
            0.75f, -0.75f,    // X, Y positional coordinates
            0.25f, 0.25f,     // X, Y inner radii
            270.0f, 360.0f,   // Start/end arch angles (degrees, unit circle)
            featScale,        // Width of arch from inner radii
            segments/3,       // Number of polygons
            detailColor,      // Arch Color
            verts, colrs);    // Input Vectors

      // Define Bulb Markers
      float tmx, 
            tmy = 17.0f / 16.0f, 
            tmb = (float)numBulbs;

      // Define Geometry/Colors for one bulb
      std::vector<float> BMverts;
      std::vector<float> BMcolrs;
      defineEllipse(
            0.0f, 0.0f,
            1.0f/6.0f, 1.0f/6.0f,
            circleSegments/3,
            detailColor,
            BMverts, BMcolrs);

      for (unsigned int i = 0; i < 6; i++) {
         if (featureLevel >= 2.0 && i < numBulbs) {
            featScale = 1.0f;
         } else {
            featScale = 0.0f;
         }

         tmx = float(-1.0f + 1.0f/tmb + (i*2.0f)/tmb);

         // Copy Pre-calculated Geometry + colors
         for (unsigned int j = 0; j < BMverts.size()/2; j++) {
            verts.push_back(featScale*(BMverts[j*2+0]+tmx));
            verts.push_back(featScale*(BMverts[j*2+1]+tmy));
            colrs.push_back(BMcolrs[j*4+0]);
            colrs.push_back(BMcolrs[j*4+1]);
            colrs.push_back(BMcolrs[j*4+2]);
            colrs.push_back(BMcolrs[j*4+3]);
         }
      }

      // Define Bulb Halos

      // Define Geometry/Colors for one halo
      std::vector<float> BHverts;
      std::vector<float> BHcolrs;
      defineArch(
            0.0f, 0.0f,    // X, Y positional coordinates
            0.22f, 0.22f,  // X, Y inner radii
            0.0f, 360.0f,  // Start/end arch angles (degrees, unit circle)
            1.0f/13.0f,    // Width of arch from inner radii
            segments/3,    // Number of polygons
            detailColor,   // Arch Color
            BHverts, BHcolrs); // Input Vectors

      for (int i = 0; i < 6; i++) {
         if (featureLevel >= 3 && i < numBulbs) {
            //featScale = 1.0f / 13.0f;
            featScale = 1.0f;
         } else {
            featScale = 0.0f;
         }

         tmx = float(-1.0f + 1.0f/tmb + (i*2.0f)/tmb);

         unsigned int bulbStart = verts.size()/2;

         // Copy Pre-calculated Geometry + colors
         for (unsigned int j = 0; j < BHverts.size()/2; j++) {
            verts.push_back(featScale*(BHverts[j*2+0]+tmx));
            verts.push_back(featScale*(BHverts[j*2+1]+tmy));
            colrs.push_back(BHcolrs[j*4+0]);
            colrs.push_back(BHcolrs[j*4+1]);
            colrs.push_back(BHcolrs[j*4+2]);
            colrs.push_back(BHcolrs[j*4+3]);
         }

         // Prevent Bulb Halos from overlaping, clip overextending vertices
         float clipLimit = 1.0f / tmb;
         if ( i == 0 )
            for (unsigned int i = bulbStart; i < verts.size()/2; i++)
               verts[i*2] = constrain(verts[i*2], -2.0f, tmx+clipLimit);
         else if ( i == numBulbs-1 )
            for (unsigned int i = bulbStart; i < verts.size()/2; i++)
               verts[i*2] = constrain(verts[i*2], tmx-clipLimit, 2.0f);
         else
            for (unsigned int i = bulbStart; i < verts.size()/2; i++)
               verts[i*2] = constrain(verts[i*2], tmx-clipLimit, tmx+clipLimit);
      }

      // Define Grand Outline
      if (featureLevel >= 4) {
         featScale = 1.0f;
      } else {
         featScale = 0.0f;
      }

      // Draw Outer Straights
      defineQuad4pt(
            -0.75f*featScale, (17.0f/16.0f + 13.0f/60.0f)*featScale,
            -0.75f*featScale, (17.0f/16.0f + 18.0f/60.0f)*featScale,
            +0.75f*featScale, (17.0f/16.0f + 13.0f/60.0f)*featScale,
            +0.75f*featScale, (17.0f/16.0f + 18.0f/60.0f)*featScale,
            detailColor,
            verts, colrs);
      defineQuad4pt(
            -0.75f*featScale, -(17.0f/16.0f + 13.0f/60.0f)*featScale,
            -0.75f*featScale, -(17.0f/16.0f + 18.0f/60.0f)*featScale,
            +0.75f*featScale, -(17.0f/16.0f + 13.0f/60.0f)*featScale,
            +0.75f*featScale, -(17.0f/16.0f + 18.0f/60.0f)*featScale,
            detailColor,
            verts, colrs);
      defineQuad4pt(
            (17.0f/16.0f + 13.0f/60.0f)*featScale, -0.75f*featScale,
            (17.0f/16.0f + 18.0f/60.0f)*featScale, -0.75f*featScale,
            (17.0f/16.0f + 13.0f/60.0f)*featScale, +0.75f*featScale,
            (17.0f/16.0f + 18.0f/60.0f)*featScale, +0.75f*featScale,
            detailColor,
            verts, colrs);
      defineQuad4pt(
            -(17.0f/16.0f + 13.0f/60.0f)*featScale, -0.75f*featScale,
            -(17.0f/16.0f + 18.0f/60.0f)*featScale, -0.75f*featScale,
            -(17.0f/16.0f + 13.0f/60.0f)*featScale, +0.75f*featScale,
            -(17.0f/16.0f + 18.0f/60.0f)*featScale, +0.75f*featScale,
            detailColor,
            verts, colrs);

      // Draw Rounded Corners
      float ri = 5.0f/16.0f+13.0f/60.0f,
            ro = 5.0f/60.0f;

      defineArch(
            0.75f, 0.75f,  // X, Y positional coordinates
            ri, ri,        // X, Y inner radii
            0.0f, 90.0f,   // Start/End arch angles (degrees, unit circle)
            ro*featScale,  // Width of arch from inner radii
            segments/2,    // Number of polygons
            detailColor,   // Arch color
            verts, colrs); // Input vectors
      defineArch(
            -0.75f, 0.75f, // X, Y positional coordinates
            ri, ri,        // X, Y inner radii
            90.0f, 180.0f, // Start/End arch angles (degrees, unit circle)
            ro*featScale,  // Width of arch from inner radii
            segments/2,    // Number of polygons
            detailColor,   // Arch color
            verts, colrs); // Input vectors
      defineArch(
            -0.75f, -0.75f,// X, Y positional coordinates
            ri, ri,        // X, Y inner radii
            180.0f, 270.0f,// Start/End arch angles (degrees, unit circle)
            ro*featScale,  // Width of arch from inner radii
            segments/2,    // Number of polygons
            detailColor,   // Arch color
            verts, colrs); // Input vectors
      defineArch(
            0.75f, -0.75f, // X, Y positional coordinates
            ri, ri,        // X, Y inner radii
            270.0f, 360.0f,// Start/End arch angles (degrees, unit circle)
            ro*featScale,  // Width of arch from inner radii
            segments/2,    // Number of polygons
            detailColor,   // Arch color
            verts, colrs); // Input vectors

   return verts.size()/2;
}

unsigned int updateIconLinearGeometry(
      float          bx,               // X-Coordinate
      float          by,               // Y-Coordinate
      float          bs,               // Scale 2.0=spans display before GL scaling
      unsigned int   featureLevel,     // level of featureLevel
      unsigned int   circleSegments,   // Number of sides
      unsigned char  numBulbs,         // number of bulbs
      unsigned int   index,            // Index of where to start writing in input array
      float*         verts             // Input Vector of x,y coordinates
      ){

      float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy, featScale;
      float offset = 2.0f / 60.0f;
      unsigned int segments = 60;
      unsigned int limit = segments/numBulbs;
      unsigned int subIndex = index;

      // Draw Block of stripes
      for (unsigned int j = 0; j < numBulbs; j++) {
         for (unsigned int i = 0; i < limit; i++) {

            TLx = constrain(float(-1.0f + i*offset + j*offset*limit), -0.75f, 0.75f);
            TLy =  1.00f;

            BLx = constrain(float(-1.0f + i*offset + j*offset*limit), -0.75f, 0.75f);
            BLy = -1.00f;

            TRx = constrain(float(-1.0f + (i+1)*offset + j*offset*limit), -0.75f, 0.75f);
            TRy =  1.00f;

            BRx = constrain(float(-1.0f + (i+1)*offset + j*offset*limit), -0.75f, 0.75f);
            BRy = -1.00f;

            subIndex = updateQuad4ptGeometry(
                  TLx, TLy,
                  BLx, BLy,
                  TRx, TRy,
                  BRx, BRy,
                  subIndex,
                  verts); // Input Array
         }
      }

      // Draw rounded corners for end slices
      subIndex = updateArchGeometry(
            -0.75, 0.75f,  // X, Y positional coordinates
            0.0f, 0.0f,    // X, Y inner radii
            90.0f, 180.0f, // Start/end arch angles (degrees, unit circle)
            0.25f,         // Width of arch from inner radii
            segments/3,    // Number of polygons
            subIndex,      // Index of where to start writing
            verts);        // Input Array

      subIndex = updateArchGeometry(
            -0.75, -0.75f,    // X, Y positional coordinates
            0.0f, 0.0f,       // X, Y inner radii
            180.0f, 270.0f,   // Start/end arch angles (degrees, unit circle)
            0.25f,            // Width of arch from inner radii
            segments/3,       // Number of polygons
            subIndex,         // Index of where to start writing
            verts);           // Input Array
      subIndex = updateQuad4ptGeometry(
            -0.75f, +0.75f,
            -0.75f, -0.75f,
            -1.00f, +0.75f,
            -1.00f, -0.75f,
            subIndex,   // Index of where to start writing
            verts);     // Input Array

      subIndex = updateArchGeometry(
            0.75, -0.75f,     // X, Y positional coordinates
            0.0f, 0.0f,       // X, Y inner radii
            270.0f, 360.0f,   // Start/end arch angles (degrees, unit circle)
            0.25f,            // Width of arch from inner radii
            segments/3,       // Number of polygons
            subIndex,         // Index of where to start writing
            verts);           // Input Array

      subIndex = updateArchGeometry(
            0.75, +0.75f,  // X, Y positional coordinates
            0.0f, 0.0f,    // X, Y inner radii
            360.0f, 90.0f, // Start/end arch angles (degrees, unit circle)
            0.25f,         // Width of arch from inner radii
            segments/3,    // Number of polygons
            subIndex,      // Index of where to start writing
            verts);        // Input Array
      subIndex = updateQuad4ptGeometry(
            0.75f, +0.75f,
            0.75f, -0.75f,
            1.00f, +0.75f,
            1.00f, -0.75f,
            subIndex,   // Index of where to start writing
            verts);     // Input Array
            

      // Define OutLine
      if (featureLevel >= 1) {
         featScale = 9.0f / 8.0f;
      } else {
         featScale = 1.0f;
      }

      // Draw Outer Straights
      subIndex = updateQuad4ptGeometry(
            featScale, 0.75f,
            featScale, -0.75f,
            1.0f, 0.75f,
            1.0f, -0.75f,
            subIndex,   // Index of where to start writing
            verts);     // Input Array
      subIndex = updateQuad4ptGeometry(
            -featScale, 0.75f,
            -featScale, -0.75f,
            -1.0f, 0.75f,
            -1.0f, -0.75f,
            subIndex,   // Index of where to start writing
            verts);     // Input Array
      subIndex = updateQuad4ptGeometry(
            -0.75f, featScale,
            0.75f, featScale, 
            -0.75f, 1.0f,
            0.75f, 1.0f, 
            subIndex,   // Index of where to start writing
            verts);     // Input Array
      subIndex = updateQuad4ptGeometry(
            -0.75f, -featScale,
            0.75f, -featScale, 
            -0.75f, -1.0f,
            0.75f, -1.0f, 
            subIndex,   // Index of where to start writing
            verts);     // Input Array

      if (featureLevel >= 1) {
         featScale = 1.0f / 8.0f;
      } else {
         featScale = 0.0f;
      }

      // Draw Rounded Corners
      subIndex = updateArchGeometry(
            0.75f, +0.75f, // X, Y positional coordinates
            0.25f, 0.25f,  // X, Y inner radii
            360.0f, 90.0f, // Start/end arch angles (degrees, unit circle)
            featScale,     // Width of arch from inner radii
            segments/3,    // Number of polygons
            subIndex,      // Index of where to start writing
            verts);        // Input Array
      subIndex = updateArchGeometry(
            -0.75f, +0.75f,// X, Y positional coordinates
            0.25f, 0.25f,  // X, Y inner radii
            90.0f, 180.0f, // Start/end arch angles (degrees, unit circle)
            featScale,     // Width of arch from inner radii
            segments/3,    // Number of polygons
            subIndex,      // Index of where to start writing
            verts);        // Input Array
      subIndex = updateArchGeometry(
            -0.75f, -0.75f,   // X, Y positional coordinates
            0.25f, 0.25f,     // X, Y inner radii
            180.0f, 270.0f,   // Start/end arch angles (degrees, unit circle)
            featScale,        // Width of arch from inner radii
            segments/3,       // Number of polygons
            subIndex,         // Index of where to start writing
            verts);           // Input Array
      subIndex = updateArchGeometry(
            0.75f, -0.75f,    // X, Y positional coordinates
            0.25f, 0.25f,     // X, Y inner radii
            270.0f, 360.0f,   // Start/end arch angles (degrees, unit circle)
            featScale,        // Width of arch from inner radii
            segments/3,       // Number of polygons
            subIndex,         // Index of where to start writing
            verts);           // Input Array

      // Define Bulb Markers
      float tmx, 
            tmy = 17.0f / 16.0f, 
            tmb = (float)numBulbs;

      // Define Geometry/Colors for one bulb
      std::vector<float> BMverts;
      std::vector<float> BMcolrs;
      float tmc[4] = {1.0f, 1.0f, 1.0f, 1.0f};
      defineEllipse(
            0.0f, 0.0f,
            1.0f/6.0f, 1.0f/6.0f,
            circleSegments/3,
            tmc,
            BMverts, BMcolrs);

      // Multiply subIndex by 2 since we will be operating directly on the input array
      subIndex *= 2;

      for (unsigned int i = 0; i < 6; i++) {
         if (featureLevel >= 2.0 && i < numBulbs) {
            featScale = 1.0f;
         } else {
            featScale = 0.0f;
         }

         tmx = float(-1.0f + 1.0f/tmb + (i*2.0f)/tmb);

         // Copy Pre-calculated Geometry + colors
         for (unsigned int j = 0; j < BMverts.size()/2; j++) {
            verts[subIndex++] = featScale*(BMverts[j*2+0]+tmx);
            verts[subIndex++] = featScale*(BMverts[j*2+1]+tmy);
         }
      }

      // Define Bulb Halos

      // Define Geometry/Colors for one halo
      std::vector<float> BHverts;
      std::vector<float> BHcolrs;
      defineArch(
            0.0f, 0.0f,    // X, Y positional coordinates
            0.22f, 0.22f,  // X, Y inner radii
            0.0f, 360.0f,  // Start/end arch angles (degrees, unit circle)
            1.0f/13.0f,    // Width of arch from inner radii
            segments/3,    // Number of polygons
            tmc,
            BHverts, BHcolrs); // Input Vectors

      for (int i = 0; i < 6; i++) {
         if (featureLevel >= 3 && i < numBulbs) {
            featScale = 1.0f;
         } else {
            featScale = 0.0f;
         }

         tmx = float(-1.0f + 1.0f/tmb + (i*2.0f)/tmb);

         unsigned int bulbStart = subIndex/2;

         // Copy Pre-calculated Geometry + colors
         for (unsigned int j = 0; j < BHverts.size()/2; j++) {
            verts[subIndex++] = featScale*(BHverts[j*2+0]+tmx);
            verts[subIndex++] = featScale*(BHverts[j*2+1]+tmy);
         }

         // Prevent Bulb Halos from overlaping, clip overextending vertices
         float clipLimit = 1.0f / tmb;
         if ( i == 0 )
            for (unsigned int i = bulbStart; i < subIndex/2; i++)
               verts[i*2] = constrain(verts[i*2], -2.0f, tmx+clipLimit);
         else if ( i == numBulbs-1 )
            for (unsigned int i = bulbStart; i < subIndex/2; i++)
               verts[i*2] = constrain(verts[i*2], tmx-clipLimit, 2.0f);
         else
            for (unsigned int i = bulbStart; i < subIndex/2; i++)
               verts[i*2] = constrain(verts[i*2], tmx-clipLimit, tmx+clipLimit);
      }

      // Revert index to reflect the index of the vertex not its coordinate pair (X, Y)
      subIndex /= 2;

      // Define Grand Outline
      if (featureLevel >= 4) {
         featScale = 1.0f;
      } else {
         featScale = 0.0f;
      }

      // Draw Outer Straights
      subIndex = updateQuad4ptGeometry(
            -0.75f*featScale, (17.0f/16.0f + 13.0f/60.0f)*featScale,
            -0.75f*featScale, (17.0f/16.0f + 18.0f/60.0f)*featScale,
            +0.75f*featScale, (17.0f/16.0f + 13.0f/60.0f)*featScale,
            +0.75f*featScale, (17.0f/16.0f + 18.0f/60.0f)*featScale,
            subIndex,   // Index of where to start writing
            verts);     // Input Array
      subIndex = updateQuad4ptGeometry(
            -0.75f*featScale, -(17.0f/16.0f + 13.0f/60.0f)*featScale,
            -0.75f*featScale, -(17.0f/16.0f + 18.0f/60.0f)*featScale,
            +0.75f*featScale, -(17.0f/16.0f + 13.0f/60.0f)*featScale,
            +0.75f*featScale, -(17.0f/16.0f + 18.0f/60.0f)*featScale,
            subIndex,   // Index of where to start writing
            verts);     // Input Array
      subIndex = updateQuad4ptGeometry(
            (17.0f/16.0f + 13.0f/60.0f)*featScale, -0.75f*featScale,
            (17.0f/16.0f + 18.0f/60.0f)*featScale, -0.75f*featScale,
            (17.0f/16.0f + 13.0f/60.0f)*featScale, +0.75f*featScale,
            (17.0f/16.0f + 18.0f/60.0f)*featScale, +0.75f*featScale,
            subIndex,   // Index of where to start writing
            verts);     // Input Array
      subIndex = updateQuad4ptGeometry(
            -(17.0f/16.0f + 13.0f/60.0f)*featScale, -0.75f*featScale,
            -(17.0f/16.0f + 18.0f/60.0f)*featScale, -0.75f*featScale,
            -(17.0f/16.0f + 13.0f/60.0f)*featScale, +0.75f*featScale,
            -(17.0f/16.0f + 18.0f/60.0f)*featScale, +0.75f*featScale,
            subIndex,   // Index of where to start writing
            verts);     // Input Array

      // Draw Rounded Corners
      float ri = 5.0f/16.0f+13.0f/60.0f,
            ro = 5.0f/60.0f;

      subIndex = updateArchGeometry(
            0.75f, 0.75f,  // X, Y positional coordinates
            ri, ri,        // X, Y inner radii
            0.0f, 90.0f,   // Start/End arch angles (degrees, unit circle)
            ro*featScale,  // Width of arch from inner radii
            segments/2,    // Number of polygons
            subIndex,      // Index of where to start writing
            verts);        // Input Array
      subIndex = updateArchGeometry(
            -0.75f, 0.75f, // X, Y positional coordinates
            ri, ri,        // X, Y inner radii
            90.0f, 180.0f, // Start/End arch angles (degrees, unit circle)
            ro*featScale,  // Width of arch from inner radii
            segments/2,    // Number of polygons
            subIndex,      // Index of where to start writing
            verts);        // Input Array
      subIndex = updateArchGeometry(
            -0.75f, -0.75f,// X, Y positional coordinates
            ri, ri,        // X, Y inner radii
            180.0f, 270.0f,// Start/End arch angles (degrees, unit circle)
            ro*featScale,  // Width of arch from inner radii
            segments/2,    // Number of polygons
            subIndex,      // Index of where to start writing
            verts);        // Input Array
      subIndex = updateArchGeometry(
            0.75f, -0.75f, // X, Y positional coordinates
            ri, ri,        // X, Y inner radii
            270.0f, 360.0f,// Start/End arch angles (degrees, unit circle)
            ro*featScale,  // Width of arch from inner radii
            segments/2,    // Number of polygons
            subIndex,      // Index of where to start writing
            verts);        // Input Array

   return subIndex;
}

unsigned int updateIconLinearColor(
      //unsigned int   featureLevel,     // level of featureLevel
      unsigned int   circleSegments,   // Number of sides
      unsigned char  numBulbs,         // number of bulbs
      float          alpha,            // alpha transparency value
      float*         bulbColors,       // colors of the bulbs
      float*         detailColor,      // color of the accent details
      unsigned int   index,            // index of where to start writing to input array
      float*         colrs             // Input Vector of r,g,b values
      ){

      unsigned int subIndex = index;
      //float TLx, TRx, BLx, BRx, TLy, TRy, BLy, BRy, featScale;
      //float offset = 2.0f / 60.0f;
      float tmc[4];
      unsigned int segments = 60;
      unsigned int limit = segments/numBulbs;

      // Draw Block of stripes
      for (unsigned int j = 0; j < numBulbs; j++) {
         tmc[0] = float(bulbColors[j*3+0]);
         tmc[1] = float(bulbColors[j*3+1]);
         tmc[2] = float(bulbColors[j*3+2]);
         tmc[3] = alpha;

         for (unsigned int i = 0; i < limit; i++) {

            subIndex = updateQuadColor(
                  tmc,
                  subIndex,
                  colrs);
         }
      }

      // Draw rounded corners for end slices
      tmc[0] = float(bulbColors[0]);
      tmc[1] = float(bulbColors[1]);
      tmc[2] = float(bulbColors[2]);
      tmc[3] = alpha;
      subIndex = updateArchColor(
            segments/3,    // Number of polygons
            tmc,           // Arch Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateArchColor(
            segments/3,    // Number of polygons
            tmc,           // Arch Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateQuadColor(
            tmc,           // Quad color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array

      tmc[0] = float(bulbColors[(numBulbs-1)*3+0]);
      tmc[1] = float(bulbColors[(numBulbs-1)*3+1]);
      tmc[2] = float(bulbColors[(numBulbs-1)*3+2]);
      tmc[3] = alpha;
      subIndex = updateArchColor(
            segments/3,    // Number of polygons
            tmc,           // Arch Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateArchColor(
            segments/3,    // Number of polygons
            tmc,           // Arch Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateQuadColor(
            tmc,           // Quad color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
            

      // Define OutLine
      // Draw Outer Straights
      subIndex = updateQuadColor(
            detailColor,   // Quad Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateQuadColor(
            detailColor,   // Quad Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateQuadColor(
            detailColor,   // Quad Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateQuadColor(
            detailColor,   // Quad Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array

      // Draw Rounded Corners
      subIndex = updateArchColor(
            segments/3,    // Number of polygons
            detailColor,   // Arch Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateArchColor(
            segments/3,    // Number of polygons
            detailColor,   // Arch Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateArchColor(
            segments/3,    // Number of polygons
            detailColor,   // Arch Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateArchColor(
            segments/3,    // Number of polygons
            detailColor,   // Arch Color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array

      // Define Bulb Markers
      // Define Geometry/Colors for one bulb
      std::vector<float> BMverts;
      std::vector<float> BMcolrs;
      defineEllipse(
            0.0f, 0.0f,
            1.0f/6.0f, 1.0f/6.0f,
            circleSegments/3,
            detailColor,
            BMverts, BMcolrs);

      // Multiply subIndex by 4 since we will be operating directly on the input array
      subIndex *= 4;

      for (unsigned int i = 0; i < 6; i++) {
         // Copy Pre-calculated Geometry + colors
         for (unsigned int j = 0; j < BMverts.size()/2; j++) {
            colrs[subIndex++] = BMcolrs[j*4+0];
            colrs[subIndex++] = BMcolrs[j*4+1];
            colrs[subIndex++] = BMcolrs[j*4+2];
            colrs[subIndex++] = BMcolrs[j*4+3];
         }
      }

      // Define Bulb Halos

      // Define Geometry/Colors for one halo
      std::vector<float> BHverts;
      std::vector<float> BHcolrs;
      defineArch(
            0.0f, 0.0f,    // X, Y positional coordinates
            0.22f, 0.22f,  // X, Y inner radii
            0.0f, 360.0f,  // Start/end arch angles (degrees, unit circle)
            1.0f/13.0f,    // Width of arch from inner radii
            segments/3,    // Number of polygons
            detailColor,   // Arch Color
            BHverts, BHcolrs); // Input Vectors

      for (int i = 0; i < 6; i++) {
         unsigned int bulbStart = subIndex/4;

         // Copy Pre-calculated Geometry + colors
         for (unsigned int j = 0; j < BHverts.size()/2; j++) {
            colrs[subIndex++] = BHcolrs[j*4+0];
            colrs[subIndex++] = BHcolrs[j*4+1];
            colrs[subIndex++] = BHcolrs[j*4+2];
            colrs[subIndex++] = BHcolrs[j*4+3];
         }
      }
      
      // Revert index to reflect the index of the vertex not its coordinate pair (X, Y)
      subIndex /= 4;

      // Define Grand Outline
      // Draw Outer Straights
      subIndex = updateQuadColor(
            detailColor,
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateQuadColor(
            detailColor,
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateQuadColor(
            detailColor,
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateQuadColor(
            detailColor,
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array

      // Draw Rounded Corners
      subIndex = updateArchColor(
            segments/2,    // Number of polygons
            detailColor,   // Arch color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateArchColor(
            segments/2,    // Number of polygons
            detailColor,   // Arch color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateArchColor(
            segments/2,    // Number of polygons
            detailColor,   // Arch color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array
      subIndex = updateArchColor(
            segments/2,    // Number of polygons
            detailColor,   // Arch color
            subIndex,      // Index of where to start writing to
            colrs);        // Input Array

   return subIndex;
}
