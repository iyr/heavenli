/*
 * defines a pizza-style color wheel
 */
unsigned int defineColorWheel(
      float          bx,               // X-Coordinate
      float          by,               // Y-Coordinate
      float          bs,               // Scale 2.0=spans display before GL scaling
      unsigned int   circleSegments,   // Number of sides
      float          angOffset,        // How much, in degrees, to rotate
      unsigned char  numColors,        // number of color slices
      float          alpha,            // alpha transparency value
      float*         colors,           // colors of the wheel slices
      std::vector<float> &verts,       // Input Vector of x,y coordinates
      std::vector<float> &colrs        // Input Vector of r,g,b values
      ){

   float degPerCol = 360.0f / (float)circleSegments;
   float colOffset = 360.0f / (float)numColors;
   float tmc[4];
      
   for (unsigned int j = 0; j < numColors; j++) {
      tmc[0] = float(colors[j*3+0]);
      tmc[1] = float(colors[j*3+1]);
      tmc[2] = float(colors[j*3+2]);
      tmc[3] = alpha;

      for (unsigned int i = 0; i < circleSegments/numColors; i++ ){

         defineArch(
               bx, by,
               0.0f, 0.0f,
               i*degPerCol + j*colOffset - 90.0f,
               (i+1)*degPerCol + j*colOffset - 90.0f,
               bs,
               1,
               tmc,
               verts,
               colrs);
      }
   }

   return verts.size()/2;
}

unsigned int updateColorWheelColor(
      unsigned int   circleSegments,   // Number of sides
      unsigned char  numColors,        // number of color slices
      float          alpha,            // alpha transparency value
      float*         colors,           // colors of the wheel slices
      unsigned int   index,            // Index of where to start writing to input arrays
      float*         colrs             // Input array of r,g,b,a values to be updated
      ){

   float tmc[4];
   unsigned int subIndex = index;
      
   for (unsigned int j = 0; j < numColors; j++) {
      tmc[0] = float(colors[j*3+0]);
      tmc[1] = float(colors[j*3+1]);
      tmc[2] = float(colors[j*3+2]);
      tmc[3] = alpha;

      for (unsigned int i = 0; i < circleSegments/numColors; i++ ){

         subIndex = updateArchColor(
               1,
               tmc,
               subIndex,            // Index of where to start writing to input arrays
               colrs);
      }
   }

   return subIndex;
}

unsigned int updateColorWheelGeometry(
      float          bx,               // X-Coordinate
      float          by,               // Y-Coordinate
      float          bs,               // Scale 2.0=spans display before GL scaling
      unsigned int   circleSegments,   // Number of sides
      float          angOffset,        // How much, in degrees, to rotate
      unsigned char  numColors,        // number of color slices
      unsigned int   index,            // Index of where to start writing to input arrays
      float*         verts             // Input array of x,y values to be updated
      ){

   float degPerCol = 360.0f / (float)circleSegments;
   float colOffset = 360.0f / (float)numColors;
   unsigned int subIndex = index;
      
   for (unsigned int j = 0; j < numColors; j++) {

      for (unsigned int i = 0; i < circleSegments/numColors; i++ ){

         subIndex = updateArchGeometry(
               bx, by,
               0.0f, 0.0f,
               i*degPerCol + j*colOffset - 90.0f,
               (i+1)*degPerCol + j*colOffset - 90.0f,
               bs,
               1,
               subIndex,
               verts);
      }
   }

   return subIndex;
}
