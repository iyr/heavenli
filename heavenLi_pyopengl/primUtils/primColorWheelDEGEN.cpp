/*
 * defines a pizza-style color wheel
 */
int defineColorWheel(
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

   float degPerCol = float(360.0 / float(numColors));
   float tmc[4];
      
   for (unsigned int j = 0; j < numColors; j++) {
      tmc[0] = float(colors[j*3+0]);
      tmc[1] = float(colors[j*3+1]);
      tmc[2] = float(colors[j*3+2]);
      tmc[3] = alpha;

      defineArch(
            bx, by,
            0.0f, 0.0f,
            (j+0)*degPerCol+90.0f+angOffset,
            (j+1)*degPerCol+90.0f+angOffset,
            bs,
            circleSegments/numColors,
            tmc,
            verts,
            colrs);
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

      subIndex = updateArchColor(
            circleSegments/numColors,
            tmc,
            subIndex,
            colrs);
   }

   return subIndex;
}

/*
int updateColorWheelGeometry(
      float          bx,               // X-Coordinate                              
      float          by,               // Y-Coordinate                              
      float          bs,               // Scale 2.0=spans display before GL scaling 
      unsigned int   circleSegments,   // Number of sides                           
      float          angOffset,        // How much, in degrees, to rotate           
      int            index,            // Index of where to start writing to array  
      float*         verts             // Input Array of x,y coordinates            
      ){

   //char degSegment = 360 / circleSegments;
   float degPerCol = float(360.0 / float(numColors));
      
   for (unsigned int j = 0; j < numColors; j++) {
      updateArchGeometry(
            bx, by,
            0.0f, 0.0f,
            (j+0)*degPerCol+90.0f+angOffset,
            (j+1)*degPerCol+90.0f+angOffset,
            bs,
            circleSegments/numColors,
            tmc,
            verts,
            colrs);
   }

   delete [] tmc;

   return verts.size()/2;
}
*/
