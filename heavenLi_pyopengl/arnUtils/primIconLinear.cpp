/*
 * Defines icon for lamps with linear arrangements
 */
unsigned int defineIconLinear(
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

   return verts.size()/2;
}
