// Convenience functions
#define degToRad(angleindegrees) ((angleindegrees) * 3.1415926535 / 180.0)
float constrain(
      float value,   // Input value to constrain
      float min,     // Minimum
      float max      // Maximum
      ) {
   if (value > max)
      return max;
   else if (value < min)
      return min;
   else
      return value;
}

/*
unsigned int constrain(
      unsigned int value,   // Input value to constrain
      unsigned int min,     // Minimum
      unsigned int max      // Maximum
      ) {
   if (value > max)
      return max;
   else if (value < min)
      return min;
   else
      return value;
}
*/

int constrain(
      int value,   // Input value to constrain
      int min,     // Minimum
      int max      // Maximum
      ) {
   if (value > max)
      return max;
   else if (value < min)
      return min;
   else
      return value;
}

float rangeShift(
      float value,   // Input value to shift
      float oldMin,  // Input value range minimum
      float oldMax,  // Input value range maximum
      float newMin,  // Output value range minimum
      float newMax   // Output value range maximum
      ) {
   float oldRange = (oldMax - oldMin);

   if (oldRange == 0.0f) {
      return newMin;
   } else {
      float newRange = (newMax - newMin);
      float tmv = (((value - oldMin)*newRange) / oldRange) + newMin;
      return tmv;
   }
}

#include "primUtils/primEllipseDEGEN.cpp"
#include "primUtils/primArchDEGEN.cpp"
#include "primUtils/primPillDEGEN.cpp"
#include "primUtils/primQuadDEGEN.cpp"
#include "primUtils/primRoundRectDEGEN.cpp"

#include "primUtils/primColorWheelDEGEN.cpp"
#include "primUtils/primBulbDEGEN.cpp"

#include "primUtils/defineElementCoords.cpp"
#include "primUtils/primMenuOverflowDEGEN.cpp"
#include "primUtils/primMenuNormalDEGEN.cpp"

#include "primUtils/primTexQuadDEGEN.cpp"
#include "primUtils/primTexEllipseDEGEN.cpp"

#include "primUtils/drawArch.cpp"
#include "primUtils/drawEllipse.cpp"
#include "primUtils/drawPill.cpp"

#include "primUtils/hsv2rgb.cpp"
#include "primUtils/rgb2hsv.cpp"

// :v
