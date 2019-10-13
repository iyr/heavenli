
// Convenience functions
#define degToRad(angleindegrees) ((angleindegrees) * 3.1415926535 / 180.0)
float constrain(float value, float min, float max) {
   if (value > max)
      return max;
   else if (value < min)
      return min;
   else
      return value;
}

#include "primUtils/primEllipseDEGEN.cpp"
#include "primUtils/primArchDEGEN.cpp"
#include "primUtils/primPillDEGEN.cpp"
#include "primUtils/primQuadDEGEN.cpp"

#include "primUtils/primColorWheelDEGEN.cpp"
#include "primUtils/primBulbDEGEN.cpp"

#include "primUtils/hsv2rgb.cpp"
#include "primUtils/rgb2hsv.cpp"

// :v
