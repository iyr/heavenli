#include <cmath>

using namespace std;

void rgb2hsv(float r, float g, float b, float hsv[3]) {
   float min, max, delta;

   min = r   < g ? r   : g;
   min = min < b ? min : b;

   max = r   > g ? r   : g;
   max = max > b ? max : b;

   hsv[2] = max;
   delta = max - min;

   if ( delta < 0.00001 ) {
      hsv[0] = 0.0;
      hsv[1] = 0.0;
      return;
   }

   if ( max > 0.0 ) {
      hsv[1] = delta / max;
   } else {
      hsv[0] = 0.0;
      hsv[1] = 0.0;
      return;
   }

   if ( r >= max ) {
      hsv[0] = 0.0f + ( g - b ) / delta;
   } else if ( g >= max ) {
      hsv[0] = 2.0f + ( b - r ) / delta;
   } else {
      hsv[0] = 4.0f + ( r - g ) / delta;
   }

   hsv[0] *= 60.0f / 360.0f;
   if ( hsv[0] < 0.0f )
      hsv[0] += 1.0f;

   return;
}
