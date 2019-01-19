#include <cmath>

using namespace std;

void hsv2rgb(double h, double s, double v, double rgb[3]) {
   if ( s <= 0.0 ) {
      rgb[0] = v;
      rgb[1] = v;
      rgb[2] = v;
      return;
   }

   h *= 360.0;
   double R, G, B;

   long hi = long(floor(h / 60.0)) % 6;
   double f = (h / 60.0) - hi;
   double p = v * (1.0 - s);
   double q = v * (1.0 - s * f);
   double t = v * (1.0 - s * (1.0 - f));

   switch(hi) {
      case 0: R = v, G = t, B = p; break;
      case 1: R = q, G = v, B = p; break;
      case 2: R = p, G = v, B = t; break;
      case 3: R = p, G = q, B = v; break;
      case 4: R = t, G = p, B = v; break;
      case 5: R = v, G = p, B = q; break;
      default: R = v, G = p, B = q; break;
   }

   rgb[0] = R;
   rgb[1] = G;
   rgb[2] = B;

   return;
}
