#include <cmath>

using namespace std;

void hsv2rgb(float h, float s, float v, float rgb[3]) {
   h *= 360.0f;
   float R, G, B;

   int hi = int(h / 60.0f) % 6;
   float f = (h / 60.0f) - hi;
   float p = v * (1.0f - s);
   float q = v * (1.0f - s * f);
   float t = v * (1.0f - s * (1.0f - f));
   switch(hi) {
      case 0: R = v, G = t, B = p; break;
      case 1: R = q, G = v, B = p; break;
      case 2: R = p, G = v, B = t; break;
      case 3: R = p, G = q, B = v; break;
      case 4: R = t, G = p, B = v; break;
      case 5: R = v, G = p, B = q; break;
   }
   rgb[0] = R;
   rgb[1] = G;
   rgb[2] = B;

   return;
}
