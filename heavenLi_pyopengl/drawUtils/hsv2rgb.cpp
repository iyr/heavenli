#include <cmath>

using namespace std;

void hsv2rgb(float h, float s, float v, float rgb[3]) {
   float C = s*v;
   float X = float(C * (1.0 - abs(h - 1.0)));
   float m = v - C;
   float R, G, B;

   if (h >= 0.0 && h < 1.0/6.0) {
      R = C;
      G = X;
      B = 0.0;
   } else
   if (h >= 1.0/6.0 && h < 2.0/6.0) {
      R = X;
      G = C;
      B = 0.0;
   } else
   if (h >= 2.0/6.0 && h < 3.0/6.0) {
      R = 0.0;
      G = C;
      B = X;
   } else
   if (h >= 3.0/6.0 && h < 4.0/6.0) {
      R = 0.0;
      G = X;
      B = C;
   } else
   if (h >= 4.0/6.0 && h < 5.0/6.0) {
      R = X;
      G = 0.0;
      B = C;
   } else
   {
      R = C;
      G = 0.0;
      B = X;
   }
   rgb[0] = (R + m);
   rgb[1] = (G + m);
   rgb[2] = (B + m);
}
