#include "FastLED.h"

#define NUM_LEDS 60
#define DATA_PIN 10

CRGB strip[NUM_LEDS];

float bulbsCurRGB[6][3];
float bulbsTarRGB[6][3];

char inByte;
float tDiff = 0.01;
int numBulbs = 1;
int curBulb = 0;

void setup() {
  Serial.begin(57600);
  FastLED.addLeds<NEOPIXEL, DATA_PIN>(strip, NUM_LEDS);
  FastLED.setBrightness(64);

  for (int i = 0; i < 6; i++) {
    bulbsCurRGB[i][0] = 0.0;
    bulbsCurRGB[i][1] = 0.0;
    bulbsCurRGB[i][2] = 0.0;

    bulbsTarRGB[i][0] = 0.5;
    bulbsTarRGB[i][1] = 0.0;
    bulbsTarRGB[i][2] = 1.0;
  }
  //FastLED.setBrightness(255);

  //while (!Serial) {
  // ;
  //}
  //establishContact();
}

void loop() {

  if (Serial.available() > 0 ) {
    //Serial.println("Contact!");
    //Serial.readBytesUntil('t', inPacket, 64);
    char inMessage[8];

    Serial.readBytes(inMessage, 5);

    numBulbs  = constrain(inMessage[0], 1, 6);
    curBulb   = constrain(inMessage[1], 0, 5);

    bulbsTarRGB[curBulb][0] = pow(float(inMessage[2])/128.0, 2.0);
    bulbsTarRGB[curBulb][1] = pow(float(inMessage[3])/128.0, 2.0);
    bulbsTarRGB[curBulb][2] = pow(float(inMessage[4])/128.0, 2.0);
  }

  /*
  Serial.print(" tmr: ");
  Serial.print(bulbsTarRGB[0][0]);
  Serial.print(" tmg: ");
  Serial.print(bulbsTarRGB[0][1]);
  Serial.print(" tmb: ");
  Serial.println(bulbsTarRGB[0][2]);
  */
  
  updateLEDs(tDiff / 2.0);
  int LEDsPerBulb = NUM_LEDS / constrain(numBulbs, 1, 6);
  for (int i = 0; i < LEDsPerBulb; i++) {
    int tmr = int(bulbsCurRGB[curBulb][0] * 255.0);
    int tmg = int(bulbsCurRGB[curBulb][1] * 255.0);
    int tmb = int(bulbsCurRGB[curBulb][2] * 255.0);
    strip[constrain(i + curBulb * LEDsPerBulb, 0, 59)] = CRGB(tmr, tmg, tmb);
  }

  FastLED.show();
}

void updateLEDs(float frameTime) {
  for (int i = 0; i < numBulbs; i++) {
    float r = bulbsCurRGB[i][0];
    float g = bulbsCurRGB[i][1];
    float b = bulbsCurRGB[i][2];

    float rT = bulbsTarRGB[i][0];
    float gT = bulbsTarRGB[i][1];
    float bT = bulbsTarRGB[i][2];

    if (  r != rT  ||
          g != gT  ||
          b != bT  ) {
      float difR = abs(r - rT);
      float difG = abs(g - gT);
      float difB = abs(b - bT);
      float rd = 0.0;
      float gd = 0.0;
      float bd = 0.0;
      float delta = frameTime;
      float threshold = 0.05;
      float tmf = float((i + 2) * 2) / float(constrain(numBulbs * 3, 1, 18));
      delta *= tmf;
      //Serial.println(tmf);

      if (difR > threshold)
        if (rT > r)
          rd = delta;
        else
          rd = -delta;

      if (difG > threshold)
        if (gT > g)
          gd = delta;
        else
          gd = -delta;

      if (difB > threshold)
        if (bT > b)
          bd = delta;
        else
          bd = -delta;

      if (difR > threshold)
        difR = r + rd;
      else
        difR = rT;

      if (difG > threshold)
        difG = g + gd;
      else
        difG = gT;

      if (difB > threshold)
        difB = b + bd;
      else
        difB = bT;

      if (difR >= 1.0)
        difR = 1.0;
      else if (difR <= 0.0)
        difR = 0.0;

      if (difB >= 1.0)
        difB = 1.0;
      else if (difB <= 0.0)
        difB = 0.0;

      if (difG >= 1.0)
        difG = 1.0;
      else if (difG <= 0.0)
        difG = 0.0;

      //         r = difR;
      //         g = difG;
      //         b = difB;
      bulbsCurRGB[i][0] = difR;
      bulbsCurRGB[i][1] = difG;
      bulbsCurRGB[i][2] = difB;
    }
  }
}

void establishContact() {
  while (Serial.available() <= 0) {
    Serial.println("Waiting for connection...");
    delay(2500);
  }
}
