#include "FastLED.h"

#define NUM_LEDS 10
#define DATA_PIN 17

CRGB strip[NUM_LEDS];

//char inPacket[64];

char inByte;
float r = 0.0;
float g = 0.0;
float b = 0.0;
float rT = 0.5;
float gT = 1.0;
float bT = 0.0;
float tDiff = 0.005;

void setup() {
   Serial.begin(38400);
   FastLED.addLeds<NEOPIXEL, DATA_PIN>(strip, NUM_LEDS);

   //while (!Serial) {
     // ;
   //}
   //establishContact();
}

void loop() {

   if (Serial.available() > 0 ) {
      //Serial.println("Contact!");
      //Serial.readBytesUntil('t', inPacket, 64);
      switch(Serial.read())
      {
         case 'r':    
            int tmr = Serial.parseInt();
            rT = pow(float(tmr)/255.0, 2.0);
            //rT = Serial.parseFloat();
            //Serial.println();
         //break;

         case 'g':
            int tmg = Serial.parseInt();
            gT = pow(float(tmg)/255.0, 2.0);
         //break;

         case 'b':
            int tmb = Serial.parseInt();
            bT = pow(float(tmb)/255.0, 2.0);
            //bT = Serial.parseFloat();
         break;

         //case 't':
           // tDiff = 0.005;//Serial.parseFloat();
         //break;
      }
  }

  updateLEDs(tDiff/2.0);
  for (int i = 0; i < NUM_LEDS; i++){
     int tmr = int(r*255.0);
     int tmg = int(g*255.0);
     int tmb = int(b*255.0);
     strip[i] = CRGB(tmr, tmg, tmb);
     //strip[i] = CRGB(128, 128, 128);
     //strip[i] = CRGB(int(r*255.0), int(g*255.0), int(b*255.0));
  }

  FastLED.show();
}

void updateLEDs(float frameTime) {
   int numBulbs = 1;
   for (int i = 0; i < numBulbs; i++) {
      //Serial.print("g: ");
      //Serial.println(g);
      //Serial.print("gT: ");
      //Serial.println(gT);
      if (  r != rT  ||
            g != gT  ||
            b != bT  ){
         float difR = abs(r - rT);
         float difG = abs(g - gT);
         float difB = abs(b - bT);
         float rd = 0.0;
         float gd = 0.0;
         float bd = 0.0;
         float delta = frameTime;
         float threshold = 0.05;
         float tmf = float((i+2)*2)/float(numBulbs*3);
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

         //difR = constrain(difR, 0.0, 1.0);
         //difG = constrain(difG, 0.0, 1.0);
         //difB = constrain(difB, 0.0, 1.0);

         //difR *= difR;
         //difG *= difG;
         //difB *= difB;
         
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

         r = difR;
         g = difG;
         b = difB;
      }
   }
}

void establishContact() {
   while (Serial.available() <= 0) {
      Serial.println("Waiting for connection...");
      delay(2500);
   }
}
