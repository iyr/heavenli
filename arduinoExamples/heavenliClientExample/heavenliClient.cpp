#include "Arduino.h"
#include "heavenliClient.h"

heavenliClient::heavenliClient() {
   this->connectionEstablished = false; 
   this->synackReceived = false;
}

void heavenliClient::init() {
   this->numLamps = 0;
   return;
}

//void heavenliClient::update()
//{
   //return;
//}

void heavenliClient::update(hliLamp lamp) {
   lamp.update(2.72);
   return;
}

//void heavenliClient::update(hliLamp* lamps)
//{
   //return;
//}

void heavenliClient::processPacket(const uint8_t* buffer, size_t size) {
   //char tmp[size];
   //memcpy(tmp, buffer, size);
   //if (tms == "This is a synack packet.") {
   char tmp[] = "SYNACK";
   if (strcmp(tmp, buffer) == 0) {
      this->synackReceived = true;
      this->connectionEstablished = true;
   }

   return;
}

bool heavenliClient::establishConnection() {
   return false;
}

size_t heavenliClient::outPacket(uint8_t*& buffer) {
   size_t n = 0;
   if (this->synackReceived) {
      String message = "ACK";
      n = message.length()+1;
      buffer = new uint8_t[n];
      message.toCharArray(buffer, n);
   } else {
      String message = "SYN";
      n = message.length()+1;
      buffer = new uint8_t[n];
      message.toCharArray(buffer, n);
   }
   return n;
}

/*
 * Implements a heavenli lamp
 */
hliLamp::hliLamp() {
   this->numBulbs = 1;
   this->isMetaLamp = false;
   this->bulbsTargetRGB[10][3];
   this->bulbsCurrentRGB[10][3];
   this->alias[4] = 'demo';
   this->id[2] = 'FF';


   float RGB[3] = {0.5, 0.0, 0.3};
   this->setBulbsTargetRGB(RGB);
   RGB[0]=0.0; RGB[1]=0.0; RGB[2]=0.0;
   this->setBulbsCurrentRGB(RGB);
}

void hliLamp::init() {
   return;
}

void hliLamp::setBulbsTargetRGB(float* TargetRGB) {
   float RGB[3] = {0.0, 0.0, 0.0};
   if (sizeof(RGB)/sizeof(*RGB) != sizeof(TargetRGB)/sizeof(*TargetRGB))
      return;
   else
   {
      for (int i = 0; i < 10; i++)
      {
         this->bulbsTargetRGB[i][0] = TargetRGB[0];
         this->bulbsTargetRGB[i][1] = TargetRGB[1];
         this->bulbsTargetRGB[i][2] = TargetRGB[2];
      }
      return;
   }
}

void hliLamp::update(float frameTime) {
   int r;
   int g;
   int b;
   int rT;
   int gT;
   int bT;
   int difR;
   int difG;
   int difB;
   int rd = 0;
   int gd = 0;
   int bd = 0;
   int delta = frameTime;
   int threshold = 13;
   int tmf;
   for (int i = 0; i < this.numBulbs; i++) {
     r = this->bulbsCurrentRGB[i][0];
     g = this->bulbsCurrentRGB[i][1];
     b = this->bulbsCurrentRGB[i][2];
  
     rT = this->bulbsTargetRGB[i][0];
     gT = this->bulbsTargetRGB[i][1];
     bT = this->bulbsTargetRGB[i][2];

      if (  r != rT  ||
            g != gT  ||
            b != bT  ){
         difR = abs(r - rT);
         difG = abs(g - gT);
         difB = abs(b - bT);
         rd = 0;
         gd = 0;
         bd = 0;
         delta = frameTime;
         threshold = 12
         tmf = float((i+2)*2)/float(this->numBulbs*3);
         delta *= tmf;

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
         
         if (difR >= 255)
            difR = 255;
         else if (difR <= 0)
            difR = 0;

         if (difB >= 255)
            difB = 255;
         else if (difB <= 0)
            difB = 0;

         if (difG >= 255)
            difG = 255;
         else if (difG <= 0)
            difG = 0;

         this->bulbsCurrentRGB[i][0] = difR;
         this->bulbsCurrentRGB[i][1] = difG;
         this->bulbsCurrentRGB[i][2] = difB;
      }
   }

   return;
}

void hliLamp::setBulbsCurrentRGB(float* CurrentRGB) {
   float RGB[3] = {0.0, 0.0, 0.0};
   if (sizeof(RGB)/sizeof(*RGB) != sizeof(CurrentRGB)/sizeof(*CurrentRGB))
      return;
   else
   {
      for (int i = 0; i < 10; i++)
      {
         this->bulbsCurrentRGB[i][0] = CurrentRGB[0];
         this->bulbsCurrentRGB[i][1] = CurrentRGB[1];
         this->bulbsCurrentRGB[i][2] = CurrentRGB[2];
      }
      return;
   }
}

void hliLamp::getBulbCurrentRGB(unsigned int bulb, float* RGB) {
   RGB[0] = this->bulbsCurrentRGB[bulb][0];
   RGB[1] = this->bulbsCurrentRGB[bulb][1];
   RGB[2] = this->bulbsCurrentRGB[bulb][2];
   return;
}

void hliLamp::setNumBulbs(unsigned int newNumBulbs) {
   if (newNumBulbs > 6)
      this->numBulbs = 6;
   else if (newNumBulbs < 1)
      this->numBulbs = 1;
   else
      this->numBulbs = newNumBulbs;
   return;
}
