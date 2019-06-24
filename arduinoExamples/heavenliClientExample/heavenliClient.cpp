#include "Arduino.h"
#include "heavenliClient.h"

heavenliClient::heavenliClient()
{
   connectionEstablished = false; 
   ackReceived = false;
}

void heavenliClient::init()
{
   numLamps = 0;
   return;
}

//void heavenliClient::update()
//{
   //return;
//}

void heavenliClient::update(hliLamp lamp)
{
   return;
}

//void heavenliClient::update(hliLamp* lamps)
//{
   //return;
//}

void heavenliClient::processPacket(const uint8_t* buffer, size_t size)
{
   char tmp[size];
   memcpy(tmp, buffer, size);
   String tms = String(tmp);
   if (tms == "This is an ack packet.") {
      ackReceived = true;
   }

   return;
}

bool heavenliClient::establishConnection()
{
   return false;
}

size_t heavenliClient::outPacket(uint8_t*& buffer)
{
   size_t n = 0;
   if (ackReceived) {
      String message = "This is a synack packet.";
      n = message.length()+1;
      buffer = new uint8_t[n];
      message.toCharArray(buffer, n);
   } else {
      String message = "This is a syn packet.";
      n = message.length()+1;
      buffer = new uint8_t[n];
      message.toCharArray(buffer, n);
   }
   return n;
}

/*
 * Implements a heavenli lamp
 */
hliLamp::hliLamp()
{
   numBulbs = 1;
   isMetaLamp = 0;
   bulbsTargetRGB[10][3];
   bulbsCurrentRGB[10][3];
   alias[4] = 'demo';
   id[2] = 'FF';


   float RGB[3] = {0.5, 0.0, 0.3};
   setBulbsTargetRGB(RGB);
   RGB[0]=0.0; RGB[1]=0.0; RGB[2]=0.0;
   setBulbsCurrentRGB(RGB);
}

void hliLamp::init()
{
   return;
}

void hliLamp::setBulbsTargetRGB(float* TargetRGB)
{
   float RGB[3] = {0.0, 0.0, 0.0};
   if (sizeof(RGB)/sizeof(*RGB) != sizeof(TargetRGB)/sizeof(*TargetRGB))
      return;
   else
   {
      for (int i = 0; i < 10; i++)
      {
         bulbsTargetRGB[i][0] = TargetRGB[0];
         bulbsTargetRGB[i][1] = TargetRGB[1];
         bulbsTargetRGB[i][2] = TargetRGB[2];
      }
      return;
   }
}

void hliLamp::update(float frameTime) {
   float 
      r, 
      g, 
      b, 
      rT, 
      gT, 
      bT, 
      difR,
      difG,
      difB,
      rd = 0.0,
      gd = 0.0,
      bd = 0.0,
      delta = frameTime,
      threshold = 0.05,
      tmf;
   for (int i = 0; i < numBulbs; i++) {
     float r = bulbsCurrentRGB[i][0];
     float g = bulbsCurrentRGB[i][1];
     float b = bulbsCurrentRGB[i][2];
  
     float rT = bulbsTargetRGB[i][0];
     float gT = bulbsTargetRGB[i][1];
     float bT = bulbsTargetRGB[i][2];

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

         bulbsCurrentRGB[i][0] = difR;
         bulbsCurrentRGB[i][1] = difG;
         bulbsCurrentRGB[i][2] = difB;
      }
   }

   return;
}

void hliLamp::setBulbsCurrentRGB(float* CurrentRGB)
{
   float RGB[3] = {0.0, 0.0, 0.0};
   if (sizeof(RGB)/sizeof(*RGB) != sizeof(CurrentRGB)/sizeof(*CurrentRGB))
      return;
   else
   {
      for (int i = 0; i < 10; i++)
      {
         bulbsCurrentRGB[i][0] = CurrentRGB[0];
         bulbsCurrentRGB[i][1] = CurrentRGB[1];
         bulbsCurrentRGB[i][2] = CurrentRGB[2];
      }
      return;
   }
}

void hliLamp::setNumBulbs(unsigned int newNumBulbs)
{
   numBulbs = newNumBulbs;
   return;
}
