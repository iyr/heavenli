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

int heavenliClient::getID() {
   return this->id;
}

void heavenliClient::setID(int newID) {
   this->id = newID;
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
   //this->isMetaLamp = false;
   this->bulbsTargetRGB[10][3];
   this->bulbsCurrentRGB[10][3];
   this->alias[16] = 'demo';
   this->id = 255;


   uint8_t RGB[3] = {128, 0, 77};
   this->setBulbsTargetRGB(RGB);
   RGB[0]=0; RGB[1]=0; RGB[2]=0;
   this->setBulbsCurrentRGB(RGB);
}

void hliLamp::init() {
   return;
}

void     hliLamp::getAlias(char*& knickname) {
   knickname = new char[16];
   for (uint8_t i = 0; i < 16; i++) {
      knickname[i] = this->alias[i];
   }
   return;
}

void     hliLamp::setAlias(char* newKnickname) {
   this->alias = new char[16];
   for (uint8_t i = 0; i < 16; i++) {
      this->alias[i] = newKnickname[i];
   }
   return;
}

float    hliLamp::getAngularOffset() {
   return this->angularOffset;
}

void     hliLamp::setAngularOffset(float newAO) {
   this->angularOffset = newAO;
   return;
}

void     hliLamp::setAngularOffset(int8_t newAO) {
   this->angularOffset = float(newAO);
   return;
}

uint8_t  hliLamp::getArrangement() {
   return this->arrangement;
}

void     hliLamp::setArrangement(uint8_t newArn) {
   this->arrangement = constrain(newArn, 0, 1);
   return;
}

void     hliLamp::getBulbCurrentRGB(uint8_t bulb, uint8_t* RGB) {
   RGB[0] = this->bulbsCurrentRGB[bulb][0];
   RGB[1] = this->bulbsCurrentRGB[bulb][1];
   RGB[2] = this->bulbsCurrentRGB[bulb][2];
   return;
}

void     hliLamp::setBulbCurrentRGB(uint8_t bulb, uint8_t* newRGB) {
   this->bulbsCurrentRGB[bulb][0] = newRGB[0];
   this->bulbsCurrentRGB[bulb][1] = newRGB[1];
   this->bulbsCurrentRGB[bulb][2] = newRGB[2];
   return;
}

void     hliLamp::getBulbsCurrentRGB(uint8_t** RGB) {
   return;
}

void     hliLamp::setBulbsCurrentRGB(uint8_t* newRGB) {
   uint8_t RGB[3] = {0, 0, 0};
   // Sanity Check to verrify bounds
   if (sizeof(RGB)/sizeof(*RGB) != sizeof(newRGB)/sizeof(*newRGB))
      return;
   else
   {
      for (int i = 0; i < 10; i++)
      {
         this->bulbsCurrentRGB[i][0] = newRGB[0];
         this->bulbsCurrentRGB[i][1] = newRGB[1];
         this->bulbsCurrentRGB[i][2] = newRGB[2];
      }
      return;
   }
}

void     hliLamp::getBulbTargetRGB(uint8_t bulb, uint8_t* RGB) {
   RGB[0] = this->bulbsTargetRGB[bulb][0];
   RGB[1] = this->bulbsTargetRGB[bulb][1];
   RGB[2] = this->bulbsTargetRGB[bulb][2];
   return;
}

void     hliLamp::setBulbTargetRGB(uint8_t bulb, uint8_t* newRGB) {
   this->bulbsTargetRGB[bulb][0] = newRGB[0];
   this->bulbsTargetRGB[bulb][1] = newRGB[1];
   this->bulbsTargetRGB[bulb][2] = newRGB[2];
   return;
}

void     hliLamp::getBulbsTargetRGB(uint8_t** RGB) {
   return;
}

void     hliLamp::setBulbsTargetRGB(uint8_t* newRGB) {
   uint8_t RGB[3] = {0, 0, 0};
   // Sanity Check
   if (sizeof(RGB)/sizeof(*RGB) != sizeof(newRGB)/sizeof(*newRGB))
      return;
   else
   {
      for (int i = 0; i < 10; i++)
      {
         this->bulbsTargetRGB[i][0] = newRGB[0];
         this->bulbsTargetRGB[i][1] = newRGB[1];
         this->bulbsTargetRGB[i][2] = newRGB[2];
      }
      return;
   }
}

int      hliLamp::getID() {
   return this->id;
}

void     hliLamp::setID(int newID) {
   this->id = newID;
   return;
}
      
char     hliLamp::getMasterSwitchBehavior() {
   return this->masterSwitchBehavior;
}

void     hliLamp::setMasterSwitchBehavior(char newBehavior) {
   this->masterSwitchBehavior = newBehavior;
   return;
}

uint8_t  hliLamp::getNumBulbs() {
   return this->numBulbs;
}

void     hliLamp::setNumBulbs(uint8_t newNumBulbs) {
   if (newNumBulbs > 6)
      this->numBulbs = 6;
   else if (newNumBulbs < 1)
      this->numBulbs = 1;
   else
      this->numBulbs = newNumBulbs;
   return;
}

bool     hliLamp::getBulbCountMutability() {
   return this->mutableBulbCount;
}

uint8_t  hliLamp::getMetaLampLevel() {
   return this->metaLampLevel;
}

void     hliLamp::getValidBulbQuantities(uint8_t*& quantities) {
   quantities = new uint8_t[10];
   for (int i = 0; i < 10; i++) {
      quantities[i] = this->validBulbCounts[i];
   }
   return;
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
   for (int i = 0; i < this->numBulbs; i++) {
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
         threshold = 12;
         tmf = int(float((i+2)*2)/float(this->numBulbs*3));
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
        /* 
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
         */

         difR = constrain(difR, 0, 255);
         difG = constrain(difG, 0, 255);
         difB = constrain(difB, 0, 255);

         this->bulbsCurrentRGB[i][0] = difR;
         this->bulbsCurrentRGB[i][1] = difG;
         this->bulbsCurrentRGB[i][2] = difB;
      }
   }

   return;
}

