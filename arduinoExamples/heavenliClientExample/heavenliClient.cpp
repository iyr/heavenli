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

