#include "Arduino.h"
#include "heavenliClient.h"
//#include <PacketSerial.h>

heavenliClient::heavenliClient()
{
   //PacketSerial __client;
   connectionEstablished = false; 
   ackReceived = false;
}

void heavenliClient::init() {
   numLamps = 0;
   return;
}

void heavenliClient::update() {
   return;
}

void heavenliClient::processPacket(const uint8_t* buffer, size_t size) {
   char tmp[size];
   memcpy(tmp, buffer, size);
   String tms = String(tmp);
   if (tms = "This is an ack packet.") {
      ackReceived = true;
   }

   return;
}

bool heavenliClient::establishConnection() {
   return false;
}

size_t heavenliClient::outPacket(uint8_t*& buffer) {
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
