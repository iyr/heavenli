#include "Arduino.h"
#include "heavenliClient.h"
#include <PacketSerial.h>

heavenliClient::heavenliClient()
{
   PacketSerial __client;
   connectionEstablished = false; 
}

void heavenliClient::init() {
   numLamps = 0;
   return;
}

void heavenliClient::update() {
   return;
}

void heavenliClient::processPacket(const uint8_t* buffer, size_t size) {
   return;
}

bool heavenliClient::establishConnection() {
   return false;
}

void heavenliClient::outPacket(uint8_t* buffer, size_t size) {
   uint8_t tmp[size] = {
      'T','h','i','s',' ',
      'i','s',' ',
      'a',' ',
      's','y','n',' ',
      'p','a','c','k','e','t','.',
      'Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q'
   };

   memcpy(buffer, tmp, size);
   return;
}
