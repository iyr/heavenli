#include "Arduino.h"
#include "heavenliClient.h"

heavenliClient::heavenliClient()
{
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
