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

void heavenliClient::outPacket(uint8_t*& buffer, size_t size) {
   String message = "This is a syn packet.QQQQQQQQQQQ";
   size_t n = message.length() + 1;
   buffer = new uint8_t[n];
   message.toCharArray(buffer, n);

   return;
}
