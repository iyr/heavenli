#include "Arduino.h"
#include "heavenliClient.h"
#include "heavenliLamp.h"
#include <EEPROM.h>

heavenliClient::heavenliClient() {
   //this->IDaddress      = 'h';
   this->isConnected    = false; 
   this->synackReceived = false;
   this->outBufferFull  = false;

   // Determine if device has an ID set (cannot be 0, 255, or FF)
   this->id = EEPROM.read(this->IDaddress);
   if (  this->id == 255   ||
         this->id == 0     ){
      this->__ID_requested = true;
   } else {
      this->__ID_requested = false;
   }
}

void heavenliClient::init() {
   this->numLamps = 0;
   return;
}

void heavenliClient::update()
{
   return;
}

void heavenliClient::update(hliLamp lamp) {
   lamp.update(2.72);
   return;
}

void heavenliClient::update(hliLamp* lamps)
{
   return;
}

void heavenliClient::processPacket(const uint8_t* buffer, size_t size) {
   //char tmp[size];
   //memcpy(tmp, buffer, size);
   //if (tms == "This is a synack packet.") {

   char tmp[] = "SYNACK";
   if (strcmp(tmp, buffer) == 0) {
      this->synackReceived = true;
      this->connectionEstablished = true;
   } else {

      // Read buffer for paramters
      for (int i = 0; i < size; i++) {

         if (i+4 <= size) {

            if (  buffer[i+0] == 'C'   && 
                  buffer[i+1] == 'I'   &&
                  buffer[i+2] == 'D'   &&
                  buffer[i+3] == '?'   &&
                  buffer[i+4] == 'R'   ){
               this->__CID_requested = true;
            }

         }
      }
   }
   return;
}

// Performs necessary three-way handshake with client device
bool heavenliClient::establishConnection() {
   return false;
}

int heavenliClient::getID() {
   return this->id;
}

void heavenliClient::setID(int newID) {

   // Ensure ID is not 0, 255, or FF
   if (  newID == 255   ||
         newID == 0     ){
      return;
   } else {
      this->id = newID;
      EEPROM.update(this->IDaddress, this->id);
      return;
   }
}

size_t heavenliClient::outPacket(uint8_t*& buffer) {
   size_t   numBytes = 0;
   uint8_t  byteLimit = 56;
   uint8_t  tmb[byteLimit];
   
   // If not a real client, listen for syn packets
   if (this->isConnected == false) {
      if (this->synackReceived == true) {
         String message = "ACK";
         numBytes = message.length()+1;
         buffer = new uint8_t[numBytes];
         message.toCharArray(buffer, numBytes);
         this->isConnected = true;
      } else {
         String message = "SYN";
         numBytes = message.length()+1;
         buffer = new uint8_t[numBytes];
         message.toCharArray(buffer, numBytes);
      }
   } else {

      // Plugin has requested the ID of the Client Device (HOST: getClientID)
      if (  this->__CID_requested   == true  && 
            this->outBufferFull     == false ){

         // Number of bytes it will take to send parameter information
         uint8_t paramBytes = 6;

         // Check if we have enough space in out output buffer
         if (paramBytes + numBytes >= byteLimit) {
            this->outBufferFull = true;
         } else {
            // Get upper and lower bytes of ID;
            uint8_t idul = this->id & 65280;
            uint8_t idll = this->id & 255;
            tmb[numBytes] = 'C'; numBytes++;
            tmb[numBytes] = 'I'; numBytes++;
            tmb[numBytes] = 'D'; numBytes++;
            tmb[numBytes] = ':'; numBytes++;
            tmb[numBytes] = 'U'; numBytes++;
            tmb[numBytes] = 'idul'; numBytes++;
            tmb[numBytes] = 'idll'; numBytes++;
            this->__CID_requested = false;
         }
      }

      // Write contents to buffer
      buffer = new uint8_t[numBytes];
      for (int i = 0; i < numBytes; i++)
         buffer[i] = tmb[i];
   }
   return numBytes;
}

