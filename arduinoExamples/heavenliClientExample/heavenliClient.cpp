
#include "Arduino.h"
#include "heavenliClient.h"
#include "heavenliLamp.h"
#include <EEPROM.h>

heavenliClient::heavenliClient() {
   this->isConnected       = false; 
   this->synackReceived    = false;
   this->outBufferFull     = false;
   this->__CID_sent        = false;
   this->runtimeCounter1   = millis();
   this->timeoutCounter    = millis();

   // Determine if device has an ID set (cannot be 0, 255, or FF)
   this->id[0] = EEPROM.read(this->IDaddress+0);
   this->id[1] = EEPROM.read(this->IDaddress+1);
}

void heavenliClient::init(hliLamp* lamp) {
   this->numLamps = 1;
   this->lamp = lamp;
   this->lamp->init();
   return;
}

void heavenliClient::init(hliLamp* lamps, uint8_t numLamps) {
   this->numLamps = numLamps;
   for (int i = 0; i < numLamps; i++){
      lamps[i].init();
   }
   return;
}

void heavenliClient::update()
{
   if ((millis() - this->timeoutCounter) > 1500) {
      this->isConnected = false;
      this->synackReceived = false;
      this->__CID_sent = false;
      this->timeoutCounter = millis();
   }
   return;
}

void heavenliClient::update(hliLamp* lamp) {

   /*
   // Check for new lamps
   char* tmid;
   if (lampIDs == NULL) {
      lamp.getID(tmid);
      this->lampIDs = new uint8_t[1][2];
      this->lampIDs[0][0] = tmid[0];
      this->lampIDs[0][1] = tmid[1];
      numLamps++;
   } else {
      lamp.getID(tmid);
      bool isKnownLamp = false;
      for (int i = 0; i < numLamps; i++){
         // Check the lamp ID against list of known lampIDs
         if (  this->lampIDs[i][0] == tmid[0] &&
               this->lampIDs[i][1] == tmid[1] )
            isKnownLamp = true;
      }

      if (isKnownLamp == false) {
         // Allocate Temporary buffer for current array of lamp IDs
         tml = [numLamps][2];

         // Copy the lamp IDs to the temporary array
         for (int j = 0; j < numLamps; j++) {
            tml[j][0] = this->lampIDs[j][0];
            tml[j][1] = this->lampIDs[j][1];
         }

         // Deallocate current array
         delete [] this->lampIDs;

         // Update number of lamp IDs
         numLamps++;

         // Allocate new array of lamps
         this->lampIDs = new uint8_t[numLamps][2];

         // Copy old array of lamp IDs over
         for (int j = 0; j < numLamps-1; j++) {
            this->lampIDs[j][0] = tml[j][0];
            this->lampIDs[j][1] = tml[j][1];
         }

         // Copy the new lamp ID to client array
         this->lampIDs[numLamps][0] = tmid[0];
         this->lampIDs[numLamps][1] = tmid[1];
      }
   }
   */

   this->lamp->update(2.72);
   if ((millis() - this->timeoutCounter) > 1500) {
      this->isConnected = false;
      this->synackReceived = false;
      this->__CID_sent = false;
      this->timeoutCounter = millis();
   }
   return;
}

void heavenliClient::update(hliLamp* lamps, uint8_t numLamps)
{
   if ((millis() - this->timeoutCounter) > 1500) {
      this->isConnected = false;
      this->synackReceived = false;
      this->__CID_sent = false;
      this->timeoutCounter = millis();
   }
   return;
}

/*
 * Generates Packet to be sent to host
 */
size_t heavenliClient::outPacket(uint8_t*& buffer) {
   size_t   numBytes = 0;
   uint8_t  byteLimit = 56;
   uint8_t  message[byteLimit];
   

   if (millis() - this->runtimeCounter1 > 1000) {
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
      } 
      else // Respond to Data Requests
      {

         // Plugin has requested the ID of the Client Device (HOST: getClientID)
         if (  this->__CID_requested   == true  && 
               this->__CID_sent        == false &&
               this->outBufferFull     == false ){

            // Number of bytes it will take to send parameter information
            uint8_t paramBytes = 0;
            char tmb[10];

            tmb[paramBytes] = 'C'; paramBytes++;
            tmb[paramBytes] = 'I'; paramBytes++;
            tmb[paramBytes] = 'D'; paramBytes++;
            tmb[paramBytes] = '!'; paramBytes++;
            tmb[paramBytes] = this->id[0]; paramBytes++;
            tmb[paramBytes] = this->id[1]; paramBytes++;

            // Check if we have enough space in out output buffer
            if (paramBytes + numBytes >= byteLimit) {
               this->outBufferFull = true;
            } else {
               for (int i = 0; i < paramBytes; i++)
                  message[numBytes+i] = tmb[i];
               numBytes += paramBytes;
               this->__CID_requested = false;
               this->__CID_sent = true;
            }
         }

         // Plugin has requested the ID of the Client Device (HOST: getClientID)
         if (  this->__CNL_requested   == true  && 
               this->__CNL_sent        == false &&
               this->outBufferFull     == false ){

            // Number of bytes it will take to send parameter information
            uint8_t paramBytes = 0;
            char tmb[10];

            tmb[paramBytes] = 'C'; paramBytes++;
            tmb[paramBytes] = 'N'; paramBytes++;
            tmb[paramBytes] = 'L'; paramBytes++;
            tmb[paramBytes] = '!'; paramBytes++;
            tmb[paramBytes] = this->numLamps; paramBytes++;

            // Check if we have enough space in out output buffer
            if (paramBytes + numBytes >= byteLimit) {
               this->outBufferFull = true;
            } else {
               for (int i = 0; i < paramBytes; i++)
                  message[numBytes+i] = tmb[i];
               numBytes += paramBytes;
               this->__CNL_requested = false;
               this->__CNL_sent = true;
            }
         }

         // Write contents to buffer
         buffer = new uint8_t[numBytes];
         for (int i = 0; i < numBytes; i++)
            buffer[i] = message[i];
      }

      this->runtimeCounter1 = millis();
   }
   return numBytes;
}

void heavenliClient::processPacket(const uint8_t* buffer, size_t size) {

   // Before we can start sending or receiving data
   //
   char tmp[] = "SYNACK";
   if (strcmp(tmp, buffer) == 0) {
      this->synackReceived = true;
      this->connectionEstablished = true;
   } else {

      // Read buffer for paramters
      for (int i = 0; i < size; i++) {

         if (i+3 <= size) {

            // Host has requested the Client ID
            if (  buffer[i+0] == 'C'   && 
                  buffer[i+1] == 'I'   &&
                  buffer[i+2] == 'D'   &&
                  buffer[i+3] == '?'   ){
               this->__CID_requested = true;
            }

            // Host has requested the Client Number of Lamps
            if (  buffer[i+0] == 'C'   && 
                  buffer[i+1] == 'N'   &&
                  buffer[i+2] == 'L'   &&
                  buffer[i+3] == '?'   ){
               this->__CNL_requested = true;
            }

            // Host is assigning Client ID
            if (  buffer[i+0] == 'C'   && 
                  buffer[i+1] == 'I'   &&
                  buffer[i+2] == 'D'   &&
                  buffer[i+3] == '!'   ){
               uint8_t tmp[2];
               tmp[0] = buffer[i+4];
               tmp[1] = buffer[i+5];
               this->setID(tmp);
               this->__CID_requested = false;
            }

         }
      }
   }

   this->timeoutCounter = millis();
   return;
}

// Performs necessary three-way handshake with client device
bool heavenliClient::establishConnection() {
   return false;
}

// IMPLEMENTED THIS FOR FUTURE USE NOW PLS
int heavenliClient::getID() {
   return this->id;
}

void heavenliClient::setID(uint8_t* newID) {
   // Ensure ID is not 0, 255, or FF
   if (  (newID[0] == 0 && newID[1] == 0)    ||
         (newID[0] == 255 && newID[1] == 255)){
      return;
   } else {
      this->id[0] = newID[0];
      this->id[1] = newID[1];
      EEPROM.update(this->IDaddress+0, this->id[0]);
      EEPROM.update(this->IDaddress+1, this->id[1]);
      return;
   }
}
