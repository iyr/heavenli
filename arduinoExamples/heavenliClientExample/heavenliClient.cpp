#include "Arduino.h"
#include "heavenliClient.h"
#include "heavenliLamp.h"
#include <EEPROM.h>

heavenliClient::heavenliClient() {
   this->isConnected       = false; 
   this->synackReceived    = false;
   this->outBufferFull     = false;
   this->__CID_sent        = false;
   this->client_addressed  = false;
   this->tma               = 0;
   this->tmb               = 0;
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
   this->lamp->update(2.72);
   if ((millis() - this->timeoutCounter) > 1500) {
      //this->isConnected = false;
      this->synackReceived = false;
      this->__CID_sent = false;
      this->timeoutCounter = millis();
   }
   return;
}

/*
 * Update Lamp state, keep tab on connection to host
 */
void heavenliClient::update(hliLamp* lamp) {
   this->lamp = lamp;
   this->lamp->update(2.72);
   if ((millis() - this->timeoutCounter) > 1500) {
      //this->isConnected = false;
      this->synackReceived = false;
      this->__CID_sent = false;
      this->timeoutCounter = millis();
   }
   return;
}

/*
 * Update Lamp states, keep tab on connection to host
 */
void heavenliClient::update(hliLamp* lamps, uint8_t numLamps)
{
   if ((millis() - this->timeoutCounter) > 15) {
      //this->isConnected = false;
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

         // Prepend Client ID to packet for host to address
         if (  this->client_addressed  == true ){
         //if ( true ){
            // Number of bytes it will take to send parameter information
            uint8_t paramBytes = 0;
            char tmb[10];

            tmb[paramBytes] = 'C'; paramBytes++;
            tmb[paramBytes] = 'I'; paramBytes++;
            tmb[paramBytes] = 'D'; paramBytes++;
            tmb[paramBytes] = ':'; paramBytes++;
            tmb[paramBytes] = this->id[0]; paramBytes++;
            tmb[paramBytes] = this->id[1]; paramBytes++;

            // Check if we have enough space in out output buffer
            if (paramBytes + numBytes >= byteLimit) {
               this->outBufferFull = true;
            } else {
               for (int i = 0; i < paramBytes; i++)
                  message[numBytes+i] = tmb[i];
               numBytes += paramBytes;
            }
         }

         // Plugin has requested the Number of lamps on the Client Device (HOST: requestNumLamps)
         if (  this->__CNL_requested   == true  && 
               this->client_addressed  == true  &&
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
            }
         }

         // Plugin has requested all base parameters of a lamp
         if (  this->__ALL_requested   == true  && 
               this->client_addressed  == true  &&
               this->outBufferFull     == false ){

            // Number of bytes it will take to send parameter information
            uint8_t paramBytes = 0;
            uint8_t tmb[50];

            uint8_t* tmid;
            this->lamp->getID(tmid);
            tmb[paramBytes] = 'L'; paramBytes++;
            tmb[paramBytes] = 'I'; paramBytes++;
            tmb[paramBytes] = 'D'; paramBytes++;
            tmb[paramBytes] = ':'; paramBytes++;
            tmb[paramBytes] = tmid[0]; paramBytes++;
            tmb[paramBytes] = tmid[1]; paramBytes++;
            
            tmb[paramBytes] = 'N'; paramBytes++;
            tmb[paramBytes] = 'B'; paramBytes++;
            tmb[paramBytes] = '!'; paramBytes++;
            tmb[paramBytes] = this->lamp->getNumBulbs(); paramBytes++;

            tmb[paramBytes] = 'C'; paramBytes++;
            tmb[paramBytes] = 'M'; paramBytes++;
            tmb[paramBytes] = '!'; paramBytes++;
            tmb[paramBytes] = this->lamp->getBulbCountMutability(); paramBytes++;

            tmb[paramBytes] = 'A'; paramBytes++;
            tmb[paramBytes] = 'R'; paramBytes++;
            tmb[paramBytes] = '!'; paramBytes++;
            tmb[paramBytes] = this->lamp->getArrangement(); paramBytes++;

            //tmb[paramBytes] = 'A'; paramBytes++;
            //tmb[paramBytes] = 'O'; paramBytes++;
            //tmb[paramBytes] = '!'; paramBytes++;

            tmb[paramBytes] = 'L'; paramBytes++;
            tmb[paramBytes] = 'L'; paramBytes++;
            tmb[paramBytes] = '!'; paramBytes++;
            tmb[paramBytes] = this->lamp->getMetaLampLevel(); paramBytes++;

            tmb[paramBytes] = 'S'; paramBytes++;
            tmb[paramBytes] = 'B'; paramBytes++;
            tmb[paramBytes] = '!'; paramBytes++;
            tmb[paramBytes] = this->lamp->getMasterSwitchBehavior(); paramBytes++;

            //tmb[paramBytes] = 'V'; paramBytes++;
            //tmb[paramBytes] = 'Q'; paramBytes++;
            //tmb[paramBytes] = '!'; paramBytes++;

            // Check if we have enough space in out output buffer
            if (paramBytes + numBytes >= byteLimit) {
               this->outBufferFull = true;
            } else {
               for (int i = 0; i < paramBytes; i++)
                  message[numBytes+i] = tmb[i];
               numBytes += paramBytes;
               this->__CNL_requested = false;
            }
         }

         // Write contents to buffer
         buffer = new uint8_t[numBytes];
         for (int i = 0; i < numBytes; i++)
            buffer[i] = message[i];
      }
      this->runtimeCounter1 = millis();
      this->client_addressed = false;
   }

   //this->client_addressed = false;
   return numBytes;
}

void heavenliClient::processPacket(const uint8_t* buffer, size_t size) {

   // Before we can start sending or receiving data
   //
   char tmp[] = "SYNACK";
   if (strcmp(tmp, buffer) == 0) {
      this->synackReceived = true;
      this->connectionEstablished = true;
   } else 
   if (this->connectionEstablished == true) {

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

            // Host is addressing this client
            if (  buffer[i+0] == 'C'   &&
                  buffer[i+1] == 'I'   &&
                  buffer[i+2] == 'D'   &&
                  buffer[i+3] == ':'   &&
                  //buffer[i+3] == ':'   ){
                  buffer[i+4] == this->id[0] &&
                  buffer[i+5] == this->id[1] ){
               this->tma = buffer[i+4];
               this->tmb = buffer[i+5];
               this->client_addressed = true;
            }

            // Host has requested the Client Number of Lamps
            if (  buffer[i+0] == 'C'   && 
                  buffer[i+1] == 'N'   &&
                  buffer[i+2] == 'L'   &&
                  //buffer[i+3] == '?'   ){
                  buffer[i+3] == '?'   &&
                  this->client_addressed == true){
               this->__CNL_requested = true;
            }

            // Host has requested lamp parameters
            if (  buffer[i+0] == 'L'   &&
                  buffer[i+1] == 'I'   &&
                  buffer[i+2] == 'D'   &&
                  buffer[i+3] == ':'   &&
                  buffer[i+4] == 255   &&
                  buffer[i+5] == 255   ){
               this->__ALL_requested = true;
            }

            // Host has requested lamp parameters
            if (  buffer[i+0] == 'P'   &&
                  buffer[i+1] == 'A'   &&
                  buffer[i+2] == 'R'   &&
                  buffer[i+3] == '?'   ){
               this->__ALL_requested = true;
            }

            // Host has requested lamp parameters
            uint8_t* tmlid;
            this->lamp->getID(tmlid);
            if (  buffer[i+0] == 'L'   &&
                  buffer[i+1] == 'I'   &&
                  buffer[i+2] == 'D'   &&
                  buffer[i+3] == ':'   &&
                  buffer[i+4] == tmlid[0]   &&
                  buffer[i+5] == tmlid[1]   ){
               this->__lamp_addressed = true;
            }

            // Host is setting lamp bulbs target colors
            if (  buffer[i+0] == 'B'   &&
                  buffer[i+1] == 'T'   &&
                  buffer[i+2] == 'C'   &&
                  buffer[i+3] == '!'   ){

               uint8_t tmc[3];
               for (int j = 0; j < 10; j++) {
                  tmc[0] = buffer[j*3 + i + 4];
                  tmc[1] = buffer[j*3 + i + 5];
                  tmc[2] = buffer[j*3 + i + 6];
                  this->lamp->setBulbTargetRGB(j, tmc);
               }
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
