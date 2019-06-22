#ifndef hliClient
#define hliClient

#include "Arduino.h"

class heavenliClient 
{
   public:
      heavenliClient();
      bool establishConnection();
      bool connectionEstablished;
      void init();
      void processPacket(const uint8_t* buffer, size_t size);
      void update();
      size_t outPacket(uint8_t*& buffer);

   private:
      int numLamps;
      bool ackReceived;
};

void packetReceived();

#endif 
