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
      void outPacket(uint8_t*& buffer, size_t size);

   private:
      int numLamps;
};

void packetReceived();

#endif 
