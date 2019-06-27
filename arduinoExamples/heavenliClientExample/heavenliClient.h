#ifndef hliClient
#define hliClient

#include "Arduino.h"
#include "heavenliLamp.h"

class heavenliClient 
{
   public:
      heavenliClient();
      bool           establishConnection();
      bool           connectionEstablished;
      void           init();
      void           processPacket(const uint8_t* buffer, size_t size);
      //void           update(hliLamp);
      void           update(hliLamp lamp);
      //void           update(hliLamp* lamps);
      size_t         outPacket(uint8_t*& buffer);

   private:
      int            getID();
      void           setID(int id);

      int            numLamps;
      bool           synackReceived;
      char*          id;

};

#endif 
