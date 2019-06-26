#ifndef hliClient
#define hliClient

#include "Arduino.h"

/*
 * heavenli lampClass
 */
class hliLamp
{
   public:
      hliLamp();
      void           setBulbsTargetRGB(byte* bulbsTargetRGB);
      void           setBulbsCurrentRGB(byte* CurrentRGB);
      void           getBulbCurrentRGB(byte bulb, byte* RGB);
      void           setNumBulbs(byte newNumBulbs);
      void           init();
      void           update(float frameTime);

   private:
      float          angularOffset;
      
      uint8_t**      bulbsTargetRGB;
      uint8_t**      bulbsCurrentRGB;
      uint8_t        numBulbs;
      uint8_t        arrangement;

      char           masterSwitchBehavior = -1;

      char*          alias;
      char*          id;
      bool           isMetaLamp;
};

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
      int            numLamps;
      bool           synackReceived;

};


#endif 
