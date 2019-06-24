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
      void           setBulbsTargetRGB(float* bulbsTargetRGB);
      void           setBulbsCurrentRGB(float* CurrentRGB);
      void           setNumBulbs(unsigned int newNumBulbs);
      void           init();
      void           update(float frameTime);

   private:
      float**        bulbsTargetRGB;
      float**        bulbsCurrentRGB;
      unsigned int   numBulbs;
      unsigned int   angularOffset;
      unsigned int   arrangement;

      int            masterSwitchBehavior = -1;

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
      bool           ackReceived;

};


#endif 
