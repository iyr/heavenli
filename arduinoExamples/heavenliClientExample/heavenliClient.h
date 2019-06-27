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
      void           getAlias(char*& knickname);
      void           setAlias(char* newKnickname);

      float          getAngularOffset();
      void           setAngularOffset(float newAO);
      void           setAngularOffset(int8_t newAO);

      uint8_t        getArrangement();
      void           setArrangement(uint8_t newArn);

      void           getBulbCurrentRGB(uint8_t bulb, uint8_t* RGB);
      void           setBulbCurrentRGB(uint8_t bulb, uint8_t* newRGB);
      void           getBulbsCurrentRGB(uint8_t** RGB);
      void           setBulbsCurrentRGB(uint8_t* newRGB);

      void           getBulbTargetRGB(uint8_t bulb, uint8_t* RGB);
      void           setBulbTargetRGB(uint8_t bulb, uint8_t* newRGB);
      void           getBulbsTargetRGB(uint8_t** RGB);
      void           setBulbsTargetRGB(uint8_t* newRGB);

      int            getID();
      void           setID(int newID);
      
      char           getMasterSwitchBehavior();
      void           setMasterSwitchBehavior(char newBehavior);

      void           setNumBulbs(uint8_t newNumBulbs);
      uint8_t        getNumBulbs();

      bool           getBulbCountMutability();
      uint8_t        getMetaLampLevel();
      void           getValidBulbQuantities(uint8_t*& quantities);

      void           init();
      void           update(float frameTime);

   private:
      float          angularOffset;
      
      uint8_t**      bulbsTargetRGB;
      uint8_t**      bulbsCurrentRGB;
      uint8_t*       validBulbCounts;
      uint8_t        numBulbs;
      uint8_t        arrangement;
      uint8_t        metaLampLevel;


      char*          alias;
      char*          id;
      char           masterSwitchBehavior;

      bool           mutableBulbCount;
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
      int            getID();
      void           setID(int id);

      int            numLamps;
      bool           synackReceived;
      char*          id;

};


#endif 
