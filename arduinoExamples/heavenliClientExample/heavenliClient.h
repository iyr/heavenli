#ifndef hliClient
#define hliClient

#include "Arduino.h"
#include "heavenliLamp.h"

class heavenliClient 
{
   public:
      heavenliClient();
      bool        establishConnection();
      bool        connectionEstablished;
      void        init();
      void        processPacket(const uint8_t* buffer, size_t size);
      void        update();
      void        update(hliLamp lamp);
      void        update(hliLamp* lamps);
      size_t      outPacket(uint8_t*& buffer);

   private:
      uint32_t    runtimeCounter1;  // Used for timing
      uint32_t    timeoutCounter;   // 
      int         getID();
      void        setID(char* newID);
      const int   IDaddress = 'h';
      uint32_t    timeOut;

      int         numLamps;
      char        id[2];
      bool        synackReceived;
      bool        isConnected;      // True if the connected device is a valid heavenli client
      bool        outBufferFull;    // True if the output buffer is full
      
      bool        __NL_requested;   // Plugin has requested total number of lamps of client
      bool        __CID_requested;  // Plugin has requested the ID of the client device
      bool        __CID_sent;       // CID packet has been sent

      bool        __ALL_requested;  // Plugin has requested all base parameters of a lamp
      bool        __LID_requested;  // Lamp has requested a unique ID from the plugin
      bool        NB_requested;     // Plugin has requested number of bulbs of the lamp
      bool        CM_requested;     // Plugin has requested bulb-count mutability of the lamp
      bool        AR_requested;     // Plugin has requested the current arrangement of the lamp
      bool        VQ_requested;     // Plugin has requested valid bulb quantities of the lamp
      bool        AO_requested;     // Plugin has requested the angular offset of the lamp
      bool        KN_requested;     // Plugin has requested the alias of the lamp
      bool        ID_requested;     // Plugin has requested the ID of the lamp
      bool        LL_requested;     // Plugin has requested the meta-lamp level of the lamp
      bool        SB_requested;     // Plugin has requested the master switch behavior of the lamp
      bool        CC_requested;     // Plugin has requested the current colors of the lamp
      bool        TC_requested;     // Plugin has requested the target colors of the lamp
};

#endif 
