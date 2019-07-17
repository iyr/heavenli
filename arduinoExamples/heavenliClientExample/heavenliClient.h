#ifndef hliClient #define hliClient

#include "Arduino.h"
#include "heavenliLamp.h"

class heavenliClient 
{
   public:
      heavenliClient();
      bool        establishConnection();
      bool        connectionEstablished;
      void        init(hliLamp* lamp);
      void        init(hliLamp* lamps, uint8_t numLamps);
      void        processPacket(const uint8_t* buffer, size_t size);
      void        update();
      void        update(hliLamp* lamp);
      void        update(hliLamp* lamps, uint8_t numLamps);
      int         getNumLamps();
      size_t      outPacket(uint8_t*& buffer);

   private:
      hliLamp*    lamp;
      void        assignLampID();
      uint32_t    runtimeCounter1;  // Used for timing
      uint32_t    timeoutCounter;   // 
      uint32_t**  lampIDs = NULL;
      int         getID();
      void        setID(uint8_t* newID);
      const int   IDaddress = 'h';
      uint32_t    timeOut;

      int         numLamps;
      uint8_t     id[2];
      uint8_t     tma;
      uint8_t     tmb;
      bool        synackReceived    = false;
      bool        isConnected;      // True if the connected device is a valid heavenli client
      bool        outBufferFull     = false; // True if the output buffer is full
      
      bool        client_addressed  = false; // True iff packet received has addressed this client by ID

      bool        __CNL_requested   = false; // Plugin has requested total number of lamps of client
      bool        __CNL_sent        = false; // Plugin has requested total number of lamps of client
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
