#include "heavenliClient.h"
#include <PacketSerial.h>

heavenliClient client;
PacketSerial commPort;

void setup() {
   client.init();
   commPort.setPacketHandler(&packetReceived);
   commPort.begin(115200);
}

void loop() {
   client.update();
   commPort.update();
}

void packetReceived(const uint8_t* buffer, size_t size) {
   client.processPacket(buffer, size);
   return;
}
