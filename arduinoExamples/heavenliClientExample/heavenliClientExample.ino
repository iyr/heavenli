#include "heavenliClient.h"
#include <PacketSerial.h>

heavenliClient client;
PacketSerial commPort;

void setup() {
   pinMode(LED_BUILTIN, OUTPUT);
   client.init();
   commPort.setPacketHandler(&packetReceived);
   commPort.begin(115200);
   digitalWrite(LED_BUILTIN, LOW);
   delay(1000);
   digitalWrite(LED_BUILTIN, HIGH);
   delay(1000);
   digitalWrite(LED_BUILTIN, LOW);
   delay(1000);
   digitalWrite(LED_BUILTIN, HIGH);
   delay(200);
   digitalWrite(LED_BUILTIN, LOW);
   delay(200);
   digitalWrite(LED_BUILTIN, HIGH);
   delay(200);
   digitalWrite(LED_BUILTIN, LOW);
   delay(200);
   digitalWrite(LED_BUILTIN, HIGH);
   delay(200);
   digitalWrite(LED_BUILTIN, LOW);
}

void loop() {
   size_t size;
   uint8_t* buffer;
   digitalWrite(LED_BUILTIN, HIGH);
   delay(20);

   size = client.outPacket(buffer);
   commPort.send(buffer, size);
   client.update();
   commPort.update();
   digitalWrite(LED_BUILTIN, LOW);
   delay(1000);
}

void packetReceived(const uint8_t* buffer, size_t size) {
   client.processPacket(buffer, size);
   return;
}
