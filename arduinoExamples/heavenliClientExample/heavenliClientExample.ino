#include "heavenliClient.h"
#include <PacketSerial.h>

heavenliClient client;
hliLamp        quack;
PacketSerial   commPort;

void setup() {
   pinMode(LED_BUILTIN, OUTPUT);
   client.init(&quack);
   commPort.setPacketHandler(&packetReceived);
   commPort.begin(115200);
   digitalWrite(LED_BUILTIN, HIGH);
   delay(500);
   digitalWrite(LED_BUILTIN, LOW);
   delay(500);
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
   client.update(&quack);
   commPort.update();
   digitalWrite(LED_BUILTIN, LOW);
   //delay(100);
}

void packetReceived(const uint8_t* buffer, size_t size) {
   client.processPacket(buffer, size);
   //commPort.send(buffer, size);
   /*
   char tmp[] = "SYNACK";
   if (strcmp(tmp, buffer) == 0) {
      char tmm[] = "synack received";
      commPort.send(tmm, 16);
   } else {
      char tmm[] = "synack, waiting";
      commPort.send(tmm, 16);
      commPort.send(buffer, size);
   }
   */
   return;
}
