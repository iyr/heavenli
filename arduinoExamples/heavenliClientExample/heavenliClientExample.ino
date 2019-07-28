#include "heavenliClient.h"
#include <PacketSerial.h>
//#include <Adafruit_NeoPixel.h>

#define LED_PIN   17
#define LED_COUNT 10

//Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
heavenliClient client;
PacketSerial   commPort;

bool  flipflop = false;
long  timer = 0;

void setup() {
   pinMode(LED_BUILTIN, OUTPUT);
   client.init();
   commPort.setPacketHandler(&packetReceived);
   commPort.begin(115200);

   //strip.begin();
   //strip.show();
   //strip.setBrightness(64);

   /*
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
   */
   timer = millis();
}

void loop() {
   if (millis() - timer > 1000) {
      if (flipflop) {
         digitalWrite(LED_BUILTIN, LOW);
         flipflop = !flipflop;
      } else {
         digitalWrite(LED_BUILTIN, HIGH);
         flipflop = !flipflop;
      }
      timer = millis();
   }
   uint8_t tmc[3];
   client.lamp.getBulbCurrentRGB(0, tmc);
   /*
   tmc[0] = 128;
   tmc[1] = 64;
   tmc[2] = 8;
   */
   for (int i = 0; i < 10; i++) {
      //strip.setPixelColor(i, strip.Color(tmc[0], tmc[1], tmc[2]));
   }

   //strip.show();
   size_t size;
   uint8_t buffer[56];
   size = client.outPacket(buffer);
   client.update();
   commPort.send(buffer, size);
   commPort.update();
   //delete [] buffer;
}

void packetReceived(const uint8_t* buffer, size_t size) {
   client.processPacket(buffer, size);
   //commPort.send(buffer, size);
   return;
}
