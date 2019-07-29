
/*
 * Arduino Code that implements a HeavenLi client device
 */

// HeavenLi Client Library
#include "heavenliClient.h"
// Necessary for HeavenLi host to send/receive data
#include <PacketSerial.h>
// Drive Adafruit NeoPixel LEDs
#include <Adafruit_NeoPixel.h>

// Set data pin that NeoPixel strip is attached to
#define LED_PIN   8     // (Circuit Playground Express)
//#define LED_PIN   17    // (Circuit Playground Classic)

// Set number of NeoPixel LEDs
#define LED_COUNT 10

// Initialize NeoPixel Strip object
Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

// Initialize HeavenLi client object
heavenliClient client;

// Initialize PacketSerial object
PacketSerial   commPort;

void setup() {
   // Start HeavenLi Client
   client.init();
   // Set Number of 'bulbs' on the client's virtual lamp to 2
   client.lamp.setNumBulbs(2);

   // Set which function is called when a serial packet is received
   commPort.setPacketHandler(&packetReceived);
   // Set baud rate for PacketSerial objcet
   commPort.begin(115200);

   // Start NeoPixel Strip
   strip.begin();
   // Set all LEDs to black
   strip.show();
   // Set max brightness
   strip.setBrightness(128);
}

void loop() {
   // Initialize empty array of bytes to store bulb colors
   uint8_t rgb[3];

   // Get RGB values of the indexed bulb on the virtual lamp
   client.lamp.getBulbCurrentRGB(0, rgb);
   // Set NeoPixel LED color to bulb color with basic color calibration
   for (int i = 0; i < 5; i++) {
      strip.setPixelColor(i, strip.Color(rgb[0], (rgb[1]*2)/3, rgb[2]/2));
   }

   // Get RGB values of the indexed bulb on the virtual lamp
   client.lamp.getBulbCurrentRGB(1, rgb);
   // Set NeoPixel LED color to bulb color with basic color calibration
   for (int i = 5; i < 10; i++) {
      strip.setPixelColor(i, strip.Color(rgb[0], (rgb[1]*2)/3, rgb[2]/2));
   }

   // Update the colors of the LED strip
   strip.show();

   // Update HeavenLi Client state
   client.update();

   // Stores the size, in bytes, of the packet the client will send to the host
   size_t size;
   // Initialize an empty buffer that can store up to 56 bytes
   uint8_t buffer[56];
   // Prepare serial packet for client to send to host
   size = client.outPacket(buffer);
   // Send client packet to host
   commPort.send(buffer, size);
   // Update the PacketSerial state
   commPort.update();
}

void packetReceived(const uint8_t* buffer, size_t size) {
   // Process packet from HeavenLi host to update state
   client.processPacket(buffer, size); 
   return;
}
