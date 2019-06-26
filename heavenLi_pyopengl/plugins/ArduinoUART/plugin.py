
# Necessary import for all plugins
from lampClass import *

import glob
import serial
import sys
#from time import *
import time
from cobs import cobs

class Plugin():

    def __init__(self):
        self.lamps = []
        self.devices = []
        self.clients = []
        self.getDevices()
        self.curTime = time.time()
        self.t0 = 0
        pass

    # All Clients are devices, but not all devices are clients
    class Device():
        def __init__(self, port = None):
            print("ARDUINO PLUGIN: creating device on port: ", port)

            self.port = port
            self.isClient = False
            self.synReceived = False
            self.serialDevice = serial.Serial(port, 115200)
            self.serialDevice.close()
            self.serialDevice.open()

        def listen(self):
            bytesToRead = self.serialDevice.inWaiting()
            if (bytesToRead > 0):
                try:
                    print("[HOST] Incoming Bytes: " + str(int(bytesToRead)))
                    zeroByte = b'\x00'
                    mess = self.serialDevice.read_until( zeroByte )
                    print("Data BEFORE COBS decoding: ", mess)
                    mess = str(cobs.decode( mess[0:-1] )[:-1])[2:-1]
                    if (mess == "SYN"):
                        print("Syn packet received. Packet:", mess)
                        print("Sending synack packet")
                        self.synReceived = True
                        enmass = cobs.encode(b'SYNACK')+b'\x01'+b'\x00'
                        self.serialDevice.write(enmass)
                    elif (mess == "ACK" and self.synReceived == True):
                        self.isClient = True
                        print("CONNECTION ESTABLISHED :D")
                    else:
                        pass
                        print("Data received. Packet:", mess)
                except Exception as OOF:
                    print("Error Decoding Packet")
                    print("Error: ", OOF)

        def __del__(self):
            print("Removing device on port:", self.port)
            self.serialDevice.close()
            return

    # 
    class Client():
        def __init__(self, 
                alias="quack",
                clientID=None):
            self.alias = alias
            self.clientID = clientID
            self.connected = False
            self.device = None
            self.numLamps = 0

        def setDevice(self, device):
            self.device = device
            return

        def requestLamps(self):
            pass
            return

    # *** END OF CLIENT CLASS ***

    # Necessary for Heavenli integration
    def update(self):
        if (time.time() - self.curTime > 1.0):
            self.getDevices()
            for i in range(len(self.devices)):
                try:
                    if (not self.devices[i].isClient):
                        self.clients
                        #self.devices[i].establishConnection()
                    else:
                        self.devices[i].listen()

                except Exception as OOF:
                    print("Error:", OOF)
                    del self.devices[i]
        pass
        return

    # Scans Serial ports for potential Heavenli clients
    def getDevices(self):
        ports = self.getSerialPorts()
        if (len(ports) <= 0):
            pass
            #print("No Serial devices available :(")
        else:
            if (len(self.clients) <= 0):
                print("Found Serial devices on ports: " + str(ports))
                for i in range(len(ports)):
                    self.devices.append(self.Device(ports[i]))
        return
                    

    def getLamps(self):
        return self.lamps

    # Largely based on solution by Thomas on stackoverflow
    # https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
    def getSerialPorts(self):
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i+1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # excludes current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('dev/tty.*')
        else:
            raise EnvironmetError('Unupported platform, darn shame')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                #print("Port: " + str(port) + " not available")
                pass

        return result
