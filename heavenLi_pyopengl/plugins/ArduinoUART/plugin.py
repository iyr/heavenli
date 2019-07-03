
# Necessary import for all plugins
from lampClass import *

import glob, serial, sys, time, traceback, random
from cobs import cobs

# Largely based on solution by Thomas on stackoverflow
# https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
def getSerialPorts():
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

class Plugin():

    def __init__(self):
        self.lamps = []
        self.clients = []
        self.devices = []
        self.getDevices()
        self.curTime = time.time()
        self.t0 = 0
        pass

    # This class abstracts Serial Devices connected to the system
    # Note: All HeavenLi Client Devices are Serial Devices, but not all Serial Devices are Clients
    class Device():
        def __init__(self, port = None):
            print("ARDUINO PLUGIN: creating device on port: ", port)

            # Current Serial Port of the client device, may change based on system
            self.port = port

            # Whether or not device is a heavenli client:
            # 0:  Unknown, ambiguous, default
            # 1:  Device is a heavenli client
            # -1: Device is known NOT to be a heavenli client (mutable)
            self.isClient = 0

            # Whether or not a heavenli client device is trying to connect,
            # Used to establish three-way handshake for tcp-style connection
            self.synReceived = False

            # Whether or not the client device has established a connection
            self.connectionEstablished = False

            # Serial Port of the device
            self.serialDevice = serial.Serial(port, 115200)
            self.serialDevice.close()
            self.serialDevice.open()

            # List of all lamps handled by device
            self.connectedLamps = []

            # ID of the client device
            self.clientID = [None, None]

        def __del__(self):
            self.serialDevice.close()
            del self.serialDevice
            pass

        # See if device has received any data
        def listen(self):
            try:
                bytesToRead = self.serialDevice.inWaiting()
                if (bytesToRead > 0):
                    print("[HOST] Incoming Bytes: " + str(int(bytesToRead)))
                    zeroByte = b'\x00'
                    mess = self.serialDevice.read_until( zeroByte )
                    print("Data BEFORE COBS decoding: ", mess)
                    #mess = str(cobs.decode( mess[0:-1] )[:-1])[2:-1] ORIGINAL
                    mess = str(cobs.decode( mess[:-1] ) )[2:-1]
                    print("Data received. Packet:", mess)

                    if ("CID!" in str(mess)):
                        pos = str(mess).index("CID!")+4
                        if ("\\x" in mess):
                            print("CID contains improperly formatted bytes")
                            print("Ungarbling bytes...")
                            demess = mess[pos:pos+8].encode(encoding="utf-8")
                            ID_a = int(demess[2:4], 16)
                            ID_b = int(demess[6:8], 16)
                            print("ungarbled demess:", ID_a, ID_b, chr(ID_a), chr(ID_b))
                        else:
                            demess = mess[pos:pos+2]
                            ID_a = ord(demess[0])
                            ID_b = ord(demess[1])
                            print("demess:", ID_a, ID_b)

                        if (    ID_a == 255 or
                                ID_b == 255 ):
                            print("Invalid Client ID:", ID_a, ID_b)
                            newID = [random.randint(1, 254), random.randint(1, 254)]
                            self.setClientID(newID)
                        else:
                            print("Received client ID:", ID_a, ID_b)
                            self.clientID[0] = ID_a
                            self.clientID[1] = ID_b
                            self.serialDevice.flushInput()

            except Exception as OOF:
                self.synReceived = False
                self.connectionEstablished = False
                self.serialDevice.close()
                del self.serialDevice
                print(traceback.format_exc())
                print("Error Decoding Packet: ", OOF)

        # This function performs the TCP-like three-way handshake
        # to connect to heavenli client devices
        def establishConnection(self):
            try:
                bytesToRead = self.serialDevice.inWaiting()
                if (bytesToRead > 0):
                    # Listen for Synchronize Packet from client devices
                    print("[HOST] Incoming Bytes: " + str(int(bytesToRead)))
                    zeroByte = b'\x00'
                    mess = self.serialDevice.read_until( zeroByte )
                    print("Data BEFORE COBS decoding: ", mess)
                    mess = str(cobs.decode( mess[0:-1] )[:-1])[2:-1]

                    # If Synchronize Packet received, note it, then send a synack packet
                    if (mess == "SYN"):
                        print("Syn packet received. Packet:", mess)
                        print("Sending synack packet")
                        self.synReceived = True
                        enmass = cobs.encode(b'SYNACK')+b'\x01'+b'\x00'
                        self.serialDevice.write(enmass)

                    # If Ack Packet received, we know this device is a client
                    elif (  mess == "ACK" and 
                            self.synReceived == True and
                            self.connectionEstablished == False):
                        print("CONNECTION ESTABLISHED :D")
                        self.isClient = 1
                        self.connectionEstablished = True
                        self.requestClientID()
                    else:
                        pass
                        print("Data received. Packet:", mess)

            except Exception as OOF:
                self.synReceived = False
                self.connectionEstablished = False
                self.serialDevice.close()
                del self.serialDevice
                print(traceback.format_exc())
                print("Error Decoding Packet: ", OOF)

        def requestNumLamps(self):
            enmass = cobs.encode(b'NL?')+b'\x01'+b'\x00'
            self.serialDevice.write(enmass)
            pass
            return

        def requestNumBulbs(self, lamp):
            enmass = cobs.encode(b'NB?')+b'\x01'+b'\x00'
            self.serialDevice.write(enmass)
            pass
            return

        def setNumBulbs(self, lamp, newNumBulbs):
            varInBytes = (newNumBulbs).to_bytes(1, byteorder='little')
            enmass = cobs.encode(b'NB:')+varInBytes+b'\x01'+b'\x00'
            self.serialDevice.write(enmass)
            pass
            return

        def requestClientID(self):
            enmass = cobs.encode(b'CID?')+b'\x01'+b'\x00'
            self.serialDevice.write(enmass)
            #print("Requesting Client ID on port:", self.port)
            pass
            return

        def setClientID(self, newID):
            enmass = cobs.encode(b'CID!'+bytes(newID))+b'\x01'+b'\x00'
            print(enmass)
            self.serialDevice.write(enmass)
            pass
            return
            
        def requestAllParamters(self, lamp):
            pass
            return

        def __del__(self):
            print("Removing device on port:", self.port)
            self.serialDevice.close()
            del self.serialDevice
            return

    # Necessary for Heavenli integration
    def update(self):
        if (time.time() - self.curTime > 1.0):
            pass
            self.getDevices()
            #print("Number of Devices:", len(self.devices))
            #print(self.devices)
            try:
                for i in range(len(self.devices)):
                    if (self.devices[i].isClient == 1):
                        self.devices[i].listen()
                        self.devices[i].requestClientID()
                    else:
                        pass
                        #print("Listening for SYN packets on: ", self.devices[i].port)
                        self.devices[i].establishConnection()

            except Exception as OOF:
                print(traceback.format_exc())
                print("Error:", OOF)
                del self.devices[i]

            self.curTime = time.time()
        pass
        return

    # Scans Serial ports for potential Heavenli clients
    def getDevices(self):
        ports = getSerialPorts()
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

