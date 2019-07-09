
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
            if (str(port) not in "/dev/ttyAMA0" and
                str(port) not in "/dev/ttyS0"):
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

    # Necessary for Heavenli integration
    def update(self):
        if (time.time() - self.curTime > 1.0):
            pass
            #if (len(self.devices) <= len(ports)):
            if (len(self.devices) <= 0):
                self.getDevices()
            #self.getDevices()
            try:
                # Iterate through all connected devices
                for i in range(len(self.devices)):

                    # Listen for data on known client devices
                    if (self.devices[i].isClient == 1):
                        self.devices[i].listen()

                        # Device is a client, get ID
                        if (len(self.devices[i].clientID) != 2):
                            self.devices[i].requestClientID()

                        # If Device has no lamps (ready), 
                        # ping until lamps are available
                        if (self.devices[i].getNumLamps() == 0 and
                                len(self.devices[i].clientID) == 2):
                            self.devices[i].requestNumLamps()
                    else:
                        # Attempt to establish connection
                        # if the device is a heavenli client.
                        self.devices[i].establishConnection()
                        pass

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

            # First call
            if (len(self.devices) <= 0):
                print("Found Serial devices on ports: " + str(ports))
                for i in range(len(ports)):
                    self.devices.append(self.Device(ports[i]))

            # Successive calls
            else:
                for i in range(len(self.devices)):
                    if (self.devices[i].port not in ports):
                        print("Found Serial devices on port: " + str(ports))
                        self.devices.append(self.Device(ports[i]))
        print("ARDUINO PLUGIN: number of devices: " + str(len(self.devices)))
        return
                    
    # Collect lamps from devices
    def getLamps(self):

        # Iterate through connected devices
        for i in range(len(self.devices)):

            # Iterate through connected lamps on devices
            for j in range(len(self.devices[i].getConnectedLamps())):

                # Check for any new lamps not already connected
                if (self.devices[i].getConnectedLamps()[j] not in self.lamps):
                    self.lamps.append(self.devices[i].getConnectedLamps()[j])
        print("ARDUINO PLUGIN: number of lamps: " + str(len(self.lamps)))
        return self.lamps

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
            self.synackSent = False
            self.synackTimeout = time.time()

            # Whether or not the client device has established a connection
            self.connectionEstablished = False

            # Serial Port of the device
            self.serialDevice = serial.Serial()
            self.serialDevice.port = port
            self.serialDevice.baudrate = 115200
            self.serialDevice.xonxoff = True
            self.serialDevice.setDTR(True)
            self.serialDevice.setRTS(True)
            self.serialDevice.timeout = 1.0
            self.serialDevice.write_timeout = 10.0
            self.serialDevice.open()

            # List of all lamps handled by device
            self.connectedLamps = []

            # ID of the client device
            self.clientID = [255, 255]


        def __del__(self):
            self.serialDevice.close()
            print("Deleting Serial Device on ", self)
            del self.serialDevice
            pass

        def getLampIDs(self):
            tml = []
            if (len(self.connectedLamps) <= 0):
                return tml
            else:
                for i in range(len(self.connectedLamps)):
                    tml.append(self.connectedLamps[i].lid)

        # See if device has received any data
        def listen(self):
            try:
                bytesToRead = self.serialDevice.inWaiting()
                if (bytesToRead > 0):
                    print("[HOST] Incoming Bytes: " + str(int(bytesToRead)))
                    zeroByte = b'\x00'
                    mess = self.serialDevice.read_until( zeroByte )
                    print("Data BEFORE COBS decoding: ", mess)
                    mess = str(cobs.decode( mess[:-1] ) )[2:-1]

                    # Fix non-character bytes
                    while (mess.count("\\x") > 0):
                        badBytePos = mess.index("\\x")
                        tmc = int(mess[badBytePos+2:badBytePos+4], 16)

                        # Splice bad byte out of string
                        tmsb = mess[badBytePos+4:]
                        tmsa = mess[:badBytePos]+str(chr(tmc))
                        mess = tmsa+tmsb
                    print("Data received. Packet:", mess)

                    # Get Client ID
                    if ("CID!" in str(mess)):

                        # Parse Client ID from packet
                        pos = str(mess).index("CID!")+4
                        demess = mess[pos:pos+2]

                        ID_a = ord(demess[0])
                        ID_b = ord(demess[1])
                        print("demess:", ID_a, ID_b, chr(ID_a), chr(ID_b))

                        # Check if ID is valid
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
                    elif (("CID:" + str(self.clientID)) in mess):
                        print("Received packet from CID:" + str(self.clientID))

                    # Get Client number of lamps
                    tms = chr(self.clientID[0]) + chr(self.clientID[1])
                    if (("CID:" + tms) in str(mess) and ("CNL!" in str(mess))):
                        pos = mess.index("CNL!")+4
                        demess = mess[pos:pos+1]
                        numLamps = ord(demess[0])
                        tml = []
                        if (numLamps > 1):
                            print("Client: " + str(self.clientID) + " has " + str(numLamps) + " lamps connected")
                        elif (numLamps == 1):
                            print("Client: " + str(self.clientID) + " has " + str(numLamps) + " lamp connected")

                        for i in range(numLamps):
                            tml.append(Lamp())

                        self.connectedLamps = tml
                        print("connectedLamps", self.connectedLamps)

                    # Get Lamp ID
                    if ((("CID:" + str(self.clientID)) in str(mess)) and ("LID:" in str(mess))):

                        # Parse Lamp ID from packet
                        pos = str(mess).index("LID:")+4
                        demess = mess[pos:pos+2]
                        ID_a = ord(demess[0])
                        ID_b = ord(demess[1])

                        # Check if ID is valid
                        if (    ID_a == 255 or
                                ID_b == 255 ):
                            print("Invalid Lamp ID:", ID_a, ID_b)
                            newID = [random.randint(1, 254), random.randint(1, 254)]
                            #self.setClientID(newID)
                        else:
                            print("Received Lamp ID:", ID_a, ID_b)
                            if (len(self.connectedLamps) <= 0):
                                tml = Lamp()
                                tml.setID(str(ID_a)+str(ID_b))
                                self.connectedLamps.append(tml)
                            self.serialDevice.flushInput()

            except Exception as OOF:
                self.synackSent = False
                self.synReceived = False
                self.connectionEstablished = False
                self.serialDevice.reset_output_buffer()
                self.serialDevice.close()
                if (self.isClient < 2):
                    del self.serialDevice
                print(traceback.format_exc())
                print("Error Decoding Packet: ", OOF)

            return  # END: listen()

        # This function performs the TCP-like three-way handshake
        # to connect to heavenli client devices
        def establishConnection(self):
            try:
                bytesToRead = self.serialDevice.in_waiting
                if (bytesToRead > 0):
                    # Listen for Synchronize Packet from client devices
                    print("[HOST] Incoming Bytes: " + str(int(bytesToRead)))
                    zeroByte = b'\x00'
                    mess = self.serialDevice.read_until( zeroByte )
                    print("Data BEFORE COBS decoding: ", mess)
                    mess = str(cobs.decode( mess[0:-1] )[:-1])[2:-1]

                    # If Synchronize Packet received, note it, then send a synack packet
                    #if (mess == "SYN" and
                        #self.synackSent == False):
                    if (mess == "SYN"):
                        print("Syn packet received. Packet:", mess)
                        print("Sending SynAck packet")
                        self.synReceived = True

                        # Clear Output Buffer
                        while (self.serialDevice.out_waiting > 0):
                            self.serialDevice.reset_output_buffer()
                            pass
                        enmass = cobs.encode(b'SYNACK')+b'\x01'+b'\x00'
                        time.sleep(0.1)
                        self.serialDevice.write(enmass)
                        self.synackSent = True
                        print("SynAck sent")
                        self.synackTimeout = time.time()

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
                        #print("Data received. Packet:", mess)

                else:
                    self.synReceived = False

                #if (time.time() - self.synackTimeout > 1.0):
                    #synackSent = False
                    #self.synackTimeout = time.time()

            except Exception as OOF:
                self.synReceived = False
                self.synackSent = False
                self.connectionEstablished = False
                self.serialDevice.reset_output_buffer()
                self.serialDevice.close()
                if (self.isClient <= 1):
                    del self.serialDevice
                print(traceback.format_exc())
                print("Error Decoding Packet: ", OOF)

        # Returns a list of currently connected lamps
        def getConnectedLamps(self):
            return self.connectedLamps

        # Number of lamps host is coordinating
        def getNumLamps(self):
            return len(self.getConnectedLamps())

        # Request number of lamps on the client device (for reference)
        def requestNumLamps(self):
            print("Requesting number of lamps on client:" + str(self.clientID))
            tms = [(self.clientID[0]), (self.clientID[1])]
            #enmass = cobs.encode(b'CID:'+bytearray(tms))+b'\x01'+b'\x00'
            enmass = cobs.encode(b'CID:'+bytearray(tms)+b'CNL?')+b'\x01'+b'\x00'
            print("enmass: ", enmass)
            self.serialDevice.write(enmass)
            pass
            return

        # Get alias of lamp
        def requestAlias(self, lamp):
            enmass = cobs.encode(b'LID:')+bytes(lamp.getID())
            enmass += cobs.encode(b'KN?')+b'\x01'+b'\x00'
            self.serialDevice.write(enmass)
            pass
            return

        def requestNumBulbs(self, lamp):
            enmass = cobs.encode(b'LID:')+bytes(lamp.getID())
            enmass += cobs.encode(b'NB?')+b'\x01'+b'\x00'
            self.serialDevice.write(enmass)
            pass
            return

        def setNumBulbs(self, lamp, newNumBulbs):
            varInBytes = (newNumBulbs).to_bytes(1, byteorder='little')
            enmass = cobs.encode(b'NB!')+varInBytes+b'\x01'+b'\x00'
            self.serialDevice.write(enmass)
            pass
            return

        def requestClientID(self):
            enmass = cobs.encode(b'CID?')+b'\x01'+b'\x00'
            try:
                self.serialDevice.write(enmass)
            except Exception as OOF:
                self.synReceived = False
                self.connectionEstablished = False
                self.serialDevice.close()
                if (self.isClient <= 1):
                    del self.serialDevice
                print(traceback.format_exc())
            #print("Requesting Client ID on port:", self.port)
            pass
            return

        def setClientID(self, newID):
            enmass = cobs.encode(b'CID!'+bytes(newID))+b'\x01'+b'\x00'
            print(enmass)
            self.serialDevice.write(enmass)
            pass
            return
            
        def requestAllParameters(self, lamp):
            requestAlias(lamp)
            pass
            return

