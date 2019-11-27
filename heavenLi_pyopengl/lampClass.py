import colorsys, traceback
from rangeUtils import *

class Lamp:
    
    def __init__(self):#,
            #numBulbs=3,
            #arrangement=0,
            #alias='demo',
            #angularOffset=0,
            #masterSwitchBehavior=-1,
            #):

        #
        #   INITIALIZE LAMP VARIABLES
        #

        # Lamp name
        self.alias = []

        # Angle In which bulbs are arranged
        self.angularOffset = 0

        # 0 - Lamp is Circularly Arranged
        # 1 - Lamp is LinearLy Arranged
        self.arrangement = 0

        # Arrays that store the colors of the bulbs
        self.bulbsPreviousHSV = []
        self.bulbsCurrentHSV = []
        self.bulbsTargetHSV = []

        # Unique ID
        self.lid = [255, 255]

        # Restricts which quantities of bulbs this lamp can use if
        # is not a meta lamp and it has user-changeable bulb quantities
        # Default (empty list) is all quantities
        self.validBulbCounts = []

        # Determines whether the user can change the bulb quantities from within the interface.
        # Is set to 'False' if only 1 validBulbCount is provided or
        # if the lamp is a meta lamp
        self.mutableBulbCount = False

        # -1 - Bulbs turn to White Light when clock is touched
        # -2 - Bulbs turn to prev color(s) when clock is touched, default is white when no prev
        # -3 - Bulbs unaffected by Central Clock input
        # 0+ - Bulbs turn to selected (by index) user-saved color profile
        self.masterSwitchBehavior = -1

        # 0: recursive base-case for lamp tree, lamps comprised of bulbs
        # 1+: lamps comprised of lamps
        self.metaLampLevel = 0

        self.mainLightOn = True

        # Number of Bulbs
        self.numBulbs = 0

        for i in range(10):
            self.bulbsCurrentHSV.append((i*(1.0/3)+0.16667, 1.00, 1.00))
        self.bulbsTargetHSV = self.bulbsCurrentHSV.copy()

    # Return True iff any of lamp's bulbs/sub-lamps are emitting any amount of light
    def isOn(self):
        for i in range(self.numBulbs):
            if (self.bulbsTargetHSV[i][2] > 0.0):
                return True
        return False

    # Returns True iff all lamp parameters are set
    # and lamp is ready to send/receive color data streams from heavenli
    # Returns False if a requred parameter is not set
    # Dispass: Display lamp checks
    def isReady(self, dispChecks=False):
        # Number of errors
        Err = 0

        #
        # BEGIN: Parameter checks
        #

        # Check if id is set and 
        # set place-holder ID
        if (len(self.lid) != 2):
            tmid = b'\xFF\xFF'
            if (dispChecks):
                print(str(tmid) + ": ERROR: Lamp ID not set or valid")
            Err += 1
        elif (self.lid[0] == 255 and self.lid[1] == 255):
            tmid = b'\xFF\xFF'
            if (dispChecks):
                print(str(tmid) + ": ERROR: Lamp ID not set or valid")
            Err += 1
        elif (dispChecks):
            tmid = bytes(self.lid)
            if (dispChecks):
                if (tmid == b'\xFF\xFF'):
                    print(str(tmid) + ": ERROR: Lamp ID not set or valid")
                    Err += 1
                else:
                    print(str(tmid) + ": PASSED: ID")
        else:
            tmid = bytes(self.lid)

        # Check if alias is valid
        if (len(self.alias) <= 0):
            if (dispChecks):
                print(str(tmid) + ": WARNING: alias not set")
            #Err += 1
            pass
        elif (len(self.alias) > 16):
            if (dispChecks):
                print(str(tmid) + ": ERROR: alias too long: len(alias)" + str(len(self.alias)))
            Err += 1
        elif (dispChecks):
            print(str(tmid) + ": PASSED: Alias: " + self.alias)

        # Check if angularOffset is valid
        if (self.angularOffset == None):
            if (dispChecks):
                print(str(tmid) + ": WARNING: angularOffset not set")
            Err += 1
        elif (dispChecks):
            print(str(tmid) + ": PASSED: Angular Offset:" + str(self.angularOffset))

        # Check if arrangement is valid
        if ((self.arrangement < 0) or (self.arrangement > 1)):
            if (dispChecks):
                print(str(tmid) + ": ERROR: invalid arrangement or no arrangement set: " + str(self.arrangement))
            Err += 1
        elif (dispChecks):
            print(str(tmid) + ": PASSED: Arrangement: " + str(self.arrangement))

        # Check if Number of Bulbs is set and valid
        if ((self.numBulbs <= 0) or (self.numBulbs > 10)):
            if (dispChecks):
                print(str(tmid) + ": ERROR: invalid number of bulbs:" + str(self.numBulbs))
            Err += 1
        elif (dispChecks):
            print(str(tmid) + ": PASSED: Number of Bulbs: " + str(self.numBulbs))

        #
        # END: Parameter checks
        #

        # Lamp passed all checks
        if (Err == 0):
            if (dispChecks):
                print("Lamp: " + str(tmid) + " \"" + str(self.alias) + "\"" + " passed all checks, woot.")
            return True

        # Lamp failed some checks
        else:
            print("Lamp: " + str(tmid) + " failed " + str(Err) + " checks, boo.")
            return False

    # Return the knick-name of the lamp
    def getAlias(self):
        return self.alias

    # Set the knick-name of the lamp
    def setAlias(self, newAlias):
        self.alias = newAlias
        return

    # Get angular offset for lamp
    def getAngle(self):
        return self.angularOffset

    # Set angular offset for lamp
    def setAngle(self, ang):
        if ang >= 360:
            ang -= 360
        if ang <= 0:
            ang += 360
        self.angularOffset = ang
        return

    # Set/Get arrangement for lamp
    def getArn(self):
        return self.arrangement

    def setArn(self, n):
        if (n > 1):
            n = 1
        elif (n < 0):
            n = 0
        self.arrangement = n
        return

    # Get list of target RGB values for each bulb
    def getBulbTargetRGB(self, n):
        tmp = colorsys.hsv_to_rgb(
                self.bulbsTargetHSV[n][0],
                self.bulbsTargetHSV[n][1],
                self.bulbsTargetHSV[n][2])
        return tmp

    # Set target RGB values for 'bulb'
    def setBulbTargetRGB(self, bulb, RGBcolor):
        self.bulbsTargetHSV[bulb] = colorsys.rgb_to_hsv(RGBcolor[0], RGBcolor[1], RGBcolor[2])

    # Set/Get target Hue/Sat/Val for 'bulb'
    def getBulbTargetHSV(self, bulb):
        return self.bulbsTargetHSV[bulb]

    def setBulbTargetHSV(self, bulb, HSVcolor):
        self.bulbsTargetHSV[bulb] = HSVcolor

    def setBulbsTargetHSV(self, HSVcolor):
        for i in range(self.numBulbs):
            self.bulbsTargetHSV[i] = HSVcolor

    # Get current Hue/Sat/Vale for 'bulb'
    def getBulbCurrentHSV(self, bulb):
        return self.bulbsCurrentHSV[bulb]

    # Get list of current RGB values for each bulb
    def getBulbCurrentRGB(self, n):
        tmp = colorsys.hsv_to_rgb(
                self.bulbsCurrentHSV[n][0],
                self.bulbsCurrentHSV[n][1],
                self.bulbsCurrentHSV[n][2])
        return tmp

    # Get a list of tuples representing RGB values for each bulb
    def getBulbsCurrentRGB(self):
        tmp = []
        for i in range(self.numBulbs):
            tmp.append(colorsys.hsv_to_rgb(
                self.bulbsCurrentHSV[i][0],
                self.bulbsCurrentHSV[i][1],
                self.bulbsCurrentHSV[i][2]))
        return tmp

    # Set current colors of all bulbs
    def setBulbsCurrentRGB(self, RGBs):
        for i in range(self.numBulbs):
            self.bulbsTargetHSV[i] = colorsys.rgb_to_hsv(RGBs[i])

    # Returns ID of the lamp
    def getID(self):
        return self.lid

    # Sets ID of the lamp
    def setID(self, n):
        self.lid = n
        return
            
    # Returns the control mode for the lamp
    def getMasterSwitchBehavior(self):
        return self.masterSwitchBehavior

    # Sets the control mode for the lamp
    def setMasterSwitchBehavior(self, n):
        self.masterSwitchBehavior = n;
        return

    
    def getMetaLampLevel(self):
        return self.metaLampLevel

    def setMetaLampLevel(self, newLevel):
        self.metaLampLevel = newLevel
        return

    # Set/get functions for the number of bulbs belonging to the lamp
    def getNumBulbs(self):
        return self.numBulbs

    def setNumBulbs(self, n):
        if n < 1:
            n = 1
        elif n > 6:
            n = 6
        if n > len(self.bulbsCurrentHSV):
            self.bulbsCurrentHSV.append((n*(1/self.numBulbs), 1.0, n/self.numBulbs))
            self.bulbsTargetHSV.append((n*(1/self.numBulbs), 1.0, n/self.numBulbs))
        self.numBulbs = n
        return

    def setBulbCountMutability(self, mutability):
        self.mutableBulbCount = mutability
        return

    def getBulbCountMutability(self):
        return self.mutableBulbCount

    # If current colors =/= target colors, animate smooth color transition
    def updateBulbs(self, frametime):
        if (self.numBulbs is None or self.numBulbs < 1):
            pass
        else:
            for i in range(0, self.numBulbs):
                colC = colorsys.hsv_to_rgb(
                        self.bulbsCurrentHSV[i][0],
                        self.bulbsCurrentHSV[i][1],
                        self.bulbsCurrentHSV[i][2])
                colT = colorsys.hsv_to_rgb(
                        self.bulbsTargetHSV[i][0],
                        self.bulbsTargetHSV[i][1],
                        self.bulbsTargetHSV[i][2])

                if (colC != colT):
                    curR = colC[0]
                    curG = colC[1]
                    curB = colC[2]
                    tarR = colT[0]
                    tarG = colT[1]
                    tarB = colT[2]
                    difR = abs(curR - tarR)
                    difG = abs(curG - tarG)
                    difB = abs(curB - tarB)
                    rd = 0
                    gd = 0
                    bd = 0
                    threshold = 0.05
                    #delta = frametime + i/self.numBulbs
                    #delta /= 2
                    delta = frametime
                    delta *= ((i+2)*2)/(self.numBulbs*3)
    
                    if (difR > threshold):
                        if (tarR > curR):
                            rd = delta
                        else:
                            rd = -(delta)
                    if (difG > threshold):
                        if (tarG > curG):
                            gd = delta
                        else:
                            gd = -(delta)
                    if (difB > threshold):
                        if (tarB > curB):
                            bd = delta
                        else:
                            bd = -(delta)

                    if (difR > threshold):
                        difR = curR + rd
                    else:
                        difR = tarR

                    if (difG > threshold):
                        difG = curG + gd
                    else:
                        difG = tarG

                    if (difB > threshold):
                        difB = curB + bd
                    else:
                        difB = tarB

                    difR = constrain(difR, 0.0, 1.0)
                    difG = constrain(difG, 0.0, 1.0)
                    difB = constrain(difB, 0.0, 1.0)
                    self.bulbsCurrentHSV[i] = colorsys.rgb_to_hsv(difR, difG, difB)
                #self.bulbsCurrentHSV[i] = (
                        #min(self.bulbsTargetHSV[i][0]+frametime, 1.0),
                        #min(self.bulbsTargetHSV[i][1]+frametime, 1.0),
                        #min(self.bulbsTargetHSV[i][2]+frametime, 1.0)
                        #)

    # Set/Get functions for turning lamp on/off
    def getMainLight(self):
        return self.mainLightOn

    def setMainLight(self, lightOn):

        # Turn Lamp On
        if (lightOn):
            try:
                self.setBulbsTargetHSV((1.0, 0.0, 1.0))
            except Exception as OOF:
                print(traceback.format_exc())
                print("Error:", OOF)

        # Turn Lamp Off
        else:
            try:
                self.setBulbsTargetHSV((0.0, 0.0, 0.0))
            except Exception as OOF:
                print(traceback.format_exc())
                print("Error:", OOF)

        # Lamp is being turned on
        #if (self.mainLightOn == False) and (lightOn == True):
            #if (self.masterSwitchBehavior == -1):
                #try:
                    #self.setBulbsTargetHSV((1.0, 0.0, 1.0))
                #except Exception as OOF:
                    #print(traceback.format_exc())
                    #print("Error:", OOF)

        # Lamp is being turn off
        #if (self.mainLightOn == True) and (lightOn == False):
            #if self.masterSwitchBehavior >= -2:
                #try:
                    #self.setBulbsTargetHSV((0.0, 0.0, 0.0))
                #except Exception as OOF:
                    #print(traceback.format_exc())
                    #print("Error:", OOF)
        #self.mainLightOn = lightOn
        return

