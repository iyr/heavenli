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
        self.alias = None

        # Angle In which bulbs are arranged
        self.angularOffset = None

        # 0 - Lamp is Circularly Arranged
        # 1 - Lamp is LinearLy Arranged
        self.arrangement = None

        # Arrays that store the colors of the bulbs
        self.bulbsPreviousHSV = []
        self.bulbsCurrentHSV = []
        self.bulbsTargetHSV = []

        # Unique ID
        self.ID = []

        # Restricts which quantities of bulbs this lamp can use if
        # is not a meta lamp and it has user-changeable bulb quantities
        # Default (empty list) is all quantities
        self.validBulbCounts = []

        # Determines whether the user can change the bulb quantities from within the interface.
        # Is set to 'False' if only 1 validBulbCount is provided or
        # if the lamp is a meta lamp
        self.mutableBulbCount = None

        # -1 - Bulbs turn to White Light when clock is touched
        # -2 - Bulbs turn to prev color(s) when clock is touched, default is white when no prev
        # -3 - Bulbs unaffected by Central Clock input
        # 0+ - Bulbs turn to selected (by index) user-saved color profile
        self.masterSwitchBehavior = -1

        # 0: recursive base-case for lamp tree, lamps comprised of bulbs
        # 1+: lamps comprised of lamps
        self.metaLampLevel = None

        self.mainLightOn = False

        # Number of Bulbs
        self.numBulbs = None

        for i in range(10):
            self.bulbsCurrentHSV.append((i*(1.0/3)+0.16667, 1.00, 1.00))
        self.bulbsTargetHSV = self.bulbsCurrentHSV.copy()

    # Returns True iff all lamp parameters are set
    # and lamp is to send/receive color data streams from heavenli
    # Returns False if a requred parameter is not set
    def isReady(self):

        # Check if id is set and valid
        if ((len(self.id) != 2):
            print("Lamp ID not set or valid")
            return False

        # Check if alias is valid
        if (len(self.alias) <= 0):
            print(self.id, "alias not set")
            return False
        if (len(self.alias) > 16):
            print(self.id, "alias too long: len(alias)", len(self.alias))
            return False

        # Check if Number of Bulbs is set and valid
        if (numBulbs is None):
            print(self.id, "number of bulbs not set")
            return False
        if ( (numBulbs < 0) or (numBulbs > 10) ):
            print(self.id, "invalid number of bulbs:", self.numBulbs)
            return False

        pass
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
    def getID():
        return self.id

    # Sets ID of the lamp
    def setID(n):
        self.id = n
        return
            
    # Returns the control mode for the lamp
    def getMasterSwitchBehavior(self):
        return self.masterSwitchBehavior

    # Sets the control mode for the lamp
    def setMasterSwitchBehavior(self, n):
        self.masterSwitchBehavior = n;
        return

    
    def getMetaLampLevel():
        return this.metaLampLevel

    def setMetaLampLevel(newLevel):
        this.metaLampLevel = newLevel
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

    def getBulbCountMutability():
        return this.mutableBulbCount

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

        # Lamp is being turned on
        if (self.mainLightOn == False) and (lightOn == True):
            if self.masterSwitchBehavior == -1:
                try:
                    for i in range(self.numBulbs):
                        self.bulbsTargetHSV[i] = (1.0, 0.0, 1.0)
                except Exception as OOF:
                    print(traceback.format_exc())
                    print("Error:", OOF)

        # Lamp is being turn off
        if (self.mainLightOn == True) and (lightOn == False):
            if self.masterSwitchBehavior >= -2:
                try:
                    for i in range(self.numBulbs):
                        self.bulbsTargetHSV[i] = (0.0, 0.0, 0.0)
                except Exception as OOF:
                    print(traceback.format_exc())
                    print("Error:", OOF)
        self.mainLightOn = lightOn
        return

