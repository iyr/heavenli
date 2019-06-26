import colorsys
from rangeUtils import *

class Lamp:
    
    def __init__(self):#,
            #numBulbs=3,
            #arrangement=0,
            #alias='demo',
            #angularOffset=0,
            #masterSwitchBehavior=-1,
            #):

        # Angle In which bulbs are arranged
        self.angularOffset = None

        # 0 - Lamp is Circularly Arranged
        # 1 - Lamp is LinearLy Arranged
        self.arrangement = None

        # -1 - Bulbs turn to White Light when clock is touched
        # -2 - Bulbs turn to prev color(s) when clock is touched, default is white when no prev
        # -3 - Bulbs unaffected by Central Clock input
        # 0+ - Bulbs turn to selected (by index) user-saved color profile
        self.masterSwitchBehavior = -1

        self.mainLightOn = False

        # Name or ID
        self.alias = None

        # Number of Bulbs
        self.numBulbs = None

        self.bulbsPreviousHSV = []
        self.bulbsCurrentHSV = []
        self.bulbsTargetHSV = []
        for i in range(10):
            self.bulbsCurrentHSV.append((i*(1.0/3)+0.16667, 1.00, 1.00))
        self.bulbsTargetHSV = self.bulbsCurrentHSV.copy()

    def alias(self):
        return self.alias

    def setAlias(self, newAlias):
        self.alias = newAlias
        return

    # Get/Set angle offset for lamp
    def getAngle(self):
        return self.angularOffset

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
        self.arrangement = n
        return

    # Get list of current RGB values for each bulb
    def getBulbRGB(self, n):
        tmp = colorsys.hsv_to_rgb(
                self.bulbsCurrentHSV[n][0],
                self.bulbsCurrentHSV[n][1],
                self.bulbsCurrentHSV[n][2])
        return tmp

    # Get list of target RGB values for each bulb
    def getBulbTargetRGB(self, n):
        tmp = colorsys.hsv_to_rgb(
                self.bulbsTargetHSV[n][0],
                self.bulbsTargetHSV[n][1],
                self.bulbsTargetHSV[n][2])
        return tmp

    # Set target RGB values for 'bulb'
    def setBulbRGB(self, bulb, RGBcolor):
        self.bulbsTargetHSV[bulb] = colorsys.rgb_to_hsv(RGBcolor[0], RGBcolor[1], RGBcolor[2])

    # Set/Get target Hue/Sat/Val for 'bulb'
    def getBulbtHSV(self, bulb):
        return self.bulbsTargetHSV[bulb]

    def setBulbTargetHSV(self, bulb, HSVcolor):
        self.bulbsTargetHSV[bulb] = HSVcolor

    def setBulbsTargetHSV(self, HSVcolor):
        for i in range(self.numBulbs):
            self.bulbsTargetHSV[i] = HSVcolor

    # Get current Hue/Sat/Vale for 'bulb'
    def getBulbHSV(self, bulb):
        return self.bulbsCurrentHSV[bulb]

    # Set/Get a list of tuples representing RGB values for each bulb
    def getBulbsRGB(self):
        tmp = []
        for i in range(self.numBulbs):
            tmp.append(colorsys.hsv_to_rgb(
                self.bulbsCurrentHSV[i][0],
                self.bulbsCurrentHSV[i][1],
                self.bulbsCurrentHSV[i][2]))
        return tmp

    def setBulbsRGB(self, RGBs):
        for i in range(self.numBulbs):
            self.bulbsTargetHSV[i] = colorsys.rgb_to_hsv(RGBs[i])


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

    # Returns the control mode for the lamp
    def getControlMode(self, n):
        return self.masterSwitchBehavior

    # Used for animating Bulb Color Transitions
    # frameCursors is and array of floats where the cursor
    # in frameCursors[n] corresponds to bulb[n]
    # Updates all frameCursors at once
    def setFrameCursors(self, frameCursors):
        self.frameCursors = frameCursors


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
                    print("Error:", OOF)

        # Lamp is being turn off
        if (self.mainLightOn == True) and (lightOn == False):
            if self.masterSwitchBehavior >= -2:
                try:
                    for i in range(self.numBulbs):
                        self.bulbsTargetHSV[i] = (0.0, 0.0, 0.0)
                except Exception as OOF:
                    print("Error:", OOF)
        self.mainLightOn = lightOn


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

