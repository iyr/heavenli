import colorsys
from rangeUtils import *

class Lamp:
    
    def __init__(self,
            numBulbs=3,
            arn=0,
            name='demo',
            angularOffset=0,
            mainsControlMode=-1,
            ):

        # Angle In which bulbs are arranged
        self.angularOffset = angularOffset

        # 0 - Lamp is Circularly Arranged
        # 1 - Lamp is LinearLy Arranged
        self.arn = arn

        self.frameCursors = [0, 0, 0, 0, 0, 0]

        # -1 - Bulbs turn to White Light when clock is touched
        # -2 - Bulbs turn to prev color(s) when clock is touched, default is white when no prev
        # -3 - Bulbs unaffected by Central Clock input
        # 0+ - Bulbs turn to selected (by index) user-saved color profile
        self.mainsControlMode = -1

        self.mainLightOn = False

        # Name or ID of Lamp
        self.name = name

        # Number of Bulbs
        self.numBulbs = numBulbs

        self.bulbsPreviousHSV = []
        self.bulbsCurrentHSV = []
        self.bulbsTargetHSV = []
        for i in range(self.numBulbs):
            self.bulbsCurrentHSV.append((i*(1.0/self.numBulbs)+0.3333, 0.75, 0.75))
            #self.bulbsCurrentHSV.append((i*(360/numBulbs)+60, 255, 255))
        self.bulbsTargetHSV = self.bulbsCurrentHSV.copy()

    def getAngle(self):
        return self.angularOffset

    def setAngle(self, ang):
        if ang >= 360:
            ang -= 360
        if ang <= 0:
            ang += 360
        self.angularOffset = ang

    def getArn(self):
        return self.arn

    def setArn(self, n):
        self.arn = n

    def getBulbRGB(self, n):
        tmp = colorsys.hsv_to_rgb(
                self.bulbsCurrentHSV[n][0],
                self.bulbsCurrentHSV[n][1],
                self.bulbsCurrentHSV[n][2])
        return tmp

    def getBulbsRGB(self):
        tmp = []
        for i in range(self.numBulbs):
            tmp.append(colorsys.hsv_to_rgb(
                self.bulbsCurrentHSV[i][0],
                self.bulbsCurrentHSV[i][1],
                self.bulbsCurrentHSV[i][2]))
        return tmp

    def setBulbRGB(self, n, RGB):
        self.bulbsTargetHSV[n] = colorsys.rgb_to_hsv(RGB[0], RGB[1], RGB[2])

    def setBulbsRGB(self, RGBs):
        for i in range(self.numBulbs):
            self.bulbsTargetHSV[i] = colorsys.rgb_to_hsv(RGBs[i])
        return tmp

    def updateBulbs(self, frametime):

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
                threshold = 0.01
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

    def getControlMode(self, n):
        return self.mainsControlMode

    # Used for animating Bulb Color Transitions
    # frameCursors is and array of floats where the cursor
    # in frameCursors[n] corresponds to bulb[n]
    # Updates all frameCursors at once
    def setFrameCursors(self, frameCursors):
        self.frameCursors = frameCursors

    def getMainLight(self):
        return self.mainLightOn

    def setMainLight(self, lightOn):

        # Lamp is being turned on
        if (self.mainLightOn == False) and (lightOn == True):
            if self.mainsControlMode == -1:
                for i in range(self.numBulbs):
                    self.bulbsTargetHSV[i] = (1.0, 0.0, 1.0)

        # Lamp is being turn off
        if (self.mainLightOn == True) and (lightOn == False):
            if self.mainsControlMode >= -2:
                for i in range(self.numBulbs):
                    self.bulbsTargetHSV[i] = (0.0, 0.0, 0.0)
        self.mainLightOn = lightOn

    def setNumBulbs(self, n):
        if n < 1:
            n = 1
        elif n > 6:
            n = 6
        if n > len(self.bulbsCurrentHSV):
            self.bulbsCurrentHSV.append((n*(1/self.numBulbs), 1.0, n/self.numBulbs))
            self.bulbsTargetHSV.append((n*(1/self.numBulbs), 1.0, n/self.numBulbs))
        self.numBulbs = n

    def getNumBulbs(self):
        return self.numBulbs

