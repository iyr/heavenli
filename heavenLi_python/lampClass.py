import colorsys

class Lamp:
    
    def __init__(self,
            numBulbs=3,
            arn=0,
            name='demo',
            angularOffset=0,
            ):

        # Number of Bulbs
        self.numBulbs = numBulbs

        # 0 - Lamp is Circularly Arranged
        # 1 - Lamp is LinearLy Arranged
        self.arn = arn

        # Name or ID of Lamp
        self.name = name

        # 0 - Bulbs turn to White Light when clock is touched
        # 1 - Bulbs turn to prev color(s) when clock is touched, default is white when no prev
        # 2 - Bulbs unaffected by Central Clock input
        self.mainsControlMode = 0

        # Angle In which bulbs are arranged
        self.angularOffset = angularOffset


        self.bulbsCurrentHSV = []
        self.bulbsTargetHSV = []
        for i in range(numBulbs):
            self.bulbsCurrentHSV.append((i*(360/numBulbs)+60, 255, 255))
        self.bulbsTargetHSV = self.bulbsCurrentHSV

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
        return colorsys.hsv_to_rgb(
                self.bulbsCurrentHSV[n][0]/255,
                self.bulbsCurrentHSV[n][1]/255,
                self.bulbsCurrentHSV[n][2]/255)

    def getControlMode(self, n):
        return self.mainsControlMode

    def getBulbsRGB(self):
        tmp = []
        for i in range(self.numBulbs):
            tmp.append(colorsys.hsv_to_rgb(
                self.bulbsCurrentHSV[i][0]/255,
                self.bulbsCurrentHSV[i][1]/255,
                self.bulbsCurrentHSV[i][2]/255))
        return tmp

    def setNumBulbs(self, n):
        if n < 1:
            n = 1
        elif n > 6:
            n = 6
        if n > len(self.bulbsCurrentHSV):
            self.bulbsCurrentHSV.append((32*n, 255, 32*n))
        self.numBulbs = n

    def getNumBulbs(self):
        return self.numBulbs

