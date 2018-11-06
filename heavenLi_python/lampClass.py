import colorsys

class Lamp:
    
    def __init__(self,
            numBulbs=3,
            arn=0,
            name='demo',
            angularOffset=0,
            ):
        self.numBulbs = numBulbs
        self.arn = arn
        self.name = name
        self.angularOffset = angularOffset
        self.bulbsCurrentHSV = []
        self.bulbsTargetHSV = []
        for i in range(numBulbs):
            self.bulbsCurrentHSV.append((i*(360/numBulbs)+60, 255, 255))
        self.bulbsTargetHSV = self.bulbsCurrentHSV

    def getBulbRGB(self, n):
        return colorsys.hsv_to_rgb(
                self.bulbsCurrentHSV[n][0]/255,
                self.bulbsCurrentHSV[n][1]/255,
                self.bulbsCurrentHSV[n][2]/255)

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
        print(self.numBulbs)

    def getNumBulbs(self):
        return self.numBulbs

    def setAngle(self, ang):
        if ang >= 360:
            ang -= 360
        if ang <= 0:
            ang += 360
        self.angularOffset = ang

    def getAngle(self):
        return self.angularOffset

