# Classes and utilities for UI animation

from lampClass import *
from drawArn import *

# Abstracts Lamp icon drawing
def drawIcon(ix, iy, scale, color, w2h, Lamp):
    if (Lamp.getArn() == 0):
        drawIconCircle(ix, iy, 
                scale, 
                Lamp.metaLampLevel+2,
                color,
                Lamp.getNumBulbs(), 
                Lamp.getAngle(), 
                w2h, 
                Lamp.getBulbsCurrentRGB())

    if (Lamp.getArn() == 1):
        drawIconLinear(ix, iy, 
                scale, 
                Lamp.metaLampLevel+2,
                color,
                Lamp.getNumBulbs(), 
                Lamp.getAngle(), 
                w2h, 
                Lamp.getBulbsCurrentRGB())
    return

# This class helps UI elements (buttons) 
# manage their own animation with needing to be 
# precisely choreographed
class UIelement:
    # heavenLi UI elements have five base parameters to describe their location:
    # X & Y position coordinates
    # X & Y scaling
    # rotation
    # additional parameters for more niche uses may be added after the fact
    def __init__(self):
        self.params = {}
        self.params["coordX"] = UIparam()
        self.params["coordY"] = UIparam()
        self.params["scaleX"] = UIparam()
        self.params["scaleY"] = UIparam()
        self.params["rotAng"] = UIparam()
        self.tDiff = 1.0

    # Set time-slice for animation speed
    def setTimeSlice(self, tDiff):
        self.tDiff = tDiff
        for i in self.params:
            self.params[i].setTimeSlice(self.tDiff)
        return

    # General-purpose setter
    def setTarget(self, key, val):
        self.params[key].setTargetVal(val)
        return

    # General-purpose getter
    def getValue(self, key):
        return self.params[key].getVal()

    # Update all parameters
    def updateParams(self):
        for i in self.params:
            self.params[i].updateVal()
        return

    # Add another parameter
    def addParam(self, key):
        self.params[key] = UIelement
        # set animation speed pre-emptively
        self.params[key].setTimeSlice(self.tDiff)
        return

    # Used to determine if UI element should invoke OpenGL draw-call
    def isVisible(self):
        if (abs(self.params["scaleX"]) <= 0.0001):
            return False
        elif (abs(self.params["scaleY"]) <= 0.0001):
            return False
        else:
            return True

    # Useful overloads for base parameters
    def setTarPosX(self, val):
        self.params["coordX"].setTargetVal(val)
        return

    def setTarPosY(self, val):
        self.params["coordY"].setTargetVal(val)
        return

    def setTarSizeX(self, val):
        self.params["scaleX"].setTargetVal(val)
        return

    def setTarSizeY(self, val):
        self.params["scaleY"].setTargetVal(val)
        return

    def setTarAngle(self, val):
        self.params["rotAng"].setTargetVal(val)
        return

    def setAccel(self, value):
        for i in self.params:
            self.params[i].setAccel(value)
        return

    def setTarSize(self, val):
        self.params["scaleX"].setTargetVal(val)
        self.params["scaleY"].setTargetVal(val)
        return

    def getSize(self):
        return self.params["scaleX"].getVal()

    def getPosX(self):
        return self.params["coordX"].getVal()

    def getPosY(self):
        return self.params["coordY"].getVal()

    def getSizeX(self):
        return self.params["scaleX"].getVal()

    def getSizeY(self):
        return self.params["scaleY"].getVal()

    def getAngle(self):
        return self.params["rotAng"].getVal()

# helper class for UIelement
class UIparam:
    def __init__(self):
        self.currentVal = 0.0
        self.previousVal = 0.0
        self.targetVal = 0.0
        self.targetReached = True
        self.tDiff = 1.0
        self.accel = 1.0
        self.curveBias = 0.5
        self.prevDeltaSign = None
        return

    # Used for tuning animation speed
    def setTimeSlice(self, tDiff):
        self.tDiff = tDiff
        return

    # Set Target Value for the Current Value to transition to
    def setTargetVal(self, target):
        self.previousVal = self.currentVal
        self.currentVal = self.targetVal
        self.targetVal = target
        self.targetReached = False

        delta = self.targetVal - self.currentVal

        # Avoid divide by zero
        if (delta == 0.0 or delta == -0.0):
            deltaSign = 1.0
        else:
            deltaSign = delta/abs(delta)
        self.prevDeltaSign = deltaSign
        return

    # True Value of the parameter
    def getVal(self):
        return self.currentVal

    # Used for making real-time changes, corrections.
    # Use sparringly
    def setVal(self, value):
        self.currentVal = value
        return

    # Use to temporarily tune animation speed (multiplies tDiff)
    # Resets to 1.0 (default) after animation completes
    def setAccel(self, value):
        self.accel = value
        return

    def updateVal(self):
        delta = self.targetVal - self.currentVal

        # Avoid divide by zero
        if (delta == 0.0 or delta == -0.0):
            deltaSign = 1.0
        else:
            deltaSign = delta/abs(delta)

        if (abs(delta) > self.tDiff*0.01 and self.prevDeltaSign == deltaSign):
            change = (self.currentVal-self.previousVal)/(self.targetVal-self.previousVal)
            if (abs(change) <= 0.1):
                change = 0.1
            elif(abs(change) > 0.90):
                change = self.targetVal - self.currentVal

            if (delta <= 0.0):
                self.currentVal -= float(self.accel*self.tDiff*abs(change))
            if (delta > 0.0):
                self.currentVal += float(self.accel*self.tDiff*abs(change))
        else:
            self.currentVal = self.targetVal
            self.targetReached = True
            self.accel = 1.0
            self.prevDeltaSign = None
        return
