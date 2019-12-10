# Classes and utilities for UI animation

from lampClass import *
import pytweening
print("Loading hliGLutils...")
from hliGLutils import *
print("Done!")

# function for printing statemachine information
def drawInfo(stateMach):
    infoStr = "//~~~~~~~~~~~~~~~infoo debugeroo, eletric boogaloo~~~~~~~~~~~~~~~\\\\"
    infoStr += "\nFPS: " + str(int(stateMach['fps']))
    infoStr += "\nResolution: " + str(stateMach['windowDimW'])+", "+str(stateMach['windowDimH'])
    infoStr += "\nwidth to height: " + str(stateMach['w2h'])
    infoStr += "\nCursor: " + str(stateMach['cursorX']) + ', ' + str(stateMach['cursorY'])
    infoStr += "\nInput State: " + str(stateMach['currentState'])
    infoStr += "\nMouse Button: " + str(stateMach['mouseButton'])
    infoStr += "\nCursor Velocity Raw, Polar Ang: " + str(stateMach['cursorVelocity'][0])[:5] + " deg"
    infoStr += "\nCursor Velocity Raw, Polar Mag: " + str(stateMach['cursorVelocity'][1])[:5] + " px/s"
    infoStr += "\nCursor Velocity Smoothed, Polar Ang: " + str(stateMach['cursorVelSmoothPol'][0])[:5] + " deg"
    infoStr += "\nBall Position: " + str(stateMach['BallPosition'])
    infoStr += "\nBall Velocity: " + str(stateMach['BallVelocity'])
    tmc = ( stateMach['faceColor'][0], 
            stateMach['faceColor'][1], 
            stateMach['faceColor'][2], 
            stateMach['faceColor'][3]/2)
    drawText(infoStr, 0.0, -1.0, 0.85, 0.25, 0.25, stateMach['w2h'], stateMach['detailColor'], tmc)

# This class helps the management and animation of colors
class UIcolor:
    def __init__(self):
        self.params = {}
        self.params["R"] = UIparam()
        self.params["G"] = UIparam()
        self.params["B"] = UIparam()
        self.params["A"] = UIparam()
        self.tDiff = 1.0
        
    def setAccel(self, value):
        for i in self.params:
            self.params[i].setAccel(value)
        return

    # Set color to fade to
    def setTargetColor(self, color):
        self.params["R"].setTargetVal(color[0])
        self.params["G"].setTargetVal(color[1])
        self.params["B"].setTargetVal(color[2])
        self.params["A"].setTargetVal(color[3])
        return

    # Return 4-tuple of RGBA
    def getColor(self):
        return tuple([
            self.params["R"].getVal(), 
            self.params["G"].getVal(), 
            self.params["B"].getVal(), 
            self.params["A"].getVal()])

    # Set time-slice for animation speed
    def setTimeSlice(self, tDiff):
        if (self.tDiff != tDiff):
            self.tDiff = tDiff
            for i in self.params:
                self.params[i].setTimeSlice(self.tDiff)
        return

    # Update all parameters
    def updateParams(self):
        for i in self.params:
            self.params[i].updateVal()
        return

    # Update all parameters
    def updateVal(self):
        for i in self.params:
            self.params[i].updateVal()
        return

    # General-purpose setter
    def setTarget(self, key, val):
        self.params[key].setTargetVal(val)
        return

    # General-purpose getter
    def getTarget(self, key):
        return self.params[key].getTar()

    # General-purpose setter
    def setValue(self, key, val):
        self.params[key].setValue(val)
        return

    # General-purpose getter
    def getValue(self, key):
        return self.params[key].getVal()


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
        self.params["facCol"] = UIcolor()
        self.params["detCol"] = UIcolor()
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
    def getTarget(self, key):
        return self.params[key].getTar()

    # General-purpose setter
    def setValue(self, key, val):
        self.params[key].setValue(val)
        return

    # General-purpose getter
    def getValue(self, key):
        return self.params[key].getVal()

    # Manually set size, use sparringly
    def setSize(self, val):
        self.params["scaleX"].setValue(val)
        self.params["scaleY"].setValue(val)
        return

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
        if (self.params["scaleX"] == 0.0):
            return False
        elif (self.params["scaleY"] == 0.0):
            return False
        else:
            return True

    def setTargetFaceColor(self, targetColor):
        self.params['facCol'].setTargetColor(targetColor)
        return

    def setTargetDetailColor(self, targetColor):
        self.params['detCol'].setTargetColor(targetColor)
        return

    # Get element face color
    def getFaceColor(self):
        return self.params['facCol'].getColor()

    # Get element accent color
    def getDetailColor(self):
        return self.params['detCol'].getColor()

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

    def getTarSize(self):
        return self.params["scaleX"].getTar()

    def getSize(self):
        return self.params["scaleX"].getVal()

    def getPosX(self):
        return self.params["coordX"].getVal()

    def getPosY(self):
        return self.params["coordY"].getVal()

    def getTarPosX(self):
        return self.params["coordX"].getTar()

    def getTarPosY(self):
        return self.params["coordY"].getTar()

    def getSizeX(self):
        return self.params["scaleX"].getVal()

    def getSizeY(self):
        return self.params["scaleY"].getVal()

    def getAngle(self):
        return self.params["rotAng"].getVal()

# helper class for UIelement
class UIparam:
    def __init__(self):
        self.currentVal     = 0.0   # True value of the parameter
        self.previousVal    = 0.0   # Parameter value before a new target value was set
        self.targetVal      = 0.0   # Target Value that current value will 'drift' to
        self.targetReached  = True
        self.tDiff          = 1.0   # Time-slice for adjusting animation speed
        self.accel          = 1.0   # temporarily adjust tDiff until animation finishes
        self.curveBias      = 0.5
        self.prevDeltaSign  = None
        self.cursor         = 1.0
        self.curve          = "Default"
        return

    # Used for tuning animation speed
    def setTimeSlice(self, tDiff):
        self.tDiff = tDiff
        return

    # Set Target Value for the Current Value to transition to
    def setTargetVal(self, target):
        self.cursor         = 0.0
        self.previousVal    = self.currentVal
        self.accel          = 1.0
        self.targetVal      = target
        self.targetReached  = False

        delta = self.targetVal - self.currentVal

        # Avoid divide by zero
        if (delta == 0.0 or delta == -0.0):
            deltaSign = 1.0
        else:
            deltaSign = delta/abs(delta)
        self.prevDeltaSign = deltaSign
        return

    # Target Value Getter
    def getTar(self):
        return self.targetVal

    # True Value of the parameter
    def getVal(self):
        return self.currentVal

    # Used for making real-time changes, corrections.
    # Use sparringly
    def setValue(self, value):
        self.currentVal = value
        return

    # Use to temporarily tune animation speed (multiplies tDiff)
    # Resets to 1.0 (default) after animation completes
    def setAccel(self, value):
        self.accel = value
        return

    # Choose which animation curve to use
    # For a good refernce on different animation curves, visit:
    # https://easings.net/
    def setCurve(self, curve):
        self.curve = curve
        return

    def updateVal(self):
        if (self.cursor < 1.0):
            if (self.cursor < 0.0):
                self.cursor = 0.0
            delta           = self.targetVal - self.previousVal
            self.currentVal = delta*self._curve(self.cursor) + self.previousVal
            self.cursor     += self.tDiff*self.accel*0.25
        else:
            self.currentVal = self.targetVal
            self.cursor     = 1.0

        return

    def _curve(self, x):
        if self.curve is "Default":
            return pytweening.easeOutQuint(x)

        elif self.curve is "easeInQuad":
            return pytweening.easeInQuad(x)

        elif self.curve is "easeOutQuade":
            return pytweening.easeOutQuade(x)

        elif self.curve is "easeInOutQuad":
            return pytweening.easeInOutQuad(x)

        elif self.curve is "easeInQubic":
            return pytweening.easeInQubic(x)

        elif self.curve is "easeOutQubic":
            return pytweening.easeOutQubic(x)

        elif self.curve is "easeInOutQubic":
            return pytweening.easeInOutQubic(x)

        elif self.curve is "easeInQuart":
            return pytweening.easeInQuart(x)

        elif self.curve is "easeOutQuart":
            return pytweening.easeOutQuart(x)

        elif self.curve is "easeInOutQuart":
            return pytweening.easeInOutQuart(x)

        elif self.curve is "easeInQuint":
            return pytweening.easeInQuint(x)

        elif self.curve is "easeOutQuint":
            return pytweening.easeOutQuint(x)

        elif self.curve is "easeInOutQuint":
            return pytweening.easeInOutQuint(x)

        elif self.curve is "easeInSine":
            return pytweening.easeInSine(x)

        elif self.curve is "easeOutSine":
            return pytweening.easeOutSine(x)

        elif self.curve is "easeInOutSine":
            return pytweening.easeInOutSine(x)

        elif self.curve is "easeInExpo":
            return pytweening.easeInExpo(x)

        elif self.curve is "easeOutExpo":
            return pytweening.easeOutExpo(x)

        elif self.curve is "easeInOutExpo":
            return pytweening.easeInOutExpo(x)

        elif self.curve is "easeInCirc":
            return pytweening.easeInCirc(x)

        elif self.curve is "easeOutCirc":
            return pytweening.easeOutCirc(x)

        elif self.curve is "easeInOutCirc":
            return pytweening.easeInOutCirc(x)

        elif self.curve is "easeInElastic":
            return pytweening.easeInElastic(x)

        elif self.curve is "easeOutElastic":
            return pytweening.easeOutElastic(x)

        elif self.curve is "easeInOutElastic":
            return pytweening.easeInOutElastic(x)

        elif self.curve is "easeInBack":
            return pytweening.easeInBack(x)

        elif self.curve is "easeOutBack":
            return pytweening.easeOutBack(x)

        elif self.curve is "easeInOutBack":
            return pytweening.easeInOutBack(x)

        elif self.curve is "easeInBounce":
            return pytweening.easeInBounce(x)

        elif self.curve is "easeOutBounce":
            return pytweening.easeOutBounce(x)

        elif self.curve is "easeInOutBounce":
            return pytweening.easeInOutBounce(x)

        else:
            return pytweening.easeOutCirc(x)
