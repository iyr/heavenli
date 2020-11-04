
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

        # Used to manually control visibility
        self.isVis = True

    # Set time-slice for animation speed
    def setTimeSlice(self, tDiff):
        self.tDiff = tDiff
        for i in self.params:
            self.params[i].setTimeSlice(self.tDiff)
        return

    # General-purpose setter
    def setParamTarget(self, key, val):
        self.params[key].setTargetVal(val)
        return

    # General-purpose getter
    def getParamTarget(self, key):
        return self.params[key].getTar()

    # General-purpose setter
    def setParamValue(self, key, val):
        self.params[key].setValue(val)
        return

    # General-purpose getter
    def getParamValue(self, key):
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
        elif (self.isVis == False):
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
    def setTarXYS(self, vals):
        self.params["coordX"].setTargetVal(vals[0])
        self.params["coordY"].setTargetVal(vals[1])
        self.params["scaleX"].setTargetVal(vals[2])
        self.params["scaleY"].setTargetVal(vals[2])
        return

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
