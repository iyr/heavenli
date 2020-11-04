# Classes and utilities for UI animation
# Additional imports may be appended at end of file

from hliImports import *

# function for printing statemachine information
def drawInfo(stateMach):
    w2h = stateMach['w2h']
    infoStr = "//~~~~~~~~~~~~~~~statemachine shenanigans~~~~~~~~~~~~~~~\\\\"
    infoStr += "\nglutMainLoopFrequency (Hz): " + str(int(stateMach['glutFreq']))
    infoStr += "\nUI Engine FPS (Hz): " + str(int(stateMach['SMfreq']))
    infoStr += "\nOpenGL FPS (Hz): " + str(int(stateMach['GLfreq']))
    infoStr += "\nResolution: " + str(stateMach['windowDimW'])+", "+str(stateMach['windowDimH'])
    infoStr += "\nwidth to height: " + str(w2h)
    infoStr += "\nwindow position: " + str(stateMach['windowPosX'])+", "+str(stateMach['windowPosY'])
    infoStr += "\nCursor: " + str(stateMach['cursorX']) + ', ' + str(stateMach['cursorY'])
    #tmx = mapRanges(stateMach['cursorX'], 0, stateMach['windowDimW'], -w2h, w2h)
    #tmy = mapRanges(stateMach['cursorY'], 0, stateMach['windowDimH'], 1.0, -1.0)
    #infoStr += "\nCursor (GL): " + str(tmx) + ', ' + str(tmy)
    infoStr += "\nCursor (GL): " + str(stateMach['cursorXgl']) + ', ' + str(stateMach['cursorYgl'])
    infoStr += "\nCursor (Desktop): " + str(stateMach['pynputMouse'].position[0]) + ', ' + str(stateMach['pynputMouse'].position[1])
    infoStr += "\nMouse Button Input State: " + str(stateMach['currentMouseButtonState'])
    infoStr += "\nMouse Button: " + str(stateMach['mouseButton'])
    #infoStr += "\nCursor Velocity Raw, Polar Ang: " + str(stateMach['cursorVelocity'][0]) + " deg"
    #infoStr += "\nCursor Velocity Raw, Polar Mag: " + str(stateMach['cursorVelocity'][1]) + " px/s"
    #infoStr += "\nCursor Velocity Smoothed, Polar Ang: " + str(stateMach['cursorVelSmoothPol'][0]) + " deg"
    #infoStr += "\nCursor Velocity Smoothed, Cart X: " + str(stateMach['cursorVelSmoothed'][0])
    #infoStr += "\nCursor Velocity Smoothed, Cart Y: " + str(stateMach['cursorVelSmoothed'][1])
    #infoStr += "\nBall Position: " + str(stateMach['BallPosition'])
    #infoStr += "\nBall Velocity: " + str(stateMach['BallVelocity'])
    infoStr += "\ntextBaseScalar: " + str(stateMach['textBaseScalar'])
    infoStr += "\ntextDPIscalar: " + str(stateMach['textDPIscalar'])
    infoStr += "\ntextGlyphRes: " + str(stateMach['textGlyphRes'])
    tmc = ( stateMach['faceColor'][0], 
            stateMach['faceColor'][1], 
            stateMach['faceColor'][2], 
            stateMach['faceColor'][3]/2)
    drawText(infoStr, 0.0, 0.0, -1.0, 0.85, 0.25*stateMach['textDPIscalar']*stateMach['textBaseScalar'], 0.25*stateMach['textDPIscalar']*stateMach['textBaseScalar'], stateMach['w2h'], stateMach['detailColor'], tmc)
    #drawText(infoStr, 0.0, 0.0, -1.0, 0.85, 0.25*stateMach['textDPIscalar'], 0.25*stateMach['textDPIscalar'], stateMach['w2h'], stateMach['detailColor'], tmc)

# This class helps regulate how often functions are run
# for efficiency or performance
class UIrateRegulator:
    def __init__(self):
        self.frequency = UIparam()
        self.minimizedFreq = 1.0
        self.backgroundFreq = 2.0
        self.foregroundFreq = 3.0

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

def runScript(filepath):
    if (os.path.isfile(filepath)):
        return exec('file=open("'+filepath+'","r")\rexec(file.read())\nfile.close\r',globals())
    elif (os.path.isdir(filepath)):
        for script in os.scandir(filepath):
            if (script.path.endswith(".py") and script.is_file()):
                runScript(script.path)

#runScript("./hliUIutils/drawImage.py")
runScript("./hliUIutils")

#scriptFilepath = "./hliUIutils/drawImage.py"
#exec('file = open(scriptFilepath, "r")\rexec(file.read())\nfile.close\r')

from menuClass import *
