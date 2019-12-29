from hliUIutils import UIparam, UIelement, watchDot, watchBox, watchPolygon
from hliGLutils import *
from rangeUtils import *
from math import degrees, atan, sin, cos, sqrt, radians
import time

# Class definition for a drop/slide out menu
class Menu:

    def __init__(self):

        # Type of drop menu
        self.menuType = 0   # 0: Carousel that loops back on itself
                            # 1: Linear Strip that stops at two ends
                            # 2: Range slider(?)

        # Cursor for animating menu slide-out (0 = closed, 1 = open)
        self.deployed = UIparam()

        # Couple the UIelement to the menu
        self.UIelement = UIelement()

        # distance from the center of the element closest to the center
        self.deltaCenter = 0.0

        # Current selected element of the menu (index: list, key: dict, value: range-tuple)
        self.selectedElement = 0.0

        # Which direction (N, E, S, W), the menu slides out, for input
        self.dir = "E"

        # Which angle, in degrees about the unit circle, the menu slides out, for animation/drawing
        self.angle = 0.0

        # Whether or not to display the index of the elemen
        self.dispIndex = True

        # Floating cursor for selecting via dragging or flicking (inertial scrolling)
        # One-dimensional value parallel to the menu deployment angle
        self.selectionCursorPosition = 0.0
        self.selectionCursorVelocity = 0.0

        # Number of elements
        self.numElements = 0

        # Timer for updating physics
        self.tPhys = time.time()

    # index draw getter
    def getIndexDraw(self):
        return bool(self.dispIndex)

    # Get direction (NESW) angle, for watching input
    def getDirAng(self):
        if self.dir == "E":
            return 0.0
        if self.dir == "N":
            return 90.0
        if self.dir == "W":
            return 180.0
        if self.dir == "S":
            return 270.0

    # Set deployment direction
    def setDir(self, direction):
        if( (direction != "N")
        and (direction != "E")
        and (direction != "S")
        and (direction != "W")
        ):
            self.dir = "E"
        else:
            self.dir = direction

    # Get Menu deployment angle
    def getAng(self):
        return float(self.angle)

    # Watch menu for input
    def watch(self, sm):
        w2h = sm['w2h']

        # Watch menu to toggle open
        tmx = self.UIelement.getTarPosX()*sm['w2h']
        tmy = self.UIelement.getTarPosY()
        tms = self.UIelement.getTarSize()

        if sm['w2h'] < 1.0:
            tms *= sm['w2h']

        if (watchDot(
                tmx, 
                tmy, 
                tms,
                sm['cursorXgl'],
                sm['cursorYgl'],
                sm['w2h'],
                sm['drawInfo']
                )
            and
            sm['mousePressed'] == 0
            ):
            self.toggleOpen()

        # Watch Menu for scroll
        polygon = []
        ang = self.getAng()
        da  = 90.0-degrees(atan((23/4)+float(self.getIndexDraw())))

        rad = sqrt(2)
        polygon.append((rad*cos(radians(ang+45)), rad*sin(radians(ang+45))))   # A
        polygon.append((rad*cos(radians(ang-45)), rad*sin(radians(ang-45))))   # B
        rad = (23/4)+float(self.getIndexDraw())
        polygon.append((rad*cos(radians(ang-da)), rad*sin(radians(ang-da))))     # C
        polygon.append((rad*cos(radians(ang+da)), rad*sin(radians(ang+da))))     # D
        for i in range(len(polygon)):
            tmx = polygon[i][0]
            tmy = polygon[i][1]
            tmx *= self.UIelement.getTarSize()
            tmy *= tms

            polygon2[i] = (tmx,tmy)

        rad *= self.UIelement.getSize()
        if w2h < 1.0:
            tmx *= w2h
        #if sm['w2h'] < 1.0:
            #px *= sm['w2h']
            #qx *= sm['w2h']

        if (self.isOpen()
            and
            (watchPolygon(sm['cursorXgl'], sm['cursorYgl'], polygon, w2h, sm['drawInfo'])
            or
            watchDot(
                rad*cos(radians(ang)), 
                rad*sin(radians(ang)),
                tms,
                sm['cursorXgl'],
                sm['cursorYgl'],
                sm['w2h'],
                sm['drawInfo']
                )
            )):
            print("quack")
            pass

        pass
        return

    # Set the cursor's velocity magnitude (speed)
    def setCurVel(self, velocity):

        # Input is a single value
        if type(velocity) is float:
            self.selectionCursorVelocity = float(velocity)

        # Input is a tuple velocity vector
        elif type(velocity) is tuple:
            tmvx = velocity[0]*cos(self.angle)
            tmvy = velocity[1]*sin(self.angle)
            self.selectionCursorVelocity = hypot(tmvx, tmvy)

        # Input unknown
        else:
            self.selectionCursorVelocity = 0.0
        return

    # Set the selection cursor's position
    def setCurPos(self, position):

        # Input is a single value
        if type(position) is float:
            self.selectionCursorPosition = float(position)

        # Input is a tuple velocity vector
        elif type(position) is tuple:
            tmvx = position[0]*cos(self.angle)
            tmvy = position[1]*sin(self.angle)
            self.selectionCursorPosition = hypot(tmvx, tmvy)

        # Input unknown
        else:
            self.selectionCursorPosition = 0.0
        return

    # Set the deployment slide-out angle
    def setAng(self, angle):
        self.angle = float(angle)
        return

    # Returns True if menu is fully deployed and ready to use
    def isOpen(self):
        if self.deployed.getVal() == 1.0:
            return True
        else:
            return False

    # Opens the menu
    def open(self):
        self.deployed.setTargetVal(1.0)
        return

    # Closes the menu
    def close(self):
        self.deployed.setTargetVal(0.0)
        return

    # Toggles menu
    def toggleOpen(self):
        if (self.deployed.getTar() == 1.0):
            self.deployed.setTargetVal(0.0)
        elif (self.deployed.getTar() == 0.0):
            self.deployed.setTargetVal(1.0)
        return

    # Return value of deployed to be used as a cursor
    def getDeployed(self):
        return self.deployed.getVal()

    # Update parameters n' stuff
    def update(self):

        # Set cursor acceleration + decay rate
        cDrag   = 1.0
        accel   = self.selectionCursorVelocity*cDrag

        # Get time since last update to decouple update rate from framerate
        tDelta  = time.time()-self.tPhys

        # Update Cursor position
        self.selectionCursorPosition += 2.0*self.selectionCursorVelocity*tDelta
        self.selectionCursorPosition += accel*pow(tDelta, 2.0)

        # Decay Cursor velocity over time to simulate drag
        self.selectionCursorVelocity += accel*tDelta
        if self.selectionCursorPosition <= 0.01:
            self.selectionCursorVelocity = 0.0

        self.deployed.updateVal()
        self.UIelement.updateParams()
        return

    # Set animation speed
    def setTimeSlice(self, tDiff):
        self.deployed.setTimeSlice(tDiff)
        self.UIelement.setTimeSlice(tDiff)
        return

    # Set whether or not to draw index
    def setIndexDraw(self, doDraw):
        self.dispIndex = bool(doDraw)
        return

    # The actual OpenGL draw call
    def draw(self, stateMach):
        w2h = stateMach['w2h']

        tms = self.UIelement.getSize()
        mx = self.UIelement.getTarPosX()
        my = self.UIelement.getTarPosY()

        drawMenu(
                mx,
                my,
                tms,
                self.angle,
                self.deployed.getVal(),
                self.selectedElement,
                self.numElements,
                self.menuType,
                self.dispIndex,
                w2h,
                stateMach['faceColor'],
                stateMach['detailColor']
                )

        pass
        return

