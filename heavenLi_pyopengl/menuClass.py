from hliUIutils import UIparam
from hliGLutils import *
from rangeUtils import *
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

        # distance from the center of the element closest to the center
        self.deltaCenter = 0.0

        # Current selected element of the menu (index: list, key: dict, value: range-tuple)
        self.selectedElement = 0.0

        # Which direction, in degrees about the unit circle, the menu slides out
        self.direction = 90.0

        # Whether or not to display the index of the elemen
        self.dispIndex = True

        # Floating cursor for selecting via dragging or flicking (inertial scrolling)
        # One-dimensional value parallel to the menu deployment direction
        self.selectionCursorPosition = 0.0
        self.selectionCursorVelocity = 0.0

        # Number of elements
        self.numElements = 0

        # Timer for updating physics
        self.tPhys = time.time()

    # Watch menu for scroll input
    def watch(self, mmx, mmy, tmx, tmy, tms, tmv):
        #if (tms >= hypot(tmx, tmy)
        pass
        return

    # Set the cursor's velocity magnitude (speed)
    def setCurVel(self, velocity):

        # Input is a single value
        if type(velocity) is float:
            self.selectionCursorVelocity = float(velocity)

        # Input is a tuple velocity vector
        elif type(velocity) is tuple:
            tmvx = velocity[0]*cos(self.direction)
            tmvy = velocity[1]*sin(self.direction)
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
            tmvx = position[0]*cos(self.direction)
            tmvy = position[1]*sin(self.direction)
            self.selectionCursorPosition = hypot(tmvx, tmvy)

        # Input unknown
        else:
            self.selectionCursorPosition = 0.0
        return

    # Set the deployment slide-out direction
    def setDir(self, angle):
        self.direction = float(angle)
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
        return

    # Set animation speed
    def setTimeSlice(self, tDiff):
        self.deployed.setTimeSlice(tDiff)
        return

    # Set whether or not to draw index
    def setIndexDraw(self, doDraw):
        self.dispIndex = bool(doDraw)
        return

    # The actual OpenGL draw call
    def draw(self, stateMach):
        w2h = stateMach['w2h']

        tms = stateMach['UIelements']['testMenu'].getSize()
        mx = stateMach['UIelements']['testMenu'].getTarPosX()
        my = stateMach['UIelements']['testMenu'].getTarPosY()

        drawMenu(
                mx,
                my,
                tms,
                self.direction,
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

