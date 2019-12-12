from hliUIutils import UIparam
from hliGLutils import *
from rangeUtils import *

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
        self.selectedElement = None

        # True iff deployed == 1.0
        self.isOpen = False

        # Which direction, in degrees about the unit circle, the menu slides out
        self.direction = 90.0

        # Whether or not to display the index of the elemen
        self.dispIndex = True

        # Floating cursor for selecting via dragging or flicking (inertial scrolling)
        self.selectionCursor = 0.0

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
        self.deployed.updateVal()
        return

    # Set animation speed
    def setTimeSlice(self, tDiff):
        self.deployed.setTimeSlice(tDiff)
        return

    def draw(self, stateMach):
        w2h = stateMach['w2h']

        tms = stateMach['UIelements']['testMenu'].getSize()
        mx = stateMach['UIelements']['testMenu'].getPosX()
        my = stateMach['UIelements']['testMenu'].getPosY()

        drawMenu(
                mx,
                my,
                tms,
                self.direction,
                self.deployed.getVal(),
                w2h,
                stateMach['faceColor'],
                stateMach['detailColor']
                )

        pass
        return

