from hliUIutils import UIparam, UIelement, watchDot, watchBox, watchPolygon
from hliGLutils import *
from rangeUtils import *
from math import degrees, atan, sin, cos, sqrt, radians, hypot
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

        # Used to animate scrolling snapping to a target
        self.scrollSnap = UIparam()

        # Couple the UIelement to the menu
        self.UIelement = UIelement()

        # distance from the center of the element closest to the center
        self.deltaCenter = 0.0

        # Current selected element of the menu (index: list, key: dict, value: range-tuple)
        self.selectedElement = 0.0

        # Index of the previously selected element when changing elements / scrolling
        self.prevSelectionIndex = 0

        # Which angle, in degrees about the unit circle, the menu slides out, for animation/drawing
        self.angle = 0.0

        # Whether or not to display the index of the elemen
        self.dispIndex = True

        # Floating cursor for selecting via dragging or flicking (inertial scrolling)
        # One-dimensional value parallel to the menu deployment angle
        self.selectionCursorPosition = 0.0
        self.selectionCursorVelocity = 0.0

        # Used for tracking scroll input when cursor has moved off the menu
        self.isTrackingScroll = False

        # Number of elements
        self.numElements = 0

        # Timer for updating physics
        self.tPhys = time.time()


    # index draw getter
    def getIndexDraw(self):
        return bool(self.dispIndex)

    # Get Menu deployment angle
    def getAng(self):
        return float(self.angle)

    # Watch menu for input
    def watch(self, sm):
        # Watch menu to toggle open
        w2h = sm['w2h']
        tmx = self.UIelement.getTarPosX()*w2h
        tmy = self.UIelement.getTarPosY()
        tms = self.UIelement.getTarSize()

        if w2h < 1.0:
            tms *= w2h

        if (watchDot(
                tmx, 
                tmy, 
                tms,
                sm['cursorXgl'],
                sm['cursorYgl'],
                w2h,
                sm['drawInfo']
                )
            and
            not self.isTrackingScroll
            ):

            if (sm['mousePressed'] == 0):
                self.toggleOpen()

            return True

        # Watch Menu Body for scroll / select
        polygon = []
        ang = self.getAng()
        da  = 90.0-degrees(atan((23/4)+float(self.getIndexDraw())))

        ofx     = self.UIelement.getTarPosX()
        ofy     = self.UIelement.getTarPosY()

        # Proximal Box Corners
        radius = sqrt(2)
        radx = radius*self.UIelement.getTarSize()
        rady = radius*tms

        tmr = radians(ang+45)
        polygon.append( (
            ofx+cos(tmr)*radx, 
            ofy+sin(tmr)*rady
            ) )

        tmr = radians(ang-45)
        polygon.append( (
            ofx+cos(tmr)*radx, 
            ofy+sin(tmr)*rady
            ) )

        # Distil Box Corners
        radius  = (23/4)+float(self.getIndexDraw())
        radx    = radius*self.UIelement.getTarSize()
        rady    = radius*tms

        tmr = radians(ang-da)
        polygon.append( (
            ofx+cos(tmr)*radx, 
            ofy+sin(tmr)*rady
            ) )

        tmr = radians(ang+da)
        polygon.append( (
            ofx+cos(tmr)*radx, 
            ofy+sin(tmr)*rady
            ) )


        # Aspect correct radius for watchdot
        radius *= self.UIelement.getSize()
        if w2h < 1.0:
            ofy = ofy+radius*sin(radians(ang))*w2h
        else:
            ofy = (ofy+radius*sin(radians(ang)))

        if (    self.isOpen()
                and
                (
                    watchPolygon(
                        sm['cursorXgl'], 
                        sm['cursorYgl'], 
                        polygon, 
                        w2h, 
                        sm['drawInfo']
                        )
                    or
                    watchDot(
                        ofx+radius*cos(radians(ang)), 
                        ofy,
                        tms,
                        sm['cursorXgl'],
                        sm['cursorYgl'],
                        w2h,
                        sm['drawInfo']
                        )
                    or
                    self.isTrackingScroll
                )
            ):

            tmx = mapRanges(sm['cursorX'], 0, sm['windowDimW'], -w2h, w2h)
            tmy = mapRanges(sm['cursorY'], 0, sm['windowDimH'], 1.0, -1.0)
            tmx /= 1.75*self.UIelement.getTarSize()
            tmy /= 1.75*self.UIelement.getTarSize()
            tmx += self.UIelement.getPosX()*self.UIelement.getSize()
            tmy += self.UIelement.getPosY()*self.UIelement.getSize()
            radAng = radians(self.angle)
            self.selectionCursorPosition = tmx*cos(radAng) + tmy*sin(radAng)
            self.isTrackingScroll = True
            
            pass
            if (sm['mouseReleased'] >= 0):
                self.isTrackingScroll = False
                self.selectionCursorVelocity = (
                    (2.0*sm['cursorVelSmoothed'][0]*cos(self.angle)) + 
                    (2.0*sm['cursorVelSmoothed'][1]*sin(self.angle))
                )

            if (sm['mousePressed'] >= 0):
                self.prevSelectionIndex = self.selectionCursorPosition

            return True

        pass
        return False

    # Set menu number of elements
    def setNumElements(self, numElements):

        # Sanity Check
        if (numElements < 0):
            self.numElements = 0

        # Sanity Check
        elif (type(numElements) != int):

            # Sanity Check
            if (type(numElements) == float and numElements > 0.0):
                self.numElements = int(numElements)
            else:
                self.numElements = 0.0
        else:
            self.numElements = numElements

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
    def update(self, sm):

        # Set cursor acceleration + decay rate
        cDrag   = 1.0
        accel   = -self.selectionCursorVelocity*cDrag

        # Get time since last update to decouple update rate from framerate
        tDelta  = time.time()-self.tPhys

        # Update Cursor position
        if (sm['currentState'] == 1):
            tmp = self.selectionCursorPosition + 2.0*self.selectionCursorVelocity*tDelta + accel*pow(tDelta, 2.0)
            self.selectionCursorPosition = tmp

            # Decay cursor velocity
            tmp = self.selectionCursorVelocity + accel*tDelta

            # Don't let cursor take forever to coast to a stop
            if (abs(tmp) <= 0.10):
                self.selectionCursorVelocity = 0.0

                # Snap cursor to nearest whole number, animate
                tmn = normalizeCursor(self.prevSelectionIndex, self.selectionCursorPosition)
                #print(tmn, self.prevSelectionIndex, self.selectionCursorPosition)
                while tmn < 0.0:
                    tmn += 1.0
                if (    tmn >= 0.001
                        and
                        tmn <= 0.999
                        and
                        not self.scrollSnap.isAnimating()
                        ):
                    self.scrollSnap.setValue(self.selectionCursorPosition)
                    self.scrollSnap.setTargetVal(round(self.selectionCursorPosition))
                else:
                    self.selectionCursorPosition = self.scrollSnap.getVal()
                    pass
            else:
                # Update Cursor velocity
                self.selectionCursorVelocity = tmp

        self.tPhys = time.time()

        self.scrollSnap.updateVal()
        self.deployed.updateVal()
        self.UIelement.updateParams()
        return

    # Set animation speed
    def setTimeSlice(self, tDiff):
        self.scrollSnap.setTimeSlice(tDiff)
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
        sc = normalizeCursor(self.prevSelectionIndex, self.selectionCursorPosition)
        if sc < 0.0:
            sc += 1.0

        drawMenu(
                mx,
                my,
                tms,
                self.angle,
                self.deployed.getVal(),
                self.selectedElement,
                sc,
                self.numElements,
                self.menuType,
                self.dispIndex,
                w2h,
                stateMach['faceColor'],
                stateMach['detailColor']
                )

        tmx = self.selectionCursorPosition*cos(self.angle)
        tmy = self.selectionCursorPosition*sin(self.angle)

        pass
        return

