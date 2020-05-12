from hliUIutils import UIparam, UIelement, watchDot, watchBox, watchPolygon
from hliGLutils import *
from rangeUtils import *
from math import degrees, atan, sin, cos, sqrt, radians, hypot
import time

# Class definition for a drop/slide out menu
class Menu:

    def __init__(self):

        # Type of drop menu
        self.menuLayout = 0   # 0: Carousel that loops back on itself
                            # 1: Linear Strip that stops at two ends
                            # 2: Range slider(?)

        # Cursor for animating menu slide-out (0 = closed, 1 = open)
        self.deployed = UIparam()

        # list of 3-tuples contain posx, posy, and scale
        self.elementCoords = []

        # Used to animate scrolling snapping to a target
        self.scrollSnap = UIparam()

        # Couple the UIelement to the menu
        self.UIelement = UIelement()

        # Menu layout: 
        # 0=carousel w/ rollover, 
        # 1=linear strip w/ terminals, 
        # 2=value slider w/ min/max
        self.menuLayout = 0

        # Current selected element of the menu (index: list, key: dict, value: range-tuple)
        self.selectedElement = 0

        # index of the previously selected element
        self.prevSelectedElement = 0

        # Index of the previously selected element when changing elements / scrolling
        self.prevSelectedElement = 0

        # Which angle, in degrees about the unit circle, the menu slides out, for animation/drawing
        self.angle = 0.0

        # Whether or not to display the index of the element
        self.dispIndex = True

        # Floating cursor for selecting via dragging or flicking (inertial scrolling)
        # One-dimensional value parallel to the menu deployment angle
        self.selectionCursorPosition = 0.0
        self.selectionCursorVelocity = 0.0

        # previous position of the cursor, for animating
        self.prevSelectionCursorPosition = 0

        # Used for tracking relative delta for moving selectionCursor
        self.mouseCursorAtPressX = 0.0
        self.mouseCursorAtPressY = 0.0

        # index of the element when input is first registered
        self.clickedElement = None

        # Used for tracking scroll input when cursor has moved off the menu
        self.isTrackingScroll = False

        # Number of elements
        self.numElements = 0

        # Timer for updating physics
        self.tPhys = time.time()

    # change menu layout
    def setLayout(self, layout):
        if layout == 0:
            self.menuLayout = 0
        elif layout == 1:
            #if (self.selectionCursorPosition == 0.0):
                #self.selectionCursorPosition = 1.0
            self.menuLayout = 1
        elif layout == 1:
            self.menuLayout = 2
        else:
            self.menuLayout = 1

        return

    # helper function for getting the indices of the elements strattling the selected element
    def cursor2indices(self, diff):
        indices = []

        tmc = self.delimitValue(self.selectionCursorPosition)
        #tmc = constrain(self.selectionCursorPosition, 1.0, self.numElements-2.0)
        if (self.menuLayout == 0):
            for i in range(
                    floor(tmc - diff),
                    ceil(tmc + diff)+1
                    ):
                indices.append(rollover(i, self.numElements))
            return indices
        elif (self.menuLayout == 1):
            if (tmc == 0.0 or tmc <= 1.0):
                return [0, 1, 2]
            elif (tmc >= self.numElements-2.0 and tmc <= self.numElements-1.0):
                return [self.numElements-3, self.numElements-2, self.numElements-1]
            else:
                for i in range(
                        floor(tmc - diff),
                        ceil(tmc + diff)+1
                        ):
                    indices.append(rollover(i, self.numElements))
                return indices

    # Return a list of tuples containing the index, posX, posY, and scale of visible elements
    def getElements(self):
        tml = self.cursor2indices(1.0);
        tme = []

        for i in range(len(tml)):
            tme.append(
                        (   
                        tml[len(tml)-1-i],
                        self.elementCoords[i][0],
                        self.elementCoords[i][1],
                        self.elementCoords[i][2]
                        )
                    )

        return tme

    # restrict value range based on menu type, convenice function
    def delimitValue(self, value):
        if (self.menuLayout == 0):
            return value
        elif (self.menuLayout == 1):
            return constrain(value, 0.0, self.numElements-1.0)

    # Simplify mouse wheel scrolling
    def scrollButton(self, button):
        self.selectionCursorVelocity = 0.0
        # Mouse wheel scroll up
        if (button == 3):
            self.scrollSnap.setTargetVal(self.delimitValue(self.scrollSnap.getTar()+1.0))

        # Mouse wheel scroll down
        if (button == 4):
            self.scrollSnap.setTargetVal(self.delimitValue(self.scrollSnap.getTar()-1.0))

    # get index for the currently selected
    def getSelection(self):
        #return rollover(round(self.selectionCursorPosition), self.numElements)
        return self.selectedElement

    # index draw getter
    def getIndexDraw(self):
        return bool(self.dispIndex)

    # Get number of elements menu is managing
    def getNumElements(self):
        return self.numElements

    # Get Menu deployment angle
    def getAng(self):
        return float(self.angle)

    # Watch menu for input
    def watch(self, sm):

        # Watch menu core to toggle open
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

        # Convenience Variables
        ang     = self.getAng()
        ofx     = self.UIelement.getTarPosX()
        ofy     = self.UIelement.getTarPosY()

        # Watch increment/decrement buttons for input
        tmx0    = 5.75*self.UIelement.getTarSize()
        tmy0    = 5.75*self.UIelement.getTarSize()
        tmx2    = 2.0*self.UIelement.getTarSize()
        tmy2    = 2.0*self.UIelement.getTarSize()
        thr     = 0.1*self.UIelement.getSize()
        tmes0   = 0.15
        tmes2   = tmes0

        if w2h < 1.0:
            tmy0 *= w2h
            tmes0 *= w2h
            tmy2 *= w2h
            tmes2 *= w2h
        if (    self.isOpen()
                and
                not self.isTrackingScroll
                and
                watchDot(
                    ofx + cos(radians(ang))*tmx0,
                    ofy + sin(radians(ang))*tmy0,
                    tmes0,
                    sm['cursorXgl'],
                    sm['cursorYgl'],
                    w2h,
                    sm['drawInfo']
                    )
                ):

            self.scrollButton(sm['mousePressed'])

            # Record position of cursor when left-clicked
            if (sm['mousePressed'] == 0):
                self.mouseCursorAtPressX = mapRanges(sm['cursorX'], 0, sm['windowDimW'], -w2h, w2h)
                self.mouseCursorAtPressY = mapRanges(sm['cursorY'], 0, sm['windowDimH'], 1.0, -1.0)
                if (self.scrollSnap.isAnimating()):
                    self.prevSelectionCursorPosition = self.scrollSnap.getTar()
                else:
                    self.prevSelectionCursorPosition = self.selectionCursorPosition

            if (sm['mouseReleased'] == 0):
                self.selectionCursorVelocity = 0.0
                self.scrollSnap.setTargetVal(self.delimitValue(self.scrollSnap.getTar()-1.0))

            # Swipe to scroll if cursor is dragged
            dfx = mapRanges(sm['cursorX'], 0, sm['windowDimW'], -w2h, w2h) - self.mouseCursorAtPressX
            dfy = mapRanges(sm['cursorY'], 0, sm['windowDimH'], 1.0, -1.0) - self.mouseCursorAtPressY
            if (hypot(dfx, dfy) > thr):
                self.isTrackingScroll = True

            return True

        if (    self.isOpen()
                and
                not self.isTrackingScroll
                and
                watchDot(
                    ofx + cos(radians(ang))*tmx2,
                    ofy + sin(radians(ang))*tmy2,
                    tmes2,
                    sm['cursorXgl'],
                    sm['cursorYgl'],
                    w2h,
                    sm['drawInfo']
                    )
                ):

            self.scrollButton(sm['mousePressed'])

            # Record position of cursor when left-clicked
            if (sm['mousePressed'] == 0):
                self.mouseCursorAtPressX = mapRanges(sm['cursorX'], 0, sm['windowDimW'], -w2h, w2h)
                self.mouseCursorAtPressY = mapRanges(sm['cursorY'], 0, sm['windowDimH'], 1.0, -1.0)
                if (self.scrollSnap.isAnimating()):
                    self.prevSelectionCursorPosition = self.scrollSnap.getTar()
                else:
                    self.prevSelectionCursorPosition = self.selectionCursorPosition

            if (sm['mouseReleased'] == 0):
                self.selectionCursorVelocity = 0.0
                self.scrollSnap.setTargetVal(self.delimitValue(self.scrollSnap.getTar()+1.0))

            # Swipe to scroll if cursor is dragged
            dfx = mapRanges(sm['cursorX'], 0, sm['windowDimW'], -w2h, w2h) - self.mouseCursorAtPressX
            dfy = mapRanges(sm['cursorY'], 0, sm['windowDimH'], 1.0, -1.0) - self.mouseCursorAtPressY
            if (hypot(dfx, dfy) > thr):
                self.isTrackingScroll = True

            return True

        # Watch Menu Body for scroll / select
        polygon = []
        da  = 90.0-degrees(atan((5.75)+float(self.getIndexDraw())))

        # Proximal Box Corners
        radius = sqrt(2)
        radx = radius*self.UIelement.getTarSize()
        rady = radius*tms

        tmr = radians(ang+45)
        polygon.append( (ofx+cos(tmr)*radx, ofy+sin(tmr)*rady) )

        tmr = radians(ang-45)
        polygon.append( (ofx+cos(tmr)*radx, ofy+sin(tmr)*rady) )

        # Distil Box Corners
        radius  = (5.75)+float(self.getIndexDraw())
        radx    = radius*self.UIelement.getTarSize()
        rady    = radius*tms

        tmr = radians(ang-da)
        polygon.append( (ofx+cos(tmr)*radx, ofy+sin(tmr)*rady) )

        tmr = radians(ang+da)
        polygon.append( (ofx+cos(tmr)*radx, ofy+sin(tmr)*rady) )
        
        # Aspect correct radius for watchdot
        radius *= self.UIelement.getSize()
        if w2h < 1.0:
            ofy = ofy+radius*sin(radians(ang))*w2h
        else:
            ofy = (ofy+radius*sin(radians(ang)))

        # Watch body for input
        if (    self.isOpen()
                and
                (
                    self.isTrackingScroll
                    or
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
                )
            ):

            self.scrollButton(sm['mousePressed'])

            # Swipe to scroll
            if (self.isTrackingScroll):
                tmx = mapRanges(sm['cursorX'], 0, sm['windowDimW'], -w2h, w2h) - self.mouseCursorAtPressX
                tmy = mapRanges(sm['cursorY'], 0, sm['windowDimH'], 1.0, -1.0) - self.mouseCursorAtPressY
                tmx /= 1.75*self.UIelement.getTarSize()
                tmy /= 1.75*self.UIelement.getTarSize()
                tmx += self.UIelement.getPosX()*self.UIelement.getSize()
                tmy += self.UIelement.getPosY()*self.UIelement.getSize()
                radAng = radians(self.angle)
                self.selectionCursorPosition = self.delimitValue(
                        tmx*cos(radAng) + tmy*sin(radAng) + self.prevSelectionCursorPosition
                        )
        
            # Add mouse cursor's velocity to menu cursor's velocity (kinetic scrolling)
            if (sm['mouseReleased'] == 0):
                #if (not self.scrollSnap.isAnimating()):
                self.isTrackingScroll = False
                tmv = (
                    (2.0*sm['cursorVelSmoothed'][0]*cos(self.angle)) + 
                    (2.0*sm['cursorVelSmoothed'][1]*sin(self.angle))
                )
                self.selectionCursorVelocity += tmv

            if (sm['mousePressed'] == 0):
                self.mouseCursorAtPressX = mapRanges(sm['cursorX'], 0, sm['windowDimW'], -w2h, w2h)
                self.mouseCursorAtPressY = mapRanges(sm['cursorY'], 0, sm['windowDimH'], 1.0, -1.0)
                if (self.scrollSnap.isAnimating()):
                    self.prevSelectionCursorPosition = self.scrollSnap.getTar()
                else:
                    self.prevSelectionCursorPosition = self.selectionCursorPosition
                self.isTrackingScroll = True

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
        self.selectionCursorVelocity = 0.0
        return

    # Toggles menu
    def toggleOpen(self):

        # Menu is Open, close
        if (self.deployed.getTar() == 1.0):
            self.deployed.setTargetVal(0.0)
            self.selectionCursorVelocity = 0.0

        # Menus is Closed, Open
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

        # Idle, no input
        if (sm['currentState'] == 1):

            # Update Cursor position
            tmp = self.selectionCursorPosition + 2.0*self.selectionCursorVelocity*tDelta + accel*pow(tDelta, 2.0)
            self.selectionCursorPosition = self.delimitValue(tmp)

            # Decay cursor velocity
            tmp = self.selectionCursorVelocity + accel*tDelta

            # Don't let cursor take forever to coast to a stop
            if (abs(tmp) <= 0.15):
                self.selectionCursorVelocity = 0.0

                # Snap cursor to nearest whole number, animate
                tmn = normalizeCursor(self.prevSelectionCursorPosition, self.selectionCursorPosition)
                # shift range to positive
                while tmn < 0.0:
                    tmn += 1.0

                if (    tmn >= 0.000001
                        and
                        tmn <= 0.999999
                        and
                        not self.scrollSnap.isAnimating()
                        ):
                    self.scrollSnap.setValue(self.delimitValue(self.selectionCursorPosition))
                    self.scrollSnap.setTargetVal(round(self.delimitValue(self.selectionCursorPosition)))
                else:

                    # Resolve weird edge case bug related to scrolling to bottom of list
                    #if (    self.selectionCursorPosition == 1.0
                            #or
                            #self.selectionCursorPosition == float(self.numElements-2)
                            #):
                        #pass
                    #else:
                        #self.selectionCursorPosition = self.delimitValue(self.scrollSnap.getVal())
                    self.selectionCursorPosition = self.delimitValue(self.scrollSnap.getVal())
            else:
                # Update Cursor velocity
                self.selectionCursorVelocity = tmp

        if (self.scrollSnap.isTargetReached()):
            self.selectedElement = rollover(round(self.delimitValue(self.selectionCursorPosition)), self.numElements)
        else:
            self.selectedElement = rollover(round(self.delimitValue(self.scrollSnap.getTar())), self.numElements)

        if (self.prevSelectedElement != self.selectedElement):
            #print(self.selectedElement, self.selectionCursorPosition)
            self.prevSelectedElement = self.selectedElement

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
        sc = normalizeCursor(self.prevSelectionCursorPosition, self.selectionCursorPosition)
        if sc < 0.0:
            sc += 1.0

        tmc = 0.0
        #if (self.scrollSnap.isAnimating()):
            #tmc = self.scrollSnap.getTar()
        #else:
            #tmc = self.selectionCursorPosition
        tmc = self.selectionCursorPosition

        self.elementCoords = drawMenu(
                mx,
                my,
                tms,
                self.angle,
                self.deployed.getVal(),
                tmc,
                sc,
                self.numElements,
                self.menuLayout,
                self.dispIndex,
                w2h,
                stateMach['faceColor'],
                stateMach['detailColor']
                )

        return

