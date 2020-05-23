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

        # (experimental) number of listings to display
        self.numListings = 7

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

        diff = floor(self.numListings/2.0)
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
            if (tmc <= diff):
                return [i for i in range(self.numListings)]
            elif (tmc >= self.numElements-1.0-diff):#2.0 and tmc <= self.numElements-1.0):
                return [self.numElements-self.numListings+i for i in range(self.numListings)]
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
        #print(tml)
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

    # increase number of visible elements
    def increaseDensity(self):
        self.numListings = constrain(self.numListings + 2, 3, 7)
    # decrease number of visible elements
    def decreaseDensity(self):
        self.numListings = constrain(self.numListings - 2, 3, 7)

    # Simplify mouse wheel scrolling
    def scrollButton(self, sm):
        self.selectionCursorVelocity = 0.0
        # Mouse wheel scroll up
        if (sm['mousePressed'] == 3):
            if (sm['CtrlActive']):
                self.decreaseDensity()
                pass
            else:
                if (self.scrollSnap.isAnimating()):
                    self.scrollSnap.setTargetVal(self.delimitValue(self.scrollSnap.getTar()+1.0))
                else:
                    self.scrollSnap.setTargetVal(self.delimitValue(round(self.selectionCursorPosition)+1.0))

        # Mouse wheel scroll down
        if (sm['mousePressed'] == 4):
            if (sm['CtrlActive']):
                self.increaseDensity()
                pass
            else:
                if (self.scrollSnap.isAnimating()):
                    self.scrollSnap.setTargetVal(self.delimitValue(self.scrollSnap.getTar()-1.0))
                else:
                    self.scrollSnap.setTargetVal(self.delimitValue(round(self.selectionCursorPosition)-1.0))

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
        thr     = 0.1*self.UIelement.getSize()
        diff    = floor(self.numListings/2)
        tmes    = 0.125
        endOffset   = 1.0/(self.numListings-1.0)
        elementSpacing  = ((6.0-endOffset)-(1.5+endOffset))/(self.numListings-1.0)
        if w2h < 1.0:
            tmes *= w2h

        # watch elements directly for selection/scroll
        for i in range(self.numListings):
            tmx = (1.5 + endOffset + i*elementSpacing)*self.UIelement.getTarSize()
            tmy = (1.5 + endOffset + i*elementSpacing)*self.UIelement.getTarSize()
            ofwx = cos(radians(ang))*tmx
            if w2h < 1.0:
                tmy *= w2h
                ofwx *= w2h
            if (    self.isOpen()
                    and
                    not self.isTrackingScroll
                    and
                    watchDot(
                        #(ofx + cos(radians(ang))*tmx)*w2h,
                        ofx*w2h + ofwx,
                        ofy + sin(radians(ang))*tmy,
                        tmes,
                        sm['cursorXgl'],
                        sm['cursorYgl'],
                        w2h,
                        sm['drawInfo']
                        )
                    ):
                # Scroll with mouse wheel
                self.scrollButton(sm)

                # Record position of cursor when left-clicked
                if (sm['mousePressed'] == 0):
                    self.mouseCursorAtPressX = mapRanges(sm['cursorX'], 0, sm['windowDimW'], -w2h, w2h)
                    self.mouseCursorAtPressY = mapRanges(sm['cursorY'], 0, sm['windowDimH'], 1.0, -1.0)
                    self.prevSelectionCursorPosition = self.selectionCursorPosition

                if (sm['mouseReleased'] == 0):
                    #self.selectionCursorVelocity = 0.0
                    tml = self.cursor2indices(1.0)
                    self.scrollSnap.setTargetVal(tml[self.numListings-1-i])
                    #self.scrollSnap.setTargetVal(self.delimitValue(self.scrollSnap.getTar()-1.0))

                # Swipe to scroll if cursor is dragged
                dfx = mapRanges(sm['cursorX'], 0, sm['windowDimW'], -w2h, w2h) - self.mouseCursorAtPressX
                dfy = mapRanges(sm['cursorY'], 0, sm['windowDimH'], 1.0, -1.0) - self.mouseCursorAtPressY
                if (    hypot(dfx, dfy) > thr
                        and
                        sm['mouseButton'] == 0
                        ):
                    self.isTrackingScroll = True

                return True

        ofwx = ofx
        if w2h > 1.0:
            ofwx *= w2h

        # Watch Menu Body for scroll / select
        polygon = []
        da  = 90.0-degrees(atan((5.75)+float(self.getIndexDraw())))

        # Proximal Box Corners
        radius = sqrt(2)
        radx = radius*self.UIelement.getTarSize()
        rady = radius*tms

        tmr = radians(ang+45)
        polygon.append( (ofwx+cos(tmr)*radx, ofy+sin(tmr)*rady) )

        tmr = radians(ang-45)
        polygon.append( (ofwx+cos(tmr)*radx, ofy+sin(tmr)*rady) )

        # Distil Box Corners
        radius  = (5.75)+float(self.getIndexDraw())
        radx    = radius*self.UIelement.getTarSize()
        rady    = radius*tms

        tmr = radians(ang-da)
        polygon.append( (ofwx+cos(tmr)*radx, ofy+sin(tmr)*rady) )

        tmr = radians(ang+da)
        polygon.append( (ofwx+cos(tmr)*radx, ofy+sin(tmr)*rady) )
        
        # Aspect correct radius for watchdot
        radius *= self.UIelement.getSize()
        if w2h < 1.0:
            ofwx = (ofwx+radius*cos(radians(ang)))*w2h
            ofy = ofy+radius*sin(radians(ang))*w2h
        else:
            ofwx = (ofwx+radius*cos(radians(ang)))
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
                        #ofwx+radius*cos(radians(ang)), 
                        ofwx,
                        ofy,
                        tms,
                        sm['cursorXgl'],
                        sm['cursorYgl'],
                        w2h,
                        sm['drawInfo']
                        )
                )
            ):

            # Scroll with mouse wheel
            self.scrollButton(sm)

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

                # normalize velocity to menu's slide-out direction
                tmv = (
                    (4.0*sm['cursorVelSmoothed'][0]*cos(self.angle)) + 
                    (4.0*sm['cursorVelSmoothed'][1]*sin(self.angle))
                )
                self.selectionCursorVelocity += tmv

            # Begin dragging menu elements
            if (sm['mousePressed'] == 0):

                # Record initial mouse coordinates to get delta
                self.mouseCursorAtPressX = mapRanges(sm['cursorX'], 0, sm['windowDimW'], -w2h, w2h)
                self.mouseCursorAtPressY = mapRanges(sm['cursorY'], 0, sm['windowDimH'], 1.0, -1.0)

                # Re/Set animation, initial position, tracking flag
                #if (self.scrollSnap.isAnimating()):
                    #self.prevSelectionCursorPosition = self.scrollSnap.getTar()
                #else:
                    #self.prevSelectionCursorPosition = self.selectionCursorPosition
                self.prevSelectionCursorPosition = self.selectionCursorPosition
                self.isTrackingScroll = True

            return True

        # Resolve peculiar edge-case bug
        if (    sm['currentMouseButtonState'] == 1
                or
                sm['mousePressed'] != 0):
            self.isTrackingScroll = False

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
        if (sm['currentMouseButtonState'] == 1):

            # Update Cursor position
            tmp = self.selectionCursorPosition + 2.0*self.selectionCursorVelocity*tDelta + accel*pow(tDelta, 2.0)
            self.selectionCursorPosition = self.delimitValue(tmp)

            # Decay cursor velocity
            tmv = self.selectionCursorVelocity + accel*tDelta

            # Zero cursor velocity if cursor reaches end of linear strip terminated list
            if (    self.menuLayout == 1
                    and
                    (   self.selectionCursorPosition == 0.0
                        and
                        self.selectionCursorVelocity < 0.0
                        )
                    or
                    (   self.selectionCursorPosition == self.numElements-1.0
                        and
                        self.selectionCursorVelocity > 0.0
                        )
                    ):
                tmv = 0.0

            # Prematurely zero cursor velocity so it doesn't take forever to coast to a stop,
            # animate snap to index of nearest whole number
            if (abs(tmv) <= 0.15):
                self.selectionCursorVelocity = 0.0

                # Snap cursor to nearest whole number, animate
                tmn = normalizeCursor(self.prevSelectionCursorPosition, self.selectionCursorPosition)
                while tmn < 0.0:
                    tmn += 1.0

                # Resolve edge-case bug related to kinetic scroll reached list terminals
                if (    self.menuLayout == 1
                        and
                        (   self.selectionCursorPosition == 0.0
                            or
                            self.selectionCursorPosition == self.numElements-1.0
                            )
                        and
                        not self.scrollSnap.isAnimating()
                        ):
                    self.scrollSnap.setValue(self.delimitValue(self.selectionCursorPosition))
                    self.scrollSnap.setTargetVal(round(self.delimitValue(self.selectionCursorPosition)))

                # Snap-to-whole number animation finished
                if (    tmn >= 0.000001
                        and
                        tmn <= 0.999999
                        and
                        not self.scrollSnap.isAnimating()
                        ):
                    self.scrollSnap.setValue(self.delimitValue(self.selectionCursorPosition))
                    self.scrollSnap.setTargetVal(round(self.delimitValue(self.selectionCursorPosition)))
                else: # Run Snap-to-whole number animation
                    self.selectionCursorPosition = self.scrollSnap.getVal()
            else:
                # Update Cursor velocity
                self.selectionCursorVelocity = tmv

        # Update Index of the element ultimately returned, always a whole number
        if (self.scrollSnap.isTargetReached()):
            self.selectedElement = rollover(round(self.delimitValue(self.selectionCursorPosition)), self.numElements)
        else:
            self.selectedElement = rollover(round(self.delimitValue(self.scrollSnap.getTar())), self.numElements)

        # Detect changes in selected element
        if (self.prevSelectedElement != self.selectedElement):
            #self.scrollSnap.setTargetVal(self.selectedElement)
            #self.scrollSnap.setValue(self.selectedElement)
            #print(self.selectedElement, self.selectionCursorPosition)
            self.prevSelectedElement = self.selectedElement

        # Update Timer
        self.tPhys = time.time()

        # Update animations
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
        scrollCursor = normalizeCursor(
                self.prevSelectionCursorPosition, 
                self.selectionCursorPosition
                )
        if scrollCursor < 0.0:
            scrollCursor += 1.0

        tmc = 0.0
        tmc = self.selectionCursorPosition

        while tmc > self.numElements-1.0:
            tmc -= self.numElements

        while tmc < 0.0:
            tmc += self.numElements
        #print(self.selectionCursorVelocity, self.selectionCursorPosition, self.selectedElement, self.scrollSnap.getTar(), self.scrollSnap.getVal())
        self.elementCoords = drawMenu(
                mx,
                my,
                tms,
                self.angle,
                self.deployed.getVal(),
                tmc,
                scrollCursor,
                self.numElements,
                self.menuLayout,
                self.numListings,
                self.dispIndex,
                w2h,
                stateMach['faceColor'],
                stateMach['detailColor']
                )

        return
