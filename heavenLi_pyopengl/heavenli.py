#!/usr/bin/python3

print("Now Loading...")
from hliImports import *
from lampClass import *
print("All Imports Loaded...")

#import profile

def init():
    global stateMach
    makeFont()

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    print("Loading Shaders...")
    initShaders()
    print("Done!")

    stateMach['second'] = datetime.datetime.now().second
    stateMach['minute'] = datetime.datetime.now().minute + stateMach['second']/60
    stateMach['hour'] = datetime.datetime.now().hour + stateMach['minute']/60
    if (stateMach['hour'] > 11):
        stateMach['hour'] -= 12

    stateMach['windowDimW'] = glutGet(GLUT_WINDOW_WIDTH)
    stateMach['windowDimH'] = glutGet(GLUT_WINDOW_HEIGHT)
    stateMach['w2h'] = stateMach['windowDimW']/stateMach['windowDimH']

    print("Initialization Finished")
    return

# Throttle FPS and update timers
def framerate():
    global stateMach
    t = time.time()
    stateMach['frames'] += 1.0
    seconds = t - stateMach['t0']
    if (seconds <= 0.0):
        seconds = 0.00000000001
    stateMach['someVar'] += stateMach['someInc']
    if (stateMach['someVar'] > 100) or (stateMach['someVar'] < 0):
        stateMach['someInc'] = -stateMach['someInc']

    try:
        stateMach['fps'] = stateMach['frames']/seconds
    except:
        stateMach['fps'] = 60
        print("Too Fast, Too Quick!!")

    if t - stateMach['t1'] >= (0.0125):
        #calcCursorVelocity(0)
        stateMach['t1'] = t

    if t - stateMach['t0'] >= 1.0:
        #print("%.0f frames in %3.1f seconds = %6.3f FPS" % (stateMach['frames'],seconds,stateMach['fps']))
        stateMach['lamps']  = plugins.pluginLoader.getAllLamps()
        stateMach['t0']     = t
        stateMach['frames'] = 0
        ct                  = datetime.datetime.now()
        stateMach['second'] = ct.second
        stateMach['minute'] = ct.minute + stateMach['second']*0.016666667
        stateMach['hour']   = ct.hour + stateMach['minute']*0.016666667
        if (stateMach['hour'] > 11):
            stateMach['hour'] -= 12

    if stateMach['frameLimit'] and (stateMach['fps'] > 66):
        pass
        time.sleep(0.015)

    return

# Function for testing drawcode
def drawTest():
    try:
        #stateMach['Menus']['testMenu'].setAng(360.0*stateMach['someVar']/100.0)
        #stateMach['Menus']['testMenu'].setAng(00.0)
        stateMach['Menus']['testMenu'].setAng(90.0)
        #stateMach['Menus']['testMenu'].setLayout(1)
        w2h = stateMach['w2h']

        # test color that changes over time
        tmc = ( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 1.0)
        tmx = stateMach['BallPosition'][0]
        tmy = stateMach['BallPosition'][1]

        if (w2h < 1.0):
            tmx /= w2h
            tmy /= w2h

        # Draw Purple Ball
        drawEllipse(
                tmx,
                tmy,
                0.1, 0.1,
                w2h,
                #(0.42, 0.0, 0.85, 1.0)
                stateMach['testList'][stateMach['Menus']['testMenu'].getSelection()]
                )

        cDrag  = 1.0
        accelX = -stateMach['BallVelocity'][0]*cDrag
        accelY = -stateMach['BallVelocity'][1]*cDrag
        # Update Ball position based on its cartesian velocity vector
        PxDs = min(stateMach['windowDimW'], stateMach['windowDimH'])
        tDelta = (time.time()-stateMach['tPhys'])

        # Update Ball Position based on velocity + acceleration
        tmx = stateMach['BallPosition'][0]+2.0*stateMach['BallVelocity'][0]*tDelta+accelX*pow(tDelta, 2.0)
        tmy = stateMach['BallPosition'][1]+2.0*stateMach['BallVelocity'][1]*tDelta+accelY*pow(tDelta, 2.0)

        # Wrap ball around to opposite side if it hits edge of screen
        if (tmx < -w2h):
            tmx = w2h
        if (tmx > w2h):
            tmx = -w2h
        if (tmy < -1.0):
            tmy = 1.0
        if (tmy > 1.0):
            tmy = -1.0

        stateMach['BallPosition'] = (tmx, tmy)

        tmx = stateMach['BallVelocity'][0]
        tmy = stateMach['BallVelocity'][1]

        ## Reverse Ball's Velocity component(s) if it hits edge of screen
        #if (stateMach['BallPosition'][0] < -w2h):
            #tmx = abs(stateMach['BallVelocity'][0])
        #if (stateMach['BallPosition'][0] > w2h):
            #tmx = -abs(stateMach['BallVelocity'][0])
        #if (stateMach['BallPosition'][1] < -1.0):
            #tmy = abs(stateMach['BallVelocity'][1])
        #if (stateMach['BallPosition'][1] > 1.0):
            #tmy = -abs(stateMach['BallVelocity'][1])

        # Reduce Ball's velocity over time to simulate drag
        tvx = tmx + accelX*tDelta
        tvy = tmy + accelY*tDelta
        if (sqrt(pow(tvx, 2.0)+pow(tvy, 2.0)) <= 0.01):
            stateMach['BallVelocity'] = (0.0, 0.0)
        else:
            stateMach['BallVelocity'] = (tvx, tvy)
        stateMach['tPhys'] = time.time()

        red = (1.0, 0.0, 0.0, 1.0)
        blu = (0.0, 0.0, 1.0, 1.0)

        if (w2h >= 1.0):
            tmx = stateMach['BallPosition'][0]
            tmy = stateMach['BallPosition'][1]
        else:
            tmx = stateMach['BallPosition'][0]/w2h
            tmy = stateMach['BallPosition'][1]/w2h

        drawPill(
                tmx,
                tmy,
                tmx + stateMach['BallVelocity'][0]*0.1,
                tmy + stateMach['BallVelocity'][1]*0.1,
                0.02, 
                w2h, 
                blu,
                red 
                )

        if (w2h >= 1.0):
            tmx = mapRanges(stateMach['cursorX'], 0, stateMach['windowDimW'], -w2h, w2h)
            tmy = mapRanges(stateMach['cursorY'], 0, stateMach['windowDimH'], 1.0, -1.0)
        else:
            tmx = mapRanges(stateMach['cursorX'], 0, stateMach['windowDimW'], -1.0, 1.0)
            tmy = mapRanges(stateMach['cursorY'], 0, stateMach['windowDimH'], 1.0/w2h, -1.0/w2h)

        drawPill(
                tmx,
                tmy,
                tmx + stateMach['cursorVelSmoothed'][0]*0.1,
                tmy + stateMach['cursorVelSmoothed'][1]*0.1,
                0.02, 
                w2h, 
                blu,
                red 
                )

        stateMach['Menus']['testMenu'].draw(stateMach)

        # Get relative positions / sizes of visible elements
        quack = stateMach['Menus']['testMenu'].getElements()

        # Loop through drawing menu elements, draw first and last elements before middle elements for proper z-layering draw-order
        for i in range(-1, len(quack)-1):
            tmx = quack[i][1]*stateMach['Menus']['testMenu'].UIelement.getSize()
            tmy = quack[i][2]*stateMach['Menus']['testMenu'].UIelement.getSize()

            if w2h < 1.0:
                tmx += stateMach['Menus']['testMenu'].UIelement.getPosX()
                tmy += stateMach['Menus']['testMenu'].UIelement.getPosY()/w2h
            else:
                tmx += stateMach['Menus']['testMenu'].UIelement.getPosX()*w2h
                tmy += stateMach['Menus']['testMenu'].UIelement.getPosY()

            drawEllipse(
                    tmx,
                    tmy,
                    quack[i][3]*0.125,
                    quack[i][3]*0.125,
                    w2h,
                    stateMach['testList'][quack[i][0]]
                    )

        # Draw selected element in menu core
        tmx = stateMach['Menus']['testMenu'].UIelement.getPosX()
        tmy = stateMach['Menus']['testMenu'].UIelement.getPosY()
        if w2h < 1.0:
            tmy /= w2h
        else:
            tmx *= w2h
        drawEllipse(
                tmx,
                tmy,
                stateMach['Menus']['testMenu'].UIelement.getSize()*0.7,
                stateMach['Menus']['testMenu'].UIelement.getSize()*0.7,
                w2h,
                stateMach['testList'][stateMach['Menus']['testMenu'].getSelection()]
                )

        pass

    except Exception as OOF:
        print(traceback.format_exc())
        print("Error:", OOF)
    return

def watchTest():
    try:
        w2h = stateMach['w2h']
        if (    watchScreen()
                ):

            if (stateMach['mouseButton'] == 2):
                tmx = mapRanges(stateMach['cursorX'], 0, stateMach['windowDimW'], -w2h, w2h)
                tmy = mapRanges(stateMach['cursorY'], 0, stateMach['windowDimH'], 1.0, -1.0)
                stateMach['BallPosition'] = (tmx, tmy)

            #if (stateMach['Menus']['testMenu'].watch(stateMach)):
            if (stateMach['Menus']['testMenu'].watch(filterKeys(stateMach, 
                [
                    'AltActive',
                    'CtrlActive',
                    'currentMouseButtonState',
                    'cursorVelSmoothed',
                    'cursorX',
                    'cursorY',
                    'cursorXgl',
                    'cursorYgl',
                    'drawInfo',
                    'keyPressed',
                    'mouseButton',
                    'mousePressed',
                    'mouseReleased',
                    'ShiftActive',
                    'w2h',
                    'windowDimW',
                    'windowDimH'
                ]
                    )
                )
                ):
                stateMach['noButtonsPressed'] = False
                pass

            # Drag window if no buttons pressed
            dragWindow()

        if (stateMach['mouseReleased'] == 2):
            tmv = (
                    stateMach['BallVelocity'][0] + stateMach['cursorVelSmoothed'][0],
                    stateMach['BallVelocity'][1] + stateMach['cursorVelSmoothed'][1]
                    )
            stateMach['BallVelocity'] = tmv
        pass
    except Exception as OOF:
        print(traceback.format_exc())
        print("Error:", OOF)
    return

def drawElements():
    global stateMach

    # Placeholder variable
    Light = 0

    iconSize = 0.15

    fcc = stateMach['faceColor']
    dtc = stateMach['detailColor']
    w2h = stateMach['w2h']

    try:
        # Draw Background
        if (len(stateMach['lamps']) > 0):

            if (stateMach['lamps'][Light].getArn() == 0):
                pass
                drawHomeCircle(0.0, 0.0, 
                        stateMach['windowDimW'], stateMach['windowDimH'], 
                        stateMach['lamps'][Light].getNumBulbs(), 
                        stateMach['lamps'][Light].getAngle(), 
                        w2h,
                        stateMach['lamps'][Light].getBulbsCurrentRGB()
                        )

            if (stateMach['lamps'][Light].getArn() == 1):
                pass
                drawHomeLinear(0.0, 0.0, 
                        stateMach['windowDimW'], stateMach['windowDimH'],
                        stateMach['lamps'][Light].getNumBulbs(), 
                        stateMach['lamps'][Light].getAngle(), 
                        w2h,
                        stateMach['lamps'][Light].getBulbsCurrentRGB()
                        )

        # Draw Menus
        #for key in stateMach['Menus']:
            #stateMach['Menus'][key].draw(stateMach)

        # Draw Granularity Rocker Underneath Clock
        if (w2h <= 1.0):
            gctmy = stateMach['UIelements']['GranChanger'].getPosY()*stateMach['w2h']
        else:
            gctmy = stateMach['UIelements']['GranChanger'].getPosY()

        limit = -0.90
        if (stateMach['UIelements']['GranChanger'].getPosY() >= limit):
            drawGranChanger(
                    stateMach['UIelements']['GranChanger'].getPosX(),
                    gctmy,
                    stateMach['UIelements']['GranChanger'].getFaceColor(),
                    stateMach['UIelements']['GranChanger'].getDetailColor(),
                    stateMach['numHues'],
                    0.0,
                    w2h,
                    stateMach['UIelements']['GranChanger'].getSize(),
                    stateMach['tDiff']
                    )
    
        # Draw Clock
        drawClock(
            stateMach['UIelements']['MasterSwitch'].getPosX(),              # X-Coordinate of position
                stateMach['UIelements']['MasterSwitch'].getPosY(),          # Y-Coordinate of position
                stateMach['hour'],                                          # Hour to be displayed
                stateMach['minute'],                                        # Minute to be displayed
                stateMach['UIelements']['MasterSwitch'].getSize(),          # Size of Clock
                w2h,                                           # w2h for proper aspect ratio scaling
                stateMach['UIelements']['MasterSwitch'].getFaceColor(),     # color of the clock face
                stateMach['UIelements']['MasterSwitch'].getDetailColor()    # color of the clock hands
                )

        # Draw Granularity Rocker on top of Clock
        if (stateMach['UIelements']['GranChanger'].getPosY() < limit):
            drawGranChanger(
                    stateMach['UIelements']['GranChanger'].getPosX(),
                    gctmy,
                    stateMach['UIelements']['GranChanger'].getFaceColor(),
                    stateMach['UIelements']['GranChanger'].getDetailColor(),
                    stateMach['numHues'],
                    0.0,
                    w2h,
                    stateMach['UIelements']['GranChanger'].getSize(),
                    stateMach['tDiff']
                    )

        # Draw Lamp buttons
        if (len(stateMach['lamps']) > 0):

            # Draw Bulb Buttons
            if (stateMach['UIelements']['BulbButtons'].isVisible()):
                stateMach['bulbButtons'] = drawBulbButton(
                        stateMach['lamps'][Light].getArn(),
                        stateMach['lamps'][Light].getNumBulbs(),
                        60,
                        stateMach['lamps'][Light].getAngle(),
                        stateMach['UIelements']['BulbButtons'].getSize(),
                        stateMach['UIelements']['BulbButtons'].getFaceColor(),
                        stateMach['UIelements']['BulbButtons'].getDetailColor(),
                        stateMach['lamps'][Light].getBulbsCurrentRGB(),
                        w2h
                        )

            # Draw All-Set Button
            if (stateMach['UIelements']['AllSetButton'].isVisible()):
                drawIcon(
                        stateMach['UIelements']['AllSetButton'].getPosX(), 
                        stateMach['UIelements']['AllSetButton'].getPosY(), 
                        stateMach['UIelements']['AllSetButton'].getSize(), 
                        stateMach['lamps'][Light].getArn(),
                        stateMach['lamps'][Light].getAlias(),
                        stateMach['features'],
                        stateMach['UIelements']['AllSetButton'].getFaceColor(),
                        stateMach['UIelements']['AllSetButton'].getDetailColor(),
                        stateMach['lamps'][Light].getNumBulbs(),
                        stateMach['lamps'][Light].getAngle(),
                        w2h,
                        stateMach['lamps'][Light].getBulbsCurrentRGB()
                        )

        if (stateMach['wereColorsTouched']):
            selectRingColor = dtc
        else:
            selectRingColor = fcc

        # Draw Ring of Dots with different hues
        if (stateMach['UIelements']['HueRing'].isVisible()):
            stateMach['hueButtons'] = drawHueRing(
                    stateMach['UIelements']['HueRing'].getPosX(),
                    stateMach['UIelements']['HueRing'].getPosY(),
                    stateMach['UIelements']['HueRing'].getSize(), 
                    stateMach['currentHue'], 
                    stateMach['numHues'], 
                    selectRingColor, 
                    w2h, 
                    stateMach['tDiff']*0.5
                    )

        # Draw Triangle of Dots with different brightness/saturation
        if (stateMach['UIelements']['ColorTriangle'].isVisible()):
            stateMach['satValButtons'] = drawColrTri(
                    stateMach['UIelements']['ColorTriangle'].getPosX(),
                    stateMach['UIelements']['ColorTriangle'].getPosY(),
                    stateMach['UIelements']['ColorTriangle'].getSize(), 
                    stateMach['currentHue'], 
                    stateMach['currentSat'], 
                    stateMach['currentVal'],
                    int(stateMach['numHues']/2), 
                    selectRingColor,
                    w2h, 
                    stateMach['tDiff']*0.5
                    )

        # Draw Confirm Button
        if (stateMach['UIelements']['ConfirmButton'].isVisible()):
            extraColor = colorsys.hsv_to_rgb(
                    stateMach['currentHue'], 
                    stateMach['currentSat'], 
                    stateMach['currentVal']
                    )
            extraColor = (extraColor[0], extraColor[1], extraColor[2], 1.0)

            drawConfirm(
                    stateMach['UIelements']['ConfirmButton'].getPosX(), 
                    stateMach['UIelements']['ConfirmButton'].getPosY(), 
                    stateMach['UIelements']['ConfirmButton'].getSize(), 
                    w2h, 
                    fcc, 
                    extraColor, 
                    dtc
                    )

        # Draw Back Button
        if (stateMach['UIelements']['BackButton'].isVisible()):
            extraColor = colorsys.hsv_to_rgb(
                    stateMach['prevHue'], 
                    stateMach['prevSat'], 
                    stateMach['prevVal']
                    )
            extraColor = (extraColor[0], extraColor[1], extraColor[2], 1.0)

            # Draw Back Button
            drawArrow(
                    stateMach['UIelements']['BackButton'].getPosX(), 
                    stateMach['UIelements']['BackButton'].getPosY(), 
                    180.0,
                    stateMach['UIelements']['BackButton'].getSize(), 
                    w2h, 
                    fcc, 
                    extraColor, 
                    dtc
                    )

        #tmc = ( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 1.0)
    except Exception as OOF:
        print(traceback.format_exc())
        print("Error:", OOF)
    return


# Sets UI elements to go to home screen
def goHome(xOrigin=0.0, yOrigin=0.0):
    stateMach['wereColorsTouched'] = False
    stateMach['UIelements']['MasterSwitch'].setTarSize(1.0)
    stateMach['UIelements']['MasterSwitch'].setTargetFaceColor(stateMach['faceColor'])
    stateMach['UIelements']['MasterSwitch'].setTargetDetailColor(stateMach['detailColor'])
    stateMach['UIelements']['ColorTriangle'].setTarSize(0.0)
    stateMach['UIelements']['HueRing'].setTarSize(10.0)
    stateMach['UIelements']['BulbButtons'].setTarSize(0.4)
    stateMach['UIelements']['AllSetButton'].setTarSize(0.1275)
    stateMach['UIelements']['BackButton'].setTarSize(0.0)
    stateMach['UIelements']['ConfirmButton'].setTarSize(0.0)
    stateMach['UIelements']['GranChanger'].setTarSize(0.0)
    stateMach['UIelements']['GranChanger'].setTarPosY(0.0)
    stateMach['UIelements']['GranChanger'].setAccel(0.33)
    stateMach['targetScreen'] = 0


# Sets UI elements for color picker screen
def engageColorPicker(xOrigin=0.0, yOrigin=0.0):
    stateMach['targetScreen'] = 1
    stateMach['UIelements']['MasterSwitch'].setTarSize(1.75)
    stateMach['UIelements']['MasterSwitch'].setTargetFaceColor((0.0, 0.0, 0.0, 1.0))
    stateMach['UIelements']['MasterSwitch'].setTargetDetailColor((0.0, 0.0, 0.0, 0.0))
    stateMach['UIelements']['AllSetButton'].setTarSize(0.0)
    stateMach['UIelements']['BulbButtons'].setTarSize(0.0)

    stateMach['UIelements']['ColorTriangle'].setTarSize(1.0)
    stateMach['UIelements']['ColorTriangle'].setValue("coordX", xOrigin)
    stateMach['UIelements']['ColorTriangle'].setValue("coordY", yOrigin)
    stateMach['UIelements']['ColorTriangle'].setTarPosX(0.0)
    stateMach['UIelements']['ColorTriangle'].setTarPosY(0.0)

    stateMach['UIelements']['HueRing'].setSize(0.0)
    stateMach['UIelements']['HueRing'].setTarSize(1.0)
    stateMach['UIelements']['HueRing'].setValue("coordX", xOrigin)
    stateMach['UIelements']['HueRing'].setValue("coordY", yOrigin)
    stateMach['UIelements']['HueRing'].setTarPosX(0.0)
    stateMach['UIelements']['HueRing'].setTarPosY(0.0)

    stateMach['UIelements']['BackButton'].setTarSize(0.2)
    stateMach['UIelements']['BackButton'].setValue("coordX", xOrigin)
    stateMach['UIelements']['BackButton'].setValue("coordY", yOrigin)
    stateMach['UIelements']['BackButton'].setTarPosX(-0.75)
    stateMach['UIelements']['BackButton'].setTarPosY(-0.75)

    stateMach['UIelements']['ConfirmButton'].setTarSize( 0.2)
    stateMach['UIelements']['ConfirmButton'].setValue("coordX", xOrigin)
    stateMach['UIelements']['ConfirmButton'].setValue("coordY", yOrigin)
    stateMach['UIelements']['ConfirmButton'].setTarPosX( 0.75)
    stateMach['UIelements']['ConfirmButton'].setTarPosY(-0.75)

    stateMach['UIelements']['GranChanger'].setTarSize(0.3)
    stateMach['UIelements']['GranChanger'].setTarPosY(-0.91)
    return

# Watch Home Screen for inputs
def watchHomeInput():
    global stateMach

    # convenience / readibility variable
    w2h = stateMach['w2h']

    # Placeholder variable
    Light = 0
    #stateMach['noButtonsPressed'] = True

    # Watch Home Screen for input
    if (    watchScreen()
            and
            stateMach['WindowBarHeight'] != False
            ):
    
        #tmx = stateMach['UIelements']['testMenu'].getTarPosX()*w2h
        #tmy = stateMach['UIelements']['testMenu'].getTarPosY()
        #tms = stateMach['UIelements']['testMenu'].getTarSize()

        #if (w2h < 1.0):
            #tms *= w2h
            
        #if (watchDot(tmx, tmy, tms, stateMach['cursorXgl'], stateMach['cursorYgl'], w2h, stateMach['drawInfo'])
            #and
            #stateMach['mousePressed'] == 0
            #):
            #stateMach['Menus']['testMenu'].toggleOpen()

        tms = stateMach['UIelements']['MasterSwitch'].getTarSize()
        # Watch Clock for input
        if (w2h < 1.0):
            tms *= w2h

        if (watchDot(
            0.0,    # Middle of Screen
            0.0,    # Middle of Screen
            tms*0.5,
            stateMach['cursorXgl'],
            stateMach['cursorYgl'],
            stateMach['w2h'],
            stateMach['drawInfo']# Clock Radius
            )
            and
            stateMach['mousePressed'] == 0
            ):                 # Button Must be clicked
            stateMach['noButtonsPressed'] = False
            AreLightsOn = False
            for i in range(len(stateMach['lamps'])):
                if(stateMach['lamps'][i].isOn()):
                    AreLightsOn = True

            if (AreLightsOn):
                for i in range(len(stateMach['lamps'])):
                    stateMach['lamps'][i].setMainLight(False)
            else:
                for i in range(len(stateMach['lamps'])):
                    stateMach['lamps'][i].setMainLight(True)

        # Watch bulb buttons for input
        if (len(stateMach['lamps']) > 0):
            for i in range(len(stateMach['bulbButtons'])):

                posX = stateMach['bulbButtons'][i][0]   # X coord of button
                posY = stateMach['bulbButtons'][i][1]   # Y coord of button
                tms  = 0.16    # Bulb Radius, 0.4(GL buttonScale) times 0.4(UIele tarSize)
                if (w2h < 1.0):
                    tms  *= w2h

                if (watchDot(
                    posX, 
                    posY,
                    tms,
                    stateMach['cursorXgl'],
                    stateMach['cursorYgl'],
                    stateMach['w2h'],
                    stateMach['drawInfo']
                    )   
                    and
                    len(stateMach['lamps']) > 0
                    and
                    stateMach['mousePressed'] == 0
                    ): # Button Must be clicked
                    stateMach['noButtonsPressed'] = False

                    # Set Color Picker as target Screen selecting bulb i
                    stateMach['targetBulb'] = i
                    engageColorPicker(
                            stateMach['bulbButtons'][i][0]/w2h,
                            stateMach['bulbButtons'][i][1]
                            )

                    # Record previous color(s)
                    stateMach['prevHue'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[0]
                    stateMach['prevSat'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[1]
                    stateMach['prevVal'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[2]

            # Watch all-set for input
            posX = stateMach['UIelements']['AllSetButton'].getTarPosX()*w2h
            posY = stateMach['UIelements']['AllSetButton'].getTarPosY()
            tms  = stateMach['UIelements']['AllSetButton'].getTarSize()*1.75
            if (w2h < 1.0):
                tms  *= w2h
            if (watchDot(
                posX,
                posY,
                tms,
                stateMach['cursorXgl'],
                stateMach['cursorYgl'],
                stateMach['w2h'],
                stateMach['drawInfo']
                )
                and
                len(stateMach['lamps']) > 0
                and
                stateMach['mousePressed'] == 0
                ): # Button Must be clicked
                stateMach['noButtonsPressed'] = False

                # Set Color Picker as target Screen selecting bulb all bulbs
                stateMach['targetBulb'] = stateMach["lamps"][Light].getNumBulbs()
                engageColorPicker(
                        stateMach['UIelements']['AllSetButton'].getPosX(),
                        stateMach['UIelements']['AllSetButton'].getPosY()
                        )

                # Record previous color(s)
                stateMach['prevHue'] = stateMach['lamps'][Light].getBulbCurrentHSV(0)[0]
                stateMach['prevSat'] = stateMach['lamps'][Light].getBulbCurrentHSV(0)[1]
                stateMach['prevVal'] = stateMach['lamps'][Light].getBulbCurrentHSV(0)[2]
                for i in range(stateMach['targetBulb']):
                    stateMach['prevHues'][i] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[0]
                    stateMach['prevSats'][i] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[1]
                    stateMach['prevVals'][i] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[2]

        # Drag window if no buttons pressed
        dragWindow()

def watchColrSettingInput():
    global stateMach

    Light = 0
    # Convenience variable
    w2h = stateMach['w2h']

    # Watch Color Picker Screen for input
    if (watchScreen()):

        gctmx = stateMach['UIelements']['GranChanger'].getTarSize()*(24.0/36.0)
        tmr = min(stateMach['windowDimW'], stateMach['windowDimH']*(12.0/36.0)*0.3)
        if (w2h <= 1.0):
            gctmy = stateMach['UIelements']['GranChanger'].getPosY()*w2h
        else:
            gctmy = stateMach['UIelements']['GranChanger'].getPosY()

        if (w2h <= 1.0):
            gctmx *= w2h
            tmr *= w2h

        # Watch Granularity Rocker for Input
        posX = stateMach['UIelements']['GranChanger'].getTarSize()*(24.0/36.0)
        posY = stateMach['UIelements']['GranChanger'].getTarPosY()
        tms  = stateMach['UIelements']['GranChanger'].getTarSize()*0.35
        if (w2h < 1.0):
            posX *= w2h
            posY *= w2h
            tms  *= w2h

        if (watchDot(
            posX,
            posY,
            tms,
            stateMach['cursorXgl'],
            stateMach['cursorYgl'],
            stateMach['w2h'],
            stateMach['drawInfo']
            )
            and
            stateMach['mousePressed'] == 0
            ):
                stateMach['noButtonsPressed'] = False
                stateMach['numHues'] += 2
                if stateMach['numHues'] > 14:
                    stateMach['numHues'] = 14

        if (watchDot(
            -posX,
            posY,
            tms,
            stateMach['cursorXgl'],
            stateMach['cursorYgl'],
            stateMach['w2h'],
            stateMach['drawInfo']
            )
            and
            stateMach['mousePressed'] == 0
            ):
                stateMach['noButtonsPressed'] = False
                stateMach['numHues'] -= 2
                if stateMach['numHues'] < 10:
                    stateMach['numHues'] = 10

        # Watch Hue Ring for Input
        for i in range(len(stateMach['hueButtons'])):
            if (w2h <= 1.0):
                stateMach['hueButtons'][i] = (
                        stateMach['hueButtons'][i][0]*w2h, 
                        stateMach['hueButtons'][i][1]*w2h, 
                        stateMach['hueButtons'][i][2])

            posX = stateMach['hueButtons'][i][0]
            posY = stateMach['hueButtons'][i][1]
            tms  = 0.15*(12.0/float(stateMach['numHues']))
            if (w2h < 1.0):
                tms  *= w2h

            if (watchDot(
                posX, 
                posY, 
                tms, 
                stateMach['cursorXgl'], 
                stateMach['cursorYgl'], 
                w2h, 
                stateMach['drawInfo'] )
                and
                stateMach['mousePressed'] == 0
                ):
                stateMach['noButtonsPressed'] = False

                stateMach['wereColorsTouched'] = True
                stateMach['currentHue'] = stateMach['hueButtons'][i][2]
                tmcHSV = (
                        stateMach['currentHue'], 
                        stateMach['currentSat'], 
                        stateMach['currentVal'])

                if (len(stateMach['lamps']) > 0):
                    if (stateMach['targetBulb'] == stateMach['lamps'][Light].getNumBulbs()):
                        stateMach['lamps'][Light].setBulbsTargetHSV(tmcHSV)
                    else:
                        stateMach['lamps'][Light].setBulbTargetHSV(stateMach['targetBulb'], tmcHSV)


        # Watch Sat / Val Triangle for input
        for i in range(len(stateMach['satValButtons'])):
            if (w2h <= 1.0):
                stateMach['satValButtons'][i] = (
                        stateMach['satValButtons'][i][0]*w2h,     # X-Coord of Button
                        stateMach['satValButtons'][i][1]*w2h,     # Y-Coord of Button
                        stateMach['satValButtons'][i][2],                    # Saturation of Button
                        stateMach['satValButtons'][i][3])                    # Value of Button

            # Map relative GL-Coord to screen aspect-Coord
            posX = stateMach['satValButtons'][i][0]
            posY = stateMach['satValButtons'][i][1]
            tms  = 0.073
            if (w2h < 1.0):
                tms  *= w2h

            if (watchDot(
                posX,
                posY,
                tms,
                stateMach['cursorXgl'],
                stateMach['cursorYgl'],
                stateMach['w2h'],
                stateMach['drawInfo']
                )
                and
                stateMach['currentMouseButtonState'] == 0
                ):
                stateMach['noButtonsPressed'] = False

                stateMach['wereColorsTouched'] = True
                stateMach['currentSat'] = stateMach['satValButtons'][i][2]
                stateMach['currentVal'] = stateMach['satValButtons'][i][3]
                tmcHSV = (
                        stateMach['currentHue'], 
                        stateMach['currentSat'], 
                        stateMach['currentVal'])
                if (len(stateMach['lamps']) > 0):
                    if (stateMach['targetBulb'] == stateMach['lamps'][Light].getNumBulbs()):
                        stateMach['lamps'][Light].setBulbsTargetHSV(tmcHSV)
                    else:
                        stateMach['lamps'][Light].setBulbTargetHSV(stateMach['targetBulb'], tmcHSV)

        # Watch Confirm Button for input
        posX = stateMach['UIelements']['ConfirmButton'].getTarPosX()*w2h
        posY = stateMach['UIelements']['ConfirmButton'].getTarPosY()
        tms  = stateMach['UIelements']['ConfirmButton'].getTarSize()
        if (w2h < 1.0):
            tms  *= w2h
        if (watchDot(
            posX,
            posY,
            tms,
            stateMach['cursorXgl'],
            stateMach['cursorYgl'],
            stateMach['w2h'],
            stateMach['drawInfo']
            )
            and
            stateMach['mousePressed'] == 0
            ):
            stateMach['noButtonsPressed'] = False
            stateMach['wereColorsTouched'] = False
            if (len(stateMach['lamps']) > 0):
                if (stateMach['targetBulb'] == stateMach['lamps'][Light].getNumBulbs()):
                    stateMach['lamps'][Light].setBulbsTargetHSV( (
                        stateMach['currentHue'], 
                        stateMach['currentSat'], 
                        stateMach['currentVal'] ) )
                else:
                    stateMach['lamps'][Light].setBulbTargetHSV(stateMach['targetBulb'], (
                        stateMach['currentHue'], 
                        stateMach['currentSat'], 
                        stateMach['currentVal'] ) )

            goHome()

        if ( stateMach['wereColorsTouched'] and len(stateMach['lamps']) > 0):
            if (stateMach['targetBulb'] == stateMach['lamps'][Light].getNumBulbs()):
                extraColor = colorsys.hsv_to_rgb(
                        stateMach['prevHues'][0], 
                        stateMach['prevSats'][0], 
                        stateMach['prevVals'][0])
                extraColor = (extraColor[0], extraColor[1], extraColor[2], 1.0)
            else:
                extraColor = colorsys.hsv_to_rgb(
                        stateMach['prevHue'], 
                        stateMach['prevSat'], 
                        stateMach['prevVal'])
                extraColor = (extraColor[0], extraColor[1], extraColor[2], 1.0)
        else:
            extraColor = stateMach['detailColor']


        # Watch Back Button for input
        posX = stateMach['UIelements']['BackButton'].getTarPosX()*w2h
        posY = stateMach['UIelements']['BackButton'].getTarPosY()
        tms  = stateMach['UIelements']['BackButton'].getTarSize()
        if (w2h < 1.0):
            tms  *= w2h
        if (watchDot(
            posX,
            posY,
            tms,
            stateMach['cursorXgl'],
            stateMach['cursorYgl'],
            stateMach['w2h'],
            stateMach['drawInfo']
            )
        and
        stateMach['mousePressed'] == 0
        ):
            stateMach['noButtonsPressed'] = False
            stateMach['wereColorsTouched'] = False
            if (len(stateMach['lamps']) > 0):
                if (stateMach['targetBulb'] == stateMach['lamps'][Light].getNumBulbs()):
                    for i in range(stateMach['lamps'][Light].getNumBulbs()):
                        stateMach['lamps'][Light].setBulbTargetHSV(i, (
                            stateMach['prevHues'][i], 
                            stateMach['prevSats'][i], 
                            stateMach['prevVals'][i] ) )
                else:
                    stateMach['lamps'][Light].setBulbTargetHSV(stateMach['targetBulb'], (
                        stateMach['prevHue'], 
                        stateMach['prevSat'], 
                        stateMach['prevVal'] ) )

            goHome()

        # Drag window if no buttons pressed
        dragWindow()

# Used to process user input
def watchScreen():
    global stateMach

    # Calibrate window title bar height if not set
    if (    type(stateMach['WindowBarHeight']) == str
            and
            abs(glutGet(GLUT_WINDOW_Y) - stateMach['prevWindowPosY']) != 0
            ):
        #print(stateMach['WindowBarHeight'], glutGet(GLUT_WINDOW_Y), stateMach['prevWindowPosY'])
        stateMach['WindowBarHeight'] = glutGet(GLUT_WINDOW_Y) - stateMach['prevWindowPosY'] 
        print("Window Title Bar Height set: " + str(stateMach['WindowBarHeight']))

    if (    stateMach['currentMouseButtonState'] == 0 
            or 
            stateMach['mousePressed'] >= 0
            or
            stateMach['drawInfo']
            or
            stateMach['mouseReleased'] >= 0
            ):
        return True
    else:
        return False

# Move window when it is being dragged
def dragWindow():
    if (    stateMach['currentMouseButtonState'] == 0
            and
            stateMach['mouseButton'] == 0
            and
            stateMach['noButtonsPressed'] == True
            and
            type(stateMach['WindowBarHeight']) != str
            and
            stateMach['isFullScreen'] == False
            ):
        pass
        if (type(stateMach['WindowBarHeight']) == type(None)
                ):
            if (stateMach['mousePressed'] == 0):
                stateMach['prevWindowPosX'] = glutGet(GLUT_WINDOW_X)
                stateMach['prevWindowPosY'] = glutGet(GLUT_WINDOW_Y)

            #print(stateMach['WindowBarHeight'], glutGet(GLUT_WINDOW_Y), stateMach['prevWindowPosY'])
            stateMach['WindowBarHeight'] = "noSet"
            tmx = glutGet(GLUT_WINDOW_X)
            tmy = glutGet(GLUT_WINDOW_Y)
        else:
            curPos = stateMach['pynputMouse'].position
            tmx = curPos[0] - stateMach['mousePosAtPressX']
            tmy = curPos[1] - stateMach['mousePosAtPressY'] - stateMach['WindowBarHeight']

        for menu in stateMach['Menus']:
            stateMach['Menus'][menu].close()
            stateMach['Menus'][menu].update(stateMach)
        glutPositionWindow(tmx, tmy)

# function is called when the mouse is being moved while registering button input
def mouseActive(mouseX, mouseY):
    global stateMach

    stateMach['cursorX'] = mouseX
    stateMach['cursorY'] = mouseY

    # Map cursor coordinates to GL coordinates
    w2h = stateMach['w2h']
    if w2h >= 1.0:
        stateMach['cursorXgl'] = mapRanges(mouseX, 0, stateMach['windowDimW'], -w2h, w2h)
        stateMach['cursorYgl'] = mapRanges(mouseY, 0, stateMach['windowDimH'], 1.0, -1.0)
    else:
        stateMach['cursorXgl'] = mapRanges(mouseX, 0, stateMach['windowDimW'], -1.0, 1.0)
        stateMach['cursorYgl'] = mapRanges(mouseY, 0, stateMach['windowDimH'], 1/w2h, -1/w2h)

# function is called when mouse is moving over window with no input
def mousePassive(mouseX, mouseY):
    global stateMach
    stateMach['cursorX'] = mouseX
    stateMach['cursorY'] = mouseY
    #print("quack",glutGetModifiers())

    # Map cursor coordinates to GL coordinates
    w2h = stateMach['w2h']
    if w2h >= 1.0:
        stateMach['cursorXgl'] = mapRanges(mouseX, 0, stateMach['windowDimW'], -w2h, w2h)
        stateMach['cursorYgl'] = mapRanges(mouseY, 0, stateMach['windowDimH'], 1.0, -1.0)
    else:
        stateMach['cursorXgl'] = mapRanges(mouseX, 0, stateMach['windowDimW'], -1.0, 1.0)
        stateMach['cursorYgl'] = mapRanges(mouseY, 0, stateMach['windowDimH'], 1/w2h, -1/w2h)

# function is called when mouse buttons are pressed/held/released
def mouseInteraction(button, state, mouseX, mouseY):
    global stateMach

    stateMach['cursorX'] = mouseX
    stateMach['cursorY'] = mouseY
    #print(glutGetModifiers())
    decodeModifiers(glutGetModifiers())

    # Map cursor coordinates to GL coordinates
    w2h = stateMach['w2h']
    if w2h >= 1.0:
        stateMach['cursorXgl'] = mapRanges(mouseX, 0, stateMach['windowDimW'], -w2h, w2h)
        stateMach['cursorYgl'] = mapRanges(mouseY, 0, stateMach['windowDimH'], 1.0, -1.0)
    else:
        stateMach['cursorXgl'] = mapRanges(mouseX, 0, stateMach['windowDimW'], -1.0, 1.0)
        stateMach['cursorYgl'] = mapRanges(mouseY, 0, stateMach['windowDimH'], 1/w2h, -1/w2h)

    # State = 0: button is pressed, low
    # State = 1: button is released, high
    if (stateMach['currentMouseButtonState'] == 1 and state == 0):
        stateMach['mousePosAtPressX'] = stateMach['cursorX']
        stateMach['mousePosAtPressY'] = stateMach['cursorY']

        stateMach['mousePressed'] = button

    if (stateMach['currentMouseButtonState'] == 0 and state > 0):
        stateMach['noButtonsPressed'] = True
        stateMach['mouseReleased'] = button

    stateMach['currentMouseButtonState'] = state
    if (state == 0):
        stateMach['mouseButton'] = button
    elif (button > 2):
        stateMach['mouseButton'] = button
    else:
        stateMach['mouseButton'] = -1

    #print("state: " + str(state) + ", currentMouseButtonState: " + str(stateMach['currentMouseButtonState']))
    return

# Convert a vector from Cartesian coordinates to Polar coordinates
def vecCart2Pol(vector):
    compx   = vector[0]
    compy   = vector[1]
    mag     = sqrt(pow(compx, 2.0) + pow(compy, 2.0))
    # Prevent divide by 0, account for float sign-bit weirdness
    if (compx == 0.0 or compx == -0.0):
        compx = 0.00000000001

    # Calculate Correct angle based on quadrant vector points to
    if   (compx > 0 and compy >= 0):
        ang = (degrees(atan(compy/compx)))
    elif (compx < 0 and compy >= 0):
        ang = (degrees(atan(compy/compx))+180.0)
    elif (compx < 0 and compy <= 0):
        ang = (degrees(atan(compy/compx))+180.0)
    elif (compx > 0 and compy <= 0):
        ang = (degrees(atan(compy/compx))+360.0)
    else:
        ang = 0.0

    return (ang, mag)

# Calculates the velocity vector of the mouse cursor
def calcCursorVelocity(millis):
    global stateMach

    PxDs = min(stateMach['windowDimW'], stateMach['windowDimH'])

    w2h = stateMach['w2h']

    # Get time since louse mouse callback
    deltaT = float((time.time() - stateMach['tCursor']))

    # Calculate Cursor Velocity Vector
    ccx = mapRanges(stateMach['cursorX'], 0, stateMach['windowDimW'], -w2h, w2h)
    ccy = mapRanges(stateMach['cursorY'], 0, stateMach['windowDimH'], 1.0, -1.0)
    pcx = mapRanges(stateMach['prevCursorX'], 0, stateMach['windowDimW'], -w2h, w2h)
    pcy = mapRanges(stateMach['prevCursorY'], 0, stateMach['windowDimH'], 1.0, -1.0)
    deltaX = ccx-pcx
    deltaY = ccy-pcy

    if deltaT <= 0:
        speedX = 0
        speedY = 0
    else:
        speedX = deltaX/deltaT
        speedY = deltaY/deltaT

    # Convert vector to polar coordinates
    tmv = vecCart2Pol((speedX, speedY))
    ang = tmv[0]
    speed = tmv[1]
    
    # Flip vector due for translation from pixel-coord space to GL-coord space
    if ang != 0.0:
        ang = 360.0 - ang

    # Insert vector magnitude into list of vectors magnitudes, remove oldest element
    tmn  = len(stateMach['prevCurVelMags'])
    stateMach['prevCurVelMags'].insert(0, speed)
    stateMach['prevCurVelMags'] = stateMach['prevCurVelMags'][0:tmn]

    # Compute weighted average of previous values, to smooth out jitter
    tmm = 0.0
    for i in range(tmn):
        tmm += stateMach['prevCurVelMags'][i]*((tmn-i)/tmn)
    speed = tmm/float(tmn)

    stateMach['cursorVelocity'] = (ang, speed)
    stateMach['tCursor'] = time.time()
    stateMach['prevCursorX'] = stateMach['cursorX']
    stateMach['prevCursorY'] = stateMach['cursorY']

    tma = radians(stateMach['cursorVelocity'][0])
    tms = stateMach['cursorVelocity'][1]
    if (abs(tms) <= 0.000001):
        tms = 0.0

    tmnx = speedX
    tmn  = len(stateMach['prevCurVelXs'])
    stateMach['prevCurVelXs'].insert(0, tmnx)
    stateMach['prevCurVelXs'] = stateMach['prevCurVelXs'][0:tmn]

    tmny = speedY
    tmn  = len(stateMach['prevCurVelYs'])
    stateMach['prevCurVelYs'].insert(0, tmny)
    stateMach['prevCurVelYs'] = stateMach['prevCurVelYs'][0:tmn]

    # Smooth out jitter over n frames
    tmnx = 0.0
    for i in range(tmn):
        tmnx += stateMach['prevCurVelXs'][i]*((tmn-i)/tmn)
    tmnx /= float(tmn)

    # Smooth out jitter over n frames
    tmny = 0.0
    for i in range(tmn):
        tmny += stateMach['prevCurVelYs'][i]*((tmn-i)/tmn)
    tmny /= float(tmn)

    stateMach['cursorVelSmoothed']  = (tmnx, tmny)
    stateMach['cursorVelSmoothPol'] = vecCart2Pol((tmnx, tmny))

    return

# Main screen drawing routine
# Passed to glutDisplayFunc()
def display():
    global stateMach

    glClear(GL_COLOR_BUFFER_BIT)# | GL_DEPTH_BUFFER_BIT)

    #stateMach['tDiff'] = 0.70568/stateMach['fps']
    #stateMach['tDiff'] = 1.30568/stateMach['fps']
    stateMach['tDiff'] = 2.0*(2.71828/stateMach['fps'])
    #stateMach['tDiff'] = 3.14159/stateMach['fps']
    #stateMach['tDiff'] = 6.28318/stateMach['fps']

    drawTestObjects = False
    #drawTestObjects = True
    calcCursorVelocity(0)

    if (not drawTestObjects):
        drawElements()

        if (stateMach['targetScreen'] == 0):
            watchHomeInput()
        elif (stateMach['targetScreen'] == 1):
            watchColrSettingInput()

    else:
        drawTest()
        watchTest()
    
    # Update animation speed of UI elements
    for key in stateMach['UIelements']:
        stateMach['UIelements'][key].setTimeSlice(stateMach['tDiff'])
    for key in stateMach['Menus']:
        stateMach['Menus'][key].setTimeSlice(stateMach['tDiff'])

    # Update Colors of Lamps
    for i in range(len(stateMach['lamps'])):
        stateMach['lamps'][i].updateBulbs(stateMach['tDiff']*0.5)

    # Draw info dialog
    if (stateMach['drawInfo']):
        drawInfo(stateMach)

    if (stateMach['currentMouseButtonState'] != 0):
        stateMach['mouseButton'] = -1

    stateMach['keyPressed']     = None
    stateMach['keyReleased']    = None
    stateMach['mousePressed']   = -1
    stateMach['mouseReleased']  = -1
    stateMach['ShiftActive']    = False
    stateMach['CtrlActive']     = False
    stateMach['AltActive']      = False

    # Update UI animation
    for key in stateMach['UIelements']:
        stateMach['UIelements'][key].updateParams()

    # Update Menu States
    for key in stateMach['Menus']:
        stateMach['Menus'][key].update(stateMach)

    stateMach['windowPosX'] = glutGet(GLUT_WINDOW_X)
    stateMach['windowPosY'] = glutGet(GLUT_WINDOW_Y)
    glutSwapBuffers()

def idleWindowOpen():
    framerate()
    glutPostRedisplay()
    plugins.pluginLoader.updatePlugins()
    pass

def idleWindowMinimized():
    framerate()
    plugins.pluginLoader.updatePlugins()
    pass

# Parse byte from glutGetModifiers()
def decodeModifiers(key):
    if (    key < 0
            or 
            key > 7
            or
            type(key) is not int
            ):
        return
    else:
        if (1&key):
            stateMach['ShiftActive'] = True
        if (2&key):
            stateMach['CtrlActive'] = True
        if (4&key):
            stateMach['AltActive'] = True
        return

# change view angle
# Respond to user input from "special" keys
def special(k, x, y):
    global stateMach

    Light = 0

    decodeModifiers(glutGetModifiers())

    if k == GLUT_KEY_LEFT:
        stateMach['textBaseScalar'] -= 0.05
        pass
        #if (len(stateMach['lamps']) > 0):
            #stateMach['lamps'][Light].setAngle(stateMach['lamps'][Light].getAngle() + 5)

    elif k == GLUT_KEY_RIGHT:
        stateMach['textBaseScalar'] += 0.05
        pass
        #if (len(stateMach['lamps']) > 0):
            #stateMach['lamps'][Light].setAngle(stateMach['lamps'][Light].getAngle() - 5)

    elif k == GLUT_KEY_UP:
        stateMach['textGlyphRes'] += 1
        stateMach['textDPIscalar'] = 32.0/float(stateMach['textGlyphRes'])
        makeFont(fontFile="fonts/expressway_regular.ttf", numChars=128, size=stateMach['textGlyphRes'])
        #if (len(stateMach['lamps']) > 0):
            #stateMach['lamps'][Light].setNumBulbs(stateMach['lamps'][Light].getNumBulbs()+1)

    elif k == GLUT_KEY_DOWN:
        stateMach['textGlyphRes'] -= 1
        stateMach['textDPIscalar'] = 32.0/float(stateMach['textGlyphRes'])
        makeFont(fontFile="fonts/expressway_regular.ttf", numChars=128, size=stateMach['textGlyphRes'])
        #if (len(stateMach['lamps']) > 0):
            #stateMach['lamps'][Light].setNumBulbs(stateMach['lamps'][Light].getNumBulbs()-1)

    elif k == GLUT_KEY_F1:
        if stateMach['targetScreen'] == 0:
            stateMach['targetScreen'] = 1
            stateMach['targetBulb'] = 0
        elif stateMach['targetScreen'] == 1:
            stateMach['targetScreen'] = 0

    elif k == GLUT_KEY_F3:
        stateMach['drawInfo'] = not stateMach['drawInfo']

    elif k == GLUT_KEY_F10:
        makeFont()

    elif k == GLUT_KEY_F11:
        if stateMach['isFullScreen'] == False:
            stateMach['prevWindowPosX'] = glutGet(GLUT_WINDOW_X)
            stateMach['prevWindowPosY'] = glutGet(GLUT_WINDOW_Y)-37
            stateMach['prevWindowDimW'] = glutGet(GLUT_WINDOW_WIDTH)
            stateMach['prevWindowDimH'] = glutGet(GLUT_WINDOW_HEIGHT)
            stateMach['isFullScreen'] = True
            glutFullScreen()
        elif stateMach['isFullScreen'] == True:
            glutPositionWindow(stateMach['prevWindowPosX'], stateMach['prevWindowPosY'])
            glutReshapeWindow(stateMach['prevWindowDimW'], stateMach['prevWindowDimH'])
            stateMach['isFullScreen'] = False
    
    if k == GLUT_KEY_F12:
        stateMach['frameLimit'] = not stateMach['frameLimit']
        print("frameLimit is now {}".format("ON" if stateMach['frameLimit'] else "OFF"))

    else:
        return
    glutPostRedisplay()

# callback for anytime a normal key is pressed
def key(ch, x, y):
    global stateMach
    Light = 0
    #print(glutGetModifiers())

    decodeModifiers(glutGetModifiers())
    tmk = ch.decode("ascii")
    stateMach['keyPressed'] == tmk

    if tmk == 'q':
        sys.exit(0)
    if ord(ch) == 27: # ESC
        sys.exit(0)

    if ch == as_8_bit('a'):
        if (len(stateMach['lamps']) > 0):
            if stateMach['lamps'][Light].getArn() == 0:
                stateMach['lamps'][Light].setArn(1)
            elif stateMach['lamps'][Light].getArn() == 1:
                stateMach['lamps'][Light].setArn(0)

    if ch == as_8_bit('p'):
        print(stateMach)
        #filename = "quack.sm"
        #fileObj  = open(filename, 'wb')
        ##pickle.dump(stateMach, fileObj)
        #pickle.dump(stateMach['lamps'][0], fileObj)
        #fileObj.close()

    if ch == as_8_bit('h'):
        goHome()
        #stateMach['wereColorsTouched'] = False
        #stateMach['targetScreen'] = 0

    if ch == as_8_bit(']'):
        stateMach['features'] += 1
        if stateMach['features'] > 4:
            stateMach['features'] = 4

    if ch == as_8_bit('['):
        stateMach['features'] -= 1
        if stateMach['features'] < 0:
            stateMach['features'] = 0

    if ch == as_8_bit('m'):
        glutIconifyWindow()

# new window size or exposure
# this function is called everytime the window is resized
def reshape(width, height):
    global stateMach

    if height > 0:
        stateMach['w2h'] = width/height
    else:
        stateMach['w2h'] = 1
    stateMach['windowDimW'] = width
    stateMach['windowDimH'] = height
    stateMach['textGlyphRes'] = int(height/15)
    stateMach['textDPIscalar'] = 32.0/float(stateMach['textGlyphRes'])
    makeFont(fontFile="fonts/expressway_regular.ttf", numChars=128, size=stateMach['textGlyphRes'])
    glViewport(0, 0, width, height)

# Only Render if the window (any pixel of it at all) is visible
def visible(vis):
    if vis == GLUT_VISIBLE:
        glutIdleFunc(idleWindowOpen)
    else:
        glutIdleFunc(idleWindowMinimized)

# Equivalent to "main()" in C/C++
if __name__ == '__main__':
    global stateMach
    stateMach = {}
    print("Initializing...")
    stateMach['textGlyphRes']       = 32                        # base vertical resolution in pixels of character glyphs
    stateMach['textDPIscalar']      = 1.0                       # scalar for adjusting text resolution with respect to screen resolution
    stateMach['textBaseScalar']     = 1.0                       # scalar for adjusting text size based on user-defined setting
    stateMach['WindowBarHeight']    = None                      # height of the window title bar (calibrated, used for window dragging)
    stateMach['noButtonsPressed']   = True                      
    stateMach['prevHues']           = [None for i in range(6)]
    stateMach['prevSats']           = [None for i in range(6)]
    stateMach['prevVals']           = [None for i in range(6)]
    stateMach['curBulb']            = 0
    stateMach['faceColor']          = (0.3, 0.3, 0.3, 1.0)      # Base (background) color for all UI elements
    stateMach['detailColor']        = (0.9, 0.9, 0.9, 1.0)      # Detail (accent) color for all UI elements
    stateMach['tStart']             = time.time()
    stateMach['t0']                 = time.time()
    stateMach['t1']                 = time.time()
    stateMach['tCursor']            = time.time()
    stateMach['tPhys']              = time.time()
    stateMach['frames']             = 0
    stateMach['lamps']              = []
    stateMach['screen']             = 0
    stateMach['masterSwitch']       = False
    stateMach['fps']                = 60
    stateMach['windowPosX']         = 0                         # current position of the window on the desktop
    stateMach['windowPosY']         = 0                         # current position of the window on the desktop
    stateMach['prevWindowPosX']     = 0                         # previous position of the window on the desktop (for returning from fullscreen)
    stateMach['prevWindowPosY']     = 0                         # previous position of the window on the desktop (for returning from fullscreen)
    stateMach['windowDimW']         = 800                       # current dimensions of the window
    stateMach['windowDimH']         = 480                       # current dimensions of the window
    stateMach['prevWindowDimW']     = 800                       # previous dimensions of the window (for returning from fullscreen)
    stateMach['prevWindowDimH']     = 480                       # previous dimensions of the window (for returning from fullscreen)
    stateMach['cursorXgl']          = 0
    stateMach['cursorYgl']          = 0
    stateMach['cursorX']            = 0
    stateMach['cursorY']            = 0
    stateMach['prevCursorX']        = 0
    stateMach['prevCursorY']        = 0
    stateMach['mousePosAtPressX']   = 0                         # x-coordinate of the mouse cursor when a button was pressed
    stateMach['mousePosAtPressY']   = 0                         # y-coordinate of the mouse cursor when a button was pressed
    stateMach['isFullScreen']       = False                     # whether or not heavenli is running fullscreen
    stateMach['isAnimating']        = False
    stateMach['targetScreen']       = 0                         # target screen to display/watch/animate to
    stateMach['touchState']         = 1                         # indicates if a touchscreen input is active
    stateMach['currentMouseButtonState'] = 1                   # indicates a mouse button is being pressed
    stateMach['prvState']           = stateMach['touchState']   # used for monitoring changes in mouse-button states
    stateMach['targetBulb']         = 0                         # selected bulb for color picker screen, operates on all bulbs if ==numBulbs
    stateMach['frameLimit']         = True                      # whether or not to restrict framerate
    stateMach['someVar']            = 0
    stateMach['someInc']            = 0.1
    stateMach['features']           = 4
    stateMach['numHues']            = 12                        # number of hues to display on color picker screen
    stateMach['currentHue']         = 0
    stateMach['currentSat']         = 0
    stateMach['currentVal']         = 0
    stateMach['prevHue']            = -1
    stateMach['prevVal']            = -1
    stateMach['prevSat']            = -1
    stateMach['wereColorsTouched']  = False                     # used for registering first-input on the color-picker screen
    stateMach['pynputMouse']        = Controller()
    stateMach['mousePressed']       = -1                        # int of the mouse button pressed on the first frame it's registered
    stateMach['mouseReleased']      = -1                        # int of the released mouse button on the first frame it's registered
    stateMach['mouseButton']        = -1                        # int of the mouse button currently being pressed/held
    stateMach['keyPressed']         = None
    stateMach['keyReleased']        = None
    stateMach['keyButton']          = None
    stateMach['ShiftActive']        = False
    stateMach['CtrlActive']         = False
    stateMach['AltActive']          = False
    stateMach['drawInfo']           = False                     # whether or not to draw extra information
    stateMach['cursorVelocity']     = (0.0, 0.0)                # velocity of the mouse cursor
    stateMach['prevCurVelMags']     = [0.0 for i in range(9)]   # used for smoothing out mouse cursor velocity over several frames
    stateMach['prevCurVelXs']       = [0.0 for i in range(9)]   # used for smoothing out mouse cursor velocity over several frames
    stateMach['prevCurVelYs']       = [0.0 for i in range(9)]   # used for smoothing out mouse cursor velocity over several frames
    stateMach['cursorVelSmoothed']  = (0.0, 0.0)                # velocity of the mouse cursor smoothed out over several frames
    stateMach['cursorVelSmoothPol'] = (0.0, 0.0)                # velocity of the mouse cursor smoothed out over several frames in polar coords
    stateMach['BallPosition']       = (0.0, 0.0)
    stateMach['BallVelocity']       = (0.0, 0.0)
    stateMach['UIelements']         = {}                        # Dictionary of objects used for positioning/animating buttons and what-not
    stateMach['Menus']              = {}                        # Dictionary of objects for managing drop/slide menus

    stateMach['testList'] = []
    stateMach['testList'].append((1.0, 1.0, 0.0, 1.0))  # Yellow
    stateMach['testList'].append((1.0, 0.0, 1.0, 1.0))  # Magenta
    stateMach['testList'].append((0.0, 1.0, 1.0, 1.0))  # Cyan
    stateMach['testList'].append((1.0, 0.0, 0.0, 1.0))  # Red
    stateMach['testList'].append((0.0, 0.0, 1.0, 1.0))  # Blue
    stateMach['testList'].append((0.0, 1.0, 0.0, 1.0))  # Green
    #stateMach['testList'].append((0.4, 0.0, 1.0, 1.0))  # Purple
    #stateMach['testList'].append((1.0, 0.4, 0.0, 1.0))  # Orange
    for i in range(0):
        stateMach['testList'].append((
            random.random(),
            random.random(),
            random.random(),
            1.0
            ))

    stateMach['Menus']['testMenu']  = Menu()
    stateMach['Menus']['testMenu'].setIndexDraw(True)
    #stateMach['Menus']['testMenu'].setIndexDraw(False)
    #stateMach['Menus']['testMenu'].setAng(45.0)
    stateMach['Menus']['testMenu'].setAng(0)

    stateMach['Menus']['testMenu'].UIelement.setTarSize(0.15)
    #stateMach['Menus']['testMenu'].UIelement.setTarSize(0.2)
    stateMach['Menus']['testMenu'].UIelement.setAccel(0.125)
    stateMach['Menus']['testMenu'].UIelement.setTarPosX(-0.775)
    stateMach['Menus']['testMenu'].UIelement.setTarPosY(-0.775)
    #stateMach['Menus']['testMenu'].UIelement.setTarPosX(-0.0)
    #stateMach['Menus']['testMenu'].UIelement.setTarPosY(-0.0)
    stateMach['Menus']['testMenu'].setNumElements(len(stateMach['testList']))

    # Setup UI animation objects, initial parameters

    stateMach['UIelements']['HueRing'] = UIelement()
    #stateMach['UIelements']['HueRing'].setTargetFaceColor(stateMach["faceColor"])
    #stateMach['UIelements']['HueRing'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['UIelements']['HueRing'].params["coordX"].setCurve("easeOutSine")
    stateMach['UIelements']['HueRing'].params["coordY"].setCurve("easeOutSine")
    stateMach['UIelements']['HueRing'].params["scaleX"].setCurve("easeInOutQuint")

    stateMach['UIelements']['BackButton'] = UIelement()
    stateMach['UIelements']['BackButton'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['UIelements']['BackButton'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['UIelements']['BackButton'].params["coordX"].setCurve("easeOutCubic")
    stateMach['UIelements']['BackButton'].params["coordY"].setCurve("easeInOutCubic")

    stateMach['UIelements']['BulbButtons'] = UIelement()
    stateMach['UIelements']['BulbButtons'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['UIelements']['BulbButtons'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['UIelements']['BulbButtons'].setTarSize(0.4)
    stateMach['UIelements']['BulbButtons'].setAccel(0.125)

    stateMach['UIelements']['GranChanger'] = UIelement()
    stateMach['UIelements']['GranChanger'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['UIelements']['GranChanger'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['UIelements']['GranChanger'].setTarSize(0.0)
    stateMach['UIelements']['GranChanger'].setTarPosX(0.0)
    stateMach['UIelements']['GranChanger'].setTarPosY(0.0)
    stateMach['UIelements']['GranChanger'].params["coordY"].setCurve("easeOutBack")

    stateMach['UIelements']['MasterSwitch'] = UIelement()
    stateMach['UIelements']['MasterSwitch'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['UIelements']['MasterSwitch'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['UIelements']['MasterSwitch'].setTarSize(1.0)
    stateMach['UIelements']['MasterSwitch'].setAccel(0.125)
    stateMach['UIelements']['MasterSwitch'].setTarPosX(0.0)
    stateMach['UIelements']['MasterSwitch'].setTarPosY(0.0)

    stateMach['UIelements']['AllSetButton'] = UIelement()
    stateMach['UIelements']['AllSetButton'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['UIelements']['AllSetButton'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['UIelements']['AllSetButton'].setTarSize(0.1275)
    stateMach['UIelements']['AllSetButton'].setAccel(0.125)
    stateMach['UIelements']['AllSetButton'].setTarPosX(-0.775)
    stateMach['UIelements']['AllSetButton'].setTarPosY(0.775)

    stateMach['UIelements']['ConfirmButton'] = UIelement()
    stateMach['UIelements']['ConfirmButton'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['UIelements']['ConfirmButton'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['UIelements']['ConfirmButton'].params["coordX"].setCurve("easeOutCubic")
    stateMach['UIelements']['ConfirmButton'].params["coordY"].setCurve("easeInOutCubic")

    stateMach['UIelements']['ColorTriangle'] = UIelement()
    #stateMach['UIelements'][''].setTargetFaceColor(stateMach["faceColor"])
    #stateMach['UIelements'][''].setTargetDetailColor(stateMach["detailColor"])
    stateMach['UIelements']['ColorTriangle'].params["coordX"].setCurve("easeInOutCirc")
    stateMach['UIelements']['ColorTriangle'].params["coordY"].setCurve("easeInOutCirc")
    stateMach['UIelements']['ColorTriangle'].params["scaleX"].setCurve("easeOutCirc")

    plugins.pluginLoader.initPlugins()

    stateMach['lamps'] = plugins.pluginLoader.getAllLamps()

    glutInit(sys.argv)

    # Disable anti-aliasing if running on a Raspberry Pi Zero
    #if (machine() == "armv6l" or machine() == "armv7l"):
        #print("Disabling Antialiasing")
        #glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    #else:
        #glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE)# | GLUT_BORDERLESS)

    # Initialize glut display mode
    GLUT_INIT = GLUT_RGBA | GLUT_DOUBLE

    if "-aa" in sys.argv:
        GLUT_INIT = GLUT_INIT | GLUT_MULTISAMPLE

    if "-borderless" in sys.argv:
        stateMach['WindowBarHeight'] = int(0)
        GLUT_INIT = GLUT_INIT | GLUT_BORDERLESS

    glutInitDisplayMode(GLUT_INIT)

    glutInitWindowPosition(stateMach['windowPosX'], stateMach['windowPosY'])
    glutInitWindowSize(stateMach['windowDimW'], stateMach['windowDimH'])

    glutCreateWindow("HeavenLi")
    glutMouseFunc(mouseInteraction)
    glutMotionFunc(mouseActive)
    glutPassiveMotionFunc(mousePassive)
    glEnable(GL_LINE_SMOOTH)# | GL_DEPTH_TEST)
    #glDepthFunc(GL_EQUAL);

    init()

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutSpecialFunc(special)
    glutKeyboardFunc(key)
    glutVisibilityFunc(visible)

    if "-f" in sys.argv:
        stateMach['isFullScreen'] = True
        glutFullScreen()

    if "-nocursor" in sys.argv:
        glutSetCursor(GLUT_CURSOR_NONE)

    if "-info" in sys.argv:
        print("GL_RENDERER   = ", glGetString(GL_RENDERER))
        print("GL_VERSION    = ", glGetString(GL_VERSION))
        print("GL_VENDOR     = ", glGetString(GL_VENDOR))
        print("GL_SHADING_LANGUAGE_VERSION = ", glGetString(GL_SHADING_LANGUAGE_VERSION))
        glext = str(glGetString(GL_EXTENSIONS))
        print("GL_EXTENSIONS = ", glext.replace(" ", "\n"))

    glutMainLoop()
