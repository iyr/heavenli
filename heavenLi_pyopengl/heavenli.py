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

    stateMach['wx'] = glutGet(GLUT_WINDOW_WIDTH)
    stateMach['wy'] = glutGet(GLUT_WINDOW_HEIGHT)
    stateMach['w2h'] = stateMach['wx']/stateMach['wy']

    print("Initialization Finished")
    return

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
        calcCursorVelocity(0)
        stateMach['t1'] = t

    if t - stateMach['t0'] >= 1.0:
        stateMach['lamps'] = plugins.pluginLoader.getAllLamps()
        #print("%.0f frames in %3.1f seconds = %6.3f FPS" % (stateMach['frames'],seconds,stateMach['fps']))
        stateMach['t0'] = t
        stateMach['frames'] = 0
        ct = datetime.datetime.now()
        stateMach['second'] = ct.second
        stateMach['minute'] = ct.minute + stateMach['second']/60
        stateMach['hour'] = ct.hour + stateMach['minute']/60
        if (stateMach['hour'] > 11):
            stateMach['hour'] -= 12

        #if (machine() == "armv6l" or machine() == "armv7l"):
            #pass
            #glutSetCursor(GLUT_CURSOR_NONE)

    if stateMach['frameLimit'] and (stateMach['fps'] > 66):
        pass
        time.sleep(0.015)

    return

# Function for testing drawcode
def drawTest():
    try:
        w2h = stateMach['w2h']

        tmc = ( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 1.0)
        drawEllipse(
                stateMach['BallPosition'][0]/w2h,
                stateMach['BallPosition'][1],
                0.1, 0.1,
                w2h,
                (0.42, 0.0, 0.85, 1.0)
                )

        # Update Ball position based on its cartesian velocity vector
        tmx = stateMach['BallPosition'][0]
        tmy = stateMach['BallPosition'][1]
        tmx += stateMach['BallVelocity'][0]*0.2
        tmy += stateMach['BallVelocity'][1]*0.2

        stateMach['BallPosition'] = (tmx, tmy)

        # Reverse Ball's Velocity component(s) if it hits edge of screen
        tmx = stateMach['BallVelocity'][0]
        tmy = stateMach['BallVelocity'][1]
        if (stateMach['BallPosition'][0] < -w2h):
            tmx = abs(stateMach['BallVelocity'][0])
        if (stateMach['BallPosition'][0] > w2h):
            tmx = -abs(stateMach['BallVelocity'][0])
        if (stateMach['BallPosition'][1] < -1.0):
            tmy = abs(stateMach['BallVelocity'][1])
        if (stateMach['BallPosition'][1] > 1.0):
            tmy = -abs(stateMach['BallVelocity'][1])

        # Reduce Ball's velocity over time to simulate drag
        cDrag = 0.99
        stateMach['BallVelocity'] = (tmx*cDrag, tmy*cDrag)

        red = (1.0, 0.0, 0.0, 1.0)
        blu = (0.0, 0.0, 1.0, 1.0)
        tmx = mapRanges(stateMach['cursorX'], 0, stateMach['windowDimW'], -w2h, w2h)
        tmy = mapRanges(stateMach['cursorY'], 0, stateMach['windowDimH'], 1.0, -1.0)

        drawPill(
                tmx,
                tmy,
                tmx + stateMach['cursorVelSmoothed'][0],
                tmy + stateMach['cursorVelSmoothed'][1],
                0.02, 
                w2h, 
                blu,
                red 
                )
        pass
    except Exception as OOF:
        print(traceback.format_exc())
        print("Error:", OOF)
    return

def watchTest():
    try:
        w2h = stateMach['w2h']
        if (watchScreen()):
            tmx = mapRanges(stateMach['cursorX'], 0, stateMach['windowDimW'], -w2h, w2h)
            tmy = mapRanges(stateMach['cursorY'], 0, stateMach['windowDimH'], 1.0, -1.0)
            stateMach['BallPosition'] = (tmx, tmy)
            pass
        if (stateMach['mouseReleased']):
            stateMach['BallVelocity'] = stateMach['cursorVelSmoothed']

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

    try:
        # Draw Background
        if (len(stateMach['lamps']) > 0):

            if (stateMach['lamps'][Light].getArn() == 0):
                pass
                drawHomeCircle(0.0, 0.0, 
                        stateMach['wx'], stateMach['wy'], 
                        stateMach['lamps'][Light].getNumBulbs(), 
                        stateMach['lamps'][Light].getAngle(), 
                        stateMach['w2h'],
                        stateMach['lamps'][Light].getBulbsCurrentRGB()
                        )

            if (stateMach['lamps'][Light].getArn() == 1):
                pass
                drawHomeLinear(0.0, 0.0, 
                        stateMach['wx'], stateMach['wy'],
                        stateMach['lamps'][Light].getNumBulbs(), 
                        stateMach['lamps'][Light].getAngle(), 
                        stateMach['w2h'],
                        stateMach['lamps'][Light].getBulbsCurrentRGB()
                        )

        # Draw Granularity Rocker Underneath Clock
        if (stateMach['w2h'] <= 1.0):
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
                    stateMach['w2h'],
                    stateMach['UIelements']['GranChanger'].getSize(),
                    stateMach['tDiff']
                    )
    
        # Draw Clock
        drawClock(
            stateMach['UIelements']['MasterSwitch'].getPosX(),            # X-Coordinate of position
                stateMach['UIelements']['MasterSwitch'].getPosY(),        # Y-Coordinate of position
                stateMach['hour'],                          # Hour to be displayed
                stateMach['minute'],                        # Minute to be displayed
                stateMach['UIelements']['MasterSwitch'].getSize(),        # Size of Clock
                stateMach['w2h'],                           # w2h for proper aspect ratio scaling
                stateMach['UIelements']['MasterSwitch'].getFaceColor(),   # color of the clock face
                stateMach['UIelements']['MasterSwitch'].getDetailColor()  # color of the clock hands
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
                    stateMach['w2h'],
                    stateMach['UIelements']['GranChanger'].getSize(),
                    stateMach['tDiff']
                    )

        if (len(stateMach['lamps']) > 0):
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
                        stateMach['w2h']
                        )

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
                        stateMach['w2h'],
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
                    stateMach['w2h'], 
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
                    stateMach['w2h'], 
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
                    stateMach['w2h'], 
                    fcc, 
                    extraColor, 
                    dtc
                    )

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
                    stateMach['w2h'], 
                    fcc, 
                    extraColor, 
                    dtc
                    )

        #tmc = ( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 1.0)
    except Exception as OOF:
        print(traceback.format_exc())
        print("Error:", OOF)
    return

# Watch Home Screen for inputs
def watchHomeInput():
    global stateMach

    # Placeholder variable
    Light = 0

    # Watch Home Screen for input
    if (watchScreen()):
    
        # Watch Clock for input
        if (watchDot(
            stateMach['wx'],                            # Middle of Screen
            stateMach['wy'],                            # Middle of Screen
            min(stateMach['wx'], stateMach['wy'])/2)    # Clock Radius
            and
            stateMach['mousePressed']):                 # Button Must be clicked
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

                posX = mapRanges(stateMach['bulbButtons'][i][0], -stateMach['w2h'], stateMach['w2h'], 0, stateMach['wx']*2)# X coord of button
                posY = mapRanges(stateMach['bulbButtons'][i][1],               1.0,             -1.0, 0, stateMach['wy']*2)# Y coord of button

                if (watchDot(
                    posX, posY,
                    min(stateMach['wx'], stateMach['wy'])*0.5*0.3)                                      # Button Radius
                    and
                    len(stateMach['lamps']) > 0
                    and
                    stateMach['mousePressed']): # Button Must be clicked

                    # Set Color Picker as target Screen selecting bulb i
                    stateMach['targetScreen'] = 1
                    stateMach['targetBulb'] = i
                    stateMach['UIelements']['MasterSwitch'].setTarSize(1.75)
                    stateMach['UIelements']['MasterSwitch'].setTargetFaceColor((0.0, 0.0, 0.0, 1.0))
                    stateMach['UIelements']['MasterSwitch'].setTargetDetailColor((0.0, 0.0, 0.0, 0.0))
                    stateMach['UIelements']['AllSetButton'].setTarSize(0.0)
                    stateMach['UIelements']['BulbButtons'].setTarSize(0.0)

                    stateMach['UIelements']['ColorTriangle'].setTarSize(1.0)
                    stateMach['UIelements']['ColorTriangle'].setValue("coordX", stateMach['bulbButtons'][i][0]/stateMach['w2h'])
                    stateMach['UIelements']['ColorTriangle'].setValue("coordY", stateMach['bulbButtons'][i][1])
                    stateMach['UIelements']['ColorTriangle'].setTarPosX(0.0)
                    stateMach['UIelements']['ColorTriangle'].setTarPosY(0.0)

                    stateMach['UIelements']['HueRing'].setSize(0.0)
                    stateMach['UIelements']['HueRing'].setTarSize(1.0)
                    stateMach['UIelements']['HueRing'].setValue("coordX", stateMach['bulbButtons'][i][0]/stateMach['w2h'])
                    stateMach['UIelements']['HueRing'].setValue("coordY", stateMach['bulbButtons'][i][1])
                    stateMach['UIelements']['HueRing'].setTarPosX(0.0)
                    stateMach['UIelements']['HueRing'].setTarPosY(0.0)

                    stateMach['UIelements']['BackButton'].setTarSize(0.2)
                    stateMach['UIelements']['BackButton'].setValue("coordX", stateMach['bulbButtons'][i][0]/stateMach['w2h'])
                    stateMach['UIelements']['BackButton'].setValue("coordY", stateMach['bulbButtons'][i][1])
                    stateMach['UIelements']['BackButton'].setTarPosX(-0.75)
                    stateMach['UIelements']['BackButton'].setTarPosY(-0.75)

                    stateMach['UIelements']['ConfirmButton'].setTarSize( 0.2)
                    stateMach['UIelements']['ConfirmButton'].setValue("coordX", stateMach['bulbButtons'][i][0]/stateMach['w2h'])
                    stateMach['UIelements']['ConfirmButton'].setValue("coordY", stateMach['bulbButtons'][i][1])
                    stateMach['UIelements']['ConfirmButton'].setTarPosX( 0.75)
                    stateMach['UIelements']['ConfirmButton'].setTarPosY(-0.75)

                    stateMach['UIelements']['GranChanger'].setTarSize(0.3)
                    stateMach['UIelements']['GranChanger'].setTarPosY(-0.91)

                    # Record previous color(s)
                    stateMach['prevHue'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[0]
                    stateMach['prevSat'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[1]
                    stateMach['prevVal'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[2]

            # Watch all-set for input
            if (watchDot(
                mapRanges(stateMach['UIelements']['AllSetButton'].getTarPosX(), -1.0,  1.0, 0, stateMach['wx']*2),  # X coord of button
                mapRanges(stateMach['UIelements']['AllSetButton'].getTarPosY(),  1.0, -1.0, 0, stateMach['wy']*2),  # Y coord of button
                min(stateMach['wx'], stateMach['wy'])*0.2)          # Button Radius
                and
                len(stateMach['lamps']) > 0
                and
                stateMach['mousePressed']): # Button Must be clicked

                # Set Color Picker as target Screen selecting bulb all bulbs
                stateMach['targetScreen'] = 1
                stateMach['targetBulb'] = stateMach["lamps"][Light].getNumBulbs()
                stateMach['UIelements']['MasterSwitch'].setTarSize(1.75)
                stateMach['UIelements']['MasterSwitch'].setTargetFaceColor((0.0, 0.0, 0.0, 1.0))
                stateMach['UIelements']['MasterSwitch'].setTargetDetailColor((0.0, 0.0, 0.0, 0.0))
                stateMach['UIelements']['AllSetButton'].setTarSize(0.0)
                stateMach['UIelements']['BulbButtons'].setTarSize(0.0)

                stateMach['UIelements']['ColorTriangle'].setTarSize(1.0)
                stateMach['UIelements']['ColorTriangle'].setValue("coordX", stateMach['UIelements']['AllSetButton'].getPosX())
                stateMach['UIelements']['ColorTriangle'].setValue("coordY", stateMach['UIelements']['AllSetButton'].getPosY())
                stateMach['UIelements']['ColorTriangle'].setTarPosX(0.0)
                stateMach['UIelements']['ColorTriangle'].setTarPosY(0.0)

                stateMach['UIelements']['HueRing'].setSize(0.0)
                stateMach['UIelements']['HueRing'].setTarSize(1.0)
                stateMach['UIelements']['HueRing'].setValue("coordX", stateMach['UIelements']['AllSetButton'].getPosX())
                stateMach['UIelements']['HueRing'].setValue("coordY", stateMach['UIelements']['AllSetButton'].getPosY())
                stateMach['UIelements']['HueRing'].setValue("scaleX", 0.0)
                stateMach['UIelements']['HueRing'].setTarPosX(0.0)
                stateMach['UIelements']['HueRing'].setTarPosY(0.0)

                stateMach['UIelements']['BackButton'].setTarSize(0.2)
                stateMach['UIelements']['BackButton'].setValue("coordX", stateMach['UIelements']['AllSetButton'].getPosX()/stateMach['w2h'])
                stateMach['UIelements']['BackButton'].setValue("coordY", stateMach['UIelements']['AllSetButton'].getPosY())
                stateMach['UIelements']['BackButton'].setTarPosX(-0.75)
                stateMach['UIelements']['BackButton'].setTarPosY(-0.75)

                stateMach['UIelements']['ConfirmButton'].setTarSize( 0.2)
                stateMach['UIelements']['ConfirmButton'].setValue("coordX", stateMach['UIelements']['AllSetButton'].getPosX()/stateMach['w2h'])
                stateMach['UIelements']['ConfirmButton'].setValue("coordY", stateMach['UIelements']['AllSetButton'].getPosY())
                stateMach['UIelements']['ConfirmButton'].setTarPosX( 0.75)
                stateMach['UIelements']['ConfirmButton'].setTarPosY(-0.75)

                stateMach['UIelements']['GranChanger'].setTarSize(0.3)
                stateMach['UIelements']['GranChanger'].setTarPosY(-0.91)

                # Record previous color(s)
                stateMach['prevHue'] = stateMach['lamps'][Light].getBulbCurrentHSV(0)[0]
                stateMach['prevSat'] = stateMach['lamps'][Light].getBulbCurrentHSV(0)[1]
                stateMach['prevVal'] = stateMach['lamps'][Light].getBulbCurrentHSV(0)[2]
                for i in range(stateMach['targetBulb']):
                    stateMach['prevHues'][i] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[0]
                    stateMach['prevSats'][i] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[1]
                    stateMach['prevVals'][i] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[2]

def watchColrSettingInput():
    global stateMach

    Light = 0

    # Watch Color Picker Screen for input
    if (watchScreen()):

        gctmx = stateMach['UIelements']['GranChanger'].getTarSize()*(24.0/36.0)
        tmr = min(stateMach['wx'], stateMach['wy']*(12.0/36.0)*0.3)
        if (stateMach['w2h'] <= 1.0):
            gctmy = stateMach['UIelements']['GranChanger'].getPosY()*stateMach['w2h']
        else:
            gctmy = stateMach['UIelements']['GranChanger'].getPosY()

        if (stateMach['w2h'] <= 1.0):
            gctmx *= stateMach['w2h']
            tmr *= stateMach['w2h']

        # Watch Granularity Rocker for Input
        if (watchDot(
            mapRanges(gctmx, -stateMach['w2h'], stateMach['w2h'], 0, stateMach['wx']*2),
            mapRanges(gctmy, 1.0, -1.0, 0, stateMach['wy']*2),
            tmr)
            and
            stateMach['mousePressed']):
                stateMach['numHues'] += 2
                if stateMach['numHues'] > 14:
                    stateMach['numHues'] = 14

        # Watch Granularity Rocker for Input
        if (watchDot(
            mapRanges(-gctmx, -stateMach['w2h'], stateMach['w2h'], 0, stateMach['wx']*2),
            mapRanges(gctmy, 1.0, -1.0, 0, stateMach['wy']*2),
            tmr)
            and
            stateMach['mousePressed']):
                stateMach['numHues'] -= 2
                if stateMach['numHues'] < 10:
                    stateMach['numHues'] = 10

        # Watch Hue Ring for Input
        for i in range(len(stateMach['hueButtons'])):
            tmr = 1.0
            if (stateMach['w2h'] <= 1.0):
                stateMach['hueButtons'][i] = (
                        stateMach['hueButtons'][i][0]*stateMach['w2h'], 
                        stateMach['hueButtons'][i][1]*stateMach['w2h'], 
                        stateMach['hueButtons'][i][2])
                tmr = stateMach['w2h']

            if (watchDot(
                mapRanges(stateMach['hueButtons'][i][0], -1.0*stateMach['w2h'], 1.0*stateMach['w2h'], 0, stateMach['wx']*2),
                mapRanges(stateMach['hueButtons'][i][1], 1.0, -1.0, 0, stateMach['wy']*2),
                min(stateMach['wx'], stateMach['wy'])*0.15*(12.0/float(stateMach['numHues'])))
                and
                stateMach['mousePressed']):

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
            tmr = 1.0
            if (stateMach['w2h'] <= 1.0):
                stateMach['satValButtons'][i] = (
                        stateMach['satValButtons'][i][0]*stateMach['w2h'],     # X-Coord of Button
                        stateMach['satValButtons'][i][1]*stateMach['w2h'],     # Y-Coord of Button
                        stateMach['satValButtons'][i][2],                    # Saturation of Button
                        stateMach['satValButtons'][i][3])                    # Value of Button
                tmr = stateMach['w2h']

            # Map relative x-Coord to screen x-Coord
            if (watchDot(
                mapRanges(stateMach['satValButtons'][i][0], -1.0*stateMach['w2h'], 1.0*stateMach['w2h'], 0, stateMach['wx']*2),
                mapRanges(stateMach['satValButtons'][i][1], 1.0, -1.0, 0, stateMach['wy']*2),
                min(stateMach['wx'], stateMach['wy'])*0.073)):

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
        if (watchDot(
        mapRanges(stateMach['UIelements']['ConfirmButton'].getPosX(), -1.0,  1.0, 0, stateMach['wx']*2),
        mapRanges(stateMach['UIelements']['ConfirmButton'].getPosY(),  1.0, -1.0, 0, stateMach['wy']*2),
        min(stateMach['wx'], stateMach['wy'])*0.2)
        and
        stateMach['mousePressed']):
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
        if (watchDot(
        mapRanges(stateMach['UIelements']['BackButton'].getPosX(), -1.0,  1.0, 0, stateMach['wx']*2),
        mapRanges(stateMach['UIelements']['BackButton'].getPosY(),  1.0, -1.0, 0, stateMach['wy']*2),
        min(stateMach['wx'], stateMach['wy'])*0.2)
        and
        stateMach['mousePressed']):
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

            stateMach['UIelements']['MasterSwitch'].setTarSize(1.0)
            stateMach['UIelements']['MasterSwitch'].setTargetFaceColor(stateMach['faceColor'])
            stateMach['UIelements']['MasterSwitch'].setTargetDetailColor(stateMach['detailColor'])
            stateMach['UIelements']['ColorTriangle'].setTarSize(0.0)
            stateMach['UIelements']['HueRing'].setTarSize(10.0)
            stateMach['targetScreen'] = 0
            stateMach['UIelements']['BulbButtons'].setTarSize(0.4)
            stateMach['UIelements']['AllSetButton'].setTarSize(0.1275)
            stateMach['UIelements']['BackButton'].setTarSize(0.0)
            stateMach['UIelements']['ConfirmButton'].setTarSize(0.0)
            stateMach['UIelements']['GranChanger'].setTarSize(0.0)
            stateMach['UIelements']['GranChanger'].setTarPosY(0.0)
            stateMach['UIelements']['GranChanger'].setAccel(0.33)


# Check if user is clicking in circle
def watchDot(px, py, pr):
    global stateMach
    if (1.0 >= pow((stateMach['cursorX']-px/2), 2) / pow(pr/2, 2) + pow((stateMach['cursorY']-py/2), 2) / pow(pr/2, 2)):
        return True
    else:
        return False

# Used to process user input
def watchScreen():
    global stateMach
    if (stateMach['currentState'] == 0 or stateMach['mousePressed']):
        return True
    else:
        return False

def mouseActive(mouseX, mouseY):
    global stateMach

    stateMach['cursorX'] = mouseX
    stateMach['cursorY'] = mouseY

def mousePassive(mouseX, mouseY):
    global stateMach
    stateMach['cursorX'] = mouseX
    stateMach['cursorY'] = mouseY

def mouseInteraction(button, state, mouseX, mouseY):
    global stateMach

    stateMach['cursorX'] = mouseX
    stateMach['cursorY'] = mouseY

    # State = 0: button is pressed, low
    # State = 1: button is released, high
    if (stateMach['currentState'] == 1 and state == 0):
        stateMach['mousePressed'] = True

    if (stateMach['currentState'] == 0 and state > 0):
        stateMach['mouseReleased'] = True

    stateMach['currentState'] = state
    if (state == 0):
        stateMach['mouseButton'] = button
    elif (button > 2):
        stateMach['mouseButton'] = button
    else:
        stateMach['mouseButton'] = "None"

    #print("state: " + str(state) + ", currentState: " + str(stateMach['currentState']))
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

    # Get time since louse mouse callback
    deltaT = float((time.time() - stateMach['tCursor'])*1000.0)

    # Calculate Cursor Velocity Vector
    deltaX  = float(stateMach['cursorX'] - stateMach['prevCursorX'])
    deltaY  = float(stateMach['cursorY'] - stateMach['prevCursorY'])
    speedX  = deltaX/deltaT
    speedY  = deltaY/deltaT

    # Convert vector to polar coordinates
    tmv = vecCart2Pol((speedX, speedY))
    ang = tmv[0]
    speed = tmv[1]
    
    # Flip vector due for translation from pixel-coord space to GL-coord space
    if ang != 0:
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
    tms = stateMach['cursorVelocity'][1]*2.0
    if (tms == 0.0):
        tms = 0.000000000001

    tmnx = cos(tma)*tms
    tmn  = len(stateMach['prevCurVelXs'])
    stateMach['prevCurVelXs'].insert(0, tmnx)
    stateMach['prevCurVelXs'] = stateMach['prevCurVelXs'][0:tmn]

    tmny = sin(tma)*tms
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

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #stateMach['tDiff'] = 0.70568/stateMach['fps']
    #stateMach['tDiff'] = 1.30568/stateMach['fps']
    stateMach['tDiff'] = 2.0*(2.71828/stateMach['fps'])
    #stateMach['tDiff'] = 3.14159/stateMach['fps']
    #stateMach['tDiff'] = 6.28318/stateMach['fps']

    drawTestObjects = False

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

    # Update Colors of Lamps
    for i in range(len(stateMach['lamps'])):
        stateMach['lamps'][i].updateBulbs(stateMach['tDiff']*0.5)

    # Draw info dialog
    if (stateMach['drawInfo']):
        drawInfo(stateMach)

    stateMach['mousePressed'] = False
    stateMach['mouseReleased'] = False

    # Update UI animation
    for key in stateMach['UIelements']:
        stateMach['UIelements'][key].updateParams()

    glutSwapBuffers()
    plugins.pluginLoader.updatePlugins()
    #glutTimerFunc(1, calcCursorVelocity, 0)

def idleWindowOpen():
    framerate()
    glutPostRedisplay()
    pass

def idleWindowMinimized():
    framerate()
    pass

# change view angle
# Respond to user input from "special" keys
def special(k, x, y):
    global stateMach

    Light = 0

    if k == GLUT_KEY_LEFT:
        if (len(stateMach['lamps']) > 0):
            stateMach['lamps'][Light].setAngle(stateMach['lamps'][Light].getAngle() + 5)

    elif k == GLUT_KEY_RIGHT:
        if (len(stateMach['lamps']) > 0):
            stateMach['lamps'][Light].setAngle(stateMach['lamps'][Light].getAngle() - 5)

    elif k == GLUT_KEY_UP:
        if (len(stateMach['lamps']) > 0):
            stateMach['lamps'][Light].setNumBulbs(stateMach['lamps'][Light].getNumBulbs()+1)

    elif k == GLUT_KEY_DOWN:
        if (len(stateMach['lamps']) > 0):
            stateMach['lamps'][Light].setNumBulbs(stateMach['lamps'][Light].getNumBulbs()-1)

    elif k == GLUT_KEY_F1:
        if stateMach['targetScreen'] == 0:
            stateMach['targetScreen'] = 1
            stateMach['targetBulb'] = 0
        elif stateMach['targetScreen'] == 1:
            stateMach['targetScreen'] = 0

    elif k == GLUT_KEY_F3:
        stateMach['drawInfo'] = not stateMach['drawInfo']

    elif k == GLUT_KEY_F11:
        if stateMach['isFullScreen'] == False:
            stateMach['windowPosX'] = glutGet(GLUT_WINDOW_X)
            stateMach['windowPosY'] = glutGet(GLUT_WINDOW_Y)
            stateMach['windowDimW'] = glutGet(GLUT_WINDOW_WIDTH)
            stateMach['windowDimH'] = glutGet(GLUT_WINDOW_HEIGHT)
            stateMach['isFullScreen'] = True
            glutFullScreen()
        elif stateMach['isFullScreen'] == True:
            glutPositionWindow(stateMach['windowPosX'], stateMach['windowPosY'])
            glutReshapeWindow(stateMach['windowDimW'], stateMach['windowDimH'])
            stateMach['isFullScreen'] = False
    
    if k == GLUT_KEY_F12:
        stateMach['frameLimit'] = not stateMach['frameLimit']
        print("frameLimit is now {}".format("ON" if stateMach['frameLimit'] else "OFF"))

    else:
        return
    glutPostRedisplay()

def key(ch, x, y):
    global stateMach
    Light = 0
    if ch == as_8_bit('q'):
        sys.exit(0)
    if ord(ch) == 27: # ESC
        sys.exit(0)

    if ch == as_8_bit('a'):
        if (len(stateMach['lamps']) > 0):
            if stateMach['lamps'][Light].getArn() == 0:
                stateMach['lamps'][Light].setArn(1)
            elif stateMach['lamps'][Light].getArn() == 1:
                stateMach['lamps'][Light].setArn(0)

    if ch == as_8_bit('h'):
        stateMach['wereColorsTouched'] = False
        stateMach['targetScreen'] = 0

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
    stateMach['wx'] = width
    stateMach['wy'] = height
    stateMach['windowDimW'] = width
    stateMach['windowDimH'] = height
    stateMach['windowPosX'] = glutGet(GLUT_WINDOW_X)
    stateMach['windowPosY'] = glutGet(GLUT_WINDOW_Y)
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
    stateMach['prevHues']           = [None for i in range(6)]
    stateMach['prevSats']           = [None for i in range(6)]
    stateMach['prevVals']           = [None for i in range(6)]
    stateMach['curBulb']            = 0
    stateMach['faceColor']          = (0.3, 0.3, 0.3, 1.0)
    stateMach['detailColor']        = (0.9, 0.9, 0.9, 1.0)
    stateMach['tStart']             = time.time()
    stateMach['t0']                 = time.time()
    stateMach['t1']                 = time.time()
    stateMach['tCursor']            = time.time()
    stateMach['frames']             = 0
    stateMach['lamps']              = []
    stateMach['screen']             = 0
    stateMach['masterSwitch']       = False
    stateMach['fps']                = 60
    stateMach['windowPosX']         = 0
    stateMach['windowPosY']         = 0
    stateMach['windowDimW']         = 800
    stateMach['windowDimH']         = 480
    stateMach['cursorX']            = 0
    stateMach['cursorY']            = 0
    stateMach['prevCursorX']        = 0
    stateMach['prevCursorY']        = 0
    stateMach['isFullScreen']       = False
    stateMach['isAnimating']        = False
    stateMach['wx']                 = 0
    stateMach['wy']                 = 0
    stateMach['targetScreen']       = 0
    stateMach['touchState']         = 1
    stateMach['currentState']       = 1
    stateMach['prvState']           = stateMach['touchState']
    stateMach['targetBulb']         = 0
    stateMach['frameLimit']         = True
    stateMach['someVar']            = 0
    stateMach['someInc']            = 0.1
    stateMach['features']           = 4
    stateMach['numHues']            = 12
    stateMach['currentHue']         = 0
    stateMach['currentSat']         = 0
    stateMach['currentVal']         = 0
    stateMach['prevHue']            = -1
    stateMach['prevVal']            = -1
    stateMach['prevSat']            = -1
    stateMach['wereColorsTouched']  = False
    stateMach['mousePressed']       = False
    stateMach['mouseReleased']      = False
    stateMach['mouseButton']        = "None"
    stateMach['drawInfo']           = False
    stateMach['cursorVelocity']     = (0.0, 0.0)
    stateMach['prevCurVelMags']     = [0.0 for i in range(9)]
    stateMach['prevCurVelXs']       = [0.0 for i in range(9)]
    stateMach['prevCurVelYs']       = [0.0 for i in range(9)]
    stateMach['cursorVelSmoothed']  = (0.0, 0.0)
    stateMach['cursorVelSmoothPol'] = (0.0, 0.0)
    stateMach['BallPosition']       = (0.0, 0.0)
    stateMach['BallVelocity']       = (0.0, 0.0)

    # Setup UI animation objects, initial parameters
    stateMach['UIelements']         = {}

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
    if (machine() == "armv6l" or machine() == "armv7l"):
        print("Disabling Antialiasing")
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    else:
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE)

    glutInitWindowPosition(stateMach['windowPosX'], stateMach['windowPosY'])
    glutInitWindowSize(stateMach['windowDimW'], stateMach['windowDimH'])

    glutCreateWindow("HeavenLi")
    glutMouseFunc(mouseInteraction)
    glutMotionFunc(mouseActive)
    glutPassiveMotionFunc(mousePassive)
    glEnable(GL_LINE_SMOOTH)

    init()

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutSpecialFunc(special)
    glutKeyboardFunc(key)
    glutVisibilityFunc(visible)

    if "-info" in sys.argv:
        print("GL_RENDERER   = ", glGetString(GL_RENDERER))
        print("GL_VERSION    = ", glGetString(GL_VERSION))
        print("GL_VENDOR     = ", glGetString(GL_VENDOR))
        print("GL_EXTENSIONS = ", glGetString(GL_EXTENSIONS))
        print("GL_SHADING_LANGUAGE_VERSION = ", glGetString(GL_SHADING_LANGUAGE_VERSION))

    glutMainLoop()
