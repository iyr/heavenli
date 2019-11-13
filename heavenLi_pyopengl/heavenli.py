#!/usr/bin/python3

print("Now Loading...")
#import hliUtilImports
from hliImports import *
from lampClass import *
print("All Imports Loaded...")

#import profile

def init():
    global stateMach
    stateMach['curBulb'] = 0
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
        print("Too Fast, Too Quick!!")

    if t - stateMach['t1'] >= (1/60.0):
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
                        stateMach['lamps'][Light].getBulbsCurrentRGB());

            if (stateMach['lamps'][Light].getArn() == 1):
                pass
                drawHomeLinear(0.0, 0.0, 
                        stateMach['wx'], stateMach['wy'],
                        stateMach['lamps'][Light].getNumBulbs(), 
                        stateMach['lamps'][Light].getAngle(), 
                        stateMach['w2h'],
                        stateMach['lamps'][Light].getBulbsCurrentRGB());

        # Draw Granularity Rocker Underneath Clock
        if (stateMach['w2h'] <= 1.0):
            gctmy = stateMach['GranChanger'].getPosY()*stateMach['w2h']
        else:
            gctmy = stateMach['GranChanger'].getPosY()

        limit = -0.90
        if (stateMach['GranChanger'].getPosY() >= limit):
            drawGranChanger(
                    stateMach['GranChanger'].getPosX(),
                    gctmy,
                    stateMach['GranChanger'].getFaceColor(),
                    stateMach['GranChanger'].getDetailColor(),
                    stateMach['numHues'],
                    0.0,
                    stateMach['w2h'],
                    stateMach['GranChanger'].getSize(),
                    stateMach['tDiff'])
    
        # Draw Clock
        drawClock(
            stateMach['MasterSwitch'].getPosX(),            # X-Coordinate of position
                stateMach['MasterSwitch'].getPosY(),        # Y-Coordinate of position
                stateMach['hour'],                          # Hour to be displayed
                stateMach['minute'],                        # Minute to be displayed
                stateMach['MasterSwitch'].getSize(),        # Size of Clock
                stateMach['w2h'],                           # w2h for proper aspect ratio scaling
                stateMach['MasterSwitch'].getFaceColor(),   # color of the clock face
                stateMach['MasterSwitch'].getDetailColor()  # color of the clock hands
                )

        # Draw Granularity Rocker on top of Clock
        if (stateMach['GranChanger'].getPosY() < limit):
            drawGranChanger(
                    stateMach['GranChanger'].getPosX(),
                    gctmy,
                    stateMach['GranChanger'].getFaceColor(),
                    stateMach['GranChanger'].getDetailColor(),
                    stateMach['numHues'],
                    0.0,
                    stateMach['w2h'],
                    stateMach['GranChanger'].getSize(),
                    stateMach['tDiff'])

        if (len(stateMach['lamps']) > 0):
            if (stateMach['BulbButtons'].isVisible()):
                stateMach['bulbButtons'] = drawBulbButton(
                        stateMach['lamps'][Light].getArn(),
                        stateMach['lamps'][Light].getNumBulbs(),
                        60,
                        stateMach['lamps'][Light].getAngle(),
                        stateMach['BulbButtons'].getSize(),
                        stateMach['BulbButtons'].getFaceColor(),
                        stateMach['BulbButtons'].getDetailColor(),
                        stateMach['lamps'][Light].getBulbsCurrentRGB(),
                        stateMach['w2h'])

            if (stateMach['AllSetButton'].isVisible()):
                drawIcon(
                        stateMach['AllSetButton'].getPosX(), 
                        stateMach['AllSetButton'].getPosY(), 
                        stateMach['AllSetButton'].getSize(), 
                        stateMach['lamps'][Light].getArn(),
                        stateMach['lamps'][Light].getAlias(),
                        stateMach['features'],
                        stateMach['AllSetButton'].getFaceColor(),
                        stateMach['AllSetButton'].getDetailColor(),
                        stateMach['lamps'][Light].getNumBulbs(),
                        stateMach['lamps'][Light].getAngle(),
                        stateMach['w2h'],
                        stateMach['lamps'][Light].getBulbsCurrentRGB())

        if (stateMach['wereColorsTouched']):
            selectRingColor = dtc
        else:
            selectRingColor = fcc

        # Draw Ring of Dots with different hues
        if (stateMach['HueRing'].isVisible()):
            stateMach['hueButtons'] = drawHueRing(
                    stateMach['HueRing'].getPosX(),
                    stateMach['HueRing'].getPosY(),
                    stateMach['HueRing'].getSize(), 
                    stateMach['currentHue'], 
                    stateMach['numHues'], 
                    selectRingColor, 
                    stateMach['w2h'], 
                    stateMach['tDiff'],
                    stateMach['interactionCursor'])

        # Draw Triangle of Dots with different brightness/saturation
        if (stateMach['ColorTriangle'].isVisible()):
            stateMach['satValButtons'] = drawColrTri(
                    stateMach['ColorTriangle'].getPosX(),
                    stateMach['ColorTriangle'].getPosY(),
                    stateMach['ColorTriangle'].getSize(), 
                    stateMach['currentHue'], 
                    stateMach['currentSat'], 
                    stateMach['currentVal'],
                    int(stateMach['numHues']/2), 
                    selectRingColor,
                    stateMach['w2h'], 
                    stateMach['tDiff'])

        # Draw Confirm Button
        if (stateMach['ConfirmButton'].isVisible()):
            extraColor = colorsys.hsv_to_rgb(
                    stateMach['currentHue'], 
                    stateMach['currentSat'], 
                    stateMach['currentVal'])
            extraColor = (extraColor[0], extraColor[1], extraColor[2], 1.0)

            drawConfirm(
                    stateMach['ConfirmButton'].getPosX(), 
                    stateMach['ConfirmButton'].getPosY(), 
                    stateMach['ConfirmButton'].getSize(), 
                    stateMach['w2h'], 
                    fcc, 
                    extraColor, 
                    dtc)

        if (stateMach['BackButton'].isVisible()):
            extraColor = colorsys.hsv_to_rgb(
                    stateMach['prevHue'], 
                    stateMach['prevSat'], 
                    stateMach['prevVal'])
            extraColor = (extraColor[0], extraColor[1], extraColor[2], 1.0)

            # Draw Back Button
            drawArrow(
                    stateMach['BackButton'].getPosX(), 
                    stateMach['BackButton'].getPosY(), 
                    180.0,
                    stateMach['BackButton'].getSize(), 
                    stateMach['w2h'], 
                    fcc, 
                    extraColor, 
                    dtc)

        #tmc = ( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 1.0)
    except Exception as OOF:
        print(traceback.format_exc())
        print("Error:", OOF)
    return

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
            stateMach['masterSwitch'] = not stateMach['masterSwitch']
            for i in range(len(stateMach['lamps'])):
                stateMach['lamps'][i].setMainLight(stateMach['masterSwitch'])

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
                    stateMach['MasterSwitch'].setTarSize(1.75)
                    stateMach['MasterSwitch'].setTargetFaceColor((0.0, 0.0, 0.0, 0.75))
                    stateMach['MasterSwitch'].setTargetDetailColor((0.0, 0.0, 0.0, 0.0))
                    stateMach['AllSetButton'].setTarSize(0.0)
                    stateMach['BulbButtons'].setTarSize(0.0)

                    stateMach['ColorTriangle'].setTarSize(1.0)
                    stateMach['ColorTriangle'].setValue("coordX", stateMach['bulbButtons'][i][0]/stateMach['w2h'])
                    stateMach['ColorTriangle'].setValue("coordY", stateMach['bulbButtons'][i][1])
                    stateMach['ColorTriangle'].setTarPosX(0.0)
                    stateMach['ColorTriangle'].setTarPosY(0.0)

                    stateMach['HueRing'].setSize(0.0)
                    stateMach['HueRing'].setTarSize(1.0)
                    stateMach['HueRing'].setValue("coordX", stateMach['bulbButtons'][i][0]/stateMach['w2h'])
                    stateMach['HueRing'].setValue("coordY", stateMach['bulbButtons'][i][1])
                    stateMach['HueRing'].setTarPosX(0.0)
                    stateMach['HueRing'].setTarPosY(0.0)

                    stateMach['BackButton'].setTarSize(0.2)
                    stateMach['BackButton'].setValue("coordX", stateMach['bulbButtons'][i][0]/stateMach['w2h'])
                    stateMach['BackButton'].setValue("coordY", stateMach['bulbButtons'][i][1])
                    stateMach['BackButton'].setTarPosX(-0.75)
                    stateMach['BackButton'].setTarPosY(-0.75)

                    stateMach['ConfirmButton'].setTarSize( 0.2)
                    stateMach['ConfirmButton'].setValue("coordX", stateMach['bulbButtons'][i][0]/stateMach['w2h'])
                    stateMach['ConfirmButton'].setValue("coordY", stateMach['bulbButtons'][i][1])
                    stateMach['ConfirmButton'].setTarPosX( 0.75)
                    stateMach['ConfirmButton'].setTarPosY(-0.75)

                    stateMach['GranChanger'].setTarSize(0.3)
                    stateMach['GranChanger'].setTarPosY(-0.91)

                    # Record previous color(s)
                    stateMach['prevHue'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[0]
                    stateMach['prevSat'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[1]
                    stateMach['prevVal'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[2]

            # Watch all-set for input
            if (watchDot(
                mapRanges(stateMach['AllSetButton'].getTarPosX(), -1.0,  1.0, 0, stateMach['wx']*2),  # X coord of button
                mapRanges(stateMach['AllSetButton'].getTarPosY(),  1.0, -1.0, 0, stateMach['wy']*2),  # Y coord of button
                min(stateMach['wx'], stateMach['wy'])*0.2)          # Button Radius
                and
                len(stateMach['lamps']) > 0
                and
                stateMach['mousePressed']): # Button Must be clicked

                # Set Color Picker as target Screen selecting bulb all bulbs
                stateMach['targetScreen'] = 1
                stateMach['targetBulb'] = stateMach["lamps"][Light].getNumBulbs()
                stateMach['MasterSwitch'].setTarSize(1.75)
                stateMach['MasterSwitch'].setTargetFaceColor((0.0, 0.0, 0.0, 0.75))
                stateMach['MasterSwitch'].setTargetDetailColor((0.0, 0.0, 0.0, 0.0))
                stateMach['AllSetButton'].setTarSize(0.0)
                stateMach['BulbButtons'].setTarSize(0.0)

                stateMach['ColorTriangle'].setTarSize(1.0)
                stateMach['ColorTriangle'].setValue("coordX", stateMach['AllSetButton'].getPosX())
                stateMach['ColorTriangle'].setValue("coordY", stateMach['AllSetButton'].getPosY())
                stateMach['ColorTriangle'].setTarPosX(0.0)
                stateMach['ColorTriangle'].setTarPosY(0.0)

                stateMach['HueRing'].setSize(0.0)
                stateMach['HueRing'].setTarSize(1.0)
                stateMach['HueRing'].setValue("coordX", stateMach['AllSetButton'].getPosX())
                stateMach['HueRing'].setValue("coordY", stateMach['AllSetButton'].getPosY())
                stateMach['HueRing'].setValue("scaleX", 0.0)
                stateMach['HueRing'].setTarPosX(0.0)
                stateMach['HueRing'].setTarPosY(0.0)

                stateMach['BackButton'].setTarSize(0.2)
                stateMach['BackButton'].setValue("coordX", stateMach['AllSetButton'].getPosX()/stateMach['w2h'])
                stateMach['BackButton'].setValue("coordY", stateMach['AllSetButton'].getPosY())
                stateMach['BackButton'].setTarPosX(-0.75)
                stateMach['BackButton'].setTarPosY(-0.75)

                stateMach['ConfirmButton'].setTarSize( 0.2)
                stateMach['ConfirmButton'].setValue("coordX", stateMach['AllSetButton'].getPosX()/stateMach['w2h'])
                stateMach['ConfirmButton'].setValue("coordY", stateMach['AllSetButton'].getPosY())
                stateMach['ConfirmButton'].setTarPosX( 0.75)
                stateMach['ConfirmButton'].setTarPosY(-0.75)

                stateMach['GranChanger'].setTarSize(0.3)
                stateMach['GranChanger'].setTarPosY(-0.91)

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

        gctmx = stateMach['GranChanger'].getTarSize()*(24.0/36.0)
        tmr = min(stateMach['wx'], stateMach['wy']*(12.0/36.0)*0.3)
        if (stateMach['w2h'] <= 1.0):
            gctmy = stateMach['GranChanger'].getPosY()*stateMach['w2h']
        else:
            gctmy = stateMach['GranChanger'].getPosY()

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
        mapRanges(stateMach['ConfirmButton'].getPosX(), -1.0,  1.0, 0, stateMach['wx']*2),
        mapRanges(stateMach['ConfirmButton'].getPosY(),  1.0, -1.0, 0, stateMach['wy']*2),
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

            stateMach['MasterSwitch'].setTarSize(1.0)
            stateMach['MasterSwitch'].setTargetFaceColor(stateMach['faceColor'])
            stateMach['MasterSwitch'].setTargetDetailColor(stateMach['detailColor'])
            stateMach['ColorTriangle'].setTarSize(0.0)
            stateMach['HueRing'].setTarSize(10.0)
            stateMach['BulbButtons'].setTarSize(0.4)
            stateMach['AllSetButton'].setTarSize(0.1275)
            stateMach['BackButton'].setTarSize(0.0)
            stateMach['ConfirmButton'].setTarSize(0.0)
            stateMach['GranChanger'].setTarSize(0.0)
            stateMach['GranChanger'].setTarPosY(0.0)
            stateMach['GranChanger'].setAccel(0.33)
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
        mapRanges(stateMach['BackButton'].getPosX(), -1.0,  1.0, 0, stateMach['wx']*2),
        mapRanges(stateMach['BackButton'].getPosY(),  1.0, -1.0, 0, stateMach['wy']*2),
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

            stateMach['MasterSwitch'].setTarSize(1.0)
            stateMach['MasterSwitch'].setTargetFaceColor(stateMach['faceColor'])
            stateMach['MasterSwitch'].setTargetDetailColor(stateMach['detailColor'])
            stateMach['ColorTriangle'].setTarSize(0.0)
            stateMach['HueRing'].setTarSize(10.0)
            stateMach['targetScreen'] = 0
            stateMach['BulbButtons'].setTarSize(0.4)
            stateMach['AllSetButton'].setTarSize(0.1275)
            stateMach['BackButton'].setTarSize(0.0)
            stateMach['ConfirmButton'].setTarSize(0.0)
            stateMach['GranChanger'].setTarSize(0.0)
            stateMach['GranChanger'].setTarPosY(0.0)
            stateMach['GranChanger'].setAccel(0.33)


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
    # State = 0: button is depressed, low
    # State = 1: button is released, high
    if (stateMach['currentState'] == 1 and state == 0):
        stateMach['mousePressed'] = True
    #else:
        #stateMach['mousePressed'] = False

    stateMach['currentState'] = state
    stateMach['cursorX'] = mouseX
    stateMach['cursorY'] = mouseY

    if (state == 0):
        stateMach['mouseButton'] = button
    elif (button > 2):
        stateMach['mouseButton'] = button
    else:
        stateMach['mouseButton'] = "None"

    #print("state: " + str(state) + ", currentState: " + str(stateMach['currentState']))
    return

# Main screen drawing routine
# Passed to glutDisplayFunc()
def display():
    global stateMach

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #drawBackground()

    #stateMach['tDiff'] = 0.70568/stateMach['fps']
    #stateMach['tDiff'] = 1.30568/stateMach['fps']
    stateMach['tDiff'] = 2.71828/stateMach['fps']
    #stateMach['tDiff'] = 3.14159/stateMach['fps']
    #stateMach['tDiff'] = 6.28318/stateMach['fps']

    drawElements()
    
    if (stateMach['targetScreen'] == 0):
        watchHomeInput()
    elif (stateMach['targetScreen'] == 1):
        watchColrSettingInput()

    # Constrain Animation Cursor
    #if (stateMach['currentState'] == 1):# or stateMach['prvState'] == 0):
        #stateMach['interactionCursor'] = 1.0
    #else:
        #stateMach['interactionCursor'] = constrain(stateMach['interactionCursor'] - stateMach['tDiff'], 0.0, 1.0)

    #if (stateMach['targetScreen'] == 0):
        #if (stateMach['colrSettingCursor'] > 0):
            #stateMach['colrSettingCursor'] = constrain(stateMach['colrSettingCursor']-stateMach['tDiff'], 0, 1)
            #drawSettingColor()
        #if (stateMach['targetScreen'] == 0) and (stateMach['colrSettingCursor'] == 0):
            #drawHome()

    #elif (stateMach['targetScreen'] == 1):
        #if (stateMach['colrSettingCursor'] < 1):
            #stateMach['colrSettingCursor'] = constrain(stateMach['colrSettingCursor']+stateMach['tDiff'], 0, 1)
        #drawSettingColor()

    stateMach['AllSetButton'].setTimeSlice(stateMach['tDiff']*2)
    stateMach['BackButton'].setTimeSlice(stateMach['tDiff']*2)
    stateMach['ConfirmButton'].setTimeSlice(stateMach['tDiff']*2)
    stateMach['ColorTriangle'].setTimeSlice(stateMach['tDiff']*2)
    stateMach['HueRing'].setTimeSlice(stateMach['tDiff']*2)
    stateMach['BulbButtons'].setTimeSlice(stateMach['tDiff']*2)
    stateMach['MasterSwitch'].setTimeSlice(stateMach['tDiff']*2)
    stateMach['GranChanger'].setTimeSlice(stateMach['tDiff']*2)

    # Update Colors of Lamps
    for i in range(len(stateMach['lamps'])):
        stateMach['lamps'][i].updateBulbs(stateMach['tDiff']/2)

    #drawPrim(0.0, 0.0, 1.0, 0.0, stateMach['w2h'], stateMach['faceColor'], stateMach['detailColor'], (1.0, 1.0, 1.0, 1.0))

    if (stateMach['drawInfo']):
        drawInfo(stateMach)

    stateMach['mousePressed'] = False

    stateMach['AllSetButton'].updateParams()
    stateMach['BackButton'].updateParams()
    stateMach['ConfirmButton'].updateParams()
    stateMach['ColorTriangle'].updateParams()
    stateMach['HueRing'].updateParams()
    stateMach['BulbButtons'].updateParams()
    stateMach['MasterSwitch'].updateParams()
    stateMach['GranChanger'].updateParams()

    glutSwapBuffers()
    plugins.pluginLoader.updatePlugins()

def idleWindowOpen():
    framerate()
    glutPostRedisplay()

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
    stateMach['colrSettingCursor']  = 0.0
    stateMach['interactionCursor']  = 0.0
    stateMach['mousePressed']       = False
    stateMach['mouseButton']        = "None"
    stateMach['drawInfo']           = False

    # Setup UI animation objects, initial parameters
    #stateMach['faceColor'] = UIcolor()
    #stateMach['faceColor'].setTargetColor((0.3, 0.3, 0.3, 1.0))
    #stateMach['detailColor'] = UIcolor()
    #stateMach['detailColor'].setTargetColor((0.9, 0.9, 0.9, 1.0))

    stateMach['HueRing'] = UIelement()
    #stateMach['HueRing'].setTargetFaceColor(stateMach["faceColor"])
    #stateMach['HueRing'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['HueRing'].params["coordX"].setCurve("easeOutSine")
    stateMach['HueRing'].params["coordY"].setCurve("easeOutSine")
    stateMach['HueRing'].params["scaleX"].setCurve("easeInOutQuint")

    stateMach['BackButton'] = UIelement()
    stateMach['BackButton'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['BackButton'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['BackButton'].params["coordX"].setCurve("easeOutCubic")
    stateMach['BackButton'].params["coordY"].setCurve("easeInOutCubic")

    stateMach['BulbButtons'] = UIelement()
    stateMach['BulbButtons'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['BulbButtons'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['BulbButtons'].setTarSize(0.4)
    stateMach['BulbButtons'].setAccel(0.125)

    stateMach['GranChanger'] = UIelement()
    stateMach['GranChanger'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['GranChanger'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['GranChanger'].setTarSize(0.0)
    stateMach['GranChanger'].setTarPosX(0.0)
    stateMach['GranChanger'].setTarPosY(0.0)
    stateMach['GranChanger'].params["coordY"].setCurve("easeOutBack")

    stateMach['MasterSwitch'] = UIelement()
    stateMach['MasterSwitch'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['MasterSwitch'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['MasterSwitch'].setTarSize(1.0)
    stateMach['MasterSwitch'].setAccel(0.125)
    stateMach['MasterSwitch'].setTarPosX(0.0)
    stateMach['MasterSwitch'].setTarPosY(0.0)

    stateMach['AllSetButton'] = UIelement()
    stateMach['AllSetButton'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['AllSetButton'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['AllSetButton'].setTarSize(0.1275)
    stateMach['AllSetButton'].setAccel(0.125)
    stateMach['AllSetButton'].setTarPosX(-0.775)
    stateMach['AllSetButton'].setTarPosY(0.775)

    stateMach['ConfirmButton'] = UIelement()
    stateMach['ConfirmButton'].setTargetFaceColor(stateMach["faceColor"])
    stateMach['ConfirmButton'].setTargetDetailColor(stateMach["detailColor"])
    stateMach['ConfirmButton'].params["coordX"].setCurve("easeOutCubic")
    stateMach['ConfirmButton'].params["coordY"].setCurve("easeInOutCubic")

    stateMach['ColorTriangle'] = UIelement()
    #stateMach[''].setTargetFaceColor(stateMach["faceColor"])
    #stateMach[''].setTargetDetailColor(stateMach["detailColor"])
    stateMach['ColorTriangle'].params["coordX"].setCurve("easeInOutCirc")
    stateMach['ColorTriangle'].params["coordY"].setCurve("easeInOutCirc")
    stateMach['ColorTriangle'].params["scaleX"].setCurve("easeOutCirc")


    #stateMach['lamps'].append(getAllLamps()[0])

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
