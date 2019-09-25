#!/usr/bin/python3

print("Now Loading...")
#import hliUtilImports
from hliUtilImports import *
from lampClass import *
print("Done!")

#import profile

def init():
    global stateMach
    stateMach['curBulb'] = 0
    makeFont()

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
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
        print("%.0f frames in %3.1f seconds = %6.3f FPS" % (stateMach['frames'],seconds,stateMach['fps']))
        stateMach['t0'] = t
        stateMach['frames'] = 0
        stateMach['second'] = datetime.datetime.now().second
        stateMach['minute'] = datetime.datetime.now().minute + stateMach['second']/60
        stateMach['hour'] = datetime.datetime.now().hour + stateMach['minute']/60
        if (stateMach['hour'] > 11):
            stateMach['hour'] -= 12

        if (machine() == "armv6l" or machine() == "armv7l"):
            glutSetCursor(GLUT_CURSOR_NONE)

    if stateMach['frameLimit'] and (stateMach['fps'] > 60):
        pass
        #time.sleep(2*float(stateMach['fps'])/10000.0)
        time.sleep(0.015)

    return

def drawBackground():
    global stateMach
    if (len(stateMach['lamps']) > 0):

        # NEEDS IMPLEMENTATION FOR SELECTING LAMP, SPACE, ZONE, ETC
        Light = 0

        if (stateMach['lamps'][Light].getArn() == 0):
            drawHomeCircle(0.0, 0.0, 
                    stateMach['wx'], stateMach['wy'], 
                    stateMach['lamps'][Light].getNumBulbs(), 
                    stateMach['lamps'][Light].getAngle(), 
                    stateMach['w2h'],
                    stateMach['lamps'][Light].getBulbsCurrentRGB());

        elif (stateMach['lamps'][Light].getArn() == 1):
            drawHomeLinear(0.0, 0.0, 
                    stateMach['wx'], stateMach['wy'],
                    stateMach['lamps'][Light].getNumBulbs(), 
                    stateMach['lamps'][Light].getAngle(), 
                    stateMach['w2h'],
                    stateMach['lamps'][Light].getBulbsCurrentRGB());

def drawHome():
    global stateMach

    # NEEDS IMPLEMENTATION FOR SELECTING LAMP, SPACE, ZONE, ETC
    Light = 0

    iconSize = 0.15
    try:
        drawClock(
                stateMach['MasterSwitch'].getPosX(),    # X-Coordinate of position
                stateMach['MasterSwitch'].getPosY(),    # Y-Coordinate of position
                stateMach['hour'],                      # Hour to be displayed
                stateMach['minute'],                    # Minute to be displayed
                stateMach['MasterSwitch'].getSize(),    # Size of Clock
                stateMach['w2h'],                       # w2h for proper aspect ratio scaling
                stateMach['faceColor'],                 # color of the clock face
                stateMach['detailColor'])               # color of the clock hands

        if (len(stateMach['lamps']) > 0):
            buttons = drawBulbButton(
                    stateMach['lamps'][Light].getArn(),
                    stateMach['lamps'][Light].getNumBulbs(),
                    60,
                    stateMach['lamps'][Light].getAngle(),
                    iconSize*2.66,
                    stateMach['faceColor'],
                    stateMach['detailColor'],
                    stateMach['lamps'][Light].getBulbsCurrentRGB(),
                    stateMach['w2h'])

            tmc = ( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100))
            drawIcon(0.75, 0.75, iconSize*0.85, tmc, stateMach['w2h'], stateMach['lamps'][Light])

        drawText("Hello World!", 0.0, 0.0, 1.0, stateMach['w2h'], (1.0, 0.2, 1.0, 1.0))
        #printText(
                #-0.9, -0.6,
                #0.05, 
                #stateMach['w2h'], 
                #'AaBbCcDdEeFfGgHhIiJjKkLlMm',
                #( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100)))

        #printText(
                #-0.9, -0.875,
                #0.05, 
                #stateMach['w2h'], 
                #'NnOoPpQqRrSsTtUuVvWwXxYyZz',
                #( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100)))

        #primTest(stateMach['someVar']/100-0.3333333333, 0.0, 
                #0.75, stateMach['w2h'],
                #(0.5, stateMach['someVar']/100, 1.0, 1.0),
                #(1.0, 0.5, 0.0, 1.0),
                #(0.0, 1.0, 0.5, 1.0))

        # Watch Home Screen for input
        if (watchScreen()):
    
            # Watch Clock for input
            if (watchDot(
                stateMach['wx'],                            # Middle of Screen
                stateMach['wy'],                            # Middle of Screen
                min(stateMach['wx'], stateMach['wy'])/2)):  # Clock Radius
                stateMach['masterSwitch'] = not stateMach['masterSwitch']
                for i in range(len(stateMach['lamps'])):
                    stateMach['lamps'][i].setMainLight(stateMach['masterSwitch'])

            # Watch bulb buttons for input
            elif (len(stateMach['lamps']) > 0):
                for i in range(len(buttons)):
                    if (watchDot(
                        mapRanges(buttons[i][0], -stateMach['w2h'], stateMach['w2h'], 0, stateMach['wx']*2),# X coord of button
                        mapRanges(buttons[i][1],      1.0,    -1.0, 0, stateMach['wy']*2),                  # Y coord of button
                        min(stateMach['wx'], stateMach['wy'])*0.5*0.3)                                      # Button Radius
                        and
                        len(stateMach['lamps']) > 0):

                        # Set Color Picker as target Screen selecting bulb i
                        stateMach['targetScreen'] = 1
                        stateMach['targetBulb'] = i
                        stateMach['MasterSwitch'].setTarSize(1.75)

                        # Record previous color(s)
                        stateMach['prevHue'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[0]
                        stateMach['prevSat'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[1]
                        stateMach['prevVal'] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[2]

            # Watch all-set for input
                if (watchDot(
                    mapRanges(0.75, -1.0,  1.0, 0, stateMach['wx']*2),  # X coord of button
                    mapRanges(0.75,  1.0, -1.0, 0, stateMach['wy']*2),  # Y coord of button
                    min(stateMach['wx'], stateMach['wy'])*0.2)          # Button Radius
                    and
                    len(stateMach['lamps']) > 0):

                    # Set Color Picker as target Screen selecting bulb all bulbs
                    stateMach['targetScreen'] = 1
                    stateMach['targetBulb'] = stateMach["lamps"][Light].getNumBulbs()
                    stateMach['MasterSwitch'].setTarSize(1.75)

                    # Record previous color(s)
                    stateMach['prevHue'] = stateMach['lamps'][Light].getBulbCurrentHSV(0)[0]
                    stateMach['prevSat'] = stateMach['lamps'][Light].getBulbCurrentHSV(0)[1]
                    stateMach['prevVal'] = stateMach['lamps'][Light].getBulbCurrentHSV(0)[2]
                    for i in range(stateMach['targetBulb']):
                        stateMach['prevHues'][i] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[0]
                        stateMach['prevSats'][i] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[1]
                        stateMach['prevVals'][i] = stateMach['lamps'][Light].getBulbCurrentHSV(i)[2]

    except Exception as OOF:
        print(traceback.format_exc())
        print("Error:", OOF)
    return

#def drawSettingColor(cursor, targetLamp, targetBulb):
def drawSettingColor():
    global stateMach

    # NEEDS IMPLEMENTATION FOR SELECTING LAMP, SPACE, ZONE, ETC
    Light = 0

    if (len(stateMach['lamps']) > 0):
        if (stateMach['targetBulb'] == stateMach['lamps'][Light].getNumBulbs()):
            # Set All Lamp Elements
            tmcHSV = stateMach['lamps'][Light].getBulbTargetHSV(0)
            if (stateMach['currentHue'] == None):
                stateMach['currentHue'] = stateMach['lamps'][Light].getBulbHSV(0)[0]
            if (stateMach['currentSat'] == None):
                stateMach['currentSat'] = stateMach['lamps'][Light].getBulbHSV(0)[1]
            if (stateMach['currentVal'] == None):
                stateMach['currentVal'] = stateMach['lamps'][Light].getBulbHSV(0)[2]
        else:
            # Set selected Lamp Element (targetBulb)
            tmcHSV = stateMach['lamps'][Light].getBulbTargetHSV(stateMach['targetBulb'])
            if (stateMach['currentHue'] == None):
                stateMach['currentHue'] = stateMach['lamps'][Light].getBulbHSV(targetBulb)[0]
            if (stateMach['currentSat'] == None):
                stateMach['currentSat'] = stateMach['lamps'][Light].getBulbHSV(targetBulb)[1]
            if (stateMach['currentVal'] == None):
                stateMach['currentVal'] = stateMach['lamps'][Light].getBulbHSV(targetBulb)[2]

    acbic = animCurveBounce(1.0-stateMach['colrSettingCursor'])
    acic = animCurve(1.0-stateMach['colrSettingCursor'])
    acbc = animCurveBounce(stateMach['colrSettingCursor'])
    acc = animCurve(stateMach['colrSettingCursor'])
    faceColor = (stateMach['faceColor'][0]*acc, 
            stateMach['faceColor'][1]*acc,
            stateMach['faceColor'][2]*acc,
            1.0)
    detailColor = (stateMach['detailColor'][0]*acc, 
            stateMach['detailColor'][1]*acc,
            stateMach['detailColor'][2]*acc,
            stateMach['detailColor'][3]*acc)
    cmx = 0.15
        
    if (stateMach['wereColorsTouched']):
        selectRingColor = stateMach['detailColor']
    else:
        selectRingColor = stateMach['faceColor']

    iconSize = 0.15
    if (len(stateMach['lamps']) > 0):
        drawBulbButton(
                stateMach['lamps'][Light].getArn(),
                stateMach['lamps'][Light].getNumBulbs(),
                60,
                stateMach['lamps'][Light].getAngle(),
                iconSize*2.66*pow(acc, 4),
                faceColor,
                detailColor,
                stateMach['lamps'][Light].getBulbsCurrentRGB(),
                stateMach['w2h'])

        tmc = ( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100))
        drawIcon(0.75, 0.75, iconSize*0.85*pow(acc, 4), tmc, stateMach['w2h'], stateMach['lamps'][Light])

    # Draw Granularity Rocker Underneath Clock
    limit = 0.85
    if (stateMach['colrSettingCursor'] < limit):
        if (stateMach['w2h'] <= 1.0):
            tmy = -0.91*acbic*stateMach['w2h']
        else:
            tmy = -0.91*acbic

        drawGranRocker(
                0.0, tmy,
                faceColor,
                detailColor,
                stateMach['numHues'],
                0.0,
                stateMach['w2h'],
                0.30*acic,
                stateMach['tDiff'])
    
    # Draw Clock
    drawClock(
            stateMach['MasterSwitch'].getPosX(),    # X-Coordinate of position
            stateMach['MasterSwitch'].getPosY(),    # Y-Coordinate of position
            stateMach['hour'],                      # Hour to be displayed
            stateMach['minute'],                    # Minute to be displayed
            stateMach['MasterSwitch'].getSize(),    # Size of Clock
            stateMach['w2h'],                       # w2h for proper aspect ratio scaling
            faceColor,                              # color of the clock face
            tuple([acc*x for x in detailColor]))    # color of the clock hands

    # Draw Granularity Rocker on top of Clock
    if (stateMach['colrSettingCursor'] >= limit):
        if (stateMach['w2h'] <= 1.0):
            tmy = -0.91*acbic*stateMach['w2h']
        else:
            tmy = -0.91*acbic
        drawGranRocker(
                0.0, tmy,
                faceColor,
                detailColor,
                stateMach['numHues'],
                0.0,
                stateMach['w2h'],
                0.30*acic,
                stateMach['tDiff'])

    # Draw Ring of Dots with different hues
    hueButtons = drawHueRing(
            stateMach['currentHue'], 
            stateMach['numHues'], 
            selectRingColor, 
            stateMach['w2h'], 
            acbic, 
            stateMach['tDiff'],
            stateMach['interactionCursor'])

    # Draw Triangle of Dots with different brightness/saturation
    satValButtons = drawColrTri(
            stateMach['currentHue'], 
            stateMach['currentSat'], 
            stateMach['currentVal'],
            int(stateMach['numHues']/2), 
            selectRingColor,
            stateMach['w2h'], acbic, 
            stateMach['tDiff'])

    #if ( stateMach['wereColorsTouched'] ):
    if ( True ):
        extraColor = colorsys.hsv_to_rgb(
                stateMach['currentHue'], 
                stateMach['currentSat'], 
                stateMach['currentVal'])
        extraColor = (extraColor[0], extraColor[1], extraColor[2], 1.0)
    #else:
        #extraColor = stateMach['detailColor']

    # Draw Confirm Button
    drawConfirm(
            0.75-0.4*(1.0-acbic), 
            -0.75-0.5*acbc, 
            0.2*(1.0-acbc), stateMach['w2h'], 
            stateMach['faceColor'], 
            extraColor, 
            stateMach['detailColor']);

    #if ( stateMach['wereColorsTouched'] ):
    if ( True ):
        extraColor = colorsys.hsv_to_rgb(
                stateMach['prevHue'], 
                stateMach['prevSat'], 
                stateMach['prevVal'])
        extraColor = (extraColor[0], extraColor[1], extraColor[2], 1.0)
    #else:
        #extraColor = stateMach['detailColor']

    # Draw Back Button
    drawArrow(
            -0.75+0.4*(1.0-acbic), 
            -0.75-0.5*acbc, 
            180.0,
            0.2*(1.0-acbc), stateMach['w2h'], 
            stateMach['faceColor'], 
            extraColor, 
            stateMach['detailColor']);

    # Watch Color Picker Screen for input
    if (watchScreen()):

        if (stateMach['w2h'] <= 1.0):
            tmx = (0.3*24.0/36.0)*stateMach['w2h']
            tmy = -0.91*acbic*stateMach['w2h']
            tmux = 1.0*stateMach['w2h']
            tmuy = 1.0
            tmr = min(stateMach['wx'],stateMach['wy']*(12.0/36.0)*0.3)*stateMach['w2h']
        else:
            tmx = (0.3*24.0/36.0)
            tmy = -0.91*acbic
            tmux = stateMach['w2h']
            tmuy = 1.0
            tmr = min(stateMach['wx'],stateMach['wy']*(12.0/36.0)*0.3)

        # Watch Granularity Rocker for Input
        if (watchDot(
            mapRanges(tmx, -tmux, tmux, 0, stateMach['wx']*2),
            mapRanges(tmy, tmuy, -tmuy, 0, stateMach['wy']*2),
            tmr)):
                stateMach['numHues'] += 2
                if stateMach['numHues'] > 14:
                    stateMach['numHues'] = 14

        # Watch Granularity Rocker for Input
        if (watchDot(
            mapRanges(-tmx, -tmux, tmux, 0, stateMach['wx']*2),
            mapRanges(tmy, tmuy, -tmuy, 0, stateMach['wy']*2),
            tmr)):
                stateMach['numHues'] -= 2
                if stateMach['numHues'] < 10:
                    stateMach['numHues'] = 10

        # Watch Hue Ring for Input
        for i in range(len(hueButtons)):
            tmr = 1.0
            if (stateMach['w2h'] <= 1.0):
                hueButtons[i] = (
                        hueButtons[i][0]*stateMach['w2h'], 
                        hueButtons[i][1]*stateMach['w2h'], 
                        hueButtons[i][2])
                tmr = stateMach['w2h']

            if (watchDot(
                mapRanges(hueButtons[i][0], -1.0*stateMach['w2h'], 1.0*stateMach['w2h'], 0, stateMach['wx']*2),
                mapRanges(hueButtons[i][1], 1.0, -1.0, 0, stateMach['wy']*2),
                min(stateMach['wx'], stateMach['wy'])*0.15*(12.0/float(stateMach['numHues'])))):

                stateMach['wereColorsTouched'] = True
                stateMach['currentHue'] = hueButtons[i][2]
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
        for i in range(len(satValButtons)):
            tmr = 1.0
            if (stateMach['w2h'] <= 1.0):
                satValButtons[i] = (
                        satValButtons[i][0]*stateMach['w2h'],     # X-Coord of Button
                        satValButtons[i][1]*stateMach['w2h'],     # Y-Coord of Button
                        satValButtons[i][2],                    # Saturation of Button
                        satValButtons[i][3])                    # Value of Button
                tmr = stateMach['w2h']

            # Map relative x-Coord to screen x-Coord
            if (watchDot(
                mapRanges(satValButtons[i][0], -1.0*stateMach['w2h'], 1.0*stateMach['w2h'], 0, stateMach['wx']*2),
                mapRanges(satValButtons[i][1], 1.0, -1.0, 0, stateMach['wy']*2),
                min(stateMach['wx'], stateMach['wy'])*0.073)):

                stateMach['wereColorsTouched'] = True
                stateMach['currentSat'] = satValButtons[i][2]
                stateMach['currentVal'] = satValButtons[i][3]
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
        mapRanges( 0.75, -1.0,  1.0, 0, stateMach['wx']*2),
        mapRanges(-0.75,  1.0, -1.0, 0, stateMach['wy']*2),
        min(stateMach['wx'], stateMach['wy'])*0.2)):
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
            stateMach['targetScreen'] = 0

        if ( stateMach['wereColorsTouched'] and len(stateMach['lamps']) > 0):
            if (stateMach['targetBulb'] == stateMach['lamps'][Light].getNumBulbs()):
                extraColor = colorsys.hsv_to_rgb(
                        stateMach['prevHues'][0], 
                        stateMach['prevSats'][0], 
                        stateMach['prevVals'][0])
            else:
                extraColor = colorsys.hsv_to_rgb(
                        stateMach['prevHue'], 
                        stateMach['prevSat'], 
                        stateMach['prevVal'])
        else:
            extraColor = stateMach['detailColor']


        # Watch Back Button for input
        if (watchDot(
        mapRanges(-0.75, -1.0,  1.0, 0, stateMach['wx']*2),
        mapRanges(-0.75,  1.0, -1.0, 0, stateMach['wy']*2),
        min(stateMach['wx'], stateMach['wy'])*0.2)):
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
            stateMach['targetScreen'] = 0

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
    if stateMach['prvState'] == 0:
        stateMach['prvState'] = stateMach['touchState']
        return True
    else:
        stateMach['prvState'] = stateMach['touchState']
        return False

def mousePassive(mouseX, mouseY):
    global stateMach
    if (stateMach['touchState'] == 0):
        stateMach['cursorX'] = mouseX
        stateMach['cursorY'] = mouseY
        
def mouseInteraction(button, state, mouseX, mouseY):
    global stateMach
    # State = 0: button is depressed, low
    # State = 1: button is released, high
    stateMach['currentState'] = state
    if (state == 0):
        stateMach['cursorX'] = mouseX
        stateMach['cursorY'] = mouseY

    if (stateMach['touchState'] == 1) and (state != 1) and (stateMach['prvState'] == 1):
        stateMach['prvState'] = not stateMach['touchState']
        return
    elif (stateMach['touchState'] == 0) and (state != 0) and (stateMach['prvState'] == 0):
        stateMach['touchState'] = not stateMach['touchState']
        stateMach['prvState'] = not stateMach['touchState']
        return

# Main screen drawing routine
# Passed to glutDisplayFunc()
def display():
    global stateMach

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    drawBackground()

    #stateMach['tDiff'] = 0.70568/stateMach['fps']
    #stateMach['tDiff'] = 1.30568/stateMach['fps']
    #stateMach['tDiff'] = 2.71828/stateMach['fps']
    stateMach['tDiff'] = 3.14159/stateMach['fps']
    #stateMach['tDiff'] = 6.28318/stateMach['fps']

    # Constrain Animation Cursor
    if (stateMach['currentState'] == 0 or stateMach['prvState'] == 0):
        stateMach['interactionCursor'] = 1.0
    else:
        stateMach['interactionCursor'] = constrain(stateMach['interactionCursor'] - stateMach['tDiff'], 0.0, 1.0)

    if (stateMach['targetScreen'] == 0):
        if (stateMach['colrSettingCursor'] > 0):
            stateMach['colrSettingCursor'] = constrain(stateMach['colrSettingCursor']-stateMach['tDiff'], 0, 1)
            drawSettingColor()
        if (stateMach['targetScreen'] == 0) and (stateMach['colrSettingCursor'] == 0):
            drawHome()

    elif (stateMach['targetScreen'] == 1):
        if (stateMach['colrSettingCursor'] < 1):
            stateMach['colrSettingCursor'] = constrain(stateMach['colrSettingCursor']+stateMach['tDiff'], 0, 1)
        drawSettingColor()

    stateMach['MasterSwitch'].setTimeSlice(stateMach['tDiff']*2)

    # Update Colors of Lamps
    for i in range(len(stateMach['lamps'])):
        stateMach['lamps'][i].updateBulbs(stateMach['tDiff']/2)

    stateMach['MasterSwitch'].updateParams()

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
    elif k == GLUT_KEY_F1:
        if stateMach['targetScreen'] == 0:
            stateMach['targetScreen'] = 1
            stateMach['targetBulb'] = 0
        elif stateMach['targetScreen'] == 1:
            stateMach['targetScreen'] = 0
    
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
        stateMach['numHues'] += 2
        if stateMach['numHues'] > 14:
            stateMach['numHues'] = 14

    if ch == as_8_bit('['):
        stateMach['numHues'] -= 2
        if stateMach['numHues'] < 10:
            stateMach['numHues'] = 10

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

    stateMach['AllSetButton']       = UIelement()
    stateMach['BackButton']         = UIelement()
    stateMach['ConfirmButton']      = UIelement()
    stateMach['ColorTriangle']      = UIelement()
    stateMach['HueRing']            = UIelement()
    stateMach['BulbButtons']        = UIelement()
    stateMach['MasterSwitch']       = UIelement()

    stateMach['MasterSwitch'].setTarSize(1.0)
    stateMach['MasterSwitch'].setAccel(0.125)
    stateMach['MasterSwitch'].setTarPosX(0.0)
    stateMach['MasterSwitch'].setTarPosY(0.0)

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
