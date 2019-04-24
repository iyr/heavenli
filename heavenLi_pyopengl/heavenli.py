#! /usr/bin/env python3
print("Now Loading...")

print("Loading OpenGL...")
import OpenGL
OpenGL.ERROR_ON_COPY = True
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
print("Done!")

print("Loading GLUT...")
from OpenGL.GLUT import *
print("Done!")

print("Loading NumPy...")
import numpy as np
print("Done!")

print("Loading System Utilities...")
import sys, time
from math import sin,cos,sqrt,pi,radians
import datetime
import os
print("Done!")

print("Loading heavenLi Utilities...")
import colorsys
from animUtils import *
from drawArn import *
from drawButtons import *
from lampClass import *
from rangeUtils import *
from shaderUtils import *
print("Done!")

def init():
    global statMac

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)

    print("Loading Shaders...")
    initShaders()
    print("Done!")

    statMac['minute'] = datetime.datetime.now().minute
    statMac['hour'] = datetime.datetime.now().hour
    if (statMac['hour'] > 11):
        statMac['hour'] -= 12
    print(statMac['hour'])
    print(statMac['minute'])

    statMac['wx'] = glutGet(GLUT_WINDOW_WIDTH)
    statMac['wy'] = glutGet(GLUT_WINDOW_HEIGHT)
    statMac['w2h'] = statMac['wx']/statMac['wy']
    demo = Lamp()
    statMac['lamps'].append(demo)

def framerate():
    global statMac
    t = time.time()
    statMac['frames'] += 1.0
    seconds = t - statMac['t0']
    statMac['someVar'] += statMac['someInc']
    if (statMac['someVar'] > 100) or (statMac['someVar'] < 0):
        statMac['someInc'] = -statMac['someInc']

    try:
        statMac['fps'] = statMac['frames']/seconds
    except:
        print("Too Fast, Too Quick!!")
    if t - statMac['t0'] >= 1.0:
        print("%.0f frames in %3.1f seconds = %6.3f FPS" % (statMac['frames'],seconds,statMac['fps']))
        statMac['t0'] = t
        statMac['frames'] = 0
        statMac['minute'] = datetime.datetime.now().minute
        statMac['hour'] = datetime.datetime.now().hour
        if (statMac['hour'] > 11):
            statMac['hour'] -= 12
    if statMac['frameLimit'] and (statMac['fps'] > 60):
        time.sleep(2*float(statMac['fps'])/10000.0)

def drawBackground(Light = 0 # Currently Selected Lamp, Space, or *
        ):
    global statMac
    if (statMac['lamps'][Light].getArn() == 0):
         drawHomeCircle(0.0, 0.0, 
                 statMac['wx'], statMac['wy'], 
                 statMac['lamps'][Light].getNumBulbs(), 
                 statMac['lamps'][Light].getAngle(), 
                 statMac['w2h'],
                 statMac['lamps'][Light].getBulbsRGB());

    elif (statMac['lamps'][Light].getArn() == 1):
         drawHomeLinear(0.0, 0.0, 
                 statMac['wx'], statMac['wy'],
                 statMac['lamps'][Light].getNumBulbs(), 
                 statMac['lamps'][Light].getAngle(), 
                 statMac['w2h'],
                 statMac['lamps'][Light].getBulbsRGB());

def drawHome():
    global statMac

    iconSize = 0.15
    drawClock(
            0.0, 0.0,
            statMac['hour'],
            statMac['minute'],
            #12*(statMac['someVar']/100),
            #60*(1.0-(statMac['someVar']/100)), 
            1.0, statMac['w2h'], 
            statMac['faceColor'],
            statMac['detailColor'])

    # We are at the home screen
    if (statMac['screen'] == 0) and (statMac['touchState'] != statMac['prvState']):
        dBias = min(statMac['wx'], statMac['wy'])/2
        if watchPoint(statMac['wx'], statMac['wy'], dBias):
            statMac['lightOn'] = not statMac['lightOn']
            for i in range(len(statMac['lamps'])):
                statMac['lamps'][i].setMainLight(statMac['lightOn'])

    buttons = drawBulbButton(
            statMac['lamps'][0].getArn(),
            statMac['lamps'][0].getNumBulbs(),
            statMac['lamps'][0].getAngle(),
            iconSize*2.66,
            statMac['faceColor'],
            statMac['detailColor'],
            statMac['lamps'][0].getBulbsRGB(),
            statMac['w2h'])

    for i in range(len(buttons)):
        if (statMac['screen'] == 0) and (statMac['touchState'] != statMac['prvState']):
            if (watchPoint(
                mapRanges(buttons[i][0], -1.0*statMac['w2h'], 1.0*statMac['w2h'], 0, statMac['wx']*2), 
                mapRanges(buttons[i][1],      1.0,    -1.0, 0, statMac['wy']*2),
                min(statMac['wx'], statMac['wy'])*0.5*0.3)):
                statMac['targetScreen'] = 1
                statMac['targetBulb'] = i
                statMac['prevHue'] = statMac['lamps'][0].getBulbHSV(i)[0]
                statMac['prevSat'] = statMac['lamps'][0].getBulbHSV(i)[1]
                statMac['prevVal'] = statMac['lamps'][0].getBulbHSV(i)[2]

    #printText(
            #-0.9, -0.6,
            #0.05, 
            #statMac['w2h'], 
            #'AaBbCcDdEeFfGgHhIiJjKkLlMm',
            #(1.0, 1.0, 1.0))

    #printText(
            #-0.9, -0.875,
            #0.05, 
            #statMac['w2h'], 
            #'NnOoPpQqRrSsTtUuVvWwXxYyZz',
            #(1.0, 1.0, 1.0))

    drawIconCircle(0.75, 0.75, 
            iconSize*0.85, 
            4,
            ( 0.9*(statMac['someVar']/100), 0.9*(statMac['someVar']/100), 0.9*(statMac['someVar']/100)),
            statMac['lamps'][0].getNumBulbs(), 
            statMac['lamps'][0].getAngle(), 
            statMac['w2h'], 
            statMac['lamps'][0].getBulbsRGB())

    drawIconLinear(0.75, -0.75, 
            iconSize*0.85, 
            4,
            ( 0.9*(statMac['someVar']/100), 0.9*(statMac['someVar']/100), 0.9*(statMac['someVar']/100)),
            statMac['lamps'][0].getNumBulbs(), 
            statMac['lamps'][0].getAngle(), 
            statMac['w2h'], 
            statMac['lamps'][0].getBulbsRGB())

__bulbsCurrentHSB = []
        
def drawSettingColor(cursor, targetLamp, targetBulb):
    global statMac
    tmcHSV = targetLamp.getBulbtHSV(statMac['targetBulb'])
    acbic = animCurveBounce(1.0-cursor)
    acic = animCurve(1.0-cursor)
    acbc = animCurveBounce(cursor)
    acc = animCurve(cursor)
    faceColor = (statMac['faceColor'][0]*acc, 
            statMac['faceColor'][1]*acc,
            statMac['faceColor'][2]*acc)
    detailColor = (statMac['detailColor'][0]*acc, 
            statMac['detailColor'][1]*acc,
            statMac['detailColor'][2]*acc)
    cmx = 0.15
    if (statMac['currentHue'] == None):
        statMac['currentHue'] = targetLamp.getBulbHSV(targetBulb)[0]
    if (statMac['currentSat'] == None):
        statMac['currentSat'] = targetLamp.getBulbHSV(targetBulb)[1]
    if (statMac['currentVal'] == None):
        statMac['currentVal'] = targetLamp.getBulbHSV(targetBulb)[2]
        
    if (statMac['wereColorsTouched']):
        selectRingColor = (1.0, 1.0, 1.0)
    else:
        selectRingColor = (0.3, 0.3, 0.3)

    iconSize = 0.15
    drawBulbButton(
            targetLamp.getArn(),
            targetLamp.getNumBulbs(),
            targetLamp.getAngle(),
            iconSize*2.66*pow(acc, 4),
            faceColor,
            detailColor,
            targetLamp.getBulbsRGB(),
            statMac['w2h'])

    #limit = 0.85
    #if (cursor < limit):
        #drawGranRocker(
                #0.0, -0.91*acbic,
                #faceColor,
                #detailColor,
                #statMac['numHues'],
                #0.0,
                #statMac['w2h'],
                #0.30*acic,
                #statMac['tDiff'])
    
    drawClock(
            0.0, 0.0,
            statMac['hour'],
            statMac['minute'],
            1.0+acic*0.75, 
            statMac['w2h'], 
            faceColor,
            #detailColor)
            tuple([acc*x for x in detailColor]))

    #if (cursor >= limit):
        #drawGranRocker(
                #0.0, -0.91*acbic,
                #faceColor,
                #detailColor,
                #statMac['numHues'],
                #0.0,
                #statMac['w2h'],
                #0.30*acic,
                #statMac['tDiff'])

    # Watch Granularity Rocker for Input
    #if (watchPoint(
        #mapRanges(0.3*24.0/36.0, -1.0*statMac['w2h'], 1.0*statMac['w2h'],   0, statMac['wx']*2),
        #mapRanges(0.0-0.91,  1.0               ,-1.0               ,   0, statMac['wy']*2),
        #min(statMac['wx'],statMac['wy']*(12.0/36.0)*0.3) )):
            #statMac['numHues'] += 2
            #if statMac['numHues'] > 14:
                #statMac['numHues'] = 14
    # Watch Granularity Rocker for Input
    #if (watchPoint(
        #mapRanges(-0.3*24.0/36.0, -1.0*statMac['w2h'], 1.0*statMac['w2h'], 0, statMac['wx']*2),
        #mapRanges(-0.91, 1.0, -1.0, 0, statMac['wy']*2),
        #min(statMac['wx'],statMac['wy']*(12.0/36.0)*0.3) )):
            #statMac['numHues'] -= 2
            #if statMac['numHues'] < 10:
                #statMac['numHues'] = 10

    # Draw Ring of Dots with different hues
    hueButtons = drawHueRing(
            statMac['currentHue'], 
            statMac['numHues'], 
            selectRingColor, 
            statMac['w2h'], 
            acbic, 
            statMac['tDiff'],
            statMac['interactionCursor'])

    for i in range(len(hueButtons)):
        tmr = 1.0
        if (statMac['w2h'] <= 1.0):
            hueButtons[i] = (
                    hueButtons[i][0]*statMac['w2h'], 
                    hueButtons[i][1]*statMac['w2h'], 
                    hueButtons[i][2])
            tmr = statMac['w2h']

        xLim = mapRanges(hueButtons[i][0], -1.0*statMac['w2h'], 1.0*statMac['w2h'],    0, statMac['wx']*2)
        yLim = mapRanges(hueButtons[i][1],  1.0,               -1.0,                   0, statMac['wy']*2)
        if (watchPoint(
            xLim, yLim,
            min(statMac['wx'], statMac['wy'])*0.15*(12.0/float(statMac['numHues'])) )):
            statMac['wereColorsTouched'] = True
            statMac['currentHue'] = hueButtons[i][2]
            tmcHSV = (
                    statMac['currentHue'], 
                    statMac['currentSat'], 
                    statMac['currentVal'])
            targetLamp.setBulbtHSV(statMac['targetBulb'], tmcHSV)

    # Draw Triangle of Dots with different brightness/saturation
    satValButtons = drawColrTri(
            statMac['currentHue'], 
            statMac['currentSat'], 
            statMac['currentVal'],
            int(statMac['numHues']/2), 
            selectRingColor,
            statMac['w2h'], acbic, 
            statMac['tDiff'])

    for i in range(len(satValButtons)):
        tmr = 1.0
        if (statMac['w2h'] <= 1.0):
            satValButtons[i] = (
                    satValButtons[i][0]*statMac['w2h'],     # X-Coord of Button
                    satValButtons[i][1]*statMac['w2h'],     # Y-Coord of Button
                    satValButtons[i][2],                    # Saturation of Button
                    satValButtons[i][3])                    # Value of Button
            tmr = statMac['w2h']

        # Map relative x-Coord to screen x-Coord
        xLim = mapRanges(
                satValButtons[i][0], 
                -1.0*statMac['w2h'], 
                1.0*statMac['w2h'], 
                0, 
                statMac['wx']*2)

        # Map relative y-Coord to screen y-Coord
        yLim = mapRanges(
                satValButtons[i][1],      
                1.0,    
                -1.0, 
                0, 
                statMac['wy']*2)

        if (watchPoint(
            xLim, yLim,
            min(statMac['wx'], statMac['wy'])*0.073)):
            statMac['wereColorsTouched'] = True
            statMac['currentSat'] = satValButtons[i][2]
            statMac['currentVal'] = satValButtons[i][3]
            tmcHSV = (
                    statMac['currentHue'], 
                    statMac['currentSat'], 
                    statMac['currentVal'])
            targetLamp.setBulbtHSV(statMac['targetBulb'], tmcHSV)

    if ( statMac['wereColorsTouched'] ):
        extraColor = colorsys.hsv_to_rgb(
                statMac['currentHue'], 
                statMac['currentSat'], 
                statMac['currentVal'])
    else:
        extraColor = statMac['detailColor']

    #drawConfirm(
            #0.75-0.4*(1.0-acbic), 
            #-0.75-0.5*acbc, 
            #0.2*(1.0-acbc), statMac['w2h'], 
            #statMac['faceColor'], 
            #extraColor, 
            #statMac['detailColor']);

    # Watch Confirm Button for input
    #if (watchPoint(
        #mapRanges( 0.75, -1.0,  1.0, 0, statMac['wx']*2),
        #mapRanges(-0.75,  1.0, -1.0, 0, statMac['wy']*2),
        #min(statMac['wx'], statMac['wy'])*0.2)):
        #statMac['wereColorsTouched'] = False
        #targetLamp.setBulbtHSV(statMac['targetBulb'], (
            #statMac['currentHue'], 
            #statMac['currentSat'], 
            #statMac['currentVal'] ) )
        #statMac['targetScreen'] = 0

    #if ( statMac['wereColorsTouched'] ):
        #extraColor = colorsys.hsv_to_rgb(
                #statMac['prevHue'], 
                #statMac['prevSat'], 
                #statMac['prevVal'])
    #else:
        #extraColor = statMac['detailColor']

    # Draw Back Button
    #drawArrow(
            #-0.75+0.4*(1.0-acbic), 
            #-0.75-0.5*acbc, 
            #180.0,
            #0.2*(1.0-acbc), statMac['w2h'], 
            #statMac['faceColor'], 
            #extraColor, 
            #statMac['detailColor']);
    #if (watchPoint(
        #mapRanges(-0.75, -1.0,  1.0, 0, statMac['wx']*2),
        #mapRanges(-0.75,  1.0, -1.0, 0, statMac['wy']*2),
        #min(statMac['wx'], statMac['wy'])*0.2)):
        #statMac['wereColorsTouched'] = False
        #targetLamp.setBulbtHSV(statMac['targetBulb'], (
            #statMac['prevHue'], 
            #statMac['prevSat'], 
            #statMac['prevVal'] ) )
        #statMac['targetScreen'] = 0

def watchPoint(px, py, pr):
    global statMac
    if (1.0 >= pow((statMac['cursorX']-px/2), 2) / pow(pr/2, 2) + pow((statMac['cursorY']-py/2), 2) / pow(pr/2, 2)):
        if statMac['prvState'] == 0:
            statMac['prvState'] = statMac['touchState']
            return True
        else:
            statMac['prvState'] = statMac['touchState']
            return False

def mousePassive(mouseX, mouseY):
    global statMac
    if (statMac['touchState'] == 0):
        statMac['cursorX'] = mouseX
        statMac['cursorY'] = mouseY
        
def mouseInteraction(button, state, mouseX, mouseY):
    global statMac
    # State = 0: button is depressed, low
    # State = 1: button is released, high
    statMac['currentState'] = state
    if (state == 0):
        statMac['cursorX'] = mouseX
        statMac['cursorY'] = mouseY

    if (statMac['touchState'] == 1) and (state != 1) and (statMac['prvState'] == 1):
        statMac['prvState'] = not statMac['touchState']
        return
    elif (statMac['touchState'] == 0) and (state != 0) and (statMac['prvState'] == 0):
        statMac['touchState'] = not statMac['touchState']
        statMac['prvState'] = not statMac['touchState']
        return

# Main screen drawing routine
# Passed to glutDisplayFunc()
def display():
    global statMac
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    drawBackground(0)
    #if (statMac['interactionCursor'] > 0.0):
        #print(statMac['interactionCursor'])

    #statMac['tDiff'] = 0.70568/statMac['fps']
    #statMac['tDiff'] = 1.30568/statMac['fps']
    statMac['tDiff'] = 2.71828/statMac['fps']
    #statMac['tDiff'] = 3.14159/statMac['fps']
    #statMac['tDiff'] = 6.28318/statMac['fps']
    if (statMac['currentState'] == 0 or statMac['prvState'] == 0):
        statMac['interactionCursor'] = 1.0
    else:
        statMac['interactionCursor'] = constrain(statMac['interactionCursor'] - statMac['tDiff'], 0.0, 1.0)

    if (statMac['targetScreen'] == 0):
        if (statMac['colrSettingCursor'] > 0):
            statMac['colrSettingCursor'] = constrain(statMac['colrSettingCursor']-statMac['tDiff'], 0, 1)
            drawSettingColor(statMac['colrSettingCursor'], statMac['lamps'][0], statMac['targetBulb'])
        if (statMac['targetScreen'] == 0) and (statMac['colrSettingCursor'] == 0):
            drawHome()

    elif (statMac['targetScreen'] == 1):
        if (statMac['colrSettingCursor'] < 1):
            statMac['colrSettingCursor'] = constrain(statMac['colrSettingCursor']+statMac['tDiff'], 0, 1)
        drawSettingColor(statMac['colrSettingCursor'], statMac['lamps'][0], statMac['targetBulb'])

    statMac['tDiff'] = 3.14159/statMac['fps']
    for i in range(len(statMac['lamps'])):
        statMac['lamps'][i].updateBulbs(statMac['tDiff']/2)

    #glFlush()
    glutSwapBuffers()

    framerate()

def idleWindowOpen():
    #lamps[0].updateBulbs(constrain(60.0/fps, 1, 2.4))
    glutPostRedisplay()

def idleWindowMinimized():
    pass

# change view angle
# Respond to user input from "special" keys
def special(k, x, y):
    global statMac

    if k == GLUT_KEY_LEFT:
        statMac['lamps'][0].setAngle(statMac['lamps'][0].getAngle() + 5)
    elif k == GLUT_KEY_RIGHT:
        statMac['lamps'][0].setAngle(statMac['lamps'][0].getAngle() - 5)
    elif k == GLUT_KEY_UP:
        statMac['lamps'][0].setNumBulbs(statMac['lamps'][0].getNumBulbs()+1)
    elif k == GLUT_KEY_DOWN:
        statMac['lamps'][0].setNumBulbs(statMac['lamps'][0].getNumBulbs()-1)
    elif k == GLUT_KEY_F11:
        if statMac['isFullScreen'] == False:
            statMac['windowPosX'] = glutGet(GLUT_WINDOW_X)
            statMac['windowPosY'] = glutGet(GLUT_WINDOW_Y)
            statMac['windowDimW'] = glutGet(GLUT_WINDOW_WIDTH)
            statMac['windowDimH'] = glutGet(GLUT_WINDOW_HEIGHT)
            statMac['isFullScreen'] = True
            glutFullScreen()
        elif statMac['isFullScreen'] == True:
            glutPositionWindow(statMac['windowPosX'], statMac['windowPosY'])
            glutReshapeWindow(statMac['windowDimW'], statMac['windowDimH'])
            statMac['isFullScreen'] = False
    elif k == GLUT_KEY_F1:
        if statMac['targetScreen'] == 0:
            statMac['targetScreen'] = 1
            statMac['targetBulb'] = 0
        elif statMac['targetScreen'] == 1:
            statMac['targetScreen'] = 0
    
    if k == GLUT_KEY_F12:
        statMac['frameLimit'] = not statMac['frameLimit']
        print("frameLimit is now {}".format("ON" if statMac['frameLimit'] else "OFF"))

    else:
        return
    glutPostRedisplay()

def key(ch, x, y):
    global statMac
    if ch == as_8_bit('q'):
        sys.exit(0)
    if ord(ch) == 27: # ESC
        sys.exit(0)

    if ch == as_8_bit('a'):
        if statMac['lamps'][0].getArn() == 0:
            statMac['lamps'][0].setArn(1)
        elif statMac['lamps'][0].getArn() == 1:
            statMac['lamps'][0].setArn(0)

    if ch == as_8_bit('h'):
        statMac['wereColorsTouched'] = False
        statMac['targetScreen'] = 0

    if ch == as_8_bit(']'):
        statMac['numHues'] += 2
        if statMac['numHues'] > 14:
            statMac['numHues'] = 14

    if ch == as_8_bit('['):
        statMac['numHues'] -= 2
        if statMac['numHues'] < 10:
            statMac['numHues'] = 10

    if ch == as_8_bit('m'):
        glutIconifyWindow()

# new window size or exposure
# this function is called everytime the window is resized
def reshape(width, height):
    global statMac

    if height > 0:
        statMac['w2h'] = width/height
    else:
        statMac['w2h'] = 1
    statMac['wx'] = width
    statMac['wy'] = height
    statMac['windowDimW'] = width
    statMac['windowDimH'] = height
    statMac['windowPosX'] = glutGet(GLUT_WINDOW_X)
    statMac['windowPosY'] = glutGet(GLUT_WINDOW_Y)
    glViewport(0, 0, width, height)
    #glMatrixMode(GL_PROJECTION)
    #glLoadIdentity()
    #glOrtho(-1.0*statMac['w2h'], 1.0*statMac['w2h'], -1.0, 1.0, -1.0, 1.0) 
    #glMatrixMode(GL_MODELVIEW)
    #glLoadIdentity()

# Only Render if the window (any pixel of it at all) is visible
def visible(vis):
    if vis == GLUT_VISIBLE:
        glutIdleFunc(idleWindowOpen)
    else:
        glutIdleFunc(idleWindowMinimized)

# Equivalent to "main()" in C/C++
if __name__ == '__main__':
    global statMac
    statMac = {}
    print("Initializing...")
    statMac['faceColor']        = (0.3, 0.3, 0.3)
    statMac['detailColor']      = (0.9, 0.9, 0.9)
    statMac['tStart']           = time.time()
    statMac['t0']               = time.time()
    statMac['frames']           = 0
    statMac['lamps']            = []
    statMac['screen']           = 0
    statMac['lightOn']          = False
    statMac['fps']              = 60
    statMac['windowPosX']       = 0
    statMac['windowPosY']       = 0
    statMac['windowDimW']       = 800
    statMac['windowDimH']       = 480
    statMac['cursorX']          = 0
    statMac['cursorY']          = 0
    statMac['isFullScreen']     = False
    statMac['isAnimating']      = False
    statMac['wx']               = 0
    statMac['wy']               = 0
    statMac['targetScreen']     = 0
    statMac['touchState']       = 1
    statMac['currentState']     = 1
    statMac['prvState']         = statMac['touchState']
    statMac['targetBulb']       = 0
    statMac['frameLimit']       = True
    statMac['someVar']          = 0
    statMac['someInc']          = 0.1
    statMac['features']         = 4
    statMac['numHues']          = 12
    statMac['currentHue']       = None
    statMac['currentSat']       = None
    statMac['currentVal']       = None
    statMac['prevHue']          = None
    statMac['prevVal']          = None
    statMac['prevSat']          = None
    statMac['wereColorsTouched']    = False
    statMac['colrSettingCursor']    = 0.0
    statMac['interactionCursor']    = 0.0
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE)
    #glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)

    glutInitWindowPosition(statMac['windowPosX'], statMac['windowPosY'])
    glutInitWindowSize(statMac['windowDimW'], statMac['windowDimH'])

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
