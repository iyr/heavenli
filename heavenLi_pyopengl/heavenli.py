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
from platform import machine
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

print("Loading Serial...")
try:
    import serial
    print("Done!")
except:
    print("Could not load serial library")

def TXstring(message):
    try:
        if (stateMach['CircuitPlayground'].isOpen()):
            stateMach['CircuitPlayground'].write(message)
    except:
        pass
        #print("Error sending color")

def updateLEDS():
    try:
        tmc = stateMach['lamps'][0].getBulbtRGB(stateMach['curBulb'])
    except:
        tmc = (0.5, 0.5, 0.5)
    tmr = int(tmc[0] * 127)
    tmg = int(tmc[1] * 127)
    tmb = int(tmc[2] * 127)
    tmn = int(stateMach['lamps'][0].getNumBulbs())
    tmq = int(stateMach['curBulb'])
    stateMach['curBulb'] += 1
    if (stateMach['curBulb'] >= tmn):
        stateMach['curBulb'] = 0

    tmm = bytearray([
        tmn, 
        tmq, 
        tmr, 
        tmg, 
        tmb])
    TXstring(tmm)


def init():
    global stateMach
    try:
        print("Making Serial Object...")
        #stateMach['CircuitPlayground'] = serial.Serial('COM8', 57600)
        stateMach['CircuitPlayground'] = serial.Serial('/dev/serial/by-id/usb-Adafruit_Feather_32u4-if00', 57600)
        stateMach['CircuitPlayground'].open()
        print("Done!")
    except:
        print("could not establish serial uart connection :(")
    stateMach['curBulb'] = 0

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)

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
    demo = Lamp()
    stateMach['lamps'].append(demo)
    print("Initialization Finished")

def framerate():
    global stateMach
    t = time.time()
    stateMach['frames'] += 1.0
    seconds = t - stateMach['t0']
    stateMach['someVar'] += stateMach['someInc']
    if (stateMach['someVar'] > 100) or (stateMach['someVar'] < 0):
        stateMach['someInc'] = -stateMach['someInc']

    try:
        stateMach['fps'] = stateMach['frames']/seconds
    except:
        print("Too Fast, Too Quick!!")

    if t - stateMach['t1'] >= 0.5:
        #updateLEDS()
        stateMach['t1'] = t

    if t - stateMach['t0'] >= 1.0:
        #updateLEDS()
        print("%.0f frames in %3.1f seconds = %6.3f FPS" % (stateMach['frames'],seconds,stateMach['fps']))
        stateMach['t0'] = t
        stateMach['frames'] = 0
        stateMach['second'] = datetime.datetime.now().second
        stateMach['minute'] = datetime.datetime.now().minute + stateMach['second']/60
        stateMach['hour'] = datetime.datetime.now().hour + stateMach['minute']/60
        if (stateMach['hour'] > 11):
            stateMach['hour'] -= 12
    if stateMach['frameLimit'] and (stateMach['fps'] > 60):
        time.sleep(2*float(stateMach['fps'])/10000.0)

def drawBackground(Light = 0 # Currently Selected Lamp, Space, or *
        ):
    global stateMach
    if (stateMach['lamps'][Light].getArn() == 0):
         drawHomeCircle(0.0, 0.0, 
                 stateMach['wx'], stateMach['wy'], 
                 stateMach['lamps'][Light].getNumBulbs(), 
                 stateMach['lamps'][Light].getAngle(), 
                 stateMach['w2h'],
                 stateMach['lamps'][Light].getBulbsRGB());

    elif (stateMach['lamps'][Light].getArn() == 1):
         drawHomeLinear(0.0, 0.0, 
                 stateMach['wx'], stateMach['wy'],
                 stateMach['lamps'][Light].getNumBulbs(), 
                 stateMach['lamps'][Light].getAngle(), 
                 stateMach['w2h'],
                 stateMach['lamps'][Light].getBulbsRGB());

def drawHome():
    global stateMach

    iconSize = 0.15
    drawClock(
            0.0, 0.0,
            stateMach['hour'],
            stateMach['minute'],
            1.0, stateMach['w2h'], 
            stateMach['faceColor'],
            stateMach['detailColor'])

    # We are at the home screen
    if (stateMach['screen'] == 0) and (stateMach['touchState'] != stateMach['prvState']):
        dBias = min(stateMach['wx'], stateMach['wy'])/2
        if watchPoint(stateMach['wx'], stateMach['wy'], dBias):
            stateMach['lightOn'] = not stateMach['lightOn']
            for i in range(len(stateMach['lamps'])):
                stateMach['lamps'][i].setMainLight(stateMach['lightOn'])

    buttons = drawBulbButton(
            stateMach['lamps'][0].getArn(),
            stateMach['lamps'][0].getNumBulbs(),
            stateMach['lamps'][0].getAngle(),
            iconSize*2.66,
            stateMach['faceColor'],
            stateMach['detailColor'],
            stateMach['lamps'][0].getBulbsRGB(),
            stateMach['w2h'])

    for i in range(len(buttons)):
        if (stateMach['screen'] == 0) and (stateMach['touchState'] != stateMach['prvState']):
            if (watchPoint(
                mapRanges(buttons[i][0], -1.0*stateMach['w2h'], 1.0*stateMach['w2h'], 0, stateMach['wx']*2), 
                mapRanges(buttons[i][1],      1.0,    -1.0, 0, stateMach['wy']*2),
                min(stateMach['wx'], stateMach['wy'])*0.5*0.3)):
                stateMach['targetScreen'] = 1
                stateMach['targetBulb'] = i
                stateMach['prevHue'] = stateMach['lamps'][0].getBulbHSV(i)[0]
                stateMach['prevSat'] = stateMach['lamps'][0].getBulbHSV(i)[1]
                stateMach['prevVal'] = stateMach['lamps'][0].getBulbHSV(i)[2]

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

    if (stateMach['lamps'][0].getArn() == 0):
        drawIconCircle(0.75, 0.75, 
                iconSize*0.85, 
                2,
                ( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100)),
                stateMach['lamps'][0].getNumBulbs(), 
                stateMach['lamps'][0].getAngle(), 
                stateMach['w2h'], 
                stateMach['lamps'][0].getBulbsRGB())

    if (stateMach['lamps'][0].getArn() == 1):
        drawIconLinear(0.75, 0.75, 
                iconSize*0.85, 
                2,
                ( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100)),
                stateMach['lamps'][0].getNumBulbs(), 
                stateMach['lamps'][0].getAngle(), 
                stateMach['w2h'], 
                stateMach['lamps'][0].getBulbsRGB())

    if (watchPoint(
        mapRanges(0.75, -1.0,  1.0, 0, stateMach['wx']*2),
        mapRanges(0.75,  1.0, -1.0, 0, stateMach['wy']*2),
        min(stateMach['wx'], stateMach['wy'])*0.2)):
            stateMach['targetScreen'] = 1
            stateMach['targetBulb'] = stateMach["lamps"][0].getNumBulbs()
            stateMach['prevHue'] = stateMach['lamps'][0].getBulbHSV(0)[0]
            stateMach['prevSat'] = stateMach['lamps'][0].getBulbHSV(0)[1]
            stateMach['prevVal'] = stateMach['lamps'][0].getBulbHSV(0)[2]
            for i in range(stateMach['targetBulb']):
                stateMach['prevHues'][i] = stateMach['lamps'][0].getBulbHSV(i)[0]
                stateMach['prevSats'][i] = stateMach['lamps'][0].getBulbHSV(i)[1]
                stateMach['prevVals'][i] = stateMach['lamps'][0].getBulbHSV(i)[2]

def drawSettingColor(cursor, targetLamp, targetBulb):
    global stateMach
    if (stateMach['targetBulb'] == targetLamp.getNumBulbs()):
        tmcHSV = targetLamp.getBulbtHSV(0)
        if (stateMach['currentHue'] == None):
            stateMach['currentHue'] = targetLamp.getBulbHSV(0)[0]
        if (stateMach['currentSat'] == None):
            stateMach['currentSat'] = targetLamp.getBulbHSV(0)[1]
        if (stateMach['currentVal'] == None):
            stateMach['currentVal'] = targetLamp.getBulbHSV(0)[2]
    else:
        tmcHSV = targetLamp.getBulbtHSV(stateMach['targetBulb'])
        if (stateMach['currentHue'] == None):
            stateMach['currentHue'] = targetLamp.getBulbHSV(targetBulb)[0]
        if (stateMach['currentSat'] == None):
            stateMach['currentSat'] = targetLamp.getBulbHSV(targetBulb)[1]
        if (stateMach['currentVal'] == None):
            stateMach['currentVal'] = targetLamp.getBulbHSV(targetBulb)[2]
    acbic = animCurveBounce(1.0-cursor)
    acic = animCurve(1.0-cursor)
    acbc = animCurveBounce(cursor)
    acc = animCurve(cursor)
    faceColor = (stateMach['faceColor'][0]*acc, 
            stateMach['faceColor'][1]*acc,
            stateMach['faceColor'][2]*acc)
    detailColor = (stateMach['detailColor'][0]*acc, 
            stateMach['detailColor'][1]*acc,
            stateMach['detailColor'][2]*acc)
    cmx = 0.15
        
    if (stateMach['wereColorsTouched']):
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
            stateMach['w2h'])

    if (stateMach['lamps'][0].getArn() == 0):
        drawIconCircle(0.75, 0.75, 
                iconSize*0.85*pow(acc, 4), 
                2,
                ( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100)),
                stateMach['lamps'][0].getNumBulbs(), 
                stateMach['lamps'][0].getAngle(), 
                stateMach['w2h'], 
                stateMach['lamps'][0].getBulbsRGB())

    if (stateMach['lamps'][0].getArn() == 1):
        drawIconLinear(0.75, 0.75, 
                iconSize*0.85*pow(acc, 4), 
                2,
                ( 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100), 0.9*(stateMach['someVar']/100)),
                stateMach['lamps'][0].getNumBulbs(), 
                stateMach['lamps'][0].getAngle(), 
                stateMach['w2h'], 
                stateMach['lamps'][0].getBulbsRGB())

    limit = 0.85
    if (cursor < limit):
        drawGranRocker(
                0.0, -0.91*acbic,
                faceColor,
                detailColor,
                stateMach['numHues'],
                0.0,
                stateMach['w2h'],
                0.30*acic,
                stateMach['tDiff'])
    
    drawClock(
            0.0, 0.0,
            stateMach['hour'],
            stateMach['minute'],
            1.0+acic*0.75, 
            stateMach['w2h'], 
            faceColor,
            #detailColor)
            tuple([acc*x for x in detailColor]))

    if (cursor >= limit):
        drawGranRocker(
                0.0, -0.91*acbic,
                faceColor,
                detailColor,
                stateMach['numHues'],
                0.0,
                stateMach['w2h'],
                0.30*acic,
                stateMach['tDiff'])

    # Watch Granularity Rocker for Input
    if (watchPoint(
        mapRanges(0.3*24.0/36.0, -1.0*stateMach['w2h'], 1.0*stateMach['w2h'],   0, stateMach['wx']*2),
        mapRanges(0.0-0.91,  1.0               ,-1.0               ,   0, stateMach['wy']*2),
        min(stateMach['wx'],stateMach['wy']*(12.0/36.0)*0.3) )):
            stateMach['numHues'] += 2
            if stateMach['numHues'] > 14:
                stateMach['numHues'] = 14
    # Watch Granularity Rocker for Input
    if (watchPoint(
        mapRanges(-0.3*24.0/36.0, -1.0*stateMach['w2h'], 1.0*stateMach['w2h'], 0, stateMach['wx']*2),
        mapRanges(-0.91, 1.0, -1.0, 0, stateMach['wy']*2),
        min(stateMach['wx'],stateMach['wy']*(12.0/36.0)*0.3) )):
            stateMach['numHues'] -= 2
            if stateMach['numHues'] < 10:
                stateMach['numHues'] = 10

    # Draw Ring of Dots with different hues
    hueButtons = drawHueRing(
            stateMach['currentHue'], 
            stateMach['numHues'], 
            selectRingColor, 
            stateMach['w2h'], 
            acbic, 
            stateMach['tDiff'],
            stateMach['interactionCursor'])

    for i in range(len(hueButtons)):
        tmr = 1.0
        if (stateMach['w2h'] <= 1.0):
            hueButtons[i] = (
                    hueButtons[i][0]*stateMach['w2h'], 
                    hueButtons[i][1]*stateMach['w2h'], 
                    hueButtons[i][2])
            tmr = stateMach['w2h']

        xLim = mapRanges(hueButtons[i][0], -1.0*stateMach['w2h'], 1.0*stateMach['w2h'],    0, stateMach['wx']*2)
        yLim = mapRanges(hueButtons[i][1],  1.0,               -1.0,                   0, stateMach['wy']*2)
        if (watchPoint(
            xLim, yLim,
            min(stateMach['wx'], stateMach['wy'])*0.15*(12.0/float(stateMach['numHues'])) )):
            stateMach['wereColorsTouched'] = True
            stateMach['currentHue'] = hueButtons[i][2]
            tmcHSV = (
                    stateMach['currentHue'], 
                    stateMach['currentSat'], 
                    stateMach['currentVal'])

            if (stateMach['targetBulb'] == targetLamp.getNumBulbs()):
                targetLamp.setBulbstHSV(tmcHSV)
            else:
                targetLamp.setBulbtHSV(stateMach['targetBulb'], tmcHSV)

    # Draw Triangle of Dots with different brightness/saturation
    satValButtons = drawColrTri(
            stateMach['currentHue'], 
            stateMach['currentSat'], 
            stateMach['currentVal'],
            int(stateMach['numHues']/2), 
            selectRingColor,
            stateMach['w2h'], acbic, 
            stateMach['tDiff'])

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
        xLim = mapRanges(
                satValButtons[i][0], 
                -1.0*stateMach['w2h'], 
                1.0*stateMach['w2h'], 
                0, 
                stateMach['wx']*2)

        # Map relative y-Coord to screen y-Coord
        yLim = mapRanges(
                satValButtons[i][1],      
                1.0,    
                -1.0, 
                0, 
                stateMach['wy']*2)

        if (watchPoint(
            xLim, yLim,
            min(stateMach['wx'], stateMach['wy'])*0.073)):
            stateMach['wereColorsTouched'] = True
            stateMach['currentSat'] = satValButtons[i][2]
            stateMach['currentVal'] = satValButtons[i][3]
            tmcHSV = (
                    stateMach['currentHue'], 
                    stateMach['currentSat'], 
                    stateMach['currentVal'])
            if (stateMach['targetBulb'] == targetLamp.getNumBulbs()):
                targetLamp.setBulbstHSV(tmcHSV)
            else:
                targetLamp.setBulbtHSV(stateMach['targetBulb'], tmcHSV)

    if ( stateMach['wereColorsTouched'] ):
        extraColor = colorsys.hsv_to_rgb(
                stateMach['currentHue'], 
                stateMach['currentSat'], 
                stateMach['currentVal'])
    else:
        extraColor = stateMach['detailColor']

    drawConfirm(
            0.75-0.4*(1.0-acbic), 
            -0.75-0.5*acbc, 
            0.2*(1.0-acbc), stateMach['w2h'], 
            stateMach['faceColor'], 
            extraColor, 
            stateMach['detailColor']);

    # Watch Confirm Button for input
    if (watchPoint(
        mapRanges( 0.75, -1.0,  1.0, 0, stateMach['wx']*2),
        mapRanges(-0.75,  1.0, -1.0, 0, stateMach['wy']*2),
        min(stateMach['wx'], stateMach['wy'])*0.2)):
        stateMach['wereColorsTouched'] = False
        if (stateMach['targetBulb'] == targetLamp.getNumBulbs()):
            targetLamp.setBulbstHSV( (
                stateMach['currentHue'], 
                stateMach['currentSat'], 
                stateMach['currentVal'] ) )
        else:
            targetLamp.setBulbtHSV(stateMach['targetBulb'], (
                stateMach['currentHue'], 
                stateMach['currentSat'], 
                stateMach['currentVal'] ) )
        stateMach['targetScreen'] = 0

    if ( stateMach['wereColorsTouched'] ):
        if (stateMach['targetBulb'] == targetLamp.getNumBulbs()):
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

    # Draw Back Button
    drawArrow(
            -0.75+0.4*(1.0-acbic), 
            -0.75-0.5*acbc, 
            180.0,
            0.2*(1.0-acbc), stateMach['w2h'], 
            stateMach['faceColor'], 
            extraColor, 
            stateMach['detailColor']);
    if (watchPoint(
        mapRanges(-0.75, -1.0,  1.0, 0, stateMach['wx']*2),
        mapRanges(-0.75,  1.0, -1.0, 0, stateMach['wy']*2),
        min(stateMach['wx'], stateMach['wy'])*0.2)):
        stateMach['wereColorsTouched'] = False
        if (stateMach['targetBulb'] == targetLamp.getNumBulbs()):
            for i in range(targetLamp.getNumBulbs()):
                targetLamp.setBulbtHSV(i, (
                    stateMach['prevHues'][i], 
                    stateMach['prevSats'][i], 
                    stateMach['prevVals'][i] ) )
        else:
            targetLamp.setBulbtHSV(stateMach['targetBulb'], (
                stateMach['prevHue'], 
                stateMach['prevSat'], 
                stateMach['prevVal'] ) )
        stateMach['targetScreen'] = 0

def watchPoint(px, py, pr):
    global stateMach
    if (1.0 >= pow((stateMach['cursorX']-px/2), 2) / pow(pr/2, 2) + pow((stateMach['cursorY']-py/2), 2) / pow(pr/2, 2)):
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
    glLoadIdentity()
    updateLEDS()

    drawBackground(0)
    #if (stateMach['interactionCursor'] > 0.0):
        #print(stateMach['interactionCursor'])

    #stateMach['tDiff'] = 0.70568/stateMach['fps']
    #stateMach['tDiff'] = 1.30568/stateMach['fps']
    stateMach['tDiff'] = 2.71828/stateMach['fps']
    #stateMach['tDiff'] = 3.14159/stateMach['fps']
    #stateMach['tDiff'] = 6.28318/stateMach['fps']
    if (stateMach['currentState'] == 0 or stateMach['prvState'] == 0):
        stateMach['interactionCursor'] = 1.0
    else:
        stateMach['interactionCursor'] = constrain(stateMach['interactionCursor'] - stateMach['tDiff'], 0.0, 1.0)

    if (stateMach['targetScreen'] == 0):
        if (stateMach['colrSettingCursor'] > 0):
            stateMach['colrSettingCursor'] = constrain(stateMach['colrSettingCursor']-stateMach['tDiff'], 0, 1)
            drawSettingColor(stateMach['colrSettingCursor'], stateMach['lamps'][0], stateMach['targetBulb'])
        if (stateMach['targetScreen'] == 0) and (stateMach['colrSettingCursor'] == 0):
            drawHome()

    elif (stateMach['targetScreen'] == 1):
        if (stateMach['colrSettingCursor'] < 1):
            stateMach['colrSettingCursor'] = constrain(stateMach['colrSettingCursor']+stateMach['tDiff'], 0, 1)
        drawSettingColor(stateMach['colrSettingCursor'], stateMach['lamps'][0], stateMach['targetBulb'])

    stateMach['tDiff'] = 3.14159/stateMach['fps']
    for i in range(len(stateMach['lamps'])):
        stateMach['lamps'][i].updateBulbs(stateMach['tDiff']/2)

    #glFlush()
    glutSwapBuffers()

    framerate()
    updateLEDS()

def idleWindowOpen():
    glutPostRedisplay()

def idleWindowMinimized():
    pass

# change view angle
# Respond to user input from "special" keys
def special(k, x, y):
    global stateMach

    if k == GLUT_KEY_LEFT:
        stateMach['lamps'][0].setAngle(stateMach['lamps'][0].getAngle() + 5)
    elif k == GLUT_KEY_RIGHT:
        stateMach['lamps'][0].setAngle(stateMach['lamps'][0].getAngle() - 5)
    elif k == GLUT_KEY_UP:
        stateMach['lamps'][0].setNumBulbs(stateMach['lamps'][0].getNumBulbs()+1)
    elif k == GLUT_KEY_DOWN:
        stateMach['lamps'][0].setNumBulbs(stateMach['lamps'][0].getNumBulbs()-1)
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
    if ch == as_8_bit('q'):
        sys.exit(0)
    if ord(ch) == 27: # ESC
        sys.exit(0)

    if ch == as_8_bit('a'):
        if stateMach['lamps'][0].getArn() == 0:
            stateMach['lamps'][0].setArn(1)
        elif stateMach['lamps'][0].getArn() == 1:
            stateMach['lamps'][0].setArn(0)

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
    stateMach['faceColor']          = (0.3, 0.3, 0.3)
    stateMach['detailColor']        = (0.9, 0.9, 0.9)
    stateMach['tStart']             = time.time()
    stateMach['t0']                 = time.time()
    stateMach['t1']                 = time.time()
    stateMach['frames']             = 0
    stateMach['lamps']              = []
    stateMach['screen']             = 0
    stateMach['lightOn']            = False
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
    stateMach['currentHue']         = None
    stateMach['currentSat']         = None
    stateMach['currentVal']         = None
    stateMach['prevHue']            = None
    stateMach['prevVal']            = None
    stateMach['prevSat']            = None
    stateMach['wereColorsTouched']  = False
    stateMach['colrSettingCursor']  = 0.0
    stateMach['interactionCursor']  = 0.0
    glutInit(sys.argv)

    # Disable anti-aliasing if running on a Raspberry Pi Zero
    if (machine() == "armv6l" or machine() == "armv7l"):
        print("Disabling Antialiasing")
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)
    else:
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE)

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
