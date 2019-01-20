#! /usr/bin/env python
#import cProfile
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
import os
print("Done!")

print("Loading heavenLi Utilities...")
import colorsys
from animUtils import *
from drawArn import *
from drawButtons import *
#from drawUtils import *
from lampClass import *
from rangeUtils import *
print("Done!")

tStart = t0 = time.time()
frames = 0
lamps = []
screen = 0
lightOn = False
fps = 60
windowPosX = 0
windowPosY = 0
windowDimW = 800
windowDimH = 480
#windowDimW = 320
#windowDimH = 240
cursorX = 0
cursorY = 0
isFullScreen = False
isAnimating = False
wx = 0
wy = 0
colrSettingCursor = 0
targetScreen = 0
touchState = 0
targetBulb = 0
frameLimit = True
someVar = 0
someInc = 0.1
features = 4
numHues = 12
#demo = Lamp()

def init():
    global wx, wy, w2h, lamps

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)

    wx = glutGet(GLUT_WINDOW_WIDTH)
    wy = glutGet(GLUT_WINDOW_HEIGHT)
    w2h = wx/wy
    demo = Lamp()
    lamps.append(demo)

def framerate():
    global t0, frames, w2h, fps, someVar, someInc
    t = time.time()
    frames += 1.0
    seconds = t - t0
    someVar += someInc
    if (someVar > 100) or (someVar < 0):
        someInc = -someInc

    try:
        fps = frames/seconds
    except:
        print("Too Fast, Too Quick!!")
    if t - t0 >= 1.0:
        print("%.0f frames in %3.1f seconds = %6.3f FPS" % (frames,seconds,fps))
        t0 = t
        frames = 0
    if frameLimit and (fps > 60):
        time.sleep(2*float(fps)/10000.0)

def drawBackground(Light = 0 # Currently Selected Lamp, Space, or *
        ):
    global wx, wy
    if (lamps[Light].getArn() == 0):
         drawHomeCircle(0.0, 0.0, 
                 wx, wy, 
                 lamps[Light].getNumBulbs(), 
                 lamps[Light].getAngle(), 
                 w2h,
                 lamps[Light].getBulbsRGB());

    elif (lamps[Light].getArn() == 1):
         drawHomeLinear(0.0, 0.0, 
                 wx, wy,
                 lamps[Light].getNumBulbs(), 
                 lamps[Light].getAngle(), 
                 w2h,
                 lamps[Light].getBulbsRGB());

prvState = touchState

def drawHome():
    global lamps, wx, wy, w2h, screen, touchState, lightOn, prvState, targetScreen, targetBulb, colrSettingCursor, features

    iconSize = 0.15
    drawClock(
            12*(someVar/100),
            60*(1.0-(someVar/100)), 1.0, w2h, (0.95, 0.95, 0.95), (0.3, 0.3, 0.3))

    # We are at the home screen
    if (screen == 0) and (touchState != prvState):
        #wx = glutGet(GLUT_WINDOW_WIDTH)
        #wy = glutGet(GLUT_WINDOW_HEIGHT)
        dBias = min(wx, wy)/2
        if watchPoint(wx, wy, dBias):
            lightOn = not lightOn
            for i in range(len(lamps)):
                lamps[i].setMainLight(lightOn)

    buttons = drawBulbButton(
            lamps[0].getArn(),
            lamps[0].getNumBulbs(),
            lamps[0].getAngle(),
            iconSize*2.66,
            (0.3, 0.3, 0.3),
            (0.8, 0.8, 0.8),
            lamps[0].getBulbsRGB(),
            w2h)

    for i in range(len(buttons)):
        if (screen == 0) and (touchState != prvState):
            if (watchPoint(
                mapRanges(buttons[i][0], -1.0*w2h, 1.0*w2h, 0, wx*2), 
                mapRanges(buttons[i][1],      1.0,    -1.0, 0, wy*2),
                min(wx, wy)*0.5*0.3)):
                targetScreen = 1
                targetBulb = i
                prevHue = lamps[0].getBulbHSV(i)[0]
                prevSat = lamps[0].getBulbHSV(i)[1]
                prevBri = lamps[0].getBulbHSV(i)[2]

    #drawIconCircle(0.75-0.25*(someVar/100), 0.75, 
    drawIconCircle(0.75, 0.75, 
            iconSize*0.85, 
            4,
            #features,
            ( 0.9*(someVar/100), 0.9*(someVar/100), 0.9*(someVar/100)),
            lamps[0].getNumBulbs(), 
            lamps[0].getAngle(), 
            w2h, 
            lamps[0].getBulbsRGB())

    drawIconCircle(0.37, 0.75, 
            iconSize*0.85, 
            3,
            #features,
            ( 0.9*(someVar/100), 0.9*(someVar/100), 0.9*(someVar/100)),
            lamps[0].getNumBulbs(), 
            lamps[0].getAngle(), 
            w2h, 
            lamps[0].getBulbsRGB())

    drawIconCircle(0.0, 0.75, 
            iconSize*0.85, 
            4,
            #features,
            ( 0.9*(someVar/100), 0.9*(someVar/100), 0.9*(someVar/100)),
            lamps[0].getNumBulbs(), 
            lamps[0].getAngle(), 
            w2h, 
            lamps[0].getBulbsRGB())

    drawIconCircle(-0.37, 0.75, 
            iconSize*0.85, 
            2,
            #features,
            ( 0.9*(someVar/100), 0.9*(someVar/100), 0.9*(someVar/100)),
            lamps[0].getNumBulbs(), 
            lamps[0].getAngle(), 
            w2h, 
            lamps[0].getBulbsRGB())

    drawIconCircle(-0.75, 0.75, 
            iconSize*0.85, 
            1,
            #features,
            ( 0.9*(someVar/100), 0.9*(someVar/100), 0.9*(someVar/100)),
            lamps[0].getNumBulbs(), 
            lamps[0].getAngle(), 
            w2h, 
            lamps[0].getBulbsRGB())

    #cProfile.run('drawIconCircle(0.75, 0.75, 0.15, 0.15, lamps[0].getNumBulbs(), lamps[0].getAngle(), w2h, lamps[0].getBulbsRGB())')

    drawIconLinear(0.75, -0.75, 
            iconSize*0.85, 
            4,
            #features,
            ( 0.9*(someVar/100), 0.9*(someVar/100), 0.9*(someVar/100)),
            lamps[0].getNumBulbs(), 
            lamps[0].getAngle(), 
            w2h, 
            lamps[0].getBulbsRGB())

    drawIconLinear(0.37, -0.75, 
            iconSize*0.85, 
            3,
            #features,
            ( 0.9*(someVar/100), 0.9*(someVar/100), 0.9*(someVar/100)),
            lamps[0].getNumBulbs(), 
            lamps[0].getAngle(), 
            w2h, 
            lamps[0].getBulbsRGB())

    drawIconLinear(0.0, -0.75, 
            iconSize*0.85, 
            4,
            #features,
            ( 0.9*(someVar/100), 0.9*(someVar/100), 0.9*(someVar/100)),
            lamps[0].getNumBulbs(), 
            lamps[0].getAngle(), 
            w2h, 
            lamps[0].getBulbsRGB())

    drawIconLinear(-0.37, -0.75, 
            iconSize*0.85, 
            2,
            #features,
            ( 0.9*(someVar/100), 0.9*(someVar/100), 0.9*(someVar/100)),
            lamps[0].getNumBulbs(), 
            lamps[0].getAngle(), 
            w2h, 
            lamps[0].getBulbsRGB())

    drawIconLinear(-0.75, -0.75, 
            iconSize*0.85, 
            1,
            #features,
            ( 0.9*(someVar/100), 0.9*(someVar/100), 0.9*(someVar/100)),
            lamps[0].getNumBulbs(), 
            lamps[0].getAngle(), 
            w2h, 
            lamps[0].getBulbsRGB())

    #cProfile.run('drawIconLinear(-0.75, -0.75, 0.15*0.875, 0.15*0.875, lamps[0].getNumBulbs(), lamps[0].getAngle(), w2h, lamps[0].getBulbsRGB())')

__bulbsCurrentHSB = []
__pickerVerts = []
__pickerColrs = []
__ringVerts = []
__ringColrs = []
__ringPoints = []
currentHue = None
currentVal = None
currentSat = None
prevHue = None
prevBri = None
prevSat = None
wereColorsTouched = False
        
def drawSettingColor(cursor, targetLamp, targetBulb, w2h):
    global currentVal, currentSat, currentHue, wereColorsTouched, __pickerVerts, __pickerColrs, __ringVerts, __ringColrs, __ringPoints
    tmcHSV = targetLamp.getBulbtHSV(targetBulb)
    acbic = animCurveBounce(1.0-cursor)
    acic = animCurve(1.0-cursor)
    acbc = animCurveBounce(cursor)
    acc = animCurve(cursor)
    #cmx = (width >= height ? mx : cx/4)
    #global wx, wy, __bulbsCurrentHSB
    cmx = 0.15
    if (currentHue == None):
        currentHue = targetLamp.getBulbHSV(targetBulb)[0]
    if (currentSat == None):
        currentSat = targetLamp.getBulbHSV(targetBulb)[1]
    if (currentVal == None):
        currentVal = targetLamp.getBulbHSV(targetBulb)[2]

    if (cursor != 0.0) and (cursor != 1.0):
        pass

    iconSize = 0.15
    # Draw circularly arranged bulb buttons
    drawBulbButton(
            targetLamp.getArn(),
            targetLamp.getNumBulbs(),
            targetLamp.getAngle(),
            iconSize*2.66*pow(acc, 4),
            (0.3, 0.3, 0.3),
            (0.8, 0.8, 0.8),
            targetLamp.getBulbsRGB(),
            w2h)

    drawClock(
            12*(someVar/100),
            60*(1.0-(someVar/100)),
            1.0+acic*0.75, 
            w2h, 
            (0.9*acc, 0.9*acc, 0.9*acc), 
            (0.3*acc, 0.3*acc, 0.3*acc))

    # Draw Ring of Dots with different hues
    hueButtons = drawHueRing(numHues, w2h, acbic)
    for i in range(numHues):
        tmr = 1.0
        if (w2h <= 1.0):
            hueButtons[i] = (hueButtons[i][0]*w2h, hueButtons[i][1]*w2h, hueButtons[i][2])
            tmr = w2h

        if (watchPoint(
            mapRanges(hueButtons[i][0], -1.0*w2h, 1.0*w2h, 0, wx*2), 
            mapRanges(hueButtons[i][1],      1.0,    -1.0, 0, wy*2),
            min(wx, wy)*0.15*(12.0/float(numHues)) )):
            wereColorsTouched = True
            currentHue = hueButtons[i][2]
            tmcHSV = (currentHue, currentSat, currentVal)
            targetLamp.setBulbtHSV(targetBulb, tmcHSV)

    # Draw Triangle of Dots with different brightness/saturation
    satValButtons = drawColrTri(currentHue, int(numHues/2), w2h, acbic)
    for i in range(int( int(numHues/2)*( int(numHues/2) + 1) / 2 )):
        tmr = 1.0
        if (w2h <= 1.0):
            satValButtons[i] = (satValButtons[i][0]*w2h, satValButtons[i][1]*w2h, satValButtons[i][2], satValButtons[i][3])
            tmr = w2h

        if (watchPoint(
            mapRanges(satValButtons[i][0], -1.0*w2h, 1.0*w2h, 0, wx*2), 
            mapRanges(satValButtons[i][1],      1.0,    -1.0, 0, wy*2),
            min(wx, wy)*0.0725)):
            wereColorsTouched = True
            currentSat = satValButtons[i][2]
            currentVal = satValButtons[i][3]
            tmcHSV = (currentHue, currentSat, currentVal)
            targetLamp.setBulbtHSV(targetBulb, tmcHSV)

def watchPoint(px, py, pr):
    global cursorX, cursorY, touchState, prvState
    if (1.0 >= pow((cursorX-px/2), 2) / pow(pr/2, 2) + pow((cursorY-py/2), 2) / pow(pr/2, 2)):
        #os.system('cls' if os.name == 'nt' else "printf '\033c'")
        #print("cursorX: {}, cursorY: {}, px: {:.3f}, py: {:.3f}, pr: {:.3f}, touchState: {}".format(cursorX, cursorY, px, py, pr, touchState))
        if prvState == 0:
            prvState = touchState
            return True
        else:
            prvState = touchState
            return False

def mousePassive(mouseX, mouseY):
    global cursorX, cursorY, touchState
    if (touchState == 0):
        cursorX = mouseX
        cursorY = mouseY
        
def mouseInteraction(button, state, mouseX, mouseY):
    global lightOn, lamps, cursorX, cursorY, wx, wy, touchState, prvState
    # State = 0: button is depressed, low
    # State = 1: button is released, high
    if (state == 0):
        cursorX = mouseX
        cursorY = mouseY
    if (touchState == 1) and (state != 1) and (prvState == 1):
        #glutPostRedisplay()
        #touchState = not touchState
        prvState = not touchState
        return
    elif (touchState == 0) and (state != 0) and (prvState == 0):
        #glutPostRedisplay()
        touchState = not touchState
        prvState = not touchState
        return

# Main screen drawing routine
# Passed to glutDisplayFunc()
# Called with glutPostRedisplay()
def display():
    global colrSettingCursor, targetScreen, targetBulb, fps, lamps
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    #glDisable(GL_LIGHTING)
    drawBackground(0)

    tDiff = 3/fps
    if (targetScreen == 0):
        if (colrSettingCursor > 0):
            colrSettingCursor = constrain(colrSettingCursor-tDiff, 0, 1)
            #cProfile.run('drawSettingColor(colrSettingCursor, lamps[0], targetBulb, w2h)')
            drawSettingColor(colrSettingCursor, lamps[0], targetBulb, w2h)
        if (targetScreen == 0) and (colrSettingCursor == 0):
            drawHome()

    elif (targetScreen == 1):
        if (colrSettingCursor < 1):
            colrSettingCursor = constrain(colrSettingCursor+tDiff, 0, 1)
        #cProfile.run('drawSettingColor(colrSettingCursor, lamps[0], targetBulb, w2h)')
        drawSettingColor(colrSettingCursor, lamps[0], targetBulb, w2h)

    for i in range(len(lamps)):
        lamps[i].updateBulbs(tDiff/2)

    #glFlush()
    glutSwapBuffers()

    framerate()

def idle():
    #lamps[0].updateBulbs(constrain(60.0/fps, 1, 2.4))
    glutPostRedisplay()

# change view angle
# Respond to user input from "special" keys
def special(k, x, y):
    global angB, frameLimit, nz, windowPosX, windowPosY, windowDimW, windowDimH, isFullScreen, targetScreen, targetBulb

    if k == GLUT_KEY_LEFT:
        lamps[0].setAngle(lamps[0].getAngle() + 5)
    elif k == GLUT_KEY_RIGHT:
        lamps[0].setAngle(lamps[0].getAngle() - 5)
    elif k == GLUT_KEY_UP:
        lamps[0].setNumBulbs(lamps[0].getNumBulbs()+1)
    elif k == GLUT_KEY_DOWN:
        lamps[0].setNumBulbs(lamps[0].getNumBulbs()-1)
    elif k == GLUT_KEY_F11:
        if isFullScreen == False:
            windowPosX = glutGet(GLUT_WINDOW_X)
            windowPosY = glutGet(GLUT_WINDOW_Y)
            windowDimW = glutGet(GLUT_WINDOW_WIDTH)
            windowDimH = glutGet(GLUT_WINDOW_HEIGHT)
            isFullScreen = True
            glutFullScreen()
        elif isFullScreen == True:
            glutPositionWindow(windowPosX, windowPosY)
            glutReshapeWindow(windowDimW, windowDimH)
            isFullScreen = False
    elif k == GLUT_KEY_F1:
        if targetScreen == 0:
            targetScreen = 1
            targetBulb = 0
        elif targetScreen == 1:
            targetScreen = 0
    
    if k == GLUT_KEY_F12:
        frameLimit = not frameLimit
        print("frameLimit is now {}".format("ON" if frameLimit else "OFF"))

    else:
        return
    glutPostRedisplay()

def key(ch, x, y):
    global targetScreen, wereColorsTouched, features, numHues
    if ch == as_8_bit('q'):
        sys.exit(0)
    if ord(ch) == 27: # ESC
        sys.exit(0)

    if ch == as_8_bit('a'):
        if lamps[0].getArn() == 0:
            lamps[0].setArn(1)
        elif lamps[0].getArn() == 1:
            lamps[0].setArn(0)

    if ch == as_8_bit('h'):
        wereColorsTouched = False
        targetScreen = 0

    if ch == as_8_bit(']'):
        numHues += 2
        if numHues > 14:
            numHues = 14

    if ch == as_8_bit('['):
        numHues -= 2
        if numHues < 10:
            numHues = 10

    #if ch == as_8_bit('m'):
        #glutIconifyWindow()

# new window size or exposure
# this function is called everytime the window is resized
def reshape(width, height):
    global wx, wy, dBias, w2h

    if height > 0:
        w2h = width/height
    else:
        w2h = 1
    wx = width
    wy = height
    windowDimW = width
    windowDimH = height
    windowPosX = glutGet(GLUT_WINDOW_X)
    windowPosY = glutGet(GLUT_WINDOW_Y)
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0*w2h, 1.0*w2h, -1.0, 1.0, -1.0, 1.0) 
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

# Only Render if the window (any pixel of it at all) is visible
def visible(vis):
    if vis == GLUT_VISIBLE:
        glutIdleFunc(idle)
    else:
        glutIdleFunc(None)

# Equivalent to "main()" in C/C++
if __name__ == '__main__':
    #global windowDimW, windowDimH, windowPosX, windowPosY
    print("Initializing...")
    glutInit(sys.argv)
    #glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)

    glutInitWindowPosition(windowPosX, windowPosY)
    glutInitWindowSize(windowDimW, windowDimH)

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

    glutMainLoop()
