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
from drawUtils import *
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
windowDimW = 300
windowDimH = 300
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
frameLimit = False
someVar = 0
someInc = 0.1
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
    frames += 1
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
                 #wx, wy, 
                 glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT),
                 lamps[Light].getNumBulbs(), 
                 lamps[Light].getAngle(), 
                 w2h,
                 lamps[Light].getBulbsRGB());

    elif (lamps[Light].getArn() == 1):
         drawHomeLinear(0.0, 0.0, 
                 1.0, 1.0, 
                 lamps[Light].getNumBulbs(), 
                 lamps[Light].getAngle(), 
                 w2h,
                 lamps[Light].getBulbsRGB());

prvState = touchState

def drawHome():
    global lamps, wx, wy, w2h, screen, touchState, lightOn, prvState, targetScreen, targetBulb, colrSettingCursor

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
            #(0.3*(1-someVar/100), 0.3*(1-someVar/100), 0.3*(1-someVar/100)),
            #(0.8*(someVar/100), 0.8*(someVar/100), 0.8*(someVar/100)),
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

    drawIconCircle(0.75, 0.75, 
            iconSize, iconSize, 
            lamps[0].getNumBulbs(), lamps[0].getAngle(), 
            w2h, lamps[0].getBulbsRGB())

    #cProfile.run('drawIconCircle(0.75, 0.75, 0.15, 0.15, lamps[0].getNumBulbs(), lamps[0].getAngle(), w2h, lamps[0].getBulbsRGB())')

    drawIconLinear(-0.75, -0.75, 
            iconSize*0.875, iconSize*0.875, 
            lamps[0].getNumBulbs(), lamps[0].getAngle(), 
            w2h, lamps[0].getBulbsRGB())

    #cProfile.run('drawIconLinear(-0.75, -0.75, 0.15*0.875, 0.15*0.875, lamps[0].getNumBulbs(), lamps[0].getAngle(), w2h, lamps[0].getBulbsRGB())')

__bulbsCurrentHSB = []
__pickerVerts = []
__pickerColrs = []
__ringVerts = []
__ringColrs = []
__ringPoints = []
currentHue = 0
currentBri = 0
currentSat = 1
prevHue = None
prevBri = None
prevSat = None
wereColorsTouched = False
        
def drawSettingColor(cursor, targetLamp, targetBulb, w2h):
    global currentBri, currentSat, currentHue, wereColorsTouched, __pickerVerts, __pickerColrs, __ringVerts, __ringColrs, __ringPoints
    tmcl = targetLamp.getBulbRGB(targetBulb)
    tmc = colorsys.rgb_to_hsv(tmcl[0], tmcl[1], tmcl[2])
    acbic = animCurveBounce(1.0-cursor)
    acic = animCurve(1.0-cursor)
    acbc = animCurveBounce(cursor)
    acc = animCurve(cursor)
    #cmx = (width >= height ? mx : cx/4)
    #global wx, wy, __bulbsCurrentHSB
    cmx = 0.15
    tm04 = 0.04
    tm05 = 0.05
    tm06 = 0.06
    tm08 = 0.08
    tm10 = 0.1
    tm13 = 0.13
    tm15 = 0.15
    tm17 = 0.17
    tm20 = 0.2
    tm23 = 0.23
    tm37 = 0.37
    tm30 = 0.3
    tm40 = 0.4
    tm50 = 0.5
    tm70 = 0.7

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

    glPushMatrix()

    # Draw Ring of Dots with different hues
    tmr = 0.15
    if w2h <= 1.0:
        glScalef(acbic*w2h, acbic*w2h, 1)
    else:
        glScalef(acbic, acbic, 1)

    if (not __ringPoints):
        for i in range(12):
            tmf = i/12.0
            ang = 360*tmf + 90
            tmx = cos(radians(ang))*0.67#*acbic
            tmy = sin(radians(ang))*0.67#*acbic
            __ringPoints.append((tmx, tmy))

    if (not __ringVerts):
        for i in range(12):
            tmx = __ringPoints[i][0]
            tmy = __ringPoints[i][1]
            if w2h <= 1.0:
                tmx *= w2h
                tmy *= w2h
            for k in range(45):
                __ringVerts.append((tmx, tmy))
                __ringVerts.append((tmx+cos(radians(k*8))*tmr, tmy+sin(radians(k*8))*tmr))
                __ringVerts.append((tmx+cos(radians((k+1)*8))*tmr, tmy+sin(radians((k+1)*8))*tmr))
                #__ringVerts.append((tmx, tmy))

    #__ringColrs = []
    for i in range(12):
        tmx = __ringPoints[i][0]
        tmy = __ringPoints[i][1]
        tmf = i/12.0
        tmc = colorsys.hsv_to_rgb(tmf, acic, 1)
        if (len(__ringColrs) != len(__ringVerts)):
            for k in range(45):
                __ringColrs.append((tmc[0], tmc[1], tmc[2]))
                __ringColrs.append((tmc[0], tmc[1], tmc[2]))
                __ringColrs.append((tmc[0], tmc[1], tmc[2]))
                #__ringColrs.append((tmc[0], tmc[1], tmc[2]))
        else:
            for k in range(45):
                __ringColrs[i*135  +k  ] = (tmc[0], tmc[1], tmc[2])
                __ringColrs[i*135  +k*2] = (tmc[0], tmc[1], tmc[2])
                __ringColrs[i*135  +k*3] = (tmc[0], tmc[1], tmc[2])
                __ringColrs[i*135+1+k  ] = (tmc[0], tmc[1], tmc[2])
                __ringColrs[i*135+1+k*2] = (tmc[0], tmc[1], tmc[2])
                __ringColrs[i*135+1+k*3] = (tmc[0], tmc[1], tmc[2])
                __ringColrs[i*135+2+k  ] = (tmc[0], tmc[1], tmc[2])
                __ringColrs[i*135+2+k*2] = (tmc[0], tmc[1], tmc[2])
                __ringColrs[i*135+2+k*3] = (tmc[0], tmc[1], tmc[2])

        if (currentHue == tmf):
            if (wereColorsTouched == True):
                glColor3f(1,1,1)
            else:
                glColor3f(0.5,0.5,0.5)
            glBegin(GL_LINE_STRIP)
            for k in range(46):
                ttmx = tmx+cos(radians(k*8))*tmr*1.1
                ttmy = tmy+sin(radians(k*8))*tmr*1.1
                glVertex2f(ttmx, ttmy)
            glEnd()
        if (watchPoint(
            mapRanges(tmx, -1.0*w2h, 1.0*w2h, 0, wx*2),
            mapRanges(tmy, 1.0, -1.0, 0, wy*2),
            min(wx, wy)*tmr+0.05)):
            wereColorsTouched = True
            currentHue = tmf
            tmc = colorsys.hsv_to_rgb(tmf, currentSat, 1.0-currentBri)
            targetLamp.setBulbRGB(targetBulb, tmc)
            tmh = targetLamp.getBulbHSV(targetBulb)
            __pickerColrs = []
            for i in range(6):
                for j in range(6-i):
                    tmc = colorsys.hsv_to_rgb(currentHue, (i+1)/6.0 - 1/6.0, 1.0-(j)/5.0)
                    for k in range(31):
                        __pickerColrs.append(tmc)
                        __pickerColrs.append(tmc)
                        __pickerColrs.append(tmc)
                        __pickerColrs.append(tmc)
    
    ptc = np.array(__ringColrs, 'f').reshape(-1, 3)
    pnt = np.array(__ringVerts, 'f').reshape(-1, 2)
    indices = np.array(np.arange(len(__ringVerts)), 'I')
    glColorPointerf(ptc)
    glVertexPointerf(pnt)
    glDrawElementsui(GL_TRIANGLES, indices)

    # Draw Triangle of Dots with different brightness/saturation
    glPushMatrix()
    tmr = 0.05

    if (not __pickerVerts):
        #print("Caching Color Picker Vertices")
        for i in range(6):
            for j in range(6-i):
                tmx = -0.23+(i*0.13)
                tmy = +0.37-(i*0.075+j*0.15)
                for k in range(31):
                    __pickerVerts.append((tmx, tmy))
                    __pickerVerts.append((tmx+cos(radians(k*12))*tmr,tmy+sin(radians(k*12))*tmr))
                    __pickerVerts.append((tmx+cos(radians((k+1)*12))*tmr,tmy+sin(radians((k+1)*12))*tmr))
                    #__pickerVerts.append((tmx, tmy))

    tmh = targetLamp.getBulbHSV(targetBulb)
    #if (not __pickerColrs):
    if True:
        __pickerColrs = []
        #print("Caching Color Picker Vert Colors")
        #print(currentHue, tmh[0])
        for i in range(6):
            for j in range(6-i):
                tmc = colorsys.hsv_to_rgb(currentHue, (i+1)/6.0 - 1/6.0, 1.0-(j)/5.0)
                for k in range(31):
                    __pickerColrs.append(tmc)
                    __pickerColrs.append(tmc)
                    __pickerColrs.append(tmc)

    for i in range(6):
        for j in range(6-i):
            tmx = -0.23+(i*0.13)
            tmy = +0.37-(i*0.075+j*0.15)
            tmc = colorsys.hsv_to_rgb(currentHue, (i+1)/6.0 - 1/6.0, 1.0-(j)/5.0)

            if (currentBri == (j)/5.0) and (currentSat == ((i+1)/6.0 - 1/6.0)):
                if (wereColorsTouched == True):
                    glColor3f(1.0, 1.0, 1.0)
                else:
                    glColor3f(0.5, 0.5, 0.5)

                glBegin(GL_LINE_STRIP)
                for k in range(31):
                    ttmx = tmx+cos(radians(k*12))*tmr*1.1
                    ttmy = tmy+sin(radians(k*12))*tmr*1.1
                    glVertex2f(ttmx, ttmy)
                glEnd()
            if (watchPoint(
                mapRanges(tmx, -1.0*w2h, 1.0*w2h, 0, wx*2),
                mapRanges(tmy, 1.0, -1.0, 0, wy*2),
                min(wx, wy)*0.5*(tmr+0.1))):
                wereColorsTouched = True
                targetLamp.setBulbRGB(targetBulb, tmc)
                currentBri = (j)/5.0
                currentSat = (i+1)/6.0 - 1/6.0
                #print("currentBri: {:.3f}, (j)/5.0: {:.3f}, currentSat: {:.3f}, i/6.0: {:.3f}".format(currentBri, (j)/5.0, currentSat, i/6.0))

    ptc = np.array(__pickerColrs, 'f').reshape(-1, 3)
    pnt = np.array(__pickerVerts, 'f').reshape(-1, 2)
    indices = np.array(np.arange(len(__pickerVerts)), 'I')
    glColorPointerf(ptc)
    glVertexPointerf(pnt)
    glDrawElementsui(GL_TRIANGLES, indices)

    glPopMatrix()
    glPopMatrix()
    #isPro = isProfile()
    
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

    glDisable(GL_LIGHTING)
    drawBackground(0)

    if (targetScreen == 0):
        if (colrSettingCursor > 0):
            colrSettingCursor = constrain(colrSettingCursor-3/fps, 0, 1)
            #cProfile.run('drawSettingColor(colrSettingCursor, lamps[0], targetBulb, w2h)')
            drawSettingColor(colrSettingCursor, lamps[0], targetBulb, w2h)
        if (targetScreen == 0) and (colrSettingCursor == 0):
            drawHome()

    elif (targetScreen == 1):
        if (colrSettingCursor < 1):
            colrSettingCursor = constrain(colrSettingCursor+3/fps, 0, 1)
        #cProfile.run('drawSettingColor(colrSettingCursor, lamps[0], targetBulb, w2h)')
        drawSettingColor(colrSettingCursor, lamps[0], targetBulb, w2h)

    for i in range(len(lamps)):
        lamps[i].updateBulbs(1.0/fps)

    glFlush()
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
    global targetScreen, wereColorsTouched
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
    # Enfore Minimum WindowSize
    width = glutGet(GLUT_WINDOW_WIDTH)
    height = glutGet(GLUT_WINDOW_HEIGHT)

    if vis == GLUT_VISIBLE:
        glutIdleFunc(idle)
    else:
        glutIdleFunc(None)

# Equivalent to "main()" in C/C++
if __name__ == '__main__':
    #global windowDimW, windowDimH, windowPosX, windowPosY
    print("Initializing...")
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE)

    glutInitWindowPosition(windowPosX, windowPosY)
    glutInitWindowSize(windowDimW, windowDimH)

    glutCreateWindow("HeavenLi")
    glutMouseFunc(mouseInteraction)
    glutPassiveMotionFunc(mousePassive)
    glEnable(GL_LINE_SMOOTH)
    print(glutGetWindow())

    init()

    glutDisplayFunc(display)
    #glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_COLOR)
    #glEnable( GL_BLEND )
    glutReshapeFunc(reshape)
    #glutKeyboardFunc(key)
    glutSpecialFunc(special)
    glutKeyboardFunc(key)
    glutVisibilityFunc(visible)

    if "-info" in sys.argv:
        print("GL_RENDERER   = ", glGetString(GL_RENDERER))
        print("GL_VERSION    = ", glGetString(GL_VERSION))
        print("GL_VENDOR     = ", glGetString(GL_VENDOR))
        print("GL_EXTENSIONS = ", glGetString(GL_EXTENSIONS))

    glutMainLoop()
