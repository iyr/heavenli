#! /usr/bin/env python
import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
import sys, time
from math import sin,cos,sqrt,pi,radians
import os

from drawArn import *
from drawButtons import *
from drawUtils import *
from lampClass import *
from rangeUtils import *

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
    global t0, frames, w2h, fps
    t = time.time()
    frames += 1
    seconds = t - t0
    fps = frames/seconds
    if t - t0 >= 1.0:
        #print("%.0f frames in %3.1f seconds = %6.3f FPS" % (frames,seconds,fps))
        t0 = t
        frames = 0
    if fps > 60:
        time.sleep(fps/10000.0)

def drawBackground(Light = 0 # Currently Selected Lamp, Space, or *
        ):
    if (lamps[Light].getArn() == 0):
         drawHomeCircle(0.0, 0.0, 
                 1.0, 1.0, 
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

    # We are at the home screen
    if (screen == 0) and (touchState != prvState):
        #wx = glutGet(GLUT_WINDOW_WIDTH)
        #wy = glutGet(GLUT_WINDOW_HEIGHT)
        dBias = min(wx, wy)/2
        if watchPoint(wx, wy, dBias):
            lightOn = not lightOn
            for i in range(len(lamps)):
                lamps[i].setMainLight(lightOn)

    # Draw circularly arranged bulb buttons
    if lamps[0].getArn() == 0:
        tmn = lamps[0].getNumBulbs()
        for i in range(tmn):
            tmx = 0.75*cos(radians(+i*360/tmn - 90 + lamps[0].getAngle() + 180/tmn))
            tmy = 0.75*sin(radians(+i*360/tmn - 90 + lamps[0].getAngle() + 180/tmn))
            if w2h >= 1:
                tmx *= pow(w2h, 0.5)
            else:
                tmy /= pow(w2h, 0.5)
            drawBulbButton(
                    gx=tmx, 
                    gy=tmy, 
                    scale=iconSize*2.66,
                    bulbColor=lamps[0].getBulbRGB(i), 
                    w2h=w2h)
            if (screen == 0) and (touchState != prvState):
                if (watchPoint(
                    mapRanges(tmx, -1.0*w2h, 1.0*w2h, 0, wx*2), 
                    mapRanges(tmy, 1.0, -1.0, 0, wy*2),
                    min(wx, wy)*0.5*0.3)):
                    #drawSettingColor(colrSettingCursor, 0, i)
                    targetScreen = 1
                    targetBulb = i

    # Draw Linearly arranged bulb buttons
    elif lamps[0].getArn() == 1:
        tmn = lamps[0].getNumBulbs()
        for i in range(tmn):
            ang = radians(+i*180/constrain(tmn-1, 1, 5) + lamps[0].getAngle() + 180)
            if tmn == 1:
                ang -= 0.5*3.14159265
            tmx = 0.75*cos(ang)
            tmy = 0.75*sin(ang)
            if w2h >= 1:
                tmx *= pow(w2h, 0.5)
            else:
                tmy /= pow(w2h, 0.5)
            drawBulbButton(
                    gx=tmx, 
                    gy=tmy, 
                    scale=iconSize*2.66,
                    bulbColor=lamps[0].getBulbRGB(tmn-i-1), 
                    w2h=w2h)
            if (screen == 0) and (touchState != prvState):
                if (watchPoint(
                    mapRanges(tmx, -1.0*w2h, 1.0*w2h, 0, wx*2), 
                    mapRanges(tmy, 1.0, -1.0, 0, wy*2),
                    min(wx, wy)*0.5*0.3)):
                    #drawSettingColor(colrSettingCursor, 0, i)
                    targetScreen = 1
                    targetBulb = i

    drawIconCircle(0.75, 0.75, 
            iconSize, iconSize, 
            lamps[0].getNumBulbs(), lamps[0].getAngle(), 
            w2h, lamps[0].getBulbsRGB())
    drawIconLinear(-0.75, -0.75, 
            iconSize*0.875, iconSize*0.875, 
            lamps[0].getNumBulbs(), lamps[0].getAngle(), 
            w2h, lamps[0].getBulbsRGB())


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

def animCurve(c):
    return -2.25*pow(float(c)/(1.5), 2) + 1.0

def animCurveBounce(c):
    if (c >= 1.0):
        return 0
    else:
        return -3.0*pow((float(c)-(0.14167))/(1.5), 2)+1.02675926

def mousePassive(mouseX, mouseY):
    global cursorX, cursorY
    cursorX = mouseX
    cursorY = mouseY
        
def mouseInteraction(button, state, mouseX, mouseY):
    global lightOn, lamps, cursorX, cursorY, wx, wy, touchState, prvState
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
        
def drawSettingColor(cursor, targetLamp, targetBulb):
    acbic = animCurveBounce(1.0-cursor)
    acic = animCurve(1.0-cursor)
    acbc = animCurveBounce(cursor)
    acc = animCurve(cursor)
    #cmx = (width >= height ? mx : cx/4)
    global wx, wy
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
    if lamps[0].getArn() == 0:
        tmn = lamps[0].getNumBulbs()
        for i in range(tmn):
            tmx = 0.75*cos(radians(+i*360/tmn - 90 + lamps[0].getAngle() + 180/tmn))
            tmy = 0.75*sin(radians(+i*360/tmn - 90 + lamps[0].getAngle() + 180/tmn))
            if w2h >= 1:
                tmx *= pow(w2h, 0.5)
            else:
                tmy /= pow(w2h, 0.5)
            drawBulbButton(
                    gx=tmx, 
                    gy=tmy, 
                    scale=iconSize*2.66*pow(acc, 4), 
                    bulbColor=lamps[0].getBulbRGB(i), 
                    w2h=w2h)

    # Draw Linearly arranged bulb buttons
    elif lamps[0].getArn() == 1:
        tmn = lamps[0].getNumBulbs()
        for i in range(tmn):
            ang = radians(+i*180/constrain(tmn-1, 1, 5) + lamps[0].getAngle() + 180)
            if tmn == 1:
                ang -= 0.5*3.14159265
            tmx = 0.75*cos(ang)
            tmy = 0.75*sin(ang)
            if w2h >= 1:
                tmx *= pow(w2h, 0.5)
            else:
                tmy /= pow(w2h, 0.5)
            drawBulbButton(
                    gx=tmx, 
                    gy=tmy, 
                    scale=iconSize*2.66*pow(acc, 4),
                    bulbColor=lamps[0].getBulbRGB(tmn-i-1), 
                    w2h=w2h)
            
    drawClock(
            scale=acic*1.4, 
            w2h=w2h, 
            handColor=(0.9*acc, 0.9*acc, 0.9*acc), 
            faceColor=(0.3*acc, 0.3*acc, 0.3*acc))
    #isPro = isProfile()

# Main screen drawing routine
# Passed to glutDisplayFunc()
# Called with glutPostRedisplay()
def display():
    global colrSettingCursor, targetScreen, targetBulb, fps
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glDisable(GL_LIGHTING)
    drawBackground(0)
    #drawClock(1.0, (0.3, 0.3, 0.3), (0.95, 0.95, 0.95), w2h)
    drawClock(w2h=w2h)

    if (targetScreen == 0):
        if (colrSettingCursor > 0):
            colrSettingCursor = constrain(colrSettingCursor-3/fps, 0, 1)
            #drawSettingColor(colrSettingCursor, targetLamp, targetBulb)
        if (targetScreen == 0) and (colrSettingCursor == 0):
            drawHome()

    elif (targetScreen == 1):
        if (colrSettingCursor < 1):
            colrSettingCursor = constrain(colrSettingCursor+3/fps, 0, 1)
        drawSettingColor(colrSettingCursor, 0, targetBulb)

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
    global angB, nz, windowPosX, windowPosY, windowDimW, windowDimH, isFullScreen

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


    else:
        return
    glutPostRedisplay()

def key(ch, x, y):
    if ch == as_8_bit('q'):
        sys.exit(0)
    if ord(ch) == 27: # ESC
        sys.exit(0)

    if ch == as_8_bit('a'):
        if lamps[0].getArn() == 0:
            lamps[0].setArn(1)
        elif lamps[0].getArn() == 1:
            lamps[0].setArn(0)

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
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE)

    glutInitWindowPosition(windowPosX, windowPosY)
    glutInitWindowSize(windowDimW, windowDimH)

    glutCreateWindow("HeavenLi")
    glutMouseFunc(mouseInteraction)
    glutPassiveMotionFunc(mousePassive)
    glEnable(GL_LINE_SMOOTH)

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
