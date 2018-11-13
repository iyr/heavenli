#! /usr/bin/env python
import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
import sys, time
from math import sin,cos,sqrt,pi,radians
from OpenGL.constants import GLfloat

from drawArn import *
from drawButtons import *
from drawUtils import *
from lampClass import *
from rangeUtils import *
vec4 = GLfloat_4

tStart = t0 = time.time()
frames = 0
angB = 00
nz = 3
w2h = 0
lamps = []
screen = 0
lightOn = False
fps = 60
#demo = Lamp()

def init():
    global wx, wy, dBias, w2h, lamps
    dWidth = glutGet(GLUT_WINDOW_WIDTH)
    dHeight = glutGet(GLUT_WINDOW_HEIGHT)
    wx = dWidth/4.0
    wy = dHeight/4.0
    w2h = wx/wy
    dBias = min(dWidth, dHeight)
    demo = Lamp()
    lamps.append(demo)

def framerate():
    global t0, frames, w2h, fps, derp
    t = time.time()
    frames += 1
    seconds = t - t0
    fps = frames/seconds
    if t - t0 >= 1.0:
        print(derp)
        print("%.0f frames in %3.1f seconds = %6.3f FPS" % (frames,seconds,fps))
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
                 0,
                 w2h,
                 lamps[Light].getBulbsRGB());

    elif (lamps[Light].getArn() == 1):
         drawHomeLin(0.0, 0.0, 
                 1.0, 1.0, 
                 lamps[Light].getNumBulbs(), 
                 lamps[Light].getAngle(), 
                 0,
                 w2h,
                 lamps[Light].getBulbsRGB());

        
def mouseInteraction(button, state, mouseX, mouseY):
    global lightOn

    # We are at the home screen
    if (screen == 0) and (state == 1):
        wx = glutGet(GLUT_WINDOW_WIDTH)
        wy = glutGet(GLUT_WINDOW_HEIGHT)
        dBias = min(wx, wy)/2
        #if (1.0 >= pow((mouseX-wx/2), 2) / pow(dBias/2, 2) + pow(mouseY-wy/2, 2) / pow(dBias/2, 2)):
        if watchPoint(mouseX, mouseY, wx, wy, dBias):
            lightOn = not lightOn
            for i in range(len(lamps)):
                lamps[i].setMainLight(lightOn)
    return 

def watchPoint(mouseX, mouseY, px, py, pr):
    if (1.0 >= pow((mouseX-px/2), 2) / pow(pr/2, 2) + pow((mouseY-py/2), 2) / pow(pr/2, 2)):
        return True
    else:
        return False

# Main screen drawing routine
# Passed to glutDisplayFunc()
# Called with glutPostRedisplay()
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glDisable(GL_LIGHTING)
    drawBackground(0)
    #drawClock(1.0, (0.3, 0.3, 0.3), (0.95, 0.95, 0.95), w2h)
    drawClock(w2h=w2h)

    iconSize = 0.15

    # Draw circularly arranged bulb buttons
    if lamps[0].getArn() == 0:
        tmn = lamps[0].getNumBulbs()
        for i in range(tmn):
            tmx = 0.75*cos(radians(+i*360/tmn - 90 + lamps[0].getAngle() + 180/tmn))
            tmy = 0.75*sin(radians(+i*360/tmn - 90 + lamps[0].getAngle() + 180/tmn))
            if w2h >= 1:
                tmx *= w2h
            else:
                tmy /= w2h
            drawBulbButton(gx=tmx, gy=tmy, scale=iconSize*1.05, bulbColor=lamps[0].getBulbRGB(i), w2h=w2h)
    elif lamps[0].getArn() == 1:
        tmn = lamps[0].getNumBulbs()
        for i in range(tmn):
            ang = radians(+i*180/constrain(tmn-1, 1, 5) + lamps[0].getAngle() + 180)
            if tmn == 1:
                ang -= 0.5*3.14159265
            tmx = 0.75*cos(ang)
            tmy = 0.75*sin(ang)
            if w2h >= 1:
                tmx *= w2h
            else:
                tmy /= w2h
            drawBulbButton(gx=tmx, gy=tmy, scale=iconSize*1.05, bulbColor=lamps[0].getBulbRGB(tmn-i-1), w2h=w2h)
    #drawHomeCircle(0, 0, 1, 1, nz, angB, 0, w2h)
    #drawHomeLin(0, 0, 
            #1.0, 1.0, 
            #lamps[0].getNumBulbs(), lamps[0].getAngle(), 
            #0, w2h, lamps[0].getBulbsRGB())
    #drawCornerMarkers()
    for i in range(len(lamps)):
        lamps[i].updateBulbs(1.0/fps)
    drawHomeCircle(0.75, 0.75, 
            iconSize, iconSize, 
            lamps[0].getNumBulbs(), lamps[0].getAngle(), 
            2, w2h, lamps[0].getBulbsRGB())
    drawHomeLin(-0.75, -0.75, 
            iconSize*0.875, iconSize*0.875, 
            lamps[0].getNumBulbs(), lamps[0].getAngle(), 
            2, w2h, lamps[0].getBulbsRGB())
    glFlush()
    glutSwapBuffers()

    framerate()

def idle():
    #lamps[0].updateBulbs(constrain(60.0/fps, 1, 2.4))
    glutPostRedisplay()

# change view angle
# Respond to user input from "special" keys
def special(k, x, y):
    global angB, nz

    if k == GLUT_KEY_LEFT:
        lamps[0].setAngle(lamps[0].getAngle() + 5)
    elif k == GLUT_KEY_RIGHT:
        lamps[0].setAngle(lamps[0].getAngle() - 5)
    elif k == GLUT_KEY_UP:
        lamps[0].setNumBulbs(lamps[0].getNumBulbs()+1)
    elif k == GLUT_KEY_DOWN:
        lamps[0].setNumBulbs(lamps[0].getNumBulbs()-1)
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

# new window size or exposure
# this function is called everytime the window is resized
def reshape(width, height):
    global wx, wy, dBias, w2h

    if height > 0:
        w2h = width/height
    else:
        w2h = 1
    #wx = dWidth/4.0
    #wy = dHeight/4.0
    wx = width
    wy = height
    dBias = min(width, height)
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
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE)

    glutInitWindowPosition(0, 0)
    glutInitWindowSize(300, 300)
    glutCreateWindow("HeavenLi")
    glutMouseFunc(mouseInteraction)
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
