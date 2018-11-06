#! /usr/bin/env python
import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
import sys, time
from math import sin,cos,sqrt,pi,radians
from OpenGL.constants import GLfloat
from drawArn import *
from lampClass import *
vec4 = GLfloat_4

tStart = t0 = time.time()
frames = 0
angB = 00
nz = 3
cx = 0.0
cy = 0.0
wx = 0.0
wy = 0.0
w2h = 0
dBias = 0
lamps = []
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
    global t0, frames, angB, dBias, wx, wy, w2h, demo
    t = time.time()
    frames += 1
    seconds = t - t0
    fps = frames/seconds
    if t - t0 >= 1.0:
        print("%.0f frames in %3.1f seconds = %6.3f FPS" % (frames,seconds,fps))
        print(lamps[0].getBulbRGB(0))
        t0 = t
        frames = 0
    if fps > 60:
        time.sleep(fps/10000.0)

def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

def drawBackground():
    global mode
    if (mode > 0):
         drawHomeCircle(cx, cy, cx, cy, numLights, angB, 0);

def drawCornerMarkers():
    # RED
    glBegin(GL_TRIANGLE_FAN)
    glColor(1, 0, 0)
    glVertex3f(-1.0, -1.0, 1)
    for i in range(91):
        glVertex3f(-1.0 + 0.5*cos(radians(i*4)), -1.0 + 0.5*sin(radians(i*4)), 1)
    glEnd()
    
    # YELLOW
    glBegin(GL_TRIANGLE_FAN)
    glColor(1, 1, 0)
    glVertex3f( 1.0, -1.0, 1)
    for i in range(91):
        glVertex3f( 1.0 + 0.5*cos(radians(i*4)), -1.0 + 0.5*sin(radians(i*4)), 1)
    glEnd()

    # GREEN
    glBegin(GL_TRIANGLE_FAN)
    glColor(0, 1, 0)
    glVertex3f( 1.0,  1.0, 1)
    for i in range(91):
        glVertex3f( 1.0 + 0.5*cos(radians(i*4)),  1.0 + 0.5*sin(radians(i*4)), 1)
    glEnd()

    # BLUE
    glBegin(GL_TRIANGLE_FAN)
    glColor(0, 0, 1)
    glVertex3f(-1.0,  1.0, 1)
    for i in range(91):
        glVertex3f(-1.0 + 0.5*cos(radians(i*4)),  1.0 + 0.5*sin(radians(i*4)), 1)
    glEnd()

        
def mouseInteraction(button, state, mouseX, mouseY):
    return

# Main screen drawing routine
# Passed to glutDisplayFunc()
# Called with glutPostRedisplay()
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glDisable(GL_LIGHTING)
    #drawHomeCircle(0, 0, 1, 1, nz, angB, 0, w2h)
    drawHomeLin(0, 0, 
            1.0, 1.0, 
            lamps[0].getNumBulbs(), lamps[0].getAngle(), 
            0, w2h, lamps[0].getBulbsRGB())
    iconSize = 0.2
    drawHomeCircle(0.7, 0.7, 
            iconSize, iconSize, 
            lamps[0].getNumBulbs(), lamps[0].getAngle(), 
            1, w2h, lamps[0].getBulbsRGB())
    drawHomeLin(-0.7, -0.7, 
            iconSize*0.875, iconSize*0.875, 
            lamps[0].getNumBulbs(), lamps[0].getAngle(), 
            1, w2h, lamps[0].getBulbsRGB())
    glFlush()
    glutSwapBuffers()

    framerate()

def idle():
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
    if width < 200:
        glutReshapeWindow(200, height)
    if height < 200:
        glutReshapeWindow(width, 200)

    if vis == GLUT_VISIBLE:
        glutIdleFunc(idle)
    else:
        glutIdleFunc(None)

# Equivalent to "main()" in C/C++
if __name__ == '__main__':
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)

    glutInitWindowPosition(0, 0)
    glutInitWindowSize(300, 300)
    glutCreateWindow("HeavenLi")
    glutMouseFunc(mouseInteraction)

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
