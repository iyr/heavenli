import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sin,cos,sqrt,radians,hypot
from datetime import datetime
import numpy as np

__bulbVerts = np.array([], 'f')
__bulbColrs = np.array([], 'f')
__bulbIndcs = np.array([], 'f')
__prvBlbClr = []
__curBlbClr = []
__lineVerts = np.array([], 'f')
__lineColrs = np.array([], 'f')
__lineIndcs = np.array([], 'f')

def drawBulbButton(
        gx=0.0,
        gy=0.0,
        scale=1.0,
        faceColor=(0.3, 0.3, 0.3),
        lineColor=(0.9, 0.9, 0.9),
        bulbColor=(0.5, 0.5, 0.5),
        w2h=1.0):

    global __bulbVerts, __bulbColrs, __bulbIndcs, __prvBlbClr, __curBlbClr, __lineVerts, __lineColrs, __lineIndcs

    __curBlbClr = bulbColor
    glPushMatrix()
    glTranslatef(gx, gy, 0)
    if w2h <= 1.0:
        glScalef(w2h*scale, w2h*scale, 1)
        glLineWidth(w2h*3.0)
        squash = w2h
    else:
        glScalef(scale, scale, 1)
        glLineWidth(pow(1/w2h, 0.5)*3.0)
        squash = 1.0

    if (__bulbVerts.size == 0):
        tmp = []
        # Define Verts for button face
        for i in range(31):
            tmp.append((0.0, 0.0))
            tmp.append((
                0.4*cos(radians(i*12)),
                0.4*sin(radians(i*12))))
            tmp.append((
                0.4*cos(radians((i+1)*12)),
                0.4*sin(radians((i+1)*12))))

        # Define Verts for Bulb Icon
        for i in range(31):
            tmp.append((0.0, 0.05))
            tmp.append((
                0.2*cos(radians(i*12)),
                0.2*sin(radians(i*12))+0.1))
            tmp.append((
                0.2*cos(radians((i+1)*12)),
                0.2*sin(radians((i+1)*12))+0.1))

        # Define verts for bulb screw base
        tmp.append((-0.085, -0.085))
        tmp.append((+0.085, -0.085))
        tmp.append((+0.085, -0.119))
        tmp.append((-0.085, -0.085))
        tmp.append((+0.085, -0.119))
        tmp.append((-0.085, -0.119))

        tmp.append((+0.085, -0.119))
        tmp.append((-0.085, -0.119))
        tmp.append((-0.085, -0.153))

        tmp.append((+0.085, -0.136))
        tmp.append((-0.085, -0.170))
        tmp.append((-0.085, -0.204))
        tmp.append((+0.085, -0.136))
        tmp.append((+0.085, -0.170))
        tmp.append((-0.085, -0.204))

        tmp.append((+0.085, -0.187))
        tmp.append((-0.085, -0.221))
        tmp.append((-0.085, -0.255))
        tmp.append((+0.085, -0.187))
        tmp.append((+0.085, -0.221))
        tmp.append((-0.085, -0.255))

        tmp.append((+0.085, -0.238))
        tmp.append((-0.085, -0.272))
        tmp.append((-0.051, -0.306))
        tmp.append((+0.085, -0.238))
        tmp.append((+0.051, -0.306))
        tmp.append((-0.051, -0.306))

        __bulbVerts = np.array(tmp, 'f')
        __bulbIndcs = np.arange(len(__bulbVerts))

    if (__bulbColrs.size == 0) or (__curBlbClr != __prvClkColr):
        __prvBlbClr = __curBlbClr
        tmc = []
        for i in range(31):
            tmc.append(faceColor)
            tmc.append(faceColor)
            tmc.append(faceColor)

        for i in range(31):
            tmc.append(bulbColor)
            tmc.append(bulbColor)
            tmc.append(bulbColor)

        for i in range(27):
            tmc.append(lineColor)

        __bulbColrs = np.array(tmc, 'f')

    glColorPointerf( __bulbColrs )
    glVertexPointerf( __bulbVerts )
    glDrawElementsui(GL_TRIANGLES, __bulbIndcs)

    # Define Verts for Bulb Button Outline
    if (__lineVerts.size == 0):
        tmp = []
        for i in range(31):
            tmp.append((
                scale*cos(radians(i*12)),
                scale*sin(radians(i*12))
                ))
        __lineVerts = np.array(tmp, 'f')
        __lineIndcs = np.arange(len(__lineVerts))


    # Define Outline Color for Bulb Button
    if (__lineColrs.size == 0):
        tmc = []
        tmc.append(bulbColor)
        for i in range(10):
            tmc.append(bulbColor)
            tmc.append(bulbColor)
            tmc.append(bulbColor)
        __lineColrs = np.array(tmc, 'f')
    else:
        __lineColrs[0] = bulbColor
        for i in range(10):
            __lineColrs[i*3+1] = bulbColor
            __lineColrs[i*3+2] = bulbColor
            __lineColrs[i*3+3] = bulbColor

    glColorPointerf( __lineColrs)
    glVertexPointerf( __lineVerts)
    glDrawElementsui(GL_LINE_STRIP, __lineIndcs)

    glPopMatrix()

__clockVerts = []
__clockColrs = []
__prvClkColr = []

def drawClock(
        scale=1.0,      # Scale of the central clock button
        faceColor=(0.3, 0.3, 0.3),  # Color of the clock face
        handColor=(0.9, 0.9, 0.9),   # Color or the clock hands
        w2h=1.0):

    global __clockVerts, __clockColrs, __prvClkColr

    glColor3f(
            faceColor[0],
            faceColor[1],
            faceColor[2])

    glPushMatrix()
    if w2h <= 1.0:
        glScalef(w2h*scale, w2h*scale, 1)
        glLineWidth(w2h*3.0)
        squash = w2h
    else:
        glScalef(scale, scale, 1)
        glLineWidth(pow(1/w2h, 0.5)*3.0)
        squash = 1.0

    if (not __prvClkColr) or (faceColor != __prvClkColr):
        __prvClkColr = faceColor
        __clockColrs = np.array([faceColor for i in range(74)], 'f').reshape(-1, 3)

    if (not __clockVerts):
        __clockVerts.append((0, 0))
        for i in range(73):
            __clockVerts.append((0.5*cos(radians(i*5)), 0.5*sin(radians(i*5))))

    pnts = np.array(__clockVerts, 'f').reshape(-1, 2)
    indices = np.arange(len(__clockVerts))
    glColorPointerf( __clockColrs )
    glVertexPointerf( pnts )
    glDrawElementsui(GL_TRIANGLE_FAN, indices)

    glLoadIdentity()
    curHour = -(datetime.now().time().hour-12)*30+90
    curMint = -datetime.now().time().minute*6+90
    curScnd = -datetime.now().time().second/10
    glColor3f(handColor[0], handColor[1], handColor[2])
    glBegin(GL_LINES)
    glVertex2f( squash*0.333*cos(radians(curScnd+curMint)), squash*0.333*sin(radians(curScnd+curMint)) )
    glVertex2f(0.0, 0.0)

    glVertex2f(squash*0.25*cos(radians(curHour)), squash*0.25*sin(radians(curHour)))
    glVertex2f(0.0, 0.0)
    glEnd()

    glPopMatrix()

