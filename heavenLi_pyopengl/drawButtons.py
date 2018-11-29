import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sin,cos,sqrt,radians,hypot
from datetime import datetime
import numpy as np

__bulbVerts = []
__bulbColrs = []
__prvBlbClr = []
__curBlbClr = []
__lineVerts = []
__lineColrs = []

def drawBulbButton(
        gx=0.0,
        gy=0.0,
        scale=1.0,
        faceColor=(0.3, 0.3, 0.3),
        lineColor=(0.9, 0.9, 0.9),
        bulbColor=(0.5, 0.5, 0.5),
        w2h=1.0):

    global __bulbVerts, __bulbColrs, __prvBlbClr, __curBlbClr, __lineVerts, __lineColrs

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

    if (not __bulbVerts):
        # Define Verts for button face
        for i in range(31):
            __bulbVerts.append((0.0, 0.0))
            __bulbVerts.append((
                0.4*cos(radians(i*12)),
                0.4*sin(radians(i*12))))
            __bulbVerts.append((
                0.4*cos(radians((i+1)*12)),
                0.4*sin(radians((i+1)*12))))

        # Define Verts for Bulb Icon
        for i in range(31):
            __bulbVerts.append((0.0, 0.05))
            __bulbVerts.append((
                0.2*cos(radians(i*12)),
                0.2*sin(radians(i*12))+0.1))
            __bulbVerts.append((
                0.2*cos(radians((i+1)*12)),
                0.2*sin(radians((i+1)*12))+0.1))

        # Define verts for bulb screw base
        __bulbVerts.append((-0.085, -0.085))
        __bulbVerts.append((+0.085, -0.085))
        __bulbVerts.append((+0.085, -0.119))
        __bulbVerts.append((-0.085, -0.085))
        __bulbVerts.append((+0.085, -0.119))
        __bulbVerts.append((-0.085, -0.119))

        __bulbVerts.append((+0.085, -0.119))
        __bulbVerts.append((-0.085, -0.119))
        __bulbVerts.append((-0.085, -0.153))

        __bulbVerts.append((+0.085, -0.136))
        __bulbVerts.append((-0.085, -0.170))
        __bulbVerts.append((-0.085, -0.204))
        __bulbVerts.append((+0.085, -0.136))
        __bulbVerts.append((+0.085, -0.170))
        __bulbVerts.append((-0.085, -0.204))

        __bulbVerts.append((+0.085, -0.187))
        __bulbVerts.append((-0.085, -0.221))
        __bulbVerts.append((-0.085, -0.255))
        __bulbVerts.append((+0.085, -0.187))
        __bulbVerts.append((+0.085, -0.221))
        __bulbVerts.append((-0.085, -0.255))

        __bulbVerts.append((+0.085, -0.238))
        __bulbVerts.append((-0.085, -0.272))
        __bulbVerts.append((-0.051, -0.306))
        __bulbVerts.append((+0.085, -0.238))
        __bulbVerts.append((+0.051, -0.306))
        __bulbVerts.append((-0.051, -0.306))

    if (not __bulbColrs) or (__curBlbClr != __prvClkColr):
        __prvBlbClr = __curBlbClr
        __bulbColrs = []
        for i in range(31):
            __bulbColrs.append(faceColor)
            __bulbColrs.append(faceColor)
            __bulbColrs.append(faceColor)

        for i in range(31):
            __bulbColrs.append(bulbColor)
            __bulbColrs.append(bulbColor)
            __bulbColrs.append(bulbColor)

        for i in range(27):
            __bulbColrs.append(lineColor)

    ptc = np.array(__bulbColrs, 'f').reshape(-1, 3)
    pnt = np.array(__bulbVerts, 'f').reshape(-1, 2)
    indices = np.arange(len(__bulbVerts))
    glColorPointerf(ptc)
    glVertexPointerf(pnt)
    glDrawElementsui(GL_TRIANGLES, indices)

    # Define Verts for Bulb Button Outline
    if (not __lineVerts):
        for i in range(31):
            __lineVerts.append((
                scale*cos(radians(i*12)),
                scale*sin(radians(i*12))
                ))


    # Define Outline Color for Bulb Button
    if True: #(not __lineColrs):
        __lineColrs = []
        __lineColrs.append(bulbColor)
        for i in range(10):
            __lineColrs.append(bulbColor)
            __lineColrs.append(bulbColor)
            __lineColrs.append(bulbColor)

    ptc = np.array(__lineColrs, 'f').reshape(-1,3)
    pnt = np.array(__lineVerts, 'f').reshape(-1,2)
    indices = np.arange(len(__lineVerts))
    glColorPointerf( ptc )
    glVertexPointerf( pnt )
    glDrawElementsui(GL_LINE_STRIP, indices)

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

