import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sin,cos,sqrt,radians,hypot
from datetime import datetime

def drawClock(
        scale=1.0,      # Scale of the central clock button
        faceColor=(0.3, 0.3, 0.3),  # Color of the clock face
        handColor=(0.9, 0.9, 0.9)   # Color or the clock hands
        ):

    glColor3f(
            faceColor[0],
            faceColor[1],
            faceColor[2])

    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(0.0, 0.0)
    for i in range(73):
        glVertex2f(
                0.5*cos(radians(i*5)),
                0.5*sin(radians(i*5))
                )
    glEnd()

    glLineWidth(4.0)
    glLoadIdentity()
    curHour = -(datetime.now().time().hour-12)*30+90
    curMint = -datetime.now().time().minute*6+90
    curScnd = -datetime.now().time().second/10
    glColor3f(handColor[0], handColor[1], handColor[2])
    glBegin(GL_LINES)
    glVertex2f( 0.333*cos(radians(curScnd+curMint)), 0.333*sin(radians(curScnd+curMint)) )
    glVertex2f(0.0, 0.0)

    glVertex2f( 0.25*cos(radians(curHour)), 0.25*sin(radians(curHour)) )
    glVertex2f(0.0, 0.0)
    glEnd()

