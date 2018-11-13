import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sin,cos,sqrt,radians,hypot
from datetime import datetime

def drawBulbButton(
        gx=0.0,
        gy=0.0,
        scale=1.0,
        faceColor=(0.3, 0.3, 0.3),
        lineColor=(0.9, 0.9, 0.9),
        bulbColor=(0.5, 0.5, 0.5),
        w2h=1.0):

    glColor3f(
            faceColor[0],
            faceColor[1],
            faceColor[2])
    
    glPushMatrix()
    if w2h <= 1.0:
        glScalef(w2h, w2h, 1)
        glLineWidth(w2h*3.0)
        squash = w2h
    else:
        glScalef(1, 1, 1)
        glLineWidth(pow(1/w2h, 0.5)*3.0)
        squash = 1.0

    glTranslatef(gx, gy, 0)

    # Draw Button Face
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(0.0, 0.0)
    for i in range(73):
        glVertex2f(
                scale*cos(radians(i*5)),
                scale*sin(radians(i*5))
                )
    glEnd()

    # Draw Bulb with color
    glColor3f(
            bulbColor[0],
            bulbColor[1],
            bulbColor[2])

    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(0.0, 0.05)
    for i in range(73):
        glVertex2f(
                0.5*scale*cos(radians(i*5)),
                0.5*scale*sin(radians(i*5))+0.25*scale
                )
    glEnd()

    glColor3f(
            lineColor[0],
            lineColor[1],
            lineColor[2])

    glBegin(GL_LINE_STRIP)
    for i in range(73):
        glVertex2f(
                0.5*scale*cos(radians(i*5)),
                0.5*scale*sin(radians(i*5))+0.25*scale
                )
    glEnd()

    scale *= 4.25

    glBegin(GL_QUADS)
    glVertex2f(-0.05*scale, -0.05*scale)
    glVertex2f( 0.05*scale, -0.05*scale)
    glVertex2f( 0.05*scale, -0.07*scale)
    glVertex2f(-0.05*scale, -0.07*scale)

    glVertex2f( 0.05*scale, -0.05*scale)
    glVertex2f(-0.05*scale, -0.07*scale)
    glVertex2f(-0.05*scale, -0.09*scale)
    glVertex2f( 0.05*scale, -0.07*scale)

    glVertex2f( 0.05*scale, -0.08*scale)
    glVertex2f(-0.05*scale, -0.10*scale)
    glVertex2f(-0.05*scale, -0.12*scale)
    glVertex2f( 0.05*scale, -0.10*scale)

    glVertex2f( 0.05*scale, -0.11*scale)
    glVertex2f(-0.05*scale, -0.13*scale)
    glVertex2f(-0.05*scale, -0.15*scale)
    glVertex2f( 0.05*scale, -0.13*scale)

    glVertex2f( 0.05*scale, -0.14*scale)
    glVertex2f(-0.05*scale, -0.16*scale)
    glVertex2f(-0.03*scale, -0.18*scale)
    glVertex2f( 0.03*scale, -0.18*scale)
    glEnd()

    glPopMatrix()


def drawClock(
        scale=1.0,      # Scale of the central clock button
        faceColor=(0.3, 0.3, 0.3),  # Color of the clock face
        handColor=(0.9, 0.9, 0.9),   # Color or the clock hands
        w2h=1.0):

    glColor3f(
            faceColor[0],
            faceColor[1],
            faceColor[2])

    glPushMatrix()
    if w2h <= 1.0:
        glScalef(w2h, w2h, 1)
        glLineWidth(w2h*3.0)
        squash = w2h
    else:
        glScalef(1, 1, 1)
        glLineWidth(pow(1/w2h, 0.5)*3.0)
        squash = 1.0

    #if w2h <= 1.0:
    #else:

    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(0.0, 0.0)
    for i in range(73):
        glVertex2f(
                0.5*cos(radians(i*5)),
                0.5*sin(radians(i*5))
                )
    glEnd()

    glLoadIdentity()
    curHour = -(datetime.now().time().hour-12)*30+90
    curMint = -datetime.now().time().minute*6+90
    curScnd = -datetime.now().time().second/10
    glColor3f(handColor[0], handColor[1], handColor[2])
    glBegin(GL_LINES)
    glVertex2f( squash*0.333*cos(radians(curScnd+curMint)), squash*0.333*sin(radians(curScnd+curMint)) )
    glVertex2f(0.0, 0.0)

    glVertex2f( squash*0.25*cos(radians(curHour)), squash*0.25*sin(radians(curHour)) )
    glVertex2f(0.0, 0.0)
    glEnd()

    glPopMatrix()

