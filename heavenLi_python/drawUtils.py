import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sin,cos,sqrt,radians,hypot

def drawCornerMarkers():
    # YELLOW
    glBegin(GL_TRIANGLE_FAN)
    glColor(1, 1, 0)
    glVertex3f(-1.0, -1.0, 1)
    for i in range(91):
        glVertex3f(-1.0 + 0.5*cos(radians(i*4)), -1.0 + 0.5*sin(radians(i*4)), 1)
    glEnd()
    
    # RED
    glBegin(GL_TRIANGLE_FAN)
    glColor(1, 0, 0)
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

