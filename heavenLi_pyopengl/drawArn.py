import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sin,cos,sqrt,radians,hypot
import numpy as np
from ctypes import *

def drawHomeLin(
        gx,
        gy,
        dx,
        dy,
        nz,
        ao,
        drawMode,
        w2h,
        colors
        ):

    wx = glutGet(GLUT_WINDOW_WIDTH)
    wy = glutGet(GLUT_WINDOW_HEIGHT)
    glPushMatrix()
    if drawMode == 0:
        squashW = 0.1*abs(sin(radians(2*ao)))
        squashH = squashW
    else:
        squashW = 0
        squashH = 0

    glRotatef(90, 0, 0, 1)

    if drawMode == 0:
        if wx != wy:
            glScalef(1, w2h, 1)
        else:
            glScalef(
                    dx+dx*squashW, 
                    dy+dy*squashH,
                    1)
        glRotatef(ao+90, 0, 0, 1)

    elif drawMode != 0:
        glTranslatef(gx, gy*(w2h), 0)
        glRotatef(ao+90, 0, 0, 1)
        if (w2h) >= 1:
            glScalef(dx, dy/2, 0)
        else:
            glScalef(dx*(w2h), (w2h)*dy/2, 0)

    if nz > 1:
        #glBegin(GL_QUADS)
        for i in range(nz):
            tmp = []
            tmc = []

            # Special case to draw rounded corners for end slice
            if i == 0:
                if drawMode > 0:
                    #glEnd()
                    # Rounded Corner
                    glBegin(GL_TRIANGLE_FAN)
                    glVertex2f(-0.74, -1.4)
                    for j in range(13):
                        glVertex2f(
                                -0.75 + 0.25*cos(-radians(j*7.5+90)), 
                                -1.5 + 0.5*sin(-radians(j*7.5+90)))
                    glEnd()

                    # Rounded Corner
                    glBegin(GL_TRIANGLE_FAN)
                    glVertex2f(-0.74, 1.4)
                    for j in range(13):
                        glVertex2f(
                                -0.75 + 0.25*cos(radians(j*7.5+90)), 
                                1.5 + 0.5*sin(radians(j*7.5+90)))
                    glEnd()

                    tmp.append(( 0.50,  2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( i*2/nz-0.75,  2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( i*2/nz-0.75, -2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( 0.50, -2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))

                    tmp.append(( 0.01,  1.5))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append((-1.01,  1.5))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append((-1.01, -1.5))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( 0.01, -1.5))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                else:
                    tmp.append(( 2.0,  2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( i*2/nz-2.0,  2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( i*2/nz-2.0, -2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( 2.0, -2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))

            # Special case to draw rounded corners for end slice
            elif i == nz-1:
                if drawMode > 0:
                    #glEnd()
                    # Rounded Corner
                    glBegin(GL_TRIANGLE_FAN)
                    glVertex3f( 0.74, -1.4, 0)
                    for j in range(13):
                        glVertex2f(
                                0.75 - 0.25*cos(-radians(j*7.5+90)), 
                                -1.5 + 0.5*sin(-radians(j*7.5+90)))
                    glEnd()

                    # Rounded Corner
                    glBegin(GL_TRIANGLE_FAN)
                    glVertex3f( 0.740, 1.4, 0)
                    for j in range(13):
                        glVertex2f(
                                0.75 - 0.25*cos(radians(j*7.5+90)), 
                                1.5 + 0.5*sin(radians(j*7.5+90)))
                    glEnd()

                    tmp.append(( 0.75,  2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( i*2/nz-1.0, 2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( i*2/nz-1.0,-2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( 0.75, -2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))

                    tmp.append(( 0.74,  1.5))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( 1.01,  1.5))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( 1.01, -1.5))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( 0.74, -1.5))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                else:
                    tmp.append(( 2.0,  2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( i*2/nz-1.0,  2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( i*2/nz-1.0, -2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                    tmp.append(( 2.0, -2.0))
                    tmc.append((colors[i][0], colors[i][1], colors[i][2]))

            else:
                tmp.append(( 0.75,  2.0))
                tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                tmp.append(( i*2/nz-1.0, 2.0))
                tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                tmp.append(( i*2/nz-1.0,-2.0))
                tmc.append((colors[i][0], colors[i][1], colors[i][2]))
                tmp.append(( 0.75, -2.0))
                tmc.append((colors[i][0], colors[i][1], colors[i][2]))

            ptc = np.array(tmc, 'f').reshape(-1,3)
            pnt = np.array(tmp, 'f').reshape(-1,2)
            indices = np.arange(len(tmp))
            glColorPointerf( ptc )
            glVertexPointerf( pnt )
            glDrawElementsui(GL_QUADS, indices)

        # START Draw Outline
        if drawMode > 0:
            # Scale line thickness
            if w2h <= 1.0:
                glLineWidth(w2h*2.0)
            else:
                glLineWidth((1/w2h)*2.0)

            tmp = []
            tmc = []
            for j in range(13):
                tmc.append((0.95, 0.95, 0.95))
                tmp.append((
                        0.75 - 0.25*cos(radians(j*7.5+90)), 
                        1.50 + 0.5*sin(radians(j*7.5+90))))

            for j in range(13):
                tmc.append((0.95, 0.95, 0.95))
                tmp.append((
                        0.75 - 0.25*cos(+radians(j*7.5+180)), 
                        -1.5 + 0.50*sin(+radians(j*7.5+180))))

            for j in range(13):
                tmc.append((0.95, 0.95, 0.95))
                tmp.append((
                        -0.75 + 0.25*cos(-radians(j*7.5+90)), 
                        -1.5 + 0.5*sin(-radians(j*7.5+90))))

            for j in range(13):
                tmc.append((0.95, 0.95, 0.95))
                tmp.append((
                        -0.75 + 0.25*cos(-radians(j*7.5+180)), 
                        1.5 + 0.5*sin(-radians(j*7.5+180))))

            tmc.append((0.95, 0.95, 0.95))
            tmp.append((
                0.75 - 0.25*cos(radians(90)),
                1.50 + 0.50*sin(radians(90))))

            ptc = np.array(tmc, 'f').reshape(-1,3)
            pnt = np.array(tmp, 'f').reshape(-1,2)
            indices = np.arange(len(tmp))
            glColorPointerf( ptc )
            glVertexPointerf( pnt )
            glDrawElementsui(GL_LINE_STRIP, indices)

            # Draw Bulb Marker
            if drawMode == 2:
                for i in range(nz):
                    glColor3f(0.9, 0.9, 0.9)
                    glBegin(GL_TRIANGLE_FAN)
                    xCoord = 1/(nz*2)-((nz*2-1)/(nz*2)) + (2*i)/nz
                    yCoord = 2.05
                    glVertex2f( xCoord,  yCoord)
                    for j in range(13):
                        glVertex2f(xCoord + 0.16*cos(radians(j*30)), yCoord + 0.32*sin(radians(j*30)))
                    glEnd()

        # END Draw Outline

        glPopMatrix()

    else:
        glPopMatrix()
        drawHomeCircle(-gx, gy, dx*1.14285, dy*1.14285, nz, ao, drawMode, w2h, colors)

def drawHomeCircle(
        gx,
        gy,
        dx,
        dy,
        nz,
        ao,
        drawMode,
        w2h,
        colors
        ):
    wx = glutGet(GLUT_WINDOW_WIDTH)
    wy = glutGet(GLUT_WINDOW_HEIGHT)
    angOffset = 360/float(nz)
    glPushMatrix()
    if drawMode > 0:
        glTranslatef(gx*(w2h), gy, 0)

    if drawMode > 0:
        if (w2h) >= 1:
            glScalef(dx, dy, 0)
        else:
            glScalef(dx*(w2h), dy*(w2h), 0)


    if (drawMode > 0):
        # No Distortion for Iconography
        squashW = 1.0
        squashH = 1.0
    else:
        squashW = sqrt((w2h))*hypot(wx, wy)
        squashH = sqrt((wy/wx))*hypot(wx, wy)

    tmp = []
    tmc = []
    for j in range(nz):
        tmc.append((colors[j][0], colors[j][1], colors[j][2]))
        for i in range(0, int(angOffset)+1):
            if (nz == 3) and (drawMode == 0):
                tmx = ( cos(radians(ao*nz+90))*0.333)*((cos(radians(ao*nz*4))+1)/2)
                tmy = (-sin(radians(ao*nz+90))*0.333)*((cos(radians(ao*nz*4))+1)/2)
                tmp.append(tmx)
                tmp.append(tmy)
                tmc.append((colors[j][0], colors[j][1], colors[j][2]))
            else:
                tmp.append(0)
                tmp.append(0)
                tmc.append((colors[j][0], colors[j][1], colors[j][2]))
    
            tmx = cos(radians(i+ao+j*(angOffset)-90))*squashW
            tmy = sin(radians(i+ao+j*(angOffset)-90))*squashH
            tmp.append(tmx)
            tmp.append(tmy)
            tmc.append((colors[j][0], colors[j][1], colors[j][2]))

    pntcols = np.array(tmc, 'f').reshape(-1,3)
    points = np.array(tmp, 'f').reshape(-1,2)
    indices = np.arange(len(tmp)/2)
    glVertexPointerf( points )
    glColorPointerf( pntcols )
    glDrawElementsui(GL_TRIANGLE_STRIP, indices)

    #buffers = glGenBuffers(2, (len(pntcols)+len(points))*4)
    #glBindBuffer(GL_ARRAY_BUFFER, points)
    #glBufferData(GL_ARRAY_BUFFER, len(points)*4, 4, GL_STATIC_DRAW)

    # Draw Outline
    if drawMode > 0:
        if w2h <= 1.0:
            glLineWidth(w2h*2.0)
        else:
            glLineWidth((1/w2h)*2.0)

        #glColor3f(0.95, 0.95, 0.95)
        tmp = []
        tmc = []
        for j in range(31):
            tmx = cos(radians(j*12))
            tmy = sin(radians(j*12))
            tmp.append(tmx)
            tmp.append(tmy)
            tmc.append((0.95, 0.95, 0.95))

        pntcol = np.array(tmc, 'f').reshape(-1, 3)
        points = np.array(tmp, 'f').reshape(-1, 2)
        indices = np.arange(len(tmp)/2)
        glColorPointerf( pntcol )
        glVertexPointerf( points )
        glDrawElementsub(GL_LINE_STRIP, indices)

    # Draw Bulb Marker
    if drawMode == 2:
        for i in range(nz):
            #glColor3f(0.9, 0.9, 0.9)
            #glBegin(GL_TRIANGLE_FAN)
            xCoord = cos(radians(-90+ao - i*(angOffset) + 180/nz))
            yCoord = sin(radians(-90+ao - i*(angOffset) + 180/nz))
            tmp = []
            tmc = []
            tmp.append(xCoord)
            tmp.append(yCoord)
            tmc.append((0.9, 0.9, 0.9))
            for j in range(13):
                tmp.append(xCoord + 0.16*cos(radians(j*30)))
                tmp.append(yCoord + 0.16*sin(radians(j*30)))
                tmc.append((0.9, 0.9, 0.9))

            pntcol = np.array(tmc, 'f').reshape(-1, 3)
            points = np.array(tmp, 'f').reshape(-1, 2)
            indices = np.arange(len(tmp)/2)
            glColorPointerf( pntcol )
            glVertexPointerf( points )
            glDrawElementsui(GL_TRIANGLE_FAN, indices)

    glPopMatrix()

