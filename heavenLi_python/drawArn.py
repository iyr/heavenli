import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sin,cos,sqrt,radians,hypot

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
        glBegin(GL_QUADS)
        for i in range(nz):
            glColor3f(colors[i][0], colors[i][1], colors[i][2])
            if i == 0:
                if drawMode > 0:
                    glEnd()
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

                    glBegin(GL_QUADS)
                    glVertex2f( 0.50,  2.0)
                    glVertex2f( i*2/nz-0.75,  2.0)
                    glVertex2f( i*2/nz-0.75, -2.0)
                    glVertex2f( 0.50, -2.0)

                    glVertex2f( 0.01,  1.5)
                    glVertex2f(-1.01,  1.5)
                    glVertex2f(-1.01, -1.5)
                    glVertex2f( 0.01, -1.5)
                else:
                    glVertex2f( 2.0,  2.0)
                    glVertex2f( i*2/nz-2.0,  2.0)
                    glVertex2f( i*2/nz-2.0, -2.0)
                    glVertex2f( 2.0, -2.0)

            elif i == nz-1:
                if drawMode > 0:
                    glEnd()
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

                    glBegin(GL_QUADS)
                    glVertex2f( 0.75,  2.0)
                    glVertex2f( i*2/nz-1.0, 2.0)
                    glVertex2f( i*2/nz-1.0,-2.0)
                    glVertex2f( 0.75, -2.0)

                    glVertex2f( 0.74,  1.5)
                    glVertex2f( 1.01,  1.5)
                    glVertex2f( 1.01, -1.5)
                    glVertex2f( 0.74, -1.5)
                else:
                    glVertex2f( 2.0,  2.0)
                    glVertex2f( i*2/nz-1.0,  2.0)
                    glVertex2f( i*2/nz-1.0, -2.0)
                    glVertex2f( 2.0, -2.0)

            else:
                glVertex2f( 0.75,  2.0)
                glVertex2f( i*2/nz-1.0, 2.0)
                glVertex2f( i*2/nz-1.0,-2.0)
                glVertex2f( 0.75, -2.0)
        glEnd()

        # Draw Outline
        if drawMode > 0:

            # Scale line thickness
            if w2h <= 1.0:
                glLineWidth(w2h*3.0)
            else:
                glLineWidth((1/w2h)*3.0)

            # Draw Outline Straights
            glBegin(GL_LINES)
            glColor3f(0.95, 0.95, 0.95)
            glVertex2f( 1.00,  1.5)
            glVertex2f( 1.00, -1.5)

            glVertex2f(-1.00,  1.5)
            glVertex2f(-1.00, -1.5)

            glVertex2f( 0.75, -2.0)
            glVertex2f(-0.75, -2.0)

            glVertex2f( 0.75,  2.0)
            glVertex2f(-0.75,  2.0)
            glEnd()

            # Draw Outline Rounded Corners
            glBegin(GL_LINE_STRIP)
            for j in range(13):
                glVertex2f(
                        -0.75 + 0.25*cos(-radians(j*7.5+90)), 
                        -1.5 + 0.5*sin(-radians(j*7.5+90)))
            glEnd()

            glBegin(GL_LINE_STRIP)
            for j in range(13):
                glVertex2f(
                        -0.75 + 0.25*cos(radians(j*7.5+90)), 
                        1.5 + 0.5*sin(radians(j*7.5+90)))
            glEnd()

            glBegin(GL_LINE_STRIP)
            for j in range(13):
                glVertex2f(
                        0.75 - 0.25*cos(-radians(j*7.5+90)), 
                        -1.5 + 0.5*sin(-radians(j*7.5+90)))
            glEnd()

            glBegin(GL_LINE_STRIP)
            for j in range(13):
                glVertex2f(
                        0.75 - 0.25*cos(radians(j*7.5+90)), 
                        1.5 + 0.5*sin(radians(j*7.5+90)))
            glEnd()

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
    mode = 1
    angOffset = 360/nz

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

    for j in range(nz):
        glBegin(GL_TRIANGLE_FAN)
        glColor3f(colors[j][0], colors[j][1], colors[j][2])
        if (nz == 3) and (drawMode == 0):
            glVertex2f(
                    ( cos(radians(ao*nz+90))*0.333)*((cos(radians(ao*nz*4))+1)/2), 
                    (-sin(radians(ao*nz+90))*0.333)*((cos(radians(ao*nz*4))+1)/2)
                    )
        else:
            glVertex2f(0, 0)
    
        for i in range(0, int(360/nz)+1):
            glColor3f(colors[j][0], colors[j][1], colors[j][2])
            glVertex2f(
                    cos(radians(i+ao+j*(360/nz)-90))*squashW,
                    sin(radians(i+ao+j*(360/nz)-90))*squashH
                    )
        glEnd()

    # Draw Outline
    if drawMode > 0:
        if w2h <= 1.0:
            glLineWidth(w2h*3.0)
        else:
            glLineWidth((1/w2h)*3.0)

        glColor3f(0.95, 0.95, 0.95)
        glBegin(GL_LINE_STRIP)
        for j in range(31):
            glVertex2f( cos(radians(j*12)), sin(radians(j*12)))
        glEnd()

    glPopMatrix()

