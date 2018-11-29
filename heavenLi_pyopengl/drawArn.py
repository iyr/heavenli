import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sin,cos,sqrt,radians,hypot
import numpy as np
from rangeUtils import constrain

# Arrays for caching 
__homeLinearVerts = []
__homeLinearColrs = []
__curHomeLinearNZ = 0
__prvHomeLinearNZ = 0
__curHomeLinearCols = [(0.5, 0.5, 0.5)]
__prvHomeLinearCols = None

def drawHomeLinear(
        gx,
        gy,
        dx,
        dy,
        nz,
        ao,
        w2h,
        colors
        ):

    global __homeLinearVerts, __homeLinearColrs, __curHomeLinearNZ, __prvHomeLinearNZ, __prvHomeLinearCols, __curHomeLinearCols
    __curHomeLinearNZ = nz
    __curHomeLinearCols = colors
    glPushMatrix()
    glRotatef(90, 0, 0, 1)
    glScalef(1, w2h, 1)
    glRotatef(ao+90, 0, 0, 1)

    if (not __homeLinearVerts) or (__prvHomeLinearNZ != __curHomeLinearNZ):
        __prvHomeLinearNZ = __curHomeLinearNZ
        __homeLinearVerts = []
        for i in range(nz):
            if i == 0:
                __homeLinearVerts.append(( i*2/nz+2.0, -2.0))
                __homeLinearVerts.append(( i*2/nz-2.0,  2.0))
                __homeLinearVerts.append(( i*2/nz-2.0, -2.0))

                __homeLinearVerts.append(( i*2/nz+2.0, -2.0))
                __homeLinearVerts.append(( i*2/nz-2.0,  2.0))
                __homeLinearVerts.append(( i*2/nz+2.0,  2.0))

            elif i == nz-1:
                __homeLinearVerts.append(( i*2/nz+1.0, -2.0))
                __homeLinearVerts.append(( i*2/nz-1.0,  2.0))
                __homeLinearVerts.append(( i*2/nz-1.0, -2.0))

                __homeLinearVerts.append(( i*2/nz+1.0, -2.0))
                __homeLinearVerts.append(( i*2/nz-1.0,  2.0))
                __homeLinearVerts.append(( i*2/nz+1.0,  2.0))

            else:
                __homeLinearVerts.append(( i*2/nz+0.75, -2.0))
                __homeLinearVerts.append(( i*2/nz-1.0,   2.0))
                __homeLinearVerts.append(( i*2/nz-1.0,  -2.0))

                __homeLinearVerts.append(( i*2/nz+0.75, -2.0))
                __homeLinearVerts.append(( i*2/nz-1.0,   2.0))
                __homeLinearVerts.append(( i*2/nz+1.0,   2.0))


    if (not __homeLinearColrs) or (__prvHomeLinearCols != __curHomeLinearCols):
        __prvHomeLinearCols = __curHomeLinearCols
        __homeLinearColrs = []
        if nz > 1:
            for i in range(nz):
                __homeLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __homeLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __homeLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __homeLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __homeLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __homeLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))


    if nz > 1:
        ptc = np.array(__homeLinearColrs, 'f').reshape(-1,3)
        pnt = np.array(__homeLinearVerts, 'f').reshape(-1,2)
        indices = np.arange(len(__homeLinearVerts))
        glColorPointerf( ptc )
        glVertexPointerf( pnt )
        glDrawElementsui(GL_TRIANGLES, indices)
    else:
        drawHomeCircle(-gx, gy, dx*1.14285, dy*1.14285, nz, ao, w2h, colors)

    glPopMatrix()

__iconLinearVerts = []
__iconLinearColrs = []
__iconOLLineVerts = []
__iconOLLineColrs = []
__iconBlbMkLVerts = []
__iconBlbMkLColrs = []
__prvIconLinearNZ = 0
__curIconLinearNZ = 0
__prvIconLinearCols = [(0.5, 0.5, 0.5)]
__curIconLinearCols = None

# Draw Tiny Rounded Square Consisting of Bands of Color
def drawIconLinear(
        gx,
        gy,
        dx,
        dy,
        nz,
        ao,
        w2h,
        colors
        ):

    global __iconLinearVerts, __iconLinearColrs, __prvIconLinearNZ, __curIconLinearNZ, __prvIconLinearCols, __curIconLinearCols, __iconOLLineVerts, __iconOLLineColrs, __iconBlbMkLVerts, __iconBlbMkLColrs
    glPushMatrix()
    glRotatef(90, 0, 0, 1)
    glTranslatef(gx, gy*(w2h), 0)
    glRotatef(ao+90, 0, 0, 1)
    if (w2h) >= 1:
        glScalef(dx, dy/2, 0)
    else:
        glScalef(dx*(w2h), (w2h)*dy/2, 0)

    __curIconLinearNZ = nz
    __curIconLinearCols = colors

    # Update / Cache Icon vertices
    if (not __iconLinearVerts) or (__prvIconLinearNZ != __curIconLinearNZ):
        __iconLinearVerts = []
        for i in range(nz):
            # Special case to draw rounded corners for end slice
            if i == 0:
                # Rounded Corner
                for j in range(13):
                    __iconLinearVerts.append((-0.74, -1.4))
                    __iconLinearVerts.append((
                            -0.75 + 0.25*cos(-radians(j*7.5+90)), 
                            -1.5 + 0.5*sin(-radians(j*7.5+90))))
                    __iconLinearVerts.append((
                            -0.75 + 0.25*cos(-radians((j+1)*7.5+90)), 
                            -1.5 + 0.5*sin(-radians((j+1)*7.5+90))))

                # Rounded Corner
                for j in range(13):
                    __iconLinearVerts.append((-0.74, 1.4))
                    __iconLinearVerts.append((
                            -0.75 + 0.25*cos(radians(j*7.5+90)), 
                            1.5 + 0.5*sin(radians(j*7.5+90))))
                    __iconLinearVerts.append((
                            -0.75 + 0.25*cos(radians((j+1)*7.5+90)), 
                            1.5 + 0.5*sin(radians((j+1)*7.5+90))))
    
                __iconLinearVerts.append(( 0.50,  2.0))
                __iconLinearVerts.append(( i*2/nz-0.75,  2.0))
                __iconLinearVerts.append(( i*2/nz-0.75, -2.0))

                __iconLinearVerts.append(( i*2/nz-0.75, -2.0))
                __iconLinearVerts.append(( i*2/nz+0.75, -2.0))
                __iconLinearVerts.append(( 0.50,  2.0))

                __iconLinearVerts.append(( 0.01,  1.5))
                __iconLinearVerts.append((-1.01,  1.5))
                __iconLinearVerts.append((-1.01, -1.5))

                __iconLinearVerts.append((-1.01, -1.5))
                __iconLinearVerts.append(( 0.01, -1.5))
                __iconLinearVerts.append(( 0.01,  1.5))

            # Special case to draw rounded corners for end slice
            elif i == nz-1:
                # Rounded Corner
                for j in range(13):
                    __iconLinearVerts.append(( 0.74, -1.4))
                    __iconLinearVerts.append((
                        0.75 - 0.25*cos(-radians(j*7.5+90)), 
                        -1.5 + 0.5*sin(-radians(j*7.5+90))))
                    __iconLinearVerts.append((
                        0.75 - 0.25*cos(-radians((j+1)*7.5+90)), 
                        -1.5 + 0.5*sin(-radians((j+1)*7.5+90))))
                    #__iconLinearVerts.append(( 0.74, -1.4))
                # Rounded Corner
                for j in range(13):
                    __iconLinearVerts.append(( 0.740, 1.4))
                    __iconLinearVerts.append((
                            0.75 - 0.25*cos(radians(j*7.5+90)), 
                            1.5 + 0.5*sin(radians(j*7.5+90))))
                    __iconLinearVerts.append((
                            0.75 - 0.25*cos(radians((j+1)*7.5+90)), 
                            1.5 + 0.5*sin(radians((j+1)*7.5+90))))
                    #__iconLinearVerts.append(( 0.740, 1.4))

                __iconLinearVerts.append(( 0.75,  2.0))
                __iconLinearVerts.append(( i*2/nz-1.0, 2.0))
                __iconLinearVerts.append(( i*2/nz-1.0,-2.0))

                __iconLinearVerts.append(( 0.75,  -2.0))
                __iconLinearVerts.append(( 0.75,   2.0))
                __iconLinearVerts.append(( i*2/nz-1.0, -2.0))

                __iconLinearVerts.append(( 0.74,  1.5))
                __iconLinearVerts.append(( 1.01,  1.5))
                __iconLinearVerts.append(( 1.01, -1.5))

                __iconLinearVerts.append(( 0.74,  1.5))
                __iconLinearVerts.append(( 1.01, -1.5))
                __iconLinearVerts.append(( 0.74, -1.5))

            else:
                __iconLinearVerts.append(( 0.75,  2.0))
                __iconLinearVerts.append(( i*2/nz-1.0, 2.0))
                __iconLinearVerts.append(( i*2/nz-1.0,-2.0))

                __iconLinearVerts.append(( 0.75,  2.0))
                __iconLinearVerts.append(( i*0/nz+0.75, -2.0))
                __iconLinearVerts.append(( i*2/nz-1.0, -2.0))

    # Update / Cache Colors
    if (not __iconLinearColrs) or (__prvIconLinearCols != __curIconLinearCols):
        __prvIconLinearCols = __curIconLinearCols
        __iconLinearColrs = []
        for i in range(nz):
            # Special case to draw rounded corners for end slice
            if i == 0:
                # Rounded Corner
                for j in range(13):
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))

                # Rounded Corner
                for j in range(13):
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
    
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))

                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))

                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))

                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))

            # Special case to draw rounded corners for end slice
            elif i == nz-1:
                # Rounded Corner
                for j in range(13):
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                # Rounded Corner
                for j in range(13):
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                    __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))

                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))

                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
            else:
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))

                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))
                __iconLinearColrs.append((colors[i][0], colors[i][1], colors[i][2]))


    ptc = np.array(__iconLinearColrs, 'f').reshape(-1,3)
    pnt = np.array(__iconLinearVerts, 'f').reshape(-1,2)
    indices = np.arange(len(__iconLinearVerts))
    glColorPointerf( ptc )
    glVertexPointerf( pnt )
    glDrawElementsui(GL_TRIANGLES, indices)

    # START Draw Outline
    # Scale line thickness
    if (not __iconOLLineVerts) or (not __iconOLLineColrs):
        __iconOLLineVerts = []
        __iconOLLineColrs = []
        if w2h <= 1.0:
            glLineWidth(w2h*2.0)
        else:
            glLineWidth((1/w2h)*2.0)
        for j in range(13):
            __iconOLLineColrs.append((0.95, 0.95, 0.95))
            __iconOLLineVerts.append((
                    0.75 - 0.25*cos(radians(j*7.5+90)), 
                    1.50 + 0.5*sin(radians(j*7.5+90))))

        for j in range(13):
            __iconOLLineColrs.append((0.95, 0.95, 0.95))
            __iconOLLineVerts.append((
                    0.75 - 0.25*cos(+radians(j*7.5+180)), 
                    -1.5 + 0.50*sin(+radians(j*7.5+180))))

        for j in range(13):
            __iconOLLineColrs.append((0.95, 0.95, 0.95))
            __iconOLLineVerts.append((
                    -0.75 + 0.25*cos(-radians(j*7.5+90)), 
                    -1.5 + 0.5*sin(-radians(j*7.5+90))))

        for j in range(13):
            __iconOLLineColrs.append((0.95, 0.95, 0.95))
            __iconOLLineVerts.append((
                    -0.75 + 0.25*cos(-radians(j*7.5+180)), 
                    1.5 + 0.5*sin(-radians(j*7.5+180))))

        __iconOLLineColrs.append((0.95, 0.95, 0.95))
        __iconOLLineVerts.append((
            0.75 - 0.25*cos(radians(90)),
            1.50 + 0.50*sin(radians(90))))

    ptc = np.array(__iconOLLineColrs, 'f').reshape(-1,3)
    pnt = np.array(__iconOLLineVerts, 'f').reshape(-1,2)
    indices = np.arange(len(__iconOLLineVerts))
    glColorPointerf( ptc )
    glVertexPointerf( pnt )
    glDrawElementsui(GL_LINE_STRIP, indices)

    # Draw Bulb Marker
    if (not __iconBlbMkLVerts) or (not __iconBlbMkLColrs) or (__prvIconLinearNZ != __curIconLinearNZ):
        __prvIconLinearNZ = __curIconLinearNZ
        __iconBlbMkLVerts = []
        __iconBlbMkLColrs = []
        if nz > 1:
            yCoord = -2.05
        else:
            yCoord = 2.05
        for i in range(nz):
            xCoord = 1/(nz*2)-((nz*2-1)/(nz*2)) + (2*i)/nz
            for j in range(13):
                __iconBlbMkLColrs.append((0.9, 0.9, 0.9))
                __iconBlbMkLVerts.append((xCoord,  yCoord))
                __iconBlbMkLColrs.append((0.9, 0.9, 0.9))
                __iconBlbMkLVerts.append((xCoord + 0.16*cos(radians(j*30)), yCoord + 0.32*sin(radians(j*30))))
                __iconBlbMkLColrs.append((0.9, 0.9, 0.9))
                __iconBlbMkLVerts.append((xCoord + 0.16*cos(radians((j+1)*30)), yCoord + 0.32*sin(radians((j+1)*30))))

    ptc = np.array(__iconBlbMkLColrs, 'f').reshape(-1,3)
    pnt = np.array(__iconBlbMkLVerts, 'f').reshape(-1,2)
    indices = np.arange(len(__iconBlbMkLVerts))
    glColorPointerf( ptc )
    glVertexPointerf( pnt )
    glDrawElementsui(GL_TRIANGLES, indices)
    # END Draw Outline

    glPopMatrix()

__homeCircleVerts = []
__homeCircleColrs = []
__curHomeCircleNZ = 0
__prvHomeCircleNZ = 0
__curHomeCircleAO = 0
__prvHomeCircleAO = 0
__curHomeCircleCols = [(0.5, 0.5, 0.5)]
__prvHomeCircleCols = None

def drawHomeCircle(
        gx,
        gy,
        dx,
        dy,
        nz,
        ao,
        w2h,
        colors
        ):
    global __homeCircleVerts, __homeCircleColrs, __curHomeCircleNZ, __prvHomeCircleNZ, __curHomeCircleCols, __prvHomeCircleCols, __curHomeCircleAO, __prvHomeCircleAO
    wx = glutGet(GLUT_WINDOW_WIDTH)
    wy = glutGet(GLUT_WINDOW_HEIGHT)
    angOffset = 360/float(nz)
    glPushMatrix()
    glScalef(sqrt((w2h))*hypot(wx, wy), sqrt((wy/wx))*hypot(wx, wy), 1)
    __curHomeCircleNZ = nz
    __curHomeCircleAO = ao
    __curHomeCircleCols = colors

    if (not __homeCircleVerts) or (__curHomeCircleNZ != __prvHomeCircleNZ) or (__curHomeCircleAO != __prvHomeCircleAO):
        __homeCircleVerts = []
        __homeCircleColrs = []
        __prvHomeCircleAO = __curHomeCircleAO
        for j in range(nz):
            for i in range(31):
                if (nz == 3):
                    #tmx = ( cos(radians(ao*nz+90))*0.333)*((cos(radians(ao*nz*4))+1)/2)
                    #tmy = (-sin(radians(ao*nz+90))*0.333)*((cos(radians(ao*nz*4))+1)/2)
                    #tmx = ( cos(radians(ao*nz+90))*0.0005)*((cos(radians(ao*nz*4))*0.75+1)/2)
                    #tmy = (-sin(radians(ao*nz+90))*0.0005)*((cos(radians(ao*nz*4))*0.75+1)/2)
                    #__homeCircleVerts.append(tmx)
                    #__homeCircleVerts.append(tmy)
                    __homeCircleVerts.append(0)
                    __homeCircleVerts.append(0)
                else:
                    __homeCircleVerts.append(0)
                    __homeCircleVerts.append(0)
    
                tma = radians(i*12.0/nz+ao+j*(angOffset)-90)
                tmx = cos(tma)#*squashW
                tmy = sin(tma)#*squashH
                __homeCircleVerts.append(tmx)
                __homeCircleVerts.append(tmy)

    if (not __homeCircleColrs) or (__curHomeCircleNZ != __prvHomeCircleNZ) or (__prvHomeCircleAO != __curHomeCircleCols):
        __homeCircleColrs = []
        __prvHomeCircleCols = __curHomeCircleCols
        __prvHomeCircleNZ = __curHomeCircleNZ
        for j in range(nz):
            for i in range(31):
                __homeCircleColrs.append((colors[j][0], colors[j][1], colors[j][2]))
                __homeCircleColrs.append((colors[j][0], colors[j][1], colors[j][2]))

    pntcols = np.array(__homeCircleColrs, 'f').reshape(-1,3)
    points = np.array(__homeCircleVerts, 'f').reshape(-1,2)
    indices = np.arange(len(__homeCircleVerts)/2)
    glVertexPointerf( points )
    glColorPointerf( pntcols )
    glDrawElementsui(GL_TRIANGLE_STRIP, indices)

    glPopMatrix()

__iconCircleVerts = []
__iconCircleColrs = []
__iconOLCircVerts = []
__iconOLCircColrs = []
__iconBlbMkCVerts = []
__iconBlbMkCColrs = []
__curIconCircleNZ = 0
__prvIconCircleNZ = 0
__curIconCircleAO = 0
__prvIconCircleAO = 0
__curIconCircleCols = None
__prvIconCircleCols = None

def drawIconCircle(
        gx,
        gy,
        dx,
        dy,
        nz,
        ao,
        w2h,
        colors
        ):
    #wx = glutGet(GLUT_WINDOW_WIDTH)
    #wy = glutGet(GLUT_WINDOW_HEIGHT)
    global __iconCircleVerts, __iconCircleColrs, __iconOLCircVerts, __iconOLCircColrs, __iconBlbMkCVerts, __iconBlbMkCColrs, __prvIconCircleNZ, __curIconCircleNZ, __prvIconCircleCols, __curIconCircleCols, __prvIconCircleAO, __curIconCircleAO
    angOffset = 360/float(nz)
    glPushMatrix()
    glTranslatef(gx*(w2h), gy, 0)

    if (w2h) >= 1:
        glScalef(dx, dy, 0)
    else:
        glScalef(dx*(w2h), dy*(w2h), 0)
    
    __curIconCircleNZ = nz
    __curIconCircleCols = colors
    __curIconCircleAO = ao

    if (not __iconCircleVerts) or (__prvIconCircleNZ != __curIconCircleNZ) or (__prvIconCircleAO != __curIconCircleAO):
        __iconCircleVerts = []
        for j in range(nz):
            #for i in range(int(angOffset)+1):
            for i in range(31):
                __iconCircleVerts.append(0)
                __iconCircleVerts.append(0)
    
                tma = radians(i*12/nz+ao+j*(angOffset)-90)
                tmx = cos(tma)
                tmy = sin(tma)
                __iconCircleVerts.append(tmx)
                __iconCircleVerts.append(tmy)

    if (not __iconCircleColrs) or (__curIconCircleNZ != __prvIconCircleNZ) or (__prvIconCircleCols != __curIconCircleCols):
        __iconCircleColrs = []
        #__prvIconCircleNZ = __curIconCircleNZ
        __prvIconCircleCols = __curIconCircleCols
        for j in range(nz):
            __iconCircleColrs.append((colors[j][0], colors[j][1], colors[j][2]))
            for i in range(31):
                __iconCircleColrs.append((colors[j][0], colors[j][1], colors[j][2]))
                __iconCircleColrs.append((colors[j][0], colors[j][1], colors[j][2]))

    pntcols = np.array(__iconCircleColrs, 'f').reshape(-1,3)
    points = np.array(__iconCircleVerts, 'f').reshape(-1,2)
    indices = np.arange(len(__iconCircleVerts)/2)
    glVertexPointerf( points )
    glColorPointerf( pntcols )
    glDrawElementsui(GL_TRIANGLE_STRIP, indices)

    # Draw Outline
    if w2h <= 1.0:
        glLineWidth(w2h*2.0)
    else:
        glLineWidth((1/w2h)*2.0)

    if (not __iconOLCircVerts) or (not __iconOLCircColrs):
        for j in range(31):
            tmx = cos(radians(j*12))
            tmy = sin(radians(j*12))
            __iconOLCircColrs.append((0.95, 0.95, 0.95))
            __iconOLCircVerts.append((tmx, tmy))

    pntcol = np.array(__iconOLCircColrs, 'f').reshape(-1, 3)
    points = np.array(__iconOLCircVerts, 'f').reshape(-1, 2)
    indices = np.arange(len(__iconOLCircVerts))
    glColorPointerf( pntcol )
    glVertexPointerf( points )
    glDrawElementsub(GL_LINE_STRIP, indices)

    #glRotatef(ao, 0, 0, 1)
    # Draw Bulb Markers
    if (not __iconBlbMkCVerts) or (__prvIconLinearNZ != __curIconCircleNZ) or (__prvIconCircleAO != __curIconCircleAO):
        __prvIconCircleNZ = __curIconCircleNZ
        __prvIconCircleAO = __curIconCircleAO
        __iconBlbMkCColrs = []
        __iconBlbMkCVerts = []
        for i in range(nz):
            xCoord = cos(radians(-90+ao - i*(angOffset) + 180/nz))
            yCoord = sin(radians(-90+ao - i*(angOffset) + 180/nz))
            for j in range(13):
                __iconBlbMkCVerts.append((xCoord, yCoord))

                __iconBlbMkCVerts.append((
                    xCoord + 0.16*cos(radians(j*30)), 
                    yCoord + 0.16*sin(radians(j*30))))

                __iconBlbMkCVerts.append((
                    xCoord + 0.16*cos(radians((j+1)*30)), 
                    yCoord + 0.16*sin(radians((j+1)*30))))

    if (not __iconBlbMkCColrs):
        __iconBlbMkCColrs = []
        for i in range(nz):
            xCoord = cos(radians(-90+ao - i*(angOffset) + 180/nz))
            yCoord = sin(radians(-90+ao - i*(angOffset) + 180/nz))
            for j in range(13):
                __iconBlbMkCColrs.append((0.9, 0.9, 0.9))
                __iconBlbMkCColrs.append((0.9, 0.9, 0.9))
                __iconBlbMkCColrs.append((0.9, 0.9, 0.9))

    pntcol = np.array(__iconBlbMkCColrs, 'f').reshape(-1, 3)
    points = np.array(__iconBlbMkCVerts, 'f').reshape(-1, 2)
    indices = np.arange(len(__iconBlbMkCVerts))
    glColorPointerf( pntcol )
    glVertexPointerf( points )
    glDrawElementsui(GL_TRIANGLES, indices)

    glPopMatrix()
