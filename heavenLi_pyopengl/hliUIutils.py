# Classes and utilities for UI animation

from lampClass import *
from drawArn import *
def drawIcon(ix, iy, scale, color, w2h, Lamp):
    if (Lamp.getArn() == 0):
        drawIconCircle(ix, iy, 
                scale, 
                Lamp.metaLampLevel+2,
                color,
                Lamp.getNumBulbs(), 
                Lamp.getAngle(), 
                w2h, 
                Lamp.getBulbsCurrentRGB())

    if (Lamp.getArn() == 1):
        drawIconLinear(ix, iy, 
                scale, 
                Lamp.metaLampLevel+2,
                color,
                Lamp.getNumBulbs(), 
                Lamp.getAngle(), 
                w2h, 
                Lamp.getBulbsCurrentRGB())
