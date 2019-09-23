print("Loading OpenGL...")
import OpenGL
OpenGL.ERROR_ON_COPY = True
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
print("Done!")

print("Loading GLUT...")
from OpenGL.GLUT import *
print("Done!")

print("Loading NumPy...")
import numpy as np
print("Done!")

print("Loading System Utilities...")
import sys, time, traceback, datetime, os, serial
from math import sin,cos,sqrt,pi,radians,hypot
from cobs import cobs
from platform import machine
print("Done!")

print("Loading heavenLi Utilities...")
import colorsys
try:
    from animUtils import *
    from drawUtils import *
    from lampClass import *
    from rangeUtils import *
    import plugins.pluginLoader
    from hliUIutils import *
    from textUtils import *
except Exception as OOF:
    print(traceback.format_exc())
    print("Error:", OOF)
