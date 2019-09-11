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
from animUtils import *
from drawArn import *
from drawButtons import *
#from lampClass import *
from rangeUtils import *
from shaderUtils import *
#from plugins.pluginLoader import *
import plugins.pluginLoader
