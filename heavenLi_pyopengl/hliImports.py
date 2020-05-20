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
import sys, time, traceback, datetime, os, serial, random
from math import sin,cos,sqrt,pi,radians,hypot,atan,degrees
from cobs import cobs
from platform import machine
from pynput.mouse import Controller
#import json
#import pickle
print("Done!")

#print("Loading Geometry Utilities...")
#from shapely.geometry import Point
#from shapely.geometry.polygon import Polygon
#print("Done!")

print("Loading heavenLi Utilities...")
import colorsys
from lampClass import *
from rangeUtils import *
import plugins.pluginLoader
from hliUIutils import *
from textUtils import *
from menuClass import *
