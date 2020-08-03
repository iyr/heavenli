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

print("Loading System Utilities...")
import sys, time, traceback, datetime, os, serial, random
from math import sin,cos,sqrt,pi,radians,hypot,atan,degrees
from cobs import cobs
from platform import machine
from pynput.mouse import Controller
import numpy as np
from PIL import Image
import pytweening
print("Done!")

print("Loading heavenLi Utilities...")
import colorsys
from rangeUtils import *
import plugins.pluginLoader
from textUtils import *
