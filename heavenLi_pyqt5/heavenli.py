from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import QGLWidget

import OpenGL
from OpenGL.GL import *

import sys, time
from math import sin, cos, pi, radians

from drawArn import *
from lampClass import *

tStart = t0 = time.time()
frames = 0
screen = 0
lightOn = True
fps = 60
lamps = []

class HeavenLiWindow(QGLWidget):

    def __init__(self, *args, **kwargs):
        global lamps
        QGLWidget.__init__(self, *args, **kwargs)
        self.setAutoBufferSwap(True)
        self.w2h = 1.0
        self._timer = QBasicTimer()
        self._timer.start(1000/fps, self)
        self.setFocusPolicy(Qt.StrongFocus)

        self.demo = Lamp()
        self.demo.setAngle(45)
        self.demo.setArn(1)
        lamps.append(self.demo)

        #self.shortcut = QShortcut(QKeySequence("q"), self)
        #self.shortcut.activated.connect(self.exitapp)

    def keyPressEvent(self, event):
        key = event.key()

        print(key)
        # Close on Escape or q
        if (key == Qt.Key_Escape) or (key == 81):
            self.close()
            quit()
        if key == Qt.Key_Up:
            self.demo.setNumBulbs(self.demo.getNumBulbs()+1)
            #lamps[0].setNumBulbs(lamps[0].getNumBulbs()+1)
        if key == Qt.Key_Down:
            self.demo.setNumBulbs(self.demo.getNumBulbs()-1)
            #lamps[0].setNumBulbs(lamps[0].getNumBulbs()-1)
        if key == Qt.Key_Left:
            self.demo.setAngle(self.demo.getAngle()-5)
        if key == Qt.Key_Right:
            self.demo.setAngle(self.demo.getAngle()+5)
        if key == 65: # 'A'
            if self.demo.getArn() == 0:
                self.demo.setArn(1)
            elif self.demo.getArn() == 1:
                self.demo.setArn(0)


    def timerEvent(self, QTimerEvent):
        global t0, frames, fps
        t = time.time()
        frames += 1
        seconds = t - t0
        fps = frames/seconds

        if t-t0 >= 1.0:
            print("%.0f frames in %3.1f seconds = %6.3f FPS" % (frames,seconds,fps))
            t0 = t
            frames = 0

        self.update()

    #def exitapp(self):
        #print("quack")
        #self.close()
        #sys.exit(app.exec_())

    # default window size
    width, height = 2400, 600

    def initializeGL(self):
        glEnable(GL_POLYGON_SMOOTH)
        self.w2h = self.width/self.height
        # background color
        #glClearColor(0,0,0,0)


    def drawBackground(self,
            Light = 0 # Currently Selected Lamp, Space, or *
            ):
        if (lamps[Light].getArn() == 0):
            drawHomeCircle(self,
                    0.0, 0.0, 
                    1.0, 1.0, 
                    lamps[Light].getNumBulbs(), 
                    lamps[Light].getAngle(), 
                    0,
                    self.w2h,
                    lamps[Light].getBulbsRGB());

        elif (lamps[Light].getArn() == 1):
            drawHomeLin(self,
                    0.0, 0.0, 
                    1.0, 1.0, 
                    lamps[Light].getNumBulbs(), 
                    lamps[Light].getAngle(), 
                    0,
                    self.w2h,
                    lamps[Light].getBulbsRGB());

    def paintGL(self):
        # clear the buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        #glColor3f(1, 0, 1)
        #glBegin(GL_TRIANGLES)
        #glVertex2f(-0.5, -0.5)
        #glVertex2f( 0.5, -0.5)
        #glVertex2f( 0.0,  0.5)
        #glEnd()

        self.drawBackground(0)
        
        iconSize = 0.15
        drawHomeCircle(self,
                0.75,
                0.75,
                iconSize,
                iconSize,
                self.demo.getNumBulbs(),
                self.demo.getAngle(),
                2,
                self.w2h,
                self.demo.getBulbsRGB()
                )

        drawHomeLin(self,
                -0.75,
                0.75,
                iconSize,
                iconSize,
                self.demo.getNumBulbs(),
                self.demo.getAngle(),
                2,
                self.w2h,
                self.demo.getBulbsRGB()
                )


        glDisable(GL_LIGHTING)

        glFlush()
        #swapBuffers()

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        # update the window size
        self.width, self.height = width, height
        if height > 0:
            self.w2h = width/height
        else:
            self.w2h = 1.0
        # paint within the whole window
        glViewport(0, 0, width, height)
        # set orthographic projection (2D only)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # the window corner OpenGL coordinates are (-+1, -+1)
        glOrtho(-1.0*self.w2h, 1.0*self.w2h, 1.0, -1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()


if __name__ == '__main__':
    # import numpy for generating random data points
    import sys
    import numpy as np
    import numpy.random as rdn

    # define a Qt window with an OpenGL widget inside it
    #PyQt5
    class TestWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            # initialize the GL widget
            self.widget = HeavenLiWindow()
            # put the window at the screen position (100, 100)
            self.setGeometry(100, 100, self.widget.width, self.widget.height)
            self.setCentralWidget(self.widget)
            self.show()

    # create the Qt App and window
    app = QtWidgets.QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())
