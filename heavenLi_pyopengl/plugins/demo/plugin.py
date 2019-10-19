from lampClass import *

class Plugin:

    def __init__(self):
        print("Hello from demo class")
        self.quack = []
        self.quack.append(Lamp())
        self.quack[0].setAlias('quack')
        self.quack[0].setNumBulbs(3)
        self.quack[0].setArn(0)
        self.quack[0].setAngle(0)
        self.quack[0].setID([ord('q'), ord('1')])
        pass
        return

    def update(self):
        pass

    def getLamps(self):
        return self.quack

#if __name__ == '__main__':
