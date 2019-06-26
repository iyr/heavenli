from lampClass import *

class Plugin:

    def __init__(self):
        print("Hello from demo2 class")

    def update(self):
        pass

    def getLamps(self):
        quack = []
        quack.append(Lamp())
        #quack[0].setAlias('quackDeux')
        #quack[0].setNumBulbs(3)
        #quack[0].setArn(0)
        #quack[0].setAngle(0)

        return quack

#if __name__ == '__main__':
