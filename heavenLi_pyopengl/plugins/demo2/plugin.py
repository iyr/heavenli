from lampClass import *

class Plugin:

    def __init__(self):
        print("Hello from demo2 class")

    def update(self):
        pass

    def getLamps(self):
        quack = []
        quack.append(Lamp())
        return quack

#if __name__ == '__main__':
