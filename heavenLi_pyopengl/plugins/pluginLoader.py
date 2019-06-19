import sys
import os

# Setup subdirectories to be loaded
PACKAGE_SUB = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_SUB)))

plugins = []

def initPlugins():
    global plugins
    print("Initializing plugins...")
    tmd = [x[0] for x in os.walk('./plugins')][1:-1]
    dirs = []
    for i in range(len(tmd)):
        if '__pycache__' not in tmd[i]:
            dirs.append(tmd[i][10:] + '.plugin')
    
    if (len(dirs) == 1):
        print("1 plugin found")
    else:
        print("{:} plugins found".format(len(dirs)))

    modules = list(map(__import__, dirs))

    for i in range(len(modules)):
        print("Now Loading:", dirs[i])
        try:
            plugins.append(modules[i].plugin.Plugin())
        except Exception as OOF:
            print("Error loading", dirs[i])
            print("Error: ", OOF)

def getAllLamps():
    global plugins
    tml = []

    for i in range(len(plugins)):
        try: 
            tmp = plugins[i].getLamps()
            if tmp is not None:
                tml.append(tmp)
        except Exception as OOF:
            print("Error getting lamps from ", str(plugins[i]))
            print("Error: ", OOF)

    print("Number of lamps: ", len(tml))
    return tml
