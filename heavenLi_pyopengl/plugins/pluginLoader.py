import sys, os, traceback

# Setup subdirectories to be loaded
PACKAGE_SUB = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_SUB)))

plugins = []

# Searches the folder "plugins" for compatible plugins and loads them
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
            print(traceback.format_exc())
            print("Error: ", OOF)

def updatePlugins():
    global plugins

    for i in range(len(plugins)):
        try:
            plugins[i].update()
        except Exception as OOF:
            print("Error updating", plugins[i])
            print(traceback.format_exc())
            print("Error: ", OOF)


# Returns of a list of all lamps provided by plugins
def getAllLamps():
    global plugins
    tml = []

    for i in range(len(plugins)):
        try: 
            tmp = plugins[i].getLamps()
            if len(tmp) > 0 and tmp is not None:
                for i in range(len(tmp)):
                    tml.append(tmp[i])
        except Exception as OOF:
            print(traceback.format_exc())
            print("Error: ", OOF)

    print("Number of lamps: ", len(tml))
    print(tml)
    return tml
