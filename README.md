# HeavenLi: 
##### HeavenLi Light interface
---
![N|Solid](https://forwardsweep.net/upload/2019/08/08/20190808230018-4e4b5151.gif)

Intuitive, form-centric, performant. 
HeavenLi Light interface is an extensible application for controlling color-capable (RGB) lights. 
HeavenLi is written in Python3 with all graphical draw code written in C/C++ and OpenGL.

### Contents
 - [Builds](#builds)
 - [Supported Platforms](#supported-platforms)
 - [Build Instructions](#build-instructions)

### Builds
| Platform | Build |
| ------ | ------ |
| Windows (portable) | [heavenli-alpha1.0-windows.zip][heavenli_alpha1.0_windows] |

### Supported platforms
- Windows (tested: Windows 10)
- Linux (tested: Fedora 23)
- Raspbian (tested: Stretch, Buster, Pi3 & Pi0)

### Build Instructions
##### Linux/Raspbian (may be incomplete on some systems):
__1.)__ Install Python3 and required packages via OS package manager:
Fedora 22 and newer:
```sh
$ sudo dnf install -y python3 mesa-utils freeglut3-dev
```
Raspbian Stretch and newer:
```sh
$ sudo apt-get install python3 mesa-utils freeglut3-dev
```
__2.)__ Install necessary python3 packages:
```sh
$ pip3 install pyopengl numpy cobs
```
__3.)__ Grab Repo and build:
```sh
$ cd ~
$ git clone https://github.com/iyr/heavenli.git
$ cd heavenli/heavenli_pyopengl/
$ ./make_ext.sh
```
__Note for Raspberry Pi users:__
Compilation may take a while on slower models (eg. Raspberry Pi Zero).
If you wish to run HeavenLi on an HDMI display, it is recommended to enable the "Full KMS" OpenGL driver:
```sh
$ sudo raspi-config
... 7 Advanced Settings ->  Enable Full KMS
$ reboot
```
For users looking to run HeavenLi on a non-HDMI display including the Raspberry Pi Official 7" touchscreen, and various Adafruit PiTFT touchscreens, you must enable to the "Fake KMS" OpenGL driver:
```sh
$ sudo raspi-config
... 7 Advanced Settings ->  Enable Fake KMS
$ reboot
```

##### Windows 10 (may be incomplete on some systems):
__1.)__ Install __Microsoft Visual Studio__ (Recommended: Community), __include Windows 10 SDKs during installation__
[Microsoft Visual Studio][mvs]

__2.)__ Install __Miniconda x86 Python 3.x__ ( 64-bit HeavenLi is not supported on Windows at this time )
When, prompted __add Miniconda to PATH__ otherwise compilation will likely fail due not finding requisite files.
[Miniconda][miniconda]

__3.)__ Install __PyOpenGL__:
download PyOpenGL wheel from UC Irvine's python libs, grab the version matching your python subversion (eg. if using Python 3.6, download "PyOpenGL‑3.1.3b2‑cp36‑cp36m‑win32.whl" )
[PyOpenGL][pyopenglwindl]
Open a windows command line:
```sh
cd c:\**PATH TO WHEEL FILE**\
pip install PyOpenGL*.whl
```
__"Why do I need to download the .whl file? Can't I just install PyOpenGL from PyPI with pip3?"__
As of writing, the PyOpenGL package from the official Python Package Index is broken on Windows and HeavenLi will fail to build/run.

__4.)__ Install necessary python3 packages:
```sh
$ pip3 install numpy cobs pyserial
```

__5.)__ Install OpenGL Extensions (glext)
[glext][glextdl]
Move all files in `glext\include\gl\` to `c:\Program Files(x86)\Windows Kits\10\include\**VERSION**\um\gl\`
Move `glext\lib\glext.lib` to `c:\Program Files(x86)\Windows Kits\10\Lib\**VERSION**\um\x86\`
Move `glext\lib\glext.dll` to `c:\windows\system32\`

[heavenli_alpha1.0_windows]: <https://github.com/iyr/heavenli/raw/master/builds/heavenli_alpha1.0_windows.zip>

[mvs]:<https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2019>

[miniconda]:<https://docs.conda.io/en/latest/miniconda.html>

[pyopenglwindl]:<https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl>

[glextdl]:<https://sourceforge.net/projects/glextwin32/>

__6.)__ Download and extract github repo

__7.)__ Navigate to repo directory and build
In a windows command line:
```sh
cd c:\**PATH TO HEAVENLI REPO**\heavenli_pyopengl
.\make_ext.bat
```
