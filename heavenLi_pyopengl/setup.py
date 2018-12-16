from distutils.core import setup, Extension
import platform

if (platform.system() == 'Windows'):
    sfc_module = Extension(
            'drawButtons', 
            sources = ['drawButtons.cpp'],
            extra_link_args=['opengl32.lib'])
else:
    sfc_module = Extension(
            'drawButtons', 
            sources = ['drawButtons.cpp'],
            extra_link_args=['lGL'])


setup(name='drawButtons', version = '0.1',
        description = 'draw Buttons implemented in c',
        ext_modules = [sfc_module]
        )

#sfc_module = Extension('animUtils', sources = ['animUtils.c'])

#setup(name='animUtils', version = '0.1',
        #description = 'Animation Curves implemented in c',
        #ext_modules = [sfc_module]
        #)
