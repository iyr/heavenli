from distutils.core import setup, Extension
import platform

if (platform.system() == 'Windows'):
    drawButtons_sfc_module = Extension(
            'drawButtons', 
            sources         = ['drawButtons.cpp'],
            extra_link_args = ['opengl32.lib'])
    drawArn_sfc_module = Extension(
            'drawArn', 
            sources         = ['drawArn.cpp'],
            extra_link_args = ['opengl32.lib'])
else:
    drawButtons_sfc_module = Extension(
            'drawButtons', 
            sources             = ['drawButtons.cpp'],
            extra_compile_args  = ['-fopenmp', '-O3', '-march=native'],
            extra_link_args     = ['-lGL', '-fopenmp'])

    drawArn_sfc_module = Extension(
            'drawArn', 
            sources             = ['drawArn.cpp'],
            extra_compile_args  = ['-fopenmp', '-O3', '-march=native'],
            extra_link_args     = ['-lGL', '-fopenmp'])


setup(name='drawButtons', version = '0.1',
        description = 'draw Buttons implemented in c',
        ext_modules = [drawButtons_sfc_module]
        )

setup(name='drawArn', version = '0.1',
        description = 'draw Arrangements implemented in c',
        ext_modules = [drawArn_sfc_module]
        )

sfc_module = Extension('animUtils', sources = ['animUtils.c'])

setup(name='animUtils', version = '0.1',
        description = 'Animation Curves implemented in c',
        ext_modules = [sfc_module]
        )
