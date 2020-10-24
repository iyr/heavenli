from distutils.core import setup, Extension
from platform import system, machine
import numpy
print("Building for", system(), machine())

if (system() == 'Windows'):
    hliGLutils_sfc_module = Extension(
            'hliGLutils', 
            sources         = ['hliGLutils.cpp'],
            extra_link_args = ['opengl32.lib', 'glext.lib'])
else:
    hliGLutils_sfc_module = Extension(
            'hliGLutils', 
            sources             = ['hliGLutils.cpp'],
            extra_compile_args  = ['-march=native'],
            extra_link_args     = ['-lGL', '-fopenmp'])

#sfc_module = Extension('animUtils', sources = ['animUtils.c'])

setup(name='hliGLutils', version = '0.2',
        description = 'HeavenLi OpenGL utility set',
        ext_modules = [hliGLutils_sfc_module],
        include_dirs= [numpy.get_include()]
        )

#setup(name='animUtils', version = '0.1',
        #description = 'Animation Curves implemented in c',
        #ext_modules = [sfc_module]
        #)
