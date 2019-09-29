from distutils.core import setup, Extension
from platform import system, machine
print("Building for", system(), machine())

if (system() == 'Windows'):
    hliGLutils_sfc_module = Extension(
            'hliGLutils', 
            sources         = ['hliGLutils.cpp'],
            extra_link_args = ['opengl32.lib', 'glext.lib'])
    #drawArn_sfc_module = Extension(
            #'drawArn', 
            #sources         = ['drawArn.cpp'],
            #extra_link_args = ['opengl32.lib', 'glext.lib'])
    #drawButtons_sfc_module = Extension(
            #'drawButtons', 
            #sources         = ['drawButtons.cpp'],
            #extra_link_args = ['opengl32.lib', 'glext.lib'])
    #fontUtils_sfc_module = Extension(
            #'fontUtils', 
            #sources         = ['fontUtils.cpp'],
            #extra_link_args = ['opengl32.lib', 'glext.lib'])
    #shaderUtils_sfc_module = Extension(
            #'shaderUtils', 
            #sources         = ['shaderUtils.cpp'],
            #extra_link_args = ['opengl32.lib', 'glext.lib'])
else:
    hliGLutils_sfc_module = Extension(
            'hliGLutils', 
            sources             = ['hliGLutils.cpp'],
            extra_compile_args  = ['-fopenmp', '-O3', '-march=native'],
            extra_link_args     = ['-lGL', '-fopenmp'])

    #drawArn_sfc_module = Extension(
            #'drawArn', 
            #sources             = ['drawArn.cpp'],
            #extra_compile_args  = ['-fopenmp', '-O3', '-march=native'],
            #extra_link_args     = ['-lGL', '-fopenmp'])

    #drawButtons_sfc_module = Extension(
            #'drawButtons', 
            #sources             = ['drawButtons.cpp'],
            #extra_compile_args  = ['-fopenmp', '-O3', '-march=native'],
            #extra_link_args     = ['-lGL', '-fopenmp'])

    #fontUtils_sfc_module = Extension(
            #'fontUtils', 
            #sources             = ['fontUtils.cpp'],
            #extra_compile_args  = ['-fopenmp', '-O3', '-march=native'],
            #extra_link_args     = ['-lGL', '-fopenmp'])

    #shaderUtils_sfc_module = Extension(
            #'shaderUtils', 
            #sources             = ['shaderUtils.cpp'],
            #extra_compile_args  = ['-fopenmp', '-O3', '-march=native'],
            #extra_link_args     = ['-lGL', '-fopenmp'])


sfc_module = Extension('animUtils', sources = ['animUtils.c'])

setup(name='hliGLutils', version = '0.2',
        description = 'HeavenLi OpenGL utility set',
        ext_modules = [hliGLutils_sfc_module]
        )

#setup(name='drawArn', version = '0.2',
        #description = 'Light Arrangment Representations in C/C++',
        #ext_modules = [drawArn_sfc_module]
        #)

#setup(name='drawButtons', version = '0.1',
        #description = 'draw Buttons implemented in c',
        #ext_modules = [drawButtons_sfc_module]
        #)

#setup(name='fontUtils', version = '0.1',
        #description = 'draw text implemented in c++',
        #ext_modules = [fontUtils_sfc_module]
        #)

#setup(name='shaderUtils', version = '0.1',
        #description = 'Utilities for loading, building and linking shaders',
        #ext_modules = [shaderUtils_sfc_module]
        #)

setup(name='animUtils', version = '0.1',
        description = 'Animation Curves implemented in c',
        ext_modules = [sfc_module]
        )
