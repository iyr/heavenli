from distutils.core import setup, Extension
import platform

if (platform.system() == 'Windows'):
    drawArn_sfc_module = Extension(
            'drawArn', 
            sources         = ['drawArn.cpp'],
            extra_link_args = ['opengl32.lib', 'glext.lib'])#, 'glew32.lib'])
    drawButtons_sfc_module = Extension(
            'drawButtons', 
            sources         = ['drawButtons.cpp'],
            extra_link_args = ['opengl32.lib'])
    shaderUtils_sfc_module = Extension(
            'shaderUtils', 
            sources         = ['shaderUtils.cpp'],
            extra_link_args = ['opengl32.lib', 'glext.lib'])#, 'glew32.lib'])
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

    shaderUtils_sfc_module = Extension(
            'shaderUtils', 
            sources             = ['shaderUtils.cpp'],
            extra_compile_args  = ['-fopenmp', '-O3', '-march=native'],
            extra_link_args     = ['-lGL', '-fopenmp'])


sfc_module = Extension('animUtils', sources = ['animUtils.c'])

setup(name='drawArn', version = '0.2',
        description = 'Light Arrangment Representations in C/C++',
        ext_modules = [drawArn_sfc_module]
        )

setup(name='drawButtons', version = '0.1',
        description = 'draw Buttons implemented in c',
        ext_modules = [drawButtons_sfc_module]
        )

setup(name='shaderUtils', version = '0.1',
        description = 'Utilities for loading, building and linking shaders',
        ext_modules = [shaderUtils_sfc_module]
        )

setup(name='animUtils', version = '0.1',
        description = 'Animation Curves implemented in c',
        ext_modules = [sfc_module]
        )
