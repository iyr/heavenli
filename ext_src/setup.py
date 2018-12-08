from distutils.core import setup, Extension

sfc_module = Extension('animUtils', sources = ['animUtils.c'])

setup(name='animUtils', version = '0.1',
        description = 'Animation Curves implemented in c',
        ext_modules = [sfc_module]
        )
