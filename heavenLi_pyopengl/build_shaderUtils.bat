cls
del shaderUtils*.pyd
python setup.py build_ext -i -c msvc
python heavenli.py