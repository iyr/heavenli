#!/bin/bash
clear
rm *.so
python3 setup.py build_ext -i -f
python3 heavenli.py
