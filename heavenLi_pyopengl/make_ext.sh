#!/bin/bash
clear
rm *.so
python3 setup.py build_ext -i -f
python3 -q -X faulthandler heavenli.py
