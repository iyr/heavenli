#!/bin/bash
clear
rm drawButtons*.so
python3 setup.py build_ext -i
python3 heavenli.py
