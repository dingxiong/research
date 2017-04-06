#!/bin/bash

# usage:
# ./createAll 0 => compile all cases
# ./createAll 1 => single thread caes

rm -rf bin pylib
b2 && mv pylib/py_CQCGL1dSub.so /usr/local/home/xiong/00git/research/lib/boostPython/
rm -rf bin pylib

