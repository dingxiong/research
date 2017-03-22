#!/bin/bash

# usage:
# ./createAll 

DEST=/usr/local/home/xiong/00git/research/lib/boostPython/ 

rm -rf bin pylib
b2 && mv pylib/py_CQCGL1dEIDc.so $DEST  && rm -rf bin pylib


