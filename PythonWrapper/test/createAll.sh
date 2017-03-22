#!/bin/bash

# usage:
# ./createAll

rm -rf bin pylib
b2 && mv pylib/py_test.*so /usr/local/home/xiong/00git/research/lib/boostPython/ && rm -rf bin pylib


