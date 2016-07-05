#!/bin/bash

# usage:
# ./createAll 0 => compile all cases
# ./createAll 1 => single thread caes

# compile single thread library
if [ $1 -eq 0 ] || [ $1 -eq 1 ]; then
    perl -p -i -e 's/choice = \d/choice = 1/' Jamroot
    perl -p -i -e 's/MODULE\(py_CQCGL\S*\)/MODULE\(py_CQCGL\)/' py_CQCGL.cc
    rm -rf bin pylib
    b2 && mv pylib/py_CQCGL*.so /usr/local/home/xiong/00git/research/lib/boostPython/
    rm -rf bin pylib
    echo "compile single thread library"
fi

# compile multi fftw threads library
if [ $1 -eq 0 ] || [ $1 -eq 2 ]; then
    perl -p -i -e 's/choice = \d/choice = 2/' Jamroot
    perl -p -i -e 's/MODULE\(py_CQCGL\S*\)/MODULE\(py_CQCGL_threads\)/' py_CQCGL.cc
    rm -rf bin pylib
    b2 && mv pylib/py_CQCGL*.so /usr/local/home/xiong/00git/research/lib/boostPython/
    rm -rf bin pylib
    echo "compile multi fftw threads library"
fi
