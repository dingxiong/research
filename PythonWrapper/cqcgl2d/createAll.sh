#!/bin/bash

# usage:
# ./createAll 0 => compile all cases
# ./createAll 1 => single thread caes

DEST=/usr/local/home/xiong/00git/research/lib/boostPython/ 

# compile single thread library
if [ $1 -eq 0 ] || [ $1 -eq 1 ]; then
    mv Jamroot jam.tmp && cp jam.tmp Jamroot
    perl -p -i -e 's/choice = \d/choice = 1/' Jamroot
    perl -p -i -e 's/py_CQCGL2d.cc/pytmp.cc/' Jamroot
    perl -p -e 's/MODULE\(py_CQCGL2d\S*\)/MODULE\(py_CQCGL2d\)/' py_CQCGL2d.cc > pytmp.cc
    rm -rf bin pylib
    b2 && mv pylib/py_CQCGL2d*.so $DEST && mv jam.tmp Jamroot && rm pytmp.cc && rm -rf bin pylib
    echo
    echo "compile single thread library"
fi

# compile multi fftw threads library
if [ $1 -eq 0 ] || [ $1 -eq 2 ]; then
    mv Jamroot jam.tmp && cp jam.tmp Jamroot
    perl -p -i -e 's/choice = \d/choice = 2/' Jamroot
    perl -p -i -e 's/py_CQCGL2d.cc/pytmp.cc/' Jamroot
    perl -p -e 's/MODULE\(py_CQCGL2d\S*\)/MODULE\(py_CQCGL2d_threads\)/' py_CQCGL2d.cc > pytmp.cc
    rm -rf bin pylib
    b2 && mv pylib/py_CQCGL2d*.so $DEST && mv jam.tmp Jamroot && rm pytmp.cc && rm -rf bin pylib
    echo 
    echo "compile multi fftw threads library"
fi

