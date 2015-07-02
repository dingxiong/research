#!/bin/bash -e

# compile single thread library
perl -p -i -e 's/choice = \d/choice = 1/' Jamroot
perl -p -i -e 's/MODULE\(py_cqcgl1d\S*\)/MODULE\(py_cqcgl1d\)/' py_cqcgl1d.cc
rm -rf bin pylib
b2 && mv pylib/py_cqcgl1d*.so /usr/local/home/xiong/00git/research/lib/boostPython/

# compile multi threads library
perl -p -i -e 's/choice = \d/choice = 2/' Jamroot
perl -p -i -e 's/MODULE\(py_cqcgl1d\S*\)/MODULE\(py_cqcgl1d_threads\)/' py_cqcgl1d.cc
rm -rf bin pylib
b2 && mv pylib/py_cqcgl1d*.so /usr/local/home/xiong/00git/research/lib/boostPython/

