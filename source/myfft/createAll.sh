#!/bin/bash

make && make move

perl -pe 's/CLEANUP = no/CLEANUP = yes/' Makefile > tmp
make -f tmp && mv libmyfft.so libmyfft_clean.so \
&& mv libmyfft.a libmyfft_clean.a && make move

rm tmp
