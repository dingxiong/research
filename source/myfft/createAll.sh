#!/bin/bash

make && make move

perl -pe 's/FFTW_THREADS = no/FFTW_THREADS = yes/' Makefile > tmp
make -f tmp && mv libmyfft.so libmyfft_threads.so \
 && mv libmyfft.a libmyfft_threads.a && make move

perl -pe 's/CLEANUP = no/CLEANUP = yes/' Makefile > tmp
make -f tmp && mv libmyfft.so libmyfft_clean.so \
 && mv libmyfft.a libmyfft_clean.a && make move

perl -pe 's/CLEANUP = no/CLEANUP = yes/; s/FFTW_THREADS = no/FFTW_THREADS = yes/' Makefile > tmp
make -f tmp && mv libmyfft.so libmyfft_clean_threads.so \
 && mv libmyfft.a libmyfft_clean_threads.a && make move

rm tmp
