# This makefile tries to compile the shared library for matlab
# You should note the difference with the ordinary C/C++ complilation procedure.
# 
#  1) As with Boost.Python, mex cannot deal with shared linkages, so except the bottom
#     library, all other source files must be compiled.
#     

CC = mex
CXX = '/usr/bin/g++-4.7'
CXXFLAGS = '-std=c++11 -fPIC -O3'
CXXFLAGS_THREADS = '-std=c++11 -fPIC -O3 -DTFFT'


INCLUDE = -I../../../include -I/usr/local/home/xiong/apps/eigen/include/eigen3
LIB = -L../../../lib -L/usr/lib/x86_64-linux-gnu
LINK = -lm -lfftw3_threads -lfftw3 -lped -lsparseRoutines -ldenseRoutines -literMethod


SOURCE = ../cqcgl1d.cc ../../myfft/myfft.cc


all : cqcglIntgv cqcglNdim cqcglGintgv

cqcglIntgv : cqcglIntgv.cpp $(SOURCE)
	$(CC) CXX=$(CXX) CXXFLAGS=$(CXXFLAGS_THREADS) $(INCLUDE) $(LIB) $(LINK) $^
	mv $@.mexa64 $@_threads.mexa64
	$(CC) CXX=$(CXX) CXXFLAGS=$(CXXFLAGS) $(INCLUDE) $(LIB) $(LINK) $^

cqcglNdim : cqcglNdim.cpp $(SOURCE)
	$(CC) CXX=$(CXX) CXXFLAGS=$(CXXFLAGS) $(INCLUDE) $(LIB) $(LINK) $^

cqcglGintgv : cqcglGintgv.cpp $(SOURCE)
	$(CC) CXX=$(CXX) CXXFLAGS=$(CXXFLAGS_THREADS) $(INCLUDE) $(LIB) $(LINK) $^
	mv $@.mexa64 $@_threads.mexa64
	$(CC) CXX=$(CXX) CXXFLAGS=$(CXXFLAGS) $(INCLUDE) $(LIB) $(LINK) $^

clean:
	rm -f *.mexa64
move:
	mv *.mexa64  /usr/local/home/xiong/00git/research/lib/mex
