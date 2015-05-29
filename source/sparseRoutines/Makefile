CC = g++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3

CFLAGS = -lm -lfftw3 -std=c++0x 

SOURCE = sparseRoutines.cc
SHARED = libsparseRoutines.so
STATIC = libsparseRoutines.a

all : $(SHARED) $(STATIC) 

$(SHARED): $(SOURCE)
	$(CC) -shared -fpic $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE:.cc=.o) : $(SOURCE)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC): $(SOURCE:.cc=.o)
	$(AR) crs $@ $^

clean:
	rm -f *.so *.o *.a
