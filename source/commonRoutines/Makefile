CC = g++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3

CFLAGS = -lm -lfftw3 -std=c++0x 

SOURCE = sparseRoutines.cc
SHARED = libsparseRoutines.so
STATIC = libsparseRoutines.a

SOURCE2 = denseRoutines.cc
SHARED2 = libdenseRoutines.so
STATIC2 = libdenseRoutines.a

all : $(SHARED) $(STATIC) $(SHARED2) $(STATIC2) 

$(SHARED): $(SOURCE)
	$(CC) -shared -fpic $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE:.cc=.o) : $(SOURCE)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC): $(SOURCE:.cc=.o)
	$(AR) crs $@ $^

$(SHARED2): $(SOURCE2)
	$(CC) -shared -fpic $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE2:.cc=.o) : $(SOURCE2)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC2): $(SOURCE2:.cc=.o)
	$(AR) crs $@ $^

clean:
	rm -f *.so *.o *.a
move:
	mv *.so *.a /usr/local/home/xiong/00git/research/lib
