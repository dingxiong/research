CC = g++
H5 = h5c++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3

CFLAGS = -lm -lfftw3 -std=c++0x 


SOURCE = ksint.cc ksintM1.cc 
SHARED = libksint.so
STATIC = libksint.a

REFINE_SOURCE = ksrefine.cc
REFINE_SHARED = libksrefine.so
REFINE_STATIC = libksrefine.a

DIM_SOURCE = ksdim.cc
DIM_STATIC = libksdim.a

FEFV_SOURCE = ksFEFV.cc
FEFV_STATIC = libksFEFV.a

all : $(SHARED) $(STATIC) $(REFINE_SHARED) $(REFINE_STATIC) $(DIM_STATIC) $(FEFV_STATIC)
int : $(SHARED) $(STATIC)
dim : $(DIM_STATIC)
fefv : $(FEFV_STATIC)

$(SHARED): $(SOURCE)
	$(CC) -shared -fpic $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE:.cc=.o) : $(SOURCE)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC): $(SOURCE:.cc=.o)
	$(AR) crs $@ $^

$(REFINE_SHARED)$ : $(REFINE_SOURCE)
	$(CC) -shared -fpic $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) \
	-L../../lib -lksint $^ -o $@
$(REFINE_SOURCE:.cc=.o) : $(REFINE_SOURCE)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(REFINE_STATIC) : $(REFINE_SOURCE:.cc=.o)
	$(AR) crs $@ $^


$(DIM_SOURCE:.cc=.o) : $(DIM_SOURCE)
	$(H5) -c -std=c++11 $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(DIM_STATIC) : $(DIM_SOURCE:.cc=.o)
	$(AR) crs $@ $^

$(FEFV_SOURCE:.cc=.o) : $(FEFV_SOURCE)
	$(H5) -c -std=c++11 $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(FEFV_STATIC) : $(FEFV_SOURCE:.cc=.o)
	$(AR) crs $@ $^

clean:
	rm -f *.so *.o *.a
move:
	mv *.so *.a /usr/local/home/xiong/00git/research/lib
