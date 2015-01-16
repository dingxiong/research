CC = g++
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

all : $(SHARED) $(STATIC) $(REFINE_SHARED) $(REFINE_STATIC)

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
clean:
	rm -f *.so *.o *.a