CC = g++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3

CFLAGS = -std=c++0x

FFTW_THREADS = no
ifeq ($(FFTW_THREADS), yes)
CFLAGS += -DTFFT
endif

MULTISHOOT_THREADS = yes
ifeq ($(MULTISHOOT_THREADS), yes)
CFLAGS += -DMULTISHOOT
endif


SOURCE = cqcgl1d.cc 
SHARED = libcqcgl1d.so
STATIC = libcqcgl1d.a

all : $(SHARED) $(STATIC) 

$(SHARED): $(SOURCE)
	$(CC) -shared -fpic $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE:.cc=.o) : $(SOURCE)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC): $(SOURCE:.cc=.o)
	$(AR) crs $@ $^

clean:
	rm -f *.so *.o *.a
