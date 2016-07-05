# usage: make t=***

CC = g++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3
CFLAGS = -std=c++11 

ifndef t
t = int
endif

ifeq ($(t), int)
SOURCE = ksint.cc 

else ifeq ($(t), dim)
SOURCE = ksdim.cc

else ifeq ($(t), fefv)
SOURCE = ksFEFV.cc

else ifeq ($(t), refine)
SOURCE = ksrefine.cc

else ifeq ($(t), ei)
SOURCE = KSEIDr.cc

endif

SHARED = $(addprefix lib, $(addsuffix .so, $(basename $(word 1, $(SOURCE)))))
STATIC = $(addprefix lib, $(addsuffix .a, $(basename $(word 1, $(SOURCE)))))

all : $(SHARED) $(STATIC) 

$(SHARED): $(SOURCE)
	$(CC) -shared -fpic $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE:.cc=.o) : $(SOURCE)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC): $(SOURCE:.cc=.o)
	$(AR) crs $@ $^

clean:
	rm -f *.so *.o *.a
move:
	mv *.so *.a /usr/local/home/xiong/00git/research/lib
