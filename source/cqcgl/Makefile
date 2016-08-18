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

ifeq ($(t), int)		# for 1d integration
SOURCE = CQCGL1d.cc 

else ifeq ($(t), req1d)
SOURCE = CQCGL1dReq.cc

else ifeq ($(t), rpo1d)		# for RPO code
SOURCE = CQCGL1dRpo.cc 

else ifeq ($(t), int2d)		# for 2d CQCGL class
SOURCE = CQCGL2d.cc CQCGLgeneral2d.cc 

else ifeq ($(t), req2d)
SOURCE = CQCGL2dReq.cc

else ifeq ($(t), ani)
SOURCE = CQCGL2dDislin.cc

else
SOURCE = CQCGL1d.cc 

endif

SHARED = $(addprefix lib, $(addsuffix .so, $(basename $(word 1, $(SOURCE)))))
STATIC = $(addprefix lib, $(addsuffix .a, $(basename $(word 1, $(SOURCE)))))


all : $(SHARED) # $(STATIC) 

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
