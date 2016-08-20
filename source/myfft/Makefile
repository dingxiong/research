CC = g++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3
CFLAGS = -std=c++11 
LIB = ../../lib

# for pace cluster 
ifdef pace
OPTIM = -O3
EIGEN = /nv/hp16/xding35/data/apps/eigen/include/eigen3
endif
####

CLEANUP = no
ifeq ($(CLEANUP), yes)
CFLAGS += -DCLEAN
endif

SOURCE = myfft.cc
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
	mv *.so *.a  $(LIB)
