CC = g++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3
CFLAGS = -lm -lfftw3 -std=c++11
LIB = ../../lib

ifdef pace
OPTIM = -O3
EIGEN = /nv/hp16/xding35/data/apps/eigen/include/eigen3
endif
####

SOURCE = sparseRoutines.cc
SHARED = $(addprefix lib, $(addsuffix .so, $(basename $(word 1, $(SOURCE)))))
STATIC = $(addprefix lib, $(addsuffix .a, $(basename $(word 1, $(SOURCE)))))

SOURCE2 = denseRoutines.cc
SHARED2 = $(addprefix lib, $(addsuffix .so, $(basename $(word 1, $(SOURCE2)))))
STATIC2 = $(addprefix lib, $(addsuffix .a, $(basename $(word 1, $(SOURCE2)))))

all : $(SHARED) $(SHARED2) # $(STATIC2) $(STATIC)

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
	mv *.so *.a $(LIB)
