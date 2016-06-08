CC = g++
CC2 = h5c++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3

# for integration
CFLAGS = -std=c++11

SOURCE = CQCGLgeneral.cc CQCGL.cc
SHARED = libCQCGL.so
STATIC = libCQCGL.a

# for 2d CQCGL class
SOURCE4 = CQCGLgeneral2d.cc CQCGL2d.cc	
SHARED4 = libCQCGL2d.so
STATIC4 = libCQCGL2d.a 	

# for RPO code
CFLAGS2 = -std=c++11

SOURCE2 = CQCGLRPO.cc 
SHARED2 = $(addprefix lib, $(addsuffix .so, $(basename $(SOURCE2))))
STATIC2 = $(addprefix lib, $(addsuffix .a, $(basename $(SOURCE2))))

# for ETD code
CFLAGS3 = -std=c++11

SOURCE3 = CqcglETD.cc 
SHARED3 = $(addprefix lib, $(addsuffix .so, $(basename $(SOURCE3))))
STATIC3 = $(addprefix lib, $(addsuffix .a, $(basename $(SOURCE3))))


all : int rpo etd int2d
int : $(SHARED) $(STATIC)
rpo : $(SHARED2) $(STATIC2) 
etd : $(SHARED3) $(STATIC3)
int2d : $(SHARED4) $(STATIC4)

$(SHARED): $(SOURCE)
	$(CC) -shared -fpic $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE:.cc=.o) : $(SOURCE)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC): $(SOURCE:.cc=.o)
	$(AR) crs $@ $^

$(SHARED2): $(SOURCE2)
	$(CC) -shared -fpic $(CFLAGS2) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE2:.cc=.o) : $(SOURCE2)
	$(CC) -c $(CFLAGS2) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC2): $(SOURCE2:.cc=.o)
	$(AR) crs $@ $^

$(SHARED3): $(SOURCE3)
	$(CC) -shared -fpic $(CFLAGS3) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE3:.cc=.o) : $(SOURCE3)
	$(CC) -c $(CFLAGS3) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC3): $(SOURCE3:.cc=.o)
	$(AR) crs $@ $^

$(SHARED4): $(SOURCE4)
	$(CC) -shared -fpic $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE4:.cc=.o) : $(SOURCE4)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC4): $(SOURCE4:.cc=.o)
	$(AR) crs $@ $^


clean:
	rm -f *.so *.o *.a
move:
	mv *.so *.a /usr/local/home/xiong/00git/research/lib
