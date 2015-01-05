CC = h5c++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
LD = ../../lib
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3

CFLAGS = -std=c++0x
LINK = -lksint -lped -lfftw3 -lm

SRC = readks.cc
SHARED = $(addprefix lib, $(SRC:.cc=.so))
STATIC = $(addprefix lib, $(SRC:.cc=.a))

# I do not know why it cannot generate shared library
#all : $(SHARED) $(STATIC)
all : $(STATIC)
$(SHARED): $(SRC)
	$(CC) -shared -fPIC $(CFLAGS) $(OPTIM) -I$(INCLUDE) \
	 -I$(EIGEN) -L$(LD) $^ -o $@ $(LINK)
$(SRC:.cc=.o) : $(SRC)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ \
	$(LINK)
$(STATIC): $(SRC:.cc=.o)
	$(AR) crs $@ $^
clean:
	rm -f *.so *.o *.a