CC = h5c++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3

CFLAGS = -std=c++11

SOURCE = myH5.cc
SHARED = libmyH5.so
STATIC = libmyH5.a

# H5Cpp does not support shared library
all : $(STATIC)

$(SHARED): $(SOURCE)
	$(CC) -shared -fPIC $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE:.cc=.o) : $(SOURCE)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC): $(SOURCE:.cc=.o)
	$(AR) crs $@ $^


clean:
	rm -f *.so *.o *.a
move:
	mv *.so *.a /usr/local/home/xiong/00git/research/lib
