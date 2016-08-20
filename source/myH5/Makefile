CC = h5c++
GCC = g++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3
CFLAGS = -std=c++11
LIB = ../../lib
hdf5 = /usr/lib/x86_64-linux-gnu


# for pace cluster 
ifdef pace
OPTIM = -O3
EIGEN = /nv/hp16/xding35/data/apps/eigen/include/eigen3
hdf5 = /nv/hp16/xding35/data/apps/hdf5
endif
####

H5FLAGS = -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_BSD_SOURCE -D_FORTIFY_SOURCE=2 -g -O2 -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security

H5LIB = $(hdf5)/libhdf5_hl_cpp.so $(hdf5)/libhdf5_cpp.so $(hdf5)/libhdf5_hl.so $(hdf5)/libhdf5.so -Wl,-Bsymbolic-functions -Wl,-z,relro -lpthread -lz -ldl -lm -Wl,-rpath -Wl,


# H5FLAGS = -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_BSD_SOURCE -D_FORTIFY_SOURCE=2 -g -O2 -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security

# H5LIB = -L/usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/libhdf5_hl_cpp.so /usr/lib/x86_64-linux-gnu/libhdf5_cpp.so /usr/lib/x86_64-linux-gnu/libhdf5_hl.so /usr/lib/x86_64-linux-gnu/libhdf5.so -Wl,-Bsymbolic-functions -Wl,-z,relro -lpthread -lz -ldl -lm -Wl,-rpath -Wl,/usr/lib/x86_64-linux-gnu


SOURCE = myH5.cc
SHARED = $(addprefix lib, $(addsuffix .so, $(basename $(SOURCE))))
STATIC = $(addprefix lib, $(addsuffix .a, $(basename $(SOURCE))))

# use flag -show to h5c++ to find out what library it uses
all : $(STATIC) $(SHARED)

$(SHARED): $(SOURCE)
	$(GCC) $(H5FLAGS) -c -shared -fPIC $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
	$(GCC) $(H5FLAGS) $(SOURCE:.cc=.o) -shared -fPIC $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) -o $@ $(H5LIB)
$(SOURCE:.cc=.o) : $(SOURCE)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC): $(SOURCE:.cc=.o)
	$(AR) crs $@ $^


clean:
	rm -f *.so *.o *.a
move:
	mv *.so *.a /usr/local/home/xiong/00git/research/lib
