# usage: make t=***
# or on pace cluster : make t=** pace=**

CC = g++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7 
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3
CFLAGS = -std=c++11 -DEIGEN_FFTW_DEFAULT
LIB = ../../lib

# for pace cluster 
ifdef pace
OPTIM = -O3
EIGEN = /nv/hp16/xding35/data/apps/eigen/include/eigen3
endif
####


ifndef t
t = int
endif

ifeq ($(t), int)		# for 1d integration
SOURCE = CQCGL1d.cc 

else ifeq ($(t), sub1d)		# sub1d integration
SOURCE = CQCGL1dSub.cc

else ifeq ($(t), req1d)		# for REQ code
SOURCE = CQCGL1dReq.cc

else ifeq ($(t), sub1dreq)	# sub1d req
SOURCE = CQCGL1dSubReq.cc

else ifeq ($(t), rpo1d)		# for RPO code
SOURCE = CQCGL1dRpo.cc 

else ifeq ($(t), rpo1dar)	# for 1d Rpo with arpack support
SOURCE = CQCGL1dRpo_arpack.cc
CFLAGS += -I$(XDAPPS)/arpackpp/include # -llapack -larpack -lsuperlu -lopenblas

else ifeq ($(t), int2d)		# for 2d CQCGL class
SOURCE = CQCGL2d.cc

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
	mv *.so *.a $(LIB)
