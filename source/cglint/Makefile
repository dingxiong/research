CC = g++
AR = ar

OPTIM = -O3 -msse2 -msse4 -march=corei7
INCLUDE = ../../include
EIGEN = /usr/local/home/xiong/apps/eigen/include/eigen3

CFLAGS = -std=c++11

SOURCE = cqcgl1d.cc 
SHARED = libcqcgl1d.so
STATIC = libcqcgl1d.a

# for RPO code
CFLAGS2 = -std=c++11

SOURCE2 = cqcglRPO.cc 
SHARED2 = $(addprefix lib, $(addsuffix .so, $(basename $(SOURCE2))))
STATIC2 = $(addprefix lib, $(addsuffix .a, $(basename $(SOURCE2))))
SHARED3 = $(addprefix lib, $(addsuffix _print.so, $(basename $(SOURCE2)))) 
STATIC3 = $(addprefix lib, $(addsuffix _print.a, $(basename $(SOURCE2)))) 


all : int rpo
int : $(SHARED) $(STATIC)
rpo : $(SHARED2) $(STATIC2) $(SHARED3) $(STATIC3) $(SHARED4) $(STATIC4) 

$(SHARED): $(SOURCE)
	$(CC) -shared -fpic $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SOURCE:.cc=.o) : $(SOURCE)
	$(CC) -c $(CFLAGS) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC): $(SOURCE:.cc=.o)
	$(AR) crs $@ $^

$(SHARED2): $(SOURCE2)
	$(CC) -shared -fpic $(CFLAGS2) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@
$(SHARED3): $(SOURCE2)
	$(CC) -shared -fpic $(CFLAGS2) -DGMRES_PRINT -DINB_PRINT $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^ -o $@

$(SOURCE2:.cc=.o) : $(SOURCE2)
	$(CC) -c $(CFLAGS2) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC2): $(SOURCE2:.cc=.o)
	$(AR) crs $@ $^
NAME1 = $(addsuffix _print.o, $(basename $(SOURCE2)))
$(NAME1): $(SOURCE2)
	$(CC) -c -o $@ -DGMRES_PRINT -DINB_PRINT $(CFLAGS2) $(OPTIM) -I$(INCLUDE) -I$(EIGEN) $^
$(STATIC3): $(NAME1)
	$(AR) crs $@ $^


clean:
	rm -f *.so *.o *.a
move:
	mv *.so *.a /usr/local/home/xiong/00git/research/lib
