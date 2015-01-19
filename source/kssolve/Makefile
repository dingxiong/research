# use this makefile to build shared libaray.
CC = g++
ifeq ($(CC),g++) 
	CPP = g++
	OPTIM = -O3 -mssse3 
else 
	CPP=icpc
	OPTIM = -O3 -xSSE4.2 
endifd

OBJ = libkssolve.so
SOURCE = kssolve.cc kssolveM1.cc

INCPATH = ../../include
CFLAGS = -shared -fpic -lm -lfftw3 -std=c++11

$(OBJ): $(SOURCE)
	$(CPP) $(CFLAGS) $(OPTIM) -I$(INCPATH)   $^ -o $@ 


clean:
	rm -f *.so *.o 
