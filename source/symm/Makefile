# use this makefile to build shared libaray.
CC = g++
ifeq ($(CC),g++) 
	CPP = g++
	OPTIM = -O3 -mssse3 
else 
	CPP=icpc
	OPTIM = -O3 -xSSE4.2 
endif

OBJ = libkssym.so
SOURCE = kssym.cc

INCPATH = ../../include
CFLAGS = -shared -fpic -lm 

$(OBJ): $(SOURCE)
	$(CPP) $(CFLAGS) $(OPTIM) -I$(INCPATH)   $^ -o $@ 


clean:
	rm -f *.so *.o 
