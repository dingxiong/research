import ctypes
from ctypes import *
from numpy import *
a0 = ones(30)*0.1;
h=0.25
nstp=10
np=1
nqr=1
d=22
# get the truncation numbe
N = len(a0) + 2

arr = ctypes.c_double * (N-2)
arr2 = ctypes.c_double * (nstp/np*(N-2)) # ctypes array object.
arr3 = ctypes.c_double * (nstp/nqr*(N-2)**2)

h = ctypes.c_double(h);
nstp = ctypes.c_int(nstp);
np = ctypes.c_int(np);
nqr = ctypes.c_int(nqr);
d = ctypes.c_double(d);

v = a0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
xx = ones(60); aa = xx.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
dxx=ones(1800);daa= dxx.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


ks = ctypes.cdll.LoadLibrary('./lib/libkssolve.so')

ks.ksfj(v, d, h, nstp, np, nqr, aa, daa)
#aa = numpy.ctypeslib.as_array(aa[:60])
#daa = numpy.ctypeslib.as_array(daa)
