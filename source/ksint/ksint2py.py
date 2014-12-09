from ctypes import *
from numpy import *

class AT(Structure):
    _fields_ = [("aa", POINTER(c_double)),
                ("tt", POINTER(c_double))]

ks = cdll.LoadLibrary('./libks2py.so')

def ksint(a0, h, nstp, np, d):
    
    N = len(a0) + 2
    
    ch = c_double(h)
    cnstp = c_int(nstp)
    cnp = c_int(np)
    cd = c_double(d)
    
    pa0 = a0.ctypes.data_as(POINTER(c_double))
    aa = empty((nstp/np+1)*(N-2)) 
    paa = aa.ctypes.data_as(POINTER(c_double))
    
    ks.ksf(paa, pa0, cd, ch, cnstp, cnp)
    aa = aa.reshape((nstp/np+1, N-2))
    # aa = aa.reshape((N-2, nstp/np+1), order='F')
    
    return aa

