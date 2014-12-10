from ctypes import *
from numpy import *

class AT(Structure):
    _fields_ = [("aa", POINTER(c_double)),
                ("tt", POINTER(c_double))]



def ksint(a0, h, nstp, np = 1, d = 22):
    """
    % integrate KS system.
    % input:
    %      a0 : initial condition (must be 30x1)
    %      h : time step
    %      nstp: number of integration steps
    %      d : size of system. Default = 22
    %      np : saving spacing. Default = 1
    % output:
    %      aa : the orbit.  size [30, nstp/np+1]
    % exmple usage:
    %   aa = ksint(a0, 0.25, 1000);
    """
    ks, N, ch, cnstp, cnp, cd, pa0, aa, paa = init(a0, h, nstp, np, d)
    ks.ksf(paa, pa0, cd, ch, cnstp, cnp)
    aa = aa.reshape((nstp/np+1, N-2))
    # aa = aa.reshape((N-2, nstp/np+1), order='F')
    
    return aa

def ksintM1(a0, h, nstp, np = 1, d = 22):
    """
    % integrate KS system on the 1st mode slice.
    % input:
    %      a0 : initial condition (must be 30x1, a0(2) = 0)
    %      h : time step
    %      nstp: number of integration steps
    %      d : size of system. Default = 22
    %      np : saving spacing. Default = 1
    % output:
    %    tt : time sequence in the full state space [nstp/np+1]
    %    aa : trajectory in the 1st mode slice [30, nstp/np+1]
    % exmple usage:
    %   tt, aa = ksintM1(a0, 0.25, 1000);

    """
    ks, N, ch, cnstp, cnp,cd, pa0, aa, paa = init(a0, h, nstp, np, d)
    tt = empty(nstp/np+1)
    ptt = tt.ctypes.data_as(POINTER(c_double))
    
    ks.ksfM1(paa, ptt, pa0, cd, ch, cnstp, cnp)
    aa = aa.reshape((nstp/np+1, N-2))
    
    return tt, aa

def ksint2M1(a0, h, T, np = 1, d = 22):
    """
    % integrate KS system on the 1st mode slice.
    % input:
    %      a0 : initial condition (must be 30x1, a0(2) = 0)
    %      h : time step
    %      nstp: number of integration steps
    %      d : size of system. Default = 22
    %      np : saving spacing. Default = 1
    % output:
    %      tt : time in the full state space 
    %      aa : trajectory on the 1st mode slice
    % exmple usage:
    %   [tt,aa] = ksint2M1(a0, 0.25, 1000);
    """
    ks = cdll.LoadLibrary('./libks2py.so')
    ks.ksf2M1.argtypes = [POINTER(AT), POINTER(c_double), 
                          c_double, c_double, c_double, c_int]
    ks.ksf2M1.restype = c_int
    ks.freeks.argtypes = [POINTER(AT)]

    N = len(a0) + 2
    
    ch = c_double(h)
    cT = c_double(T)
    cnp = c_int(np)
    cd = c_double(d)
    
    pa0 = a0.ctypes.data_as(POINTER(c_double))
    at = AT()
    M = ks.ksf2M1(byref(at), pa0, cd, ch, cT, cnp)
    aa = fromiter(at.aa, dtype=double, count=M*(N-2))
    aa = aa.reshape((M, N-2))
    tt = fromiter(at.tt, dtype=double, count=M)
    ks.freeks(at) # free the allocated memory. Just once

    return tt, aa
    
def init(a0, h, nstp, np, d):
    
    ks = cdll.LoadLibrary('./libks2py.so')
    
    N = len(a0) + 2
    
    ch = c_double(h)
    cnstp = c_int(nstp)
    cnp = c_int(np)
    cd = c_double(d)
    
    pa0 = a0.ctypes.data_as(POINTER(c_double))
    aa = empty((nstp/np+1)*(N-2)) 
    paa = aa.ctypes.data_as(POINTER(c_double))

    return ks, N, ch, cnstp, cnp, cd, pa0, aa, paa

