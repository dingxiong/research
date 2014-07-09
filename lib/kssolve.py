import ctypes, numpy
def ksfjaco(a0, h, nstp, d = 22, np = 1, nqr = 1, isJ = True):

    # get the truncation number
    N = len(a0) + 2

    ch = ctypes.c_double(h);
    cnstp = ctypes.c_int(nstp);
    cnp = ctypes.c_int(np);
    cnqr = ctypes.c_int(nqr);
    cd = ctypes.c_double(d);

    pa0 = a0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    aa = numpy.empty(nstp/np*(N-2));
    paa = aa.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    
    ks = ctypes.cdll.LoadLibrary('./libkssolve.so')

    if isJ:
        daa = numpy.empty(nstp/nqr*(N-2)**2);
        pdaa = daa.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ks.ksfj(pa0, cd, ch, cnstp, cnp, cnqr, paa, pdaa)
        aa = aa.reshape((N-2, nstp/np), order = 'F')
        daa = daa.reshape( ((N-2)**2, nstp/nqr), order = 'F')

        return aa, daa
    
    else:
        ks.ksf(pa0, cd, ch, cnstp, cnp, paa)
        aa = aa.reshape((N-2, nstp/np), order = 'F')
        
        return aa

    
