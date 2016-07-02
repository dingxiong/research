import numpy  as np 
from numpy.fft import rfft, irfft
from pdb import set_trace

def ksintIm(a0, h, nstp, d = 22, osp = 1, jsp = 1, isJ = False):
     N = len(a0); 
     a0p = np.zeros(N*2)
     a0p[1::2] = a0
     if isJ == True:
         tt, aa, daa = ksfjaco(a0p, h, nstp, osp, jsp, d, isJ)
         aa = aa[1::2, :]
         daa = daa.T.reshape(-1, 30)[1::2, 1::2]
         return aa.T, daa
     else :
         tt, aa = ksfjaco(a0p, h, nstp, osp, jsp, d, isJ)
         aa = aa[1::2, :]
         return aa.T
         
def ksfjaco(a0, h, nstp, osp = 1, jsp = 1, d =22, isJ = True):
    """
    KS integrator.

    Parameters
    ----------
    a0: one dimensional array or a list
    initial condition. It shoud be a one dimensional array or a list.
    h : double
    time step.
    nstp: int
    number of integration steps.
    osp: int, optional
    spacing for storing state vectors. If not given, osp = 1.
    jsp: int, optional
    spacing for Jacobian matrix. If not given, jsp = 1.
    d: int, optional
    length of KS system. If not given, d = 22.
    isJ: bool, optional
    Whethe to calculate the Jacobian matrix. If not given, isJ = True.

    Returns
    -------
    tt: one dimensional array
    (n,) time samples
    aa: narray
    (n,m) each column represent a state vector.
    daa: narray (only if isJ = True)
    (n**2, m) each column represent a row-stacked Jacobian.

    Examples
    --------
    >>> from numpy inport ones
    >>> tt, aa, daa= ksjaco(ones(30)*0.1, 0.25, 1000)
    >>> tt, aa = ksjaco(ones(30)*0.1, 0.25, 1000, isJ = False)
    >>> tt.shape
    (1000,)
    >>> aa.shape
    (30, 1000)
    >>> daa.shape
    (900, 1000)

    """
    N = len(a0)+2; Nh = N/2;
    a0=np.array(a0);

    if isJ:
        v = np.zeros((Nh+1, N-1), np.complex128);
        v[1:Nh, 0] = a0[::2] + 1j*a0[1::2];
        iniJ(v[:,1:]); 
    else:
        v = np.zeros([Nh+1,1], np.complex128);
        v[1:Nh, 0] = a0[::2] + 1j*a0[1::2];

    k, E, E2, Q, f1, f2, f3 = calCoe(h, Nh, d);
    g = 0.5j * np.r_[k[:-1], 0] * N; g = g[:, np.newaxis];
    E = E[:,np.newaxis]; E2 = E2[:,np.newaxis]; Q = Q[:,np.newaxis];
    f1 = f1[:,np.newaxis]; f2 = f2[:,np.newaxis]; f3 = f3[:,np.newaxis];
    if isJ:
        g = np.hstack((g, np.tile(2.*g, N-2)));
        daa = np.empty([(N-2)**2, nstp/jsp]);

    aa = np.empty([N-2, nstp/osp+1]); tt = np.empty([1, nstp/osp+1]);
    aa[:,0]=a0; tt[0,0]=0; 

    for n in range(1, nstp+1):
        t = n*h; rfv = irfft(v, axis=0); 
        Nv = g*rfft(rfv[:,0:1] * rfv, axis = 0); 

        a = E2*v + Q*Nv; rfv = irfft(a, axis=0); 
        Na = g*rfft(rfv[:,0:1] * rfv, axis = 0);

        b = E2*v + Q*Na; rfv =irfft(b, axis = 0);
        Nb = g*rfft(rfv[:,0:1] * rfv, axis = 0);

        c = E2*a + Q*(2*Nb - Nv); rfv =irfft(c, axis = 0);
        Nc = g*rfft(rfv[:,0:1] * rfv, axis = 0);

        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3; 

        if n % osp == 0:
            y1 = np.c_[v[1:Nh,0].real, v[1:Nh,0].imag];
            aa[:, n/osp] = y1.reshape(-1); tt[0,n/osp] =t;
        #set_trace();  
        if isJ and n % jsp == 0: 
            da = np.zeros((N-2, N-2)); da[::2,:]= v[1:Nh, 1:].real
            da[1::2,:] = v[1:Nh, 1:].imag; daa[:,n/jsp-1] = da.reshape(-1);
            iniJ(v[:,1:]);

    if isJ:
        return tt, aa, daa
    else:
        return tt, aa

def iniJ(J):
    """
    initialize the Jacobian matrix to be
    [ 0, 0, 0, ...
    1, 1j,0, ...
    0, 0, 1, 1j, ..
    ...
    0, 0, .... 1, 1j
    0, 0, 0, ...
    ]

    Parameters
    ----------
    J : array_like
    Jacobian matrix with size (Nh+1) x (N-2).

    """
    J[:,:]=0;
    for i in range(1, J.shape[0]-1):
        J[i, 2*i-2] = 1;
        J[i, 2*i-1] = 1j;

    return None

def calCoe(h, Nh, d, M = 16):
    k = 2.*np.pi/d * np.r_[:Nh+1]; # wave number: 0, 1, 2,.. Nh-1, Nh
    L = k**2 - k**4;
    E = np.exp(h * L);
    E2 = np.exp(h * L/ 2.);
    r = np.exp(1j*np.pi*(np.r_[1:M+1]-0.5)/M);
    LR = h*np.tile(L,(M,1)).T + np.tile(r,(Nh+1,1));
    Q = h*((np.exp(LR/2)-1)/LR).mean(1).real;
    f1 = h*((-4 - LR + np.exp(LR)*(4-3*LR+LR**2))/LR**3).mean(1).real;
    f2 = h*((2 + LR + np.exp(LR)*(-2+LR))/LR**3).mean(1).real;
    f3 = h*((-4 - 3*LR - LR**2 + np.exp(LR)*(4 - LR))/LR**3).mean(1).real;

    return k, E, E2, Q, f1, f2, f3;

def transJ(J):
    n, m = J.shape; n = n**0.5; #set_trace();
    Jp = np.empty([n, m*n]);
    for i in range(m):
        Jp[:, i*n:(i+1)*n] = J[:,i].reshape([n,n]);

    return Jp
