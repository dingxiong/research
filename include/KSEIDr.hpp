#ifndef KSEIDR_H
#define KSEIDR_H

#include "EIDr.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

////////////////////////////////////////////////////////////////////////////////
class KSEIDr {
    
public:
    typedef std::complex<double> dcp;
    
    const int N;
    const double d;
    Eigen::ArrayXd L;
    Eigen::ArrayXd K;
    Eigen::ArrayXcd G;
    
    FFT<double> fft;
    
    
    KSEIDr(int N, double d) :  N(N), d(d) {
	K = ArrayXd::LinSpaced(N/2+1, 0, N/2) * 2 * M_PI / d;
	K(N/2) = 0;
	L = K*K - K*K*K*K;
	G = 0.5 * dcp(0,1) * K * N; 

	fft.SetFlag(fft.HalfSpectrum);
    }
    
    ~KSEIDr(){};

    inline
    ArrayXXd C2R(const ArrayXXcd &v){
	int n = v.rows();
	int m = v.cols();
	ArrayXXcd vt = v.middleRows(1, n-2);
	ArrayXXd vp(2*(n-2), m);
	vp = Map<ArrayXXd>((double*)&vt(0,0), 2*(n-2), m);

	return vp;
    }
    
    inline
    ArrayXXcd R2C(const ArrayXXd &v){
	int n = v.rows();
	int m = v.cols();
	assert( 0 == n%2);
	
	ArrayXXcd vp = ArrayXXcd::Zero(n/2+2, m);
	vp.middleRows(1, n/2) = Map<ArrayXXcd>((dcp*)&v(0,0), n/2, m);
	
	return vp;
    }

    inline 
    ArrayXXd
    intgC(const Eigen::ArrayXd &a0, const double tend, const double h, const int skip_rate,
	  int method, bool adapt){
	assert( N-2 == a0.size());
	ArrayXcd Yv[5], Nv[5];
	for (int i = 0; i < 5; i++) {
	    Yv[i].resize(N/2+1);
	    Nv[i].resize(N/2+1);
	}

	ArrayXcd u0 = R2C(a0); 
	auto nl = [this](ArrayXcd &x, ArrayXcd &dxdt, double t){
	    VectorXd u;
	    Map<VectorXcd> xv(x.data(), x.size());
	    Map<VectorXcd> dxdtv(dxdt.data(), dxdt.size());
	    fft.inv(u, xv);
	    u = u.array().square();
	    fft.fwd(dxdtv, u);
	    dxdt *=  G;
	};
	
	const int Nt = (int)round(tend/h);
	const int M = (Nt+skip_rate-1)/skip_rate + 1;
	ArrayXXcd aa(N/2+1, M);
	int ks = 0;
	auto ss = [this, &ks, &aa](ArrayXcd &x, double t){
	    aa.col(ks++) = x;
	};
	
	EIDr eidr(L, Yv, Nv);
        eidr.intgC(nl, ss, 0, u0, tend, h, skip_rate); 
	
	return C2R(aa);
    }

};


#endif	/* KSEIDR_H */
