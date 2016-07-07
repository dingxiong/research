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
    
    
    KSEIDr(int N, double d);
    ~KSEIDr();

    Eigen::ArrayXXd C2R(const Eigen::ArrayXXcd &v);
    Eigen::ArrayXXcd R2C(const Eigen::ArrayXXd &v);

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
	ArrayXXd aa(N-2, M);
	int ks = 0;
	auto ss = [this, &ks, &aa](ArrayXcd &x, double t){
	    Map<ArrayXcd> xv(x.data(), x.size());
	    aa.col(ks++) = C2R(xv);
	};
	
	std::vector<ArrayXcd*> Ys = {&Yv[0], &Yv[1], &Yv[2], &Yv[3], &Yv[4]};
	std::vector<ArrayXcd*> Ns = {&Nv[0], &Nv[1], &Nv[2], &Nv[3], &Nv[4]};
	EIDr eidr(L, Ys, Ns);
        eidr.intgC(nl, ss, 0, u0, tend, h, skip_rate); 
	
	return aa;
    }

};


#endif	/* KSEIDR_H */
