#ifndef KSEIDR_H
#define KSEIDR_H

#include "EIDr.hpp"
#include "myfft.hpp"
#include <Eigen/Dense>

////////////////////////////////////////////////////////////////////////////////
class KSEIDr {
    
public:
    typedef std::complex<double> dcp;
    
    const int N;
    const double d;
    Eigen::ArrayXd L;
    Eigen::ArrayXd K;
    Eigen::ArrayXcd G;
    
    MyFFT::RFFT F[5];
    
    
    KSEIDr(int N, double d);
    ~KSEIDr();

    Eigen::ArrayXXd C2R(const Eigen::ArrayXXcd &v);
    Eigen::ArrayXXcd R2C(const Eigen::ArrayXXd &v);
    void 
    etd(const Eigen::ArrayXd &a0, const double tend, const double h, const int skip_rate,
	int method, bool adapt){
	assert( N-2 == a0.size());
	ArrayXcd u0 = R2C(a0); 
    
	int k = 0;
	auto nl = [this, &k](ArrayXd &x, ArrayXd &dxdt, double t){
	    F[k].ifft();
	    F[k].vr2 = F[k].vr2 * F[k].vr2;
	    F[i].vc3 *= G;
	    k = (k+1) % 5;
	};
    
	EIDr eidr(L, nl);
	EIDr.intgC(0, u0, tend, h, skip_rate); 
    }

};


#endif	/* KSEIDR_H */
