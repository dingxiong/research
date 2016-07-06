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
	    F[k].vc3 *= G;
	    k = (k+1) % 5;
	};
	
	std::vector<ArrayXcd*> Y = {&F[0].vc1, &F[1].vc1, &F[2].vc1, &F[3].vc1, &F[4].vc1};
	std::vector<ArrayXcd*> N = {&F[0].vc3, &F[1].vc3, &F[2].vc3, &F[3].vc3, &F[4].vc3};
	EIDr eidr(L, Y, N);
        eidr.intgC(nl, 0, u0, tend, h, skip_rate); 
    }

};


#endif	/* KSEIDR_H */
