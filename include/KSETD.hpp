#ifndef KSETD_H
#define KSETD_H

#include "ETDRK4.hpp"
#include "myfft.hpp"

#include <Eigen/Dense>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
template<class Ary>
struct KSNL {
    typedef std::complex<double> dcp;

    int N;
    Eigen::ArrayXcd G;
    std::shared_ptr<MyFFT::RFFT> F;

    
    KSNL(int N, Eigen::ArrayXcd G) : N(N), G(G){
	F = std::make_shared<MyFFT::RFFT>(N, 1);
    }
    
    ~KSNL(){};
    
    Eigen::ArrayXcd operator()(double t, const Eigen::ArrayXcd x){
	F->vc1 = x; 
	F->ifft();
	F->vr2 = (F->vr2).square();
	F->fft();
	
	return F->vc3 * G;
    }
};


////////////////////////////////////////////////////////////////////////////////
class KSETD {
    
public:
    typedef std::complex<double> dcp;

    const int N;
    const double d;
    Eigen::ArrayXd L;
    Eigen::ArrayXd K;
    Eigen::ArrayXcd G;

    std::shared_ptr<ETDRK4<Eigen::ArrayXcd, Eigen::ArrayXXcd, KSNL>> etdrk4;
    
    KSETD(int N, double d);
    ~KSETD();

    Eigen::ArrayXXd C2R(const Eigen::ArrayXXcd &v);
    Eigen::ArrayXXcd R2C(const Eigen::ArrayXXd &v);
    std::pair<Eigen::VectorXd, Eigen::ArrayXXd>
    etd(const Eigen::ArrayXd &a0, const double tend, const double h, const int skip_rate,
	int method, bool adapt);

};


#endif	/* KSETD_H */
