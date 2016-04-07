#ifndef CQCGLETD_H
#define CQCGLETD_H

#include "ETDRK4c.hpp"
#include "myfft.hpp"

#include <Eigen/Dense>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
template<class Ary>
struct CqcglNL {
    typedef std::complex<double> dcp;

    int N, Nplus, Nalias;
    double C, DR, DI;

    Eigen::ArrayXcd G;
    std::shared_ptr<MyFFT::FFT> F;

    
    CqcglNL(int N, double C, double DR, double DI) : N(N), C(C), DR(DR), DI(DI){
	int Ne = (N/3) * 2 - 1;
	int Nplus = (Ne+1) / 2;
	int Nalias = N - Ne;
	F = std::make_shared<MyFFT::FFT>(N, 1);
    }
    
    ~CqcglNL(){};
    
    Eigen::ArrayXcd operator()(double t, const Eigen::ArrayXcd x){
	F->v1 = x;
	F->ifft();
	ArrayXd A2 = (F->v2).real().square() + (F->v2).imag().square(); 
	F->v2 =  dcp(1, C) * F->v2 * A2 - dcp(DR, DI) * F->v2 * A2.square();
	F->fft();
	
	(F->v3).middleRows(Nplus, Nalias) = ArrayXXcd::Zero(Nalias, (F->v3).cols());

	return F->v3;

    }
    
};


////////////////////////////////////////////////////////////////////////////////
class CqcglETD {
    
public:
    typedef std::complex<double> dcp;

    const int N;
    const double d;
    const double W;
    const double B, C, DR, DI;

    int Ne, Ndim, Nplus, Nminus, Nalias;
    Eigen::ArrayXcd L;
    Eigen::ArrayXd K;

    std::shared_ptr<ETDRK4c<Eigen::ArrayXcd, Eigen::ArrayXXcd, CqcglNL>> etdrk4;
    
    CqcglETD(int N, double d, double W, double B, double C, double DR, double DI);
    ~CqcglETD();

    Eigen::ArrayXXd C2R(const Eigen::ArrayXXcd &v);
    Eigen::ArrayXXcd R2C(const Eigen::ArrayXXd &v);
    Eigen::ArrayXXd pad(const Eigen::Ref<const Eigen::ArrayXXd> &aa);
    Eigen::ArrayXXd unpad(const Eigen::Ref<const Eigen::ArrayXXd> &paa);
    
    std::pair<Eigen::VectorXd, Eigen::ArrayXXd>
    etd(const Eigen::ArrayXd &a0, const double tend, const double h, const int skip_rate,
	int method, bool adapt);

};


#endif	/* CQCGLETD_H */
