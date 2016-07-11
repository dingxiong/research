#ifndef CQCGL1DEIDC_H
#define CQCGL1DEIDC_H

#include "EIDr.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

class CQCGL1dEIDc {
    
public :

    typedef std::complex<double> dcp;
    
    const int N;		/* dimension of FFT */
    const double d;		/* system domain size */

    int Ne, Ndim, Nplus, Nminus, Nalias;
    
    double Br, Bi, Gr, Gi, Dr, Di, Mu;
    double Omega = 0;
    ArrayXd K, K2, QK;

    ArrayXcd L;
	

    CQCGL1dEIDc(int N, double d, double W, double B, double C, double DR, double DI):
	N(N), d(d),
	Mu(Mu), Br(Br), Bi(Bi),
	Dr(Dr), Di(Di), Gr(Gr),
	Gi(Gi),
    {
	Ne = (N/3) * 2 - 1;
	Ndim = 2 * Ne;
	Nplus = (Ne + 1) / 2;
	Nminus = (Ne - 1) / 2;
	Nalias = N - Ne;

	ArrayXd Kindex(N);
	Kindex << ArrayXd::LinSpaced(N/2, 0, N/2-1), 0, ArrayXd::LinSpaced(N/2-1, -N/2+1, -1);
      
	K = 2*M_PI/d * Kindex;
	L = dcp(1, W) - dcp(1, B) * K.square();

	CqcglNL<ArrayXcd> nl(N, C, DR, DI);
	etdrk4 = std::make_shared<ETDRK4c<ArrayXcd, ArrayXXcd, CqcglNL>>(L, nl);
    }
    
   ~CQCGL1dEIDc(){}


    ArrayXXd CQCGL1dEIDc::C2R(const ArrayXXcd &v){
	// allocate memory for new array, so it will not change the original array.
	return Map<ArrayXXd>((double*)&v(0,0), 2*v.rows(), v.cols());
    }

    ArrayXXcd CQCGL1dEIDc::R2C(const ArrayXXd &v){
	assert( 0 == v.rows() % 2);
	return Map<ArrayXXcd>((dcp*)&v(0,0), v.rows()/2, v.cols());
    }

    ArrayXXd CQCGL1dEIDc::pad(const Ref<const ArrayXXd> &aa){
	int n = aa.rows();		
	int m = aa.cols();
	assert(Ndim == n);
	ArrayXXd paa(2*N, m);
	paa << aa.topRows(2*Nplus), ArrayXXd::Zero(2*Nalias, m), 
	    aa.bottomRows(2*Nminus);
	return paa;
    }

    ArrayXXd CQCGL1dEIDc::unpad(const Ref<const ArrayXXd> &paa){
	int n = paa.rows();
	int m = paa.cols();
	assert(2*N == n);
	ArrayXXd aa(Ndim, m);
	aa << paa.topRows(2*Nplus), paa.bottomRows(2*Nminus);
	return aa;
    }


    std::pair<VectorXd, ArrayXXd>
    CQCGL1dEIDc::etd(const ArrayXd &a0, const double tend, const double h, const int skip_rate, 
		  int method, bool adapt){
	
	assert( Ndim == a0.size() );
	ArrayXcd u0 = R2C(pad(a0)); 
    
	etdrk4->Method = method;
    
	std::pair<VectorXd, ArrayXXcd> tmp;
	if (adapt) tmp = etdrk4->intg(0, u0, tend, h, skip_rate); 
	else tmp = etdrk4->intgC(0, u0, tend, h, skip_rate); 
    
	return std::make_pair(tmp.first, unpad(C2R(tmp.second)));
    }

};

#endif	/* CQCGL1DEIDC_H */
