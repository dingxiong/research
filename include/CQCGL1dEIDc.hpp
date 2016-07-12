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
    ArrayXd K, QK;

    ArrayXcd L;
	
    FFT<double> fft;

    VectorXd hs;	      /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    
    struct NL {
	FFT<double> *fft;
	dcp B, G;

	NL(){}
	NL(FFT<double> &fft, dcp B, dcp G) : fft(&fft), B(B), G(G) {}
	~NL(){}

	void init(FFT<double> &fft, dcp B, dcp G){
	    this->G = &G;
	    this->B = B;
	    this->G = G;	    
	}

	VectorXcd A;
	void operator()(ArrayXcd &x, ArrayXcd &dxdt, double t){
	    Map<VectorXcd> xv(x.data(), x.size());
	    Map<VectorXcd> dxdtv(dxdt.data(), dxdt.size());
	    fft->inv(A, xv);
	    ArrayXcd A2 = u * u.transpose();
	    A = B * A * A2 + G * A * A2.square();
	    fft->fwd(dxdtv, A);
	    dxdtv.segment(Nplus, Nalias).setZero(); /* dealias */
	}
    };
    
    NL nl;

    ArrayXcd Yv[10], Nv[10];
    EIDc eidc;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    CQCGL1dEIDc(int N, double d, double W, double B, double C, double DR, double DI):
	N(N), d(d),
	Mu(Mu), Br(Br), Bi(Bi),
	Dr(Dr), Di(Di), Gr(Gr),
	Gi(Gi) 
    {
	Ne = (N/3) * 2 - 1;
	Ndim = 2 * Ne;
	Nplus = (Ne + 1) / 2;
	Nminus = (Ne - 1) / 2;
	Nalias = N - Ne;

	// calculate the Linear part
	K.resize(N,1);
	K << ArrayXd::LinSpaced(N/2, 0, N/2-1), N/2, ArrayXd::LinSpaced(N/2-1, -N/2+1, -1);       
	QK = 2*M_PI/d * K; 
	L = dcp(Mu, -Omega) - dcp(Dr, Di) * QK.square(); 
	L.segment(Nplus, Nalias) = ArrayXcd::Zero(Nalias); 

	nl.init(fft, dcp(Br, Bi), dcp(Gr, Gi));
	
	int nnl0 = eidc.nnls[eidc.scheme];
	for(int i = 0; i < nnl0; i++){
	    Yv[i].resize(N/2+1);
	    Nv[i].resize(N/2+1);
	}
	eidc.init(&L, Yv, Nv);
    }
    
   ~CQCGL1dEIDc(){}


    ////////////////////////////////////////////////////////////////////////////////////////////////////
    inline void 
    setScheme(std::string x){
	int nnl0 = eidr.nnls[eidr.scheme];	
	eidr.scheme = eidr.names[x];
	int nnl1 = eidr.nnls[eidr.scheme];
	for (int i = nnl0; i < nnl1; i++) {
	    Yv[i].resize(N/2+1);
	    Nv[i].resize(N/2+1);
	}
    }

    inline void 
    changeOmega(double w){
	Omega = w;
	L = dcp(Mu, -Omega) - dcp(Dr, Di) * QK.square();
	L.segment(Nplus, Nalias) = ArrayXcd::Zero(Nalias);
    }

    inline 
    ArrayXXd
    intg(const ArrayXd &a0, const double tend, const double h, const int skip_rate,
	 bool adapt=true){
	assert( Ndim == a0.size());
	
	ArrayXcd u0 = R2C(a0); 
	const int Nt = (int)round(tend/h);
	const int M = (Nt+skip_rate-1)/skip_rate;
	ArrayXXcd aa(N, M);
	lte.resize(M);
	int ks = 0;
	auto ss = [this, &ks, &aa](ArrayXcd &x, double t, double h, double err){
	    aa.col(ks) = x;
	    lte(ks++) = err;
	};
	
        eidr.intgC(nl, ss, 0, u0, tend, h, skip_rate); 
	
	return C2R(aa);
    }

    /** @brief transform conjugate matrix to its real form */
    ArrayXXd C2R(const ArrayXXcd &v){
	int n = v.rows();
	int m = v.cols();
	assert(N == n);
	ArrayXXcd vt(Ne, m);
	vt << v.topRows(Nplus), v.bottomRows(Nminus);

	return Map<ArrayXXd>((double*)(vt.data()), 2*vt.rows(), vt.cols());
    }

    ArrayXXcd R2C(const ArrayXXd &v){
	int n = v.rows();
	int m = v.cols();
	assert( n == Ndim);
    
	Map<ArrayXXcd> vp((dcp*)&v(0,0), Ne, m);
	ArrayXXcd vpp(N, m);
	vpp << vp.topRows(Nplus), ArrayXXcd::Zero(Nalias, m), vp.bottomRows(Nminus);

	return vpp;
    }


};

#endif	/* CQCGL1DEIDC_H */
