#ifndef CQCGL1DEIDC_H
#define CQCGL1DEIDC_H

#include "EIDc.hpp"
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
	
    FFT<double> fft;		// support MatrixBase but not ArrayBase

    VectorXd hs;	      /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    int cellSize = 500;
    
    struct NL {
	CQCGL1dEIDc *cgl;
	dcp B, G;
	VectorXcd A;
	
	NL(CQCGL1dEIDc *cgl) : cgl(cgl), B(cgl->Br, cgl->Bi), G(cgl->Gr, cgl->Gi) {
	    A.resize(cgl->N);
	}
	~NL(){}	

	void operator()(ArrayXXcd &x, ArrayXXcd &dxdt, double t){
	    Map<VectorXcd> xv(x.data(), x.size());
	    Map<VectorXcd> dxdtv(dxdt.data(), dxdt.size());
	    cgl->fft.inv(A, xv);
	    ArrayXcd A2 = A.array() * A.array().conjugate();
	    A = B * A.array() * A2 + G * A.array() * A2.square();
	    cgl->fft.fwd(dxdtv, A);
	    dxdtv.segment(cgl->Nplus, cgl->Nalias).setZero(); /* dealias */
	}
    };
    
    NL nl;

    ArrayXXcd Yv[10], Nv[10];
    EIDc eidc;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    CQCGL1dEIDc(int N, double d, double Mu, double Dr, double Di, 
		double Br, double Bi, double Gr, double Gi):
	N(N), d(d), Mu(Mu), Dr(Dr), Di(Di), Br(Br), Bi(Bi), Gr(Gr), Gi(Gi),
	nl(this)
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
	L.segment(Nplus, Nalias).setZero();
	
	int nYN0 = eidc.names.at(eidc.scheme).nYN; // do not call setScheme here. Different.
	for(int i = 0; i < nYN0; i++){
	    Yv[i].resize(N, 1);
	    Nv[i].resize(N, 1);
	}
	eidc.init(&L, Yv, Nv);
    }
    
    CQCGL1dEIDc(int N, double d,
		double b, double c,
		double dr, double di)
	: CQCGL1dEIDc(N, d, -1, 1, b, 1, c, -dr, -di)
    { }

   ~CQCGL1dEIDc(){}


    ////////////////////////////////////////////////////////////////////////////////////////////////////
    inline void 
    setScheme(std::string x){
	int nYN0 = eidc.names.at(eidc.scheme).nYN;
	eidc.scheme = x;
	int nYN1 = eidc.names.at(eidc.scheme).nYN;
	for (int i = nYN0; i < nYN1; i++) {
	    Yv[i].resize(N, 1);
	    Nv[i].resize(N, 1);
	}
    }

    inline void 
    changeOmega(double w){
	Omega = w;
	L = dcp(Mu, -Omega) - dcp(Dr, Di) * QK.square();
	L.segment(Nplus, Nalias).setZero();
    }

    inline 
    ArrayXXd
    intgC(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
	assert( Ndim == a0.size());
	
	ArrayXXcd u0 = R2C(a0); 
	const int Nt = (int)round(tend/h);
	const int M = (Nt + skip_rate - 1) / skip_rate;
	ArrayXXcd aa(N, M);
	lte.resize(M);
	int ks = 0;
	auto ss = [this, &ks, &aa](ArrayXXcd &x, double t, double h, double err){
	    aa.col(ks) = x;
	    lte(ks++) = err;
	};
	
        eidc.intgC(nl, ss, 0, u0, tend, h, skip_rate); 
	
	return C2R(aa);
    }

    inline 
    ArrayXXd
    intg(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
	assert( Ndim == a0.size());

	ArrayXXcd u0 = R2C(a0); 
	const int Nt = (int)round(tend/h);
	const int M = (Nt+skip_rate-1)/skip_rate;
	ArrayXXcd aa(N, M);
	Ts.resize(M);
	hs.resize(M);
	lte.resize(M);
	int ks = 0;
	auto ss = [this, &ks, &aa](ArrayXXcd &x, double t, double h, double err){
	    int m = Ts.size();
	    if (ks >= m ) {
		Ts.conservativeResize(m+cellSize);
		aa.conservativeResize(Eigen::NoChange, m+cellSize); // rows not change, just extend cols
		hs.conservativeResize(m+cellSize);
		lte.conservativeResize(m+cellSize);
	    }
	    hs(ks) = h;
	    lte(ks) = err;
	    aa.col(ks) = x;
	    Ts(ks++) = t;
	};
	
	eidc.intg(nl, ss, 0, u0, tend, h, skip_rate);
	
	hs.conservativeResize(ks);
	lte.conservativeResize(ks);
	Ts.conservativeResize(ks);
	aa.conservativeResize(Eigen::NoChange, ks);

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
