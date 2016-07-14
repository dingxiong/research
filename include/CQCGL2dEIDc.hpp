#ifndef CQCGL1DEIDC_H
#define CQCGL1DEIDC_H

#include "EIDc.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

class CQCGL2dEIDc {
    
public :

    typedef std::complex<double> dcp;
    
    const int N, M;		/* dimension of FFT */
    const double dx, dy;	/* system domain size */
    int Ne, Me;
    int Nplus, Nminus, Nalias;
    int Mplus, Mminus, Malias;
    
    double Br, Bi, Gr, Gi, Dr, Di, Mu;
    double Omega = 0;
    ArrayXd Kx, QKx, Ky, QKy;
    ArrayXXcd L;
	
    FFT<double> fft;

    VectorXd hs;	      /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    int cellSize = 500;
    
    struct NL {
	CQCGL2dEIDc *cgl;
	dcp B, G;
	MatrixXcd A;
	
	NL(CQCGL1dEIDc *cgl) : cgl(cgl), B(cgl->Br, cgl->Bi), G(cgl->Gr, cgl->Gi) {
	    A.resize(cgl->M, cgl->N);
	}
	~NL(){}	
	
	void operator()(ArrayXcd &x, ArrayXcd &dxdt, double t){
	    Map<MatrixXcd> xv(x.data(), cgl->M, cgl->N);
	    Map<MatrixXcd> dxdtv(dxdt.data(), cgl->M, cgl->N);
	    cgl->fft.inv2(A, xv);
	    ArrayXXcd A2 = A.array() * A.array().conjugate();
	    A = B * A.array() * A2 + G * A.array() * A2.square();
	    cgl->fft.fwd(dxdtv, A);
	    dxdtv.middleRows(cgl->Mplus, cgl->Malias).setZero(); /* dealias */
	    dxdtv.middlecols(cgl->Nplus, cgl->Nalias).setZero(); /* dealias */
	}
    };
    
    NL nl;

    ArrayXcd Yv[10], Nv[10];
    EIDc eidc;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    CQCGL1dEIDc(int N, double d, double Mu, double Dr, double Di, 
		double Br, double Bi, double Gr, double Gi):
	N(N), d(d), Mu(Mu), Dr(Dr), Di(Di), Br(Br), Bi(Bi), Gr(Gr), Gi(Gi),
	nl(this)
    {
	Ne = (N/3) * 2 - 1;
	Nplus = (Ne + 1) / 2;
	Nminus = (Ne - 1) / 2;
	Nalias = N - Ne;

	Me = (M/3) * 2 - 1;
	Mplus = (Me + 1) / 2;
	Mminus = (Me - 1) / 2;
	Malias = M - Me;

	// calculate the Linear part
	Kx.resize(N,1);
	Kx << ArrayXd::LinSpaced(N/2, 0, N/2-1), N/2, ArrayXd::LinSpaced(N/2-1, -N/2+1, -1);
	Ky.resize(M,1);
	Ky << ArrayXd::LinSpaced(M/2, 0, M/2-1), 0, ArrayXd::LinSpaced(M/2-1, -M/2+1, -1);        

	QKx = 2*M_PI/dx * Kx;  
	QKy = 2*M_PI/dy * Ky;
    
	L = dcp(Mu, -Omega) - dcp(Dr, Di) * (QKx.square().replicate(1, M).transpose() + 
					     QKy.square().replicate(1, N)); 
	L.middleRows(Mplus, Malias).setZero();
	L.middleCols(Nplus, Nalias).setZero();
	L.resize(M*N, 1);
	
	int nnl0 = eidc.nnls[eidc.scheme];
	for(int i = 0; i < nnl0; i++){
	    Yv[i].resize(M*N);
	    Nv[i].resize(M*N);
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
	int nnl0 = eidc.nnls[eidc.scheme];	
	eidc.scheme = eidc.names[x];
	int nnl1 = eidc.nnls[eidc.scheme];
	for (int i = nnl0; i < nnl1; i++) {
	    Yv[i].resize(M*N);
	    Nv[i].resize(M*N);
	}
    }

    inline void 
    changeOmega(double w){
	Omega = w;
	L = dcp(Mu, -Omega) - dcp(Dr, Di) * (QKx.square().replicate(1, M).transpose() + 
					     QKy.square().replicate(1, N)); 
	L.middleRows(Mplus, Malias) = ArrayXXcd::Zero(Malias, N); 
	L.middleCols(Nplus, Nalias) = ArrayXXcd::Zero(M, Nalias);
	L.resize(M*N, 1);
    }

    inline 
    ArrayXXd
    intgC(const ArrayXXcd &a0, const double h, const double tend, const int skip_rate,
	  const bool doSaveDisk, const string fileName){
	
	ArrayXXcd aa;
	H5File file;
	if (doSaveDisk){
	    file = H5File(fileName, H5F_ACC_TRUNC); /* openFile fails */
	    saveState(file, 0, a0, v0, onlyOrbit ? 1 : 0);
	}
	else{
	    const int Nt = (int)round(tend/h);
	    const int m = (Nt+skip_rate-1)/skip_rate;
	    aa.resize(Me, Ne*m);
	    lte.resize(m);
	}
	
	ArrayXcd u0 = pad(a0); 

	int ks = 0;
	auto ss = [this, &ks, &aa, &file](ArrayXcd &x, double t, double h, double err){
	    Map<ArrayXXcd> xv(x.data(), M, N);
	    if(doSaveDisk){
		char groupName[10];
		sprintf (groupName, "%.6d", ks);
		std::string s = "/"+std::string(groupName);
		Group group(file.createGroup(s));
		std:: string DS = s + "/";
		
		writeMatrixXd(file, DS + "ar", xv.real());
		writeMatrixXd(file, DS + "ai", xv.imag());
	    }
	    else{
		aa.middleCols((ks)*Ne, Ne) = unpad(xv);
	    }
	    lte(ks++) = err;
	};
	
        eidc.intgC(nl, ss, 0, u0, tend, h, skip_rate); 
	
	return aa;
    }

    inline 
    ArrayXXd
    intg(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
	assert( Ndim == a0.size());

	ArrayXcd u0 = R2C(a0); 
	const int Nt = (int)round(tend/h);
	const int M = (Nt+skip_rate-1)/skip_rate;
	ArrayXXcd aa(N, M);
	Ts.resize(M);
	hs.resize(M);
	lte.resize(M);
	int ks = 0;
	auto ss = [this, &ks, &aa](ArrayXcd &x, double t, double h, double err){
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

    ArrayXXcd unpad(const ArrayXXcd &v){
	int m = v.rows();
	int n = v.cols();
	assert(m == M && n % N == 0);
	int s = n / N;
    
	ArrayXXcd vt(Me, Ne*s);
	for (int i = 0; i < s; i++){
	
	    vt.middleCols(i*Ne, Ne) <<
	    
		v.block(0, i*N, Mplus, Nplus), 
		v.block(0, i*N+Nplus+Nalias, Mplus, Nminus),

		v.block(Mplus+Malias, i*N, Mminus, Nplus),
		v.block(Mplus+Malias, i*N+Nplus+Nalias, Mminus, Nminus)
		;
	}
    
	return vt;
    }

    ArrayXXcd pad(const ArrayXXcd &v){
	int m = v.rows();
	int n = v.cols();
	assert( n % Ne == 0 && m == Me);
	int s = n / Ne;
    
	ArrayXXcd vp(M, N*s);
	for (int i = 0; i < s; i++){
	
	    vp.middleCols(i*N, N) << 
	    
		v.block(0, i*Ne, Mplus, Nplus), 
		ArrayXXcd::Zero(Mplus, Nalias), 
		v.block(0, i*Ne+Nplus, Mplus, Nminus),
	    
		ArrayXXcd::Zero(Malias, N),
	    
		v.block(Mplus, i*Ne, Mminus, Nplus), 
		ArrayXXcd::Zero(Mminus, Nalias), 
		v.block(Mplus, i*Ne+Nplus, Mminus, Nminus)
		;
	}
    
	return vp;
    }

};

#endif	/* CQCGL1DEIDC_H */
