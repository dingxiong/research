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
    ArrayXd L;
    ArrayXd K;
    ArrayXcd G;
    
    FFT<double> fft;

    VectorXd hs;	      /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    int cellSize = 500;

    struct NL {
	FFT<double> *fft;
	ArrayXcd *G;

	NL(){}
	NL(FFT<double> &fft, ArrayXcd &G) : fft(&fft), G(&G) {}
	~NL(){}

	void init(FFT<double> &fft, ArrayXcd &G){
	    this->G = &G;
	    this->fft = &fft;
	}

	VectorXd u;
	void operator()(ArrayXcd &x, ArrayXcd &dxdt, double t){
	    Map<VectorXcd> xv(x.data(), x.size());
	    Map<VectorXcd> dxdtv(dxdt.data(), dxdt.size());
	    fft->inv(u, xv);
	    u = u.array().square();
	    fft->fwd(dxdtv, u);
	    dxdt *=  (*G);
	}
    };
    
    NL nl;
    
    ArrayXcd Yv[10], Nv[10];
    EIDr eidr;

    /* ============================================================ */
    KSEIDr(int N, double d) :  N(N), d(d) {
	K = ArrayXd::LinSpaced(N/2+1, 0, N/2) * 2 * M_PI / d;
	K(N/2) = 0;
	L = K*K - K*K*K*K;
	G = 0.5 * dcp(0,1) * K * N; 

	fft.SetFlag(fft.HalfSpectrum);
	nl.init(fft, G);
	
	int nYN0 = eidr.nYNs[eidr.scheme];
	for(int i = 0; i < nYN0; i++){
	    Yv[i].resize(N/2+1);
	    Nv[i].resize(N/2+1);
	}
	eidr.init(&L, Yv, Nv);
	
    }
    
    ~KSEIDr(){};

    /* ============================================================ */
    inline void 
    setScheme(std::string x){
	int nYN0 = eidr.nYNs[eidr.scheme];	
	eidr.scheme = eidr.names[x];
	int nYN1 = eidr.nYNs[eidr.scheme];
	for (int i = nYN0; i < nYN1; i++) {
	    Yv[i].resize(N/2+1);
	    Nv[i].resize(N/2+1);
	}
    }

    inline 
    ArrayXXd
    intgC(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
	assert( N-2 == a0.size());
	
	ArrayXcd u0 = R2C(a0); 
	const int Nt = (int)round(tend/h);
	const int M = (Nt+skip_rate-1)/skip_rate;
	ArrayXXcd aa(N/2+1, M);
	lte.resize(M);
	int ks = 0;
	auto ss = [this, &ks, &aa](ArrayXcd &x, double t, double h, double err){
	    aa.col(ks) = x;
	    lte(ks++) = err;
	};
	
        eidr.intgC(nl, ss, 0, u0, tend, h, skip_rate); 
	
	return C2R(aa);
    }

    inline 
    ArrayXXd
    intg(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
	assert( N-2 == a0.size());

	ArrayXcd u0 = R2C(a0); 
	const int Nt = (int)round(tend/h);
	const int M = (Nt+skip_rate-1)/skip_rate;
	ArrayXXcd aa(N/2+1, M);
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
	
	eidr.intg(nl, ss, 0, u0, tend, h, skip_rate);
	
	hs.conservativeResize(ks);
	lte.conservativeResize(ks);
	Ts.conservativeResize(ks);
	aa.conservativeResize(Eigen::NoChange, ks);

	return C2R(aa);
    }



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

};


#endif	/* KSEIDR_H */
