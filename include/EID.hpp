#ifndef EID_H
#define EID_H

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>
#include "denseRoutines.hpp"

using namespace std;
using namespace Eigen;

// using namespace denseRoutines;

//////////////////////////////////////////////////////////////////////////////////
//                        Real ETDRK4 class
//////////////////////////////////////////////////////////////////////////////////

/**
 * @brief ETDRK4 integration class
 *
 * DT   : data type (double/complex)
 * NL   : the nonlinear function type
 */
template<typename DT>
class EID {
    
public:

    typedef std::complex<double> dcp;
    typedef Array<DT, Dynamic, 1> Ary;

    //////////////////////////////////////////////////////////////////////

    Ary &L;
    ArrayXcd *N, *Y;

    enum Scheme {
	Cox_Matthews,
	Krogstad,
	Hochbruck_Ostermann,
	Luan_Ostermann
    };
    Scheme scheme = Cox_Matthews; /* scheme  */
    std::unordered_map<int, int> nstages = { /* number of stages */
        {Cox_Matthews,        4},
        {Krogstad,            4},
        {Hochbruck_Ostermann, 5},
	{Luan_Ostermann,      8}
    };

    Ary a[9][9], b[9], c[9];
    double err;
    
    int M = 64;			/* number of sample points */
    int R = 1;			/* radius for evaluating phi(z) */
    
    ////////////////////////////////////////////////////////////
    // time adaptive method related parameters
    double rtol = 1e-8;
    double nu = 0.9;		/* safe factor */
    double mumax = 2.5;		/* maximal time step increase factor */
    double mumin = 0.4;		/* minimal time step decrease factor */
    double mue = 1.25;		/* upper lazy threshold */
    double muc = 0.85;		/* lower lazy threshold */

    int NCalCoe = 0;		/* times to evaluate coefficient */
    int NReject = 0;		/* times that new state is rejected */
    int NCallF = 0;	       /* times to call velocity function f */
    int NSteps = 0;	      /* total number of integrations steps */
    
    VectorXd hs;	       /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    
    ////////////////////////////////////////////////////////////
    // constructor and desctructor
    EID(Ary &L, ArrayXcd *Y, ArrayXcd *N) : L(L), Y(Y), N(N){}
    ~EID(){}
    
    ////////////////////////////////////////////////////////////
 
    /**
     * nonlinear class NL provides container for intermediate steps
     */
    template<class NL>
    void 
    oneStep(double t, double h, NL nl){

	switch (scheme) {
      
	case  Cox_Matthews : 
	    nl(Y[0], N[0], t);	
	    Y[1] = c[1] * Y[0] + a[1][0] * N[0];
	    
	    nl(Y[1], N[1], t+h/2);	
	    Y[2] = c[1] * Y[0] + a[1][0] * N[1];
	    
	    nl(Y[2], N[2], t+h/2);	
	    Y[3] = c[1] * Y[0] + a[1][0] * (2*N[2]-N[0]);
	
	    nl(Y[3], N[3], t+h);	
	    Y[4] = c[3] * Y[0] + b[0] * N[0] + b[1] * (N[1] + N[2]) + b[3] * N[3];
	    
	    nl(Y[4], N[4], t+h);
	
	    err = (b[3]*(N[4] - N[3])).abs().maxCoeff() / Y[4].abs().maxCoeff();

	    break;
	
	case  Krogstad : 
	    nl(Y[0], N[0], t);	
	    Y[1] = c[1] * Y[0] + a[1][0] * N[0];

	    nl(Y[1], N[1], t+h/2);
	    Y[2] = c[1] * Y[0] + a[2][0] * N[0] + a[2][1] * N[1];

	    nl(Y[2], N[2], t+h/2);
	    Y[3] = c[3] * Y[0] + a[3][0] * N[0] + a[3][2] * N[2];
	
	    nl(Y[3], N[3], t+h);
	    Y[4] = c[3]*Y[0] + b[0]*N[0] + b[1]*(N[1] + N[2]) + b[3]*N[3];
	
	    nl(Y[4], N[4], t+h);
	
	    err = (b[3]*(N[4] - N[3])).abs().maxCoeff() / Y[4].abs().maxCoeff();

	    break;

	case Hochbruck_Ostermann :
	    break;

	case Luan_Ostermann :
	    break;

	default :
	    fprintf(stderr, "Please indicate the method !\n");
	
	}

    }

#if 0
    std::pair<VectorXd, Mat>
    intg(const double t0, const Ary &u0, const double tend, const double h0, 
	 const int skip_rate){
	
	double h = h0;
	calCoe(h);

	const int N = u0.size();    
	const int Nt = (int)round((tend-t0)/h);
	const int M = Nt /skip_rate + 1;

	Mat uu(N, M);
	VectorXd tt(M);
	uu.col(0) = u0;
	tt(0) = t0;
	initState(M);

	Ary unext;
	double du;
	double t = t0;
	Ary u = u0;
	int num = 1;
	bool doChange, doAccept;

	int i = 0;
	while(t < tend){
	    i++;

	    if ( t + h > tend){
		h = tend - t;
		calCoe(h);
		NCalCoe++;
	    }

	    oneStep(u, unext, du, t, h);
	    NCallF += 5;		
	    double s = nu * std::pow(rtol/du, 1.0/4);
	    double mu = adaptTs(doChange, doAccept, s);
	
	    if (doAccept){
		t += h;
		u = unext; 
		if ( (i+1) % skip_rate == 0 ) {
		    if (num >= tt.size() ) addSpace(tt, uu);
		    saveState(tt, uu, t, u, du, h, num++);
		}
	    }
	    else {
		NReject++;
	    }
	
	    if (doChange) {
		h *= mu;
		calCoe(h);
		NCalCoe++;
	    }
	}
    
	// duu = duu.head(num) has aliasing problem 
	hs.conservativeResize(num);
	duu.conservativeResize(num);
	return std::make_pair(tt.head(num), uu.leftCols(num));
    }
#endif
    
    template<class NL, class SS>
    void 
    intgC(NL nl, SS saveState, const double t0, const ArrayXcd &u0, const double tend, const double h, 
	  const int skip_rate){
	int ns = nstages[scheme];
	calCoe(h);
	const int Nt = (int)round((tend-t0)/h);

	double t = t0;
	Y[0] = u0;
	saveState(Y[0], 0);
	for(int i = 0; i < Nt; i++){
	    oneStep(t, h, nl);
	    NCallF += ns+1;
	    t += h;
	    Y[0] = Y[ns]; 
	    if((i+1)%skip_rate == 0 || i == Nt-1) saveState(Y[0], t);
	}
    }

    /**
     * @brief calculat the damping factor of time step
     *
     * @param[out] doChange    true if time step needs change
     * @param[out] doAccept    true if accept current time step
     * @param[in]  s           estimate damping factor
     * @return     mu          final dampling factor 
     */
    inline 
    double
    adaptTs(bool &doChange, bool &doAccept, const double s){
	double mu = 1;
	doChange = true;
	doAccept = true;

	if ( s > mumax) mu = mumax;
	else if (s > mue) mu = s;
	else if (s >= 1) {
	    mu = 1;
	    doChange = false;
	}
	else {
	    doAccept = false;
	    if (s > muc) mu = muc;
	    else if (s > mumin) mu = s;
	    else mu = mumin;
	}

	return mu;
    }
    
    inline
    virtual
    void calCoe(double h){}

    inline
    virtual 
    ArrayXXcd ZR(ArrayXd &z){}
    
};


#endif	/* EID_H */



