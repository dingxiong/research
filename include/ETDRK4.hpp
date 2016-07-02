#ifndef ETDRK4_H
#define ETDRK4_H

#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "denseRoutines.hpp"

using std::cout;
using std::endl;

using Eigen::ArrayXXcd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::ArrayXcd;

// using namespace denseRoutines;

//////////////////////////////////////////////////////////////////////////////////
//                        Real ETDRK4 class
//////////////////////////////////////////////////////////////////////////////////

/**
 * @brief ETDRK4 integration class
 *
 * Ary : type of the state varible
 * Mat : type of the collection of state variable
 * NL   : the nonlinear function type
 */
template<class Ary, class Mat, template<class> class NL>
class ETDRK4 {
    
public:

    typedef std::complex<double> dcp;

    //////////////////////////////////////////////////////////////////////

    enum Scheme {
	Cox_Matthews,
	Krogstad,
	Hochbruck_Ostermann,
	Luan_Ostermann
    };

    Scheme scheme = Cox_Matthews;	

    ArrayXd Ta[9][9], Tb[9], Tc[9];
    ArrayXd L;
    
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
    VectorXd hs;	       /* time step sequnce */
    VectorXd duu;	       /* local relative error estimation */

    int cellSize = 500;	/* size of cell when resize output container */
    
    
    ////////////////////////////////////////////////////////////
    // constructor and desctructor
    ETDRK4(ArrayXd linear, NL<Ary> nonlinear) : L(linear), nl(nonlinear){}
    ~ETDRK4(){}
    
    ////////////////////////////////////////////////////////////
    void 
    oneStep(Ary &u, Ary &unext, double &du, double t, double h);

    std::pair<VectorXd, Mat>
    intg(const double t0, const Ary &u0, const double tend, const double h0, 
	 const int skip_rate);

    std::pair<VectorXd, Mat>
    intgC(const double t0, const Ary &u0, const double tend, const double h, 
	  const int skip_rate);

    double
    adaptTs(bool &doChange, bool &doAccept, const double s);
  
    void 
    initState(const int M);

    void
    addSpace(VectorXd &tt, Mat &uu);

    void    
    saveState(VectorXd &tt, Mat &uu, const double t, const Ary &u,
	      const double du, const double h, const int num);
    
    virtual void
    calCoe(double h);

    virtual ArrayXXcd
    ZR(ArrayXd &z);
    
};


template<class Ary, class Mat, template<class> class NL>
std::pair<VectorXd, Mat>
ETDRK4<Ary, Mat, NL>::
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

template<class Ary, class Mat, template<class> class NL>
std::pair<VectorXd, Mat>
ETDRK4<Ary, Mat, NL>::
intgC(const double t0, const Ary &u0, const double tend, const double h, 
      const int skip_rate){
    
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
    for(int i = 0; i < Nt; i++){
	oneStep(u, unext, du, t, h);
	NCallF += 5;
	t += h;
	u = unext;
	if ( (i+1)%skip_rate == 0 )saveState(tt, uu, t, u, du, h, num++);
    }

    return std::make_pair(tt, uu);
}



template<class Ary, class Mat, template<class> class NL>
void
ETDRK4<Ary, Mat, NL>::
addSpace(VectorXd &tt, Mat &uu){
    int m = tt.size();
    tt.conservativeResize(m+cellSize);
    uu.conservativeResize(Eigen::NoChange, m+cellSize); // rows not change, just extend cols
    hs.conservativeResize(m+cellSize);
    duu.conservativeResize(m+cellSize);
}

/**
 * @brief calculat the damping factor of time step
 *
 * @param[out] doChange    true if time step needs change
 * @param[out] doAccept    true if accept current time step
 * @param[in]  s           estimate damping factor
 * @return     mu          final dampling factor 
 */
template<class Ary, class Mat, template<class> class NL>
double
ETDRK4< Ary, Mat, NL>::
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


template<class Ary, class Mat, template<class> class NL>
void 
ETDRK4<Ary, Mat, NL>::
initState(const int M){
    NCalCoe = 0;
    NReject = 0;
    NCallF = 0;
    
    hs.resize(M);
    duu.resize(M);
    //hs = VectorXd(M);
    //duu = VectorXd(M);
}


template<class Ary, class Mat, template<class> class NL>
void 
ETDRK4<Ary, Mat, NL>::
saveState(VectorXd &tt, Mat &uu, const double t, const Ary &u, 
	  const double du, const double h, const int num){
    uu.col(num) = u;
    tt(num) = t;
    hs(num) = h;
    duu(num) = du;
}

template<class Ary, class Mat, template<class> class NL>
void 
ETDRK4<Ary, Mat, NL>::
oneStep(Ary &u, Ary &unext, double &du, double t, double h){
    
    if (1 == Method) {
	Ary N1 = nl(t, u);
    
	Ary U2 = E2*u + a21*N1;
	Ary N2 = nl(t+h/2, U2);

	Ary U3 = E2*u + a21*N2;
	Ary N3 = nl(t+h/2, U3);

	Ary U4 = E2*U2 + a21*(2*N3 - N1);
	Ary N4 = nl(t+h, U4);
    
	unext = E*u + b1*N1 + b2*(N2+N3) + b4*N4;
	Ary &U5 = unext;
	Ary N5 = nl(t+h, U5);

	du = (b4*(N5-N4)).matrix().norm() / U5.matrix().norm();

    }
    else {
	Ary N1 = nl(t, u);
    
	Ary U2 = E2*u + a21*N1;
	Ary N2 = nl(t+h/2, U2);

	Ary U3 = E2*u + a31*N1 + a32*N2;
	Ary N3 = nl(t+h/2, U3);

	Ary U4 = E*u + a41*N1 + a43*N3; 
	Ary N4 = nl(t+h, U4);
    
	unext = E*u + b1*N1 + b2*(N2+N3) + b4*N4;
	Ary &U5 = unext;
	Ary N5 = nl(t+h, U5);

	du = (b4*(N5-N4)).matrix().norm() / U5.matrix().norm();

    }
}

/**
 * nonlinear class NL provides container for intermediate steps
 *
 * U1, U2, U3, U4, U5
 * N1, N2, N3, N4, N5
 */
template<class Ary, class Mat, template<class> class NL>
void 
ETDRK4<Ary, Mat, NL>::
oneStep(double &du, double t, double h){
    
    if (1 == Method) {
	nl.N1 = nl(t, 1);
    
	nl.U2 = E2*nl.U1 + a21*nl.N1;
	nl.N2 = nl(t+h/2, 2);

	nl.U3 = E2*nl.U1 + a21*nl.N2;
	nl.N3 = nl(t+h/2, 3);

	nl.U4 = E2*nl.U2 + a21*(2*nl.N3 - nl.N1);
	nl.N4 = nl(t+h, 4);
    
	nl.U5 = E*nl.U1 + b1*nl.N1 + b2*(nl.N2+nl.N3) + b4*nl.N4;
	nl.N5 = nl(t+h, 5);

	du = (b4*(nl.N5-nl.N4)).matrix().norm() / nl.U5.matrix().norm();
    }

}

/**
 * @brief Calculate the coefficients of ETDRK4.
 */
template<class Ary, class Mat, template<class> class NL>
void
ETDRK4<Ary, Mat, NL>::
calCoe(double h){

    ArrayXd hL = h*L;
   
    switch (scheme) 
	{
    
	case Cox_Matthews : {
	    Tc[1] = (hL/2).exp();
	    Tc[3] = hL.exp();
	    a[1][0] = 
	}

	case Krogstad : {

	}
	case Hochbruck_Ostermann : {

	}
	case Luan_Ostermann : {

	}
	    
	default : {
	    fprintf(stderr, "Please indicate the method !\n");
	}
	
    }
    
    ArrayXXcd LR = ZR(hL);
    
    E = hL.exp();
    E2 = (hL/2).exp();

    ArrayXXcd LR2 = LR.square();
    ArrayXXcd LR3 = LR.cube();
    ArrayXXcd LRe = LR.exp();
    ArrayXXcd LReh = (LR/2).exp();
    
    a21 = h * ( (LReh - 1)/LR ).rowwise().mean().real(); 
    b1 = h * ( (-4.0 - LR + LRe*(4.0 - 3.0 * LR + LR2)) / LR3 ).rowwise().mean().real();
    b2 = h * 2 * ( (2.0 + LR + LRe*(-2.0 + LR)) / LR3 ).rowwise().mean().real();
    b4 = h * ( (-4.0 - 3.0*LR -LR2 + LRe*(4.0 - LR) ) / LR3 ).rowwise().mean().real();

    if (Method == 2) {
	a31 = h * ( (LReh*(LR - 4) + LR + 4) / LR2 ).rowwise().mean().real();
	a32 = h * 2 * ( (2*LReh - LR - 2) / LR2 ).rowwise().mean().real();
	a41 = h * ( (LRe*(LR-2) + LR + 2) / LR2 ).rowwise().mean().real();
	a43 = h * 2 * ( (LRe - LR - 1)  / LR2 ).rowwise().mean().real();
    }
 
}


/**
 * @brief calcuate the matrix to do averge of phi(z). 
 */
template<class Ary, class Mat, template<class> class NL>
ArrayXXcd
ETDRK4<Ary, Mat, NL>::
ZR(ArrayXd &z){
    
    int M1 = z.size();
    ArrayXd K = ArrayXd::LinSpaced(M, 1, M); // 1,2,3,...,M 

    ArrayXXcd r = R * ((K-0.5)/M * dcp(0, M_PI)).exp().transpose();

    return z.template cast<std::complex<double>>().replicate(1, M) + r.replicate(M1, 1);
}


#endif	/* ETDRK4_H */
