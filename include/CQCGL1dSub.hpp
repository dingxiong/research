#ifndef CQCGL1DSUB_H
#define CQCGL1DSUB_H

#include <complex>
#include <utility>
#include <algorithm>
#include <vector>
#include <unsupported/Eigen/FFT>
#include "denseRoutines.hpp"
#include "EIDc.hpp"

//////////////////////////////////////////////////////////////////////
//                       class CQCGL1dSub                           //
//////////////////////////////////////////////////////////////////////
class CQCGL1dSub {
  
public:
    typedef std::complex<double> dcp;
    
    const int N;		/* dimension of FFT */
    const double d;		/* system domain size */
    
    const int Ne;		// effective number of modes
    const int Ndim;		// dimension of state space
    const int DimTan;		// the dimension of tangent space
    
    bool IsQintic = true;	//  False => cubic equation
    
    double Mu, Dr, Di, Br, Bi, Gr, Gi;
    double Omega = 0;		/* used for comoving frame */
    ArrayXd K, QK;

    ArrayXcd L;

    FFT<double> fft;

    VectorXd hs;	      /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    int cellSize = 500;	/* size of cell when resize output container */

    struct NL {
	CQCGL1dSub *cgl;
	int N, Ne;
	dcp B, G;
	ArrayXXcd modes, field;	// Fourier modes and physical field
	
	NL();
	NL(CQCGL1dSub *cgl, int cols);
	~NL();
	void operator()(ArrayXXcd &x, ArrayXXcd &dxdt, double t);
    };
    
    NL nl, nl2;
    
    ArrayXXcd Yv[10], Nv[10], Yv2[10], Nv2[10];
    EIDc eidc, eidc2;
    
    ////////////////////////////////////////////////////////////
    //         constructor, destructor, copy assignment.      //
    ////////////////////////////////////////////////////////////

    // A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
    CQCGL1dSub(int N, double d,
	       double Mu, double Dr, double Di, double Br, double Bi, 
	       double Gr, double Gi, int dimTan);
    
    // A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
    CQCGL1dSub(int N, double d, 
	       double b, double c, double dr, double di, 
	       int dimTan);
    
    // iA_z + D A_{tt} + |A|^2 A + \nu |A|^4 A = i \delta A + i \beta A_{tt} + i |A|^2 A + i \mu |A|^4 A
    CQCGL1dSub(int N, double d,
	       double delta, double beta, double D, double epsilon,
	       double mu, double nu, int dimTan);
    ~CQCGL1dSub();
    CQCGL1dSub & operator=(const CQCGL1dSub &x);

    ////////////////////////////////////////////////////////////
    //                    member functions.                   //
    ////////////////////////////////////////////////////////////

    //============================================================    
    void setScheme(std::string x);
    void changeOmega(double w);

    ArrayXXd 
    intgC(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    std::pair<ArrayXXd, ArrayXXd>
    intgjC(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    ArrayXXd
    intgvC(const ArrayXd &a0, const ArrayXXd &v, const double h, const double tend);
    ArrayXXd
    intg(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    std::pair<ArrayXXd, ArrayXXd>
    intgj(const ArrayXd &a0, const double h, const double tend, const int skip_rate);
    ArrayXXd 
    intgv(const ArrayXXd &a0, const ArrayXXd &v, const double h, const double tend);
    
    ArrayXXd C2R(const ArrayXXcd &v);
    ArrayXXcd R2C(const ArrayXXd &v);
    
    //============================================================  

    ArrayXXcd Fourier2Config(const Ref<const ArrayXXd> &aa);
    ArrayXXd Config2Fourier(const Ref<const ArrayXXcd> &AA);
    ArrayXXd calPhase(const Ref<const ArrayXXcd> &AA);
    VectorXd calQ(const Ref<const ArrayXXd> &aa);
    VectorXd calMoment(const Ref<const ArrayXXd> &aa, const int p = 1);
    
    ArrayXd velocity(const ArrayXd &a0);
    ArrayXd velocityReq(const ArrayXd &a0, const double phi);
    MatrixXd stab(const ArrayXd &a0);
    MatrixXd stabReq(const ArrayXd &a0, double phi);
    VectorXcd eReq(const ArrayXd &a0, double wphi);
    MatrixXcd vReq(const ArrayXd &a0, double wphi);
    std::pair<VectorXcd, MatrixXcd>
    evReq(const ArrayXd &a0, double wphi);

    ArrayXXd phaseRotate(const Ref<const ArrayXXd> &aa, const double phi);
    ArrayXXd phaseTangent(const Ref<const ArrayXXd> &aa);
    MatrixXd phaseGenerator();
};



#endif  /* CQCGL1DSUB_H */
