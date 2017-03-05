/* to compile the class
 *  
 *  compile with Mutithreads FFT support:
 *  g++ cqcgl1d.cc  -shared -fpic -march=corei7 -O3 -msse2 -msse4 -lfftw3_threads -lfftw3 -lm -fopenmp -DTFFT  -I $XDAPPS/eigen/include/eigen3 -I../include -std=c++0x
 *
 *  complile with only one thread:
 *  g++ cqcgl1d.cc -shared -fpic -lfftw3 -lm -fopenmp  -march=corei7 -O3 -msse2 -msse4 -I/usr/include/eigen3 -I../../include -std=c++0x
 *
 *  */
#ifndef CQCGLGENERAL_H
#define CQCGLGENERAL_H

// #include <fftw3.h>
#include <complex>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "sparseRoutines.hpp"
#include "denseRoutines.hpp"
#include "iterMethod.hpp"
#include "ped.hpp"
#include "myfft.hpp"
using std::pair; using std::make_pair;
using Eigen::MatrixXd; using Eigen::VectorXd;
using Eigen::MatrixXcd; using Eigen::VectorXcd;
using Eigen::ArrayXXcd; using Eigen::ArrayXcd;
using Eigen::ArrayXXd; using Eigen::ArrayXd;
using Eigen::ConjugateGradient;
using Eigen::PartialPivLU;
using Eigen::Map; using Eigen::Ref;

//////////////////////////////////////////////////////////////////////
//                       class CQCGL1d                              //
//////////////////////////////////////////////////////////////////////
class CQCGL1d {
  
public:
    typedef std::complex<double> dcp;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::Triplet<double> Tri;
    
    const int N;		/* dimension of FFT */
    const double d;		/* system domain size */
    bool IsQintic = true;	//  False => cubic equation

    int DimTan;    		/* true tangent space dimension
				   dimTan > 0 => dimTan
				   dimTan = 0 => Ndim
				   dimTan < 0 => 0 
				*/
    int Ne;			/* effective number of modes */
    int Ndim;			/* dimension of state space */
    int Nplus, Nminus, Nalias;
    
    double Br, Bi, Gr, Gi, Dr, Di, Mu;
    double Omega = 0;		/* used for comoving frame */
    ArrayXd K, K2, QK;

    ArrayXcd L, E, E2, a21, a31, a32, a41, a43, b1, b2, b4;

    MyFFT::FFT F[5], JF[5];

    /* for time step adaptive ETDRK4 and Krogstad  */
    ////////////////////////////////////////////////////////////
    // time adaptive method related parameters
    double rtol = 1e-10;
    double nu = 0.9;		/* safe factor */
    double mumax = 2.5;		/* maximal time step increase factor */
    double mumin = 0.4;		/* minimal time step decrease factor */
    double mue = 1.25;		/* upper lazy threshold */
    double muc = 0.85;		/* lower lazy threshold */

    int NCalCoe = 0;	      /* times to evaluate coefficient */
    int NReject = 0;	      /* times that new state is rejected */
    int NCallF = 0;	      /* times to call velocity function f */
    int NSteps = 0;	      /* total number of integrations steps */
    VectorXd hs;	      /* time step sequnce */
    VectorXd lte;	      /* local relative error estimation */
    VectorXd Ts;	      /* time sequnence for adaptive method */
    
    int cellSize = 500;	/* size of cell when resize output container */
    int M = 64;			/* number of sample points */
    int R = 1;			/* radius for evaluating phi(z) */

    int Method = 1;
    ////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////
    //         constructor, destructor, copy assignment.      //
    ////////////////////////////////////////////////////////////

    // A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
    CQCGL1d(int N, double d,
	    double Mu, double Dr, double Di, double Br, double Bi, 
	    double Gr, double Gi, int dimTan);
    
    // A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
    CQCGL1d(int N, double d, 
	    double b, double c, double dr, double di, 
	    int dimTan);
    
    // iA_z + D A_{tt} + |A|^2 A + \nu |A|^4 A = i \delta A + i \beta A_{tt} + i |A|^2 A + i \mu |A|^4 A
    CQCGL1d(int N, double d,
	    double delta, double beta, double D, double epsilon,
	    double mu, double nu, int dimTan);
    ~CQCGL1d();
    CQCGL1d & operator=(const CQCGL1d &x);

    ////////////////////////////////////////////////////////////
    //                    member functions.                   //
    ////////////////////////////////////////////////////////////

    //============================================================    
    void CGLInit(int dimTan);
    void changeOmega(double w);
    void changeMu(double Mu);

    void calCoe(const double h);
    void oneStep(double &du, const bool onlyOrbit);
    ArrayXXcd ZR(ArrayXcd &z);
    double adaptTs(bool &doChange, bool &doAccept, const double s);

    ArrayXXd 
    constETD(const ArrayXXd a0, const double h, const int Nt, 
	     const int skip_rate, const bool onlyOrbit, bool reInitTan);
    ArrayXXd
    adaptETD(const ArrayXXd &a0, const double h0, const double tend, 
	     const int skip_rate, const bool onlyOrbit, bool reInitTan);
    ArrayXXd 
    intg(const ArrayXd &a0, const double h, const int Nt, const int skip_rate);
    std::pair<ArrayXXd, ArrayXXd>
    intgj(const ArrayXd &a0, const double h, const int Nt, const int skip_rate);
    ArrayXXd
    aintg(const ArrayXd &a0, const double h, const double tend, 
	  const int skip_rate);
    std::pair<ArrayXXd, ArrayXXd>
    aintgj(const ArrayXd &a0, const double h, const double tend, 
	   const int skip_rate);
    ArrayXXd
    intgv(const ArrayXd &a0, const ArrayXXd &v, const double h,
	  const int Nt);
    ArrayXXd 
    aintgv(const ArrayXXd &a0, const ArrayXXd &v, const double h,
	   const double tend);
    
    void dealias(const int k, const bool onlyOrbit);
    void NL(const int k, const bool onlyOrbit);
    ArrayXXd C2R(const ArrayXXcd &v);
    ArrayXXcd R2C(const ArrayXXd &v);
    ArrayXXd c2r(const ArrayXXcd &v);
    ArrayXXcd r2c(const ArrayXXd &v);
    
    //============================================================  

    ArrayXXcd Fourier2Config(const Ref<const ArrayXXd> &aa);
    ArrayXXd Config2Fourier(const Ref<const ArrayXXcd> &AA);
    ArrayXXd Fourier2ConfigMag(const Ref<const ArrayXXd> &aa);
    ArrayXXd calPhase(const Ref<const ArrayXXcd> &AA);
    ArrayXXd Fourier2Phase(const Ref<const ArrayXXd> &aa);
    VectorXd calQ(const Ref<const ArrayXXd> &aa);
    VectorXd calMoment(const Ref<const ArrayXXd> &aa, const int p = 1);
    
    ArrayXd velocity(const ArrayXd &a0);
    ArrayXd velocityReq(const ArrayXd &a0, const double th,
			const double phi);
    VectorXd velSlice(const Ref<const VectorXd> &aH);
    VectorXd velPhase(const Ref<const VectorXd> &aH);
    MatrixXd rk4(const VectorXd &a0, const double dt, const int nstp, const int nq);
    MatrixXd velJ(const MatrixXd &xj);
    std::pair<MatrixXd, MatrixXd>
    rk4j(const VectorXd &a0, const double dt, const int nstp, const int nq, const int nqr);
    
    ArrayXcd
    Lyap(const Ref<const ArrayXXd> &aa);
    ArrayXd
    LyapVel(const Ref<const ArrayXXd> &aa);
    MatrixXd stab(const ArrayXd &a0);
    MatrixXd stabReq(const ArrayXd &a0, double th, double phi);
    VectorXcd eReq(const ArrayXd &a0, double wth, double wphi);
    MatrixXcd vReq(const ArrayXd &a0, double wth, double wphi);
    std::pair<VectorXcd, MatrixXcd>
    evReq(const ArrayXd &a0, double wth, double wphi);

    ArrayXXd reflect(const Ref<const ArrayXXd> &aa);
    inline ArrayXd rcos2th(const ArrayXd &x, const ArrayXd &y);
    inline ArrayXd rsin2th(const ArrayXd &x, const ArrayXd &y);
    inline double rcos2thGrad(const double x, const double y);
    inline double rsin2thGrad(const double x, const double y);
    ArrayXXd reduceRef1(const Ref<const ArrayXXd> &aaHat);
    ArrayXXd reduceRef2(const Ref<const ArrayXXd> &step1);
    std::vector<int> refIndex3();
    ArrayXXd reduceRef3(const Ref<const ArrayXXd> &aa);
    ArrayXXd reduceReflection(const Ref<const ArrayXXd> &aaHat);
    MatrixXd refGrad1();
    MatrixXd refGrad2(const ArrayXd &x);
    MatrixXd refGrad3(const ArrayXd &x);
    MatrixXd refGradMat(const ArrayXd &x);
    MatrixXd reflectVe(const MatrixXd &veHat, const Ref<const ArrayXd> &xHat);
    MatrixXd reflectVeAll(const MatrixXd &veHat, const MatrixXd &aaHat,
			  const int trunc = 0);
    
    ArrayXXd transRotate(const Ref<const ArrayXXd> &aa, const double th);
    ArrayXXd transTangent(const Ref<const ArrayXXd> &aa);
    MatrixXd transGenerator();

    ArrayXXd phaseRotate(const Ref<const ArrayXXd> &aa, const double phi);
    ArrayXXd phaseTangent(const Ref<const ArrayXXd> &aa);
    MatrixXd phaseGenerator();

    ArrayXXd Rotate(const Ref<const ArrayXXd> &aa, const double th, const double phi);
    ArrayXXd rotateOrbit(const Ref<const ArrayXXd> &aa, const ArrayXd &th,
			 const ArrayXd &phi);
    std::tuple<ArrayXXd, ArrayXd, ArrayXd>
    orbit2slice(const Ref<const ArrayXXd> &aa, const int method);
    MatrixXd ve2slice(const ArrayXXd &ve, const Ref<const ArrayXd> &x, int flag);
    std::tuple<ArrayXXd, ArrayXd, ArrayXd>
    reduceAllSymmetries(const Ref<const ArrayXXd> &aa, int flag);
    MatrixXd reduceVe(const ArrayXXd &ve, const Ref<const ArrayXd> &x, int flag);
    
    VectorXd multiF(const ArrayXXd &x, const int nstp, const double th, const double phi);
    pair<SpMat, VectorXd>
    multishoot(const ArrayXXd &x, const int nstp, const double th,
	       const double phi, bool doesPrint = false);
    std::pair<MatrixXd, VectorXd>
    newtonReq(const ArrayXd &a0, const double th, const double phi);
    std::tuple<ArrayXd, double, double, double>
    findReq(const ArrayXd &a0, const double wth0, const double wphi0,
	    const int MaxN = 100, const double tol = 1e-14,
	    const bool doesUseMyCG = true, const bool doesPrint = true);
    std::vector<double>
    optThPhi(const ArrayXd &a0);
    
    std::tuple<ArrayXd, double, double>
    planeWave(int k, bool isPositve);
    VectorXcd planeWaveStabE(int k, bool isPositve);
    std::pair<VectorXcd, MatrixXcd>
    planeWaveStabEV(int k, bool isPositve);
};



#endif  /* CQCGLGENERAL_H */
