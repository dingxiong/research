/* to compile the class
 *  
 *  compile with Mutithreads FFT support:
 *  g++ cqcgl1d.cc  -shared -fpic -march=corei7 -O3 -msse2 -msse4 -lfftw3_threads -lfftw3 -lm -fopenmp -DTFFT  -I $XDAPPS/eigen/include/eigen3 -I../include -std=c++0x
 *
 *  complile with only one thread:
 *  g++ cqcgl1d.cc -shared -fpic -lfftw3 -lm -fopenmp  -march=corei7 -O3 -msse2 -msse4 -I/usr/include/eigen3 -I../../include -std=c++0x
 *
 *  */
#ifndef CQCGLGENERAL2D_H
#define CQCGLGENERAL2D_H

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
//                       class CQCGLgeneral                              //
//////////////////////////////////////////////////////////////////////
class CQCGLgeneral2d {
  
public:
    typedef std::complex<double> dcp;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::Triplet<double> Tri;
    
    const int N, M;		/* dimension of FFT */
    const double dx, dy;	/* system domain size */
    
    int Ne, Me;			/* effective number of modes */
    int Nplus, Nminus, Nalias;
    int Mplus, Mminus, Malias;
    
    double Br, Bi, Gr, Gi, Dr, Di, Mu;
    double Omega = 0;		/* used for comoving frame */
    ArrayXd Kx, Kx2, Ky, Ky2, QKx, QKy;

    ArrayXXcd L, E, E2, a21, a31, a32, a41, a43, b1, b2, b4;

    MyFFT::FFT2d F[5], JF[5];
    
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
    int MC = 64;	/* number of sample points */
    int R = 1;		/* radius for evaluating phi(z) */

    int Method = 1;
    ////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////
    //         constructor, destructor, copy assignment.      //
    ////////////////////////////////////////////////////////////
    CQCGLgeneral2d(int N, int M, double dx, double dy,
		   double Mu, double Dr, double Di,
		   double Br, double Bi, double Gr,
		   double Gi,  
		   int threadNum);
    ~CQCGLgeneral2d();
    CQCGLgeneral2d & operator=(const CQCGLgeneral2d &x);
    
    ////////////////////////////////////////////////////////////
    //                    member functions.                   //
    ////////////////////////////////////////////////////////////

    //============================================================    
    inline int calNe(const double N);
    void CGLInit();
    void changeOmega(double w);
    void calCoe(const double h);
    void oneStep(double &du, const bool onlyOrbit);
    ArrayXXcd ZR(const Ref<const ArrayXcd> &z);
    double adaptTs(bool &doChange, bool &doAccept, const double s);

    ArrayXXcd 
    constETD(const ArrayXXcd &a0, const ArrayXXcd &v0, 
	     const double h, const int Nt, 
	     const int skip_rate, const bool onlyOrbit);
    ArrayXXcd
    adaptETD(const ArrayXXcd &a0, const ArrayXXcd &v0, 
	     const double h0, const double tend, 
	     const int skip_rate, const bool onlyOrbit);
    ArrayXXcd 
    intg(const ArrayXXcd &a0, const double h, const int Nt, const int skip_rate);
    ArrayXXcd
    aintg(const ArrayXXcd &a0, const double h, const double tend, 
	  const int skip_rate);
    ArrayXXcd
    intgv(const ArrayXXcd &a0, const ArrayXXcd &v0, const double h,
	  const int Nt, const int skip_rate);
    ArrayXXcd 
    aintgv(const ArrayXXcd &a0, const ArrayXXcd &v0, const double h,
	   const double tend, const int skip_rate);
    
    void dealias(const int k, const bool onlyOrbit);
    void NL(const int k, const bool onlyOrbit);
    ArrayXXcd unpad(const ArrayXXcd &v);
    ArrayXXcd pad(const ArrayXXcd &v);
    ArrayXXd c2r(const ArrayXXcd &v);
    ArrayXXcd r2c(const ArrayXXd &v);
    
    //============================================================  

    ArrayXXcd Fourier2Config(const Ref<const ArrayXXcd> &aa);
    ArrayXXcd Config2Fourier(const Ref<const ArrayXXcd> &AA);
    
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
    orbit2sliceWrap(const Ref<const ArrayXXd> &aa);
    std::tuple<ArrayXXd, ArrayXd, ArrayXd>
    orbit2slice(const Ref<const ArrayXXd> &aa);
    ArrayXXd orbit2sliceSimple(const Ref<const ArrayXXd> &aa);
    MatrixXd ve2slice(const ArrayXXd &ve, const Ref<const ArrayXd> &x);
    std::tuple<ArrayXXd, ArrayXd, ArrayXd>
    reduceAllSymmetries(const Ref<const ArrayXXd> &aa);
    MatrixXd reduceVe(const ArrayXXd &ve, const Ref<const ArrayXd> &x);
    
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

};



#endif  /* CQCGLGENERAL2D_H */
