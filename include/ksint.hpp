/**
 * To compile this class, you need to have g++ >= 4.6, eigen >= 3.1
 * g++ ksint.cc -march=corei7 -O3 -msse4.2 -I/usr/include/eigen3
 * -lm -lfftw3 -std=c++0x
 *
 * Some notes about this class
 * 
 *  */
#ifndef KSINT_H
#define KSINT_H

#include <fftw3.h>
#include <complex>
#include <tuple>
#include <utility>
#include <cmath>
#include <memory>
#include <Eigen/Dense>
#include "myfft.hpp"
//#include "iterMethod.hpp"

using Eigen::ArrayXXcd; 
using Eigen::MatrixXcd; 
using Eigen::ArrayXXd;
using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::MatrixXd; using Eigen::VectorXd;
using Eigen::VectorXcd; 
using Eigen::Matrix2d;
using Eigen::VectorXi;
using Eigen::Map; using Eigen::Ref;

/*============================================================
 *                       Class : KS integrator
 *============================================================*/
class KS{
  
public:
    typedef std::complex<double> dcp;

    //////////////////////////////////////////////////////////////////////
    /* member variables */
    const int N;
    const double d;

    ArrayXd K, L, E, E2, a21, a31, a32, a41, a43, b1, b2, b4;
    ArrayXcd G;
    ArrayXXcd jG;

    MyFFT::RFFT F[5], JF[5];

    /* for time step adaptive ETDRK4 and Krogstad  */
    ////////////////////////////////////////////////////////////
    // time adaptive method related parameters
    double rtol = 1e-8;
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
    int M = 32;			/* number of sample points */
    int R = 1;			/* radius for evaluating phi(z) */

    int Method = 1;
    ////////////////////////////////////////////////////////////
    

    //////////////////////////////////////////////////////////////////////
    /* constructor, destructor, copy assignment */
    KS(int N, double d);
    KS(const KS &x);
    KS & operator=(const KS &x);
    ~KS();
  
    /* member functions */
    /* ------------------------------------------------------------ */
    /* related to integration */
    void calCoe(const double h);
    ArrayXXcd ZR(ArrayXd &z);
    void ksInit();
    void oneStep(double &du, const bool onlyOrbit);
    double adaptTs(bool &doChange, bool &doAccept, const double s);
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
    constETD(const ArrayXXd a0, const double h, const int Nt, 
	     const int skip_rate, const bool onlyOrbit, bool reInitTan);
    ArrayXXd
    adaptETD(const ArrayXXd &a0, const double h0, const double tend, 
	     const int skip_rate, const bool onlyOrbit, bool reInitTan);
    void NL(const int k, const bool onlyOrbit);

    /* ------------------------------------------------------------ */
    std::pair<ArrayXXd, ArrayXXd>    
    intgjMulti(const MatrixXd aa0, size_t nstp, size_t np, size_t nqr);
    std::tuple<MatrixXd, MatrixXd, VectorXd>
    calReqJJF(const Ref<const VectorXd> &x);
    std::tuple<VectorXd, double, double>
    findReq(const Ref<const VectorXd> &x, const double tol, 
	    const int maxit, const int innerMaxit);
    std::tuple<MatrixXd, MatrixXd, VectorXd>
    calEqJJF(const Ref<const VectorXd> &x);
    std::pair<VectorXd, double>
    findEq(const Ref<const VectorXd> &x, const double tol,
	   const int maxit, const int innerMaxit);


    VectorXd 
    velocity(const Ref<const ArrayXd> &a0);
    VectorXd
    velg(const Ref<const VectorXd> &a0, const double c);
    MatrixXd stab(const Ref<const ArrayXd> &a0);
    MatrixXd stabReq(const Ref<const VectorXd> &a0, const double theta);
    std::pair<VectorXcd, Eigen::MatrixXcd>
    stabEig(const Ref<const VectorXd> &a0);
    std::pair<VectorXcd, Eigen::MatrixXcd>
    stabReqEig(const Ref<const VectorXd> &a0, const double theta);


    ArrayXXd 
    C2R(const ArrayXXcd &v);
    ArrayXXcd 
    R2C(const ArrayXXd &v);
    double pump(const ArrayXcd &vc);
    double disspation(const ArrayXcd &vc);
    std::tuple<ArrayXXd, VectorXd, VectorXd> 
    intgDP(const ArrayXd &a0, size_t nstp, size_t np);
    ArrayXXd 
    Reflection(const Ref<const ArrayXXd> &aa);
    ArrayXXd 
    half2whole(const Ref<const ArrayXXd> &aa);
    ArrayXXd 
    Rotation(const Ref<const ArrayXXd> &aa, const double th);
    MatrixXd 
    gTangent(const MatrixXd &x);
    MatrixXd 
    gGenerator();
    std::pair<MatrixXd, VectorXd> 
    orbitToSlice(const Ref<const MatrixXd> &aa);
    MatrixXd 
    veToSlice(const MatrixXd &ve, const Ref<const VectorXd> &x);
    MatrixXd 
    veToSliceAll(const MatrixXd &eigVecs, const MatrixXd &aa,
		 const int trunc = 0);
    std::pair<ArrayXXd, ArrayXXd>
    orbitAndFvWhole(const ArrayXd &a0, const ArrayXXd &ve,
		    const double h,
		    const size_t nstp, const std::string ppType
		    );
    MatrixXd veTrunc(const MatrixXd ve, const int pos, const int trunc = 0);
    std::pair<ArrayXXd, ArrayXXd>
    orbitAndFvWholeSlice(const ArrayXd &a0, const ArrayXXd &ve,
			 const double h,
			 const size_t nstp, const std::string ppType,
			 const int pos
			 );
    std::vector<int> reflectIndex();
    ArrayXXd reduceReflection(const Ref<const ArrayXXd> &aaHat);
    MatrixXd GammaMat(const Ref<const ArrayXd> &xHat);
    MatrixXd reflectVe(const MatrixXd &ve, const Ref<const ArrayXd> &xHat);
    MatrixXd reflectVeAll(const MatrixXd &veHat, const MatrixXd &aaHat,
			  const int trunc = 0);

    MatrixXd calMag(const Ref<const MatrixXd> &aa);
    std::pair<MatrixXd, MatrixXd>
    toPole(const Ref<const MatrixXd> &aa);
    MatrixXcd
    a2f(const Ref<const MatrixXd> &aa);
    MatrixXd
    f2a(const Ref<const MatrixXcd> &f);

    std::pair<MatrixXd, VectorXd>
    redSO2(const Ref<const MatrixXd> &aa, const int p, const bool toY);
    std::pair<MatrixXd, VectorXi>
    fundDomain(const Ref<const MatrixXd> &aa, const int pSlice, const int pFund);
    std::tuple<MatrixXd, VectorXi, VectorXd>
    redO2f(const Ref<const MatrixXd> &aa, const int pSlice, const int pFund);
    MatrixXd redR1(const Ref<const MatrixXd> &aa);
    MatrixXd redR2(const Ref<const MatrixXd> &cc);
    MatrixXd redRef(const Ref<const MatrixXd> &aa);
    std::pair<MatrixXd, VectorXd>
    redO2(const Ref<const MatrixXd> &aa, const int p, const bool toY);
    MatrixXd Gmat1(const Ref<const VectorXd> &x);
    MatrixXd Gmat2(const Ref<const VectorXd> &x);
    MatrixXd
    redV(const Ref<const MatrixXd> &v, const Ref<const VectorXd> &a,
	 const int p, const bool toY);
    MatrixXd redV2(const Ref<const MatrixXd> &v, const Ref<const VectorXd> &a);

};

/*============================================================
 *                       Class : Calculate KS Req LM 
 *============================================================*/
template<class Mat>
struct KSReqJJF {
    KS &ks;
    KSReqJJF(KS &ks_) : ks(ks_){}
    
    std::tuple<MatrixXd, MatrixXd, VectorXd>
    operator()(const VectorXd &x) {
	return ks.calReqJJF(x);
    }	
};

/*============================================================
 *                       Class : Calculate KS Eq LM 
 *============================================================*/
template<class Mat>
struct KSEqJJF {
    KS &ks;
    KSEqJJF(KS &ks_) : ks(ks_){}
    
    std::tuple<MatrixXd, MatrixXd, VectorXd>
    operator()(const VectorXd &x) {
	return ks.calEqJJF(x);
    }	
};


#endif	/* KSINT_H */
