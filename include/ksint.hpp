/** To compile this class, you need to have g++ >= 4.6, eigen >= 3.1
 * g++ ksint.cc -march=corei7 -O3 -msse4.2 -I/usr/include/eigen3
 * -lm -lfftw3 -std=c++0x
 *  */
#ifndef KSINT_H
#define KSINT_H

#include <fftw3.h>
#include <complex>
#include <tuple>
#include <utility>
#include <Eigen/Dense>
#include "ETDRK4.hpp"
//#include "iterMethod.hpp"

using Eigen::ArrayXXcd; 
using Eigen::MatrixXcd; 
using Eigen::ArrayXXd;
using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::MatrixXd; using Eigen::VectorXd;
using Eigen::VectorXcd;
using Eigen::Matrix2d;
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
    const double h;  
    ArrayXd K, L, E, E2, Q, f1, f2, f3;
    ArrayXcd G;
    ArrayXXcd jG;
    
    //////////////////////////////////////////////////////////////////////
    /* constructor, destructor, copy assignment */
    KS(int N = 32, double h = 0.25, double d = 22);
    explicit KS(const KS &x);
    KS & operator=(const KS &x);
    ~KS();
  
    /* member functions */
    ArrayXXd 
    intg(const ArrayXd &a0, size_t nstp, size_t np = 1);
    std::pair<ArrayXXd, ArrayXXd>
    intgj(const ArrayXd &a0, size_t nstp, size_t np = 1, size_t nqr = 1);
    std::pair<ArrayXXd, ArrayXXd>

    
    intgjMulti(const MatrixXd aa0, size_t nstp, size_t np = 1, size_t nqr = 1);
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
		    const size_t nstp, const std::string ppType
		    );
    MatrixXd veTrunc(const MatrixXd ve, const int pos, const int trunc = 0);
    std::pair<ArrayXXd, ArrayXXd>
    orbitAndFvWholeSlice(const ArrayXd &a0, const ArrayXXd &ve,
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
    redSO2(const Ref<const MatrixXd> &aa);
    MatrixXd redR1(const Ref<const MatrixXd> &aa);
    MatrixXd redR2(const Ref<const MatrixXd> &cc);
    MatrixXd redRef(const Ref<const MatrixXd> &aa);
    std::pair<MatrixXd, VectorXd>
    redO2(const Ref<const MatrixXd> &aa);
    MatrixXd Gmat1(const Ref<const VectorXd> &x);
    MatrixXd Gmat2(const Ref<const VectorXd> &x);
    std::pair<VectorXd, MatrixXd>
    redV(const Ref<const MatrixXd> &v, const Ref<const VectorXd> &a);
    MatrixXd redV2(const Ref<const MatrixXd> &v, const Ref<const VectorXd> &a);

    std::pair<VectorXd, ArrayXXd>
    etd(const ArrayXd &a0, const double tend, const double h, const int skip_rate, int method);
    
    //protected:
    enum { M = 16 }; // number used to approximate the complex integral.
  
    struct KSfft{ // nested class for fft/ifft.      
	/* member variables */
	/* 3 different stage of ETDRK4:
	 * v --> ifft(v) --> Nv : c1 --> r2 --> c2
	 * */
	fftw_plan p, rp;
	double *r2;
	fftw_complex *c1, *c3;
	Map<ArrayXXd> vr2;
	Map<ArrayXXcd> vc1, vc3;
  
	/* constructor, destructor */
	KSfft() : vc1(NULL, 0, 0), vc3(NULL, 0, 0), vr2(NULL, 0, 0){}
    }; 

    KSfft Fv, Fa, Fb, Fc; 
    KSfft jFv, jFa, jFb, jFc;
  
    void ksInit();
    virtual void NL(KSfft &f);
    virtual void jNL(KSfft &f);
    void initFFT(KSfft &f, int M);
    void freeFFT(KSfft &f);
    void fft(KSfft &f);
    void ifft(KSfft &f);
    

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


/*============================================================
 *                       Class : Nonliear part of KS 
 *============================================================*/
template<class Ary>
struct KSNL {
    KS &ks;
    KSNL(KS &ks_) : ks(ks_) {}
    ArrayXcd operator()(double t, const ArrayXcd &x){
	ks.Fv.vc1 = x;
	ks.ifft(ks.Fv);
	ks.Fv.vr2 = ks.Fv.vr2 * ks.Fv.vr2;
	ks.fft(ks.Fv);
       
	return ks.Fv.vc3 * ks.G; 
    }
};

#endif	/* KSINT_H */
