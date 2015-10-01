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

using Eigen::ArrayXXcd; 
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

    /* member variables */
    const int N;
    const double d;
    const double h;  
    ArrayXd K, L, E, E2, Q, f1, f2, f3;
    ArrayXcd G;
    ArrayXXcd jG;
  
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
    VectorXd 
    velocity(const Ref<const ArrayXd> &a0);
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
protected:
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

#endif	/* KSINT_H */
