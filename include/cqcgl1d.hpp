/* to compile the class
 *  
 *  compile with Mutithreads FFT support:
 *  g++ cqcgl1d.cc  -shared -fpic -march=corei7 -O3 -msse2 -msse4 -lfftw3_threads -lfftw3 -lm -fopenmp -DTFFT  -I $XDAPPS/eigen/include/eigen3 -I../include -std=c++0x
 *
 *  complile with only one thread:
 *  g++ cqcgl1d.cc -shared -fpic -o libcqcgl1d.so -lfftw3 -lm -fopenmp  -march=corei7 -O3 -msse2 -msse4 -I/usr/include/eigen3 -I../../include -std=c++0x
 *
 *  */
#ifndef CQCGL1D_H
#define CQCGL1D_H

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
//                       class Cqcgl1d                              //
//////////////////////////////////////////////////////////////////////
class Cqcgl1d {
  
public:
    typedef std::complex<double> dcp;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::Triplet<double> Tri;
    
    const int N;		/* dimension of FFT */
    const double d;		/* system domain size */
    const bool enableJacv;	/* wether turn on tangent dynamics */
    const int Njacv;		/* number of tangent dimension. 0 => full */
    
    double h;			/* integration time step */
    int trueNjacv;    		/* true tangent space dimension */
    int Ne;			/* effective number of modes */
    int Ndim;			/* dimension of state space */
    int aliasStart, aliasEnd;	/* start and end index of
				   dealias part */
    int Nplus, Nminus, Nalias;
    
    double Br, Bi, Gr, Gi, Dr, Di, Mu; 
    ArrayXd K, Kindex, KindexUnpad;    
    ArrayXcd L, E, E2, Q, f1, f2, f3;

    // parameters only work for the second constructor
    double b = 0;		
    double c = 0;
    double dr = 0;
    double di = 0;

    ////////////////////////////////////////////////////////////
    //         constructor, destructor, copy assignment.      //
    ////////////////////////////////////////////////////////////
    Cqcgl1d(int N = 1024, double d = 30, double h = 0.01,
	    bool enableJacv = false, int Njacv = 1,
	    double Mu = -0.1, double Br = 1.0, double Bi = 0.8,
	    double Dr = 0.125, double Di = 0.5, double Gr = -0.1,
	    double Gi = -0.6, int threadNum = 4);
    Cqcgl1d(int N, double d, double h,
	    bool enableJacv, int Njacv,
	    double b, double c,
	    double dr, double di,
	    int threadNum);
    ~Cqcgl1d();
    Cqcgl1d & operator=(const Cqcgl1d &x);

    ////////////////////////////////////////////////////////////
    //                    member functions.                   //
    ////////////////////////////////////////////////////////////
    
    void changeh(const double hnew);
    ArrayXXd
    intg(const ArrayXd &a0, const size_t nstp, const size_t np = 1);
    pair<ArrayXXd, ArrayXXd>
    intgj(const ArrayXd &a0, const size_t nstp, const size_t np = 1,
	  const size_t nqr = 1);
    ArrayXXd
    intgv(const ArrayXd &a0, const ArrayXXd &v,
	  const size_t nstp);

    ArrayXXd pad(const Ref<const ArrayXXd> &aa);
    ArrayXXd generalPadding(const Ref<const ArrayXXd> &aa);
    ArrayXXcd padcp(const Ref<const ArrayXXcd> &x);
    ArrayXXd unpad(const Ref<const ArrayXXd> &paa);
    ArrayXXcd initJ();
    ArrayXXd C2R(const ArrayXXcd &v);
    ArrayXXcd R2C(const ArrayXXd &v);
    ArrayXXd Fourier2Config(const Ref<const ArrayXXd> &aa);
    ArrayXXd Config2Fourier(const Ref<const ArrayXXd> &AA);
    ArrayXXd calMag(const Ref<const ArrayXXd> &AA);
    ArrayXXd Fourier2ConfigMag(const Ref<const ArrayXXd> &aa);
    ArrayXXd calPhase(const Ref<const ArrayXXd> &AA);
    ArrayXXd Fourier2Phase(const Ref<const ArrayXXd> &aa);
    
    ArrayXd velocity(const ArrayXd &a0);
    ArrayXd velocityReq(const ArrayXd &a0, const double th,
			const double phi);
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
    std::tuple<ArrayXXd, ArrayXd, ArrayXd>
    reduceIntg(const ArrayXd &a0, const size_t nstp, const size_t np = 1);
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

    std::tuple<ArrayXd, double, double>
    planeWave(int k, bool isPositve);
    void planeWave(ArrayXd &a0, double &a, double &w, 
		   int k, bool isPositve);
    VectorXcd planeWaveStabE(int k, bool isPositve);
    std::pair<VectorXcd, MatrixXcd>
    planeWaveStabEV(int k, bool isPositve);
    
protected:
    /****    global variable definition.   *****/
    enum { M = 64 }; // number used to approximate the complex integral.
    
    FFT Fv, Fa, Fb, Fc; // create four fft&ifft structs for forward/backward fft transform.
    FFT jFv, jFa, jFb, jFc;
  
    void CGLInit();
    void calculateCoefficients();
    virtual void NL( FFT &f);
    virtual void jNL(FFT &f);
    void dealias(FFT &Fv);
    inline int calNe();
    inline int calJacv();
};



//////////////////////////////////////////////////////////////////////
//                        Class Cgl1d                             //
//////////////////////////////////////////////////////////////////////

class Cgl1d : public Cqcgl1d {
    
public:
    Cgl1d(int N, double d, double h,
		 bool enableJacv, int Njacv,
		 double b, double c,
		 int threadNum);
protected:
    void NL(FFT &f);
    void jNL(FFT &f);
};



#endif  /* CQCGL1D_H */
