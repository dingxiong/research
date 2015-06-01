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

#include <fftw3.h>
#include <complex>
#include <utility>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "sparseRoutines.hpp"
using std::pair; using std::make_pair;
using Eigen::MatrixXd; using Eigen::VectorXd;
using Eigen::MatrixXcd; using Eigen::VectorXcd;
using Eigen::ArrayXXcd; using Eigen::ArrayXcd;
using Eigen::ArrayXXd; using Eigen::ArrayXd;
using Eigen::Map; using Eigen::Ref;

class Cqcgl1d {
  
public:
    typedef std::complex<double> dcp;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::Triplet<double> Tri;
    
    const int N;
    const double d;
    const double h;

    double Br, Bi, Gr, Gi, Dr, Di, Mu;
    ArrayXd K, Kindex;
    ArrayXcd L, E, E2, Q, f1, f2, f3;

    // constructor, destructor, copy assignment.
    Cqcgl1d(int N = 256, double d = 50, double h = 0.01, 
	    double Mu = -0.1, double Br = 1.0, double Bi = 0.8,
	    double Dr = 0.125, double Di = 0.5, double Gr = -0.1,
	    double Gi = -0.6);
    explicit Cqcgl1d(const Cqcgl1d &x);
    ~Cqcgl1d();
    Cqcgl1d & operator=(const Cqcgl1d &x);

    // member functions.
    ArrayXXd intg(const ArrayXd &a0, const size_t nstp, const size_t np = 1);
    pair<ArrayXXd, ArrayXXd>
    intgj(const ArrayXd &a0, const size_t nstp, const size_t np = 1,
	  const size_t nqr = 1);
    ArrayXXd C2R(const ArrayXXcd &v);
    ArrayXXcd R2C(const ArrayXXd &v);
    ArrayXXd Fourier2Config(const Ref<const ArrayXXd> &aa);
    ArrayXXd Config2Fourier(const Ref<const ArrayXXd> &AA);
    ArrayXXd calMag(const Ref<const ArrayXXd> &AA);
    ArrayXXd Fourier2ConfigMag(const Ref<const ArrayXXd> &aa);
    
    ArrayXd velocity(const ArrayXd &a0);
    ArrayXd velocityReq(const ArrayXd &a0, const double th,
			const double phi);
    MatrixXd stab(const ArrayXd &a0);
    MatrixXd stabReq(const ArrayXd &a0, double th, double phi);

    ArrayXXd reflect(const Ref<const ArrayXXd> &aa);
    
    ArrayXXd transRotate(const Ref<const ArrayXXd> &aa, const double th);
    ArrayXXd transTangent(const Ref<const ArrayXXd> &aa);
    MatrixXd transGenerator();

    ArrayXXd phaseRotate(const Ref<const ArrayXXd> &aa, const double phi);
    ArrayXXd phaseTangent(const Ref<const ArrayXXd> &aa);
    MatrixXd phaseGenerator();

    ArrayXXd Rotate(const Ref<const ArrayXXd> &aa, const double th, const double phi);
    std::tuple<ArrayXXd, ArrayXd, ArrayXd>
    orbit2slice(const Ref<const ArrayXXd> &aa);
    MatrixXd ve2slice(const ArrayXXd &ve, const Ref<const ArrayXd> &x);
    
    VectorXd multiF(const ArrayXXd &x, const int nstp, const double th, const double phi);
    pair<SpMat, VectorXd>
    multishoot(const ArrayXXd &x, const int nstp, const double th, const double phi, bool doesPrint = false);
    std::pair<MatrixXd, VectorXd>
    newtonReq(const ArrayXd &a0, const double th, const double phi);
    
protected:
    /****    global variable definition.   *****/
    enum { M = 64 }; // number used to approximate the complex integral.

    /** @brief Structure for convenience of rfft.   */  
    struct CGLfft{ 
	/* 3 different stage os ETDRK4:
	 *  v --> ifft(v) --> fft(g(ifft(v)))
	 * */
	fftw_plan p, rp;  // plan for fft/ifft.
	fftw_complex *c1, *c2, *c3; // c1 = v, c2 = ifft(v), c3 = fft(g(ifft(v)))
	Map<ArrayXXcd> v1, v2, v3;

	CGLfft() : v1(NULL, 0, 0), v2(NULL, 0, 0), v3(NULL, 0, 0) {}
    };
  
    CGLfft Fv, Fa, Fb, Fc; // create four fft&ifft structs for forward/backward fft transform.
    CGLfft jFv, jFa, jFb, jFc;
  
    void CGLInit();
    void NL( CGLfft &f);
    void jNL(CGLfft &f);
    void fft(CGLfft &f);
    void ifft(CGLfft &f);
    void initFFT(CGLfft &f, int M);
    void freeFFT(CGLfft &f);

};


#endif  /* CQCGL1D_H */
