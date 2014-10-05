/** To compile this class, you need to have g++ >= 4.6, eigen >= 3.1
 * g++ ksint.cc -march=corei7 -O3 -msse4.2 -I/usr/include/eigen3
 * -lm -lfftw3 -std=c++0x
 *  */
#ifndef KSINT_H
#define KSINT_H

#include <fftw3.h>
#include <complex>
#include <Eigen/Dense>

using Eigen::ArrayXXcd; 
using Eigen::ArrayXXd;
using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::Map;

/*============================================================
 *                       Class : KS integrator
 *============================================================*/
class KS{
  
public:
  typedef std::complex<double> dcp;
  typedef struct{
    ArrayXXd aa;
    ArrayXXd daa;
  } KSaj;
  
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
  ArrayXXd intg(const ArrayXd &a0, size_t nstp, size_t np = 1);
  KSaj intgj(const ArrayXd &a0, size_t nstp, size_t np = 1, size_t nqr = 1);
  ArrayXXd C2R(const ArrayXXcd &v);
  ArrayXXcd R2C(const ArrayXXd &v);

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
  void NL(KSfft &f);
  void jNL(KSfft &f);
  void initFFT(KSfft &f, int M);
  void freeFFT(KSfft &f);
  void fft(KSfft &f);
  void ifft(KSfft &f);
};

#endif	/* KSINT_H */
