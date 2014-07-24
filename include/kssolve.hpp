#ifndef KSSOLVE_H
#define KSSOLVE_H

#include <complex>
#include <fftw3.h>
#include <new>

class Ks {

public:
  /*****    structure definition.        ******/
  typedef std::complex<double> dcomp;
  
  typedef struct{
    double *E, *E2;
    double *k, *L;
    double *Q, *f1, *f2, *f3;
    dcomp *g;
  } KsCoe;
  
  // member variables.
  const int N;
  const double d;
  const double h;
  KsCoe coe;
  
  // constructors, destructor, copy assignment.
  Ks(int N = 32, double d = 22, double h = 0.25);
  explicit Ks(const Ks &x);
  ~Ks();
  Ks & operator=(const Ks &x);

  // member function
  /* @brief KS solver without calculating Jacobian matrix.
   * 
   * @param[in] a0 initial condition, size N-2 array
   * @param[in] h time step
   * @param[in] nstp number of steps to be integrated
   * @param[in] np state saving spacing.
   * @param[out] aa saved state vector size = (nstp/np)*(N-2)
   * eg. if state column vector is v0, v1, ... vn-1, then
   * aa is a row vector [ v0^{T}, v1^{T}, ... vn-1^{T}]. */
  void kssolve(double *a0, int nstp, int np, double *aa);

  /* @brief KS solver with calculating Jacobian (size (N-2)*(N-2)).
   * 
   * @param[in] nqr Jacobian saving spacing spacing
   * @param[out] daa saved Jacobian matrix. size = (nstp/nqr)*(N-2)*(N-2).
   * eg. If Jacobian matrix is J=[v1, v2,..., vn] each vi is a
   * column vector,  then 
   * daa is a row vector [vec(J1), vec(J2), vec(Jn)], where
   * vec(J)= [v1^{T}, v2^{T},...,vn^{T}] with each element
   * visted column-wise. */
  void kssolve(double *a0, int nstp, int np, int nqr, double *aa, double *daa);

protected:   
  /****    global variable definition.   *****/
  static constexpr double PI=3.14159265358979323;

  /** @brief Structure for convenience of rfft.
   * For forward real Fourier transform, 'in' is real
   * and 'out' is complex.
   * For inverse real Fourier transform, the situation reverses. */  
  typedef struct{
    fftw_plan p;
    fftw_complex *c; // complex array.
    double *r; // real array
  } FFT;



  /* member function */
  virtual void initKs(double *a0, dcomp *v, double *aa, FFT &rp, FFT &p);
  virtual void initKs(double *a0, dcomp *v, double *aa, FFT *rp, FFT *p);
  virtual void initJ(dcomp *v);
  void cleanKs(FFT &rp, FFT &p);
  void cleanKs(FFT *rp, FFT *p);
  void calcoe(const int M = 16);
  void onestep(dcomp *v, FFT &p, FFT &rp);
  void onestep(dcomp *v, FFT *p, FFT *rp);
  virtual void calNL(dcomp *u, dcomp *Nv, const FFT &p, const FFT &rp);
  virtual void calNL(dcomp *u, dcomp *Nv, const FFT *p, const FFT *rp );
  void irfft(const dcomp *u, const FFT &rp);
  void rfft(double *u, const FFT &p);
  void initFFT(FFT &p,  int a);
  void freeFFT(FFT &p);
  
};



#endif /* KSSOLVE_H */
