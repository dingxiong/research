#ifndef KSSOLVEM1_H
#define KSSOLVEM1_H

#include "kssolve.hpp"
#include <vector>

class KsM1 : public Ks {

public:
  /** structure definition */
  typedef std::vector<double> dvec;
  typedef std::vector<dcomp> dcvec;

  /** @brief KS solver in the 1st mode slice without Jacobian.
   *
   *  If np != 1, the time sequence is not accurate.
   *  */
  void kssolve(double *a0, int nstp, int np, double *aa, double *tt);
  void kssolve(double *a0, int nstp, int np, int nqr, double *aa,
	       double *daa, double *tt);

  /** @brief velocity in the 1st mode slice
   * 
   * @param[in] a0 vector of size N-3
   * @return velocity vector of size N-3
   */
  dvec velo(dvec &a0);

  /** @brief integrate point a0 to poincare section defined by x0
   * 
   * with its velocity filed. a0 should be below the section which means the
   * direction of the poincare section is parallel to the velocity field.
   * 
   * @param[in] x0 template point of the Poincare section 
   *            U(x) = v0 * (x - x0)
   * @param[in] a0 state point that needed to be integrated onto
   *            Poincare section.
   * @return  N-1 vector with last two elements be the time and error
   */
  dvec ks2poinc(dvec &x0, dvec &a0);

  

  // constructors, destructor, copy assignment.
  KsM1(int N = 32, double d = 22, double h = 0.25);
  explicit KsM1(const KsM1 &x);
  KsM1 & operator=(const KsM1 &x);

  /* ------------------- ---------------------------------------- */

protected:
  /* function onestep() keeps unchanged. */
  void initKs(double *a0, dcomp *v, double *aa, FFT &rp, FFT &p) override;
  void initKs(double *a0, dcomp *v, double *aa, FFT *rp, FFT *p) override;
  void calNL(dcomp*u, dcomp *Nv, const FFT &p, const FFT &rp) override;
  void calNL(dcomp *u, dcomp *Nv, const FFT *p, const FFT *rp ) override;
  void initJ(dcomp *v) override;

  dvec cv2rv(const dcvec &a0);

  /* ------------------------------------------------------------ */

private:
  class Int2p{
    
  public:
    KsM1 &ks;
    dvec v0;

    Int2p(KsM1 &k): ks(k){}
    void operator() (const dvec &x, dvec &dxdt, const double /* t */);
  };
  
  Int2p int2p;

};

#endif	/* KSSOLVEM1_H */
