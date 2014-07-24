#ifndef KSSOLVEM1_H
#define KSSOLVEM1_H

#include "kssolve.hpp"

class KsM1 : public Ks {

public:

  /** @brief KS solver in the 1st mode slice without Jacobian. */
  void kssolve(double *a0, int nstp, int np, double *aa, double *tt);
  void kssolve(double *a0, int nstp, int np, int nqr, double *aa,
	       double *daa, double *tt);
  
  // constructors, destructor, copy assignment.
  KsM1(int N = 32, double d = 22, double h = 0.25);
  explicit KsM1(const KsM1 &x);
  KsM1 & operator=(const KsM1 &x);

protected:
  /* function onestep() keeps unchanged. */
  void initKs(double *a0, dcomp *v, double *aa, FFT &rp, FFT &p) override;
  void initKs(double *a0, dcomp *v, double *aa, FFT *rp, FFT *p) override;
  void calNL(dcomp*u, dcomp *Nv, const FFT &p, const FFT &rp) override;
  void calNL(dcomp *u, dcomp *Nv, const FFT *p, const FFT *rp ) override;
  void initJ(dcomp *v);
};

#endif	/* KSSOLVEM1_H */
