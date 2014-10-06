/** To compile this class, you need to have g++ >= 4.6, eigen >= 3.1
 * g++ -o libksintm1.so ksintM1.cc ksint.cc
 * -shared -fpic -lm -lfftw3 -std=c++0x
 * -march=core2 -O3 -msse2 -I/usr/include/eigen3 -I../../include
 *  */

#ifndef KSINTM1_H
#define KSINTM1_H

#include "ksint.hpp"

class KSM1 : public KS {

public:
  typedef struct{
    ArrayXXd aa;
    ArrayXd tt;
  } KSat;
  
  /* member functions */
  KSat intg(const ArrayXd &a0, size_t nstp, size_t np);
  KSat intg2(const ArrayXd &a0, double T, size_t np);

  /* constructors */
  KSM1(int N = 32, double h = 0.25, double d = 22);
  explicit KSM1(const KSM1 &x);
  KSM1 & operator=(const KSM1 &x);

protected:

  void NL(KSfft &f) override;
};

#endif	/* KSINTM1_H */
