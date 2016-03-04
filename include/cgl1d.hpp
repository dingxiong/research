
#ifndef CGL1D_H
#define CGL1D_H

#include "myfft.hpp"
#include "cqcgl1d.hpp"
#include "denseRoutines.hpp"

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

#endif CGL1D_H
