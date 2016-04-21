#include <iostream>
#include "cgl1d.hpp"

/**
 * The constructor of complex Ginzburg-Landau equation
 * A_t = A + (1 + b*i) A_{xx} - (1 + c*i) |A|^2 A
 *
 * @see Cqcgl1d()
 */
Cgl1d::Cgl1d(int N, double d, double h,
	     bool enableJacv, int Njacv,
	     double b, double c,
	     int threadNum)
    : Cqcgl1d::Cqcgl1d(N, d, h, enableJacv, Njacv, 1, -1, -c, 1, b, 0, 0, threadNum)
{ }


/** 
 * Nonlinear term without the quintic term
 */
void Cgl1d::NL(FFT &f){
    f.ifft();
    ArrayXcd A2 = f.v2 * f.v2.conjugate(); /* |A|^2 */
    f.v2 =  dcp(Br, Bi) * f.v2 * A2;
    f.fft();
}

/** 
 * Nonlinear term without the quintic term
 */
void Cgl1d::jNL(FFT &f){
    f.ifft(); 
    ArrayXcd A = f.v2.col(0);
    ArrayXcd aA2 = A * A.conjugate(); /* |A|^2 */
    ArrayXcd A2 = A.square();	      /* A^2 */
    dcp B(Br, Bi);
    dcp G(Gr, Gi);
    f.v2.col(0) = dcp(Br, Bi) * A * aA2;

    const int M = f.v2.cols() - 1;
    f.v2.rightCols(M) = f.v2.rightCols(M).conjugate().colwise() *  (B * A2) +
    	f.v2.rightCols(M).colwise() * (2.0*B*aA2);
    
    f.fft();
}
