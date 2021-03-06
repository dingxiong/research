#ifndef EIDC_H
#define EIDC_H

#include "EID.hpp"

////////////////////////////////////////////////////////////////////////////////
//   Exponential Integrator with complex data type
////////////////////////////////////////////////////////////////////////////////

class EIDc : public EID<std::complex<double>> {
    
 public:
    
    EIDc(){}
    EIDc(ArrayXcd *L, ArrayXXcd *Y, ArrayXXcd *N) : EID<std::complex<double>>(L, Y, N){}
    EIDc & operator=(const EIDc &x){
	return *this;
    }
    ~EIDc(){}
    
    inline ArrayXXcd MUL(const ArrayXcd &C, const ArrayXXcd &Y){ 
	return Y.colwise() * C;
    }
    
    inline ArrayXcd mean(const Ref<const ArrayXXcd> &x){
	return x.rowwise().mean();
    }
    
    inline ArrayXXcd ZR(ArrayXcd &z){
	int M1 = z.size();
	ArrayXd K = ArrayXd::LinSpaced(M, 1, M);
	ArrayXXcd r = R * (K/M * dcp(0,2*M_PI)).exp().transpose(); // row array.
	return z.replicate(1, M) + r.replicate(M1, 1);
    }
};

#endif /* EIDC_H */
