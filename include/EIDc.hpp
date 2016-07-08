#ifndef EIDC_H
#define EIDC_H

#include "EID.hpp"

class EIDc : public EID<std::complex<double>> {
    
 public:
    
    EIDc(ArrayXcd &L, ArrayXcd *Y, ArrayXcd *N) : EID<std::complex<double>>(L, Y, N){}
    ~EIDc(){}

    inline ArrayXcd mean(ArrayXXcd &x){
	return x.rowwise().mean();
    }
    
    inline ArrayXXcd ZR(ArrayXcd &z){
	int M1 = z.size();
	ArrayXd K = ArrayXd::LinSpaced(M, 1, M) - 0.5; 
	ArrayXXcd r = R * (K/M*dcp(0,2*M_PI)).exp().transpose(); // row array.
	return z.replicate(1, M) + r.replicate(M1, 1);
    }

};

#endif /* EIDC_H */
