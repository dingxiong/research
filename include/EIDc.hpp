#ifndef EIDC_H
#define EIDC_H

#include "EID.hpp"

////////////////////////////////////////////////////////////////////////////////
//   Exponential Integrator with complex data type
////////////////////////////////////////////////////////////////////////////////

template<int Cols>
class EIDc : public EID<std::complex<double>, Cols> {
    
public:
    
    using EID<std::complex<double>, Cols>::M; // nondependent name
    using EID<std::complex<double>, Cols>::R;

    EIDc(){}
    EIDc(ArrayXcd *L, 
	 typename EID<std::complex<double>, Cols>::Arycs *Y, // nondependent type
	 typename EID<std::complex<double>, Cols>::Arycs *N) 
	: EID<std::complex<double>, Cols>(L, Y, N){}
    EIDc & operator=(const EIDc &x){
	return *this;
    }

    ~EIDc(){}

    inline typename EID<std::complex<double>, Cols>::Arycs
    MUL(const ArrayXcd &C, typename EID<std::complex<double>, Cols>::Arycs &Y){
	return Y.colwise() * C;
    }

    inline ArrayXcd mean(const Ref<const ArrayXXcd> &x){
	return x.rowwise().mean();
    }
    
    inline ArrayXXcd ZR(ArrayXcd &z){
	int M1 = z.size();
	ArrayXd K = ArrayXd::LinSpaced(M, 1, M);
	ArrayXXcd r = R * (K/M * typename EID<std::complex<double>, Cols>::dcp(0,2*M_PI)).exp().transpose(); // row array.
	return z.replicate(1, M) + r.replicate(M1, 1);
    }

};

#endif /* EIDC_H */
