#ifndef EIDR_H
#define EIDR_H

#include "EID.hpp"

class EIDr : public EID<double> {

public :
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // constructor and destructor
    EIDr(ArrayXd &L, ArrayXcd *Y, ArrayXcd *N) : EID<double>(L, Y, N){}
    ~EIDr(){}
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // virtual functions 

    inline
    ArrayXd
    mean(ArrayXXcd &x){ 
	return x.rowwise().mean().real();
    }

    /**
     * @brief calcuate the matrix to do averge of phi(z). 
     */
    inline
    ArrayXXcd ZR(ArrayXd &z){
    
	int M1 = z.size();
	ArrayXd K = ArrayXd::LinSpaced(M, 1, M); // 1,2,3,...,M 

	ArrayXXcd r = R * ((K-0.5)/M * dcp(0, M_PI)).exp().transpose();

	return z.template cast<std::complex<double>>().replicate(1, M) + r.replicate(M1, 1);
    }

}; 


#endif	/* EIDR_H */
