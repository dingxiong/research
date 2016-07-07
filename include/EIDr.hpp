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
    void calCoe(double h){

	ArrayXd hL = h * L;
   
	switch (scheme) {
    
	case Cox_Matthews : {
	    ArrayXXcd LR = ZR(hL);

	    ArrayXXcd LR2 = LR.square();
	    ArrayXXcd LR3 = LR.cube();
	    ArrayXXcd LRe = LR.exp();
	    ArrayXXcd LReh = (LR/2).exp();

	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();

	    a[1][0] = h * ( (LReh - 1)/LR ).rowwise().mean().real(); 
	    
	    b[0] = h * ( (-4.0 - LR + LRe*(4.0 - 3.0 * LR + LR2)) / LR3 ).rowwise().mean().real();
	    b[1] = h * 2 * ( (2.0 + LR + LRe*(-2.0 + LR)) / LR3 ).rowwise().mean().real();
	    b[3] = h * ( (-4.0 - 3.0*LR -LR2 + LRe*(4.0 - LR) ) / LR3 ).rowwise().mean().real();

	    break;
	}

	case Krogstad : {
	    ArrayXXcd LR = ZR(hL);

	    ArrayXXcd LR2 = LR.square();
	    ArrayXXcd LR3 = LR.cube();
	    ArrayXXcd LRe = LR.exp();
	    ArrayXXcd LReh = (LR/2).exp();

	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();

	    a[1][0] = h * ( (LReh - 1)/LR ).rowwise().mean().real(); 
	    a[2][0] = h * ( (LReh*(LR - 4) + LR + 4) / LR2 ).rowwise().mean().real();
	    a[2][1] = h * 2 * ( (2*LReh - LR - 2) / LR2 ).rowwise().mean().real();
	    a[3][0] = h * ( (LRe*(LR-2) + LR + 2) / LR2 ).rowwise().mean().real();
	    a[3][2] = h * 2 * ( (LRe - LR - 1)  / LR2 ).rowwise().mean().real();
	    
	    b[0] = h * ( (-4.0 - LR + LRe*(4.0 - 3.0 * LR + LR2)) / LR3 ).rowwise().mean().real();
	    b[1] = h * 2 * ( (2.0 + LR + LRe*(-2.0 + LR)) / LR3 ).rowwise().mean().real();
	    b[3] = h * ( (-4.0 - 3.0*LR -LR2 + LRe*(4.0 - LR) ) / LR3 ).rowwise().mean().real();

	    break;
	}
	    
	case Hochbruck_Ostermann : {
	    break;
	}
	    
	case Luan_Ostermann : {
	    break;
	}
	    
	default : 
	    fprintf(stderr, "Please indicate the method !\n");
	    
	}
     
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
