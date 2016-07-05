#ifndef EIDC_H
#define EIDC_H

#include "EID.hpp"

template<class NL>
class EIDc : public EID<std::complex<double>, NL> {

public :
									
    typedef std::complex<double> dcp;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // constructor and destructor
    EIDc(ArrayXd L, NL nl) : EID<dcp, NL>(L, nl){}
    ~EIDc(){}
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // virtual functions 

    void calCoe(double h){

	ArrayXd hL = h*L;
   
	switch (scheme) {
    
	case Cox_Matthews : {
	    ArrayXXcd LR = ZR(hL);

	    ArrayXXcd LR2 = LR.square();
	    ArrayXXcd LR3 = LR.cube();
	    ArrayXXcd LRe = LR.exp();
	    ArrayXXcd LReh = (LR/2).exp();

	    c[1] = (hL/2).exp();
	    c[3] = hL.exp();

	    a[1][0] = h * ( (LReh - 1)/LR ).rowwise().mean(); 
	    
	    b[0] = h * ( (-4.0 - LR + LRe*(4.0 - 3.0 * LR + LR2)) / LR3 ).rowwise().mean();
	    b[1] = h * 2 * ( (2.0 + LR + LRe*(-2.0 + LR)) / LR3 ).rowwise().mean();
	    b[3] = h * ( (-4.0 - 3.0*LR -LR2 + LRe*(4.0 - LR) ) / LR3 ).rowwise().mean();
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
	    a[2][0] = h * ( (LReh*(LR - 4) + LR + 4) / LR2 ).rowwise().mean();
	    a[2][1] = h * 2 * ( (2*LReh - LR - 2) / LR2 ).rowwise().mean();
	    a[3][0] = h * ( (LRe*(LR-2) + LR + 2) / LR2 ).rowwise().mean();
	    a[3][2] = h * 2 * ( (LRe - LR - 1)  / LR2 ).rowwise().mean();
	    
	    b[0] = h * ( (-4.0 - LR + LRe*(4.0 - 3.0 * LR + LR2)) / LR3 ).rowwise().mean();
	    b[1] = h * 2 * ( (2.0 + LR + LRe*(-2.0 + LR)) / LR3 ).rowwise().mean();
	    b[3] = h * ( (-4.0 - 3.0*LR -LR2 + LRe*(4.0 - LR) ) / LR3 ).rowwise().mean();
	}
	    
	case Hochbruck_Ostermann : {
	    
	}
	    
	case Luan_Ostermann : {
	    
	}
	    
	default : 
	    fprintf(stderr, "Please indicate the method !\n");
	    
	}
     
    }

    /**
     * @brief calcuate the matrix to do averge of phi(z). 
     */
    ArrayXXcd
    ZR(ArrayXcd &z){
	
	int M1 = z.size();
	ArrayXd K = ArrayXd::LinSpaced(M, 1, M); // 1,2,3,...,M 
	
	ArrayXXcd r = R * (K/M*dcp(0,2*M_PI)).exp().transpose(); // row array.
	
	return z.replicate(1, M) + r.replicate(M1, 1);
    }

}; 


#endif	/* EIDR_H */
