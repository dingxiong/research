#ifndef ETDRK4C_H
#define ETDRK4C_H

#include <Eigen/Dense>
#include "ETDRK4"

//////////////////////////////////////////////////////////////////////////////////
//                        Complex ETDRK4 class
//////////////////////////////////////////////////////////////////////////////////

template<class Ary, class Mat, template<class> class NL>
class ETDRK4c : public ETDRK4<Ary, Mat, NL> {

public:

    ////////////////////////////////////////////////////////////
    ArrayXcd L, E, E2, a21, b1, b2, b4;
    
    
    ////////////////////////////////////////////////////////////
    ETDRK4c(ArrayXcd linear, NL<Ary> nonlinear) : L(linear), nl(nonlinear){}
    ~ETDRK4c(){}

};


/**
 * @brief Calculate the coefficients of ETDRK4.
 */
template<class Ary, class Mat, template<class> class NL>
void
ETDRK4<Ary, Mat, NL>::
calCoe(double h){

    ArrayXd hL = h*L;
    ArrayXXcd LR = ZR(hL);
    
    E = hL.exp();
    E2 = (hL/2).exp();

    ArrayXXcd LR2 = LR.square();
    ArrayXXcd LR3 = LR.cube();
    ArrayXXcd LRe = LR.exp();
    

    a21 = h * ( ((LR/2.0).exp() - 1)/LR ).rowwise().mean(); 
    b1 = h * ( (-4.0 - LR + LRe*(4.0 - 3.0 * LR + LR2)) / LR3 ).rowwise().mean();
    b2 = h * ( (2.0 + LR + LRe*(-2.0 + LR)) / LR3 ).rowwise().mean();
    b4 = h * ( (-4.0 - 3.0*LR -LR2 + LRe*(4.0 - LR) ) / LR3 ).rowwise().mean();

}

/**
 * @brief calcuate the matrix to do averge of phi(z). 
 */
template<class Ary, class Mat, template<class> class NL>
ArrayXXcd
ETDRK4<Ary, Mat, NL>::
ZR(ArrayXcd &z){
    
    int M1 = z.size();
    ArrayXd K = ArrayXd::LinSpaced(M, 1, M); // 1,2,3,...,M 

    ArrayXXcd r = R * (K/M*dcp(0,2*M_PI)).exp().transpose(); // row array.
    
    return z.replicate(1, M) + r.replicate(M1, 1);
}


#endif	/* ETDRK4C_H */
