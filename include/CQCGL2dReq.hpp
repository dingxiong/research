#ifndef CQCGL2dREQ_H		
#define CQCGL2dREQ_H

#include "CQCGL2d.hpp"

class CQCGL2dReq : public CQCGL2d {

public :
    
    double tol = 1e-10;
    double minRD = 1;
    int maxit = 100;
    int maxInnIt = 100;
    double GmresRtol = 1e-6;
    int GmresRestart = 100;
    int GmresMaxit = 10;
    int hookPrint = 1;

    ArrayXXcd invL; 
    ArrayXd GmresPre;

    ////////////////////////////////////////////////////////////
    CQCGL2dReq(int N, int M, double dx, double dy,
	       double b, double c, double dr, double di,
	       int threadNum);
    CQCGL2dReq(int N, double dx,
	       double b, double c, double dr, double di,
	       int threadNum);
    ~CQCGL2dReq();
    CQCGL2dReq & operator=(const CQCGL2dReq &x);

    ////////////////////////////////////////////////////////////
    
    void calPre();
    VectorXd Fx(const VectorXd &x);
    VectorXd DFx(const VectorXd &x, const VectorXd &dx);
    std::tuple<ArrayXXcd, double, double, double, double>
    findReq_hook(const ArrayXXcd &x0, const double wthx0, 
		 const double wthy0, const double wphi0);
    
};

#endif	/* CQCGL2dREQ_H */
