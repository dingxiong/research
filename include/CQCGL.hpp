#ifndef CQCGL_H
#define CQCGL_H

#include "CQCGLgeneral.hpp"

class CQCGL : public CQCGLgeneral {

public:
    
    //////////////////////////////////////////////////////////////////////
    double b;
    double c;
    double dr;
    double di;

    //////////////////////////////////////////////////////////////////////
    CQCGL(int N, double d,
	  double b, double c,
	  double dr, double di,
	  int dimTan, int threadNum);
    ~CQCGL();
    CQCGL & operator=(const CQCGL &x);

    //////////////////////////////////////////////////////////////////////
    std::tuple<ArrayXd, double, double>
    planeWave(int k, bool isPositve);
    void 
    planeWave(ArrayXd &a0, double &a, double &w, 
	      int k, bool isPositve);
    VectorXcd planeWaveStabE(int k, bool isPositve);
    std::pair<VectorXcd, MatrixXcd>
    planeWaveStabEV(int k, bool isPositve);

    
};

#endif  /* CQCGL_H */
