#ifndef CQCGL_H
#define CQCGL_H

#include "cqcgl1d.hpp"

class Cqcgl : public Cqcgl1d {
    
    //////////////////////////////////////////////////////////////////////
    double b;
    double c;
    double dr;
    double di;

    //////////////////////////////////////////////////////////////////////
    Cqcgl(int N, double d, double h,
	  bool enableJacv, int Njacv,
	  double b, double c,
	  double dr, double di,
	  int threadNum);
    ~Cqcgl();
    Cqcgl & operator=(const Cqcgl &x);

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
