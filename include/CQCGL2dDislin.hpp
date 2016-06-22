#ifndef CQCGL2DDISLIN_H
#define CQCGL2DDISLIN_H

#include "CQCGL2d.hpp"
#include "discpp.h"

class CQCGL2dDislin : public CQCGL2d {

public:
    
    //////////////////////////////////////////////////////////////////////
   
    
    //////////////////////////////////////////////////////////////////////
    CQCGL2dDislin(int N, int M, double dx, double dy,
		  double b, double c, double dr, double di,
		  int threadNum);
    CQCGL2dDislin(int N, double dx,
		  double b, double c, double dr, double di,
		  int threadNum);
    ~CQCGL2dDislin();
    CQCGL2dDislin & operator=(const CQCGL2dDislin &x);

    //////////////////////////////////////////////////////////////////////
    void 
    initPlot(Dislin &g, const double Lx, const double Ly, 
	     const double Lz);
    void 
    plot(Dislin &g, const double t, const double err);
    void
    endPlot(Dislin &g);

    void
    constAnim(const ArrayXXcd &a0, const double h, const int skip_rate);
    void
    adaptAnim(const ArrayXXcd &a0, const double h0, const int skip_rate);  
};

#endif  /* CQCGL2DDISLIN_H */


