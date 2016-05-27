#ifndef CQCGL2D_H
#define CQCGL2D_H

#include "CQCGLgeneral2d.hpp"

class CQCGL2d : public CQCGLgeneral2d {

public:
    
    //////////////////////////////////////////////////////////////////////
    double b;
    double c;
    double dr;
    double di;

    //////////////////////////////////////////////////////////////////////
    CQCGL2d(int N, int M, double dx, double dy,
	    double b, double c, double dr, double di,
	    int threadNum);
    CQCGL2d(int N, double dx,
	    double b, double c, double dr, double di,
	    int threadNum);
    ~CQCGL2d();
    CQCGL2d & operator=(const CQCGL2d &x);

    //////////////////////////////////////////////////////////////////////
    
};

#endif  /* CQCGL2D_H */
