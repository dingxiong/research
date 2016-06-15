#include <iostream>
#include "CQCGL2d.hpp"

using namespace denseRoutines;
using namespace Eigen;
using namespace std;

/**
 * Constructor of cubic quintic complex Ginzburg-Landau equation
 * A_t = -A + (1 + b*i) (A_{xx}+A_{yy}) + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
 */
CQCGL2d::CQCGL2d(int N, int M, double dx, double dy,
		 double b, double c, double dr, double di,
		 int threadNum)
    : CQCGLgeneral2d(N, M, dx, dy, -1, 1, b, 1, c, -dr, -di, threadNum),
      b(b), 
      c(c),
      dr(dr),
      di(di)
{				
}

CQCGL2d::CQCGL2d(int N, double dx,
		 double b, double c, double dr, double di,
		 int threadNum)
    : CQCGL2d(N, N, dx, dx, b, c, dr, di, threadNum)
{				
}


CQCGL2d::~CQCGL2d(){}

CQCGL2d & CQCGL2d::operator=(const CQCGL2d &x){
    return *this;
}



