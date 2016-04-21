#include <iostream>
#include "cqcgl.hpp"

using namespace sparseRoutines;
using namespace denseRoutines;
using namespace Eigen;
using namespace std;

/**
 * Constructor of cubic quintic complex Ginzburg-Landau equation
 * A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
 */
Cqcgl::Cqcgl(int N, double d,
	     double b, double c,
	     double dr, double di,
	     int dimTan, int threadNum)
    : Cqcgl1d(N, d, -1, 1, b, 1, c, -dr, -di, dimTan, threadNum),
      b(b), 
      c(c),
      dr(dr),
      di(di)
{				
}

Cqcgl::~Cqcgl(){}

Cqcgl & Cqcgl::operator=(const Cqcgl &x){
    return *this;
}


/**************************************************************/
/*                plane wave related                          */
/**************************************************************/

/**
 * @brief Return plane waves.
 *
 * @param[in] k            the wave index of the plane wave
 * @paran[in] isPositive   whether the sign in amplitude is positive or not
 * @return [a0, a, w]     Fourier state, amplitude, omega
 * 
 * @note This function only works for the b, c, dr, di construction
 */
std::tuple<ArrayXd, double, double>
Cqcgl::planeWave(int k, bool isPositve){
    double qk, a2, w;
    
    qk = 2 * M_PI * k / d;
    if(isPositve) a2 = 1/(2*dr) * (1 + sqrt(1-4*dr*(qk*qk+1)));
    else a2 = 1/(2*dr) * (1 - sqrt(1-4*dr*(qk*qk+1)));
    w = b*qk*qk - c*a2 + di*a2*a2;
    
    ArrayXd a0(ArrayXd::Zero(Ndim));
    if(k >= 0) a0(2*k) = sqrt(a2) * N;
    else a0(Ndim + 2*k) = sqrt(a2) * N; // please check
    
    return std::make_tuple(a0, sqrt(a2), w);
}

/**
 * @brief Return plane waves.  -- short version
 */
void 
Cqcgl::planeWave(ArrayXd &a0, double &a, double &w, 
		   int k, bool isPositve){
    auto tmp = planeWave(k, isPositve);
    a0 = std::get<0>(tmp);
    a = std::get<1>(tmp);
    w = std::get<2>(tmp);
}

/**
 * @brief Stability exponents of plane wave
 *
 * @see planeWave(), eReq()
 */
VectorXcd Cqcgl::planeWaveStabE(int k, bool isPositve){
    auto tmp = planeWave(k, isPositve);
    return eReq(std::get<0>(tmp), 0, std::get<2>(tmp));
}

std::pair<VectorXcd, MatrixXcd>
Cqcgl::planeWaveStabEV(int k, bool isPositve){
    auto tmp = planeWave(k, isPositve);
    return evReq(std::get<0>(tmp), 0, std::get<2>(tmp));
}

