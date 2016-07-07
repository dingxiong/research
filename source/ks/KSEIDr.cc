#include "KSEIDr.hpp"
#include "denseRoutines.hpp"
#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace denseRoutines;


KSEIDr::KSEIDr(int N, double d) : 
    N(N), d(d)
{
    K = ArrayXd::LinSpaced(N/2+1, 0, N/2) * 2 * M_PI / d;
    K(N/2) = 0;
    L = K*K - K*K*K*K;
    G = 0.5 * dcp(0,1) * K * N; 
    
}
    
KSEIDr::~KSEIDr(){}

ArrayXXd KSEIDr::C2R(const ArrayXXcd &v){
    int n = v.rows();
    int m = v.cols();
    ArrayXXcd vt = v.middleRows(1, n-2);
    ArrayXXd vp(2*(n-2), m);
    vp = Map<ArrayXXd>((double*)&vt(0,0), 2*(n-2), m);

    return vp;
}

ArrayXXcd KSEIDr::R2C(const ArrayXXd &v){
    int n = v.rows();
    int m = v.cols();
    assert( 0 == n%2);
    
    ArrayXXcd vp = ArrayXXcd::Zero(n/2+2, m);
    vp.middleRows(1, n/2) = Map<ArrayXXcd>((dcp*)&v(0,0), n/2, m);
    
    return vp;
}

