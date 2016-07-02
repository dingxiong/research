#include "KSETD.hpp"
#include "denseRoutines.hpp"
#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace MyFFT;
using namespace denseRoutines;


KSETD::KSETD(int N, double d): N(N), d(d) {
    K = ArrayXd::LinSpaced(N/2+1, 0, N/2) * 2 * M_PI / d;
    K(N/2) = 0;
    L = K*K - K*K*K*K;
    G = G = 0.5 * dcp(0,1) * K * N; 

    KSNL<ArrayXcd> nl(N, G);
    etdrk4 = std::make_shared<ETDRK4<ArrayXcd, ArrayXXcd, KSNL>>(L, nl);
}
    
KSETD::~KSETD(){}

ArrayXXd KSETD::C2R(const ArrayXXcd &v){
    int n = v.rows();
    int m = v.cols();
    ArrayXXcd vt = v.middleRows(1, n-2);
    ArrayXXd vp(2*(n-2), m);
    vp = Map<ArrayXXd>((double*)&vt(0,0), 2*(n-2), m);

    return vp;
}

ArrayXXcd KSETD::R2C(const ArrayXXd &v){
    int n = v.rows();
    int m = v.cols();
    assert( 0 == n%2);
    
    ArrayXXcd vp = ArrayXXcd::Zero(n/2+2, m);
    vp.middleRows(1, n/2) = Map<ArrayXXcd>((dcp*)&v(0,0), n/2, m);
    
    return vp;
}

std::pair<VectorXd, ArrayXXd>
KSETD::etd(const ArrayXd &a0, const double tend, const double h, const int skip_rate, 
	   int method, bool adapt){
	
    assert( N-2 == a0.size());
    ArrayXcd u0 = R2C(a0); 
    
    etdrk4->Method = method;
    
    std::pair<VectorXd, ArrayXXcd> tmp;
    if (adapt) tmp = etdrk4->intg(0, u0, tend, h, skip_rate); 
    else tmp = etdrk4->intgC(0, u0, tend, h, skip_rate); 
    
    return std::make_pair(tmp.first, C2R(tmp.second));
}

