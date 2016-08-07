// g++ randWalk.cc -O3 -std=c++11 -I$EIGEN -I$RESH/include -L$RESH/lib -ldenseRoutines
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>
#include "denseRoutines.hpp"

using namespace std;
using namespace Eigen;
using namespace denseRoutines;

typedef std::complex<double> dcp;

VectorXd 
walk2d(const int Np, const int Ns, const double tol){
    std::srand(std::time(0));
    double max = static_cast<double>(RAND_MAX);

    VectorXd a(Np), s(Ns);
    VectorXcd z(Np);
    a.setZero();
    z.setZero();
    for (int i = 0; i < Ns; i++){
	for(int j = 0; j < Np; j++){
	    double t, m;
	    do {
		t = std::rand() / max;
		Vector3d d;
		d << fabs(t-a(j)), fabs(t-a(j)+1), fabs(t-a(j)-1) ;
		m = d.minCoeff();
	    } while (m < tol);

	    a(j) = t;
	    z(j) += exp(t*2*M_PI*dcp(0,1));
	}
	
	s(i) = (z.array() * z.array().conjugate()).real().mean();
    }
    
    return s;
}

int main(){

    
    const int Np = 10000;
    const int Ns = 1000;
    
    int n = 4;
    MatrixXd ss(Ns, n);
    for (int i = 0; i < n; i++){
	ss.col(i) = walk2d(Np, Ns, 0.1*i);
    }
    
    savetxt("ss.dat", ss);

    return 0;
}
