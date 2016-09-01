// to compile
// g++ test_CQCGL1dRpo_arpack.cc -std=c++11 -O3  -march=corei7 -msse4 -msse2 -I$EIGEN -I$RESH/include -I$XDAPPS/arpackpp/include -L$RESH/lib -lCQCGL1dRpo_arpack -lCQCGL1dRpo -lCQCGL1d -lmyfft -lfftw3 -lm -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -llapack -larpack -lsuperlu -lopenblas -ldenseRoutines

#include <iostream>
#include <arsnsym.h>
#include <Eigen/Dense>
#include "CQCGL1dRpo_arpack.hpp"

using namespace std;
using namespace Eigen;
using namespace denseRoutines;

#define cee(x) (cout << (x) << endl << endl)

#define CASE_10


int main(int argc, char **argv){

#ifdef CASE_10
    //======================================================================
    // to visulize the limit cycle first 
    const int N = 1024;
    const double L = 50;
    double Bi = 0.8;
    double Gi = -3.6;
    
    CQCGL1dRpo_arpack cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
    string file = "../../data/cgl/rpoBiGi.h5";
    cgl.calEVParaSeq(file, std::vector<double>{Bi}, std::vector<double>{Gi}, 10, true);


#endif
#ifdef CASE_20
    //======================================================================
    // test vc2vr and vr2vc
    MatrixXd A(10, 10); 
    A.setRandom();
    VectorXcd e;
    MatrixXcd v;
    std::tie(e, v) = evEig(A, 2);

    cee(e);
    cee(v);
    
    MatrixXd vr = vc2vr(v);
    cee(vr);

    MatrixXcd vc = vr2vc(e, vr);
    cee(vc);

#endif

    return 0;
}
