// to compile
//
// add -I$XDAPPS/arpackpp/include -llapack -larpack -lsuperlu -lopenblas
//
// first use
// mpicxx --showme -O3 test_CQCGL1dRpo_mpi.cc  -L../../lib -I../../include -I$EIGEN -std=c++11 -lCQCGL1dRpo -lCQCGL1d -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -lmyfft -lfftw3 -lm
//
// then change g++ to h5c++
// 
// h5c++ -O3 test_CQCGL1dRpo_mpi.cc -L../../lib -I../../include -I/usr/local/home/xiong/apps/eigen/include/eigen3 -std=c++11 -lCQCGL1dRpo -lCQCGL1d -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -lmyfft -lfftw3 -lm -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi -pthread -L/usr//lib -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi -ldl -lhwloc 

#include <iostream>
#include <arsnsym.h>
#include <Eigen/Dense>
#include "denseRoutines.hpp"

using namespace std;
using namespace Eigen;
using namespace denseRoutines;

class Mat {
public:
    MatrixXd A;

    Mat(MatrixXd A) : A(A) {}
    ~Mat(){}

    void mul(double *v, double *w){
	Map<const VectorXd> mv(v, A.cols());
	Map<VectorXd> mw(w, A.cols());
	mw = A * mv;
    }
};

int main(int argc, char **argv){
    int n = 100;
    int ne = 5;
    MatrixXd A(n, n);
    A.setRandom();

    Mat mat(A);
    ARNonSymStdEig<double, Mat> dprob(n, ne, &mat, &Mat::mul, "LM");
    dprob.ChangeTol(1e-9);
    
    VectorXd er(ne+1), ei(ne+1);
    MatrixXd v((ne+1)*n, 1);
    double *p_er = er.data();
    double *p_ei = ei.data();
    double *p_v = v.data();
    int nconv = dprob.EigenValVectors(p_v, p_er, p_ei);
    VectorXcd e(ne+1);
    e.real() = er;
    e.imag() = ei;
    v.resize(n, ne+1);
    VectorXcd v1(n);
    v1.real() = v.col(2);
    v1.imag() = v.col(3);
    
    cout << nconv << endl << endl;
    cout << e << endl << endl;
    
    double err = (A*v1 - e(2)*v1).norm();
    cout << err << endl;
    
    return 0;
}
