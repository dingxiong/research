/* to comiple:
 * h5c++ -O3 test_CQCGL1dSubReq.cc  -L../../lib -I../../include -I$EIGEN -std=c++11 -lCQCGL1dSubReq -lCQCGL1dSub -lCQCGL1dReq -lCQCGL1d -ldenseRoutines -literMethod -lmyH5 -lfftw3 -lm
 */
#include "CQCGL1dSubReq.hpp"			
#include "CQCGL1dReq.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <H5Cpp.h>

using namespace std;
using namespace Eigen;
using namespace denseRoutines;
using namespace MyH5;

typedef std::complex<double> dcp;


#define N50

int main(int argc, char **argv){

#ifdef N10
    //======================================================================
    // use the full-space req as initial condition to find req in the
    // symmetric subspace.
    const int N = 1024;
    const double L = 50;
    double Bi = 0.8, Gi = -0.6;
    CQCGL1dReq cgl0(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
    CQCGL1dSubReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
	
    string fileName = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGi.h5";
    H5File file(fileName, H5F_ACC_RDWR);
    ArrayXd a0;
    double wth0, wphi0, err0;
    std::tie(a0, wth0, wphi0, err0) = cgl0.read(file, cgl.toStr(Bi, Gi, 2));
    a0 = a0.head(cgl.Ndim);

    ArrayXd a;
    double wphi, err;
    int flag;
    std::tie(a, wphi, err, flag) = cgl.findReq_LM(a0, wphi0, 1e-10, 100, 1000);
    fprintf(stderr, "%g %g\n", wphi, err);
    H5File fout("sub.h5", H5F_ACC_TRUNC);
    if (flag == 0) cgl.write(fout, cgl.toStr(Bi, Gi, 1), a, wphi, err);    
#endif
#ifdef N20
    //======================================================================
    // the same as N10, but navigate in the parameter plane.
    const int N = 1024;
    const double L = 50;
	
    string finName = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGi.h5";
    H5File file(finName, H5F_ACC_RDWR);
    string foutName = "sub.h5";
    H5File fout(foutName, H5F_ACC_TRUNC);
    
    ArrayXd a0;
    double wth0, wphi0, err0;
    ArrayXd a;
    double wphi, err;
    int flag;
    auto gs = scanGroup(finName);
    for(auto entry : gs) {
	double Bi = stod(entry[0]), Gi = stod(entry[1]);
	int id = stoi(entry[2]);
	fprintf(stderr, "\n %d %g %g \n", id, Bi, Gi);
	
	CQCGL1dReq cgl0(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
	CQCGL1dSubReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
	std::tie(a0, wth0, wphi0, err0) = cgl0.read(file, cgl.toStr(Bi, Gi, id));
	a0 = a0.head(cgl.Ndim);
	std::tie(a, wphi, err, flag) = cgl.findReq_LM(a0, wphi0, 1e-10, 100, 1000);
	fprintf(stderr, "%g %g\n", wphi, err);
	if (flag == 0) cgl.write(fout, cgl.toStr(Bi, Gi, id), a, wphi, err);    
    }
    
#endif
#ifdef N40
    //======================================================================
    // try to calculate the eigenvalue and eigenvector of one req
    const int N = 1024;
    const double L = 50;
    double Bi = 0.8, Gi = -0.6;
    CQCGL1dSubReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
    
    string fileName = "/usr/local/home/xiong/00git/research/data/cgl/reqsubBiGi.h5";
    H5File file(fileName, H5F_ACC_RDWR);
    int id = 1;

    ArrayXd a0;
    double wphi0, err0;
    std::tie(a0, wphi0, err0) = cgl.read(file, cgl.toStr(Bi, Gi, id));
    VectorXcd e;
    MatrixXcd v;
    std::tie(e, v) = cgl.evReq(a0, wphi0); 
    
    cout << e.head(10) << endl;
    cout << endl;
    cout << v.cols() << ' ' << v.rows() << endl;

#endif
#ifdef N50
    //======================================================================
    // calculate E/V
    const int N = 1024;
    const double L = 50;
    
    string s = "/usr/local/home/xiong/00git/research/data/cgl/reqsubBiGi";
    auto gs = scanGroup(s + ".h5");
    // H5File fin(s + ".h5", H5F_ACC_RDWR);
    H5File fout(s + "EV.h5", H5F_ACC_RDWR);
    
    ArrayXd a0;
    double wphi0, err0;
    VectorXcd e;
    MatrixXcd v;
    
    int i = 0;
    for (auto entry : gs){
	double Bi = stod(entry[0]), Gi = stod(entry[1]);
	int id = stoi(entry[2]);
	if( !checkGroup(fout, CQCGL1dSubReq::toStr(Bi, Gi, id) + "/er", false) ){
	    fprintf(stderr, "%d/%d : %d %g %g \n", ++i, (int)gs.size(), id, Bi, Gi);
	    CQCGL1dSubReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
	    std::tie(a0, wphi0, err0) = cgl.read(fout, cgl.toStr(Bi, Gi, id));
	    std::tie(e, v) = cgl.evReq(a0, wphi0); 
	    cgl.writeE(fout, cgl.toStr(Bi, Gi, id), e);
	    cgl.writeV(fout, cgl.toStr(Bi, Gi, id), v.leftCols(10));
	}
    }

#endif
    return 0;
}
