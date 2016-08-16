/* to comiple:
 * h5c++ -O3 test_CQCGL1dReq.cc  -L../../lib -I../../include -I$EIGEN -std=c++11 -lCQCGL1dReq -lCQCGL1d -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -lmyfft -lfftw3 -lm 
 */
#include "CQCGL1dReq.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <H5Cpp.h>

using namespace std;
using namespace Eigen;
using namespace denseRoutines;

typedef std::complex<double> dcp;


#define N40

int main(){

#ifdef N10
    //======================================================================
    // use the previous data as guess. But this approach does not work
    // if you change domain size L.
    const int N = 1024;
    const int L = 50;
    double Bi = 0.8;
    double Gi = -0.6;
    CQCGL1dReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
	
    string file = "/usr/local/home/xiong/00git/research/data/cgl/reqDi.h5";
    ArrayXd a0;
    double wth0, wphi0, err0;
    std::tie(a0, wth0, wphi0, err0) = cgl.readReq(file, "0.060000/1");
    a0 *= 0.316;

    ArrayXd a;
    double wth, wphi, err;
    int flag;
    std::tie(a, wth, wphi, err, flag) = cgl.findReq_LM(a0, wth0, wphi0, 1e-10, 100, 1000);
    fprintf(stderr, "%g %g %g\n", wth, wphi, err);
    if (flag == 0) cgl.writeReq("reqBiGi.h5", Bi, Gi, a, wth, wphi, err);

#endif
#ifdef N20
    //======================================================================
    // use a Gaussian shape guess to find the 2nd soliton. 
    const int N = 1024;
    const int L = 50;
    double Bi = 0.8;
    double Gi = -0.6;
    CQCGL1dReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
	
    VectorXcd A0 = Gaussian(N, N/2, N/20, 1);
    ArrayXd  a0 = cgl.Config2Fourier(A0);
    std::vector<double> th = cgl.optThPhi(a0);
    double wth0 = th[0], wphi0 = th[1];
    cout << wth0 << ' ' << wphi0 << endl;

    ArrayXd a;
    double wth, wphi, err;
    int flag;
    std::tie(a, wth, wphi, err, flag) = cgl.findReq_LM(a0, wth0, wphi0, 1e-10, 100, 1000);
    fprintf(stderr, "%g %g %g\n", wth, wphi, err);
    string file = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGi.h5";
    if (flag == 0) cgl.writeReq(file, Bi, Gi, 2, a, wth, wphi, err);

#endif
#ifdef N30
    //======================================================================
    // use a Gaussian shape guess and evove it for a while and then to find
    // the 1st soliton
    const int N = 1024;
    const int L = 50;
    double Bi = 0.8;
    double Gi = -0.6;
    CQCGL1dReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
	
    VectorXcd A0 = Gaussian(N, N/2, N/20, 3);
    ArrayXd  a0 = cgl.Config2Fourier(A0);
    ArrayXXd aa = cgl.intg(a0, 1e-3, 10000, 10000);
    a0 = aa.rightCols(1);
    std::vector<double> th = cgl.optThPhi(a0);
    double wth0 = th[0], wphi0 = th[1];
    cout << wth0 << ' ' << wphi0 << endl;

    ArrayXd a;
    double wth, wphi, err;
    int flag;
    std::tie(a, wth, wphi, err, flag) = cgl.findReq_LM(a0, wth0, wphi0, 1e-10, 100, 1000);
    fprintf(stderr, "%g %g %g\n", wth, wphi, err);
    string file = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGi.h5";
    if (flag == 0) cgl.writeReq(file, Bi, Gi, 1, a, wth, wphi, err);
    
#endif
#ifdef N40
    //======================================================================
    // extend the soliton solution in the Bi-Gi plane
    const int N = 1024;
    const int L = 50;
    double Bi = 0.8;
    double Gi = -0.6;

    int ids[] = {1, 2};
    for (int i = 1; i < 2; i++){
	int id = ids[i];
	CQCGL1dReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);

	string file = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGi.h5";
	double stepB = 0.1;
	int NsB = 20;
	cgl.findReqParaSeq(file, id, stepB, NsB, true);
	for (int i = 0; i < NsB+1; i++){
	    cgl.Bi = Bi+i*stepB;
	    cgl.findReqParaSeq(file, id, -0.1, 50, false);
	}
    }
    

#endif
    
    return 0;
}
