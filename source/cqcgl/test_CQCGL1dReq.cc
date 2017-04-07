/* to comiple:
 * h5c++ -O3 test_CQCGL1dReq.cc  -L../../lib -I../../include -I$EIGEN -std=c++11 -lCQCGL1dReq -lCQCGL1d -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -lfftw3 -lm
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
using namespace MyH5;

typedef std::complex<double> dcp;


#define N10

int main(int argc, char **argv){

#ifdef N10
    //======================================================================
    // test the accuracy of req.
    const int N = 1024, L = 50;
    double Bi = 0.8, Gi = -0.6;
    CQCGL1dReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
	
    string fileName = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGi.h5";
    H5File file(fileName, H5F_ACC_RDWR);
    ArrayXd a0;
    double wth0, wphi0, err0;
    std::tie(a0, wth0, wphi0, err0) = cgl.read(file, cgl.toStr(Bi, Gi, 1));
    a0 += ArrayXd::Random(a0.size()) * 0.1;

    ArrayXd a;
    double wth, wphi, err;
    int flag;
    std::tie(a, wth, wphi, err, flag) = cgl.findReq_LM(a0, wth0, wphi0, 1e-10, 100, 1000);
    fprintf(stderr, "%g %g %g\n", wth, wphi, err);

#endif
#ifdef N20
    //======================================================================
    // use a Gaussian shape guess to find the 2nd soliton. 
    const int N = 1024;
    const double L = 50;
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
    if (flag == 0) cgl.write(file, Bi, Gi, 2, a, wth, wphi, err);

#endif
#ifdef N30
    //======================================================================
    // use a Gaussian shape guess and evove it for a while and then to find
    // the 1st soliton
    const int N = 1024;
    const double L = 50;
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
    if (flag == 0) cgl.write(file, Bi, Gi, 1, a, wth, wphi, err);
    

#endif
#ifdef N40
    //======================================================================
    // try to calculate the eigenvalue and eigenvector of one req
    const int N = 1024;
    const double L = 50;
    double Bi = 1.8;
    double Gi = -0.6;
    CQCGL1dReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
    
    string file = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGi.h5";
    int id = 1;

    ArrayXd a0;
    double wth0, wphi0, err0;
    std::tie(a0, wth0, wphi0, err0) = cgl.read(file, cgl.toStr(Bi, Gi, id));
    VectorXcd e;
    MatrixXcd v;
    std::tie(e, v) = cgl.evReq(a0, wth0, wphi0); 
    
    cout << e.head(10) << endl;
    cout << endl;
    cout << v.cols() << ' ' << v.rows() << endl;

#endif
#ifdef N50
    //======================================================================
    // combine the calculated E/V data
   
    string s = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGiEV";
    H5File fout(s + ".h5", H5F_ACC_RDWR);
    int ids[] = {1, 2};
    
    for ( int i = 0; i < 8; i++){
	H5File fin(s + "_add_" + to_string(i) + ".h5", H5F_ACC_RDONLY);
	for (int j = 0; j < 2; j++){
	    int id = ids[j];
	    for( int k = 0; k < 24; k++){
		double Bi = 3.5 + 0.1*k;
		for(int p = 0; p < 55; p++){
		    double Gi = -5.6 + 0.1*p;
		    string g = CQCGL1dReq::toStr(Bi, Gi, id);
		    if (checkGroup(fin, g+"/vr", false)){
			fprintf(stderr, "%d %g %g\n", id, Bi, Gi);
			CQCGL1dReq::move(fin, g, fout, g, 2);
		    }
		}
	    }
	}
    }

#endif
#ifdef N60
    //======================================================================
    // extend the soliton in one direction of parameter space
    iterMethod::LM_OUT_PRINT = false;
    iterMethod::LM_IN_PRINT = false;
    iterMethod::CG_PRINT = false;

    const int N = 1024;
    const double L = 50;
    double Bi = 4.5;
    double Gi = -5.6;

    CQCGL1dReq cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 0);
    string fileName = "/usr/local/home/xiong/00git/research/data/cgl/reqBiGi.h5";
    H5File file(fileName, H5F_ACC_RDWR);

    double stepB = 0.1;
    int NsB = 55;
    
    cgl.findReqParaSeq(file, 1, stepB, NsB, false);
    cgl.findReqParaSeq(file, 2, stepB, NsB, false);
    // cgl.findReqParaSeq(file, 2, 0.1, 53, false);
    
#endif
    
    return 0;
}
