/* to compile:
 * g++ test_ksint.cc -std=c++11 -lksint -ldenseRoutines -lfftw3 -I ../../include/ -L ../../lib/ -I$EIGEN -O3 -DEIGEN_FFTW_DEFAULT && ./a.out
 */
#include "ksint.hpp"
#include "denseRoutines.hpp"
#include "myH5.hpp"
#include <Eigen/Dense>
#include <iostream>
using namespace std;
using namespace Eigen;
using namespace MyH5;
using namespace denseRoutines;

#define N10

int main(){
    
#ifdef N10
    //======================================================================
    // 
    int N = 64;
    double L = 22;
    ArrayXd a0 = ArrayXd::Random(N-2) * 0.1;
    KS ks(N, L);
    ArrayXXd aa = ks.intgC(a0, 0.01, 20, 10);
#endif
#if 0
    case 10 :{		  /* find eq and req for N = 64 from N = 32 */
	std::string file = "../../data/ksReqx32.h5";
	std::string outFile = "tmp.h5";
	const int N = 64;
	KS ks(N, 0.01, 22);

	for (int i = 0; i < 2; i++){
	    int Id = i+1;
	    auto tmp = KSreadReq(file, Id);
	    VectorXd x0(VectorXd::Zero(N-1));
	    x0.head(30) = tmp.first; 
	    x0(N-2) = tmp.second * 2*M_PI/22; 

	    auto result = ks.findReq(x0, 1e-13, 100, 20);
	
	    VectorXd &a0 = std::get<0>(result);
	    double omega = std::get<1>(result);
	    double err = std::get<2>(result);
	    
	    KSwriteReq(outFile, Id, a0, omega, err);
	}

	for (int i = 0; i < 3; i++){
	    int Id = i + 1;
	    auto a0 = KSreadEq(file, Id);
	    VectorXd x0(VectorXd::Zero(N-2));
	    x0.head(30) = a0;
	
	    auto result = ks.findEq(x0, 1e-13, 100, 20);

	    VectorXd &a = std::get<0>(result);
	    double err = std::get<1>(result);
	    
	    KSwriteEq(outFile, Id, a, err);
	}
	

	break;
    }

	
    case 11 : {		/* calculate the stability exponents of eq */
	std::string file = "../../data/ks22Reqx64.h5";
	auto eq = KSreadEq(file, 1);
	VectorXd a0 = eq.first;
	double err = eq.second;

	const int N = 64;
	KS ks(N, 0.01, 22);
	auto ev = ks.stabEig(a0);
	cout << ev.first << endl;

	break;
    }
	
    case 12 : {		/* calculate the stability exponents of req */
	std::string file = "../../data/ks22Reqx64.h5";
	auto req = KSreadReq(file, 1);
	VectorXd &a0 = std::get<0>(req);
	double w = std::get<1>(req);
	double err = std::get<2>(req);

	const int N = 64;
	KS ks(N, 0.01, 22);
	auto ev = ks.stabReqEig(a0, w);

	cout << ev. first << endl;
	
	break;
    }

    case 13: {			/* write stability exponents */
	std::string file = "../../data/ks22Reqx64.h5";
	std::string outFile = "ks22Reqx64.h5";
	const int N = 64;
	KS ks(N, 0.01, 22);
	
	for(int i = 0; i < 3 ; i++){
	    int Id = i + 1;
	    auto eq = KSreadEq(file, Id);
	    VectorXd &a0 = eq.first;
	    auto ev = ks.stabEig(a0);
	    
	    KSwriteEqE(outFile, Id, ev.first);
	}

	for(int i = 0; i < 2; i++){
	    int Id = i + 1;
	    auto req = KSreadReq(file, Id);
	    VectorXd &a0 = std::get<0>(req);
	    double w = std::get<1>(req);
	    auto ev = ks.stabReqEig(a0, w);
	    
	    KSwriteReqE(outFile, Id, ev.first);
	}

	break;
    }

    case 14: {
	
	std::string file = "../../data/ks22h001t120x64.h5";
	std::string poType = "rpo";
	
	MatrixXd a0;
	double T, r, s;
	int nstp;
	std::tie(a0, T, nstp, r, s) = KSreadRPO(file, poType, 1);
	
        KS ks(64, 22);
	auto tmp = ks.aintgj(a0, 0.01, T, 1);
	savetxt("a.dat", ks.hs);

	break;
    }
	
	
    default :
	{
	    cout << "please indicate the correct index." << endl;
	}
    }

#endif
    return 0;
}
