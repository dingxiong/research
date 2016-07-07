/* to compile:
 * h5c++ test_ksint.cc -std=c++0x -lksint -lmyfft -lmyH5 -ldenseRoutines -literMethod -lfftw3 -I ../../include/ -L ../../lib/ -I $XDAPPS/eigen/include/eigen3 -O3 && ./a.out
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

int main(){
    /// -----------------------------------------------------
    switch (1){
	
    case 1:
	{
	    ArrayXd a0 = ArrayXd::Random(30) * 0.1;
	    KS ks(32, 22);
	    ArrayXXd aa;
	    for (int i = 0; i < 1; i++) {
		aa = ks.intg(a0, 0.01, 20, 1);
	    }
	    // cout << aa.rightCols(1) << endl << endl;
	    break;
	}


    case 2 :
	{
	    ArrayXd a0 = ArrayXd::Random(62) * 0.1;
	    KS ks(64, 22);
	    pair<ArrayXXd, ArrayXXd> tmp = ks.intgj(a0, 0.01, 2000, 10);
	    // cout << tmp.second.col(0).tail(30) << endl;
	    
	    break;
	}

# if 0
    case 3: // test velocity
	{
      
	    ArrayXd a0(ArrayXd::Ones(30)*0.1);
	    KS ks(32, 0.25, 22); 
	    cout << ks.velocity(a0) << endl;
	    /*
	      ArrayXd a0(ArrayXd::Ones(62)*0.1);
	      KS ks(64, 0.1, 22);
	      cout << ks.velocity(a0) << endl;
	    */
	    break;
	}

    case 4: // test disspation
	{
	    ArrayXd a0(ArrayXd::Ones(30)*0.1);
	    KS ks(32, 0.1, 22);
	    tuple<ArrayXXd, VectorXd, VectorXd> tmp = ks.intgDP(a0, 10000, 1);
	    cout << get<1>(tmp).tail(1) << endl << endl;
	    cout << get<2>(tmp).tail(1) << endl;

	    break;
	}

    case 5: // test 1st mode integrator
	{
	    ArrayXd a0(ArrayXd::Ones(62)*0.1);
	    a0(1) = 0;
	    KSM1 ks(64, 0.01, 22);
	    pair<ArrayXXd, ArrayXd> tmp = ks.intg(a0, 10, 1);
	    cout << tmp.first.cols() << endl << endl;
	    // cout << get<2>(tmp).tail(1) << endl;

	    break;
	}

    case 6: {			// test Eq

	std::string file = "../../data/ksReqx32.h5";
	auto a0 = KSreadEq(file, 1).first;
	KS ks(32, 0.01, 22);
	auto ev = ks.stabEig(a0);
	cout << ev.first << endl;

	break;
    }

    case 7: {			// test the req 
	std::string file = "../../data/ksReqx32.h5";
	auto tmp = KSreadReq(file, 2);
	VectorXd a0 = tmp.first;
	double c = tmp.second;
	cout << c << endl << endl;
	
	KS ks(32, 0.01, 22);
	auto ev = ks.stabReqEig(a0, c*2*M_PI/22);
	cout << ks.velg(a0, c*2*M_PI/22) << endl;
	cout << endl << ev.first << endl;
	break;
    }
	
    case 8: {			// test findReq
	
	std::string file = "../../data/ksReqx32.h5";
	auto tmp = KSreadReq(file, 2);
	const int N = 64;

	VectorXd x0(VectorXd::Zero(N-1));
	x0.head(30) = tmp.first; 
	x0(N-2) = tmp.second * 2*M_PI/22; 

	KS ks(N, 0.01, 22);
	auto result = ks.findReq(x0, 1e-13, 100, 20);
	
	VectorXd &a0 = std::get<0>(result);
	double omega = std::get<1>(result);
	double err = std::get<2>(result);

	cout << std::get<0>(result) << endl << endl;
	cout << std::get<1>(result) << endl << endl;
	cout << std::get<2>(result) << endl << endl;

	VectorXd v = ks.velg(a0, omega);
	cout << v.norm() << endl;
	
	break;
    }

    case 9 :{			// test findEq
	
	std::string file = "../../data/ksReqx32.h5";
	auto a0 = KSreadEq(file, 3);
	const int N = 64;
	
	VectorXd x0(VectorXd::Zero(N-2));
	x0.head(30) = a0;
	
	KS ks(N, 0.01, 22);
	auto result = ks.findEq(x0, 1e-13, 100, 20);

	VectorXd &a = std::get<0>(result);
	double err = std::get<1>(result);
	
	cout << a << endl << endl;
	cout << err << endl << endl;

	VectorXd v = ks.velocity(a);
	cout << v.norm() << endl;
	
	break;
    }

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

#endif
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

    return 0;
}
