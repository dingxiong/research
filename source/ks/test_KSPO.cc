/* How to compile:
 * h5c++ test_KSPO.cc -L../../lib -I../../include -I$EIGEN -std=c++11 -lKSPO -lksint -lmyH5 -literMethod -ldenseRoutines -lsparseRoutines -lped -lfftw3 -lm 
 *
 * mpicxx  has flag -cxx, which can be used to specify cc compiler. eg. -cxx=h5c++
 * */
#include "KSPO.hpp"
#include "ksint.hpp"
#include "myH5.hpp"
#include <iostream>
#include <cmath>				
//#include <mpi.h>
using namespace std;
using namespace Eigen;
using namespace MyH5;

#define CASE_60

int main(int argc, char **argv) {

#ifdef CASE_10
    //================================================================================    
    // move data
    string fin = "../../data/ks22h001t120x64EV.h5";
    H5File fin2(fin, H5F_ACC_RDONLY);
    string fout = "tmp.h5";
    H5File fout2(fout, H5F_ACC_TRUNC);

    VectorXd a;
    double T, r, s;
    int nstp; 
    MatrixXd es, vs;

    vector<vector<string>> gs = scanGroup(fin);
    for(auto v : gs){
	bool isRPO = v[0] == "rpo";
	std::tie(a, T, nstp, s, r) = KSPO::read(fin2, v[0] + "/" + v[1], isRPO);
	KSPO::write(fout2, KSPO::toStr(v[0], stoi(v[1])), isRPO, a, T, nstp, -s/22*2*M_PI, r);
	if(checkGroup(fin2, v[0] + "/" + v[1] + "/e", false)){
	    es = KSPO::readE(fin2, v[0] + "/" + v[1]); 
	    vs = KSPO::readV(fin2, v[0] + "/" + v[1]);
	    KSPO::writeE(fout2, KSPO::toStr(v[0], stoi(v[1])), es);
	    KSPO::writeV(fout2, KSPO::toStr(v[0], stoi(v[1])), vs);
	}
    }
    
#endif
#ifdef CASE_20
    //================================================================================    
    // test multiF(). For rpo, the shift should be reversed.
    int N = 64;
    double L = 22;
    KSPO ks(N, L);
    
    string fileName = "../../data/ks22h001t120x64EV.h5";
    H5File file(fileName, H5F_ACC_RDONLY);
    string ppType = "rpo";
    bool isRPO = ppType == "rpo";
    int ppId = 1;

    ArrayXd a;
    double T, theta, err;
    int nstp;
    std::tie(a, T, nstp, theta, err) = ks.read(file, ks.toStr(ppType, ppId), isRPO);
    VectorXd x(ks.N);
    x << a, T, theta;

    VectorXd F = ks.MFx(x, nstp, isRPO);
    cout << F.norm() << endl;
	
#endif
#ifdef CASE_30
    //================================================================================
    // findPO from one L = 22 orbit 
    int N = 64;
    double L = 21.95;
    KSPO ks(N, L);
    
    string fileName = "../../data/ks22h001t120x64EV.h5";
    H5File file(fileName, H5F_ACC_RDONLY);
    string ppType = "rpo";
    bool isRPO = ppType == "rpo";
    int ppId = 1;
    int M = 10;

    ArrayXd a;
    double T, theta, err;
    int nstp;
    std::tie(a, T, nstp, theta, err) = ks.read(file, ks.toStr(ppType, ppId), isRPO);
    nstp /= 10;
    ArrayXXd aa = ks.intgC(a, T/nstp, T, nstp/M); 

    ArrayXXd x;
    if(isRPO){
	x.resize(ks.N, M);
	x << aa, ArrayXXd::Ones(1, M) * T/M, ArrayXXd::Zero(1, M);
	x(N-1, M-1) = theta;
    }
    else {
	x.resize(ks.N - 1, M);
	x << aa, ArrayXXd::Ones(1, M) * T/M;
    }
    ArrayXXd ap;
    double errp;
    int flag;
    std::tie(ap, errp, flag) = ks.findPO_LM(x, isRPO, nstp/M, 1e-12, 100, 30);
    double Tp = ap.row(N-2).sum();
    double thetap = isRPO ? ap.row(N-1).sum() : 0;
    H5File fout("tmp.h5", H5F_ACC_RDWR);
    if (flag == 0) ks.write(fout, ks.toStr(L, ppType, ppId), isRPO, ap.col(0).head(N-2), Tp, nstp, thetap, errp); 

#endif
#ifdef CASE_40
    //================================================================================
    // findPO sequentially change parameters
    int N = 64;
    
    string fileName = "../../data/ksh01x64.h5";
    H5File file(fileName, H5F_ACC_RDWR);
    string ppType = "rpo";
    bool isRPO = ppType == "rpo";
    int ppId = 2;
    int M = 10;
    
    ArrayXd a;
    double T, theta, err, errp;
    int nstp, flag;
    ArrayXXd ap;
    
    double dL = 0.05, L0 = 21.4;
    for(int i = 0; i < 5; i++){
	double L = L0 - i*dL;
	KSPO ks(N, L);
	
	std::tie(a, T, nstp, theta, err) = ks.read(file, ks.toStr(L + dL, ppType, ppId), isRPO);
	ArrayXXd aa = ks.intgC(a, T/nstp, T, nstp/M); 
	
	ArrayXXd x;
	if(isRPO){
	    x.resize(ks.N, M);
	    x << aa, ArrayXXd::Ones(1, M) * T/M, ArrayXXd::Zero(1, M);
	    x(N-1, M-1) = theta;
	}
	else {
	    x.resize(ks.N - 1, M);
	    x << aa, ArrayXXd::Ones(1, M) * T/M;
	}
	
	std::tie(ap, errp, flag) = ks.findPO_LM(x, isRPO, nstp/M, 1e-12, 100, 30);
	double Tp = ap.row(N-2).sum();
	double thetap = isRPO ? ap.row(N-1).sum() : 0;
	if (flag == 0) ks.write(file, ks.toStr(L, ppType, ppId), isRPO, ap.col(0).head(N-2), Tp, nstp, thetap, errp);
	else exit(1);
    }
#endif
#ifdef CASE_50
    //================================================================================
    // calculate E/V
    int N = 64;
    double L = 21.95;
    KSPO ks(N, L);
    
    string fileName = "../../data/ksh01x64.h5";
    H5File file(fileName, H5F_ACC_RDONLY);
    string ppType = "ppo";
    bool isRPO = ppType == "rpo";
    int ppId = 1;

    ArrayXd a;
    double T, theta, err;
    int nstp;
    std::tie(a, T, nstp, theta, err) = ks.read(file, ks.toStr(L, ppType, ppId), isRPO);
    
    auto ev = ks.calEV(isRPO, a, T, nstp, theta, 1e-12, 1000, 10);
    MatrixXd &e = ev.first, &v = ev.second;

    cout << e << endl;   
#endif
#ifdef CASE_60
    //================================================================================
    // calculate E/V for all rpo/ppo
    int N = 64;
    string fileName = "../../data/ksh01x64EV.h5";
    H5File file(fileName, H5F_ACC_RDWR);
    auto gs = scanGroup(fileName);
    
    ArrayXd a;
    double T, theta, err;
    int nstp;
    
    for(auto ss : gs){
	double L = stod(ss[0]);
	string ppType = ss[1];
	bool isRPO = ppType == "rpo";
	int ppId = stoi(ss[2]);	
	string groupName = KSPO::toStr(L, ppType, ppId);
	if (!checkGroup(file, groupName + "/e", false)){
	    KSPO ks(N, L);
	    std::tie(a, T, nstp, theta, err) = ks.read(file, groupName, isRPO);
	    auto ev = ks.calEV(isRPO, a, T, nstp, theta, 1e-12, 1500, 10);
	    MatrixXd &e = ev.first, &v = ev.second;
	    ks.writeE(file, groupName, e.topRows(10));
	    ks.writeV(file, groupName, v);
	}
    }

#endif

    return 0;
}


