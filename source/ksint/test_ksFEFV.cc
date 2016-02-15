/* to comiple:
 * (Note : libksdim.a is static library, so the following order is important)
 *
 * h5c++ test_ksFEFV.cc -std=c++11 -O3 -march=corei7 -msse4 -msse2 -I$XDAPPS/eigen/include/eigen3 -I$RESH/include  -L$RESH/lib -lksFEFV -lksint -lfftw3 -lmyH5 -lped
 * 
 */

#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <algorithm> 
#include "ksFEFV.hpp"
#include "myH5.hpp"


using namespace std;
using namespace Eigen;
using namespace MyH5;

int main(){
    cout.precision(16);

    switch (50) {
	
    case 1: {
	/* calculate FE, FV for a single orbit */
	const double L = 22;
	const int nqr = 5; // spacing 	
	const int MaxN = 2000;  // maximal iteration number for PED
	const double tol = 1e-15; // torlearance for PED   
	const int trunc = 30; // number of vectors	
	const size_t ppId = 33;
	const string inputfileName = "../../data/ks22h001t120x64EV.h5";
	const string outputfileName = "ex.h5";
	const string ppType = "rpo";
	// KScalWriteFEFVInit(inputfileName, outputfileName, ppType, ppId, L, MaxN, tol, nqr, trunc);
	auto tmp = KScalFEFV(inputfileName, ppType, ppId, L, MaxN, tol, nqr, trunc);
	cout << tmp.second.col(0) << endl;
	
	break;
    }

    case 20: {
	/*
	  calculate FE, FV for a set of orbits 
	 */
	const double L = 22;
	const int nqr = 5; // spacing 	
	const int MaxN = 500;  // maximal iteration number for PED
	const double tol = 1e-12; // torlearance for PED   
	const int trunc = 30; // number of vectors	

	const string inName = "../../data/ks22h001t120x64EV.h5";
	const string outName = "ex.h5";
	
	int ppIds[] = {33, 36, 59, 60, 79, 81, 109, 114};
	const string ppType = "rpo";

	for(size_t i = 0; i < sizeof(ppIds)/sizeof(int); i++){
	    printf("ppId = %d\n ", ppIds[i]);

	    // auto tmp = KSreadRPO(inName, ppType, ppIds[i]);
	    // printf("T=%f, nstp=%d, r=%g, s=%f\n", std::get<1>(tmp), (int)std::get<2>(tmp), 
	    //	   std::get<3>(tmp), std::get<4>(tmp));
	    KScalWriteFEFVInit(inName, outName, ppType, ppIds[i], L, MaxN, tol, nqr, trunc);
	}

	break;

    }

    case 30: {
	/*
	  move KS data base 
	  all ppo from one file
	  rpo from 2 files
	 */
	const string inFile1 = "../../data/ks22h001t120x64EV.h5";
	const string inFile2 = "../../data/ex.h5";
	const string outFile1 = "../../data/ks22h001t120x64_t.h5";
	const string outFile2 = "../../data/ks22h001t120x64_t2.h5";

	std::vector<int> ppIds1;
	for(int i = 0; i < 200; i++) ppIds1.push_back(i+1);
	std::vector<int> ppIds2 = {33, 36, 59, 60, 79, 81, 109, 114};

	for(int i = 0; i < ppIds1.size(); i++) {
	    printf("f1 ppo, i = %d, ppId = %d \n", i, ppIds1[i]);
	    KSmoveFE(inFile1, outFile1, "ppo", ppIds1[i]);
	    KSmoveFEFV(inFile1, outFile2, "ppo", ppIds1[i]);
	}

	for(int i = 0; i < ppIds1.size(); i++) {
	    if(std::find(ppIds2.begin(), ppIds2.end(), ppIds1[i]) == ppIds2.end()){
		printf("f1 rpo, i = %d, ppId = %d \n", i, ppIds1[i]);
		KSmoveFE(inFile1, outFile1, "rpo", ppIds1[i]);
		KSmoveFEFV(inFile1, outFile2, "rpo", ppIds1[i]);
	    }
	    else {
		printf("f2 rpo, i = %d, ppId = %d \n", i, ppIds1[i]);
		KSmoveFE(inFile2, outFile1, "rpo", ppIds1[i]);
		KSmoveFEFV(inFile2, outFile2, "rpo", ppIds1[i]);
	    }
	}
	
	break;
    }

    case 40: {
	/* move KS initial condtions 
	   The purpose is to reduce the space
	   Also it changes double nstp to int nstp
	*/
	const string inFile = "../../data/ks22h001t120x64.h5";		
	const string outFile = "../../data/ks22h001t120x64_2.h5";
	
	for(int i = 0; i < 840; i++){
	    int ppId = i+1;
	    printf("ppo, i = %d, ppId = %d \n", i, ppId);
	    KSmoveRPO(inFile, outFile, "ppo", ppId);
	}
	
	for(int i = 0; i < 834; i++){
	    int ppId = i+1;
	    printf("rpo, i = %d, ppId = %d \n", i, ppId);
	    KSmoveRPO(inFile, outFile, "rpo", ppId);
	}
	
	break;
    }

    case 50: {
	/* calculate left FE, FV for a single orbit */
	const double L = 22;
	const int nqr = 5; // spacing 	
	const int MaxN = 2000;  // maximal iteration number for PED
	const double tol = 1e-13; // torlearance for PED   
	const int trunc = 30; // number of vectors	
	const size_t ppId = 1;
	const string inputfileName = "../../data/ks22h001t120x64EV.h5";
	const string outputfileName = "left.h5";
	const string ppType = "ppo";
	KScalWriteLeftFEFV(inputfileName, outputfileName, ppType, ppId, L, MaxN, tol, nqr, trunc);
	// auto tmp = KScalLeftFEFV(inputfileName, ppType, ppId, L, MaxN, tol, nqr, trunc);
	// cout << tmp.first << endl;
	
	break;

    }
	
	
    default: {
	cout << "Please indicate the right #" << endl;
    }
	
    }


    return 0;
}
