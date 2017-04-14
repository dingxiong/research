/* How to compile:
 * h5c++ test_KSPO.cc -L../../lib -I../../include -I$EIGEN -std=c++11 -lKSPO -lksint -lmyH5 -literMethod -ldenseRoutines -lsparseRoutines -lfftw3 -lm 
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

#define CASE_40

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
    // findPO by single shooting
    int N = 64;
    double L = 21.9;
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
    
    ArrayXXd ap;
    double errp;
    int flag;
    std::tie(ap, errp, flag) = ks.findPO_LM(x, isRPO, nstp, 1e-12, 100, 30);
    // if (flag == 0) ks.write("rpoBiGi2.h5", cgl.toStr(Bi, Gi, 1), x, nstp, err);
#endif
#ifdef CASE_40
    //================================================================================
    int N = 64;
    double L = 22-0.0001;
    KSPO ks(N, L);
    
    string fileName = "../../data/ks22h001t120x64EV.h5";
    H5File file(fileName, H5F_ACC_RDONLY);
    string ppType = "rpo";
    bool isRPO = ppType == "rpo";
    int ppId = 2;
    int M = 20;

    ArrayXd a;
    double T, theta, err;
    int nstp;
    std::tie(a, T, nstp, theta, err) = ks.read(file, ks.toStr(ppType, ppId), isRPO);
    ArrayXXd aa = ks.intgC(a, T/nstp, T, nstp/M);

    ArrayXXd x(ks.N, M);
    x << aa, ArrayXXd::Ones(1, M) * T/M, ArrayXXd::Zero(1, M);
    x(N-1, M-1) = theta;
    
    ArrayXXd ap;
    double errp;
    int flag;
    std::tie(ap, errp, flag) = ks.findPO_LM(x, isRPO, nstp/M, 1e-12, 100, 30);
#endif
#if 0
 case 3: 

 case 4: // refine the inital condition for N=32
     {
	 const int Nks = 64;
	 const int N = Nks - 2;
	 const double L = 22;
	 string fileName("../../data/ks22h1t120x64");
	 string ppType("ppo");
	 ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N, Nks, L);
	
	 int NN(0);
	 if(ppType.compare("ppo") == 0) NN = 840;
	 else NN = 834;

	 const int MaxN = 30;
	 const double tol = 1e-14;
	 const int M = 10;

	 ////////////////////////////////////////////////////////////
	 // mpi part 
	 int left = 0;
	 int right = NN;
	
	 MPI_Init(&argc, &argv);
	 int rank, num;
	 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	 MPI_Comm_size(MPI_COMM_WORLD, &num);
	 int inc = ceil( (double)(right - left) / num);
	 int istart = left + rank * inc;
	 int iend = min( left + (rank+1) * inc, right);
	 printf("MPI : %d / %d; range : %d - %d \n", rank, num, istart, iend);
	 ////////////////////////////////////////////////////////////

	 for(int i = istart; i < iend; i++){
	     const int ppId = i+1; 
	     std::tuple<ArrayXd, double, double>
		 pp = readks.readKSorigin(ppType, ppId);	
	     ArrayXd &a = get<0>(pp); 
	     double T = get<1>(pp);
	     double s = get<2>(pp); 

	     const int nstp = ceil(ceil(T/0.001)/10)*10;
	     KSrefine ksrefine(Nks, L);
	     tuple<MatrixXd, double, double, double> 
		 p = ksrefine.findPOmulti(a, T, nstp, M, ppType,
					  0.1, -s/L*2*M_PI, MaxN, tol, false, false);
	  
	     printf("r = %g for %s ppId = %d \n", get<3>(p), ppType.c_str(), ppId);
	     readks.writeKSinitMulti("../../data/tmp.h5", ppType, ppId, 
				     make_tuple(get<0>(p), get<1>(p)*nstp, nstp, 
						get<3>(p), -L/(2*M_PI)*get<2>(p))
				     );
	 }

	 ////////////////////////////////////////////////////////////
	 MPI_Finalize();
	 ////////////////////////////////////////////////////////////
	
	 break;
     }
      
 case 5 : // refine step by step
     {
	 const int Nks = 64;
	 const int N = Nks - 2;
	 const double L = 22;
	 string fileName("../../data/ks22h001t120x64");
	 string ppType("rpo");
	 ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N, Nks, L);
	
	 int NN(0);
	 if(ppType.compare("ppo") == 0) NN = 840;
	 else NN = 834;

	 const int MaxN = 100;
	 const double tol = 1e-14;
	 const int M = 10;

	 ////////////////////////////////////////////////////////////
	 // mpi part 
	 int left = 0;
	 int right = NN;
	
	 MPI_Init(&argc, &argv);
	 int rank, num;
	 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	 MPI_Comm_size(MPI_COMM_WORLD, &num);
	 int inc = ceil( (double)(right - left) / num);
	 int istart = left + rank * inc;
	 int iend = min( left + (rank+1) * inc, right);
	 printf("MPI : %d / %d; range : %d - %d \n", rank, num, istart, iend);
	 ////////////////////////////////////////////////////////////

	 for(int i = istart; i < iend; i++){
	     const int ppId = i+1; 
	     // printf("\n****  ppId = %d   ****** \n", ppId); 
	     std::tuple<ArrayXd, double, double, double, double>
		 pp = readks.readKSinit(ppType, ppId);	
	     ArrayXd &a = get<0>(pp); 
	     double T = get<1>(pp);
	     int nstp = (int) get<2>(pp);
	     double r = get<3>(pp);  // printf("r = %g\n", r);
	     double s = get<4>(pp); 

	     double hinit = T / nstp;
	     // nstp *= 5;
	     KSrefine ksrefine(Nks, L);
	     tuple<MatrixXd, double, double, double> 
		 p = ksrefine.findPOmulti(a, T, nstp, M, ppType,
					  hinit, -s/L*2*M_PI, MaxN, tol, false, true);
	  
	     printf("r = %g for %s ppId = %d \n", get<3>(p), ppType.c_str(), ppId);
	     readks.writeKSinitMulti("../../data/tmp.h5", ppType, ppId, 
				     make_tuple(get<0>(p), get<1>(p)*nstp, nstp, 
						get<3>(p), -L/(2*M_PI)*get<2>(p))
				     );

	  
	 }
	
	 ////////////////////////////////////////////////////////////
	 MPI_Finalize();
	 ////////////////////////////////////////////////////////////

	 break;
     }
      
}
#endif
    return 0;
}


