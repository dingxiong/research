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

#define CASE_10

int main(int argc, char **argv) {

#ifdef CASE_10
    //================================================================================    
    // move data
    string fin = "../../data/Ruslan/ks22h1t120x64.h5";
    H5File fin2(fin, H5F_ACC_RDONLY);
    string fout = "tmp.h5";
    H5File fout2(fout, H5F_ACC_TRUNC);

    VectorXd a;
    double T, r, s;
    int nstp;
    
    vector<vector<string>> gs = scanGroup(fin);
    for(auto v : gs){
	bool isRPO = v[0] == "rpo";
	std::tie(a, T, nstp, s, r) = KSPO::read(fin2, v[0] + "/" + v[1], isRPO);
	KSPO::write(fout2, KSPO::toStr(v[0], stoi(v[1])), isRPO, a, T, 0, -s/22*2*M_PI, r);
    }
    
#endif
#ifdef CASE_20
    //================================================================================    
    // test multiF(). For rpo, the shift should be reversed.
    string fileName("../../data/ks22h02t100");
    string ppType("ppo");
    const int ppId = 1;

    ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5");
    std::tuple<ArrayXd, double, double, double, double>
	pp = readks.readKSinit(ppType, ppId);	
    ArrayXd &a = get<0>(pp); 
    double T = get<1>(pp);
    int nstp = (int)get<2>(pp);
    double r = get<3>(pp);
    double s = get<4>(pp);
	
    KS ks(32, T/nstp, 22);
    ArrayXXd aa = ks.intg(a, nstp, nstp/10);
    KSrefine ksrefine(32, 22);
    VectorXd F = ksrefine.multiF(ks, aa.leftCols(aa.cols()-1),
				 nstp/10, ppType, -s/22*2*M_PI);
    cout << F.norm() << endl;
	
#endif
#if 0
 case 2: // test findPO() 
     {
	 const int Nks = 64;
	 const int N = Nks - 2;
	 const double L = 22;
	 string fileName("../../data/ks22h1t120x64");
	 string ppType("rpo");
	 const int ppId = 23;

	 ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N, Nks, L);
	 std::tuple<ArrayXd, double, double>
	     pp = readks.readKSorigin(ppType, ppId);	
	 ArrayXd &a = get<0>(pp); 
	 double T = get<1>(pp);
	 double s = get<2>(pp); cout << s << endl;

	 const int nstp = ceil(ceil(T/0.001)/10)*10; cout << nstp << endl;
	 KSrefine ksrefine(Nks, L);
	 tuple<MatrixXd, double, double, double> 
	     p = ksrefine.findPOmulti(a, T, nstp, 10, ppType,
				      0.1, -s/L*2*M_PI, 20, 1e-14, true, false);
	 cout << get<0>(p).cols() << endl;
	 cout << get<1>(p) * nstp << endl;
	 cout << get<2>(p) << endl;
	 cout << get<3>(p) << endl;

	 KS ks(Nks, get<1>(p), L);
	 VectorXd df = ksrefine.multiF(ks, get<0>(p), nstp/10, ppType, get<2>(p));
	 VectorXd df2 = ksrefine.multiF(ks, get<0>(p).col(0), nstp, ppType, get<2>(p));
	 cout << df.norm() << endl;
	 cout << df2.norm() << endl;
	 break;
	  
     }

 case 3: // distiguish rpo5 and rpo6
     {
	 const int Nks = 32;
	 const int N = Nks - 2;
	 const double L = 22;
	 string fileName("../../data/ks22h1t120");
	 string ppType("rpo");
	 const int ppId = 17;

	 ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N, Nks, L);
	 std::tuple<ArrayXd, double, double>
	     pp = readks.readKSorigin(ppType, ppId);	
	 ArrayXd &a = get<0>(pp); 
	 double T = get<1>(pp);
	 double s = get<2>(pp); cout << s << endl;

	 const int nstp = ceil(ceil(T/0.02)/10)*10; cout << nstp << endl;
	 KSrefine ksrefine(Nks, L);
	 tuple<MatrixXd, double, double, double> 
	     p = ksrefine.findPOmulti(a, T, nstp, 10, ppType,
				      0.25, -s/L*2*M_PI, 100, 1e-15, true, false);
	 double th = get<2>(p); cout << th << endl;
	 double Tnew = get<1>(p) * nstp; std::cout << Tnew << std::endl;
	 cout << get<0>(p).cols() << endl;
	 cout << get<3>(p) << endl;

	 // calculate the Flqouet exponents 
	 KS ks(Nks, get<1>(p), L);
	 MatrixXd daa = ks.intgjMulti(get<0>(p), nstp/10, 1, 1).second;
	 PED ped;
	 ped.reverseOrderSize(daa); // reverse order.
	 if(ppType.compare("ppo") == 0)
	     daa.leftCols(N) = ks.reflect(daa.leftCols(N)); // R*J for ppo
	 else // R*J for rpo
	     daa.leftCols(N) = ks.rotate(daa.leftCols(N), th);
	 //MatrixXd eigvals = ped.EigVals(daa, 1000, 1e-15, true);
	 //eigvals.col(0) = eigvals.col(0).array()/Tnew;

	 //cout << eigvals << endl;
	
	 // write data
	 /*
	   fileName = "../../data/ks22h02t100";
	   ReadKS readks2(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N, Nks, L);
	   readks2.writeKSinit("../../data/ks22h1t120.h5",  ppType, ppId,
	   make_tuple(get<0>(p).col(0), get<1>(p)*nstp, nstp, 
	   get<3>(p), -L/(2*M_PI)*get<2>(p))
	   );
	   readks2.calKSOneOrbit(ppType, ppId, 1000, 1e-15, false);
	 */
	 break;
	  
     }

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


