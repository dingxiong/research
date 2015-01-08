/* How to compile this program:
 * h5c++ test_readks.cc readks.cc ../ped/ped.cc ../ksint/ksint.cc
 * -std=c++0x -I$XDAPPS/eigen/include/eigen3 -I../../include -lfftw3
 * -march=corei7 -msse4 -O3
 *
 * or (Note : libreadks.a is static library, so the following order is important)
 *
 * h5c++ test_readks.cc -std=c++0x -I$XDAPPS/eigen/include/eigen3 -I../../include
 * -L../../lib -lreadks -lksint -lped
 * -lfftw3 -O3 -march=corei7 -msse4 -msse2
 */
#include "readks.hpp"
#include <iostream>
using namespace std;
using namespace Eigen;

int main()
{

  
  switch (6)
    {
      
    case 1: // calculate all the orbits that converged for 100 - 120
      {
	ReadKS readks("../../data/ks22h02t120.h5",
		      "../../data/ks22h02t120E.h5",
		      "../../data/ks22h02t120EV.h5");	

	const double tolErr = 1e-8; // tolerance of the error of orbits
	const int MaxN = 8000;  // maximal iteration number for PED
	const double tol = 1e-15; // torlearance for PED	
	string ppType("ppo");
	const bool rewrite = false;

	int NN;
	if(ppType.compare("ppo") == 0) NN = 840; // number of ppo
	else NN = 834; // number of rpo

	for(size_t ppId = 1; ppId < NN+1; ppId++){
	  tuple<ArrayXd, double, double, double, double> 
	    pp = readks.readKSinit(ppType, ppId);
	  double r = get<3>(pp);
	  
	  if (r < tolErr){  
	    printf("********* ppId = %zd ***********\n", ppId);
	    readks.calKSOneOrbit(ppType, ppId, MaxN, tol, false);
	  }
	}
	
	break;
      }

    case 2: // test the existance of e/ve 
      {
	ReadKS readks("../../data/ks22h02t120.h5",
		      "../../data/ks22h02t120E.h5",
		      "../../data/ks22h02t120EV.h5");	
	string ppType("rpo");	
	int NN;
	if(ppType.compare("ppo") == 0) NN = 840; // number of ppo
	else NN = 834; // number of rpo
	
	MatrixXi status = readks.checkExistEV(ppType, NN);
	cout << status << endl;

	break;
      }
      
    case 3: // test marginalPos()
      {
	ReadKS readks("../../data/ks22h02t120.h5",
		      "../../data/ks22h02t120E.h5",
		      "../../data/ks22h02t120EV.h5");
	string ppType("ppo");
	int NN = 840;
	MatrixXi status = readks.checkExistEV(ppType, NN);
	std::vector<double> sm;
	for(size_t i = 0; i < NN; i++)
	  {
	    if( 1 == status(i,0) ){
	      int ppId = i + 1;	      
	      printf("========= i = %zd ========\n", i);
	      MatrixXd eigVals = readks.readKSe(ppType, ppId);
	      std::vector< std::pair<double, int> >  
		marginal = readks.findMarginal(eigVals.col(0));
	      sm.push_back(fabs(marginal[0].first));
	      sm.push_back(fabs(marginal[1].first));
	      // std::cout << marginal[0].first << ' '<< marginal[0].second << std::endl;
	      // std::cout << marginal[1].first << ' '<< marginal[1].second << std::endl;
	    }
	    
	  }
	cout << *std::max_element(sm.begin(), sm.end()); // about 1e-6
	break;
      }

    case 4: // test indexSubspace()
      {
	ReadKS readks("../../data/ks22h02t120.h5",
		      "../../data/ks22h02t120E.h5",
		      "../../data/ks22h02t120EV.h5");
	string ppType("ppo");
	int ppId = 2;
	MatrixXd eigVals = readks.readKSe(ppType, ppId);
	MatrixXi x = readks.indexSubspace(eigVals.col(2), eigVals.col(0));
	cout << x << endl;
	
	break;
      }
      
    case 5 : // calculate the difference between marginal FVs and velocity field or
	     // group tangent for 32 Fourier modes.
      {
	const double L = 22;
	const int Nks = 64;
	const int N = Nks - 2;
	string fileName("../../data/ks22h001t120x64");
	string ppType("ppo");
	const int ppId = 1;
	
	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N, Nks);
	// calculate Floquet exponents and vectors and only keep vectors
	MatrixXd FVs = readks.calKSFloquet(ppType, ppId, 1000, 1e-15).second;
	const int M = FVs.cols();

	cout << FVs.rows() << 'x' <<  FVs.cols() << endl;
	
	tuple<ArrayXd, double, double, double, double> 
	  pp = readks.readKSinit(ppType, ppId);
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
	KS ks(Nks, T/nstp, L);
	ArrayXXd aa = ks.intg(a, nstp);
	
	cout << aa.rows() << 'x' << FVs.cols() << endl;

	// compare the velocity with marginal Floquet vectors
	MatrixXd marg1 = FVs.middleRows(N*2, N); // the 3rd is the velocity
	for(int i = 0; i < M; i++){
	  VectorXd v = ks.velocity(aa.col(i));
	  cout << v.norm() << endl;
	  cout << v.rows() << endl;
	  v = v.array() / v.norm(); 
	  cout << (marg1.col(i) - v).norm() << endl;
	}

	break;
      }

   case 6: // calculate the first 10 ppos for N = 64
      {

	const double L = 22;
	const int Nks = 64;
	const int N = Nks - 2;
	string fileName("../../data/ks22h001t120x64");
	string ppType("ppo");
	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N, Nks);
	
	const int MaxN = 1000;  // maximal iteration number for PED
	const double tol = 1e-15; // torlearance for PED	
	const bool rewrite = false;
	const int nqr = 10;
	const int trunc = 10;

	int NN = 4;

	for(size_t ppId = NN; ppId < NN+1; ppId++){
	  printf("********* ppId = %zd ***********\n", ppId);
	  readks.calKSOneOrbit(ppType, ppId, MaxN, tol, rewrite, nqr, trunc);
	}
	
	break;
      }

    default:
      {
	cout << "Please indicate the right #" << endl;
      }
    }

  return 0;
}
