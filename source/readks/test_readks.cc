/* How to compile this program:
 * h5c++ test_readks.cc readks.cc ../ped/ped.cc ../ksint/ksint.cc
 * -std=c++0x -I$XDAPPS/eigen/include/eigen3 -I../../include -lfftw3
 * -march=corei7 -msse4 -O3
 *
 * or (Note : libreadks.a is static library, so the following order is important)
 *
 * h5c++ test_readks.cc -std=c++0x -I$XDAPPS/eigen/include/eigen3 -I../../include
 * -L../../lib -lreadks -lksint -lped -lfftw3
 * -O3 -march=corei7 -msse4 -msse2
 */
#include "readks.hpp"
#include <mpi.h>
#include <iostream>
#include <cstdio>
using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{

  cout.precision(16);
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
	const int nqr = 1;
	
	ReadKS readks(fileName+".h5",
		      fileName+"Ex"+to_string(nqr)+".h5", 
		      fileName+"EVx"+to_string(nqr)+".h5", N, Nks);
	MatrixXd FVs = readks.readKSve(ppType, ppId);
	//cout << FVs.rows() << 'x' << FVs.cols() << endl;

	tuple<ArrayXd, double, double, double, double> 
	  pp = readks.readKSinit(ppType, ppId);
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
	KS ks(Nks, T/nstp, L);
	ArrayXXd aa = ks.intg(a, nstp, nqr);
	const int M = aa.cols()-1;
	
	switch (2)
	  {
	  case 1:
	    {
	      // compare the velocity with marginal Floquet vectors
	      MatrixXd marg1 = FVs.middleRows(N*2, N); // the 3rd is the velocity
	      for(int i = 0; i < M; i++){
		VectorXd v = ks.velocity(aa.col(i));
		v = v.array() / v.norm();
		double dv1 = (marg1.col(i) - v).norm();
		double dv2 = (marg1.col(i) + v).norm();
		cout << min(dv1, dv2) << endl;
	      }
	      
	      break;
	    }
	    
	  case 2: // compare the group tangent with marginal Floquet vectors
	    {
	      MatrixXd marg = FVs.middleRows(N*3, N);
	      for(int i = 0; i < M; i++){
		VectorXd tx = ks.gTangent(aa.col(i));
		tx = tx.array() / tx.norm();
		double dv1 = (marg.col(i) - tx).norm();
		double dv2 = (marg.col(i) + tx).norm();
		cout << min(dv1, dv2) << endl;
	      }
	      
	      break;
	    }
	  }
	

	break;
      }

   case 6: // calculate the  Floquet vectors/spectrum ppos/rpos for N = 64
      {
	const double L = 22;
	const int Nks = 64;
	const int N = Nks - 2;
	string fileName("../../data/ks22h001t120x64");
	string ppType("rpo");
	const int nqr = 5; // spacing 
	ReadKS readks(fileName+".h5", fileName+"E.h5",
		      fileName+"EV.h5", N, Nks, L);
	
	const int MaxN = 8000;  // maximal iteration number for PED
	const double tol = 1e-15; // torlearance for PED	
	const bool rewrite = false; 
	const int trunc = 30; // number of vectors

	int NN(0);
	if(ppType.compare("ppo") == 0) NN = 840;
	else NN = 834;
	
	////////////////////////////////////////////////////////////
	// mpi part 
	int left = 0;
	int right = 2;
	 
	MPI_Init(&argc, &argv);
	int rank, num;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num);
	int inc = ceil( (double)(right - left) / num);
	int istart = left + rank * inc;
	int iend = min( left + (rank+1) * inc, right);
	printf("MPI : %d / %d; range : %d - %d \n", rank, num, istart, iend); 
	////////////////////////////////////////////////////////////
	
	for(size_t i = istart; i < iend; i++){
	  const size_t ppId = i+1;
	  string output = ppType + to_string(ppId);
	  freopen (output.c_str(), "w", stderr);
	  fprintf(stderr, "********* ppId = %zd ***********\n", ppId);
	  // readks.calKSOneOrbit(ppType, ppId, MaxN, tol, rewrite, nqr, trunc);
	  readks.calKSOneOrbitOnlyE(ppType, ppId, MaxN, tol, rewrite, nqr);
	  fclose(stderr);
	}
	
	////////////////////////////////////////////////////////////
	MPI_Finalize();
	////////////////////////////////////////////////////////////

	break;
      }

    case 7: // calculate the Floquet exponents of ppo1 for different spacing
      {
	const double L = 22;
	const int Nks = 64;
	const int N = Nks - 2;
	// string fileName("../../data/ks22h001t120x64");
	string fileName("../../data/ks22h005t120x64");
	string ppType("ppo");
	const int ppId = 1;
	vector<double> nqr{1, 5, 10};
	
	const int MaxN = 5000;
	const double tol = 1e-15;
	const int trunc = 2;
	ReadKS readks(fileName+".h5", fileName+".h5", fileName+".h5", N, Nks);

	for (size_t i = 0; i < nqr.size(); i++) {
	  MatrixXd eigvals = readks.calKSFloquet(ppType, ppId, MaxN, tol, nqr[i], trunc).first;
	  cout << eigvals << endl;
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
