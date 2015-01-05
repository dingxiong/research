#include "ksrefine.hpp"
#include "ksint.hpp"
#include "readks.hpp"
#include <iostream>
#include <cmath>
using namespace std;
using namespace Eigen;

int main()
{
  cout.precision(16);
  switch (3) 
    {
    case 1: // test multiF(). For rpo, the shift should be reversed.
      {
	string fileName("../../data/ks22h02t100");
	string ppType("rpo");
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
	
	break;
      }
    
    case 2: // test findPO() and write the data
      {
	const int Nks = 64;
	const int N = Nks - 2;
	string fileName("../../data/ks22h1t120x64");
	string ppType("ppo");
	const int ppId = 1;

	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N, Nks, 22);
	std::tuple<ArrayXd, double, double>
	  pp = readks.readKSorigin(ppType, ppId);	
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	double s = get<2>(pp);

	const int nstp = ceil(floor(T/0.001)/10)*10; cout << nstp << endl;
	KSrefine ksrefine(Nks, 22);
	tuple<VectorXd, double, double> 
	  p = ksrefine.findPO(a, T, nstp, 10, ppType,
			      0.1, 0, 200, 1e-14, true, false);
	cout << get<0>(p) << endl;
	
	KS ks(Nks, get<1>(p), 22);
	ArrayXXd aa = ks.intg(get<0>(p), nstp);
	double r(0);
	if(ppType.compare("ppo") == 0)
	  r = (ks.Reflection(aa.rightCols(1)) - aa.col(0)).matrix().norm();
	else
	  r = (ks.Rotation(aa.rightCols(1), get<2>(p)) - aa.col(0)).matrix().norm();
	
	cout << r << endl;
	readks.writeKSinit("../../data/ks22h001t120x64.h5", ppType, ppId, 
			   make_tuple(get<0>(p), get<1>(p)*nstp, nstp, r, get<2>(p))
			   );

	break;
	  
      }

    case 3: // refine the inital condition for N=32
      {
	const int Nks = 64;
	const int N = Nks - 2;
	string fileName("../../data/ks22h1t120x64");
	string ppType("ppo");
	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N, Nks, 22);
	
	int NN(0);
	if(ppType.compare("ppo")) NN = 840;
	else NN = 834;

	for(int i = 0; i < NN; i++){
	  const int ppId = i+1; 
	  printf("\n****  ppId = %d  ****** \n", ppId); 
	  std::tuple<ArrayXd, double, double>
	    pp = readks.readKSorigin(ppType, ppId);	
	  ArrayXd &a = get<0>(pp); 
	  double T = get<1>(pp);
	  double s = get<2>(pp);

	  const int nstp = ceil(ceil(T/0.001)/10)*10;
	  KSrefine ksrefine(Nks, 22);
	  tuple<VectorXd, double, double> 
	    p = ksrefine.findPO(a, T, nstp, 10, ppType,
				0.1, 0, 100, 1e-14, false, false);
	  KS ks(Nks, get<1>(p), 22);
	  ArrayXXd aa = ks.intg(get<0>(p), nstp);
	  double r(0);
	  if(ppType.compare("ppo") == 0)
	    r = (ks.Reflection(aa.rightCols(1)) - aa.col(0)).matrix().norm();
	  else
	    r = (ks.Rotation(aa.rightCols(1), get<2>(p)) - aa.col(0)).matrix().norm();
	  
	  printf("r = %g\n", r);
	  readks.writeKSinit("../../data/ks22h001t120x64.h5", ppType, ppId, 
			     make_tuple(get<0>(p), get<1>(p)*nstp, nstp, r, get<2>(p))
			     );
	}
	
	break;
      }
  
      
    }
  return 0;
}


