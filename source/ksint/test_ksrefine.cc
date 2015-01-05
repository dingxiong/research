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
  switch (2) 
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
    
    case 2:
      {
	const int Nks = 32;
	const int N = Nks - 2;
	string fileName("../../data/ks22h1t120");
	string ppType("ppo");
	const int ppId = 1;

	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N, Nks, 22);
	std::tuple<ArrayXd, double, double>
	  pp = readks.readKSorigin(ppType, ppId);	
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	double s = get<2>(pp);

	const int nstp = ceil(floor(T/0.02)/10)*10;
	KSrefine ksrefine(Nks, 22);
	VectorXd p = ksrefine.findPO(a, T, nstp, 10, ppType,
				     0.25, 0);
	cout << p << endl;
	
	KS ks(Nks, p(N), 22);
	ArrayXXd aa = ks.intg(p.head(N), nstp);
	cout << (ks.Reflection(aa.rightCols(1)) - aa.col(0)).matrix().norm() << endl;
	
      }
    }
  return 0;
}


