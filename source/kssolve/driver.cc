#include "kssolve.hpp"
#include <iostream>
#include <cstdlib>
using namespace std;
int main(){
	
  int i;

  double h=0.1;
  double d=22;

  int nstp=10;
  int np=1;
  int nqr =1;
  double a0[N-3];

  double *tt = (double *)malloc((nstp/np)*sizeof(double));
  double *aa = (double *)malloc((N-3)*(nstp/np)*sizeof(double));
  double *daa = (double *)malloc((N-2)*(N-2)*(nstp/nqr)*sizeof(double));
  for(i=0; i<N-3; i++) a0[i]=0.1;
  //	kssolve(a0,d,h,nstp,np,aa);
  ksfM1(a0, d, h, nstp, np, aa, tt);
  
  int k1=0; int k2=0;
  //for(int i = 0; i < 30; i++) cout<<daa[900*k1+30*k2+i] <<endl;
  for(int i = 0 ;i < 11; i++) cout<<tt[i]<<endl;
    
  free(aa);
  free(daa);

  return 0;
}
