#include "kssolve.hpp"
#include <iostream>
#include <cstdlib>
using namespace std;
int main(){
	
  int i;

  double h=0.25;
  double d=22;

  int nstp=10;
  int np=1;
  int nqr =1;
  double a0[N-2];
  double *aa = (double *)malloc((N-2)*(nstp/np)*sizeof(double));
  double *daa = (double *)malloc((N-2)*(N-2)*(nstp/nqr)*sizeof(double));
  for(i=0; i<N-2; i++) a0[i]=0.1;
  //	kssolve(a0,d,h,nstp,np,aa);
  ksf(a0, d, h, nstp, np, aa);
  
  int k1=0; int k2=0;
  //for(int i = 0; i < 30; i++) cout<<daa[900*k1+30*k2+i] <<endl;
  for(int i =0 ;i <90; i++) cout<<aa[i]<<endl;
    
  free(aa);
  free(daa);

  return 0;
}
