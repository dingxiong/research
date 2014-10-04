#include "kssolve.hpp"
#include <iostream>
#include <cstdlib>
using namespace std;
int main(){
  /// -----------------------------------------------------
  int i; 
  int N = 32;

  double h=0.1; 
  double d=22;

  int nstp=10000;
  int np=1000;
  int nqr =1000;
  double a0[N-2]; 

  double *tt = (double *)malloc((nstp/np)*sizeof(double));
  double *aa = (double *)malloc((N-2)*(nstp/np)*sizeof(double));
  double *daa = (double *)malloc((N-2)*(N-2)*(nstp/nqr)*sizeof(double));
  for(i=0; i<N-2; i++) a0[i]=0.1;
  Ks ks;
  ks.kssolve(a0,nstp,np,nqr,aa,daa);
  // ksfM1(a0, d, h, nstp, np, aa, tt);
  
  int k1=0; int k2=0;
  //for(int i = 0; i < 30; i++) cout<<daa[900*k1+30*k2+i] <<endl;
  //for(int i = 30*30*1 ; i < 30*30*1+30; i++) cout<<daa[i]<<endl;
    
  free(aa);
  free(daa);
  free(tt);

  return 0;
}
