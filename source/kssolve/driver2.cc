#include "kssolveM1.hpp"
#include <iostream>
#include <cstdlib>
using namespace std;
int main(){
  /// -----------------------------------------------------
  int i; 
  int N = 32;

  double h=0.1; 
  double d=22;

  int nstp=100;
  int np=1;
  int nqr =1;
  double a0[N-3]; 

  double *tt = (double *)malloc((nstp/np)*sizeof(double));
  double *aa = (double *)malloc((N-3)*(nstp/np)*sizeof(double));
  double *daa = (double *)malloc((N-3)*(N-3)*(nstp/nqr)*sizeof(double));
  for(i=0; i<N-3; i++) a0[i]=0.1;
  KsM1 ks;
  ks.kssolve(a0,nstp,np,aa,tt);
  //ks.kssolve(a0,nstp, np, nqr, aa, daa, tt);
  
  //for(int i = 0; i< 29*2; i++) cout<<aa[i]<<endl;
  KsM1::dvec b0(N-3, 0.2);
  KsM1::dvec c0(N-3, 0.1);
  KsM1::dvec p = ks.ks2poinc(c0, b0);
  
  for(auto &a : p) cout<<a<<endl;
  cout<<p.size()<<endl;
  
  int k1=0; int k2=0;
  //  for(int i = 0; i < 30; i++) cout<<daa[900*k1+30*k2+i] <<endl;
  //  for(int i = 29*1000 ; i < 29*1000+29; i++) cout<<aa[i]<<endl;

  //for(int i = 29*29*500 + 29; i < 29*29*500+29*2; i++) cout<<daa[i]<<endl;
    
  free(aa);
  free(daa);
  free(tt);

  return 0;
}
