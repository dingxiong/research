#include "kssolveM1.hpp"
#include <iostream>
#include <cstdlib>
#include <iomanip>
using namespace std;
int main(){
  /// -----------------------------------------------------
  int i; 
  int N = 32;

  double h=0.1; 
  double d=22;

  int nstp=2;
  int np=1;
  int nqr =1;
  double a0[N-3]; 

  double *tt = (double *)malloc( ((nstp-1)/np + 1)*sizeof(double) );
  double *aa = (double *)malloc((N-3)*( (nstp-1)/np + 1)*sizeof(double));
  double *daa = (double *)malloc((N-3)*(N-3)*(nstp/nqr)*sizeof(double));
  for(i=0; i<N-3; i++) a0[i]=0.1;
  KsM1 ks(N,d,h);

  /* -------------------- test the integrator ---------- */
  ks.kssolve(a0,nstp,np,aa,tt);
  //ks.kssolve(a0,nstp, np, nqr, aa, daa, tt);
  for(int i = 0; i< 29; i++) cout<<aa[i+29*((nstp-1)/np)]<<endl;
  cout << tt[(nstp-1)/np] << endl;
  int k1=0; int k2=0;

  /* ----------------------------------------------------------- */
  // test the integrator to poincare section
  #if 0
  KsM1::dvec b0(N-3, 0.1);
  KsM1::dvec b1(N-3, 0.2);
  KsM1::dvec v0(ks.velo(b0));
  KsM1::dvec c0 = ks.ks2poinc(b0, b1);
  for(auto i : c0) cout<<i << endl;
  double dd = 0; 
  for(int i = 0; i < b0.size(); i++) dd += v0[i]*(c0[i]-b0[i]);
  cout<<endl<<dd<<endl << endl; 
  KsM1 ks2(N, d, c0[29]/100);
  ks2.kssolve(&b1[0], 101, 1, aa, tt);
  double d2 = 0;
  for(int i = 0; i < 29; i++) d2 += v0[i]*(aa[29*100+i] - b0[i]);
  cout << endl << d2 <<endl << endl;
  for(int i = 0; i < 29; i++) cout << aa[29*100+i] <<endl;
  #endif 
  /* ------------------------------------------------------------ */
  
 
  //  for(int i = 0; i < 30; i++) cout<<daa[900*k1+30*k2+i] <<endl;
  //  for(int i = 29*1000 ; i < 29*1000+29; i++) cout<<aa[i]<<endl;

  //for(int i = 29*29*500 + 29; i < 29*29*500+29*2; i++) cout<<daa[i]<<endl;
    
  free(aa);
  free(daa);
  free(tt);

  return 0;
}
