#include "kssolveM1.hpp"
#include "kssym.hpp"
#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <fstream>

using namespace Eigen;
using std::cout; using std::endl;
using std::ofstream; using std::ios;

const double h = 0.1;
const size_t m = 20000; 
const size_t pre = 1000;
const size_t maxp = 5000;
const size_t maxn = 20;

int main(){
  Ks ks(32, 22, h);
  KsM1 ksm1(32, 22, h);
  Kssym sym;
  const int N = ks.N; 
  
  
  KsM1::dvec va0{
    0.431764456067497      ,  
      -0.415487637133904   ,
      -0.157103370964322   ,
      -0.0183007065365757  ,
      -0.486338123063609   ,
      -0.0368836028973648  ,
      -0.0781357949688922  ,
      -0.0745718350746372  ,
      -0.063335888718416   ,
      -0.0105543839724204  ,
      -0.0346044313780501  ,
      -0.00841137576989943 ,
      -0.00449758627360849 ,
      -0.00538505103730609 ,
      -0.00368394792095493 ,
      -0.00113868996750195 ,
      -0.00119372073161254 ,
      -0.0006365646094406  ,
      -0.000125985609026291,
      -0.000262290480940966,
      -0.000108939695856616,
      -6.79162926605918e-05,
      -1.85276435257783e-05,
      -3.10451367009982e-05,
      1.35763401254287e-06 ,
      -1.02295721969699e-05,
      -7.78951504046965e-07,
      -2.44471606268707e-06,
      7.31833967779438e-07           
      };

  VectorXd a0 = Map<VectorXd>(&va0[0], N-3);

  KsM1::dvec vv0 = ksm1.velo(va0);
  VectorXd v0 = Map<MatrixXd>(&vv0[0], N-3, 1);

  MatrixXd aa(N-2, pre);
  VectorXd x0 = 0.1 * VectorXd::Random(N-2);
  
  ks.kssolve(&x0(0), pre, 1, &aa(0,0));
  x0 = aa.rightCols(1); 
  aa.resize(N-2, m);

  MatrixXd poinc(N-3, maxp);
  VectorXd err(maxp);
  int ix = 0;
  
  for(int nn = 0; nn < maxn; nn++){
    cout << "==========  nn = " << nn << "   ============" <<endl;
    ks.kssolve(&x0(0), m, 1, &aa(0,0)); x0 = aa.rightCols(1);
    MatrixXd raa = sym. redSO2(aa, NULL);
    MatrixXd raa2(N-3, m);
    raa2 << raa.topRows(1), raa.middleRows(2, N-4);
    VectorXd uu = (v0.transpose() * raa2).array() - v0.transpose() * a0;
    for(int j = 0; j < m - 1; j++ ){
      
      if(uu(j) < 0 && uu(j+1) >= 0){
	
	KsM1::dvec tmp(N-3);
	Map<MatrixXd>(&tmp[0], N-3, 1) = raa2.middleCols(j,1);
	//KsM1::dvec vvt = ksm1.velo(tmp);
	//VectorXd vt = Map<VectorXd>(&vvt[0], N-3);
	//double dn = v0.transpose() * vt;

	KsM1::dvec ta = ksm1.ks2poinc(va0, tmp);
	VectorXd tta = Map<VectorXd>(&ta[0], N-3);
	
	if((tta-a0).norm() < 1.0 ){
	  err(ix) = *(ta.end()-1);
	  poinc.col(ix++) = tta;
	  if(ix == maxp) break;
	}

      }
    }
  }
  
  cout << ix <<endl;
  
  ofstream fp;
  fp.open("poinc.bin", ios::binary);
  if(fp.is_open()) fp.write((char*) &poinc(0,0), (N-3)* ix *sizeof(double));
  fp.close();

  fp.open("err.bin", ios::binary);
  if(fp.is_open()) fp.write((char*) &poinc(0,0), ix *sizeof(double));
  fp.close();

  return 0;
}
