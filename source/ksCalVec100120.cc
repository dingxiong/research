/* How to compile this program:
 * h5c++ ksCalVec100120.cc ./ksint/ksint.cc ./ped/ped.cc
 * -std=c++0x -I$XDAPPS/eigen/include/eigen3 -I../include/ -lfftw3 -O3 -march=corei7
 */
#include "ksint.hpp"
#include "ped.hpp"
#include <H5Cpp.h>
#include <Eigen/Dense>
#include <tuple>
#include <string>
#include <iostream>
#include <cmath>

using namespace std;
using namespace H5;
using namespace Eigen;

/** @brief read initial conditons of KS system.
 *
 *  @param[in] fileName hdf5 file which stores the initial conditon
 *                      structure
 *  @param[in] ppType periodic type: ppo/rpo
 *  @param[in] ppId  id of the orbit
 */
std::tuple<ArrayXd, double, double, double, double>
readKSinit(const string &fileName, const string &ppType,
       const int ppId){
  const int N = 30;
  H5File file(fileName, H5F_ACC_RDONLY);
  string DS = "/" + ppType + "/" + to_string(ppId) + "/";
  
  string DS_a = DS + "a";
  DataSet a = file.openDataSet(DS_a);
  ArrayXd a0(N);
  a.read(&a0(0), PredType::NATIVE_DOUBLE);

  string DS_T = DS + "T"; 
  DataSet T = file.openDataSet(DS_T);
  double T0(0);
  T.read(&T0, PredType::NATIVE_DOUBLE);

  string DS_nstp = DS + "nstp";
  DataSet nstp = file.openDataSet(DS_nstp);
  double nstp0(0);
  nstp.read(&nstp0, PredType::NATIVE_DOUBLE);

  string DS_r = DS + "r";
  DataSet r = file.openDataSet(DS_r);
  double r0(0);
  r.read(&r0, PredType::NATIVE_DOUBLE);

  double s0(0);
  if(ppType.compare("rpo") == 0){
    string DS_s = DS + "s";
    DataSet s = file.openDataSet(DS_s);
    s.read(&s0, PredType::NATIVE_DOUBLE);    
  }

  return make_tuple(a0, T0, nstp0, r0, s0);
  
}

/** @brief write the Floquet exponents  */
void 
writeKSe(const string &fileName, const string &ppType,
	 const int ppId, const MatrixXd &eigvals){
  const int N = eigvals.rows();
  const int M = eigvals.cols();

  H5File file(fileName, H5F_ACC_RDWR);
  string DS = "/" + ppType + "/" + to_string(ppId) + "/";
  
  string DS_e = DS + "e";
  hsize_t dims[] = {M, N};
  DataSpace dsp(2, dims);
  DataSet e = file.createDataSet(DS_e, PredType::NATIVE_DOUBLE, dsp);
  e.write(&eigvals(0,0), PredType::NATIVE_DOUBLE);
}

/** @brief write the Floquet exponents and Floquet vectors  */
void 
writeKSev(const string &fileName, const string &ppType,
	 const int ppId, const MatrixXd &eigvals, const MatrixXd &eigvecs){
  const int N = eigvals.rows();
  const int M1 = eigvals.cols();
  const int M2 = eigvecs.cols();

  H5File file(fileName, H5F_ACC_RDWR);
  string DS = "/" + ppType + "/" + to_string(ppId) + "/";
  
  string DS_e = DS + "e";
  hsize_t dims[] = {M1, N};
  DataSpace dsp(2, dims);
  DataSet e = file.createDataSet(DS_e, PredType::NATIVE_DOUBLE, dsp);
  e.write(&eigvals(0,0), PredType::NATIVE_DOUBLE);

  string DS_ve = DS + "ve";
  hsize_t dims2[] = {M2, N*N};
  DataSpace dsp2(2, dims2);
  DataSet ve = file.createDataSet(DS_ve, PredType::NATIVE_DOUBLE, dsp2);
  ve.write(&eigvecs(0,0), PredType::NATIVE_DOUBLE);
}

int main(){
  cout.precision(16);
  const int Nks = 32;
  const int N = Nks - 2;
  const double L = 22;

  switch (2)
    {
    case 1: // calculate only one orbit and write it.
      {
	string fileName("../data/ks22h02t100.h5");
	string fileName2("../data/ks22h02t100E.h5");
	string fileName3("../data/ks22h02t100EV.h5");
	string ppType("rpo");
	const int ppId = 1;
	

	tuple<ArrayXd, double, double, double, double> 
	  pp = readKSinit(fileName, ppType, ppId);
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
  
	KS ks(Nks, T/nstp, L);
	pair<ArrayXXd, ArrayXXd> tmp = ks.intgj(a, nstp);
	MatrixXd daa = tmp.second;
	
	PED ped;
	ped.reverseOrderSize(daa); 
	if(ppType.compare("ppo") == 0)
	  daa.leftCols(N) = ks.Reflection(daa.leftCols(N)); // R*J for ppo
	else // R*J for rpo
	  daa.leftCols(N) = ks.Rotation(daa.leftCols(N), -s*2*M_PI/L);
	pair<MatrixXd, MatrixXd> eigv = ped.EigVecs(daa, 5000, 1e-15, false);
	MatrixXd &eigvals = eigv.first; 
	eigvals.col(0) = eigvals.col(0).array()/T;
	MatrixXd &eigvecs = eigv.second;
	
	writeKSe(fileName2, ppType, ppId, eigvals);
	writeKSev(fileName3, ppType, ppId, eigvals, eigvecs);
	
	break;
      }

    case 2: // calculate all the orbits that converged
      {
	const double tolErr = 1e-8; // tolerance of the error of orbits
	const int MaxN = 5000;  // maximal iteration number for PED
	const double tol = 1e-15; // torlearance for PED
	
	string fileName("../data/ks22h02t100120.h5");
	string fileName2("../data/ks22h02t100120E.h5");
	string fileName3("../data/ks22h02t100120EV.h5");
	string ppType("rpo");
	
	int NN;
	if(ppType.compare("ppo") == 0) NN = 600; // number of ppo
	else NN = 595; // number of rpo
	
	for(size_t ppId = 1; ppId < NN+1; ppId++){
	  tuple<ArrayXd, double, double, double, double> 
	    pp = readKSinit(fileName, ppType, ppId);
	  ArrayXd &a = get<0>(pp); 
	  double T = get<1>(pp);
	  int nstp = (int)get<2>(pp);
	  double r = get<3>(pp);
	  double s = get<4>(pp);
	  
	  if (r < tolErr){  
	    printf("********* ppId = %zd ***********\n", ppId);

	    KS ks(Nks, T/nstp, L);
	    pair<ArrayXXd, ArrayXXd> tmp = ks.intgj(a, nstp);
	    MatrixXd daa = tmp.second;
  

	    PED ped;
	    ped.reverseOrderSize(daa); 
	    if(ppType.compare("ppo") == 0)
	      daa.leftCols(N) = ks.Reflection(daa.leftCols(N)); // R*J for ppo
	    else // R*J for rpo
	      daa.leftCols(N) = ks.Rotation(daa.leftCols(N), -s*2*M_PI/L);
	    pair<MatrixXd, MatrixXd> eigv = ped.EigVecs(daa, MaxN, tol, false);
	    MatrixXd &eigvals = eigv.first; 
	    eigvals.col(0) = eigvals.col(0).array()/T;
	    MatrixXd &eigvecs = eigv.second;
  
	    writeKSe(fileName2, ppType, ppId, eigvals);
	    writeKSev(fileName3, ppType, ppId, eigvals, eigvecs);
	  }
	}
	
	break;
      }

    case 3:
      {
	break;
      }
    }
  return 0;

}
