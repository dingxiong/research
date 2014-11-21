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
#include <fstream>
#include <cmath>

using namespace std;
using namespace H5;
using namespace Eigen;

MatrixXd checkExistEV(const string &fileName, const string &ppType, 
		      const int NN){
  H5File file(fileName, H5F_ACC_RDONLY);

  
  MatrixXd status(NN, 2);
  
  for(size_t i = 0; i < NN; i++){
    int ppId = i + 1;
    string DS_e = "/" + ppType + "/" + to_string(ppId) + "/" + "e";
    string DS_ve = "/" + ppType + "/" + to_string(ppId) + "/" + "ve";
    // check the existance of eigenvalues
    try {
      DataSet tmp = file.openDataSet(DS_e);
      status(i,0) = 1;
    }
    catch (FileIException not_found_error) {
      status(i, 0) = 0;
    }
    // check the existance of eigenvectors
    try {
      DataSet tmp = file.openDataSet(DS_ve);
      status(i,1) = 1;
    }
    catch (FileIException not_found_error) {
      status(i, 1) = 0;
    }
  }
  
  return status;
}

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

/** @brief read initial conditons of KS system.
 *
 *  @param[in] fileName hdf5 file which stores the initial conditon
 *                      structure
 *  @param[in] ppType periodic type: ppo/rpo
 *  @param[in] ppId  id of the orbit
 */
std::tuple<ArrayXd, double, double, double, double, MatrixXd, MatrixXd>
readKSve(const string &fileName, const string &ppType,
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

  string DS_e = DS + "e";
  DataSet e = file.openDataSet(DS_e);
  MatrixXd e0(N, 3);
  e.read(&e0(0,0), PredType::NATIVE_DOUBLE);

  string DS_ve = DS + "ve";
  DataSet ve = file.openDataSet(DS_ve);
  MatrixXd ve0(N*N, (int)nstp0);
  ve.read(&ve0(0,0), PredType::NATIVE_DOUBLE);
  
  return make_tuple(a0, T0, nstp0, r0, s0, e0, ve0);
  
}


/** @brief write the Floquet exponents  */
void 
writeKSe(const string &fileName, const string &ppType,
	 const int ppId, const MatrixXd &eigvals, const bool rewrite = false){
  const int N = eigvals.rows();
  const int M = eigvals.cols();

  H5File file(fileName, H5F_ACC_RDWR);
  string DS = "/" + ppType + "/" + to_string(ppId) + "/";
  
  string DS_e = DS + "e";
  hsize_t dims[] = {M, N};
  DataSpace dsp(2, dims);
  DataSet e;
  if(rewrite) e = file.openDataSet(DS_e);
  else e = file.createDataSet(DS_e, PredType::NATIVE_DOUBLE, dsp);
  e.write(&eigvals(0,0), PredType::NATIVE_DOUBLE);
}

/** @brief write the Floquet exponents and Floquet vectors  */
void 
writeKSev(const string &fileName, const string &ppType, const int ppId,
	  const MatrixXd &eigvals, const MatrixXd &eigvecs, const bool rewrite = false){
  const int N = eigvals.rows();
  const int M1 = eigvals.cols();
  const int M2 = eigvecs.cols();

  H5File file(fileName, H5F_ACC_RDWR);
  string DS = "/" + ppType + "/" + to_string(ppId) + "/";
  
  string DS_e = DS + "e";
  hsize_t dims[] = {M1, N};
  DataSpace dsp(2, dims);
  DataSet e;
  if(rewrite) e = file.openDataSet(DS_e);
  else e = file.createDataSet(DS_e, PredType::NATIVE_DOUBLE, dsp);
  e.write(&eigvals(0,0), PredType::NATIVE_DOUBLE);
  
  string DS_ve = DS + "ve";
  hsize_t dims2[] = {M2, N*N};
  DataSpace dsp2(2, dims2);
  DataSet ve;
  if(rewrite) ve = file.openDataSet(DS_ve);
  else ve = file.createDataSet(DS_ve, PredType::NATIVE_DOUBLE, dsp2);
  ve.write(&eigvecs(0,0), PredType::NATIVE_DOUBLE);
}



/** @brief calculate the cos() of the largest angle
 *         between the two subspaces spanned by the
 *         columns of matrices A and B.
 */
double angleSubspace(const Ref<const MatrixXd> &A,
		     const Ref<const MatrixXd> &B){
  assert(A.rows() == B.rows());
  const int N = A.rows();
  const int M1 = A.cols();
  const int M2 = B.cols();
  MatrixXd thinQa(MatrixXd::Identity(N,M1));
  MatrixXd thinQb(MatrixXd::Identity(N,M2));
  
  HouseholderQR<MatrixXd> qra(A);
  thinQa = qra.householderQ() * thinQa;

  HouseholderQR<MatrixXd> qrb(B);
  thinQb = qrb.householderQ() * thinQb;
  
  JacobiSVD<MatrixXd> svd(thinQa.transpose() * thinQb);
  VectorXd sv = svd.singularValues();
  
  return sv.maxCoeff();
}

/** @brief Get the supspace index.
 *
 * For each index, it has a Left value (L) and a Right value (R), which
 * denote when this index is used as the left bound or right bound
 * for a subspace. For real eigenvalues, L = index = R. For complex
 * eigenvalues, L1, L2 = index1, R1, R2 = index1+1; 
 */
MatrixXi indexSubspace(const Ref<const VectorXd> &RCP){
  const int N = RCP.size();
  MatrixXi index_sub(N, 2);
  for(size_t i = 0; i < N; i++){
    if(RCP(i) == 0){
      index_sub(i, 0) = i;
      index_sub(i, 1) = i;
    }
    else{
      index_sub(i, 0) = i; index_sub(i, 1) = i+1;
      index_sub(i+1, 0) = i; index_sub(i+1,1) = i+1;
      i++;
    }
  }
  return index_sub;
}
/** @brief  calculate the actual subspace bound and the indicator whether the actual
 *          bound is the same as specified.
 *
 * @return a pair of integer matrix. The first one is actual subspace bound.
 *         The second one indicate whether the bounds are the same as specified.
 */
std::pair<MatrixXi, MatrixXi> subspBound(const MatrixXi subspDim, const MatrixXi ixSp){
  assert(subspDim.rows() == 4); // two subspaces have 4 indices.
  const int M = subspDim.cols();
  MatrixXi boundStrict(MatrixXi::Zero(4,M));
  MatrixXi bound(4, M);
  
  for(size_t i = 0; i < M; i++){
    int L1 = ixSp(subspDim(0,i), 0); bound(0,i) = L1;
    int R1 = ixSp(subspDim(1,i), 1); bound(1,i) = R1;
    int L2 = ixSp(subspDim(2,i), 0); bound(2,i) = L2;
    int R2 = ixSp(subspDim(3,i), 1); bound(3,i) = R2;
    if(L1 == subspDim(0,i)) boundStrict(0,i) = 1;
    if(R1 == subspDim(1,i)) boundStrict(1,i) = 1;
    if(L2 == subspDim(2,i)) boundStrict(2,i) = 1;
    if(L2 == subspDim(3,i)) boundStrict(3,i) = 1;
  }
  
  return std::make_pair(bound, boundStrict);
  
}

std::pair<MatrixXd, MatrixXi> anglePO(const string fileName, const string ppType,
				      const int ppId, const MatrixXi subspDim){
  assert(subspDim.rows() == 4);
  std::tuple<ArrayXd, double, double, double, double, MatrixXd, MatrixXd>
    pp = readKSve(fileName, ppType, ppId);
  MatrixXd eigVals = get<5>(pp);
  MatrixXd eigVecs = get<6>(pp);
  MatrixXi ixSp = indexSubspace(eigVals.col(2));
  
  const int N = sqrt(eigVecs.rows());
  const int M = eigVecs.cols();
  const int M2 = subspDim.cols();
  MatrixXd ang_po(M, M2);
  
  // calculate the exact bound of this indices.
  std::pair<MatrixXi, MatrixXi> tmp = subspBound(subspDim, ixSp);
  MatrixXi &bound = tmp.first;
  MatrixXi &boundStrict = tmp.second;

  for(size_t i = 0; i < M; i++){
    MatrixXd ve = eigVecs.col(i);
    ve.resize(N, N);
    for(size_t j = 0; j < M2; j++){      
      double ang =  angleSubspace(ve.middleCols(bound(0, j), bound(1,j)-bound(0,j)+1), 
				  ve.middleCols(bound(2, j), bound(3,j)-bound(2,j)+1) );
      ang_po(i, j) = ang;
    }
  }
  
  return std::make_pair(ang_po, boundStrict);
}

int main(){
  cout.precision(16);
  const int Nks = 32;
  const int N = Nks - 2;
  const double L = 22;

  switch (5)
    {
    case 1: // calculate only one orbit and write it.
      {
	string fileName("../data/ks22h02t100120.h5");
	string fileName2("../data/ks22h02t100120E.h5");
	string fileName3("../data/ks22h02t100120EV.h5");
	string ppType("rpo");
	const int ppId = 333;
	const bool rewrite = true;

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
	pair<MatrixXd, MatrixXd> eigv = ped.EigVecs(daa, 3000, 1e-14, false);
	MatrixXd &eigvals = eigv.first; 
	eigvals.col(0) = eigvals.col(0).array()/T;
	MatrixXd &eigvecs = eigv.second;
	
	writeKSe(fileName2, ppType, ppId, eigvals, rewrite);
	writeKSev(fileName3, ppType, ppId, eigvals, eigvecs, rewrite);
	
	break;
      }

    case 2: // calculate all the orbits that converged
      {
	const double tolErr = 1e-8; // tolerance of the error of orbits
	const int MaxN = 8000;  // maximal iteration number for PED
	const double tol = 1e-15; // torlearance for PED
	
	string fileName("../data/ks22h02t100120.h5");
	string fileName2("../data/ks22h02t100120E.h5");
	string fileName3("../data/ks22h02t100120EV.h5");
	string ppType("ppo");
	const bool rewrite = false;

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
  
	    writeKSe(fileName2, ppType, ppId, eigvals, rewrite);
	    writeKSev(fileName3, ppType, ppId, eigvals, eigvecs, rewrite);
	  }
	}
	
	break;
      }

    case 3: // test the existance of e/ve 
      {
	string fileName("../data/ks22h02t100120EV.h5");
	string ppType("rpo");	
	int NN;
	if(ppType.compare("ppo") == 0) NN = 600; // number of ppo
	else NN = 595; // number of rpo
	
	MatrixXd status = checkExistEV(fileName, ppType, NN);
	cout << status << endl;

	break;
      }
      
    case 4: // small test for angle calculation.
      {
	VectorXi subspDim(3); 
	subspDim << 6, 7, 8; // 0-6, 0-7, 0-8;
	string fileName("../data/ks22h02t100120EV.h5");
	string ppType("rpo");	
	int ppId = 1;
	MatrixXd ang = anglePO(fileName, ppType, ppId, subspDim);
	
	ofstream file;
	file.precision(16);
	file.open("good.txt");
	file << ang << endl;
	file.close();

	break;
      }
 
    case 5: // calculate the angle, output to stdout.
      {
	VectorXi subspDim(3); subspDim << 6, 7, 9; // 0-6, 0-7, 0-8;
	string fileName("../data/ks22h02t100120EV.h5");
	string ppType("rpo");	
	int NN;
	if(ppType.compare("ppo") == 0) NN = 600; // number of ppo
	else NN = 595; // number of rpo
	MatrixXd statisAngle;
	
	ofstream file("ang.txt", ios::trunc);
	file.precision(16);
	
	// get the index of POs which converge.
	MatrixXd status = checkExistEV(fileName, ppType, NN);
	for(size_t i = 0; i < NN; i++)
	  {
	    if( 1 == (int)status(i,0) ){
	      int ppId = i + 1;
	      
	      std::tuple<ArrayXd, double, double, double, double, MatrixXd, MatrixXd>
		pp = readKSve(fileName, ppType, ppId);
	      MatrixXd eigVals = get<5>(pp);
	      VectorXi ixSp = indexSubspace(eigVals.col(2));
	      
	      cout << "==========  i = " << i << "   =========" << endl;
	      MatrixXd ang = anglePO(fileName, ppType, ppId, subspDim);
	      file << ang << endl;
	    }
	  }
	file.close();
	
	break;
      }


    default:
      {
	printf("please indicate the index of problem !\n");
      }
    }
  return 0;

}
