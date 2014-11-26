/* How to compile this program:
 * h5c++ ksCalVec100120.cc ./ksint/ksint.cc ./ped/ped.cc
 * -std=c++0x -I$XDAPPS/eigen/include/eigen3 -I../include/ -lfftw3 -O3 -march=corei7
 * -msse4
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
#include <ctime>

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
 * @param[in] subspDim subspace bounds. Each column is a 4-vector storing dimension of
 *                     two subspaces.
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
    if(R2 == subspDim(3,i)) boundStrict(3,i) = 1;
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

void normc(MatrixXd &A){
  int m = A.cols();
  for(size_t i = 0; i < m; i++) 
    A.col(i).array() = A.col(i).array() / A.col(i).norm();
}

/** @brief calculate the minimal distance between an ergodic
 *         trajectory and one ppo/rpo.
 * @note the cols of ppo/rpo should be the same as its corresponding
 *       Flqouet vectors, otherwise, there maybe index flow when calling
 *       difAngle(). For example, you can do it in the following way:
 *       \code
 *        minDistance(ergodicHat, aaHat.leftCols(aaHat.cols()-1), tolClose)
 *       \endcode
 */
std::tuple<MatrixXd, VectorXi, VectorXi, VectorXd>
minDistance(const MatrixXd &ergodic, const MatrixXd &aa, const double tolClose){
  const int n = ergodic.rows();
  const int m = ergodic.cols();
  const int n2 = aa.rows();
  const int m2 = aa.cols();
  assert(n2 == n);
  
  VectorXi minIndexPo(m);
  VectorXi minIndexErgodic(m);
  VectorXd minDis(m);
  MatrixXd minDifv(n, m);

  size_t tracker = 0;
  for(size_t i = 0; i < m; i++){
    MatrixXd dif = aa.colwise() - ergodic.col(i);// relation is inversed.
    VectorXd colNorm(m2);
    for(size_t j = 0; j < m2; j++) colNorm(j) = dif.col(j).norm();
    int r, c;
    double closest = colNorm.minCoeff(&r, &c); 
    if(closest < tolClose){
      minIndexPo(tracker) = r;
      minIndexErgodic(tracker) = i;
      minDis(tracker) = closest;
      minDifv.col(tracker++) = -dif.col(r);
    }
  }
  return std::make_tuple(minDifv.leftCols(tracker), minIndexErgodic.head(tracker),
			 minIndexPo.head(tracker), minDis.head(tracker));
}

std::pair<std::vector<int>, std::vector<int> >
consecutiveWindow(const VectorXi &index, const int window){
  const int n = index.size();
  std::vector<int> start, dur;

  int pos = 0;
  while (pos < n-1) {
    int span = 0;
    for (size_t i = pos; i < n-1; i++) { // note i < n-1
      if(index(i) == index(i+1)-1) span++;
      else break;
    }
    if(span >= window){
      start.push_back(pos);
      dur.push_back(span);
    }
    if(span > 0) pos += span;
    else pos++;
  }
  return make_pair(start, dur);
}

std::tuple<MatrixXd, VectorXi, VectorXi, VectorXd>
minDistance(const MatrixXd &ergodic, const MatrixXd &aa, const double tolClose, const int window){

  std::tuple<MatrixXd, VectorXi, VectorXi, VectorXd> min_distance = minDistance(ergodic, aa, tolClose); 
  MatrixXd &minDifv = std::get<0>(min_distance); 
  VectorXi &minIndexErgodic = std::get<1>(min_distance);  
  VectorXi &minIndexPo = std::get<2>(min_distance);
  VectorXd &minDis = std::get<3>(min_distance);

  std::pair<std::vector<int>, std::vector<int> > consecutive =
    consecutiveWindow(minIndexErgodic, window);   
  std::vector<int> &start = consecutive.first;
  std::vector<int> &dur = consecutive.second;
  int sum = 0;
  for (std::vector<int>::iterator i = dur.begin(); i != dur.end(); i++) {
    sum += *i;
  }

  MatrixXd minDifv2(minDifv.rows(), sum);
  VectorXi minIndexErgodic2(sum);
  VectorXi minIndexPo2(sum);
  VectorXd minDis2(sum);
  
  size_t pos = 0;
  for(size_t i = 0; i < dur.size(); i++){ 
    minDifv2.middleCols(pos, dur[i]) = minDifv.middleCols(start[i], dur[i]);
    minIndexErgodic2.segment(pos, dur[i]) = minIndexErgodic.segment(start[i], dur[i]);
    minIndexPo2.segment(pos, dur[i]) = minIndexPo.segment(start[i], dur[i]);
    minDis2.segment(pos, dur[i]) = minDis.segment(start[i], dur[i]);

    pos += dur[i];
  }

  return std::make_tuple(minDifv2, minIndexErgodic2, minIndexPo2, minDis2);
}



MatrixXd veTrunc(const MatrixXd ve, const int pos){
  const int N = ve.rows();
  const int M = ve.cols() / N;
  assert(ve.cols()%N == 0);
  
  MatrixXd newVe(N, (N-1)*M);
  for(size_t i = 0; i < M; i++){
    newVe.middleCols(i*(N-1), pos) = ve.middleCols(i*N, pos);
    newVe.middleCols(i*(N-1)+pos, N-1-pos) = ve.middleCols(i*N+pos+1, N-1-pos);
  }
  return newVe;
}

/** @brief calculate the angle between difference vectors and the subspaces spanned
 *  by Flqouet vectors.
 *
 *  @param[in] subsp number of Floquet vectors to span subspace
 *
 *  @note subsp does not stores the indices of the subspace cut. 
 */
MatrixXd difAngle(const MatrixXd &minDifv, const VectorXi &minIx, const VectorXi &subsp, 
		  const MatrixXd &ve_trunc, const int truncN){
  assert(minDifv.cols() == minIx.size());
  const int N = minDifv.rows();
  const int M = minDifv.cols();
  const int M2 = subsp.size();
  
  MatrixXd angle(M2, M);
  for(size_t i = 0; i < M; i++){
    int ix = minIx(i);
    for(size_t j = 0; j < M2; j++)
      // calculate the angle between the different vector and Floquet subspace.
      angle(j, i) = angleSubspace(ve_trunc.middleCols(truncN*ix, subsp(j)),
				  minDifv.col(i));
  }
  return angle;
}

int main(){
  cout.precision(16);
  const int Nks = 32;
  const int N = Nks - 2;
  const double L = 22;

  switch (8)
    {
    case 1: // calculate only one orbit and write it.
      {
	string fileName("../data/ks22h02t100.h5");
	string fileName2("../data/ks22h02t100E.h5");
	string fileName3("../data/ks22h02t100EV.h5");
	string ppType("ppo");
	const int ppId = 173;
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
	pair<MatrixXd, MatrixXd> eigv = ped.EigVecs(daa, 80000, 1e-14, false);
	MatrixXd &eigvals = eigv.first; 
	eigvals.col(0) = eigvals.col(0).array()/T;
	MatrixXd &eigvecs = eigv.second;
	
	writeKSe(fileName2, ppType, ppId, eigvals, rewrite);
	writeKSev(fileName3, ppType, ppId, eigvals, eigvecs, rewrite);
	
	break;
      }

    case 2: // calculate all the orbits that converged for 100 - 120
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

    case 3: // calculate all the orbits that converged for 0-100
      {
	const int MaxN = 10000;  // maximal iteration number for PED
	const double tol = 1e-14; // torlearance for PED
	
	string fileName("../data/ks22h02t100.h5");
	string fileName2("../data/ks22h02t100E.h5");
	string fileName3("../data/ks22h02t100EV.h5");
	string ppType("ppo");
	const bool rewrite = true;

	int NN;
	if(ppType.compare("ppo") == 0) NN = 240; // number of ppo
	else NN = 239; // number of rpo
	
	for(size_t ppId = 1; ppId < NN+1; ppId++){
	  printf("********* ppId = %zd ***********\n", ppId);
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
	  pair<MatrixXd, MatrixXd> eigv = ped.EigVecs(daa, MaxN, tol, false);
	  MatrixXd &eigvals = eigv.first; 
	  eigvals.col(0) = eigvals.col(0).array()/T;
	  MatrixXd &eigvecs = eigv.second;
  
	  writeKSe(fileName2, ppType, ppId, eigvals, rewrite);
	  writeKSev(fileName3, ppType, ppId, eigvals, eigvecs, rewrite);

	}
	
	break;
      }

    case 4: // test the existance of e/ve 
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
      
    case 5: // small test for angle calculation.
      {
	MatrixXi subspDim(4,3); 
	subspDim << 
	  3, 0, 0,
	  3, 7, 8,
	  4, 8, 9,
	  4,29,29; // (0-6,7-29), (0-7,8-29), (0-8,9-29)
	cout << subspDim << endl;
	string fileName("../data/ks22h02t100120EV.h5");
	string ppType("rpo");	
	int ppId = 1;
	std::pair<MatrixXd, MatrixXi> ang = 
	  anglePO(fileName, ppType, ppId, subspDim);
	
	cout << ang.second << endl;
	
	ofstream file;
	file.precision(16);
	file.open("good.txt", ios::trunc);
	file << ang.first << endl;
	file.close();

	break;
      }

    case 6: // calculate the angle, output to files.
	    // This the MAIN experiments I am doing.
      {
	/////////////////////////////////////////////////////////////////
	string fileName("../data/ks22h02t100120EV.h5");
	string ppType("ppo");	
	int NN;
	if(ppType.compare("ppo") == 0) NN = 600; // number of ppo
	else NN = 595; // number of rpo
	/////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////
	MatrixXi subspDim(4,4);
	subspDim << 
	  5,  6,  7,  8,
	  5,  7,  7,  8,
	  6,  8,  8,  9,
	  6,  9,  9,  9; // (0-6,7-29), (0-7,8-29), (0-8,9-29)
	const int M = subspDim.cols();
	ofstream file[M];
	string angName("ang");
	for(size_t i = 0; i < M; i++){
	  file[i].open(angName + to_string(i) + ".txt", ios::trunc);
	  file[i].precision(16);
	}
	/////////////////////////////////////////////////////////////////
	
	/////////////////////////////////////////////////////////////////
	// get the index of POs which converge.
	MatrixXd status = checkExistEV(fileName, ppType, NN);
	for(size_t i = 0; i < NN; i++)
	  {
	    if( 1 == (int)status(i,0) ){
	      int ppId = i + 1;	      
	      cout << "==========  i = " << i << "   =========" << endl;
	      std::pair<MatrixXd, MatrixXi> tmp =
		anglePO(fileName, ppType, ppId, subspDim);
	      MatrixXd &ang = tmp.first;
	      MatrixXi &boundStrict = tmp.second; cout << boundStrict << endl;
	      MatrixXi pro = boundStrict.row(0).array() *  boundStrict.row(1).array() *
		boundStrict.row(2).array() * boundStrict.row(3).array();
	      if(pro(0,0) == 1) file[0] << ang.col(0) << endl;
	      if(pro(0,1) == 1) file[1] << ang.col(1) << endl;
	      if(pro(0,2) == 1) file[2] << ang.col(2) << endl;
	      if(pro(0,3) == 1) file[3] << ang.col(3) << endl;
	      
	      //if(boundStrict(1,0) == 1) file[0] << ang.col(0) << endl;
	      //if(boundStrict(1,1) == 1) file[1] << ang.col(1) << endl;
	      //if(boundStrict(1,2) == 1) file[2] << ang.col(2) << endl;
	      //if(boundStrict(2,3) == 1) file[3] << ang.col(3) << endl;
	    }
	  }
	for(size_t i = 0; i < M; i++) file[i].close();
	/////////////////////////////////////////////////////////////////
	
	break;
      }
      
    case 7 : // small test of the covariant vector projection process. 
      {
	string fileName("../data/ks22h02t100EV.h5");
	string ppType("ppo");
	const int ppId = 1;
	const int gTpos = 3;

	std::tuple<ArrayXd, double, double, double, double, MatrixXd, MatrixXd>
	  pp = readKSve(fileName, ppType, ppId);	
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
  	MatrixXd eigVals = get<5>(pp);
	MatrixXd eigVecs = get<6>(pp);

	KS ks(Nks, T/nstp, L);
	ArrayXXd aa = ks.intg(a, nstp);
	std::pair<MatrixXd, VectorXd> tmp = ks.orbitToSlice(aa); 
	MatrixXd &aaHat = tmp.first; 
	MatrixXd veSlice = ks.veToSliceAll( eigVecs, aa.leftCols(aa.cols()-1) );
	MatrixXd ve_trunc = veTrunc(veSlice, gTpos);
	
	cout << veSlice.middleCols(2,3) << endl << endl;
	cout << veSlice.middleCols(2+30*100,3) << endl << endl;
	cout << ve_trunc.middleCols(2,2) << endl << endl;
	cout << ve_trunc.middleCols(2+29*100,2) << endl << endl;
	
	break;
      }
      
    case 8 : // ergodic orbit approache rpo/ppo
      {
	////////////////////////////////////////////////////////////
	// set up the system
	string fileName("../data/ks22h02t100EV.h5");
	string ppType("ppo");
	const int ppId = 4; 
	const int gTpos = 3; // position of group tangent marginal vector 
	VectorXi subsp(12); subsp << 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 21, 28; // subspace indices.
	////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////
	// prepare orbit, vectors
	std::tuple<ArrayXd, double, double, double, double, MatrixXd, MatrixXd>
	  pp = readKSve(fileName, ppType, ppId);	
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
  	MatrixXd eigVals = get<5>(pp);
	MatrixXd eigVecs = get<6>(pp);
	
	KS ks(Nks, T/nstp, L);
	ArrayXXd aa = ks.intg(a, nstp);
	std::pair<MatrixXd, VectorXd> tmp = ks.orbitToSlice(aa); 
	MatrixXd &aaHat = tmp.first; 
	// note here aa has one more column the the Floquet vectors
	MatrixXd veSlice = ks.veToSliceAll( eigVecs, aa.leftCols(aa.cols()-1) );
	MatrixXd ve_trunc = veTrunc(veSlice, gTpos);
	////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////
	// choose experiment parameters & do experiment
	const double h = 0.1;
	const double sT = 25;
	const double tolClose = 0.1;
	const int MaxSteps = floor(2000/h);
	const int MaxT = 10000;
	KS ks2(Nks, h, L); 
	srand(time(NULL));
	ArrayXd a0(0.1 * ArrayXd::Random(N));
	
	const int fileNum = 5;
	string strf[fileNum] = {"angle_", "dis_", "difv_", "indexPo_", "No_"};
	ofstream saveName[fileNum];	
	for (size_t i = 0; i < fileNum; i++) {
	  saveName[i].open(strf[i] + ppType + to_string(ppId), ios::trunc);
	  saveName[i].precision(16);
	} 

	for(size_t i = 0; i < MaxT; i++){
	  std::cout << "********** i = " << i << "**********"<< std::endl;
	  ArrayXXd ergodic = ks2.intg(a0, MaxSteps); a0 = ergodic.rightCols(1);
	  std::pair<MatrixXd, VectorXd> tmp = ks2.orbitToSlice(ergodic);
	  MatrixXd &ergodicHat = tmp.first;
	  std::tuple<MatrixXd, VectorXi, VectorXi, VectorXd> // be careful about the size of aaHat
	    dis = minDistance(ergodicHat, aaHat.leftCols(aaHat.cols()-1), tolClose, (int)(sT/h) );
	  MatrixXd &minDifv = std::get<0>(dis); 
	  VectorXi &minIndexErgodic = std::get<1>(dis);
	  VectorXi &minIndexPo = std::get<2>(dis); 
	  VectorXd &minDis = std::get<3>(dis);
	  MatrixXd angle = difAngle(minDifv, minIndexPo, subsp, ve_trunc, N-1);
	  if(angle.cols() > 0) printf("angle size = %ld x %ld\n", angle.rows(), angle.cols());

	  if(angle.cols() != 0) {
	    saveName[0] << angle.transpose() << endl;
	    saveName[1] << minDis << std::endl;
	    saveName[2] << minDifv.transpose() << std::endl;
	    saveName[3] << minIndexPo << std::endl;
	    saveName[4] << angle.cols() << std::endl;
	  }
	}
	for (size_t i = 0; i < fileNum; i++)  saveName[i].close();
	////////////////////////////////////////////////////////////
	
	break;
      }
      

      
    default:
      {
	printf("please indicate the index of problem !\n");
      }
    }
  return 0;

}
 
