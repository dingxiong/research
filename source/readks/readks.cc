#include "readks.hpp"
#include <H5Cpp.h>
#include <algorithm>
#include <vector>
#include <iostream>
using namespace std;
using namespace H5;
using namespace Eigen;

/* ================================================== 
 *                 Class : ReadKS
 *==================================================*/

/* ---------------  constructor/destructor --------------- */
ReadKS::ReadKS(string s1, string s2, string s3, int N /* = 30 */,
	       int Nks /* = 32 */, double L /* = 22*/) 
  : fileName(s1), fileNameE(s2), fileNameEV(s3), N(N), Nks(Nks), L(L) {}
ReadKS::ReadKS(const ReadKS &x) 
  : fileName(x.fileName), fileNameE(x.fileNameE), fileNameEV(x.fileNameEV),
    N(N), Nks(Nks), L(L){}
ReadKS & ReadKS::operator=(const ReadKS &x){
  return *this;
}

ReadKS::~ReadKS(){}

/* ------------    memeber functions    ------------------ */

/** @brief check the existence of Floquet vectors.
 *
 *  Some orbits with 100 < T < 120 fail to converge, so
 *  their Flqouet spectrum is not availble. The function
 *  uses exception handling, though ugly but the only method
 *  I could think of right now.
 *
 *  @param[in] ppType ppo/rpo
 *  @param[in] NN number of orbits need to be investigated
 *  @return N*2 matrix stands for exsitence of
 *          Floquet exponents and Floquet vector. '1' exist, '0' not exist.
 */
MatrixXi ReadKS::checkExistEV(const string &ppType, const int NN){
  H5File file(fileNameEV, H5F_ACC_RDONLY);  
  
  MatrixXi status(NN, 2);
  
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

/** @brief read initial condition from Ruslan's file
 *
 */
std::tuple<ArrayXd, double, double>
ReadKS::readKSorigin(const string &ppType, const int ppId){
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

  double s0(0);
  if(ppType.compare("rpo") == 0){
    string DS_s = DS + "s";
    DataSet s = file.openDataSet(DS_s);
    s.read(&s0, PredType::NATIVE_DOUBLE);    
  }

  return make_tuple(a0, T0, s0);
}

/** @brief read initial conditons of KS system.
 *
 *  For ppo, s = 0.
 *
 *  @param[in] ppType periodic type: ppo/rpo
 *  @param[in] ppId  id of the orbit
 *  @return a, T, nstp, r, s 
 */
std::tuple<ArrayXd, double, double, double, double>
ReadKS::readKSinit(const string &ppType, const int ppId){
  
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

/** @brief rewrite the refined initial condition
 *
 *  Originally, Ruslan's file has a, T, r for ppo and a, T, r, s for rpo.
 *  Now, I refine the initial conditon, so a, T, r, (s) are updated and
 *  nstp is added.
 *
 *  @param[in] ksinit the update data in order: a, T, nstp, r, s
 */
void 
ReadKS::writeKSinit(const string fileName, const string ppType, 
		    const int ppId,
		    const tuple<ArrayXd, double, double, double, double> ksinit
		    ){
  const int N = get<0>(ksinit).rows();

  H5File file(fileName, H5F_ACC_RDWR);
  string DS = "/" + ppType + "/" + to_string(ppId) + "/";

  { // rewrite a
    string DSitem = DS + "a";
    DataSet item = file.openDataSet(DSitem);
    item.write(&(get<0>(ksinit)(0)), PredType::NATIVE_DOUBLE);
  }
  
  { // rewrite T
    string DSitem = DS + "T";
    DataSet item = file.openDataSet(DSitem);
    item.write(&(get<1>(ksinit)), PredType::NATIVE_DOUBLE);
  }

  { // create nstp
    string DSitem = DS + "nstp";
    hsize_t dim[] = {1};
    DataSpace dsp(1, dim);
    DataSet item = file.createDataSet(DSitem, PredType::NATIVE_DOUBLE, dsp);
    item.write(&(get<2>(ksinit)), PredType::NATIVE_DOUBLE);
  }

  { // rewrite r
    string DSitem = DS + "r";
    DataSet item = file.openDataSet(DSitem);
    item.write(&(get<3>(ksinit)), PredType::NATIVE_DOUBLE);
  }

  if(ppType.compare("rpo") == 0) { // rewrite s
    string DSitem = DS + "s";
    DataSet item = file.openDataSet(DSitem);
    item.write(&(get<4>(ksinit)), PredType::NATIVE_DOUBLE);
  }

}
/** @brief read Floquet exponents of KS system.
 *
 *  @param[in] ppType periodic type: ppo/rpo
 *  @param[in] ppId  id of the orbit
 *  @return exponents matrix
 */
MatrixXd
ReadKS::readKSe(const string &ppType, const int ppId){
  // const int N = 30;
  H5File file(fileNameE, H5F_ACC_RDONLY);
  string DS = "/" + ppType + "/" + to_string(ppId) + "/";
  
  string DS_e = DS + "e";
  DataSet e = file.openDataSet(DS_e);
  MatrixXd e0(N, 3);
  e.read(&e0(0,0), PredType::NATIVE_DOUBLE);
  
  return e0;
  
}


/** @brief read Floquet vectors of KS system.
 *
 *  @param[in] ppType periodic type: ppo/rpo
 *  @param[in] ppId  id of the orbit
 *  @return vectors
 */
MatrixXd
ReadKS::readKSve(const string &ppType, const int ppId){
  H5File file(fileNameEV, H5F_ACC_RDONLY);
  string DS = "/" + ppType + "/" + to_string(ppId) + "/";

  string DS_ve = DS + "ve";
  DataSet ve = file.openDataSet(DS_ve);
  DataSpace ds = ve.getSpace();
  // check whether the dimension is 2
  assert(ds.getSimpleExtentNdims() == 2);
  // get the size of each dimension
  hsize_t dims[2];
  int ndims = ds.getSimpleExtentDims(dims, NULL);
  // copy out the vectors. Note that the order is reversed intentionally.
  MatrixXd ve0(dims[1], dims[0]);
  ve.read(&ve0(0,0), PredType::NATIVE_DOUBLE);
  
  return ve0;
  
}


/** @brief write the Floquet exponents
 *
 *  Since HDF5 uses row wise storage, so the Floquet exponents are
 *  stored in row order.
 */
void 
ReadKS::writeKSe(const string &ppType, const int ppId, 
		 const MatrixXd &eigvals, const bool rewrite /* = false */){
  const int N = eigvals.rows();
  const int M = eigvals.cols();

  H5File file(fileNameE, H5F_ACC_RDWR);
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
ReadKS::writeKSev(const string &ppType, const int ppId,
		  const MatrixXd &eigvals, const MatrixXd &eigvecs,
		  const bool rewrite /* = false */){
  const int N1 = eigvals.rows();
  const int M1 = eigvals.cols();
  const int N2 = eigvecs.rows();
  const int M2 = eigvecs.cols();

  H5File file(fileNameEV, H5F_ACC_RDWR);
  string DS = "/" + ppType + "/" + to_string(ppId) + "/";
  
  string DS_e = DS + "e";
  hsize_t dims[] = {M1, N1};
  DataSpace dsp(2, dims);
  DataSet e;
  if(rewrite) e = file.openDataSet(DS_e);
  else e = file.createDataSet(DS_e, PredType::NATIVE_DOUBLE, dsp);
  e.write(&eigvals(0,0), PredType::NATIVE_DOUBLE);
  
  string DS_ve = DS + "ve";
  hsize_t dims2[] = {M2, N2};
  DataSpace dsp2(2, dims2);
  DataSet ve;
  if(rewrite) ve = file.openDataSet(DS_ve);
  else ve = file.createDataSet(DS_ve, PredType::NATIVE_DOUBLE, dsp2);
  ve.write(&eigvecs(0,0), PredType::NATIVE_DOUBLE);
}

/** @brief calculate Floquet exponents and Floquet vectors of KS system.
 *
 *  @param[in] ppType periodic type: ppo/rpo
 *  @param[in] ppId  id of the orbit
 *  @param[in] MaxN maximal number of PED iteration
 *  @param[in] tol tolerance of PED
 *  @param[in] nqr spacing
 *  @return FE and FV
 */
pair<MatrixXd, MatrixXd>
ReadKS::calKSFloquet(const string ppType, const int ppId, 
		     const int MaxN /* = 80000 */,
		     const double tol /* = 1e-15 */,
		     const int nqr /* = 1 */,
		     const int trunc /* = 0 */){
  // get the initla condition of the orbit
  tuple<ArrayXd, double, double, double, double> 
    pp = readKSinit(ppType, ppId);
  ArrayXd &a = get<0>(pp); 
  double T = get<1>(pp);
  int nstp = (int)get<2>(pp);
  double r = get<3>(pp);
  double s = get<4>(pp);
  
  // solve the Jacobian of this po.
  KS ks(Nks, T/nstp, L);
  pair<ArrayXXd, ArrayXXd> tmp = ks.intgj(a, nstp, 1, nqr);
  MatrixXd daa = tmp.second;
  
  // calculate the Flqouet exponents and vectors.
  PED ped;
  ped.reverseOrderSize(daa); // reverse order.
  if(ppType.compare("ppo") == 0)
    daa.leftCols(N) = ks.Reflection(daa.leftCols(N)); // R*J for ppo
  else // R*J for rpo
    daa.leftCols(N) = ks.Rotation(daa.leftCols(N), -s*2*M_PI/L);
  pair<MatrixXd, MatrixXd> eigv = ped.EigVecs(daa, MaxN, tol, false, trunc);
  MatrixXd &eigvals = eigv.first; 
  eigvals.col(0) = eigvals.col(0).array()/T;
  MatrixXd &eigvecs = eigv.second;

  return make_pair(eigvals, eigvecs);
}

MatrixXd
ReadKS::calKSFloquetOnlyE(const string ppType, const int ppId, 
			  const int MaxN /* = 80000 */,
			  const double tol /* = 1e-15 */,
			  const int nqr /* = 1 */){
  // get the initla condition of the orbit
  tuple<ArrayXd, double, double, double, double> 
    pp = readKSinit(ppType, ppId);
  ArrayXd &a = get<0>(pp); 
  double T = get<1>(pp);
  int nstp = (int)get<2>(pp);
  double r = get<3>(pp);
  double s = get<4>(pp);
  
  // solve the Jacobian of this po.
  KS ks(Nks, T/nstp, L);
  pair<ArrayXXd, ArrayXXd> tmp = ks.intgj(a, nstp, 1, nqr);
  MatrixXd daa = tmp.second;
  
  // calculate the Flqouet exponents and vectors.
  PED ped;
  ped.reverseOrderSize(daa); // reverse order.
  if(ppType.compare("ppo") == 0)
    daa.leftCols(N) = ks.Reflection(daa.leftCols(N)); // R*J for ppo
  else // R*J for rpo
    daa.leftCols(N) = ks.Rotation(daa.leftCols(N), -s*2*M_PI/L);
  MatrixXd eigvals = ped.EigVals(daa, MaxN, tol, false);
  eigvals.col(0) = eigvals.col(0).array()/T;

  return eigvals;
}

void 
ReadKS::calKSOneOrbitOnlyE( const string ppType, const int ppId,
			    const int MaxN /* = 80000 */,
			    const double tol /* = 1e-15 */,
			    const bool rewrite /* = false */,
			    const int nqr /* = 1 */){
  MatrixXd eigvals = calKSFloquetOnlyE(ppType, ppId, MaxN, tol, nqr);
  
  writeKSe(ppType, ppId, eigvals, rewrite);
  	
}

void 
ReadKS::calKSOneOrbit( const string ppType, const int ppId,
		       const int MaxN /* = 80000 */,
		       const double tol /* = 1e-15 */,
		       const bool rewrite /* = false */,
		       const int nqr /* = 1 */,
		       const int trunc /* = 0 */){
  pair<MatrixXd, MatrixXd> eigv = calKSFloquet(ppType, ppId, MaxN, tol, nqr, trunc);
  MatrixXd &eigvals = eigv.first;
  MatrixXd &eigvecs = eigv.second; 
  
  writeKSe(ppType, ppId, eigvals, rewrite);
  writeKSev(ppType, ppId, eigvals, eigvecs, rewrite);
	
}

/** @brief find the marginal exponents and their position.
 *
 *  This function tries to find the two smallest absolute
 *  value of the exponents, and assume they are marginal.
 *
 *  @input Exponent Floquet exponent (the frist column of e)
 *  @return a two element vector. Each element is a pair of
 *          the marginal exponent and its corresponding postion
 */
std::vector< std::pair<double, int> > 
ReadKS::findMarginal(const Ref<const VectorXd> &Exponent){
  const int N = Exponent.size();
  std::vector<std::pair<double, int> > var;
  for (size_t i = 0; i < N; i++) {
    var.push_back( std::make_pair(Exponent(i), i) );
  }
  // sort the exponent from large to small by absolute value
  std::sort(var.begin(), var.end(), 
	    [](std::pair<double, int> i , std::pair<double, int> j)
	    {return fabs(i.first) < fabs(j.first);}
	    );
  
  return std::vector<std::pair<double, int> >(var.begin(), var.begin()+2);

}

/** @brief Get the supspace index.
 *
 * For each index, it has a Left value (L) and a Right value (R), which
 * denote when this index is used as the left bound or right bound
 * for a subspace. For real eigenvalues, L = index = R. For complex
 * eigenvalues, L1, L2 = index1, R1, R2 = index1+1;
 *
 * @param[in] RCP real/complex position, usually it is the third
 *                column of the Floquet exponents matrix.
 * @param[in] Exponent Floquet exponents used to get the marginal position
 * @return a Nx2 matrix
 */
MatrixXi 
ReadKS::indexSubspace(const Ref<const VectorXd> &RCP, 
		      const Ref<const VectorXd> &Exponent){
  assert(RCP.size() == Exponent.size());
  const int N = RCP.size();
  std::vector< std::pair<double, int> > 
    marginal = findMarginal(Exponent);

  // create a new real/complex position vector whose marginal
  // will be considered to complex.
  VectorXd RCP2(RCP);
  RCP2(marginal[0].second) = 1.0; RCP2(marginal[1].second) = 1.0;
  MatrixXi index_sub(N, 2);
  for(size_t i = 0; i < N; i++){
    if(RCP2(i) == 0){ // real case
      index_sub(i, 0) = i;
      index_sub(i, 1) = i;
    }
    else{ // complex case
      index_sub(i, 0) = i; index_sub(i, 1) = i+1;
      index_sub(i+1, 0) = i; index_sub(i+1,1) = i+1;
      i++;
    }
  }
  return index_sub;
}

