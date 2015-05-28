/* to comiple:
 * g++ -O3 lyapunov.cc -lcqcgl1d -L../lib -I../include -I/usr/include/eigen3
 * -std=c++0x -lfftw3
 */
#include "cqcgl1d.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/FFT>
#include <complex>
#include <ctime>
#include <H5Cpp.h>

using namespace std; 
using namespace Eigen;
using namespace H5;
typedef std::complex<double> dcp;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Tri;

const int N = 256; 
const double L = 50;

struct KeepDiag {
  inline bool operator() (const int& row, const int& col,
			  const double&) const
  { return row == col;  }
};

vector<Tri> triMat(const MatrixXd &A, const size_t M = 0, const size_t N = 0){
  vector<Tri> tri; 
  size_t m = A.cols();
  size_t n = A.rows();
  tri.reserve(m*n);
  for(size_t j = 0; j < m; j++)
    for(size_t i = 0; i < n; i++)
      tri.push_back( Tri(M+i, N+j, A(i,j) ));

  return tri;
}

vector<Tri> triDiag(const size_t n, const double x, const size_t M = 0, const size_t N = 0 ){
  vector<Tri> tri;
  tri.reserve(n);
  for(size_t i = 0; i < n; i++) tri.push_back( Tri(M+i, N+i, x) );
  return tri;
}

VectorXd multiF(Cqcgl1d &cgl, const ArrayXXd &x, const int nstp, const double th, const double phi){
  int M = x.cols();
  int N = x.rows();
  
  VectorXd F(M*N);
  for(size_t i = 0; i < M; i++){
    ArrayXXd aa = cgl.intg(x.col(i), nstp, nstp);
    if(i < M-1) F.segment(i*N, N) = aa.col(1) - x.col(i+1);
    else F.segment(i*N, N) =  cgl.S1( cgl.S2( aa.col(1), phi ), th )  - x.col((i+1)%M);
  }

  return F;
}

pair<SpMat, VectorXd> multishoot(Cqcgl1d &cgl, const ArrayXXd &x, const int nstp, const double th, const double phi){
  int M = x.cols();
  int N = x.rows();
  
  SpMat DF(M*N, M*N+3);
  VectorXd F(M*N);
  vector<Tri> nz; nz.reserve(2*M*N*N);
  printf("Forming multishooting matrix:");
  for(size_t i = 0 ; i < M; i++){
    printf("%zd ", i);
    Cqcgl1d::CGLaj aj = cgl.intgj(x.col(i), nstp, nstp, nstp); 
    
    Map<MatrixXd> J(&(aj.daa(0,0)), 2*cgl.N, 2*cgl.N);
    if(i < M-1){
      vector<Tri> triJ = triMat(J, i*N, i*N);
      nz.insert(nz.end(), triJ.begin(), triJ.end());
      F.segment(i*N, N) = aj.aa.col(1) - x.col(i+1);
    } else{
      vector<Tri> triJ = triMat(cgl.S1(cgl.S2(J, phi), th), i*N, i*N);     
      nz.insert(nz.end(), triJ.begin(), triJ.end());
      ArrayXd T1 = cgl.TS1( cgl.S1( cgl.S2(aj.aa.rightCols(1), phi), th ) );
      vector<Tri> triT1 = triMat(T1, i*N, M*N+1);
      nz.insert(nz.end(), triT1.begin(), triT1.end());
      ArrayXd T2 = cgl.TS2( cgl.S2( cgl.S1(aj.aa.rightCols(1), th), phi ) );
      vector<Tri> triT2 = triMat(T2, i*N, M*N+2);
      nz.insert(nz.end(), triT2.begin(), triT2.end());
      
      F.segment(i*N, N) = cgl.S1( cgl.S2( aj.aa.col(1), phi ), th )  - x.col((i+1)%M);
    }

    vector<Tri> triI = triDiag(N, -1, i*N, ((i+1)%M)*N);
    nz.insert(nz.end(), triI.begin(), triI.end());

    vector<Tri> triv = triMat(cgl.vel(aj.aa.rightCols(1)), i*N, M*N);
    nz.insert(nz.end(), triv.begin(), triv.end());

  }
  printf("\n");
  
  DF.setFromTriplets(nz.begin(), nz.end());
  

  return make_pair(DF, F);
}

ArrayXd findPO(const ArrayXd &a0, const double h0, const int Norbit, const double th0, const double phi0,
	    const int M, const int MaxN, const double tol){

  const int nstp = Norbit/M; 
  double h = h0;
  double th = th0;
  double phi = phi0;
  double lam = 1;

  Cqcgl1d cgl0(N, L, h);
  ArrayXXd x = cgl0.intg(a0, Norbit, nstp);
  x.conservativeResize(2*N,M); 
  
  ConjugateGradient<SpMat> CG;
  
  for(size_t i = 0; i < MaxN; i++){
    printf("********  i = %zd/%d   ******** \n", i, MaxN);
    Cqcgl1d cgl(N, L, h);
    VectorXd F = multiF(cgl, x, nstp, th, phi);
    double err = F.norm(); 
    if(err < tol){
      printf("stop at norm(F)=%g\n", err);
      break;
    }
   
    pair<SpMat, VectorXd> p = multishoot(cgl, x, nstp, th, phi); 
    SpMat JJ = p.first.transpose() * p.first;
    VectorXd JF = p.first.transpose() * p.second;
    SpMat Dia = JJ; Dia.prune(KeepDiag());
    
    for(size_t j = 0; j < 20; j++){
      printf("inner iteration j = %zd\t", j);
      SpMat H = JJ + lam * Dia; 
      CG.compute(H);     
      VectorXd dF(2*N*M+3); dF = CG.solve(-JF);
      printf("CG error %f, iteration number %d\n", CG.error(), CG.iterations());
      ArrayXXd xnew = x + Map<ArrayXXd>(&dF(0), 2*N, M);
      double hnew = h + dF(2*N*M)/nstp; // be careful here.
      double thnew = th + dF(2*N*M+1);
      double phinew = phi + dF(2*N*M+2);
      printf("\nhnew = %f, thnew = %f, phinew = %f\n", hnew, thnew, phinew);
      
      if( hnew <= 0 ){ printf("new time step is negative\n"); exit(1); }
      Cqcgl1d tcgl(N, L, hnew);
      VectorXd newF = multiF(tcgl, xnew, nstp, thnew, phinew); cout << "err = " << newF.norm() << endl;
      if (newF.norm() < err){
	x = xnew; h = hnew; th = thnew; phi = phinew;
	lam = lam/10; cout << "lam = "<< lam << endl;
	break;
      }
      else{
	lam *= 10; cout << "lam = "<< lam << endl;
	if( lam > 1e10) { printf("lam = %f too large", lam); exit(1); }
      }
      
    }
  }
  
  ArrayXd po(2*N+3);
  po << x.col(0), h, th, phi;
  return po;
}

/****************************************
pair<ArrayXXd, ArrayXd> findRecurrence(const ArrayXd &a0, const double h, 
				       const double tol, const int minN,
				       const int unitN, const int MaxIt){
  ArrayXXd
  Cqcgl1d cgl(N, L, h);
  ArrayXd a = a0;
  for(size_t i = 0; i < MaxIt, i++){
    ArrayXXd aa = cgl.intg(a, unitN, 1); a = aa.rightCols(1);
    for(size_t j = 0; j < unitN; j++)
      for(size_t k = j + minN; k < unitN; k++)
	if( (aa.col(j)-aa.col(k)).norm() < tol)
	  
  }
}
****************************************/

pair<MatrixXd, VectorXd> newtonReq(Cqcgl1d &cgl, const ArrayXd &a0, const double w1, const double w2){
  MatrixXd DF(2*N+2, 2*N+2);
  ArrayXd t1 = cgl.TS1(a0);
  ArrayXd t2 = cgl.TS2(a0);
  DF.topLeftCorner(2*N, 2*N) = cgl.stab(a0) + w1*cgl.GS1() + w2*cgl.GS2();
  DF.col(2*N).head(2*N) = t1;
  DF.col(2*N+1).head(2*N) = t2;
  //DF.row(2*N).head(2*N) = t1.transpose();
  //DF.row(2*N+1).head(2*N) = t2.transpose();
  DF.row(2*N).head(2*N) = VectorXd::Zero(2*N);
  DF.row(2*N+1).head(2*N) = VectorXd::Zero(2*N);
  DF.bottomRightCorner(2,2) = MatrixXd::Zero(2,2);

  VectorXd F(2*N+2);
  F.head(2*N) = cgl.vel(a0) + w1*t1 + w2*t2;
  F(2*N) = 0;
  F(2*N+1) = 0;

  return make_pair(DF, F);
  
}

ArrayXd findReq(const ArrayXd &a0, const double w10, const double w20, const int MaxN, const double tol){

  ArrayXd a = a0;
  double w1 = w10;
  double w2 = w20;
  double lam = 1;
  ConjugateGradient<MatrixXd> CG;
  Cqcgl1d cgl(N, L);
  
  for(size_t i = 0; i < MaxN; i++){
    if (lam > 1e10) break;
    printf("********  i = %zd/%d   ******** \n", i, MaxN);
    VectorXd F = cgl.vel(a) + w1*cgl.TS1(a) + w2*cgl.TS2(a);
    double err = F.norm(); 
    if(err < tol){
      printf("stop at norm(F)=%g\n", err);
      break;
    }
   
    pair<MatrixXd, VectorXd> p = newtonReq(cgl, a, w1, w2); 
    MatrixXd JJ = p.first.transpose() * p.first;
    VectorXd JF = p.first.transpose() * p.second;
    
    for(size_t j = 0; j < 20; j++){
      printf("inner iteration j = %zd\n", j);
      //MatrixXd H = JJ + lam * JJ.diagonal().asDiagonal(); 
      MatrixXd H = JJ; H.diagonal() *= (1+lam);
      CG.compute(H);     
      VectorXd dF = CG.solve(-JF);
      printf("CG error %f, iteration number %d\n", CG.error(), CG.iterations());
      ArrayXd anew = a + dF.head(2*N).array();
      double w1new = w1 + dF(2*N); 
      double w2new = w2 + dF(2*N+1);
      printf("w1new = %f, w2new = %f\n", w1new, w2new);
      
      VectorXd Fnew = cgl.vel(anew) + w1new*cgl.TS1(anew) + w2new*cgl.TS2(anew);
      cout << "err = " << Fnew.norm() << endl;
      if (Fnew.norm() < err){
	a = anew; w1 = w1new; w2 = w2new;
	lam = lam/10; cout << "lam = "<< lam << endl;
	break;
      }
      else{
	lam *= 10; cout << "lam = "<< lam << endl;
	if( lam > 1e10) { printf("lam = %f too large.\n", lam); break; }
      }
      
    }
  }
  
  ArrayXd req(2*N+3);
  VectorXd err = cgl.vel(a) + w1*cgl.TS1(a) + w2*cgl.TS2(a);
  req << a, w1, w2, err.norm(); 
  return req;
}

bool comp(pair<double, int> i, pair<double, int> j){return (i.first > j.first); }

int main(){
  //--------------------------------------------------
#if 0
  Cqcgl1d cgl;
  ArrayXXd x = ArrayXXd::Random(2*N,M);
  pair<SpMat, VectorXd> p = multishoot(cgl, x, 20, 1, 1);
  cout << p.first.rows() << 'x' << p.first.cols() << endl;
  cout << p.first.nonZeros() << endl;

  cout << p.second.rows() << 'x' << p.second.cols() << endl;
  cout << p.first.coeffRef(5120-1,5122) << endl;
#endif
  //--------------------------------------------------
#if 0
  cout.precision(15);

  ArrayXd tmp(2*N+3);
  ifstream fp;
  fp.open("init.bin", ios::binary);
  if(fp.is_open()) fp.read((char*)&tmp(0), (2*N+3)*sizeof(double));
  fp.close();

  ArrayXd a0 = tmp.head(2*N);
  double Norbit = tmp(2*N); cout << Norbit << endl;
  double th0 = tmp(2*N+1);
  double phi0 = tmp(2*N+2);
  
  ArrayXd po = findPO(a0, 0.01, Norbit, th0, phi0, 5, 200, 1e-10);  
  ofstream fo;
  fo.open("po.bin", ios::binary);
  if(fo.is_open()) fo.write((char*)&po(0), (2*N+3)*sizeof(double));
  fo.close();
#endif
  //--------------------------------------------------
  
  ArrayXd tmp(2*N+3);
  
  ifstream fp;
  fp.open("po200.bin", ios::binary);
  if(fp.is_open()) fp.read((char*)&tmp(0), (2*N+3)*sizeof(double));
  fp.close();
  
  ArrayXd a0 = tmp.head(2*N);
  double w1 = 0; //tmp(2*N); 
  double w2 = 17.67; //tmp(2*N+1);
  cout << w1 <<'\t' << w2<< endl;
  ArrayXd req = findReq(a0, w1, w2, 40000, 1e-14);

 // ofstream fo;
 // fo.open("req2.bin", ios::binary);
 // if(fo.is_open()) fo.write((char*)&req(0), (2*N+3)*sizeof(double));
 // fo.close();
  
  Cqcgl1d cgl(N, L);
  MatrixXd travel = cgl.stabReq(req.head(2*N), req(2*N), req(2*N+1));
  EigenSolver<MatrixXd> es;
  es.compute(travel);
  ArrayXcd eigval = es.eigenvalues();
  ArrayXXcd eigvec = es.eigenvectors();
  vector<pair<double, int> > s(2*N);
  for(size_t i = 0; i < 2*N; i++) 
	s[i] = make_pair(eigval(i).real(), i);
  sort(s.begin(), s.end(), comp );
  
  ArrayXcd val(2*N); 
  ArrayXXcd vec(2*N, 2*N);
  for(size_t i = 0; i < 2*N; i++){ 
    val(i) = eigval(s[i].second);
    vec.col(i) = eigvec.col(s[i].second);
  }
  //cout << val << endl;
  //cout <<  eigvec.rows() << 'x' << eigvec.cols() << endl;
  
  ArrayXXd vecr = vec.real();
  ArrayXXd veci = vec.imag();
  ArrayXd valr = val.real();
  ArrayXd vali = val.imag();

  hsize_t dims[2] = {2*N, 2*N};
  DataSpace dataspace(2, dims);
  hsize_t dims2[1] = {2*N};
  DataSpace dsp2(1, dims2);
  hsize_t dims3[1] = {1};
  DataSpace dsp3(1, dims3);

  H5File file("req.h5", H5F_ACC_TRUNC);
  Group group(file.createGroup("/req1"));
  
  DataSet h5_a0 = file.createDataSet("/req1/a0", PredType::NATIVE_DOUBLE, dsp2);
  h5_a0.write(&req(0), PredType::NATIVE_DOUBLE);
  
  DataSet h5_w1 = file.createDataSet("req1/w1",PredType::NATIVE_DOUBLE, dsp3);
  h5_w1.write(&req(2*N), PredType::NATIVE_DOUBLE);

  DataSet h5_w2 = file.createDataSet("req1/w2",PredType::NATIVE_DOUBLE, dsp3);
  h5_w2.write(&req(2*N+1), PredType::NATIVE_DOUBLE);

  DataSet h5_err = file.createDataSet("req1/err",PredType::NATIVE_DOUBLE, dsp3);
  h5_err.write(&req(2*N+2), PredType::NATIVE_DOUBLE);

  DataSet h5_valr = file.createDataSet("/req1/valr", PredType::NATIVE_DOUBLE, dsp2);
  h5_valr.write(&valr(0),  PredType::NATIVE_DOUBLE);

  DataSet h5_vali = file.createDataSet("/req1/vali", PredType::NATIVE_DOUBLE, dsp2);
  h5_vali.write(&vali(0),  PredType::NATIVE_DOUBLE);

  DataSet h5_vecr = file.createDataSet("/req1/vecr", PredType::NATIVE_DOUBLE, dataspace);
  h5_vecr.write(&vecr(0,0), PredType::NATIVE_DOUBLE);

  DataSet h5_veci = file.createDataSet("/req1/veci", PredType::NATIVE_DOUBLE, dataspace);
  h5_veci.write(&veci(0,0), PredType::NATIVE_DOUBLE);

  return 0;
}
