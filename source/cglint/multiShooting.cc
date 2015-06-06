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
using namespace sparseRoutines;

typedef std::complex<double> dcp;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Tri;

const int N = 256; 
const double L = 50;


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
	    printf("\nhnew = %g, thnew = %g, phinew = %f\n", hnew, thnew, phinew);
      
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
    po << x.col(0), h, th, phi;ve
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

bool comp(pair<double, int> i, pair<double, int> j){return (i.first > j.first); }

int main(){
   
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
