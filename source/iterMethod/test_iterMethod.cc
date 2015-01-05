/* to compile :
 * g++ test_iterMethod.cc  -O3 -march=corei7 -msse2 -msse4
 * -I $XDAPPS/eigen/include/eigen3 -std=c++0x -I ../../include
 */
#include "iterMethod.hpp"
#include <cmath>
#include <iostream>
using namespace std;
using namespace Eigen;

typedef Eigen::Triplet<double> Tri;
typedef Eigen::SparseMatrix<double> SpMat; 

int main()
{
  cout.precision(16);
  switch (6)
    {
 
    case 1: // test ConjGrad() with dense matrix
      {
	const int N = 10;
	MatrixXd A(N, N);
	VectorXd b(N);

	for (int i = 0; i < N; i++) {
	  for (int j = 0; j < N; j++) {
	    A(i,j) = (i+j)%11;
	  }
	  b(i) = sin(i*i);
	}
  
	pair<VectorXd, vector<double> > 
	  tmp = iterMethod::ConjGrad<MatrixXd>(A, b, VectorXd::Zero(N), 100, 1e-6);
	cout << tmp.first << endl << endl;
	for (int i = 0; i < tmp.second.size(); i++) {
	  cout << tmp.second[i] << endl;
	}
	cout << tmp.second.size() << endl;

	break;
      }
      
    case 2: //test ConjGrad() with sparse matrix
      {
	const int N  = 10;
	
	vector<Tri> tri;
	for (int i = 0; i < N; i++) {
	  tri.push_back(Tri(i, i, 2));
	  if(i < N-1) {
	    tri.push_back(Tri(i, i+1, -1));
	    tri.push_back(Tri(i+1, i, -1));
	  }
	}
	SpMat A(N, N);
	A.setFromTriplets(tri.begin(), tri.end());
	
	VectorXd b(N);
	for (int i = 0; i < N; i++) b(i) = sin(i*i);  

	// perform CG method 
	pair<VectorXd, vector<double> > 
	  tmp = iterMethod::ConjGrad<SpMat>(A, b, VectorXd::Zero(N), 100, 1e-6);
	cout << tmp.first << endl << endl;
	for (int i = 0; i < tmp.second.size(); i++) {
	  cout << tmp.second[i] << endl;
	}
	cout << tmp.second.size() << endl;
	
      }

    case 3 : // test ConjGradPre() with dense matrix
      {
	// initialization 
	const int N  = 10;
	
	MatrixXd A(N, N);
	for (int i = 0; i < N; i++) {
	  A(i, i) = 2;
	  if(i < N-1) {
	    A(i+1, i) = -1;
	    A(i, i+1) = -1;
	  }
	}

	VectorXd b(N);
	for (int i = 0; i < N; i++) b(i) = sin(i*i);  

	// compute
	MatrixXd M(MatrixXd::Identity(N, N)); cout << A << endl;
	PartialPivLU<MatrixXd> solver;
	pair<VectorXd, vector<double> > 
	  tmp = iterMethod::ConjGradPre<MatrixXd, PartialPivLU<MatrixXd> >
	  (A, b, M, solver, VectorXd::Zero(N), 100, 1e-16);

	cout << tmp.first << endl << endl;
	for (int i = 0; i < tmp.second.size(); i++) {
	  cout << tmp.second[i] << endl;
	}
	cout << tmp.second.size() << endl;
	
	break;
      }
      
    case 4: // test ConjGradPre() with sparse matrix
      {
	const int N  = 10;
	
	vector<Tri> tri;
	for (int i = 0; i < N; i++) {
	  tri.push_back(Tri(i, i, 2));
	  if(i < N-1) {
	    tri.push_back(Tri(i, i+1, -1));
	    tri.push_back(Tri(i+1, i, -1));
	  }
	}
	SpMat A(N, N);
	A.setFromTriplets(tri.begin(), tri.end());
	
	VectorXd b(N);
	for (int i = 0; i < N; i++) b(i) = sin(i*i);  

	// compute
	SpMat M(N, N); 
	vector<Tri> triM; 
	for (int i = 0; i < N; i++) triM.push_back(Tri(i, i, 1));
	M.setFromTriplets(triM.begin(), triM.end());
	cout << M << endl;

	SparseLU<SpMat> solver;
	pair<VectorXd, vector<double> > 
	  tmp = iterMethod::ConjGradPre<SpMat, SparseLU<SpMat> >
	  (A, b, M, solver, VectorXd::Zero(N), 100, 1e-16);

	cout << tmp.first << endl << endl;
	for (int i = 0; i < tmp.second.size(); i++) {
	  cout << tmp.second[i] << endl;
	}
	cout << tmp.second.size() << endl;
	
	break;
	
      }
      
    case 5 : // test SSOR preconditioning
      {
      	const int N  = 10;
	
	vector<Tri> tri;
	for (int i = 0; i < N; i++) {
	  tri.push_back(Tri(i, i, 2));
	  if(i < N-1) {
	    tri.push_back(Tri(i, i+1, -1));
	    tri.push_back(Tri(i+1, i, -1));
	  }
	}
	SpMat A(N, N);
	A.setFromTriplets(tri.begin(), tri.end());

	SpMat ssor = iterMethod::preSSOR<SpMat>(A);
	cout << ssor << endl;

	break;
      }

    case 6: // test ConjGradSSOR() with sparse matrix
      {
	const int N  = 10;
	
	vector<Tri> tri;
	for (int i = 0; i < N; i++) {
	  tri.push_back(Tri(i, i, 2));
	  if(i < N-1) {
	    tri.push_back(Tri(i, i+1, -1));
	    tri.push_back(Tri(i+1, i, -1));
	  }
	}
	SpMat A(N, N);
	A.setFromTriplets(tri.begin(), tri.end());
	
	VectorXd b(N);
	for (int i = 0; i < N; i++) b(i) = sin(i*i);  

	// compute
	SparseLU<SpMat> solver;
	pair<VectorXd, vector<double> > 
	  tmp = iterMethod::ConjGradSSOR<SpMat, SparseLU<SpMat> >
	  (A, b, solver, VectorXd::Zero(N), 100, 1e-16);

	cout << tmp.first << endl << endl;
	for (int i = 0; i < tmp.second.size(); i++) {
	  cout << tmp.second[i] << endl;
	}
	cout << tmp.second.size() << endl;
	
	break;
	
      }
      
    }
  return 0;
}


