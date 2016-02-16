/* to compile :
 * g++ test_iterMethod.cc  -O3 -march=corei7 -msse2 -msse4 -I $XDAPPS/eigen/include/eigen3 -std=c++11 -I ../../include -L ../../lib -literMethod -DGMRES_PRINT
 */
#include "iterMethod.hpp"
#include <cmath>
#include <iostream>
#include <time.h>       /* time */
using namespace std;
using namespace Eigen;

typedef Eigen::Triplet<double> Tri;
typedef Eigen::SparseMatrix<double> SpMat; 

int main()
{
    cout.precision(16);
    switch (9)
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

	case 7: // test Gmres() with dense matrix
	    {
		srand (time(NULL));
		const int N  = 10;

		MatrixXd A(N, N);
		VectorXd x(N);

		for (int i = 0; i < N; i++) {
		    for (int j = 0; j < N; j++) {
			A(i,j) = (double)rand() / RAND_MAX - 0.5; 
		    }
		    x(i) =  (double)rand() / RAND_MAX - 0.5; 
		}
		VectorXd b = A * x;
	
		//cout << A << endl << endl;
		cout << x << endl << endl;
		cout << b << endl << endl;

		// compute
		std::tuple<VectorXd, vector<double>, int> 
		    tmp = iterMethod::Gmres<MatrixXd>
		    (A, b, VectorXd::Zero(N), 20, 20, 1e-8);

		cout << std::get<0>(tmp) << endl << endl;
		for (int i = 0; i < std::get<1>(tmp).size(); i++) {
		    cout << std::get<1>(tmp)[i] << endl;
		}
		cout << std::get<2>(tmp) << endl << endl;;
		cout << std::get<1>(tmp).size() << endl << endl;
		cout << (A * std::get<0>(tmp) - b).norm() << endl;
		break;	
	    }

	case 8: // test findTrustRegion
	    {
		srand (time(NULL));
		const int N = 5;
		ArrayXd D(N), p(N);
		for(int i = 0; i < N; i++){
		    /* D(i) = i+1; */
		    /* p(i) = i+1; */
		    /* D(i) = (double)rand() / RAND_MAX - 0.5; */
		    /* p(i) = (double)rand() / RAND_MAX - 0.5; */
		    D(i) = rand() % 100;
		    p(i) = rand() % 100;
		}

		cout << D <<endl << endl;
		cout << p << endl << endl;
		cout << p / D << endl << endl;
		cout << p.matrix().norm() << endl << endl;
		
		auto tmp = iterMethod::findTrustRegion(D, p, 0.1, 1e-4, 20, 0);
		cout << std::get<0>(tmp) << endl;
		cout << std::get<1>(tmp).size() << endl;
		cout << std::get<2>(tmp) << endl;

		auto x = std::get<1>(tmp);
		for_each (x.begin(), x.end(), [](double i){cout << i << endl;});

		ArrayXd z = iterMethod::calz(D, p,  std::get<0>(tmp));
		cout << "z norm " << z.matrix().norm() << endl;
		cout << "Dz-p " << endl << (D*z - p).matrix().norm() << endl;

		break;
	    }

	case 9: // test GmresHook() with dense matrix
	    {
		srand (time(NULL));
		const int N  = 10;

		MatrixXd A(N, N);
		VectorXd x(N);

		for (int i = 0; i < N; i++) {
		    for (int j = 0; j < N; j++) {
			A(i,j) = (double)rand() / RAND_MAX - 0.5; 
		    }
		    x(i) =  (double)rand() / RAND_MAX - 0.5; 
		}
		VectorXd b = A * x;
	
		//cout << A << endl << endl;
		cout << x << endl << endl;
		cout << b << endl << endl;

		// compute
		std::tuple<VectorXd, vector<double>, int> 
		    tmp = iterMethod::GmresHook<MatrixXd>
		    (A, b, VectorXd::Zero(N), 1e-12, 1e-3, 20, 20, 1e-8, 100, 20, false, 1); 

		cout << std::get<0>(tmp) << endl << endl;
		for (int i = 0; i < std::get<1>(tmp).size(); i++) {
		    cout << std::get<1>(tmp)[i] << endl;
		}
		cout << std::get<2>(tmp) << endl << endl;;
		cout << std::get<1>(tmp).size() << endl << endl;
		cout << (A * std::get<0>(tmp) - b).norm() << endl;

		break;	
	    }

	}
    return 0;
}


