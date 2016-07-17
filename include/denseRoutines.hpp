/** \mainpage some frequently used dense matrix related routines 
 *
 *  \section sec_intro Introduction
 *  \section sec_use usage
 *  Example:
 *  \code
 *  g++ yourfile.cc /path/to/denseRoutines.cc -I/path/to/sparseRoutines.hpp -I/path/to/eigen -std=c++0x
 *  \endcode
 */

#ifndef DENSEROUTINES_H
#define DENSEROUTINES_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>

namespace denseRoutines {

    using namespace std;
    using namespace Eigen;

    double angleSubspace(const Ref<const MatrixXd> &A,
			 const Ref<const MatrixXd> &B);
    double angleSpaceVector(const Ref<const MatrixXd> &Q,
			    const Ref<const VectorXd> &V);
   
    std::vector< std::pair<double, int> > 
    findMarginal(const Ref<const VectorXd> &Exponent,
		 const int k = 2 );
    MatrixXi 
    indexSubspace(const Ref<const VectorXd> &RCP, 
		  const Ref<const VectorXd> &Exponent);
    std::pair<MatrixXi, MatrixXi> 
    subspBound(const MatrixXi subspDim, const MatrixXi ixSp);

    void normc(MatrixXd &A);
    std::vector<int> csort(const VectorXcd &e);
    VectorXcd eEig(const MatrixXd &A);
    MatrixXcd vEig(const MatrixXd &A);
    std::pair<VectorXcd, MatrixXcd> evEig(const MatrixXd &A);
    VectorXd centerRand(const int N, const double frac);
    MatrixXd realv(const MatrixXcd &v);
    MatrixXd orthAxes(const Ref<const MatrixXd> &v);
    MatrixXd orthAxes(const Ref<const VectorXd> &e1, 
		      const Ref<const VectorXd> &e2);
    MatrixXd orthAxes(const Ref<const VectorXd> &e1, 
		      const Ref<const VectorXd> &e2, 
		      const Ref<const VectorXd> &e3);
    VectorXd spacing(const Ref<const MatrixXd> &v);
    int minDisIndex(const Ref<const VectorXd> &a, 
		    const Ref<const MatrixXd> &v, double &minD);
    int minDisIndex(const Ref<const VectorXd> &a, 
		    const Ref<const MatrixXd> &v);
    std::pair<MatrixXd, MatrixXd>
    GS(const Ref<const MatrixXd> &A);
    MatrixXd
    GSsimple(const Ref<const MatrixXd> &A);

    std::pair<MatrixXd, MatrixXd>
    QR(const Ref<const MatrixXd> &A);
    MatrixXd
    randM(int M, int N);
    MatrixXcd 
    loadComplex(const std::string f1, const std::string f2);
    ArrayXXd 
    calPhase(const Ref<const ArrayXXcd> &AA);
    
    /////////////////////////// template or inline function implementation ////////////////////////////////////////////////////////////////////////////////////

    /** @brief save a matrix into a text file  */
    inline void savetxt(const std::string f, const Ref<const MatrixXd> &A){
	ofstream file(f, ios::trunc);
	file.precision(16);
	file << A << endl;
	file.close();
    }

    /** @brief read data from text file  */
    template<class T = double>
    Matrix<T, Dynamic, Dynamic>
    loadtxt(const std::string f){
	int cols = 0; 
	int rows = 0;
	std::vector<T> buff;
	buff.reserve(1000);
	
	ifstream infile(f);
	assert(!infile.fail());
	while (! infile.eof()) {
	    string line;
	    std::getline(infile, line); 
	
	    int temp_cols = 0;
	    stringstream stream(line); 
	    while(!stream.eof()){
		T x;
		stream >> x; 
		buff.push_back(x);
		temp_cols++;
	    }
	    if (rows == 0) cols = temp_cols;
	    else if (temp_cols != cols) break;

	    rows++;
	}
	infile.close();

	Matrix<T, Dynamic, Dynamic> result(rows, cols);
	for (int i = 0; i < rows; i++)
	    for (int j = 0; j < cols; j++)
		result(i, j) = buff[ cols*i+j];

	return result;
    }
    
    /* @brief create 2d centerized random variables */
    inline MatrixXcd
    center2d(const int M, const int N, const double f1, const double f2){
	MatrixXcd a(M, N);
	a.real() = MatrixXd::Random(M, N)*0.5+0.5*MatrixXd::Ones(M, N);
	a.imag() = MatrixXd::Random(M, N)*0.5+0.5*MatrixXd::Ones(M, N);
	int M2 = (int) (0.5 * M * (1-f1));
	int N2 = (int) (0.5 * N * (1-f2));
	a.topRows(M2) = MatrixXcd::Zero(M2, N);
	a.bottomRows(M2) = MatrixXcd::Zero(M2, N);
	a.leftCols(N2) = MatrixXcd::Zero(M, N2);
	a.rightCols(N2) = MatrixXcd::Zero(M, N2);

	return a;
    }
    
    /** @brief Gaussian profile
     *
     *  $f(x) = a exp(\frac{-(x-b)^2}{2 c^2})$
     */
    inline VectorXcd
    Gaussian(const int N, const int b, const double c, const double a = 1){
	ArrayXd dx = ArrayXd::LinSpaced(N, 0, N-1) - b;
	VectorXd f = (- dx.square() / (2*c*c)).exp() * a;
	return f.cast<std::complex<double>>();	
    }

    /** @brief 2d Gaussian profile
     *
     *  $f(x, y) = a exp(\frac{-(x-b_1)^2}{2 c_1^2} + \frac{-(y-b_2)^2}{2 c_2^2})$
     *  The rectangle size is [M x N] corresponding to y and x direction.
     */
    inline MatrixXcd
    Gaussian2d(const int M, const int N, const int b1, const int b2, const double c1, 
	       const double c2, const double a = 1){
	MatrixXd dx = (VectorXd::LinSpaced(N, 0, N-1)).replicate(1, M).transpose() - MatrixXd::Constant(M, N, b1);
	MatrixXd dy = (VectorXd::LinSpaced(M, 0, M-1)).replicate(1, N) - MatrixXd::Constant(M, N, b2);
	
	MatrixXd d = dx.array().square() / (2*c1*c1) + dy.array().square() / (2*c2*c2);
	MatrixXd f = (-d).array().exp() * a;
	return f.cast<std::complex<double>>();	
    }

    inline MatrixXcd
    soliton(const int M, const int N, const int x, const int y, const double a, const double b){
	
	MatrixXd dx = (VectorXd::LinSpaced(M, 0, M-1)).replicate(1, N) - MatrixXd::Constant(M, N, x);
	MatrixXd dy = (VectorXd::LinSpaced(N, 0, N-1)).replicate(1, M).transpose() - MatrixXd::Constant(M, N, y);
	MatrixXd d = (dx.array() / M / a).square() + (dy.array() / N / b).square();
	
	return (-d).array().exp().cast<std::complex<double>>();
    }

    inline MatrixXcd
    solitons(const int M, const int N, const VectorXd xs, const VectorXd ys,
	     const VectorXd as, const VectorXd bs){
	int n = xs.size();
	MatrixXcd A(MatrixXcd::Zero(M, N));
	for (int i = 0; i < n; i++){
	    A += soliton(M, N, xs(i), ys(i), as(i), bs(i));
	}
	return A;
    }
    
    inline MatrixXcd
    solitonMesh(const int M, const int N, const int nx, const int ny, const double a){
	MatrixXd px = (VectorXd::LinSpaced(nx, M/nx/2, M-1-M/nx/2)).replicate(1, ny);
	MatrixXd py = (VectorXd::LinSpaced(ny, N/ny/2, N-1-N/ny/2)).replicate(1, nx).transpose();
	
	px.resize(nx*ny, 1);
	py.resize(nx*ny, 1);
	
	VectorXd as = VectorXd::Constant(nx*ny, a);

	return solitons(M, N, px, py, as, as);
    }
	
    /** create a matrix with designed eigenvalues.
     *  A V = V E
     */
    inline MatrixXd matE(const VectorXd &e){
	int n = e.size();
	MatrixXd V(MatrixXd::Random(n, n));
	return V * e.asDiagonal() * V.inverse();
    }
    
}

#endif	// DENSEROUTINES_H

