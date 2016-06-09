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
    template<class T = double>
    Matrix<T, Dynamic, Dynamic>
    center2d(const int M, const int N, const double f1, const double f2, const int flag){
	Matrix<T, Dynamic, Dynamic> a;
	if (1 == flag){
	    a = Matrix<T, Dynamic, Dynamic>::Random(M, N); /* -1 to 1 */
	}
	else if(2 == flag){
	    a = Matrix<T, Dynamic, Dynamic>::Ones(M, N); /* all one */
	}
	int M2 = (int) (0.5 * M * (1-f1));
	int N2 = (int) (0.5 * N * (1-f2));
	a.topRows(M2) = Matrix<T, Dynamic, Dynamic>::Zero(M2, N);
	a.bottomRows(M2) = Matrix<T, Dynamic, Dynamic>::Zero(M2, N);
	a.leftCols(N2) = Matrix<T, Dynamic, Dynamic>::Zero(M, N2);
	a.rightCols(N2) = Matrix<T, Dynamic, Dynamic>::Zero(M, N2);

	return a;
    }
    
}

#endif	// DENSEROUTINES_H

