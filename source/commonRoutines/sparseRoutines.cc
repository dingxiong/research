#include "sparseRoutines.hpp"

using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Tri;

namespace sparseRoutines {
    
    /* -----------        functions    -------------- */

    /* @brief transform a dense matrix into Triplet form, which should be used to
     *        initialize a spase matrix subblock.
     *
     * @param[in] M row position of the subblock
     * @param[in] N col position of the sbublock
     * @return the triplet representation of the dense matrix
     *
     */
    vector<Tri> 
    triMat(const MatrixXd &A, const size_t M /* = 0 */, 
	   const size_t N /* = 0 */){
	vector<Tri> tri; 
	size_t rows = A.rows(), cols = A.cols();
	tri.reserve(rows*cols);
	// for efficience, loop in the row wise.
	for(size_t j = 0; j < cols; j++)
	    for(size_t i = 0; i < rows; i++)
		tri.push_back( Tri(M+i, N+j, A(i,j) ));

	return tri;
    }


    /* @brief transform a diagoal matrix (diagonal elements are the same) into Triplet
     *        form
     *        
     * @param[in] n size of the diagonal matrix
     * @param[in] x the diagonal elements
     * @param[in] M row position of the subblock
     * @param[in] N col position of the sbublock
     * @see triMat()
     */
    vector<Tri> 
    triDiag(const size_t n, const double x, const size_t M /* = 0 */, 
	    const size_t N /* = 0 */ ){
	vector<Tri> tri;
	tri.reserve(n);
	for(size_t i = 0; i < n; i++) tri.push_back( Tri(M+i, N+i, x) );
	return tri;
    }

    std::vector<Tri>
    triDiag(const VectorXd &D, const size_t M, const size_t N){
	int n = D.size();
	
	vector<Tri> tri;
	tri.reserve(n);
	for(int i = 0; i < n; i++) tri.push_back( Tri(M+i, N+1, D(i)) );

	return tri;

    }
}

