#include "sparseRoutines.hpp"

using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Tri;
    
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
sparseRoutines::triMat(const MatrixXd &A, const size_t M /* = 0 */, 
				 const size_t N /* = 0 */){
    vector<Tri> tri; 
    size_t m = A.rows();
    size_t n = A.cols();
    tri.reserve(m*n);
    // for efficience, loop in the row wise.
    for(size_t j = 0; j < n; j++)
	for(size_t i = 0; i < m; i++)
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
sparseRoutines::triDiag(const size_t n, const double x, const size_t M /* = 0 */, 
				  const size_t N /* = 0 */ ){
    vector<Tri> tri;
    tri.reserve(n);
    for(size_t i = 0; i < n; i++) tri.push_back( Tri(M+i, N+i, x) );
    return tri;
}
