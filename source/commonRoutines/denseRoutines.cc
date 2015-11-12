#include "denseRoutines.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace Eigen;

/* -----------        functions    -------------- */

//////////////////////////////////////////////////////////////////////
//                       subspace angle                             //
//////////////////////////////////////////////////////////////////////

/** @brief calculate the cos() of the largest angle
 *         between the two subspaces spanned by the
 *         columns of matrices A and B.
 */
double denseRoutines::angleSubspace(const Ref<const MatrixXd> &A,
				    const Ref<const MatrixXd> &B){
    assert(A.rows() == B.rows());
    const int N = A.rows();
    const int M1 = A.cols();
    const int M2 = B.cols();
    MatrixXd thinQa(MatrixXd::Identity(N,M1));
    MatrixXd thinQb(MatrixXd::Identity(N,M2));
  
    ColPivHouseholderQR<MatrixXd> qra(A);
    thinQa = qra.householderQ() * thinQa;

    ColPivHouseholderQR<MatrixXd> qrb(B);
    thinQb = qrb.householderQ() * thinQb;
  
    JacobiSVD<MatrixXd> svd(thinQa.transpose() * thinQb);
    VectorXd sv = svd.singularValues();
  
    return sv.maxCoeff();
}


double denseRoutines::angleSpaceVector(const Ref<const MatrixXd> &Q,
				       const Ref<const VectorXd> &V){
    assert( Q.rows() == V.rows());
    VectorXd P = Q.transpose() * V;
    double cos2 = P.squaredNorm() / V.squaredNorm();
    
    return sqrt(1-cos2);
}


//////////////////////////////////////////////////////////////////////
//                   eigenspace related                             //
//////////////////////////////////////////////////////////////////////

/** @brief find the marginal exponents and their position.
 *
 *  This function tries to find the n (default 2)
 *  smallest absolute
 *  value of the exponents, and assume they are marginal.
 *
 *  @input Exponent Floquet exponent (the frist column of e)
 *  @return a n-element vector. Each element is a pair of
 *          the marginal exponent and its corresponding postion
 */
std::vector< std::pair<double, int> > 
denseRoutines::findMarginal(const Ref<const VectorXd> &Exponent,
			    const int k /* = 2 */){
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
  
    return std::vector<std::pair<double, int> >(var.begin(), var.begin() + k);

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
 * @return    N x 2 matrix
 */
MatrixXi 
denseRoutines::indexSubspace(const Ref<const VectorXd> &RCP, 
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


/** @brief  calculate the actual subspace bound and the indicator whether the actual
 *          bound is the same as specified.
 *
 * @param[in] subspDim subspace bounds. Each column is a 4-vector storing dimension of
 *                     two subspaces.
 * @param[in] ixSp subspace index, i.e the left and right bounds
 * @see indexSubspace
 * @return a pair of integer matrix. The first one is actual subspace bound.
 *         The second one indicate whether the bounds are the same as specified.
 */
std::pair<MatrixXi, MatrixXi> 
denseRoutines::subspBound(const MatrixXi subspDim, const MatrixXi ixSp){
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


/**
 * Generate center localized random initial condition
 */
VectorXd denseRoutines::centerRand(const int N, const double frac){
    VectorXd a(VectorXd::Random(N)); /* -1 to 1 */
    int N2 = (int) 0.5 * N * (1-frac);
    a.head(N2) = VectorXd::Zero(N2);
    a.tail(N2) = VectorXd::Zero(N2);
    
    return a;
}


/** @brief normalize each column of a matrix  */
void denseRoutines::normc(MatrixXd &A){
    int m = A.cols();
    for(size_t i = 0; i < m; i++) 
	A.col(i).array() = A.col(i).array() / A.col(i).norm();
}

/**
 * @brief get the sorted indices of a complex vector in descending order
 */
std::vector<int>
denseRoutines::csort(const VectorXcd &e){
    int n = e.size(); 
    std::vector<int> id(n);
    for(int i = 0; i < n; i++) id[i] = i;
    std::sort(id.begin(), id.end(), 
	      [&e](int i, int j){return e(i).real() > e(j).real();}
	      ); 
    return id;
}

/**
 * @brief calculate the eigenvalues of a real matrix.
 *
 * The eigenvalues are sorted by their real part. It returns a complex vector.
 */
VectorXcd denseRoutines::eEig(const MatrixXd &A){
    EigenSolver<MatrixXd> es(A);
    std::vector<int> id = csort(es.eigenvalues());
    int n = id.size();
    VectorXcd re(n);
    for(size_t i = 0; i < n; i++) re(i) = es.eigenvalues()(id[i]);
    return re;
}

/**
 * @brief calculate the eigenvectors of a real matrix. It returns a complex matrix.
 *
 * The eigenvalues are sorted by their real part. It returns a complex matrix.
 */
MatrixXcd denseRoutines::vEig(const MatrixXd &A){
    EigenSolver<MatrixXd> es(A);
    std::vector<int> id = csort(es.eigenvalues());
    int n = id.size();
    MatrixXcd rv(n, n);
    for(size_t i = 0; i < n; i++) rv.col(i) = es.eigenvectors().col(id[i]);
    return rv;
}

/**
 * @brief calculate the eigenvalues and eigenvectors of a real matrix. 
 *
 * The eigenvalues are sorted by their real part. 
 */
std::pair<VectorXcd, MatrixXcd>
denseRoutines::evEig(const MatrixXd &A){
    EigenSolver<MatrixXd> es(A);
    std::vector<int> id = csort(es.eigenvalues());
    int n = id.size();
    VectorXcd re(n);
    MatrixXcd rv(n, n);
    for(size_t i = 0; i < n; i++){
	re(i) = es.eigenvalues()(id[i]);
	rv.col(i) = es.eigenvectors().col(id[i]);
    }
    return make_pair(re, rv);   
}

/**
 * @brief  Reform complex eigenvector matrix to real format
 */
MatrixXd denseRoutines::realv(const MatrixXcd &v){
    int n = v.rows();
    int m = v.cols();
    MatrixXd vp(n, m);

    VectorXd tmp = v.imag().array().abs().colwise().sum();
    for(size_t i = 0; i < m; i++){
	if(tmp(i) < 1e-6) vp.col(i) = v.col(i).real();
	else {
	    vp.col(i) = v.col(i).real();
	    vp.col(i+1) = v.col(i).imag();
	    i++;
	}
    }
    
    return vp;
}

/**
 * @brief construct orthonormal bases from the coloumn vectors of a matrix
 *
 * Note: we do not use ColPivHouseHolderQR because we want to preserve the order.
 */
MatrixXd denseRoutines::orthAxes(const MatrixXd &v){
    int n = v.rows();
    int m = v.cols();
    HouseholderQR<MatrixXd> qr(v);
    return qr.householderQ() * MatrixXd::Identity(n, m);
}

/**
 * @brief construct orthonormal vectors from a 2 vectors
 */
MatrixXd denseRoutines::orthAxes(const VectorXd &e1, const VectorXd &e2){
    int n = e1.size();
    MatrixXd tmp(n, 2);
    tmp << e1, e2;
    return orthAxes(tmp);
}

/**
 * @brief construct orthonormal vectors from a 3 vectors
 */
MatrixXd denseRoutines::orthAxes(const VectorXd &e1, const VectorXd &e2, 
				 const VectorXd &e3){
    int n = e1.size();
    MatrixXd tmp(n, 3);
    tmp << e1, e2, e3;
    return orthAxes(tmp);
}