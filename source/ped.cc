#include "ped.hpp"
#include <cmath>
#include <complex>
#include <iostream>
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Tri;

using std::cout; using std::endl;
/*============================================================*
 *            Class : periodic Eigendecomposition             *
 *============================================================*/

/*--------------------  constructor, desctructor --------------- */

/*---------------        member methods          --------------- */

MatrixXd PED::EigVals(MatrixXd &J, const int MaxN /* = 100 */,
		      const double tol/* = 1e-16 */, bool Print /* = true */){ 
  const int N = J.rows();
  pair<MatrixXd, vector<int> > tmp = PerSchur(J, MaxN, tol, Print);
  vector<int> complex_index = tmp.second;
  vector<int> real_index = realIndex(complex_index, N);
  MatrixXd eigVals(N, 2);
  
  // get the real eigenvalues
  for(vector<int>::iterator it = real_index.begin(); it != real_index.end(); it++){
    pair<double, int> tmp = product1dDiag(J, *it);
    eigVals(*it, 0) = tmp.first;
    eigVals(*it, 1) = tmp.second;
  }

  // get the complex eigenvalues
  for(vector<int>::iterator it = complex_index.begin(); it != complex_index.end(); it++){
    pair<Matrix2d, double> tmp = product2dDiag(J, *it);
    pair<Vector2d, Matrix2d> tmp2 = complexEigsMat2(tmp.first);
    
    eigVals(*it, 0) = tmp.second + tmp2.first(0);
    eigVals(*it, 1) = tmp2.first(1);

    eigVals(*it+1, 0) = tmp.second + tmp2.first(0);
    eigVals(*it+1, 1) = -tmp2.first(1);
  }
  return eigVals;
}

/** @brief Periodic Schur decomposition of a sequence of matrix stored in J
 *
 */
pair<MatrixXd, vector<int> > PED::PerSchur(MatrixXd &J, const int MaxN /* = 100 */, 
		       const double tol/* = 1e-16*/, bool Print /* = true */){
  const int N = J.rows();
  MatrixXd Q = HessTrian(J);
  vector<int> cp = PeriodicQR(J, Q, 0, N-1, MaxN, tol, Print);
  
  return make_pair(Q, cp);
}

 
/* @brief transform the matrices stored in J into Hessenberg-upper-triangular form         * 
 * 											   * 
 * Input: J = [J_m, J_{m_1}, ..., J_1] , a sequence of matrices with each of which	   * 
 *        has dimension [n,n], so J has dimension [mn,n]. Note the storage is columnwise   * 
 *        We are interested in the product J_0 = J_m * J_{m-1} *,...,* J_1.		   * 
 * Output: J' = [J'_m, J'_{m_1}, ..., J'_1] in the Hessenberg upper-triangular form.	   * 
 *         J'_m: Hessenber matrix; the others are upper-triangular.			   * 
 *         Q = [Q_m, Q_{m-1}, ..., Q_1], a sequence of orthogonal matrices, which satisfy  *   
 *         Q_i^{T} * J_i * Q_{i-1} = J'_i, with Q_0 = Q_m.                                 *
 *         
 * NOTE : THE TRANSFORM IS IN PLACE.  
 * */
MatrixXd PED::HessTrian(MatrixXd &J){
  const int N = J.rows();
  const int M = J.cols() / N;
  MatrixXd Q = (MatrixXd::Identity(N,N)).replicate(1, M);
  
  for(size_t i = 0; i < N - 1; i++){
    for(size_t j = M-1; j > 0; j--){
      HouseHolder(J.middleCols((j-1)*N, N), J.middleCols(j*N, N), 
		  Q.middleCols(j*N, N), i);
    }
    if(i < N - 2){
      HouseHolder(J.middleCols((M-1)*N, N), J.middleCols(0, N), 
		  Q.middleCols(0, N), i, true);
    }
  }
  
  return Q;
}

/**
 * PeriodicQR transforms an unreduced hessenberg-triangular sequence of
 * matrices into periodic Schur form.
 * This iteration method is based on the Implicit-Q theorem.
 *
 *
 */
vector<int> PED::PeriodicQR(MatrixXd &J, MatrixXd &Q, const int L, const int U,
			    const int MaxN, const double tol, bool Print){
  const int N = J.rows(); 
  const int M = J.cols() / N;
  
  switch(U - L){
    
    /* case 1: [1,1] matrix. No further operation needed.
     *         Just return a empty vector.
     */
  case 0 :
    {
      if(Print) printf("1x1 matrix at L = U = %d \n", L);
      return vector<int>();
    }

    /* case 2: [2,2] matrix. Need to determine whether complex or real
     * if the eigenvalues are complex pairs,no further reduction is needed;
     * otherwise, we need to turn it into diagonal form.
     * */
  case 1 :
    {
      Matrix2d mhess = MatrixXd::Identity(2,2);
      // normalize the matrix to avoid overflow/downflow
      for(size_t i = 0; i < M; i++) {
	mhess *= J.block<2,2>(L, i*N+L); // rows not change, cols increasing.
	mhess = mhess.array() / mhess.cwiseAbs().maxCoeff();
      }
      double del = deltaMat2(mhess);
      // judge the eigenvalues are complex or real number.
      if(del < 0){
	if(Print) printf("2x2 has complex eigenvalues: L = %d, U=%d\n", L, U);
	return vector<int>{L}; // nothing is needed to do.
      }else{      
	Vector2d tmp = vecMat2(mhess);
	GivensOneRound(J, Q, tmp, L);
      }
      // no break here, since the second case need to use iteration too.
    }
  
    
    /* case 3 : subproblme dimension >= 3 or
     *         real eigenvalues with dimension = 2.
     */
  default : 
    {
      vector<int> cp; // vector to store the position of complex eigenvalues.
      size_t np;
      for(np = 0; np < MaxN; np++){
	// Here we define the shift, but right now do not use any shift.
	Vector2d tmp = J.block(L, L, 2, 1);
	GivensOneIter(J, Q, tmp, L, U);
    
	vector<int> zeroID = checkSubdiagZero(J.leftCols(N), L, U, tol);
	vector<int> padZeroID = padEnds(zeroID, L-1, U); // pad L-1 and U at ends.
	const int Nz = padZeroID.size();
	if( Nz > 2 ){

	  //////////////////////////////////////////////////////
	  // print out divide conquer information.
	  if(Print) printf("subproblem L = %d, U = %d uses iteration %zd\n", L, U, np);
	  if(Print){
	    printf("divide position: ");
	    for(vector<int>::iterator it = zeroID.begin(); it != zeroID.end(); it++)
	      printf("%d ", *it);
	    printf("\n");
	  }
	  /////////////////////////////////////////////////////

	  for(size_t i = 0; i < Nz-1; i++){ 	
	    vector<int> tmp = PeriodicQR(J, Q, padZeroID[i]+1, padZeroID[i+1], MaxN, tol, Print);
	    cp.insert(cp.end(), tmp.begin(), tmp.end());
	  }
	  break; // after each subproblem is finished, then problem is solved.
	}
    
      }
      //print out the information if not converged.
      if( np == MaxN){
	printf("**********************\n");
	printf("subproblem L = %d, U = %d does not converge in %d iterations!\n", L, U, MaxN);
	printf("**********************\n");
      }
      return cp;
    }

  }
  
}


/* @brief perform Givens iteration across whole sequence of J.
 * 
 * */
void PED::GivensOneRound(MatrixXd &J, MatrixXd &Q, const Vector2d &v, 
		       const int k){
  const int N = J.rows(); 
  const int M = J.cols() / N;
  
  // Givens rotate the first matrix and the last matrix  
  Givens(J.rightCols(N), J.leftCols(N), Q.leftCols(N), v, k);
  
  // sequence of Givens rotations from the last matrix to the first matrix
  // the structure of first matrix is probabaly destroyed.
  for(size_t i = M-1; i > 0; i--)
    Givens(J.middleCols((i-1)*N, N), J.middleCols(i*N, N), Q.middleCols(i*N, N), k);
}

/* @brief perform periodic QR iteration and restore to the initial form
 * 
 * */
void PED::GivensOneIter(MatrixXd &J, MatrixXd &Q, const Vector2d &v, 
			const int L, const int U){
  // first Given sequence rotation specified by 2x1 vector v
  GivensOneRound(J, Q, v, L);

  // restore the Hessenberg upper triangular form by chasing down the bulge.
  for(size_t i = L; i < U-1; i++){ // cols from L to U-2
    Vector2d tmp = J.block(i+1, i, 2, 1); // take the subdiagonal 2 vector to form Givens.
    GivensOneRound(J, Q, tmp, i+1);
  }
}

/** @brief Givens rotation with provided 2x1 vector as parameter. 
 *
 */
void PED::Givens(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, 
		 const Vector2d &v, const int k){
  double nor = v.norm();
  double c = v(0)/nor;
  double s = v(1)/nor;

  MatrixXd tmp = B.row(k) * c + B.row(k+1) * s;
  B.row(k+1) = -B.row(k) * s + B.row(k+1) * c;
  B.row(k) = tmp;
  
  MatrixXd tmp2 = A.col(k) * c + A.col(k+1) * s;
  A.col(k+1) = -A.col(k) * s + A.col(k+1) * c;
  A.col(k) = tmp2;

  MatrixXd tmp3 = C.col(k) * c + C.col(k+1) * s;
  C.col(k+1) = -C.col(k) * s + C.col(k+1) * c;
  C.col(k) = tmp3;
  
}

/* @brief insert Givens rotation between matrix product A*B.
 * A*G^{T}*G*B
 * G = [c s
 *     -s c]
 * G^{T} = [c -s
 *          s  c]
 * */
void PED::Givens(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C,
		 const int k){
  double nor = sqrt(B(k,k)*B(k,k) + B(k+1,k)*B(k+1,k)); 
  double c = B(k,k)/nor;
  double s = B(k+1,k)/nor;

  // rows k, k+1 of B are transformed
  MatrixXd tmp = B.row(k) * c + B.row(k+1) * s;
  B.row(k+1) = -B.row(k) * s + B.row(k+1) * c;
  B.row(k) = tmp;

  // columns k, k+1 of A are transformed
  MatrixXd tmp2 = A.col(k) * c + A.col(k+1) * s;
  A.col(k+1) = -A.col(k) * s + A.col(k+1) * c;
  A.col(k) = tmp2;
  
  MatrixXd tmp3 = C.col(k) * c + C.col(k+1) * s;
  C.col(k+1) = -C.col(k) * s + C.col(k+1) * c;
  C.col(k) = tmp3;
}

/* @brief store the position where the subdiagonal element is zero.
 *
 * The criteria : J(i+1, i) < 0.5 * tol * (|J(i,i)+J(i+1,i+1)|)
 * */
vector<int> PED::checkSubdiagZero(const Ref<const MatrixXd> &J0,  const int L,
			   const int U, const double tol){
  const int N = J0.rows();
  vector<int> zeroID; // vector to store zero position.
  zeroID.reserve(U - L);
  for(size_t i = L; i < U; i++){
    double SS = ( fabs(J0(i,i)) + fabs(J0(i+1,i+1)) ) * 0.5;
    if(fabs(J0(i+1, i)) < SS * tol) zeroID.push_back(i);
  }

  return zeroID;
}

/** @brief pad two elements at the begining and end of a vector.
 *
 */
vector<int> PED::padEnds(const vector<int> &v, const int &left, const int &right){
  vector<int> vp;
  vp.push_back(left); 
  vp.insert(vp.end(), v.begin(), v.end());
  vp.push_back(right);

  return vp;
}

/** @brief realIndex() gets the positions of the real eigenvelues from the positions
 *         of complex eigenvalues.
 *
 *  Denote the sequence of complex positions: [a_0, a_1,...a_s], then the real
 *  positions between [a_i, a_{i+1}] is from a_i + 2 to a_{i+1} - 1.
 *  Example:
 *          Complex positions :  3, 7, 9
 *          Dimension : N = 12
 *      ==> Real positions : 0, 1, 2, 5, 6, 11   
 */
vector<int> PED::realIndex(const vector<int> &complexIndex, const int N){
  vector<int> padComplexIndex = padEnds(complexIndex, -2, N);
  vector<int> a;
  a.reserve(N);
  for(vector<int>::iterator it = padComplexIndex.begin(); 
      it != padComplexIndex.end()-1; it++){ // note : -1 here.
    for(int i = *it+2; i < *(it+1); i++) {
      a.push_back(i);
    }
  }
  
  return a;
}

/** @brief return the delta of a 2x2 matrix :
 *  for [a, b
 *       c, d]
 *   Delta = (a-d)^2 + 4bc
 */
double PED::deltaMat2(const Matrix2d &A){
  return (A(0,0) - A(1,1)) * (A(0,0) - A(1,1)) + 4 * A(1,0) * A(0,1);
}

/** @brief calculate the eigenvector of a 2x2 matrix which corresponds
 *   to the larger eigenvalue. 
 *  Note: MAKE SURE THAT THE INPUT MATRIX HAS REAL EIGENVALUES.
 */
Vector2d PED::vecMat2(const Matrix2d &A){
  EigenSolver<Matrix2d> eig(A); // eigenvalues must be real.
  Vector2d val = eig.eigenvalues().real();
  Matrix2d vec = eig.eigenvectors().real();
  Vector2d tmp;
  
  // chose the large eigenvalue in the upper place.
  if( val(0) > val(1) ) tmp = vec.col(0); 
  else tmp = vec.col(1);
  
  return tmp;
}

/** @brief get the eigenvalues and eigenvectors of 2x2 matrix
 *
 *  Eigenvalue is stored in exponential way : e^{mu + i*omega}.
 *  Here, omega is guarantted to be positive in [0, PI].
 *  The real and imaginary parts are splitted for eigenvector:
 *  [real(v), imag(v)]
 *  Only one eigenvalue and corresponding eigenvector are return.
 *  The other one is the conjugate.
 *  
 *  Example:
 *         for matrix  [1, -1
 *                      1,  1],
 *         it returns [ 0.346,  
 *                      0.785]
 *                and [-0.707,    0
 *                       0     0.707 ]
 *  Note : make sure that the input matrix has COMPLEX eigenvalues. 
 */
pair<Vector2d, Matrix2d> PED::complexEigsMat2(const Matrix2d &A){
  EigenSolver<Matrix2d> eig(A);
  Vector2cd val = eig.eigenvalues();
  Matrix2cd vec = eig.eigenvectors();

  Vector2d eigvals;
  Matrix2d eigvecs;
  eigvals(0) = log( abs(val(0)) );
  eigvals(1) = fabs( arg(val(0)) );
  int i = 0;
  if(arg(val(0)) < 0) i = 1;
  eigvecs.col(0) = vec.col(i).real();
  eigvecs.col(1) = vec.col(i).imag();

  return make_pair(eigvals, eigvecs);
}

pair<double, int> PED::product1dDiag(const MatrixXd &J, const int k){
  const int N = J.rows();
  const int M = J.cols() / N;
  double logProduct = 0;
  int signProduct = 1;
  for(size_t i = 0; i < M; i++){
    logProduct += log( fabs(J(k, i*N+k)) );
    signProduct *= sgn(J(k, i*N+k));
  }
  
  return make_pair(logProduct, signProduct);
}

pair<Matrix2d, double> PED::product2dDiag(const MatrixXd &J, const int k){
  const int N = J.rows();
  const int M = J.cols() / N;
  double logProduct = 0;
  Matrix2d A = MatrixXd::Identity(2,2);

  for(size_t i = 0; i < M; i++){
    A *= J.block<2,2>(k, i*N+k);
    double norm = A.cwiseAbs().maxCoeff();
    A = A.array() / norm;
    logProduct += log(norm);
  }
  return make_pair(A, logProduct);
}

/* @brief insert Householder transform between matrix product A*B -> A*H*H*B
 * The process also update the orthogonal matrix H : C -> C*H.
 * 
 * Here H is symmetric: H = I - 2vv* / (v*v). v = sign(x_1)||x||e_1 + x
 * A = A -2 (Av) v* / (v*v).
 * B = B - 2v (v*B) / (v*v)
 * Note : A denote the right cols of A, but B denotes the right bottom corner.
 * This process will change A, B and C.
 **/
void PED::HouseHolder(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, 
		      const int k, bool subDiag /* = false */){
  int shift = 0;
  if (subDiag) shift = 1;

  int br = A.rows() - k - shift; // block rows.
  int bc = A.cols() - k;  // block columns.
  VectorXd x = B.block(k + shift, k, br, 1); 
  int sx1 = sgn(x(0)); //sign of x(0)
  VectorXd e1 = VectorXd::Zero(br); e1(0) = 1;
  VectorXd v = sx1 * x.norm() * e1 + x;  
  double vnorm = v.norm(); v /= vnorm;

  A.rightCols(br) = A.rightCols(br) - 
    2 * (A.rightCols(br) * v) * v.transpose();

  C.rightCols(br) = C.rightCols(br) - 
    2 * (C.rightCols(br) * v) * v.transpose();

  B.bottomRightCorner(br, bc) = B.bottomRightCorner(br, bc) - 
    2 * v * (v.transpose() * B.bottomRightCorner(br, bc));
}

/** @brief return the sign of double precision number.
 */
int PED::sgn(const double &num){
  return (0 < num) - (num < 0);
}



/** @brief triDenseMat() creates the triplets of a dense matrix
 *  
 */
vector<Tri> PED::triDenseMat(const Ref<const MatrixXd> &A, const size_t M /* = 0 */, 
			     const size_t N /* = 0 */){
  size_t m = A.rows();
  size_t n = A.cols();

  vector<Tri> tri; 

  tri.reserve(m*n);
  for(size_t i = 0; i < n; i++) // cols
    for(size_t j = 0; j < m; j++) // rows
      tri.push_back( Tri(M+j, N+i, A(j,i) ));

  return tri;
}

/** @brief triDenseMat() creates the triplets of the Kroneck product of
 *         an IxI identity matrix and a dense matrix: Identity x R
 *  
 */
vector<Tri> PED::triDenseMatKron(const size_t I, const Ref<const MatrixXd> &A, 
				 const size_t M /* = 0 */, const size_t N /* = 0 */){
  size_t m = A.rows();
  size_t n = A.cols();
  
  vector<Tri> nz; 
  nz.reserve(m*n*I);
  
  for(size_t i = 0; i < I; i++){
    vector<Tri> tri = triDenseMat(A, M+i*m, N+i*n);
    nz.insert(nz.end(), tri.begin(), tri.end());
  }
  
  return nz;
}

/** @brief triDiagMat() creates the triplets of a diagonal matrix.
 *
 */
vector<Tri> PED::triDiagMat(const size_t n, const double x, 
			    const size_t M /* = 0 */, const size_t N /* = 0 */ ){
  vector<Tri> tri;
  tri.reserve(n);
  for(size_t i = 0; i < n; i++) tri.push_back( Tri(M+i, N+i, x) );
  return tri;
}

/** @brief triDiagMat() creates the triplets of the product
 *         of a matrix and an IxI indentity matrix.
 *
 */
vector<Tri> PED::triDiagMatKron(const size_t I, const Ref<const MatrixXd> &A,
				const size_t M /* = 0 */, const size_t N /* = 0 */ ){
  size_t m = A.rows();
  size_t n = A.cols();
  
  vector<Tri> nz;
  nz.reserve(m*n*I);

  for(size_t i = 0; i < n; i++){
    for(size_t j = 0; j < m; j++){      
      vector<Tri> tri = triDiagMat(I, A(j,i), M+j*n, N+i*n);
      nz.insert(nz.end(), tri.begin(), tri.end());
    }
  }
  
  return nz;
}

/** @brief perSylvester() create the periodic Sylvester sparse matrix and the
 *         dense vector for the reordering algorithm.
 *
 *  @param P the position of the eigenvalue
 *  @param isReal for real eigenvector or complex eigenvector
 */
pair<SpMat, VectorXd> PED::PerSylvester(const MatrixXd &J, const int &P, 
					const bool &isReal, const bool &Print){
  const int N = J.rows();
  const int M = J.cols() / N;
  if(isReal)
    {
      // real case. only need to switch 1x1 matrix on the diagoanl
      if(Print) printf("Forming periodic Sylvester matrix for a real eigenvalue:");
      SpMat per_Sylvester(M*P, M*P);
      VectorXd t12(M*P);
      vector<Tri> nz; nz.reserve(2*M*P*P);
      for(size_t i = 0; i < M; i++){
	if(Print) printf("%zd ", i);
	t12.segment(i*P, P) = -J.block(0, i*N+P, P, 1); // vector -R^{12}
	vector<Tri> triR11 = triDenseMat( J.block(0, i*N, P, P), i*P, i*P );
	vector<Tri> triR22 = triDiagMat(P, -J(P, i*N+P), i*P, ((i+1)%M)*P);
	nz.insert(nz.end(), triR11.begin(), triR11.end());
	nz.insert(nz.end(), triR22.begin(), triR22.end());
      }
      if(Print) printf("\n");
      per_Sylvester.setFromTriplets(nz.begin(), nz.end());
      
      return make_pair(per_Sylvester, t12);
    }  
  else
    {
      // complex case. Need to switch the 2x2 matrix on the diagonal.
      if(Print) printf("Forming periodic Sylvester matrix for a complex eigenvalue:");
      SpMat per_Sylvester(2*M*P, 2*M*P);
      VectorXd t12(2*M*P);
      vector<Tri> nz; nz.reserve(2*2*M*P*P);
      for(size_t i = 0; i < M; i++){
	if(Print) printf("%zd ", i);
	MatrixXd tmp = -J.block(0, i*N+P, P, 2); tmp.resize(2*P,1);
	t12.segment(i*2*P, 2*P) = tmp;
	vector<Tri> triR11 = triDenseMatKron(2, J.block(0, i*N, P, P), i*2*P, i*2*P);
	vector<Tri> triR22 = triDiagMatKron(2, -J.block(P, i*N+P, 2, 2), i*2*P, ((i+1)%M)*2*P);
	nz.insert(nz.end(), triR11.begin(), triR11.end());
	nz.insert(nz.end(), triR22.begin(), triR22.end());
      }      
      if(Print) printf("\n");
      per_Sylvester.setFromTriplets(nz.begin(), nz.end());
      
      return make_pair(per_Sylvester, t12);    
    }


}
/** @brief calculate eigenvector corresponding to the eigenvalue at 
 *         postion P given the Periodic Real Schur Form.
 *         
 */
MatrixXd oneEigVec(const MatrixXd &J, const int &P, 
	       const bool &isReal, const bool &Print){
  const int N = J.rows();
  const int M = J.cols() / N;
  if(isReal)
    {
      MatrixXd ve = MatrixXd::Zero(N, M);
      if(P == 0){
	ve.row(0) = MatrixXd::Identity(1, M);
      } 
      else{
	pair<SpMat, VectorXd> tmp = PerSylvester(J, P, isReal, Print);
	VectorXd x = tmp.first.fullPivLu().solve(tmp.second);
	x.resize(P, M);
	ve.topRows(P) = x; ve.row(P) = MatrixXd::Identity(1, M); 
      }
      ve.resize(N*M,1);
      return ve;
    }
  else
    {
      MatrixXd ve = MatrixXd::Zero(N, 2*M);
      if(P == 0){
      }
      else{
	pair<Matrix2d, double> tmp = product2dDiag(J, P);
	pair<Vector2d, Matrix2d> tmp2 = complexEigsMat2(tmp.first);
	pair<SpMat, VectorXd> tmp3 = PerSylvester(J, P, isReal, Print);
	VectorXd x = tmp3.first.fullPivLu().solve(tmp3.second);
	x.resize(P, 2*M);
      }
      
    }

}
