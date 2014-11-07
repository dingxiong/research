#include "ped.hpp"
#include <cmath>
#include <iostream>
using std::cout; using std::endl;
/*============================================================*
 *            Class : periodic Eigendecomposition             *
 *============================================================*/

/*--------------------  constructor, desctructor --------------- */

/*---------------        member methods          --------------- */

VectorXd PED::EigVals(MatrixXd &J, const int MaxN /* = 100 */,
		      const double tol/* = 1e-16 */, bool Print /* = true */){ 
  const int N = J.rows();
  pair<MatrixXd, vector<int> > tmp = PerSchur(J, MaxN, tol, Print);
  MatrixXd Q = tmp.first;
  vector<int> v = tmp.second;
  
  
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
	  if(Print) printf("subproblem L = %d, U = %d uses iteration %zd\n", L, U, np);
	  for(size_t i = 0; i < Nz-1; i++){ 	
	    vector<int> tmp = PeriodicQR(J, Q, padZeroID[i]+1, padZeroID[i+1], MaxN, tol, Print);
	    cp.insert(cp.end(), tmp.begin(), tmp.end());
	  }
	  break; // after each subproblem is finished, then problem is solved.
	}
    
      }
      //print out the information if not converged.
      if( np == MaxN-1)
	printf("subproblem L = %d, U = %d does not converge in %d iterations!\n", L, U, MaxN);
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
  Matrix2d A = MatrixXd::Identity<2,2>();
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
  return (0 < val) - (val < 0);
}
