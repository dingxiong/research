#include "ped.hpp"
#include <cmath>
#include <iostream>
using std::cout; using std::endl;
/*============================================================*
 *            Class : periodic Eigendecomposition             *
 *============================================================*/

/*--------------------  constructor, desctructor --------------- */

/*---------------        member methods          --------------- */

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
  C.col(k) = tmp2;
  
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
  double sx1 = (x(0) > 0) ? 1 : ((x(0) < 0) ? -1 : 0); //sign of x(0)
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

MatrixXd PED::PerSchur(MatrixXd &J, const int MaxN, const double tol,
		  bool Print /* = True */){
  const int N = J.rows();
  MatrixXd Q = HessTrian(J);
  PeriodicQR(J, Q, 0, N-1, MaxN, tol, Print);

  return Q;
}

void PED::PeriodicQR(MatrixXd &J, MatrixXd &Q, const int L, const int U,
		     const int MaxN, const double tol, bool Print){
  
  const int N = J.rows(); 
  const int M = J.cols() / N;
  
  /* case 1: [1,1] matrix. No further operation needed. */
  if( U - L == 0) return;

  /* case 2: [2,2] matrix. Need to determine whether complex or real
   * if the eigenvalues are complex pairs,no further reduction is needed;
   * otherwise, we need to turn it into diagonal form.
   * */
  if(U - L ==1){
    Matrix2d mhess = MatrixXd::Identity(2,2);
    // normalize the matrix to avoid overflow/downflow
    for(size_t i = 0; i < M; i++) {
      mhess *= J.block<2,2>(L, i*N+L); // rows not change, cols increasing.
      mhess = mhess.array() / mhess.cwiseAbs().maxCoeff();
    }
    double del = (mhess(0,0) - mhess(1,1)) * (mhess(0,0) - mhess(1,1))
      + 4 * mhess(1,0) * mhess(0,1);
    // judge the eigenvalues are complex or real number.
    if(del < 0){
      if(Print) printf("2x2 has complex eigenvalues: L = %d, U=%d\n", L, U);
      return; // nothing is needed to do.
    }else{
      
      EigenSolver<Matrix2d> eig(mhess); // eigenvalues must be real.
      Vector2d val = eig.eigenvalues().real(); 
      Matrix2d vec = eig.eigenvectors().real();
      Vector2d tmp;
      
      // chose the large eigenvalue in the upper place.
      if( val(0) > val(1) ) tmp = vec.col(0); 
      else tmp = vec.col(1);
      GivensOneRound(J, Q, tmp, L);
    }
  }
  
  /* case 3: subproblme dimension >= 3 or
   *         real eigenvalues with dimension = 2.
   * */  
  size_t np;
  for(np = 0; np < MaxN; np++){
    // Here we define the shift, but right now do not use any shift.
    Vector2d tmp = J.block(L, L, 2, 1);
    GivensOneIter(J, Q, tmp, L, U);
    
    vector<int> zeroID = checkSubdiagZero(J.leftCols(N), L, U, tol);
    const int Nz = zeroID.size();
    if( Nz > 2 ){
      if(Print) printf("subproblem L = %d, U = %d uses iteration %zd\n", L, U, np);
      for(size_t i = 0; i < Nz-1; i++){ 	
	PeriodicQR(J, Q, zeroID[i]+1, zeroID[i+1], MaxN, tol, Print);
      }
      break; // after each subproblem is finished, then problem is solved.
    }
    
  }
  //print out the information if not converged.
  if( np == MaxN-1)
    printf("subproblem L = %d, U = %d does not converge in %d iterations!\n", L, U, MaxN);
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

/* @brief store the position where the subdiagonal element is zero.
 * pad with L-1 and U at the ends.
 *
 * The criteria : J(i+1, i) < 0.5 * tol * (|J(i,i)+J(i+1,i+1)|)
 * */
vector<int> PED::checkSubdiagZero(const Ref<const MatrixXd> &J0,  const int L,
			   const int U,const double tol){
  const int N = J0.rows();
  vector<int> zeroID; // vector to store zero position.
  zeroID.reserve(U - L + 2);
  zeroID.push_back(L-1);
  for(size_t i = L; i < U; i++){
    double SS = ( fabs(J0(i,i)) + fabs(J0(i+1,i+1)) ) * 0.5;
    if(fabs(J0(i+1, i)) < SS * tol) zeroID.push_back(i);
  }
  zeroID.push_back(U);

  return zeroID;
}
