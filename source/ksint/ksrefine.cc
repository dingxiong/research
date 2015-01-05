#include "ksrefine.hpp"
#include "iterMethod.hpp"
#include <iostream>
#include <fstream>

using namespace std; 
using namespace Eigen;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Tri;

////////////////////////////////////////////////////////////
//                     class KSrefine                     //
////////////////////////////////////////////////////////////

/*------------          constructor        ---------------*/
KSrefine::KSrefine(int N /* = 32 */, double L /* = 22 */) : N(N), L(L) {}
KSrefine::KSrefine(const KSrefine &x) : N(x.N), L(x.L) {}
KSrefine & KSrefine::operator=(const KSrefine &x){ return *this;}
KSrefine::~KSrefine(){}

/* -----------         member functions    -------------- */
struct KeepDiag {
  inline bool operator() (const int& row, const int& col,
			  const double&) const
  { return row == col;  }
};

/* @brief transform a dense matrix into Triplet form, which should be used to
 *        initialize a spase matrix subblock.
 *
 * @param[in] M row position of the subblock
 * @param[in] N col position of the sbublock
 * @return the triplet representation of the dense matrix
 *
 */
vector<Tri> 
KSrefine::triMat(const MatrixXd &A, const size_t M /* = 0 */, 
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
KSrefine::triDiag(const size_t n, const double x, const size_t M /* = 0 */, 
	const size_t N /* = 0 */ ){
  vector<Tri> tri;
  tri.reserve(n);
  for(size_t i = 0; i < n; i++) tri.push_back( Tri(M+i, N+i, x) );
  return tri;
}

/* @brief form the multishooting difference vector
 *        [ f(x_0)       - x_1,
 *          f(x_1)       - x_2,
 *          ...
 *          R*f(x_{m-1}) - x_0]
 *        Here R is rotation for rpo, reflection for ppo.
 *
 * @param[in] ks KS object used to integrate the system
 * @param[in] x [x_0, x_1, ..., x_{n-1}] state vectors
 * @param[in] nstp integratin steps
 * @param[in] ppType "ppo" or "rpo"
 * @param[in] th rotation angle for rpo. It is not used for ppo.
 */

VectorXd 
KSrefine::multiF(KS &ks, const ArrayXXd &x, const int nstp, 
		 const string ppType, const double th /* = 0.0 */){
  int n = x.rows();
  int m = x.cols();
  VectorXd F(m*n);

  // form the first m - 1 difference vectors
  for(size_t i = 0; i < m - 1; i++){
    ArrayXXd aa = ks.intg(x.col(i), nstp, nstp);
    F.segment(i*n, n) = aa.col(1) - x.col(i+1);
  }

  // the last difference vector
  ArrayXXd aa = ks.intg(x.col(m-1), nstp, nstp);
  if(ppType.compare("ppo") == 0)
    F.segment((m-1)*n, n) = ks.Reflection(aa.col(1)) - x.col(0);
  else if(ppType.compare("rpo") == 0)
    F.segment((m-1)*n, n) = ks.Rotation(aa.col(1), th)  - x.col(0);
  else
    fprintf(stderr, "please indicate the right PO type !\n");
  
  return F;
}

/** @brief calculate the column size of the multishooting matrix
 *
 */
int
KSrefine::calColSizeDF(const int &rows, const int &cols, const string ppType){
  int colsizeDF(0);
  
  if(ppType.compare("ppo") == 0) colsizeDF = rows*cols+1;
  else if(ppType.compare("rpo") == 0) colsizeDF = rows*cols+2;
  else fprintf(stderr, "please indicate the right PO type !\n");

  return colsizeDF;
}

/* @brief not only form the difference vector but also the multishooting matrix
 *
 * @return multishooting matrix and difference vector
 * @see multiF()
 */
pair<SpMat, VectorXd> 
KSrefine::multishoot(KS &ks, const ArrayXXd &x, const int nstp, 
		     const string ppType, const double th /* = 0.0 */,
		     const bool Print /* = false */){
  int n = x.rows();
  int m = x.cols();
  
  int colsizeDF = calColSizeDF(n, m, ppType); 
  SpMat DF(m*n, colsizeDF);
  VectorXd F(m*n);
  vector<Tri> nz; 
  nz.reserve(2*m*n*n+m*n);
  if(Print) printf("Forming multishooting matrix:");

  for(size_t i = 0 ; i < m; i++){
    if(Print) printf("%zd ", i);
    pair<ArrayXXd, ArrayXXd> tmp = ks.intgj(x.col(i), nstp, nstp, nstp); 
    ArrayXXd &aa = tmp.first;
    ArrayXXd &J = tmp.second;
    // assert(J.rows() == n*n && J.cols() == 1);
    J.resize(n, n);
    
    if(i < m-1)
      {
	// J
	vector<Tri> triJ = triMat(J, i*n, i*n);
	nz.insert(nz.end(), triJ.begin(), triJ.end());
	// velocity
	vector<Tri> triv = triMat(ks.velocity(aa.col(1)), i*n, m*n);
	nz.insert(nz.end(), triv.begin(), triv.end());
	// f(x_i) - x_{i+1}
	F.segment(i*n, n) = aa.col(1) - x.col(i+1);
      } 
    else
      {
	if(ppType.compare("ppo") == 0){
	  // R*J
	  vector<Tri> triJ = triMat(ks.Reflection(J), i*n, i*n);  
	  nz.insert(nz.end(), triJ.begin(), triJ.end());
	  // R*velocity
	  vector<Tri> triv = 
	    triMat(ks.Reflection(ks.velocity(aa.col(1))), i*n, m*n);
	  nz.insert(nz.end(), triv.begin(), triv.end());
	  // R*f(x_{m-1}) - x_0
	  F.segment(i*n, n) = ks.Reflection(aa.col(1)) - x.col(0);
	} else if (ppType.compare("rpo") == 0){
	  // g*J
	  vector<Tri> triJ = triMat(ks.Rotation(J, th), i*n, i*n);
	  nz.insert(nz.end(), triJ.begin(), triJ.end());  
	  // R*velocity
	  vector<Tri> triv = 
	    triMat(ks.Rotation(ks.velocity(aa.col(1)), th), i*n, m*n);
	  nz.insert(nz.end(), triv.begin(), triv.end());
	  // T*g*f(x_{m-1})
	  VectorXd tx = ks.gTangent( ks.Rotation(aa.col(1), th) );
	  vector<Tri> tritx = triMat(tx, i*n, m*n+1);
	  nz.insert(nz.end(), tritx.begin(), tritx.end());
	  // g*f(x_{m-1}) - x_0
	  F.segment(i*n, n) = ks.Rotation(aa.col(1), th) - x.col(0);
	} else {
	  fprintf(stderr, "please indicate the right PO type !\n");
	}

      }
    
    // -I on subdiagonal
    vector<Tri> triI = triDiag(n, -1, i*n, ((i+1)%m)*n);
    nz.insert(nz.end(), triI.begin(), triI.end());

  }
  if(Print) printf("\n");
  
  DF.setFromTriplets(nz.begin(), nz.end());

  return make_pair(DF, F);
}


/** @brief find ppo/rpo in KS system
 *
 * @param[in] a0 guess of initial condition
 * @param[in] T guess of period
 * @param[in] h0 guess of time step
 * @param[in] Norbit number of pieces of the orbit
 * @param[in] M number of piece for multishooting method
 * @param[in] ppType ppo/rpo
 * @param[in] th0 guess of initial group angle. For ppo, it is zero.
 * @param[in] hinit: the initial time step to get a good starting guess.
 * @param[in] MaxN maximal number of iteration
 * @param[in] tol tolerance of convergence
 *
 */
tuple<VectorXd, double, double>
KSrefine::findPO(const ArrayXd &a0, const double T, const int Norbit, 
		 const int M, const string ppType,
		 const double hinit /* = 0.1*/,
		 const double th0 /* = 0 */,
		 const int MaxN /* = 100 */, 
		 const double tol /* = 1e-14*/, 
		 const bool Print /* = false */,
		 const bool isSingle /* = false */){ 
  bool Terminate = false;
  assert(a0.rows() == N - 2 && Norbit % M == 0);
  const int nstp = Norbit/M;
  double h = T/Norbit;
  double th = th0;
  double lam = 1;
  SparseLU<SpMat> solver; // used in the pre-CG method

  // prepare the initial state sequence
  KS ks0(N, hinit, L);
  ArrayXXd x = ks0.intg(a0, (int)ceil(T/hinit), 
			(int)floor(T/hinit/M));
  x = x.leftCols(M); 
  
  for(size_t i = 0; i < MaxN; i++){
    if(Print && i%10 == 0) printf("******  i = %zd/%d  ****** \n", i, MaxN);
    KS ks(N, h, L);
    VectorXd F;
    if(!isSingle) F = multiF(ks, x, nstp, ppType, th);
    else F = multiF(ks, x.col(0), Norbit, ppType, th);
    double err = F.norm(); 
    if(err < tol){
      fprintf(stderr, "stop at norm(F)=%g for iteration %zd\n", err, i);
      break;
    }
   
    pair<SpMat, VectorXd> p = multishoot(ks, x, nstp, ppType, th, false); 
    SpMat JJ = p.first.transpose() * p.first; 
    VectorXd JF = p.first.transpose() * p.second;
    SpMat Dia = JJ; 
    Dia.prune(KeepDiag());

    for(size_t j = 0; j < 20; j++){
      ////////////////////////////////////////
      // solve the update
      SpMat H = JJ + lam * Dia; 
      pair<VectorXd, vector<double> > cg = iterMethod::ConjGradSSOR<SpMat>
	(H, -JF, solver, VectorXd::Zero(H.rows()), H.rows(), 1e-6);
      VectorXd &dF = cg.first;
      
      ////////////////////////////////////////
      // print the CG infomation
      if(Print)
	printf("CG error %g after %lu iterations.\n", 
	       cg.second.back(), cg.second.size());

      ////////////////////////////////////////
      // update the state
      ArrayXXd xnew = x + Map<ArrayXXd>(&dF(0), N-2, M);
      double hnew = h + dF((N-2)*M)/nstp; // be careful here.
      double thnew = th;
      if(ppType.compare("rpo") == 0) thnew += dF((N-2)*M+1);
      
      // check the new time step positve
      if( hnew <= 0 ){ 
	fprintf(stderr, "new time step is negative\n");
	Terminate = true;
	break;
      }
      
      KS ks1(N, hnew, L);
      VectorXd newF;
      if(!isSingle) newF = multiF(ks1, xnew, nstp, ppType, thnew); 
      else newF = multiF(ks1, xnew.col(0), Norbit, ppType, thnew); 
      if(Print) printf("err = %g \n", newF.norm());
      if (newF.norm() < err){
	x = xnew; h = hnew; th = thnew; 
	lam = lam/10; 
	break;
      }
      else{
	lam *= 10; 
	if(Print) printf("lam = %g \n", lam);
	if( lam > 1e10) {
	  fprintf(stderr, "lam = %g too large.\n", lam); 
	  Terminate = true;
	  break; 
	}
      }
      
    }
    
    if( Terminate )  break;
  }
  
  return make_tuple(x.col(0), h, th);
}


/*
pair<MatrixXd, VectorXd> newtonReq(Cqcgl1d &cgl, const ArrayXd &a0, const double w1, const double w2){
  MatrixXd DF(2*N+2, 2*N+2);
  ArrayXd t1 = cgl.TS1(a0);
  ArrayXd t2 = cgl.TS2(a0);
  DF.topLeftCorner(2*N, 2*N) = cgl.stab(a0) + w1*cgl.GS1() + w2*cgl.GS2();
  DF.col(2*N).head(2*N) = t1;
  DF.col(2*N+1).head(2*N) = t2;
  //DF.row(2*N).head(2*N) = t1.transpose();
  //DF.row(2*N+1).head(2*N) = t2.transpose();
  DF.row(2*N).head(2*N) = VectorXd::Zero(2*N);
  DF.row(2*N+1).head(2*N) = VectorXd::Zero(2*N);
  DF.bottomRightCorner(2,2) = MatrixXd::Zero(2,2);

  VectorXd F(2*N+2);
  F.head(2*N) = cgl.vel(a0) + w1*t1 + w2*t2;
  F(2*N) = 0;
  F(2*N+1) = 0;

  return make_pair(DF, F);
  
}
*/
/*
ArrayXd findReq(const ArrayXd &a0, const double w10, const double w20, const int MaxN, const double tol){

  ArrayXd a = a0;
  double w1 = w10;
  double w2 = w20;
  double lam = 1;
  ConjugateGradient<MatrixXd> CG;
  Cqcgl1d cgl(N, L);
  
  for(size_t i = 0; i < MaxN; i++){
    if (lam > 1e10) break;
    printf("********  i = %zd/%d   ******** \n", i, MaxN);
    VectorXd F = cgl.vel(a) + w1*cgl.TS1(a) + w2*cgl.TS2(a);
    double err = F.norm(); 
    if(err < tol){
      printf("stop at norm(F)=%g\n", err);
      break;
    }
   
    pair<MatrixXd, VectorXd> p = newtonReq(cgl, a, w1, w2); 
    MatrixXd JJ = p.first.transpose() * p.first;
    VectorXd JF = p.first.transpose() * p.second;
    
    for(size_t j = 0; j < 20; j++){
      printf("inner iteration j = %zd\n", j);
      //MatrixXd H = JJ + lam * JJ.diagonal().asDiagonal(); 
      MatrixXd H = JJ; H.diagonal() *= (1+lam);
      CG.compute(H);     
      VectorXd dF = CG.solve(-JF);
      printf("CG error %f, iteration number %d\n", CG.error(), CG.iterations());
      ArrayXd anew = a + dF.head(2*N).array();
      double w1new = w1 + dF(2*N); 
      double w2new = w2 + dF(2*N+1);
      printf("w1new = %f, w2new = %f\n", w1new, w2new);
      
      VectorXd Fnew = cgl.vel(anew) + w1new*cgl.TS1(anew) + w2new*cgl.TS2(anew);
      cout << "err = " << Fnew.norm() << endl;
      if (Fnew.norm() < err){
	a = anew; w1 = w1new; w2 = w2new;
	lam = lam/10; cout << "lam = "<< lam << endl;
	break;
      }
      else{
	lam *= 10; cout << "lam = "<< lam << endl;
	if( lam > 1e10) { printf("lam = %f too large.\n", lam); break; }
      }
      
    }
  }
  
  ArrayXd req(2*N+3);
  VectorXd err = cgl.vel(a) + w1*cgl.TS1(a) + w2*cgl.TS2(a);
  req << a, w1, w2, err.norm(); 
  return req;
}
*/
