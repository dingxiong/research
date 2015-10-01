/* How to compile this program:
 * h5c++ ksDimension.cc ./ksint/ksint.cc ./ped/ped.cc ./readks/readks.cc
 * -std=c++0x -I$XDAPPS/eigen/include/eigen3 -I../include/ -lfftw3 -O3 -march=corei7
 * -msse4 -msse2
 *
 * or (Note : libreadks.a is static library, so the following order is important)
 *
 * h5c++ ksDimension.cc -std=c++0x -I$XDAPPS/eigen/include/eigen3 -I../include -L../lib -lreadks -lksint -lped  -lfftw3 -O3 -march=corei7 -msse4 -msse2
 */
#include "ksint.hpp"
#include "ped.hpp"
#include "readks.hpp"
#include <Eigen/Dense>
#include <tuple>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>

using namespace std;
using namespace Eigen;

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
subspBound(const MatrixXi subspDim, const MatrixXi ixSp){
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


/** @brief calculate the cos() of the largest angle
 *         between the two subspaces spanned by the
 *         columns of matrices A and B.
 */
double angleSubspace(const Ref<const MatrixXd> &A,
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

double angleSpaceVector(const Ref<const MatrixXd> &Q,
			const Ref<const VectorXd> &V){
  assert( Q.rows() == V.rows());
  VectorXd P = Q.transpose() * V;
  double cos2 = P.squaredNorm() / V.squaredNorm(); /* cos^2 */

  return sqrt(1-cos2);
}


/** @brief calculate angle between two subspaces along an upo
 *
 *  @param[in] fileName file that stores the ppo/rpo information
 *  @param[in] ppType ppo or rpo
 *  @param[in] ppId id of the periodic orbit
 *  @param[in] subspDim 4xM2  matrix storing the bounds of subspaces
 *  @return first matrix : each row stores the angles at one point
 *          second matrix: matrix indicating whether bounds are the
 *          same as specified
 *  @see subspBound
 */
std::pair<MatrixXd, MatrixXi> 
anglePO(ReadKS &readks, const string ppType,
	const int ppId, const MatrixXi subspDim){
  assert(subspDim.rows() == 4);
  MatrixXd eigVals = readks.readKSe(ppType, ppId); // Floquet exponents
  MatrixXd eigVecs = readks.readKSve(ppType, ppId);// Floquet vectors
  // left and right bounds
  MatrixXi ixSp = readks.indexSubspace(eigVals.col(2), eigVals.col(0)); 
  
  const int N = sqrt(eigVecs.rows());
  const int M = eigVecs.cols();
  const int M2 = subspDim.cols();
  MatrixXd ang_po(M, M2);
  
  // calculate the exact bound of this indices.
  std::pair<MatrixXi, MatrixXi> tmp = subspBound(subspDim, ixSp);
  MatrixXi &bound = tmp.first;
  MatrixXi &boundStrict = tmp.second;

  for(size_t i = 0; i < M; i++){
    MatrixXd ve = eigVecs.col(i);
    ve.resize(N, N);
    for(size_t j = 0; j < M2; j++){      
      double ang = angleSubspace(ve.middleCols(bound(0, j), bound(1,j)-bound(0,j)+1), 
				 ve.middleCols(bound(2, j), bound(3,j)-bound(2,j)+1) );
      ang_po(i, j) = ang;
    }
  }
  
  return std::make_pair(ang_po, boundStrict);
}

/** @brief normalize each row of a matrix  */
void normc(MatrixXd &A){
  int m = A.cols();
  for(size_t i = 0; i < m; i++) 
    A.col(i).array() = A.col(i).array() / A.col(i).norm();
}

/** @brief calculate the minimal distance between an ergodic
 *         trajectory and one ppo/rpo.
 * @note the cols of ppo/rpo should be the same as its corresponding
 *       Flqouet vectors, otherwise, there maybe index flow when calling
 *       difAngle(). For example, you can do it in the following way:
 *       \code
 *        minDistance(ergodicHat, aaHat.leftCols(aaHat.cols()-1), tolClose)
 *       \endcode
 */
std::tuple<MatrixXd, VectorXi, VectorXi, VectorXd>
minDistance(const MatrixXd &ergodic, const MatrixXd &aa, const double tolClose){
  const int n = ergodic.rows();
  const int m = ergodic.cols();
  const int n2 = aa.rows();
  const int m2 = aa.cols();
  assert(n2 == n);
  
  VectorXi minIndexPo(m);
  VectorXi minIndexErgodic(m);
  VectorXd minDis(m);
  MatrixXd minDifv(n, m);

  size_t tracker = 0;
  for(size_t i = 0; i < m; i++){
    MatrixXd dif = aa.colwise() - ergodic.col(i);// relation is inversed.
    VectorXd colNorm(m2);
    for(size_t j = 0; j < m2; j++) colNorm(j) = dif.col(j).norm();
    int r, c;
    double closest = colNorm.minCoeff(&r, &c); 
    if(closest < tolClose){
      minIndexPo(tracker) = r;
      minIndexErgodic(tracker) = i;
      minDis(tracker) = closest;
      minDifv.col(tracker++) = -dif.col(r);
    }
  }
  return std::make_tuple(minDifv.leftCols(tracker), minIndexErgodic.head(tracker),
			 minIndexPo.head(tracker), minDis.head(tracker));
}

std::pair<std::vector<int>, std::vector<int> >
consecutiveWindow(const VectorXi &index, const int window){
  const int n = index.size();
  std::vector<int> start, dur;

  int pos = 0;
  while (pos < n-1) {
    int span = 0;
    for (size_t i = pos; i < n-1; i++) { // note i < n-1
      if(index(i) == index(i+1)-1) span++;
      else break;
    }
    if(span >= window){
      start.push_back(pos);
      dur.push_back(span);
    }
    if(span > 0) pos += span;
    else pos++;
  }
  return make_pair(start, dur);
}

vector<int>
interp(KS &ks, const Ref<const MatrixXd> &ergodic, const MatrixXd &aa,
       const VectorXi &indx, double threashold){
  const int n = ergodic.rows();
  const int m = ergodic.cols();
  const int n2 = aa.rows();
  const int m2 = aa.cols();
  assert(n2 == n);
  
  vector<int> filter;
  for (int i = 0; i < m; i++) {
    VectorXd x = aa.col(indx(i));
    VectorXd velo = ks.veToSlice( ks.velocity(x), x ); 
    VectorXd difv = ergodic.col(i) - x;
    double u = difv.dot(velo);
    if (fabs(u) / (velo.norm() * difv.norm()) < threashold ){
      filter.push_back(i);
    }
  }
  
  return filter;
  
}

std::tuple<MatrixXd, VectorXi, VectorXi, VectorXd>
minDistance(const MatrixXd &ergodic, const MatrixXd &aa, const double tolClose, const int window){

  std::tuple<MatrixXd, VectorXi, VectorXi, VectorXd> min_distance = minDistance(ergodic, aa, tolClose); 
  MatrixXd &minDifv = std::get<0>(min_distance); 
  VectorXi &minIndexErgodic = std::get<1>(min_distance);  
  VectorXi &minIndexPo = std::get<2>(min_distance);
  VectorXd &minDis = std::get<3>(min_distance);

  std::pair<std::vector<int>, std::vector<int> > consecutive =
    consecutiveWindow(minIndexErgodic, window);   
  std::vector<int> &start = consecutive.first;
  std::vector<int> &dur = consecutive.second;
  int sum = 0;
  for (std::vector<int>::iterator i = dur.begin(); i != dur.end(); i++) {
    sum += *i;
  }

  MatrixXd minDifv2(minDifv.rows(), sum);
  VectorXi minIndexErgodic2(sum);
  VectorXi minIndexPo2(sum);
  VectorXd minDis2(sum);
  
  size_t pos = 0;
  for(size_t i = 0; i < dur.size(); i++){ 
    minDifv2.middleCols(pos, dur[i]) = minDifv.middleCols(start[i], dur[i]);
    minIndexErgodic2.segment(pos, dur[i]) = minIndexErgodic.segment(start[i], dur[i]);
    minIndexPo2.segment(pos, dur[i]) = minIndexPo.segment(start[i], dur[i]);
    minDis2.segment(pos, dur[i]) = minDis.segment(start[i], dur[i]);

    pos += dur[i];
  }

  return std::make_tuple(minDifv2, minIndexErgodic2, minIndexPo2, minDis2);
}


/** @brief
 * ve has dimension [N, M*trunc]
 */
MatrixXd veTrunc(const MatrixXd ve, const int pos, const int trunc = 0){
  int Trunc = trunc;
  if(trunc == 0) Trunc = ve.rows();

  const int N = ve.rows();
  const int M = ve.cols() / Trunc;
  assert(ve.cols()%M == 0);
  
  MatrixXd newVe(N, (Trunc-1)*M);
  for(size_t i = 0; i < M; i++){
    newVe.middleCols(i*(Trunc-1), pos) = ve.middleCols(i*Trunc, pos);
    newVe.middleCols(i*(Trunc-1)+pos, Trunc-1-pos) = 
      ve.middleCols(i*Trunc+pos+1, Trunc-1-pos);
  }
  return newVe;
}

/** @brief calculate the angle between difference vectors and the subspaces spanned
 *  by Flqouet vectors.
 *
 *  @param[in] subsp number of Floquet vectors to span subspace
 *
 *  @note subsp does not stores the indices of the subspace cut. 
 */
MatrixXd difAngle(const MatrixXd &minDifv, const VectorXi &minIx, const VectorXi &subsp, 
		  const MatrixXd &ve_trunc, const int truncN){
  assert(minDifv.cols() == minIx.size());
  const int N = minDifv.rows();
  const int M = minDifv.cols();
  const int M2 = subsp.size();
  
  MatrixXd angle(M2, M);
  for(size_t i = 0; i < M; i++){
    int ix = minIx(i);
    for(size_t j = 0; j < M2; j++)
      // calculate the angle between the different vector and Floquet subspace.
      angle(j, i) = angleSubspace(ve_trunc.middleCols(truncN*ix, subsp(j)),
      				  minDifv.col(i));
    // angle(j, i) = angleSpaceVector(ve_trunc.middleCols(truncN*ix, subsp(j)),
    // minDifv.col(i));
  }
  return angle;
}



int main(){
  cout.precision(16);
  const int Nks = 64;
  const int N = Nks - 2;
  const int N64 = 64;
  const int N62 = 62;
  const double L = 22;

  switch (4)
    {
    case 1: // small test for angle calculation.
      {
	MatrixXi subspDim(4,3); 
	subspDim << 
	  3, 0, 0,
	  3, 7, 8,
	  4, 8, 9,
	  4, 29, 29; // (0-6,7-29), (0-7,8-29), (0-8,9-29)
	cout << subspDim << endl;
	string fileName("../data/ks22h02t100");
	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5");
	string ppType("rpo");	
	int ppId = 1;
	std::pair<MatrixXd, MatrixXi> ang = 
	  anglePO(readks, ppType, ppId, subspDim);
	
	cout << ang.second << endl;
	
	ofstream file;
	file.precision(16);
	file.open("good.txt", ios::trunc);
	file << ang.first << endl;
	file.close();

	break;
      }


    case 2: // calculate the angle, output to files.
	    // This the MAIN experiments I am doing.
      {
	/////////////////////////////////////////////////////////////////
	string fileName("../data/ks22h02t120");
	string ppType("rpo");
	string spType("vector");
	string folder("./case4/");
	/////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////
	// all kinds of IF statements
	int NN;
	if(fileName.compare("../data/ks22h02t120") ==0 ){	  
	  if(ppType.compare("ppo") == 0) NN = 840; // number of ppo
	  else NN = 834; // number of rpo
	}
	else if(fileName.compare("../data/ks22h02t100") ==0 ) {
	  if(ppType.compare("ppo") == 0) NN = 240; // number of ppo
	  else NN = 239; // number of rpo
	}
	else{ 
	  printf("invalid file name !\n");
	}

       	MatrixXi subspDim(4,29);
	if(spType.compare("vector") == 0)
	  for (size_t i = 0; i < 29; i++) subspDim.col(i) << i, i, i+1, i+1;
	else if(spType.compare("space") == 0)
	  for (size_t i = 0; i < 29; i++) subspDim.col(i) << 0, i, i+1, 29;
	else {
	  printf("invalid spType.\n");
	}
	const int M = subspDim.cols();
	ofstream file[M];
	string angName("ang");
	for(size_t i = 0; i < M; i++){
	  file[i].open(folder + angName + to_string(i) + ".txt", ios::trunc);
	  file[i].precision(16);
	}
	/////////////////////////////////////////////////////////////////
	
	/////////////////////////////////////////////////////////////////
	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5");
	// get the index of POs which converge.
	MatrixXi status = readks.checkExistEV(ppType, NN);
	for(size_t i = 0; i < NN; i++)
	  {
	    if( 1 == status(i,0) ){
	      int ppId = i + 1;	      
	      printf("========= i = %zd ========\n", i);
	      std::pair<MatrixXd, MatrixXi> tmp =
		anglePO(readks, ppType, ppId, subspDim);
	      MatrixXd &ang = tmp.first;
	      MatrixXi &boundStrict = tmp.second; cout << boundStrict << endl;
	      // check whether there are degeneracy
	      MatrixXi pro = boundStrict.row(0).array() *  boundStrict.row(1).array() *
		boundStrict.row(2).array() * boundStrict.row(3).array();
	      for(size_t i = 0; i < M; i++){
		// only keep the orbits whose 4 bounds are not degenerate
		if(pro(0, i) == 1) file[i] << ang.col(i) << endl;
	      }
	    }
	  }
	for(size_t i = 0; i < M; i++) file[i].close();
	/////////////////////////////////////////////////////////////////
	
	break;
      }
      
    case 3 : // small test of the covariant vector projection process. 
      {
	string fileName("../data/ks22h02t100");
	string ppType("ppo");
	const int ppId = 1;
	const int gTpos = 3;

	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5");
	std::tuple<ArrayXd, double, double, double, double>
	  pp = readks.readKSinit(ppType, ppId);	
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
  	MatrixXd eigVals = readks.readKSe(ppType, ppId);
	MatrixXd eigVecs = readks.readKSve(ppType, ppId);

	KS ks(Nks, T/nstp, L);
	ArrayXXd aa = ks.intg(a, nstp);
	std::pair<MatrixXd, VectorXd> tmp = ks.orbitToSlice(aa); 
	MatrixXd &aaHat = tmp.first; 
	MatrixXd veSlice = ks.veToSliceAll( eigVecs, aa.leftCols(aa.cols()-1) );
	MatrixXd ve_trunc = veTrunc(veSlice, gTpos);
	
	cout << veSlice.middleCols(2,3) << endl << endl;
	cout << veSlice.middleCols(2+30*100,3) << endl << endl;
	cout << ve_trunc.middleCols(2,2) << endl << endl;
	cout << ve_trunc.middleCols(2+29*100,2) << endl << endl;
	
	break;
      }
      
    case 4 : // ergodic orbit approache rpo/ppo
	     // fields need to be changed: ppType ppId, nqr, gTpos, sT, folder
	{
	    ////////////////////////////////////////////////////////////
	    // set up the system
	    const int Nks = 64;
	    const int N = Nks - 2;
	    string fileName("../data/ks22h001t120x64");
	    string ppType("ppo");
	    const int ppId = 4; 
	    const int nqr = 5;
	    const int Trunc = 30;
	    const int gTpos = 3; // position of group tangent marginal vector 
	    VectorXi subsp(7); subsp << 3, 4, 5, 6, 7, 8, 9; // subspace indices.
	    ////////////////////////////////////////////////////////////

	    ////////////////////////////////////////////////////////////
	    // prepare orbit, vectors
	    ReadKS readks(fileName+".h5",
			  fileName+"E.h5", 
			  fileName+"EV.h5", N, Nks);
	    // ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5");
	    std::tuple<ArrayXd, double, double, double, double>
		pp = readks.readKSinit(ppType, ppId);	
	    ArrayXd &a = get<0>(pp); 
	    double T = get<1>(pp);
	    int nstp = (int)get<2>(pp);
	    double r = get<3>(pp);
	    double s = get<4>(pp);
	    MatrixXd eigVals = readks.readKSe(ppType, ppId); 
	    MatrixXd eigVecs = readks.readKSve(ppType, ppId);

	    KS ks(Nks, T/nstp, L);
	    ArrayXXd aa = ks.intg(a, nstp, nqr); 
	    ArrayXXd aaWhole, eigVecsWhole;
	    if(ppType.compare("ppo") == 0) {
		aaWhole = ks.half2whole(aa.leftCols(aa.cols()-1));
		eigVecsWhole = ks.half2whole(eigVecs); // this is correct. Think carefully.
	    } else {
		aaWhole = aa.leftCols(aa.cols()-1); // one less
		eigVecsWhole = eigVecs;
	    }
	    assert(aaWhole.cols() == eigVecsWhole.cols());
	    std::pair<MatrixXd, VectorXd> tmp = ks.orbitToSlice(aaWhole); 
	    MatrixXd &aaHat = tmp.first;
	    // note here aa has one more column the the Floquet vectors
	    MatrixXd veSlice = ks.veToSliceAll( eigVecsWhole, aaWhole, Trunc);
	    MatrixXd ve_trunc = veTrunc(veSlice, gTpos, Trunc);
	    ////////////////////////////////////////////////////////////
	
	    ////////////////////////////////////////////////////////////
	    // choose experiment parameters & do experiment
	    const double h = 0.1;
	    const double sT = 30;
	    const double tolClose = 0.1;
	    const int MaxSteps = floor(2000/h);
	    const int MaxT = 10000;
	    string folder = "./case4ppo4x5sT30/";
	    KS ks2(Nks, h, L); 
	    srand(time(NULL));
	    ArrayXd a0(0.1 * ArrayXd::Random(N));
	
	    const int fileNum = 5;
	    string strf[fileNum] = {"angle", "dis", "difv", "indexPo", "No"};
	    ofstream saveName[fileNum];	
	    for (size_t i = 0; i < fileNum; i++) {
		saveName[i].open(folder + strf[i], ios::trunc);
		saveName[i].precision(16);
	    } 
	
	    int accum = 0;
	    for(size_t i = 0; i < MaxT; i++){
		std::cout << "********** i = " << i << "**********"<< std::endl;
		ArrayXXd ergodic = ks2.intg(a0, MaxSteps); a0 = ergodic.rightCols(1);
		std::pair<MatrixXd, VectorXd> tmp = ks2.orbitToSlice(ergodic);
		MatrixXd &ergodicHat = tmp.first;
		// be careful about the size of aaHat
		std::tuple<MatrixXd, VectorXi, VectorXi, VectorXd> 
		    dis = minDistance(ergodicHat, aaHat, tolClose, (int)(sT/h) );
		MatrixXd &minDifv = std::get<0>(dis); 
		VectorXi &minIndexErgodic = std::get<1>(dis);
		VectorXi &minIndexPo = std::get<2>(dis); 
		VectorXd &minDis = std::get<3>(dis);
		MatrixXd angle = difAngle(minDifv, minIndexPo, subsp, ve_trunc, Trunc-1);
		if(angle.cols() > 0) printf("angle size = %ld x %ld, instance %d \n", 
					    angle.rows(), angle.cols(), ++accum);

		if(angle.cols() != 0) {
		    saveName[0] << angle.transpose() << endl;
		    saveName[1] << minDis << std::endl;
		    saveName[2] << minDifv.transpose() << std::endl;
		    saveName[3] << minIndexPo << std::endl;
		    saveName[4] << angle.cols() << std::endl;
		}
	    }
	    for (size_t i = 0; i < fileNum; i++)  saveName[i].close();
	    ////////////////////////////////////////////////////////////
	    break;
	}
      
    case 5: // test the projection of subspaces.
	    // not using FVs, but OVs. OVs is rotated to slice.
      {
	//////////////////////////////////////////////////
	string fileName("../data/ks22h001t120x64");
	string ppType("ppo");
	const int ppId = 4; 
	VectorXi subsp(10); subsp << 4, 5, 6, 7, 8, 9, 10, 12, 16, 28;

	//////////////////////////////////////////////////
	// prepare orbit, vectors
	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N, Nks);
	std::tuple<ArrayXd, double, double, double, double>
	  pp = readks.readKSinit(ppType, ppId);	
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
	
	KS ks(Nks, T/nstp, L);
	pair<ArrayXXd, ArrayXXd> tmp = ks.intgj(a, nstp, 5, 5);
	ArrayXXd aaWhole;
	if(ppType.compare("ppo") == 0)
	  aaWhole = ks.half2whole(tmp.first.leftCols(tmp.first.cols()-1));
	else 
	  aaWhole = tmp.first;
	MatrixXd daa = tmp.second;
	
	std::pair<MatrixXd, VectorXd> tmp2 = ks.orbitToSlice(aaWhole); 
	MatrixXd &aaHat = tmp2.first; 
	VectorXd &theta = tmp2.second;

	PED ped;
	ped.reverseOrderSize(daa); // reverse order.
	if(ppType.compare("ppo") == 0)
	  daa.leftCols(N) = ks.Reflection(daa.leftCols(N)); // R*J for ppo
	else // R*J for rpo
	  daa.leftCols(N) = ks.Rotation(daa.leftCols(N), -s*2*M_PI/L);
	pair<MatrixXd, vector<int> > psd = ped.PerSchur(daa, 1000, 1e-15, false);
	MatrixXd &Q1 = psd.first;
	int n1 = Q1.rows();
	int m1 = Q1.cols() / n1;
	MatrixXd Q2(n1, n1*m1);
	for(size_t i = 0; i < m1; i++) 
	  Q2.middleCols(i*n1, n1) = Q1.middleCols(n1*((m1-i)%m1), n1);

	MatrixXd Q;
	if(ppType.compare("ppo") == 0)
	  Q= ks.half2whole(Q2);
	else // R*J for rpo
	  Q = Q2;
	
	int n = Q.rows();
	int m = Q.cols() / n;
	MatrixXd rQ(n, m*n);
	for (size_t i = 0; i < m; i++) {
	  rQ.middleCols(i*n, n) = ks.Rotation( Q.middleCols(i*n, n),
					       -theta(i) );
	}
  
	////////////////////////////////////////////////////////////
	// choose experiment parameters & do experiment
	const double h = 0.1;
	const double sT = 50;
	const double tolClose = 0.1;
	const int MaxSteps = floor(2000/h);
	const int MaxT = 10000;
	string folder = "./case7/";
	KS ks2(Nks, h, L); 
	srand(time(NULL));
	ArrayXd a0(0.1 * ArrayXd::Random(N));
	
	const int fileNum = 5;
	string strf[fileNum] = {"angle", "dis", "difv", "indexPo", "No"};
	ofstream saveName[fileNum];	
	for (size_t i = 0; i < fileNum; i++) {
	  saveName[i].open(folder + strf[i], ios::trunc);
	  saveName[i].precision(16);
	} 

	for(size_t i = 0; i < MaxT; i++){
	  std::cout << "********** i = " << i << "**********"<< std::endl;
	  ArrayXXd ergodic = ks2.intg(a0, MaxSteps); a0 = ergodic.rightCols(1);
	  std::pair<MatrixXd, VectorXd> tmp = ks2.orbitToSlice(ergodic);
	  MatrixXd &ergodicHat = tmp.first;
	  // be careful about the size of aaHat
	  std::tuple<MatrixXd, VectorXi, VectorXi, VectorXd> 
	    dis = minDistance(ergodicHat, aaHat, tolClose, (int)(sT/h) );
	  MatrixXd &minDifv = std::get<0>(dis); 
	  VectorXi &minIndexErgodic = std::get<1>(dis);
	  VectorXi &minIndexPo = std::get<2>(dis); 
	  VectorXd &minDis = std::get<3>(dis);
	  MatrixXd angle = difAngle(minDifv, minIndexPo, subsp, rQ, N);
	  if(angle.cols() > 0) printf("angle size = %ld x %ld\n", angle.rows(),
				      angle.cols());

	  if(angle.cols() != 0) {
	    saveName[0] << angle.transpose() << endl;
	    saveName[1] << minDis << std::endl;
	    saveName[2] << minDifv.transpose() << std::endl;
	    saveName[3] << minIndexPo << std::endl;
	    saveName[4] << angle.cols() << std::endl;
	  }
	}
	for (size_t i = 0; i < fileNum; i++)  saveName[i].close();
	////////////////////////////////////////////////////////////
	  
	break;
      }

    case 6: // test the projection of subspaces.
	    // not using FVs, but OVs. Slice is not used,
	    // Experiments are conducted on the full state space.
      {
	//////////////////////////////////////////////////
	string fileName("../data/ks22h02t100");
	string ppType("ppo");
	const int ppId = 1; 
	VectorXi subsp(10); subsp << 4, 5, 6, 7, 8, 9, 10, 12, 16, 28;

	//////////////////////////////////////////////////
	// prepare orbit, vectors
	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5");
	std::tuple<ArrayXd, double, double, double, double>
	  pp = readks.readKSinit(ppType, ppId);	
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
	
	KS ks(Nks, T/nstp, L);
	pair<ArrayXXd, ArrayXXd> tmp = ks.intgj(a, nstp);
	ArrayXXd aaWhole;
	if(ppType.compare("ppo") == 0)
	  aaWhole = ks.half2whole(tmp.first.leftCols(tmp.first.cols()-1));
	else 
	  aaWhole = tmp.first;
	MatrixXd daa = tmp.second;
	
	PED ped;
	ped.reverseOrderSize(daa); // reverse order.
	if(ppType.compare("ppo") == 0)
	  daa.leftCols(N) = ks.Reflection(daa.leftCols(N)); // R*J for ppo
	else // R*J for rpo
	  daa.leftCols(N) = ks.Rotation(daa.leftCols(N), -s*2*M_PI/L);
	pair<MatrixXd, vector<int> > psd = ped.PerSchur(daa, 10000, 1e-15, false);
	MatrixXd &Q1 = psd.first;
	int n1 = Q1.rows();
	int m1 = Q1.cols() / n1;
	MatrixXd Q2(n1, n1*m1);
	for(size_t i = 0; i < m1; i++) 
	  Q2.middleCols(i*n1, n1) = Q1.middleCols(n1*((m1-i)%m1), n1);

	MatrixXd Q;
	if(ppType.compare("ppo") == 0)
	  Q= ks.half2whole(Q2);
	else // R*J for rpo
	  Q = Q2;
  
	////////////////////////////////////////////////////////////
	// choose experiment parameters & do experiment
	const double h = 0.1;
	const double sT = 20;
	const double tolClose = 0.1;
	const int MaxSteps = floor(2000/h);
	const int MaxT = 100000;
	string folder = "./case2/";
	KS ks2(Nks, h, L); 
	srand(time(NULL));
	ArrayXd a0(0.1 * ArrayXd::Random(N));
	
	const int fileNum = 5;
	string strf[fileNum] = {"angle", "dis", "difv", "indexPo", "No"};
	ofstream saveName[fileNum];	
	for (size_t i = 0; i < fileNum; i++) {
	  saveName[i].open(folder + strf[i], ios::trunc);
	  saveName[i].precision(16);
	} 

	for(size_t i = 0; i < MaxT; i++){
	  std::cout << "********** i = " << i << "**********"<< std::endl;
	  ArrayXXd ergodic = ks2.intg(a0, MaxSteps); a0 = ergodic.rightCols(1);
	  // be careful about the size of aaHat
	  std::tuple<MatrixXd, VectorXi, VectorXi, VectorXd> 
	    dis = minDistance(ergodic, aaWhole, tolClose,
			      (int)(sT/h) );
	  MatrixXd &minDifv = std::get<0>(dis); 
	  VectorXi &minIndexErgodic = std::get<1>(dis);
	  VectorXi &minIndexPo = std::get<2>(dis); 
	  VectorXd &minDis = std::get<3>(dis);
	  MatrixXd angle = difAngle(minDifv, minIndexPo, subsp, Q, N);
	  if(angle.cols() > 0) printf("angle size = %ld x %ld\n", angle.rows(),
				      angle.cols());

	  if(angle.cols() != 0) {
	    saveName[0] << angle.transpose() << endl;
	    saveName[1] << minDis << std::endl;
	    saveName[2] << minDifv.transpose() << std::endl;
	    saveName[3] << minIndexPo << std::endl;
	    saveName[4] << angle.cols() << std::endl;
	  }
	}
	for (size_t i = 0; i < fileNum; i++)  saveName[i].close();
	////////////////////////////////////////////////////////////
	  
	break;
      }

    case 7: // test rotated initial condition for N = 64
	    // after rotation, the result looks good.
      {

	string fileName("../data/ks22h001t120x64");
	string ppType("ppo");
	const int ppId = 13; 

	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5", N62, N64);
	std::tuple<ArrayXd, double, double, double, double>
	  pp = readks.readKSinit(ppType, ppId);	
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);

	const int div = 1;
	KS ks(N64, T/nstp/div, L);
	MatrixXd aa = ks.intg(a, 2*div*nstp);
	MatrixXd aaHat = ks.orbitToSlice(aa).first;
	cout << (aa.rightCols(1) - aa.col(0)).norm() << endl;
	MatrixXd paa = ks.intg(ks.Rotation(aa.col(0), 1), 2*div*nstp);
	std::cout << (paa.rightCols(1) - paa.col(0)).norm() << std::endl;

	break;
      }
      
      
    case 8: // test the linear relation of OVs
	    // the result is bad 1e-5. 
      {
	//////////////////////////////////////////////////
#if 0
	string fileName("../data/ks22h02t100");
	string ppType("rpo");
	const int ppId = 3; 
	
	ReadKS readks(fileName+".h5", fileName+"E.h5", fileName+"EV.h5");
	std::tuple<ArrayXd, double, double, double, double>
	  pp = readks.readKSinit(ppType, ppId);	
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);

	KS ks(Nks, T/nstp, L);
# endif			
	string fileName("../data/ks22h001t120x64");
	string ppType("rpo");
	const int ppId = 8;
	
	ReadKS readks(fileName+".h5", fileName+".h5", fileName+".h5", N62, N64);
	std::tuple<ArrayXd, double, double, double, double>
	  pp = readks.readKSinit(ppType, ppId);
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
	
	KS ks(N64, T/nstp, L);
	//////////////////////////////////////////////////
	
	pair<ArrayXXd, ArrayXXd> tmp = ks.intgj(a, nstp, 5, 5);
	MatrixXd aa = tmp.first;
	MatrixXd daa = tmp.second;
	PED ped;
	ped.reverseOrderSize(daa); // reverse order.
	if(ppType.compare("ppo") == 0)
	  daa.leftCols(N) = ks.Reflection(daa.leftCols(N)); // R*J for ppo
	else // R*J for rpo
	  daa.leftCols(N) = ks.Rotation(daa.leftCols(N), -s*2*M_PI/L);


	switch (2) {
	case 1:
	  {
	    pair<MatrixXd, vector<int> > psd = ped.PerSchur(daa, 3000, 1e-15, true);
	    MatrixXd &Q1 = psd.first;
	    int n1 = Q1.rows();
	    int m1 = Q1.cols() / n1;
	    MatrixXd Q2(n1*n1, m1);
	    for(size_t i = 0; i < m1; i++) {
	      MatrixXd tmp = Q1.middleCols(n1*((m1-i)%m1), n1);
	      tmp.resize(n1*n1,1);
	      Q2.col(i) = tmp;
	    }
	    MatrixXd Q = ks.veToSliceAll( Q2, aa.leftCols(aa.cols()-1) );
	    cout << Q.rows() << 'x' << Q.cols() << endl;

	    ColPivHouseholderQR<MatrixXd> qr(Q.leftCols(5));
	    MatrixXd R = qr.matrixQR().triangularView<Upper>();
	    cout << R.rows() << 'x' << R.cols() << endl;
	    cout << R.topRows<5>() << endl;

	    break;
	  }
	  
	case 2 :
	  {
	    const int N = daa.rows();
	    const int M = daa.cols() / N;
	    MatrixXd J(N, N*M);
	    for (size_t i = 0; i < M; i++) {
	      J.middleCols(i*N, N) = daa.middleCols((M-i-1)*N, N);
	    }
	    
	    HouseholderQR<MatrixXd> qr;
	    MatrixXd Q(MatrixXd::Identity(N,N));
	    MatrixXd Qp(N, N);
	    for (size_t i = 0; i < 3000; i++) {
	      if(i%10 == 0) cout << i << endl;
	      for (size_t i = 0; i < M; i++) {
		Qp = J.middleCols(i*N, N) * Q;
		qr.compute(Qp);
		Q = qr.householderQ();
	      }
	    }
	    Q.resize(N*N,1);
	    MatrixXd rQ = ks.veToSliceAll(Q, aa.col(0));
	    qr.compute(rQ.leftCols(5));
	    MatrixXd R = qr.matrixQR().triangularView<Upper>();
	    cout << R.rows() << 'x' << R.cols() << endl;
	    cout << R.topRows<5>() << endl;
	  }
	}
	
	break;
      }

    case 9 : // test the 32 modes initial condition 
      {
	string fileName("../data/ks22h001t120x64");
	string ppType("ppo");
	const int ppId = 1;
	ReadKS readks(fileName+".h5", fileName+".h5", fileName+".h5", N62, N64);
	std::tuple<ArrayXd, double, double, double, double>
	  pp = readks.readKSinit(ppType, ppId);
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
	
	KS ks(N64, T/nstp, L);
	MatrixXd daa = ks.intgj(a, nstp, 10, 10).second;
	
	PED ped ;
	ped.reverseOrderSize(daa);
	daa.leftCols(N) = ks.Reflection(daa.leftCols(N));
	MatrixXd eigVals = ped.EigVals(daa, 3000, 1e-15, true);
	
	MatrixXd FloquetE(eigVals.rows(), eigVals.cols());
	FloquetE << eigVals.col(0).array()/T, eigVals.rightCols(2);
	cout << FloquetE << endl;

	break;
      }

    case 10: // test the spacing in 32 modes orbit
      {
	string fileName("../data/ks22h001t50x64");
	string ppType("rpo");
	const int ppId = 8;
	
	ReadKS readks(fileName+".h5", fileName+".h5", fileName+".h5", N62, N64);
	std::tuple<ArrayXd, double, double, double, double>
	  pp = readks.readKSinit(ppType, ppId);
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
	
	KS ks(N64, T/nstp, L);
	MatrixXd aa = ks.intg(a, nstp, 10);
	ArrayXXd aaWhole;
	if(ppType.compare("ppo") == 0) {
	  aaWhole = ks.half2whole(aa.leftCols(aa.cols()-1));
	} else {
	  aaWhole = aa.leftCols(aa.cols()-1); // one less
	} 
	MatrixXd aaReduced = ks.orbitToSlice(aaWhole).first;
	// cout << aa.rows() << 'x' << aa.cols() << endl;
	VectorXd dis(aaReduced.cols());
	for (int i = 0; i < dis.size(); i++) {
	  dis(i) = (aaReduced.col( (i+1)%dis.size() )-aaReduced.col(i)).norm();
	}
	cout << dis << endl;
	
	break;
      }
      
    case 11: // to try Predrag's suggestions
      {
	string folder("../data/ErgodicApproch/cases32modes/case4ppo4x10/");
	string ppType("ppo");
	const int ppId = 4;
	const int N = 62;
	const int nqr = 10;

	// read files from disk
	const int fileNum = 4;
	string strf[fileNum] = {"No", "dis", "difv", "indexPo"};	
	ifstream saveName[fileNum];
	for (int i = 0; i < fileNum; i++) saveName[i].open(folder+strf[i]);

	string line;
	vector<int> No;
	while( getline(saveName[0], line) ) No.push_back(stoi(line));
	int M = accumulate(No.begin(), No.end(), 0);
	VectorXd dis(M);
	VectorXi indexPo(M);
	MatrixXd difv(N, M);
	for (int i = 0; i < M; i++) {
	  getline(saveName[1], line);
	  dis(i) = stod(line);
	  
	  getline(saveName[2], line);
	  size_t sz = 0;
	  for (int j = 0; j < N; j++){
	    size_t tmp;
	    difv(j, i) = stod(line.substr(sz), &tmp);
	    sz += tmp;
	  }

	  getline(saveName[3], line);
	  indexPo(i) = stoi(line);
	}
	for (int i = 0; i < fileNum; i++) saveName[i].close();
	
	string fileName("../data/ks22h001t50x64.h5");
	ReadKS readks(fileName, fileName, fileName, N, N+2);
	std::tuple<ArrayXd, double, double, double, double>
	  pp = readks.readKSinit(ppType, ppId);	
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
	
	KS ks(N+2, T/nstp, L);
	ArrayXXd aa = ks.intg(a, nstp, nqr); 
	ArrayXXd aaWhole;
	if(ppType.compare("ppo") == 0) {
	  aaWhole = ks.half2whole(aa.leftCols(aa.cols()-1));
	} else {
	  aaWhole = aa.leftCols(aa.cols()-1); // one less
	}
	MatrixXd aaHat = ks.orbitToSlice(aaWhole).first;
	MatrixXd ergodicHat(N, M);
	for (int i = 0; i < M; i++) {
	  ergodicHat.col(i) = aaHat.col(indexPo(i)) + difv.col(i);
	}
	//vector<int> filter = interp(ks, ergodicHat, aaHat, indexPo, 0.05);
	//for (int i = 0; i < filter.size(); i++) cout << filter[i] << endl;
	// cout << M << endl;
	cout << aaHat << endl;
	
	break;
      }

    case 12:
      {
	string fileName("../data/ks22h001t120x64");
	string ppType("ppo");
	const int ppId = 1; 
	const int nqr = 5;
	const int Trunc = 30;
	const int gTpos = 3; 
	////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////
	// prepare orbit, vectors
	const int Nks = 64;
	const int N = Nks - 2;
	ReadKS readks(fileName+".h5",
		      fileName+"E.h5", 
		      fileName+"EV.h5", N, Nks);
	std::tuple<ArrayXd, double, double, double, double>
	  pp = readks.readKSinit(ppType, ppId);	
	ArrayXd &a = get<0>(pp); 
	double T = get<1>(pp);
	int nstp = (int)get<2>(pp);
	double r = get<3>(pp);
	double s = get<4>(pp);
  	MatrixXd eigVals = readks.readKSe(ppType, ppId); 
	MatrixXd eigVecs = readks.readKSve(ppType, ppId);
	
	// printf("%ld x %ld\n", eigVecs.rows(), eigVecs.cols());
	assert(N * Trunc == eigVecs.rows());
	const int M = eigVecs.cols();
	VectorXd vol = VectorXd::Zero(Trunc);
	for (int i = 0; i < M; i++) {
	  MatrixXd v = eigVecs.col(i);
	  v.resize(N, Trunc);
	  for (int j = 0; j < Trunc; j++) {
	    double nor = v.col(j).norm();
	    v.col(j) = v.col(j) / nor;
	  }
	  for (int j = 1; j < Trunc+1; j++) {
	    MatrixXd A = v.leftCols(j).transpose();
	    MatrixXd B = v.leftCols(j);
	    double  det = (A*B).determinant();
	    // if(i == 500) cout << sqrt(det) << endl;
	    vol(j-1) = vol(j-1) + sqrt(det);
	  }
	}
	vol = vol.array() / M;
	cout << vol << endl;


	KS ks(Nks, T/nstp, L);
	ArrayXXd aa = ks.intg(a, nstp, nqr); 
	ArrayXXd aaWhole, eigVecsWhole;
	if(ppType.compare("ppo") == 0) {
	  aaWhole = ks.half2whole(aa.leftCols(aa.cols()-1));
	  eigVecsWhole = ks.half2whole(eigVecs); // this is correct. Think carefully.
	} else {
	  aaWhole = aa.leftCols(aa.cols()-1); // one less
	  eigVecsWhole = eigVecs;
	}
	assert(aaWhole.cols() == eigVecsWhole.cols());
	std::pair<MatrixXd, VectorXd> tmp = ks.orbitToSlice(aaWhole); 
	MatrixXd &aaHat = tmp.first;
	// note here aa has one more column the the Floquet vectors
	MatrixXd veSlice = ks.veToSliceAll( eigVecsWhole, aaWhole, Trunc);
	MatrixXd ve_trunc = veTrunc(veSlice, gTpos, Trunc);
	
	break;
      }

    default:
      {
	printf("please indicate the index of problem !\n");
      }
    }
  return 0;

}
 
