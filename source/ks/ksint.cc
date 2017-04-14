#include "ksint.hpp"
#include "denseRoutines.hpp"
#include <cmath>
#include <iostream>

using namespace denseRoutines;

//////////////////////////////////////////////////////////////////////
KS::NL::NL(){}
KS::NL::NL(KS *ks, int cols) : ks(ks), N(ks->N) {
    u.resize(ks->N, cols);
}
KS::NL::~NL(){}
void KS::NL::operator()(ArrayXXcd &x, ArrayXXcd &dxdt, double t){
    int cs = x.cols();
    assert(cs == dxdt.cols() && N/2+1 == x.rows() && N/2+1 == dxdt.rows());
    
    for(int i = 0; i < cs; i++)
	ks->fft.inv(u.data() + i*N, x.data() + i*(N/2+1), N); 
    ArrayXd orbit = u.col(0);
    u = u.colwise() * orbit;
    for(int i = 0; i < cs; i++)
	ks->fft.fwd(dxdt.data() + i*(N/2+1), u.data() + i*N, N); 
    dxdt.col(0) *= ks->G;
    if(cs > 1)
	dxdt.rightCols(cs-1).colwise() *= 2.0 * ks->G;
}

//////////////////////////////////////////////////////////////////////
/*-------------------- constructor, destructor -------------------- */
// dimension of physical state : N
// dimension of state space    : N - 2 (zeroth and highest mode are excluded)
// dimension of Fourier space  : N/2 + 1
KS::KS(int N, double d) : N(N), d(d), nl(this, 1), nl2(this, N-1) {
    K = ArrayXd::LinSpaced(N/2+1, 0, N/2) * 2 * M_PI / d; //2*PI/d*[0, 1, 2,...,N/2]
    K(N/2) = 0;
    L = K*K - K*K*K*K; 
    G = 0.5 * dcp(0,1) * K * N;   
    
    fft.SetFlag(fft.HalfSpectrum); // very important
    
    int nYN0 = eidr.names.at(eidr.scheme).nYN; // do not call setScheme here. Different.
    for(int i = 0; i < nYN0; i++){
	Yv[i].resize(N/2+1, 1);
	Nv[i].resize(N/2+1, 1);
	Yv2[i].resize(N/2+1, N-1);
	Nv2[i].resize(N/2+1, N-1);
    }
    eidr.init(&L, Yv, Nv);
    eidr2.init(&L, Yv2, Nv2);
}
KS & KS::operator=(const KS &x){
    return *this;
}
KS::~KS(){}

/*------------------- member methods ------------------ */
void KS::setScheme(std::string x){
    int nYN0 = eidr.names.at(eidr.scheme).nYN;
    eidr.scheme = x;
    int nYN1 = eidr.names.at(eidr.scheme).nYN;
    for (int i = nYN0; i < nYN1; i++) {
	Yv[i].resize(N/2+1, 1);
	Nv[i].resize(N/2+1, 1);
	Yv2[i].resize(N/2+1, N-1);
	Nv2[i].resize(N/2+1, N-1);
    }
}

ArrayXXd 
KS::intgC(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
    assert(a0.size() == N-2);
    ArrayXXcd u0 = R2C(a0); 
    const int Nt = (int)round(tend/h);
    const int M = (Nt + skip_rate - 1) / skip_rate;
    ArrayXXcd aa(N/2+1, M);
    lte.resize(M);
    int count = 0;
    auto ss = [this, &count, &aa](ArrayXXcd &x, double t, double h, double err){
	aa.col(count) = x;
	lte(count++) = err;
    };
    eidr.intgC(nl, ss, 0, u0, tend, h, skip_rate); 
    
    return C2R(aa);
}

std::pair<ArrayXXd, ArrayXXd>
KS::intgjC(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
    assert(a0.size() == N-2);
    ArrayXXd v0(N-2, N-1); 
    v0 << a0, MatrixXd::Identity(N-2, N-2);
    ArrayXXcd u0 = R2C(v0);
    
    const int Nt = (int)round(tend/h);
    const int M = (Nt + skip_rate - 1) / skip_rate;
    ArrayXXcd aa(N/2+1, M), daa(N/2+1, (N-2)*M);
    lte.resize(M);
    int count = 0;
    auto ss = [this, &count, &aa, &daa](ArrayXXcd &x, double t, double h, double err){
	aa.col(count) = x.col(0);
	daa.middleCols(count*(N-2), N-2) = x.rightCols(N-2);
	lte(count++) = err;
	x.rightCols(N-2) = R2C(MatrixXd::Identity(N-2, N-2));
    };
    
    eidr2.intgC(nl2, ss, 0, u0, tend, h, skip_rate);
    return std::make_pair(C2R(aa), C2R(daa));
}


ArrayXXd
KS::intg(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
    assert(a0.size() == N-2);
    ArrayXXcd u0 = R2C(a0); 
    const int Nt = (int)round(tend/h);
    const int M = (Nt+skip_rate-1)/skip_rate;
    ArrayXXcd aa(N/2+1, M);
    Ts.resize(M);
    hs.resize(M);
    lte.resize(M);
    int count = 0;
    auto ss = [this, &count, &aa](ArrayXXcd &x, double t, double h, double err){
	int m = Ts.size();
	if (count >= m ) {
	    Ts.conservativeResize(m+cellSize);
	    aa.conservativeResize(Eigen::NoChange, m+cellSize); 
	    hs.conservativeResize(m+cellSize);
	    lte.conservativeResize(m+cellSize);
	}
	hs(count) = h;
	lte(count) = err;
	aa.col(count) = x;
	Ts(count++) = t;
    };
    
    eidr.intg(nl, ss, 0, u0, tend, h, skip_rate);
	
    hs.conservativeResize(count);
    lte.conservativeResize(count);
    Ts.conservativeResize(count);
    aa.conservativeResize(Eigen::NoChange, count);
    
    return C2R(aa);
}

std::pair<ArrayXXd, ArrayXXd>
KS::intgj(const ArrayXd &a0, const double h, const double tend, const int skip_rate){
    assert(a0.size() == N-2);
    ArrayXXd v0(N-2, N-1); 
    v0 << a0, MatrixXd::Identity(N-2, N-2);
    ArrayXXcd u0 = R2C(v0);
    
    const int Nt = (int)round(tend/h);
    const int M = (Nt + skip_rate - 1) / skip_rate;
    ArrayXXcd aa(N/2+1, M), daa(N/2+1, (N-2)*M);
    Ts.resize(M);
    hs.resize(M);
    lte.resize(M);
    int count = 0;
    auto ss = [this, &count, &aa, &daa](ArrayXXcd &x, double t, double h, double err){
	int m = Ts.size();
	if (count >= m ) {
	    Ts.conservativeResize(m+cellSize);	    
	    hs.conservativeResize(m+cellSize);
	    lte.conservativeResize(m+cellSize);
	    aa.conservativeResize(Eigen::NoChange, m+cellSize); // rows not change, just extend cols
	    daa.conservativeResize(Eigen::NoChange, (m+cellSize)*(N-2));
	}
	hs(count) = h;
	lte(count) = err;
	Ts(count) = t;
	aa.col(count) = x.col(0);
	daa.middleCols(count*(N-2), N-2) = x.rightCols(N-2);
	x.rightCols(N-2) = R2C(MatrixXd::Identity(N-2, N-2));
	count++;
    };

    eidr2.intg(nl2, ss, 0, u0, tend, h, skip_rate);
    return std::make_pair(C2R(aa), C2R(daa));
}

/* @brief complex matrix to the corresponding real matrix.
 * [N/2+1, M] --> [N-2, M]
 * Since the Map is not address continous, the performance is
 * not good enough.
 */
ArrayXXd KS::C2R(const ArrayXXcd &v){
    int rs = v.rows(), cs = v.cols();
    assert(N/2+1 == rs);
    ArrayXXcd vt = v.middleRows(1, rs-2);
    ArrayXXd vp = Map<ArrayXXd>((double*)(vt.data()), N-2, cs);
    return vp;
}

ArrayXXcd KS::R2C(const ArrayXXd &v){
    int rs = v.rows(), cs = v.cols();
    assert( N - 2 == rs); 
    ArrayXXcd vp(N/2+1, cs); 
    vp << ArrayXXcd::Zero(1, cs), 
	Map<ArrayXXcd>((dcp*)(v.data()), N/2-1, cs), 
	ArrayXXcd::Zero(1, cs);
    return vp;
}


/*************************************************** 
 *           stability ralated                     *
 ***************************************************/

/** @brief calculate the velocity 
 *
 * @param[in] a0 state vector
 * @return velocity field at a0
 */
ArrayXd
KS::velocity(const Ref<const ArrayXd> &a0){
    assert( N - 2 == a0.rows() );
    ArrayXXcd u = R2C(a0);
    ArrayXXcd v(N/2+1, 1);
    nl(u, v, 0);
    return C2R(L*u + v);
}

/* return v(x) + theta *t(x) */
ArrayXd
KS::velReq(const Ref<const VectorXd> &a0, const double theta){
    return velocity(a0) + theta * gTangent(a0);
}

/** @brief calculate the stability matrix 
 *    A = (qk^2 - qk^4) * v  - i*qk* F( F^{-1}(a0) * F^{-1}(v))
 */
MatrixXd KS::stab(const Ref<const ArrayXd> &a0){
    assert(a0.size() == N-2);
    ArrayXXd v0(N-2, N-1); 
    v0 << a0, MatrixXd::Identity(N-2, N-2);
    ArrayXXcd u0 = R2C(v0);
    
    ArrayXXcd v(N/2+1, N-1); 
    nl2(u0, v, 0); 
	
    ArrayXXcd j0 = R2C(MatrixXd::Identity(N-2, N-2));
    MatrixXcd Z = L.matrix().asDiagonal() * j0.matrix()  + v.rightCols(N-2).matrix();
    
    return C2R(Z);   
}

/* the stability matrix of a relative equilibrium */
MatrixXd KS::stabReq(const Ref<const VectorXd> &a0, const double theta){
    return stab(a0) + theta * gGenerator();
}

/* Eigenvalues/Eigenvectors of equilibrium */
std::pair<VectorXcd, MatrixXcd>
KS::evEq(const Ref<const VectorXd> &a0){
    return evEig(stab(a0), 1);
}

/* Eigenvalues/Eigenvectors of equilibrium */
std::pair<VectorXcd, MatrixXcd>
KS::evReq(const Ref<const VectorXd> &a0, const double theta){
    return evEig(stabReq(a0, theta), 1);
}

/*************************************************** 
 *           energe ralated                        *
 ***************************************************/
double KS::pump(const ArrayXcd &vc){
    VectorXcd tmp = vc * K;
    return tmp.squaredNorm();
}

double KS::disspation(const ArrayXcd &vc){
    VectorXcd tmp = vc * K * K;
    return tmp.squaredNorm();
}

/*************************************************** 
 *           Symmetry related                      *
 ***************************************************/

/** @brief apply reflection on each column of input  */
ArrayXXd KS::reflect(const Ref<const ArrayXXd> &aa){
    int n = aa.rows();
    int m = aa.cols();
    assert( 0 == n%2 );
  
    ArrayXd R(n);
    for(size_t i = 0; i < n/2; i++) {
	R(2*i) = -1;
	R(2*i+1) = 1;
    }
  
    ArrayXXd Raa = aa.colwise() * R;
    return Raa;
}


ArrayXXd KS::half2whole(const Ref<const ArrayXXd> &aa){
    int n = aa.rows();
    int m = aa.cols();
  
    ArrayXXd raa = reflect(aa);
    ArrayXXd aaWhole(n, 2*m);
    aaWhole << aa, raa;
  
    return aaWhole;
}

/** @brief apply rotation to each column of input with the
 *         same angle specified
 *    a_k => a_k * e^{ik*th}
 */
ArrayXXd KS::rotate(const Ref<const ArrayXXd> &aa, const double th){
    assert(aa.rows() == N - 2);
    ArrayXd indices = ArrayXd::LinSpaced(N/2+1, 0, N/2); // different from K
    return C2R( R2C(aa).colwise() * (dcp(0, th) * indices).exp() );
}

/** @brief group tangent of SO(2)
 *
 *  x=(b1, c1, b2, c2, ...) ==> tx=(-c1, b1, -2c2, 2b2, ...)
 *  That is a_k => a_k * {ik}
 */
ArrayXXd KS::gTangent(const Ref<const ArrayXXd> &x){
    assert(x.rows() == N - 2);
    ArrayXd indices = ArrayXd::LinSpaced(N/2+1, 0, N/2); // different from K
    return  C2R(R2C(x).colwise() * (dcp(0, 1) * indices) );
}

/* group generator matrix T */
MatrixXd KS::gGenerator(){
    MatrixXd T(MatrixXd::Zero(N-2, N-2));
    for (int i = 0; i < N/2-1; i++ ){
	T(2*i, 2*i+1) = -(i+1);
	T(2*i+1, 2*i) = i+1;
    }
    return T;
}

/**
 * @brief transform the SO2 symmetry to C_p.
 *
 * Use p-th mode to reduce SO2 symmetry. If p > 1, then the symmetry is not reduced
 * but tranformed to cyclic symmetry.
 *
 * For example, if p = 2, then SO2 is reduced to C2, so together with reflection,
 * we have D2 symmetry. If we rotate to the imaginary axis, then we have rules
 * R : ( - +  - +  - + ...)
 * C2: ( - -  + +  - - ...)
 * 
 *
 * @param[in]   p          index of Fourier mode 
 * @param[in]  toY         True => tranform to positive imaginary axis.
 */
std::pair<MatrixXd, VectorXd>
KS::redSO2(const Ref<const MatrixXd> &aa, const int p, const bool toY){
    int n = aa.rows();
    int m = aa.cols();
    assert( 0 == n%2 && p > 0);
    MatrixXd raa(n, m);
    VectorXd ang(m);
    
    double s = toY ? M_PI/2 : 0; /* pi/2 or 0 */
    
    for(int i = 0; i < m; i++){
	double th = (atan2(aa(2*p-1, i), aa(2*p-2, i)) - s )/ p;
	ang(i) = th;
	raa.col(i) = rotate(aa.col(i), -th);
    }
    return std::make_pair(raa, ang);
}

/**
 * If we use the 2nd mode to reduce SO2, then we use 1st mode or 3rd mode
 * to define fundamental domain. b_1 > 0 && c_1 > 0
 *
 * If we use the 1st mode to reduce SO2, then we use 2nd mode
 * to define fundamental domain. b_2 > 0
 *
 * In all these cases, we assume SO2 is reduced to positive y-axis.
 *
 * @param[in] pSlice     index of Fourier mode used to define slice
 * @param[in] pFund      index of Fourier mode used to define fundamental domain
 * 
 */
std::pair<MatrixXd, VectorXi>
KS::fundDomain(const Ref<const MatrixXd> &aa, const int pSlice, const int pFund){
    int n = aa.rows();
    int m = aa.cols();
    assert( 0 == n%2 && pSlice > 0 && pFund > 0 && pSlice != pFund);
 
    MatrixXd faa(n, m);
    VectorXi DomainIds(m);
    
    ArrayXd R(n);		// reflection matrix
    for(int i = 0; i < n/2; i++){
	R(2*i) = -1;
	R(2*i+1) = 1;
    }
    
    if (pSlice == 1){		//  1st mode slice	
	int id = 2 * (pFund - 1);
	for (int i = 0; i < m; i++) {
	    if(aa(id, i) > 0 ) {
		faa.col(i) = aa.col(i);
		DomainIds(i) = 1;
	    }
	    else {
		faa.col(i) = aa.col(i).array() * R;
		DomainIds(i) = 2;
	    }
	}
    }
    
    else if (pSlice == 2){	// 2nd mode slice	
	ArrayXd C(n);		// rotation by pi
	int s = -1;
	for(int i = 0; i < n/2; i++){
	    C(2*i) = s;
	    C(2*i+1) = s;
	    s *= -1;
	}
    
	ArrayXd RC = R * C;
	
	int id = 2 * (pFund - 1);
	for(int i = 0; i < m; i++){
	    bool x = aa(id, i) > 0;
	    bool y = aa(id+1, i) > 0;
	    if(x && y) {
		faa.col(i) = aa.col(i);
		DomainIds(i) = 1;
	    }
	    else if (!x && y) {
		faa.col(i) = aa.col(i).array() * R;
		DomainIds(i) = 2;
	    }
	    else if (!x && !y) {
		faa.col(i) = aa.col(i).array() * C;
		DomainIds(i) = 3;
	    }
	    else { // if (x && !y) {
		faa.col(i) = aa.col(i).array() * RC;
		DomainIds(i) = 4;
	    }
	}
	
    }

    else if (pSlice == 3){ 
	ArrayXcd C1(n/2+2), C2(n/2+2);
	for (int i = 0; i < n/2; i++){
	    double th = (i+1) * 2 * M_PI / 3;
	    C1[i+1] = dcp(cos(th), sin(th));
	    C2[i+1] = dcp(cos(2*th), sin(2*th));
	}

	int id = 2 * (pFund - 1); 

	for(int i = 0; i < m; i++){
	    int x = aa(id, i), y = aa(id+1, i);
	    if ( y <= x/2 && y >= -x/2){ // domain 1
		faa.col(i) = aa.col(i);
		DomainIds(i) = 1;
	    }
	    else if (y >= x/2 && y <= -x/2){ // domain 4
		faa.col(i) = aa.col(i).array() * R;
		DomainIds(i) = 4;
	    }
	    else if (y >= x/2 && x >= 0) { // domain 2 
		faa.col(i) = C2R(C1 * R2C(aa.col(i)).array()) * R;
		DomainIds(i) = 2;
	    }
	    else if(y <= x/2 && x <= 0) { // domain 5 
		faa.col(i) = C2R(C1 * R2C(aa.col(i)).array());
		DomainIds(i) = 5; 
	    }
	    else if (y >= -x/2 && x <= 0){ // domain 3
		faa.col(i) = C2R(C2 * R2C(aa.col(i)).array());
		DomainIds(i) = 3;
	    }
	    else {		// domain 6
		faa.col(i) = C2R(C2 * R2C(aa.col(i)).array()) * R;
		DomainIds(i) = 6;
	    }
	}
    }
    
    else {
	fprintf(stderr, "wrong index of Fourier mode !\n");
    }

    return std::make_pair(faa, DomainIds);
}
    
/** @brief reduce O2 symmetry to fundamental domain
 *
 *
 * @param[in] aa         states in the full state space
 * @param[in] pSlice     index of Fourier mode used to define slice
 * @param[in] pFund      index of Fourier mode used to define fundamental domain
 * @return    state in fundamental domain, domain indices, angles
 **/
std::tuple<MatrixXd, VectorXi, VectorXd>
KS::redO2f(const Ref<const MatrixXd> &aa, const int pSlice, const int pFund){
    auto tmp = redSO2(aa, pSlice, true);
    auto tmp2 = fundDomain(tmp.first, pSlice, pFund);

    return std::make_tuple(tmp2.first, tmp2.second, tmp.second);
}

// projection matrix is h= (I - |tx><tp|/<tx|tp>) * g(-th)
MatrixXd
KS::redV(const Ref<const MatrixXd> &v, const Ref<const VectorXd> &a,
	 const int p, const bool toY){
    auto tmp = redSO2(a, p, toY);
    MatrixXd &aH = tmp.first;
    double th = tmp.second(0);
    VectorXd tx = gTangent(aH);

    VectorXd x0(VectorXd::Zero(N-2));
    if (toY) x0(2*p-1) = 1;
    else x0(2*p-2) = 1;    
    VectorXd t0 = gTangent(x0);
    
    MatrixXd vep = rotate(v, -th);
    MatrixXd dot = (t0.transpose() * vep) / (t0.transpose() * tx);
    vep = vep - tx * dot;
    
    return vep;
}

#if 0
/** @beief project the sequence of Floquet vectors to 1st mode slice
 *
 *  Usaully, aa has one more column the the Floquet vectors, so you can
 *  call this function like:
 *  \code
 *      veToSliceAll(eigVecs, aa.leftCols(aa.cols()-1))
 *  \endcode
 *  
 *  @param[in] eigVecs Floquet vectors along the orbit. Dimension: [N*Trunc, M]
 *  @param[in] aa the orbit
 *  @return projected vectors on the slice with dimension [N, M*Trunc]
 *
 *  @note vectors are not normalized
 */
MatrixXd KS::veToSliceAll(const MatrixXd &eigVecs, const MatrixXd &aa,
			  const int trunc /* = 0*/){
    int Trunc = trunc;
    if(trunc == 0) Trunc = sqrt(eigVecs.rows());

    assert(eigVecs.rows() % Trunc == 0);
    const int n = eigVecs.rows() / Trunc ;  
    const int m = eigVecs.cols();
    const int n2 = aa.rows();
    const int m2 = aa.cols();

    assert(m == m2 && n == n2);
    MatrixXd newVe(n, Trunc*m);
    for(size_t i = 0; i < m; i++){
	MatrixXd ve = eigVecs.col(i);
	ve.resize(n, Trunc);
	newVe.middleCols(i*Trunc, Trunc) = veToSlice(ve, aa.col(i));
    }

    return newVe;
}


/**
 * @brief get the full orbit and full set of Fvs
 * 
 * Given the inital point and the set of Fvs, the whole set is
 * twice bigger for ppo, but for rpo, it is the single piece.
 *
 * @param[in] a0       the inital condition of the ppo/rpo
 * @param[in] ve       the set of Fvs. Dimension [(N-2)*NFV, M]
 * @param[in] nstp     number of integration steps
 * @param[in] ppTpe    ppo/rpo
 * @return             the full orbit and full set of Fvs.
 */
std::pair<ArrayXXd, ArrayXXd>
KS::orbitAndFvWhole(const ArrayXd &a0, const ArrayXXd &ve,
		    const double h,
		    const size_t nstp, const std::string ppType
		    ){
    assert(N-2 == a0.rows());
    const int M = ve.cols();
    assert(nstp % M == 0);
    const int space = nstp / M;
    
    ArrayXXd aa = intg(a0, h, nstp, space);
    if(ppType.compare("ppo") == 0) 
	return std::make_pair(
			      half2whole(aa.leftCols(aa.cols()-1)),
			      half2whole(ve) // this is correct. Think carefully.
			      );
    
    else 
	return std::make_pair(
			      aa.leftCols(aa.cols()-1), // one less
			      ve
			      );
}

/**
 * @brief get rid of the marginal direction of the Fvs
 * 
 * @param[in] ve        dimension [N, M*trunc]
 * @param[in] pos       the location of the group tangent margianl Floquet vector.
 *                      pos starts from 0.
 * @param[in] return    the clean set of Fvs
 */
MatrixXd KS::veTrunc(const MatrixXd ve, const int pos, const int trunc /* = 0 */){
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


/**
 * @brief get the full orbit and full set of Fvs on the slice
 *
 * @return             the full orbit and full set of Fvs on slice
 * @see                orbitAndFvWhole(),  veTrunc()
 */
std::pair<ArrayXXd, ArrayXXd>
KS::orbitAndFvWholeSlice(const ArrayXd &a0, const ArrayXXd &ve,
			 const double h,
			 const size_t nstp, const std::string ppType,
			 const int pos
			 ){
    assert(ve.rows() % (N-2) == 0);
    const int NFV = ve.rows() / (N-2);
    auto tmp = orbitAndFvWhole(a0, ve, h, nstp, ppType);
    auto tmp2 = orbitToSlice(tmp.first);
    MatrixXd veSlice = veToSliceAll(tmp.second, tmp.first, NFV);
    MatrixXd ve_trunc = veTrunc(veSlice, pos, NFV);

    return std::make_pair(tmp2.first, ve_trunc);
}

#endif
/*************************************************** 
 *                  Others                         *
 ***************************************************/

/* calculate mode modulus */
MatrixXd KS::calMag(const Ref<const MatrixXd> &aa){
    assert(aa.rows() == N - 2);
    ArrayXXcd u = R2C(aa).middleRows(1, N/2-1);
    return (u * u.conjugate()).real();
}

std::pair<MatrixXd, MatrixXd>
KS::toPole(const Ref<const MatrixXd> &aa){
    assert(aa.rows() == N - 2);
    ArrayXXcd u = R2C(aa).middleRows(1, N/2-1);
    
    MatrixXd mag = (u * u.conjugate()).real();
    MatrixXd th(u.rows(), u.cols());    
    for(int i = 0; i < u.rows(); i++){
	for(int j = 0; j < u.cols(); j++){
	    th(i, j) = arg(u(i, j));
	}
    }
    
    return std::make_pair(mag, th);

}


