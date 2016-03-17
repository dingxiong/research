#include<iostream>
#include <cmath>
#include "lorenz.hpp"

using namespace std;
using namespace Eigen;
using namespace denseRoutines;

Lorenz::Lorenz(){
}

Lorenz::~Lorenz(){
}

Lorenz & Lorenz::operator=(const Lorenz &x){
    return *this;
}

/* ====================================================================== */

/* velocity field of Lorenz flow
 *        
 *        | sigma * (y -x)     |
 * v(x) = | rho * y - y - x*z  |
 *        | x*y - b*z          |
 */

Vector3d
Lorenz::vel(const Ref<const Vector3d> &x){
    Vector3d v;
    v <<
	Sigma * (x(1) - x(0)),
	Rho * x(0) - x(1) - x(0)*x(2),
	x(0)*x(1) - B * x(2);

    return v;
}

/* stability matrix  */
Matrix3d
Lorenz::stab(const Ref<const Vector3d> &x){
    Matrix3d A;
    A << 
	-Sigma,    Sigma,   0,
	Rho-x(2),   -1,     -x(0),
	x(1),       x(0),   -B;
    
    return A;
}

MatrixXd
Lorenz::velJ(const Ref<const MatrixXd> &x){
    MatrixXd vJ(3, 4);
    vJ << 
	vel(x.col(0)),
	stab(x.col(0)) * x.rightCols(3);

    return vJ;
}

MatrixXd
Lorenz::intg(const Ref<const Vector3d> &x0, const double h, 
	     const int nstp, const int nq){

    int M = nstp/nq + 1;
    MatrixXd xx(3, M);
    xx.col(0) = x0;

    Vector3d x(x0);


    for (int i = 0; i < nstp; i++){
	Vector3d k1 = vel(x); 
	Vector3d k2 = vel(x + h/2*k1);
	Vector3d k3 = vel(x + h/2*k2);
	Vector3d k4 = vel(x + h*k3);
	
	x += h/6 * (k1 + 2*k2 + 2*k3 + k4);
	
	if((i+1)%nq == 0) xx.col((i+1)/nq) = x;
    }

    return xx;
}

std::pair<MatrixXd, MatrixXd>
Lorenz::intgj(const Ref<const Vector3d> &x0, const double h,
	      const int nstp, const int xs, const int js){
    int M1 = nstp / xs + 1;
    int M2 = nstp / js;
    
    MatrixXd xx(3, M1);
    MatrixXd JJ(3, 3*M2);
    xx.col(0) = x0;

    MatrixXd x(3, 4);
    x << x0, MatrixXd::Identity(3,3);  

    for (int i = 0; i < nstp; i++){
	MatrixXd k1 = velJ(x); 
	MatrixXd k2 = velJ(x + h/2*k1);
	MatrixXd k3 = velJ(x + h/2*k2);
	MatrixXd k4 = velJ(x + h*k3);
	
	x += h/6 * (k1 + 2*k2 + 2*k3 + k4);
	
	if ((i+1)%xs == 0) xx.col((i+1)/xs) = x.col(0);
	if ((i+1)%js == 0) JJ.middleCols((i+1)/js-1, 3) = x.rightCols(3);
    }

    return std::make_pair(xx, JJ);
}


/* ============================================================ */

/**
 * If rho > 1 there are 3 equilibria :
 *  Eq0 = [0, 0, 0]
 *  Eq1 = [\sqrt{b(rho-1)}, \sqrt{b(rho-1)}, rho-1]
 *  Eq2 = [-\sqrt{b(rho-1)}, -\sqrt{b(rho-1)}, rho-1]
 *
 *  @return each column is an equilibrium
 */
Matrix3d
Lorenz::equilibria(){
    double z = Rho - 1;
    double x = sqrt(B*z);
    
    Matrix3d E;
    E << 
	0, x, -x,
	0, x, -x,
	0, z, z;

    return E;
}

/* obtain the stability matrix of the ith equilibrium */
Matrix3d
Lorenz::equilibriaStab(const int i){
    Matrix3d E = equilibria();
    return stab(E.col(i));
}

/* obtain the eigenvalues/eigenvectors of the stability matrix of
 * the ith equilibrium 
 */
std::pair<VectorXcd, MatrixXcd>
Lorenz::equilibriaEV(const int i){
    Matrix3d A = equilibriaStab(i);
    return evEig(A);
}


MatrixXd
Lorenz::equilibriaIntg(const int i, const int j, const double eps,
		       const double h, const int nstp, 
		       const int nq){
    Vector3d x0 = equilibria().col(i);
    auto e = equilibriaEV(i);
    Vector3d v = e.second.col(j).real();
    
    return intg(x0 + eps*v, h, nstp, nq); 
}
