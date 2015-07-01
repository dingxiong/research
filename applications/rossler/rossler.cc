/*
 * g++ rossler.cc -O3 -std=c++11 -I $XDAPPS/sources/boost_1_57_0 -I $XDAPPS/eigen/include/eigen3 -I $RESH/include -L $RESH/lib -literMethod
 */
#include <iostream>
#include <vector>
#include <boost/numeric/odeint.hpp>
#include <Eigen/Dense>
#include <functional>
#include "iterMethod.hpp"

using namespace std;
using namespace Eigen;
using namespace boost::numeric::odeint;
using namespace iterMethod;
namespace ph = std::placeholders;

class Rossler {
    
protected:
    //////////////////////////////////////////////////////////////////////
    //                              Inner classes                       //
    //////////////////////////////////////////////////////////////////////

    /**
     * @brief velocity of Rossler system
     *
     *   dx/dt = - y - z
     *   dy/dt = x + a * y
     *   dz/dt = b + z * (x - c)
     */
    struct Velocity{
	const double a, b, c;
	Velocity (double a = 0.2, double b=0.2, double c = 5.7) : a(a), b(b), c(c) {}
	void operator()(const std::vector<double> &x, std::vector<double> &dxdt,
			const double /* t */){
	    dxdt[0] = -x[1] - x[2];
	    dxdt[1] = x[0] + a * x[1];
	    dxdt[2] = b + x[2] * (x[0] - c);
	}
    };
    
    /**
     * @brief calculate J * dx for Rossler
     *
     *               | 0, -1, -1  |
     *  stability =  | 1,  a,  0  |
     *               | z,  0, x-c |
     *
     */
    struct Jacv{
	const double a, b, c;
	Jacv (double a = 0.2, double b = 0.2, double c = 5.7) : a(a), b(b), c(c) {}
	void operator()(const std::vector<double> &x, std::vector<double> &dxdt,
			const double /* t */){
	    dxdt[0] = -x[1] - x[2];
	    dxdt[1] = x[0] + a * x[1];
	    dxdt[2] = b + x[2] * (x[0] - c);
	    
	    dxdt[3] = -x[4] - x[5];
	    dxdt[4] = x[3] + a*x[4];
	    dxdt[5] = x[2]*x[3] + (x[0]-c)*x[5];
	}
	
    };


public:
    
    const double a, b, c;
    Velocity velocity;
    Jacv jacv;
    double odeAtol, odeRtol;


    Rossler(double odeAtol = 1e-16, double odeRtol = 1e-12,
	    double a = 0.2, double b=0.2, double c=5.7) :
	a(a), b(b), c(c),
	velocity(a, b, c),
	jacv(a, b, c),
	odeAtol(odeAtol), odeRtol(odeRtol)
    {}

    Vector3d getVelocity(Array3d x0){
	std::vector<double> x(&x0[0], &x0[0] + x0.size());
	std::vector<double> v(3);
	velocity(x, v, 0);
	Vector3d vel = Map<Vector3d>(&v[0]);
	return vel; 
    }
    
    /**
     * @brief integrator : each column is a state vector
     */
    template< typename Vel>
    std::pair<ArrayXXd, ArrayXd>
    intg0(const ArrayXd &x0, const double h, const int nstp, Vel vel, int blockSize = 100){
	ArrayXXd xx(x0.size(), blockSize);
	ArrayXd tt(blockSize);
	int totalNum = 0;
	std::vector<double> x(&x0[0], &x0[0] + x0.size());
	//integrate_adaptive(make_controlled< runge_kutta_cash_karp54<std::vector<double>> >( odeAtol , odeRtol ) ,
	integrate_const( runge_kutta4< std::vector<double> >() , 
			 // integrate(
			 vel, x, 0.0, nstp*h, h,
			 [&xx, &tt, &totalNum, &blockSize](const std::vector<double> &x, double t){		      
			     if(totalNum >= tt.size()){
				 xx.conservativeResize(NoChange, xx.cols() + blockSize);
				 tt.conservativeResize(tt.size() + blockSize);
			     }
			     xx.col(totalNum) = ArrayXd::Map(&x[0], x.size());
			     tt(totalNum++) = t;
			 });
       
	return make_pair(xx.leftCols(totalNum), tt.head(totalNum));
    }

    std::pair<ArrayXXd, ArrayXd>
    intg(const ArrayXd &x0, const double h, const int nstp, int blockSize = 100){
	return intg0(x0, h, nstp, velocity, blockSize);
    }

    std::tuple<ArrayXXd, ArrayXXd, ArrayXd>
    intgv(const ArrayXd &x0, const ArrayXd &dx0, const double h,
	  const int nstp, int blockSize = 100){
	ArrayXd xdx(x0.size() + dx0.size());
	xdx << x0, dx0;
	auto tmp = intg0(xdx, h, nstp, jacv, blockSize);
	return std::make_tuple(tmp.first.topRows(3),
			       tmp.first.bottomRows(3),
			       tmp.second);
    }

    /**
     * @brief form f(x,t) - x
     * @param[in] x   4-element vector: (x, t)
     * @return    4-element vector
     *               | f(x, t) - x|
     *               |     0      |
     */
    Vector4d Fx(Vector4d x){
	int nstp =  (int)(x(3) / 0.01); 
	Array3d fx = intg(x.head(3), x(3)/nstp, nstp).first.rightCols(1);
	Vector4d F; 
	F << fx, x(3);
	return F - x;
    }

    /**
     * @brief get the product J * dx
     *
     * Here J = | J(x, t) - I,  v(f(x,t)) | 
     *          |     v(x),          0    |
     */

    Vector4d DFx(Vector4d x, Vector4d dx){
	int nstp = (int)(x(3) / 0.01);
	double norm = dx.head(3).norm();
	auto tmp = intgv(x.head(3), dx.head(3), x(3)/nstp, nstp);
	VectorXd v1 = getVelocity(x.head(3));
	VectorXd v2 = getVelocity(std::get<0>(tmp).rightCols(1));
	Vector4d DF;
	DF << std::get<1>(tmp).rightCols(1).matrix() - dx.head(3) + v2 * dx(3),
	    v1.dot(dx.head(3));
	return DF;
    }

#if 0
    VectorXd DFx(Vector4d x, Vector4d dx){
	int nstp = (int)(x(3) / 0.01);
	auto tmp = intg(x.head(3), x(3)/nstp, nstp);
	double norm = dx.head(3).norm();
	auto tmp2 = intg(x.head(3) + 1e-7*dx.head(3)/norm, x(3)/nstp, nstp);
	Vector3d Jx = (tmp2.first.rightCols(1) - tmp.first.rightCols(1)) / 1e-7 * norm;
	VectorXd v1 = getVelocity(x.head(3));
	VectorXd v2 = getVelocity(tmp.first.rightCols(1));
	Vector4d DF;
	DF << Jx - dx.head(3) + v2 * dx(3),
	    v1.dot(dx.head(3));
	return DF;
    }

    #endif
};


int main(){
    
    switch (1){
	
    case 1 :{
	Rossler ros;
	Array3d x0;
	//x0 << 1, 6.0918, 1.2997;
	//x0 << -3.36773  ,  5.08498  ,  0.0491195;
	// T = 18
	x0 << -3.36788  ,  5.08677  ,  0.0240049;
	double T = 17.5959;
	int nstp = int(T/0.01);
	    
	Array3d dx0;
	dx0 << 0.2, 0.2, 0.2;
	auto tmp = ros.intgv(x0, dx0, T/nstp, nstp);
	cout << std::get<0>(tmp) << endl;
	break;
    }

    case 2 : {
	Rossler ros;
	Array3d x0;
	//x0 << 1, 6.0918, 1.2997;
	x0 << -3.36773  ,  5.08498  ,  0.0491195;   
	Array3d dx0;
	dx0 << 0.1, 0.1, 0.1;
	auto tmp = ros.intg(x0, 0.01, 1800);
	cout << std::get<1>(tmp) << endl;
	break;
	
    }
	
    case 3 : {
	Rossler ros;
	Vector4d x0;
	//x0 << 1, 6.0918, 1.2997, 5.88;
	x0  <<  -3.36773  ,  5.08498  ,  0.0491195, 18;   
	auto tmp = ros.Fx(x0);
	//cout << tmp << endl;
	
	/* Vector4d dx0; */
	/* dx0 << 0.1, 0.1, 0.1, 0.1; */
	/* tmp = ros.DFx(x0, dx0); */
	/* cout << tmp << endl; */
	break;
    }
	
    case 4: {			// find POs
	Rossler ros;
	Vector4d x0;
	//x0 << 1, 6.0918, 1.2997, 5.88;
	x0  <<  -3.36773  ,  5.08498  ,  0.0491195, 18;
	
	auto f = std::bind(&Rossler::Fx, ros, ph::_1);
	auto df = std::bind(&Rossler::DFx, ros, ph::_1, ph::_2);
	auto result = InexactNewtonBacktrack(f, df, x0, 1e-12, 10, 10);
	cout << std::get<0>(result) << endl;
	break;
    }
	
    }
    return 0;
}
