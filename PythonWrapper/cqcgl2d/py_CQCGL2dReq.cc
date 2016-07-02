#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "CQCGL2dReq.hpp"
#include "myBoostPython.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class pyCQCGL2dReq : public CQCGL2dReq {
  
public:

    pyCQCGL2dReq(int N, double dx,
		 double b, double c, double dr, double di,
		 int threadNum) :
	CQCGL2dReq(N, dx, b, c, dr, di, threadNum) {}
    
    bn::ndarray PYTs(){
	return copy2bn(Ts);
    }

    bn::ndarray PYlte(){
	return copy2bn(lte);
    }

    bn::ndarray PYhs(){
	return copy2bn(hs);
    }

    /* Kx */
    bn::ndarray PYKx(){
	return copy2bn(Kx);
    }

    /* Ky */
    bn::ndarray PYKy(){
	return copy2bn(Ky);
    }
    
    /* L */
    bn::ndarray PYL(){
	return copy2bnc(L);
    }

   /* wrap the integrator */
    bn::ndarray PYintg(bn::ndarray a0, double h, int Nt, int skip_rate, 
		       bool doSaveDisk, string fileName){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXXcd> tmpa((dcp*)a0.get_data(), n, m);
	return copy2bnc(intg(tmpa, h, Nt, skip_rate, doSaveDisk, fileName));
    }
    
    bn::ndarray PYaintg(bn::ndarray a0, double h, double tend, int skip_rate,
			bool doSaveDisk, string fileName){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXXcd> tmpa((dcp*)a0.get_data(), n, m);
	return copy2bnc(aintg(tmpa, h, tend, skip_rate, doSaveDisk, fileName));
    }

    bn::ndarray PYintgv(bn::ndarray a0, bn::ndarray v0, double h, int Nt, int skip_rate,
			bool doSaveDisk, string fileName){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXXcd> tmpa((dcp*)a0.get_data(), n, m);
	getDims(v0, m, n);
	Map<ArrayXXcd> tmpv((dcp*)v0.get_data(), n, m);
	return copy2bnc(intgv(tmpa, tmpv, h, Nt, skip_rate, doSaveDisk, fileName));
    }    

    bn::ndarray PYaintgv(bn::ndarray a0, bn::ndarray v0, double h, double tend, int skip_rate,
			 bool doSaveDisk, string fileName){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXXcd> tmpa((dcp*)a0.get_data(), n, m);
	getDims(v0, m, n);
	Map<ArrayXXcd> tmpv((dcp*)v0.get_data(), n, m);
	return copy2bnc(aintgv(tmpa, tmpv, h, tend, skip_rate, doSaveDisk, fileName));
    }

    bn::ndarray PYFourier2Config(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXcd> tmpaa((dcp*)aa.get_data(), n, m);
	return copy2bnc( Fourier2Config(tmpaa) );
    }

    bn::ndarray PYConfig2Fourier(bn::ndarray AA){
	int m, n;
	getDims(AA, m, n);
	Map<ArrayXXcd> tmpAA((dcp*)AA.get_data(), n, m);
	return copy2bnc( Config2Fourier(tmpAA) );
    }

    bn::ndarray PYvelocity(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXXcd> tmpa((dcp*)a0.get_data(), n, m);
	return copy2bnc(velocity(tmpa));
    }

    bn::ndarray PYvelocityReq(bn::ndarray a0, double wthx, 
			      double wthy, double wphi){
	int m, n;
	getDims(a0, m, n);	
	Map<ArrayXXcd> tmpa((dcp*)a0.get_data(), n, m);
	return copy2bnc(velocityReq(tmpa, wthx, wthy, wphi));
    }

    bp::tuple PYfindReq_hook(bn::ndarray x0, double wthx0, double wthy0, double wphi0){
	int m, n;
	getDims(x0, m, n);
	Map<ArrayXXcd> tmpx0((dcp*)x0.get_data(), n, m);
	
	auto result = findReq_hook(tmpx0, wthx0, wthy0, wphi0);
	return bp::make_tuple(copy2bnc(std::get<0>(result)),
			      std::get<1>(result),
			      std::get<2>(result),
			      std::get<3>(result),
			      std::get<4>(result)
			      );
    }
    
    bn::ndarray PYoptReqTh(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXXcd> tmpa((dcp*)a0.get_data(), n, m);
	
	return copy2bn(optReqTh(tmpa));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



BOOST_PYTHON_MODULE(py_CQCGL2dReq) {
    bn::initialize();
    
    // must provide the constructor
    bp::class_<CQCGL2dReq>("CQCGL2dReq", bp::init<
			   int, double, 
			   double, double, double, double, 
			   int>())
	;
    
    
    bp::class_<pyCQCGL2dReq, bp::bases<CQCGL2dReq> >("pyCQCGL2dReq", bp::init<
						     int, double, 
						     double, double, double, double,
						     int>())
	.def_readonly("N", &pyCQCGL2dReq::N)
	.def_readonly("M", &pyCQCGL2dReq::M)
	.def_readonly("dx", &pyCQCGL2dReq::dx)
	.def_readonly("dy", &pyCQCGL2dReq::dy)
	.def_readonly("Mu", &pyCQCGL2dReq::Mu)
	.def_readonly("Br", &pyCQCGL2dReq::Br)
	.def_readonly("Bi", &pyCQCGL2dReq::Bi)
	.def_readonly("Dr", &pyCQCGL2dReq::Dr)
	.def_readonly("Di", &pyCQCGL2dReq::Di)
	.def_readonly("Gr", &pyCQCGL2dReq::Gr)
	.def_readonly("Gi", &pyCQCGL2dReq::Gi)
	.def_readonly("Ne", &pyCQCGL2dReq::Ne)
	.def_readonly("Me", &pyCQCGL2dReq::Me)
	.def_readonly("b", &pyCQCGL2dReq::b)
	.def_readonly("c", &pyCQCGL2dReq::c)
	.def_readonly("dr", &pyCQCGL2dReq::dr)
	.def_readonly("di", &pyCQCGL2dReq::di)
	.def_readonly("Omega", &pyCQCGL2dReq::Omega)
	.def_readwrite("rtol", &pyCQCGL2dReq::rtol)
	.def_readwrite("nu", &pyCQCGL2dReq::nu)
	.def_readwrite("mumax", &pyCQCGL2dReq::mumax)
	.def_readwrite("mumin", &pyCQCGL2dReq::mumin)
	.def_readwrite("mue", &pyCQCGL2dReq::mue)
	.def_readwrite("muc", &pyCQCGL2dReq::muc)
	.def_readwrite("NCalCoe", &pyCQCGL2dReq::NCalCoe)
	.def_readwrite("NReject", &pyCQCGL2dReq::NReject)
	.def_readwrite("NCallF", &pyCQCGL2dReq::NCallF)
	.def_readwrite("NSteps", &pyCQCGL2dReq::NSteps)
	.def_readwrite("Method", &pyCQCGL2dReq::Method)
	.def_readwrite("constETDPrint", &pyCQCGL2dReq::constETDPrint)

	.def_readwrite("tol", &pyCQCGL2dReq::tol)
	.def_readwrite("minRD", &pyCQCGL2dReq::minRD)
	.def_readwrite("maxit", &pyCQCGL2dReq::maxit)
	.def_readwrite("maxInnIt", &pyCQCGL2dReq::maxInnIt)
	.def_readwrite("GmresRtol", &pyCQCGL2dReq::GmresRtol)
	.def_readwrite("GmresRestart", &pyCQCGL2dReq::GmresRestart)
	.def_readwrite("GmresMaxit", &pyCQCGL2dReq::GmresMaxit)

	.def("Ts", &pyCQCGL2dReq::PYTs)
	.def("hs", &pyCQCGL2dReq::PYhs)
	.def("lte", &pyCQCGL2dReq::PYlte)
	.def("Kx", &pyCQCGL2dReq::PYKx)
	.def("Ky", &pyCQCGL2dReq::PYKy)
	.def("L", &pyCQCGL2dReq::PYL)

	.def("changeOmega", &pyCQCGL2dReq::changeOmega)
	.def("intg", &pyCQCGL2dReq::PYintg)
	.def("aintg", &pyCQCGL2dReq::PYaintg)
	.def("intgv", &pyCQCGL2dReq::PYintgv)
	.def("aintgv", &pyCQCGL2dReq::PYaintgv)
	.def("Fourier2Config", &pyCQCGL2dReq::PYFourier2Config)
	.def("Config2Fourier", &pyCQCGL2dReq::PYConfig2Fourier)
	.def("velocity", &pyCQCGL2dReq::PYvelocity)
	.def("velocityReq", &pyCQCGL2dReq::PYvelocityReq)
	.def("findReq_hook", &pyCQCGL2dReq::PYfindReq_hook)
	.def("optReqTh", &pyCQCGL2dReq::PYoptReqTh) 
	;

}
