#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>

#include "ksint.hpp"
#include "ksintM1.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


class pyKS : public KS {
public:
    pyKS(int N, double h, double d) : KS(N, h, d) {}
    
    /* wrap the velocity */
    bn::ndarray PYvelocity(bn::ndarray a0){
	Map<ArrayXd> tmpa((double*)a0.get_data(), N-2);
	VectorXd v = velocity(tmpa);
	
	Py_intptr_t dims[1] = { N-2 };
	bn::ndarray result = 
	    bn::empty(1, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)result.get_data(), (void*)(&v(0)), 
	       sizeof(double)*(N-2));
	return result;
    }

    /* wrap the integrator */
    bn::ndarray PYintg(bn::ndarray a0, size_t nstp, size_t np){
	Map<ArrayXd> tmpa((double*)a0.get_data(), N-2);
	ArrayXXd aa = intg(tmpa, nstp, np);
	
	Py_intptr_t dims[2] = { (int)(nstp/np+1), N-2 };
	bn::ndarray result = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)result.get_data(), (void*)(&aa(0, 0)), 
	       sizeof(double) * (N-2) * (nstp/np+1) );
	return result;
    }

    /* wrap the integrator with Jacobian */
    bp::tuple PYintgj(bn::ndarray a0, size_t nstp, size_t np, size_t nqr){
	Map<ArrayXd> tmpa((double*)a0.get_data(), N-2);
	std::pair<ArrayXXd, ArrayXXd> tmp = intgj(tmpa, nstp, np, nqr);
	
	Py_intptr_t dims[2] = { nstp/np+1, N-2 };
	bn::ndarray aa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	Py_intptr_t dims2[2] = { nstp/nqr, (N-2)*(N-2)};
	bn::ndarray daa = 
	    bn::empty(2, dims2, bn::dtype::get_builtin<double>());
	memcpy((void*)aa.get_data(), (void*)(&tmp.first(0, 0)), 
	       sizeof(double) * (N-2) * (nstp/np+1) );
	memcpy((void*)daa.get_data(), (void*)(&tmp.second(0, 0)), 
	       sizeof(double) * (nstp/np) * (N-2)*(N-2) );

	return bp::make_tuple(aa, daa);
    }
};

class pyKSM1 : public KSM1 {
public :
    pyKSM1(int N, double h, double d) : KSM1(N, h, d) {}
    
    /* wrap the integrator */
    bp::tuple PYintg(bn::ndarray a0, size_t nstp, size_t np){
	Map<ArrayXd> tmpa((double*)a0.get_data(), N-2);
	std::pair<ArrayXXd, ArrayXd> tmp = intg(tmpa, nstp, np);
	
	Py_intptr_t dims[2] = { (int)(nstp/np+1), N-2 };
	bn::ndarray aa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	bn::ndarray tt = 
	    bn::empty(1, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)aa.get_data(), (void*)(&tmp.first(0, 0)), 
	       sizeof(double) * (N-2) * (nstp/np+1) );
	memcpy((void*)tt.get_data(), (void*)(&tmp.second(0)), 
	       sizeof(double) * (nstp/np+1) );

	return bp::make_tuple(aa, tt);
    }
};

BOOST_PYTHON_MODULE(py_ks) {
    bn::initialize();    
    bp::class_<KS>("KS") ;
    bp::class_<KSM1>("KSM1") ;
    
    bp::class_<pyKS, bp::bases<KS> >("pyKS", bp::init<int, double, double>())
	.def_readonly("N", &pyKS::N)
	.def_readonly("d", &pyKS::d)
	.def_readonly("h", &pyKS::h)
	.def("velocity", &pyKS::PYvelocity)
	.def("intg", &pyKS::PYintg)
	.def("intgj", &pyKS::PYintgj)
	;

    bp::class_<pyKSM1, bp::bases<KSM1> >("pyKSM1", bp::init<int, double, double>())
	.def_readonly("N", &pyKS::N)
	.def_readonly("d", &pyKS::d)
	.def_readonly("h", &pyKS::h)
	.def("intg", &pyKSM1::PYintg)
	;
}
