#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "ksintM1.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

class pyKSM1 : public KSM1 {
public :
    pyKSM1(int N, double h, double d) : KSM1(N, h, d) {}
    
    /* wrap the integrator */
    bp::tuple PYintg(bn::ndarray a0, size_t nstp, size_t np){
	Map<ArrayXd> tmpa((double*)a0.get_data(), N-2);
	std::pair<ArrayXXd, ArrayXd> tmp = intg(tmpa, nstp, np);
	
	Py_intptr_t dims[2] = { nstp/np+1, N-2 };
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

    /* wrap the second integrator */
    bp::tuple PYintg2(bn::ndarray a0, double T, size_t np){
	Map<ArrayXd> tmpa((double*)a0.get_data(), N-2);
	std::pair<ArrayXXd, ArrayXd> tmp = intg2(tmpa, T, np);
	
	int n = tmp.first.rows();
	int m = tmp.first.cols();
		
	Py_intptr_t dims[2] = { m , n };
	bn::ndarray aa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	bn::ndarray tt = 
	    bn::empty(1, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)aa.get_data(), (void*)(&tmp.first(0, 0)), 
	       sizeof(double) * n * m );
	memcpy((void*)tt.get_data(), (void*)(&tmp.second(0)), 
	       sizeof(double) * m );

	return bp::make_tuple(aa, tt);
    }
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

BOOST_PYTHON_MODULE(py_ks) {
    bn::initialize();    
    bp::class_<KSM1>("KSM1") ;



    bp::class_<pyKSM1, bp::bases<KSM1> >("pyKSM1", bp::init<int, double, double>())
	.def_readonly("N", &pyKSM1::N)
	.def_readonly("d", &pyKSM1::d)
	.def_readonly("h", &pyKSM1::h)
	.def("intg", &pyKSM1::PYintg)
	.def("intg2", &pyKSM1::PYintg2)
	;
}
