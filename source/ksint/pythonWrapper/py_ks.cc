#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "ksint.hpp"
#include "ksintM1.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


class pyKS : public KS {
  /*
   * Python interface for ksint.cc
   * Note:
   *     1) Input and out put are arrays of C form, meaning each
   *        row is a vector
   *     2) Usually input to functions should be 2-d array, so
   *        1-d arrays should be resized to 2-d before passed
   */
  
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

    /* reflection */
    bn::ndarray PYreflection(bn::ndarray aa){
	
	int m = aa.shape(0);
	int n = aa.shape(1);

	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmpraa = Reflection(tmpaa);
	
	Py_intptr_t dims[2] = {m , n};
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&tmpraa(0, 0)), 
	       sizeof(double) * m * n );

	return raa;
    }


    /* half2whole */
    bn::ndarray PYhalf2whole(bn::ndarray aa){
	
	int m = aa.shape(0);
	int n = aa.shape(1);

	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmpraa = half2whole(tmpaa);

	int n2 = tmpraa.rows();
	int m2 = tmpraa.cols();	
	Py_intptr_t dims[2] = {m2 , n2};
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&tmpraa(0, 0)), 
	       sizeof(double) * m2 * n2 );

	return raa;
    }


    /* Rotation */
    bn::ndarray PYrotation(bn::ndarray aa, const double th){
	
	int m = aa.shape(0);
	int n = aa.shape(1);

	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmpraa = Rotation(tmpaa, th);

	Py_intptr_t dims[2] = {m , n};
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&tmpraa(0, 0)), 
	       sizeof(double) * m * n );

	return raa;
    }


    /* orbitToSlice */
    bp::tuple PYorbitToSlice(bn::ndarray aa){

      
	int m = aa.shape(0);
	int n = aa.shape(1);

	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);
	std::pair<MatrixXd, VectorXd> tmp = orbitToSlice(tmpaa);

	Py_intptr_t dims[2] = {m , n};
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&tmp.first(0, 0)), 
	       sizeof(double) * m * n );
	bn::ndarray ang = 
	    bn::empty(1, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)ang.get_data(), (void*)(&tmp.second(0, 0)), 
	       sizeof(double) * m );
	      
	return bp::make_tuple(raa, ang);
    }


    /* veToSlice */
    bn::ndarray PYveToSlice(bn::ndarray ve, bn::ndarray x){
	
	int m, n;
	if(ve.get_nd() == 1){
	    m = 1;
	    n = ve.shape(0);
	} else {
	    m = ve.shape(0);
	    n = ve.shape(1);
	}
	// printf("%d %d\n", m, n);

	Map<MatrixXd> tmpve((double*)ve.get_data(), n, m);
	Map<VectorXd> tmpx((double*)x.get_data(), n);
	MatrixXd tmpvep = veToSlice(tmpve, tmpx);

	Py_intptr_t dims[2] = {m , n};
	bn::ndarray vep = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)vep.get_data(), (void*)(&tmpvep(0, 0)), 
	       sizeof(double) * m * n );
	      
	return vep;
    }





};

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
	.def("Reflection", &pyKS::PYreflection)
	.def("half2whole", &pyKS::PYhalf2whole)
	.def("Rotation", &pyKS::PYrotation)
	.def("orbitToSlice", &pyKS::PYorbitToSlice)
	.def("veToSlice", &pyKS::PYveToSlice)
	;

    bp::class_<pyKSM1, bp::bases<KSM1> >("pyKSM1", bp::init<int, double, double>())
	.def_readonly("N", &pyKS::N)
	.def_readonly("d", &pyKS::d)
	.def_readonly("h", &pyKS::h)
	.def("intg", &pyKSM1::PYintg)
	.def("intg2", &pyKSM1::PYintg2)
	;
}
