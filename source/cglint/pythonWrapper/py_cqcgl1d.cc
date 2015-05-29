#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "cqcgl1d.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


class pyCqcgl1d : public Cqcgl1d {
  /*
   * Python interface for Cqcgl1d.cc
   * Note:
   *     1) Input and out put are arrays of C form, meaning each
   *        row is a vector
   *     2) Usually input to functions should be 2-d array, so
   *        1-d arrays should be resized to 2-d before passed
   */
  
public:
    pyCqcgl1d(int N, double d, double h, double Mu,
	      double Br, double Bi, double Dr,
	      double Di, double Gr, double Gi) :
	Cqcgl1d(N, d, h, Mu, Br, Bi, Dr, Di, Gr, Gi) {}
    
    /* wrap the velocity */
    bn::ndarray PYvelocity(bn::ndarray a0){

	int m, n;
	if(a0.get_nd() == 1){
	    m = 1;
	    n = a0.shape(0);
	} else {
	    m = a0.shape(0);
	    n = a0.shape(1);
	}
	
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	VectorXd tmpv = velocity(tmpa);
	
	int n3 = tmpv.rows();
	Py_intptr_t dims[1] = {n3};
	bn::ndarray v = 
	    bn::empty(1, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)v.get_data(), (void*)(&tmpv(0)), 
	       sizeof(double) * n3 );
	
	return v;
    }


    /* wrap the integrator */
    bn::ndarray PYintg(bn::ndarray a0, size_t nstp, size_t np){

	int m, n;
	if(a0.get_nd() == 1){
	    m = 1;
	    n = a0.shape(0);
	} else {
	    m = a0.shape(0);
	    n = a0.shape(1);
	}

	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	ArrayXXd tmpaa = intg(tmpa, nstp, np);

	int m3 = tmpaa.cols();
	int n3 = tmpaa.rows();
	Py_intptr_t dims[2] = {m3 , n3};
	bn::ndarray aa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)aa.get_data(), (void*)(&tmpaa(0, 0)), 
	       sizeof(double) * m3 * n3 );

	return aa;
    }
    

    /* wrap the integrator with Jacobian */
    bp::tuple PYintgj(bn::ndarray a0, size_t nstp, size_t np, size_t nqr){

	int m, n;
	if(a0.get_nd() == 1){
	    m = 1;
	    n = a0.shape(0);
	} else {
	    m = a0.shape(0);
	    n = a0.shape(1);
	}

	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	std::pair<ArrayXXd, ArrayXXd> tmp = intgj(tmpa, nstp, np, nqr);

	int m2 = tmp.first.cols();
	int n2 = tmp.first.rows();
	Py_intptr_t dims[2] = {m2 , n2};
	bn::ndarray aa = bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)aa.get_data(), (void*)(&tmp.first(0, 0)), 
	       sizeof(double) * m2 * n2 );

	int m3 = tmp.second.cols();
	int n3 = tmp.second.rows();
	Py_intptr_t dims2[2] = {m3 , n3};
	bn::ndarray daa = bn::empty(2, dims2, bn::dtype::get_builtin<double>());
	memcpy((void*)daa.get_data(), (void*)(&tmp.second(0, 0)), 
	       sizeof(double) * m3 * n3 );

	return bp::make_tuple(aa, daa);
    }

#if 0
    /* reflection */
    bn::ndarray PYreflection(bn::ndarray aa){

	int m, n;
	if(aa.get_nd() == 1){
	    m = 1;
	    n = aa.shape(0);
	} else {
	    m = aa.shape(0);
	    n = aa.shape(1);
	}
	
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

	int m, n;
	if(aa.get_nd() == 1){
	    m = 1;
	    n = aa.shape(0);
	} else {
	    m = aa.shape(0);
	    n = aa.shape(1);
	}

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

      	int m, n;
	if(aa.get_nd() == 1){
	    m = 1;
	    n = aa.shape(0);
	} else {
	    m = aa.shape(0);
	    n = aa.shape(1);
	}

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

    /* veToSliceAll */
    bn::ndarray PYveToSliceAll(bn::ndarray eigVecs, bn::ndarray aa, const int trunc){
	
	int m, n;
	if(eigVecs.get_nd() == 1){
	    m = 1;
	    n = eigVecs.shape(0);
	} else {
	    m = eigVecs.shape(0);
	    n = eigVecs.shape(1);
	}

	int m2, n2;
	if(aa.get_nd() == 1){
	    m2 = 1;
	    n2 = aa.shape(0);
	} else {
	    m2 = aa.shape(0);
	    n2 = aa.shape(1);
	}
	// printf("%d %d %d %d\n", m, n, m2, n2);
	Map<MatrixXd> tmpeigVecs((double*)eigVecs.get_data(), n, m);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n2, m2);
	MatrixXd tmpvep = veToSliceAll(tmpeigVecs , tmpaa, trunc);
	
	int m3 = tmpvep.cols();
	int n3 = tmpvep.rows();
	Py_intptr_t dims[2] = {m3 , n3};
	bn::ndarray vep = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)vep.get_data(), (void*)(&tmpvep(0, 0)), 
	       sizeof(double) * m3 * n3 );
	      
	return vep;
    }
#endif

};


BOOST_PYTHON_MODULE(py_cqcgl1d) {
    bn::initialize();    
    bp::class_<Cqcgl1d>("Cqcgl1d") ;
    
    bp::class_<pyCqcgl1d, bp::bases<Cqcgl1d> >("pyCqcgl1d", bp::init<int, double, double,
					       double, double, double, double, double,
					       double, double >())
	.def_readonly("N", &pyCqcgl1d::N)
	.def_readonly("d", &pyCqcgl1d::d)
	.def_readonly("h", &pyCqcgl1d::h)
	.def_readonly("Mu", &pyCqcgl1d::Mu)
	.def_readonly("Br", &pyCqcgl1d::Br)
	.def_readonly("Bi", &pyCqcgl1d::Bi)
	.def_readonly("Dr", &pyCqcgl1d::Dr)
	.def_readonly("Di", &pyCqcgl1d::Di)
	.def_readonly("Gr", &pyCqcgl1d::Gr)
	.def_readonly("Gi", &pyCqcgl1d::Gi)
	.def("velocity", &pyCqcgl1d::PYvelocity)
	.def("intg", &pyCqcgl1d::PYintg)
	.def("intgj", &pyCqcgl1d::PYintgj)
	;

}
