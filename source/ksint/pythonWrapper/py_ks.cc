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

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/* get the dimension of an array */
inline void getDims(bn::ndarray x, int &m, int &n){
    if(x.get_nd() == 1){
	m = 1;
	n = x.shape(0);
    } else {
	m = x.shape(0);
	n = x.shape(1);
    }
}

/*
 * @brief used to copy content in Eigen array to boost.numpy array.
 *
 *  Only work for double array/matrix
 */
inline bn::ndarray copy2bn(const Ref<const ArrayXXd> &x){
    int m = x.cols();
    int n = x.rows();

    Py_intptr_t dims[2];
    int ndim;
    if(m == 1){
	ndim = 1;
	dims[0] = n;
    }
    else {
	ndim = 2;
	dims[0] = m;
	dims[1] = n;
    }
    bn::ndarray px = bn::empty(ndim, dims, bn::dtype::get_builtin<double>());
    memcpy((void*)px.get_data(), (void*)x.data(), sizeof(double) * m * n);
	    
    return px;
}

/**
 * @brief std::vector to bp::list
 */
template <class T>
bp::list toList(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    bp::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
	list.append(*iter);
    }
    return list;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


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
    
    bn::ndarray PYvelg(bn::ndarray a0, double theta){
	int m, n;
	getDims(a0, m, n);
	Map<VectorXd> tmpa((double*)a0.get_data(), n*m);
	
	return copy2bn(velg(tmpa, theta));
    }
    
    /* stab */
    bn::ndarray PYstab(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	
	return copy2bn(stab(tmpa));
    }

    bn::ndarray PYstabReq(bn::ndarray a0, double theta){
	int m, n;
	getDims(a0, m, n);
	Map<VectorXd> tmpa((double*)a0.get_data(), n*m);
	
	return copy2bn(stabReq(tmpa, theta));
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

	int m, n;
	getDims(aa, m, n);
	
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

    /* orbitAndFvWholeSlice  */
    bp::tuple PYorbitAndFvWholeSlice(bn::ndarray a, bn::ndarray ve,
				     const int nstp, const string ppType,
				     const int pos){
	int m, n;
	getDims(a, m, n);
	Map<ArrayXd> tmpa((double*)a.get_data(), n*m);
	getDims(ve, m, n);
	Map<ArrayXXd> tmpve((double*)ve.get_data(), n, m);
	auto tmpav = orbitAndFvWholeSlice(tmpa, tmpve, nstp, ppType, pos);

	return bp::make_tuple(copy2bn(tmpav.first), copy2bn(tmpav.second));
    }

    /* reduceReflection */
    bn::ndarray PYreduceReflection(bn::ndarray aa){
	
	int m, n;
	if(aa.get_nd() == 1){
	    m = 1;
	    n = aa.shape(0);
	} else {
	    m = aa.shape(0);
	    n = aa.shape(1);
	}

	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmpraa = reduceReflection(tmpaa);
	
	int m3 = tmpraa.cols();
	int n3 = tmpraa.rows();
	Py_intptr_t dims[2] = {m3 , n3};
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&tmpraa(0, 0)), 
	       sizeof(double) * m3 * n3 );
	      
	return raa;
    }

    /* reflectVe */
    bn::ndarray PYreflectVe(bn::ndarray ve, bn::ndarray x){
	
	int m, n;
	getDims(ve, m, n);

	int m2, n2;
	getDims(x, m2, n2);

	Map<MatrixXd> tmpve((double*)ve.get_data(), n, m);
	Map<ArrayXd> tmpx((double*)x.get_data(), m2*n2);
	MatrixXd tmprve = reflectVe(tmpve, tmpx);
	
	int m3 = tmprve.cols();
	int n3 = tmprve.rows();
	Py_intptr_t dims[2] = {m3 , n3};
	bn::ndarray rve = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)rve.get_data(), (void*)(&tmprve(0, 0)), 
	       sizeof(double) * m3 * n3 );
	      
	return rve;
    }

    /* reflectVeAll */
    bn::ndarray PYreflectVeAll(bn::ndarray ve, bn::ndarray x, int trunc){
	
	int m, n;
	getDims(ve, m, n);

	int m2, n2;
	getDims(x, m2, n2);

	Map<MatrixXd> tmpve((double*)ve.get_data(), n, m);
	Map<MatrixXd> tmpx((double*)x.get_data(), n2, m2);
	MatrixXd tmprve = reflectVeAll(tmpve, tmpx, trunc);
	
	int m3 = tmprve.cols();
	int n3 = tmprve.rows();
	Py_intptr_t dims[2] = {m3 , n3};
	bn::ndarray rve = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)rve.get_data(), (void*)(&tmprve(0, 0)), 
	       sizeof(double) * m3 * n3 );
	      
	return rve;
    }
    
    bn::ndarray PYcalMag(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);
	
	return copy2bn(calMag(tmpaa));
    }

    bp::tuple PYtoPole(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);

	auto tmp = toPole(tmpaa);
	return bp::make_tuple(copy2bn(tmp.first), 
			      copy2bn(tmp.second)
			      );
    }

    bp::tuple PYcalAB(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);
	VectorXcd AB = calAB(tmpaa);

	return bp::make_tuple(copy2bn(AB.real()),
			      copy2bn(AB.imag())
			      );
    }

    bp::tuple PYredSO2(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<MatrixXd> tmpaa((double*)aa.get_data(), n, m);

	auto tmp = redSO2(tmpaa);
	return bp::make_tuple(copy2bn(tmp.first), 
			      copy2bn(tmp.second)
			      );
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
	.def("velg", &pyKS::PYvelg)
	.def("stab", &pyKS::PYstab)
	.def("stabReq", &pyKS::PYstabReq)
	.def("intg", &pyKS::PYintg)
	.def("intgj", &pyKS::PYintgj)
	.def("Reflection", &pyKS::PYreflection)
	.def("half2whole", &pyKS::PYhalf2whole)
	.def("Rotation", &pyKS::PYrotation)
	.def("orbitToSlice", &pyKS::PYorbitToSlice)
	.def("veToSlice", &pyKS::PYveToSlice)
	.def("veToSliceAll", &pyKS::PYveToSliceAll)
	.def("orbitAndFvWholeSlice", &pyKS::PYorbitAndFvWholeSlice)
	.def("reduceReflection", &pyKS::PYreduceReflection)
	.def("reflectVe", &pyKS::PYreflectVe)
	.def("reflectVeAll", &pyKS::PYreflectVeAll)
	.def("calMag", &pyKS::PYcalMag)
	.def("toPole", &pyKS::PYtoPole)
	.def("calAB", &pyKS::PYcalAB)
	.def("redSO2", &pyKS::PYredSO2)
	;

    bp::class_<pyKSM1, bp::bases<KSM1> >("pyKSM1", bp::init<int, double, double>())
	.def_readonly("N", &pyKS::N)
	.def_readonly("d", &pyKS::d)
	.def_readonly("h", &pyKS::h)
	.def("intg", &pyKSM1::PYintg)
	.def("intg2", &pyKSM1::PYintg2)
	;
}
