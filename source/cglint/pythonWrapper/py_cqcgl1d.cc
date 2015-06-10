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

    /* get the dimension of an array */
    void getDims(bn::ndarray x, int &m, int &n){
	if(x.get_nd() == 1){
	    m = 1;
	    n = x.shape(0);
	} else {
	    m = x.shape(0);
	    n = x.shape(1);
	}
    }
    
    /* wrap the velocity */
    bn::ndarray PYvelocity(bn::ndarray a0){

	int m, n;
	getDims(a0, m, n);
	
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
    
    /* wrap velocityReq */
    bn::ndarray PYvelocityReq(bn::ndarray a0, double th, double phi){
	int m, n;
	getDims(a0, m, n);
	
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	VectorXd tmpv = velocityReq(tmpa, th, phi);
	
	int n3 = tmpv.rows();
	Py_intptr_t dims[1] = {n3};
	bn::ndarray v = 
	    bn::empty(1, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)v.get_data(), (void*)(&tmpv(0)), 
	       sizeof(double) * n3 );
	
	return v;

    }


    /* pad */
    bn::ndarray PYpad(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);

	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmppaa = pad(tmpaa);

	int m2 = tmppaa.cols();
	int n2 = tmppaa.rows();
	Py_intptr_t dims[2] = {m2 , n2};
	bn::ndarray paa = bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)paa.get_data(), (void*)(&tmppaa(0, 0)), 
	       sizeof(double) * m2 * n2 );

	return paa;
    }

    /* unpad */
    bn::ndarray PYunpad(bn::ndarray paa){
	int m, n;
	getDims(paa, m, n);

	Map<ArrayXXd> tmppaa((double*)paa.get_data(), n, m);
	ArrayXXd tmpaa = unpad(tmppaa);

	int m2 = tmpaa.cols();
	int n2 = tmpaa.rows();
	Py_intptr_t dims[2] = {m2 , n2};
	bn::ndarray aa = bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)aa.get_data(), (void*)(&tmpaa(0, 0)), 
	       sizeof(double) * m2 * n2 );

	return aa;
    }

    
    /* wrap the integrator */
    bn::ndarray PYintg(bn::ndarray a0, size_t nstp, size_t np){

	int m, n;
	getDims(a0, m, n);

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
	getDims(a0, m, n);

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

    /* wrap Fourier2Config */
    bn::ndarray PYFourier2Config(bn::ndarray aa){

	int m, n;
	getDims(aa, m, n);
	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmpAA = Fourier2Config(tmpaa);
	
	int m2 = tmpAA.cols();
	int n2 = tmpAA.rows();
	Py_intptr_t dims[2] = {m2 , n2};
	bn::ndarray AA = bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)AA.get_data(), (void*)(&tmpAA(0, 0)), 
	       sizeof(double) * m2 * n2 );

	return AA;
    }

    /* wrap Config2Fourier */
    bn::ndarray PYConfig2Fourier(bn::ndarray AA){

	int m, n;
	if(AA.get_nd() == 1){
	    m = 1;
	    n = AA.shape(0);
	} else {
	    m = AA.shape(0);
	    n = AA.shape(1);
	}
	
	Map<ArrayXXd> tmpAA((double*)AA.get_data(), n, m);
	ArrayXXd tmpaa = Config2Fourier(tmpAA);
	
	int m2 = tmpaa.cols();
	int n2 = tmpaa.rows();
	Py_intptr_t dims[2] = {m2 , n2};
	bn::ndarray aa = bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)aa.get_data(), (void*)(&tmpaa(0, 0)), 
	       sizeof(double) * m2 * n2 );

	return aa;
    }
    
    /* wrap Fourier2ConfigMag */
    bn::ndarray PYFourier2ConfigMag(bn::ndarray aa){

	int m, n;
	if(aa.get_nd() == 1){
	    m = 1;
	    n = aa.shape(0);
	} else {
	    m = aa.shape(0);
	    n = aa.shape(1);
	}
	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmpAA = Fourier2ConfigMag(tmpaa);
	
	int m2 = tmpAA.cols();
	int n2 = tmpAA.rows();
	Py_intptr_t dims[2] = {m2 , n2};
	bn::ndarray AA = bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)AA.get_data(), (void*)(&tmpAA(0, 0)), 
	       sizeof(double) * m2 * n2 );

	return AA;
    }
    
    /* orbit2slice */
    bp::tuple PYorbit2slice(bn::ndarray aa){

	int m, n;
	getDims(aa, m, n);

	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2slice(tmpaa);

	int m2 = std::get<0>(tmp).cols();
	int n2 = std::get<0>(tmp).rows();
	Py_intptr_t dims[2] = {m2 , n2};
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&(std::get<0>(tmp)(0, 0))), 
	       sizeof(double) * m2 * n2 );

	int n3 = std::get<1>(tmp).rows();
	Py_intptr_t dims2[1] = { n3 };
	bn::ndarray th = 
	    bn::empty(1, dims2, bn::dtype::get_builtin<double>());
	memcpy((void*)th.get_data(), (void*)(&(std::get<1>(tmp)(0))), 
	       sizeof(double) * n3 );

	int n4 = std::get<2>(tmp).rows();
	Py_intptr_t dims3[1] = { n4 };
	bn::ndarray phi = 
	    bn::empty(1, dims3, bn::dtype::get_builtin<double>());
	memcpy((void*)phi.get_data(), (void*)(&(std::get<2>(tmp)(0))), 
	       sizeof(double) * n4 );
	      
	return bp::make_tuple(raa, th, phi);
    }

    /* orbit2sliceUnwrap */
    bp::tuple PYorbit2sliceWrap(bn::ndarray aa){

	int m, n;
	getDims(aa, m, n);

	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2sliceWrap(tmpaa);

	int m2 = std::get<0>(tmp).cols();
	int n2 = std::get<0>(tmp).rows();
	Py_intptr_t dims[2] = {m2 , n2};
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&(std::get<0>(tmp)(0, 0))), 
	       sizeof(double) * m2 * n2 );

	int n3 = std::get<1>(tmp).rows();
	Py_intptr_t dims2[1] = { n3 };
	bn::ndarray th = 
	    bn::empty(1, dims2, bn::dtype::get_builtin<double>());
	memcpy((void*)th.get_data(), (void*)(&(std::get<1>(tmp)(0))), 
	       sizeof(double) * n3 );

	int n4 = std::get<2>(tmp).rows();
	Py_intptr_t dims3[1] = { n4 };
	bn::ndarray phi = 
	    bn::empty(1, dims3, bn::dtype::get_builtin<double>());
	memcpy((void*)phi.get_data(), (void*)(&(std::get<2>(tmp)(0))), 
	       sizeof(double) * n4 );
	      
	return bp::make_tuple(raa, th, phi);
    }

    /* stability matrix */
    bn::ndarray PYstab(bn::ndarray a0){

      	int m, n;
	if(a0.get_nd() == 1){
	    m = 1;
	    n = a0.shape(0);
	} else {
	    m = a0.shape(0);
	    n = a0.shape(1);
	}

	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	MatrixXd tmpZ = stab(tmpa);

	int m2 = tmpZ.cols();
	int n2 = tmpZ.rows();
	Py_intptr_t dims[2] = {m2 , n2};
	bn::ndarray Z = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)Z.get_data(), (void*)&(tmpZ(0, 0)), 
	       sizeof(double) * m2 * n2 );
	      
	return Z;
    }

    /* stability matrix for relative equibrium */
    bn::ndarray PYstabReq(bn::ndarray a0, double th, double phi){

	int m, n;
	getDims(a0, m, n);

	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	MatrixXd tmpZ = stabReq(tmpa, th, phi);

	int m2 = tmpZ.cols();
	int n2 = tmpZ.rows();
	Py_intptr_t dims[2] = {m2 , n2};
	bn::ndarray Z = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)Z.get_data(), (void*)&(tmpZ(0, 0)), 
	       sizeof(double) * m2 * n2 );
	      
	return Z;
    }


    /* reflection */
    bn::ndarray PYreflect(bn::ndarray aa){

	int m, n;
	getDims(aa, m, n);
	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmpraa = reflect(tmpaa);

	int m2 = tmpraa.cols();
	int n2 = tmpraa.rows();
	Py_intptr_t dims[2] = {m2 , n2};
	
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&tmpraa(0, 0)), 
	       sizeof(double) * m2 * n2 );

	return raa;
    }


    /* reduceReflection */
    bn::ndarray PYreduceReflection(bn::ndarray aa){

	int m, n;
	getDims(aa, m, n);
	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmpraa = reduceReflection(tmpaa);

	int m2 = tmpraa.cols();
	int n2 = tmpraa.rows();
	Py_intptr_t dims[2] = {m2 , n2};
	
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&tmpraa(0, 0)), 
	       sizeof(double) * m2 * n2 );

	return raa;
    }

    
    
    /* ve2slice */
    bn::ndarray PYve2slice(bn::ndarray ve, bn::ndarray x){

	int m, n;
	getDims(ve, m, n);

	int m2, n2;
	getDims(x, m2, n2);

	Map<ArrayXXd> tmpve((double*)ve.get_data(), n, m);
	Map<ArrayXd> tmpx((double*)x.get_data(), n2*m2);
	MatrixXd tmpvep = ve2slice(tmpve, tmpx);

	int m3 = tmpvep.cols();
	int n3 = tmpvep.rows();
	Py_intptr_t dims[2] = {m3 , n3};
	bn::ndarray vep = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)vep.get_data(), (void*)(&tmpvep(0, 0)), 
	       sizeof(double) * m3 * n3 );
	      
	return vep;
    }

    /* findReq */
    bp::tuple PYfindReq(bn::ndarray a0, double wth0, double wphi0,
			int MaxN, double tol, bool doesUseMyCG,
			bool doesPrint){
	int m, n;
	getDims(a0, m, n);

	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	std::tuple<ArrayXd, double, double, double> tmp = 
	    findReq(tmpa0, wth0, wphi0, MaxN, tol, doesUseMyCG, doesPrint);

	int n2 = std::get<0>(tmp).rows();
	Py_intptr_t dims[1] = { n2 };
	bn::ndarray a = bn::empty(1, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)a.get_data(), (void*)(&(std::get<0>(tmp)(0))), 
	       sizeof(double) * n2 );

	return bp::make_tuple(a, std::get<1>(tmp), std::get<2>(tmp),
			      std::get<3>(tmp));
    }
    
    /* transRotate */
    bn::ndarray PYtransRotate(bn::ndarray aa, double th){
	int m, n;
	getDims(aa, m, n);
	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmpraa = transRotate(tmpaa, th);

	int m3 = tmpraa.cols();
	int n3 = tmpraa.rows();
	Py_intptr_t dims[2] = {m3 , n3};
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&tmpraa(0, 0)), 
	       sizeof(double) * m3 * n3 );

	return raa;
    }

    /* phaseRotate */
    bn::ndarray PYphaseRotate(bn::ndarray aa, double phi){
	int m, n;
	getDims(aa, m, n);
	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmpraa = transRotate(tmpaa, phi);

	int m3 = tmpraa.cols();
	int n3 = tmpraa.rows();
	Py_intptr_t dims[2] = {m3 , n3};
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&tmpraa(0, 0)), 
	       sizeof(double) * m3 * n3 );

	return raa;
    }

    /* Rotate */
    bn::ndarray PYRotate(bn::ndarray aa, double th, double phi){
	int m, n;
	getDims(aa, m, n);
	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmpraa = Rotate(tmpaa, th, phi);

	int m3 = tmpraa.cols();
	int n3 = tmpraa.rows();
	Py_intptr_t dims[2] = {m3 , n3};
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&tmpraa(0, 0)), 
	       sizeof(double) * m3 * n3 );

	return raa;
    }

    /* rotateOrbit */
    bn::ndarray PYrotateOrbit(bn::ndarray aa, bn::ndarray th, bn::ndarray phi){
	int m, n;
	getDims(aa, m, n);
	int m2, n2;
	getDims(th, m2, n2);
	int m4, n4;
	getDims(phi, m4, n4);
	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	Map<ArrayXd> tmpth((double*)th.get_data(), n2 * m2);
	Map<ArrayXd> tmpphi((double*)phi.get_data(), n4 * m4);
	ArrayXXd tmpraa = rotateOrbit(tmpaa, tmpth, tmpphi);

	int m3 = tmpraa.cols();
	int n3 = tmpraa.rows();
	Py_intptr_t dims[2] = {m3 , n3};
	bn::ndarray raa = 
	    bn::empty(2, dims, bn::dtype::get_builtin<double>());
	memcpy((void*)raa.get_data(), (void*)(&tmpraa(0, 0)), 
	       sizeof(double) * m3 * n3 );

	return raa;
    }
    
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
	.def_readonly("Kindex", &pyCqcgl1d::Kindex)
	.def_readonly("Ndim", &pyCqcgl1d::Ndim)
	.def_readonly("KindexUnpad", &pyCqcgl1d::KindexUnpad)
	.def("velocity", &pyCqcgl1d::PYvelocity)
	.def("velocityReq", &pyCqcgl1d::PYvelocityReq)
	.def("pad", &pyCqcgl1d::PYpad)
	.def("unpad", &pyCqcgl1d::PYunpad)
	.def("intg", &pyCqcgl1d::PYintg)
	.def("intgj", &pyCqcgl1d::PYintgj)
	.def("Fourier2Config", &pyCqcgl1d::PYFourier2Config)
	.def("Config2Fourier", &pyCqcgl1d::PYConfig2Fourier)
	.def("Fourier2ConfigMag", &pyCqcgl1d::PYFourier2Config)
	.def("orbit2sliceWrap", &pyCqcgl1d::PYorbit2sliceWrap)
	.def("orbit2slice", &pyCqcgl1d::PYorbit2slice)
	.def("stab", &pyCqcgl1d::PYstab)
	.def("stabReq", &pyCqcgl1d::PYstabReq)
	.def("reflect", &pyCqcgl1d::PYreflect)
	.def("reduceReflection", &pyCqcgl1d::PYreduceReflection)
	.def("ve2slice", &pyCqcgl1d::PYve2slice)
	.def("findReq", &pyCqcgl1d::PYfindReq)
	.def("transRotate", &pyCqcgl1d::PYtransRotate)
	.def("phaseRotate", &pyCqcgl1d::PYphaseRotate)
	.def("Rotate", &pyCqcgl1d::PYRotate)
	.def("rotateOrbit", &pyCqcgl1d::PYrotateOrbit)
	;
}
