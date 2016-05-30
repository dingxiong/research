#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "CQCGL2d.hpp"
#include "myBoostPython.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class pyCQCGL2d : public CQCGL2d {
  
public:

    pyCQCGL2d(int N, double dx,
	      double b, double c, double dr, double di,
	      int threadNum) :
	CQCGL2d(N, dx, b, c, dr, di, threadNum) {}
    
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
    bn::ndarray PYintg(bn::ndarray a0, double h, int Nt, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXXcd> tmpa((dcp*)a0.get_data(), n, m);
	return copy2bnc(intg(tmpa, h, Nt, skip_rate));
    }
    
    bn::ndarray PYaintg(bn::ndarray a0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXXcd> tmpa((dcp*)a0.get_data(), n, m);
	return copy2bnc(aintg(tmpa, h, tend, skip_rate));
    }

    bn::ndarray PYintgv(bn::ndarray a0, bn::ndarray v0, double h, int Nt, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXXcd> tmpa((dcp*)a0.get_data(), n, m);
	getDims(v0, m, n);
	Map<ArrayXXcd> tmpv((dcp*)v0.get_data(), n, m);
	return copy2bnc(intgv(tmpa, tmpv, h, Nt, skip_rate));
    }    

    bn::ndarray PYaintgv(bn::ndarray a0, bn::ndarray v0, double h, double tend, int skip_rate){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXXcd> tmpa((dcp*)a0.get_data(), n, m);
	getDims(v0, m, n);
	Map<ArrayXXcd> tmpv((dcp*)v0.get_data(), n, m);
	return copy2bnc(aintgv(tmpa, tmpv, h, tend, skip_rate));
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
    
#if 0
    
    /* wrap the velocity */
    bn::ndarray PYvelocity(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(velocity(tmpa));
    }

    /* wrap the velSlice */
    bn::ndarray PYvelSlice(bn::ndarray aH){
	int m, n;
	getDims(aH, m, n);
	Map<VectorXd> tmpa((double*)aH.get_data(), m*n);
	return copy2bn(velSlice(tmpa));
    }

    /* wrap the velPhase */
    bn::ndarray PYvelPhase(bn::ndarray aH){
	int m, n;
	getDims(aH, m, n);
	Map<VectorXd> tmpa((double*)aH.get_data(), m*n);
	VectorXd tmpv = velPhase(tmpa);
	return copy2bn(tmpv);
    }
    

    /* wrap velocityReq */
    bn::ndarray PYvelocityReq(bn::ndarray a0, double th, double phi){
	int m, n;
	getDims(a0, m, n);	
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	ArrayXd tmpv = velocityReq(tmpa, th, phi);
	return copy2bn(tmpv);
    }
   
    /* orbit2slice */
    bp::tuple PYorbit2slice(const bn::ndarray &aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2slice(tmpaa);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)), copy2bn(std::get<1>(tmp)),
			      copy2bn(std::get<2>(tmp)));
    }

    /* orbit2sliceUnwrap */
    bp::tuple PYorbit2sliceWrap(const bn::ndarray &aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = orbit2sliceWrap(tmpaa);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)), copy2bn(std::get<1>(tmp)),
			      copy2bn(std::get<2>(tmp)));
    }

    /* stability matrix */
    bn::ndarray PYstab(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	return copy2bn(stab(tmpa));
    }

    /* stability matrix for relative equibrium */
    bn::ndarray PYstabReq(bn::ndarray a0, double th, double phi){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), n*m);
	return copy2bn(stabReq(tmpa, th, phi));
    }


    /* reflection */
    bn::ndarray PYreflect(const bn::ndarray &aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( reflect(tmpaa) );
    }


    /* reduceReflection */
    bn::ndarray PYreduceReflection(const bn::ndarray &aa){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( reduceReflection(tmpaa) );
    }

    /* refGradMat */
    bn::ndarray PYrefGradMat(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXd> tmpaa((double*)aa.get_data(), n * m);
	return copy2bn( refGradMat(tmpaa) );
    }

    /* reflectVe */
    bn::ndarray PYreflectVe(const bn::ndarray &veHat, const bn::ndarray &xHat){
	int m, n;
	getDims(veHat, m, n);	
	Map<MatrixXd> tmpveHat((double*)veHat.get_data(), n, m);
	int m2, n2;
	getDims(xHat, m2, n2);
	Map<ArrayXd> tmpxHat((double*)xHat.get_data(), n2*m2);
	return copy2bn( reflectVe(tmpveHat, tmpxHat) );
    }

    /* reflectVeAll */
    bn::ndarray PYreflectVeAll(const bn::ndarray &veHat, const bn::ndarray &aaHat,
			       const int trunc){
	int m, n;
	getDims(veHat, m, n);	
	Map<MatrixXd> tmpveHat((double*)veHat.get_data(), n, m);
	int m2, n2;
	getDims(aaHat, m2, n2);
	Map<ArrayXd> tmpaaHat((double*)aaHat.get_data(), n2, m2);
	return copy2bn( reflectVeAll(tmpveHat, tmpaaHat, trunc) );
    }
    
    
    /* ve2slice */
    bn::ndarray PYve2slice(bn::ndarray ve, bn::ndarray x){
	int m, n;
	getDims(ve, m, n);
	int m2, n2;
	getDims(x, m2, n2);
	Map<ArrayXXd> tmpve((double*)ve.get_data(), n, m);
	Map<ArrayXd> tmpx((double*)x.get_data(), n2*m2);
	return copy2bn( ve2slice(tmpve, tmpx) );
    }
    
    /* reduceAllSymmetries */
    bp::tuple PYreduceAllSymmetries(const bn::ndarray &aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = 
	    reduceAllSymmetries(tmpaa);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)),
			      copy2bn(std::get<1>(tmp)),
			      copy2bn(std::get<2>(tmp)));
	
    }

    /* reduceVe */
    bn::ndarray PYreduceVe(const bn::ndarray &ve, const bn::ndarray &x){
	int m, n;
	getDims(ve, m, n);
	Map<ArrayXXd> tmpve((double*)ve.get_data(), n, m);
	int m2, n2;
	getDims(x, m2, n2);
	Map<ArrayXd> tmpx((double*)x.get_data(), n2*m2);
	return copy2bn(reduceVe(tmpve, tmpx));
    }
    
    /* transRotate */
    bn::ndarray PYtransRotate(bn::ndarray aa, double th){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( transRotate(tmpaa, th) );
    }

    /* transTangent */
    bn::ndarray PYtransTangent(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( transTangent(tmpaa) );
    }

    /* phaseRotate */
    bn::ndarray PYphaseRotate(bn::ndarray aa, double phi){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( phaseRotate(tmpaa, phi) );
    }
    
    /* phaseTangent */
    bn::ndarray PYphaseTangent(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( phaseTangent(tmpaa) );
    }

    /* Rotate */
    bn::ndarray PYRotate(bn::ndarray aa, double th, double phi){
	int m, n;
	getDims(aa, m, n);	
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( Rotate(tmpaa, th, phi) );
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
	return copy2bn( rotateOrbit(tmpaa, tmpth, tmpphi) );
    }
    
#endif
    
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



BOOST_PYTHON_MODULE(py_CQCGL2d) {
    bn::initialize();
    
    // must provide the constructor
    bp::class_<CQCGL2d>("CQCGL2d", bp::init<
			int, double, 
			double, double, double, double, 
			int>())
	;
    
    
    bp::class_<pyCQCGL2d, bp::bases<CQCGL2d> >("pyCQCGL2d", bp::init<
					       int, double, 
					       double, double, double, double,
					       int>())
	.def_readonly("N", &pyCQCGL2d::N)
	.def_readonly("M", &pyCQCGL2d::M)
	.def_readonly("dx", &pyCQCGL2d::dx)
	.def_readonly("dy", &pyCQCGL2d::dy)
	.def_readonly("Mu", &pyCQCGL2d::Mu)
	.def_readonly("Br", &pyCQCGL2d::Br)
	.def_readonly("Bi", &pyCQCGL2d::Bi)
	.def_readonly("Dr", &pyCQCGL2d::Dr)
	.def_readonly("Di", &pyCQCGL2d::Di)
	.def_readonly("Gr", &pyCQCGL2d::Gr)
	.def_readonly("Gi", &pyCQCGL2d::Gi)
	.def_readonly("Ne", &pyCQCGL2d::Ne)
	.def_readonly("Me", &pyCQCGL2d::Me)
	.def_readonly("b", &pyCQCGL2d::b)
	.def_readonly("c", &pyCQCGL2d::c)
	.def_readonly("dr", &pyCQCGL2d::dr)
	.def_readonly("di", &pyCQCGL2d::di)
	.def_readonly("Omega", &pyCQCGL2d::Omega)
	.def_readwrite("rtol", &pyCQCGL2d::rtol)
	.def_readwrite("nu", &pyCQCGL2d::nu)
	.def_readwrite("mumax", &pyCQCGL2d::mumax)
	.def_readwrite("mumin", &pyCQCGL2d::mumin)
	.def_readwrite("mue", &pyCQCGL2d::mue)
	.def_readwrite("muc", &pyCQCGL2d::muc)
	.def_readwrite("NCalCoe", &pyCQCGL2d::NCalCoe)
	.def_readwrite("NReject", &pyCQCGL2d::NReject)
	.def_readwrite("NCallF", &pyCQCGL2d::NCallF)
	.def_readwrite("NSteps", &pyCQCGL2d::NSteps)
	.def_readwrite("Method", &pyCQCGL2d::Method)
	.def("Ts", &pyCQCGL2d::PYTs)
	.def("hs", &pyCQCGL2d::PYhs)
	.def("lte", &pyCQCGL2d::PYlte)
	.def("Kx", &pyCQCGL2d::PYKx)
	.def("Ky", &pyCQCGL2d::PYKy)
	.def("L", &pyCQCGL2d::PYL)
	
	.def("changeOmega", &pyCQCGL2d::changeOmega)
	.def("intg", &pyCQCGL2d::PYintg)
	.def("aintg", &pyCQCGL2d::PYaintg)
	.def("intgv", &pyCQCGL2d::PYintgv)
	.def("aintgv", &pyCQCGL2d::PYaintgv)
	.def("Fourier2Config", &pyCQCGL2d::PYFourier2Config)
	.def("Config2Fourier", &pyCQCGL2d::PYConfig2Fourier)
#if 0
	.def("velocity", &pyCQCGL2d::PYvelocity)
	.def("velSlice", &pyCQCGL2d::PYvelSlice)
	.def("velPhase", &pyCQCGL2d::PYvelPhase)
	.def("velocityReq", &pyCQCGL2d::PYvelocityReq)
	.def("orbit2sliceWrap", &pyCQCGL2d::PYorbit2sliceWrap)
	.def("orbit2slice", &pyCQCGL2d::PYorbit2slice)
	.def("stab", &pyCQCGL2d::PYstab)
	.def("stabReq", &pyCQCGL2d::PYstabReq)
	.def("reflect", &pyCQCGL2d::PYreflect)
	.def("reduceReflection", &pyCQCGL2d::PYreduceReflection)
	.def("refGradMat", &pyCQCGL2d::PYrefGradMat)
	.def("reflectVe", &pyCQCGL2d::PYreflectVe)
	.def("reflectVeAll", &pyCQCGL2d::PYreflectVeAll)
	.def("ve2slice", &pyCQCGL2d::PYve2slice)
	.def("reduceAllSymmetries", &pyCQCGL2d::PYreduceAllSymmetries)
	.def("reduceVe", &pyCQCGL2d::PYreduceVe)
	.def("transRotate", &pyCQCGL2d::PYtransRotate) 
	.def("transTangent", &pyCQCGL2d::PYtransTangent)
	.def("phaseRotate", &pyCQCGL2d::PYphaseRotate)
	.def("phaseTangent", &pyCQCGL2d::PYphaseTangent)
	.def("Rotate", &pyCQCGL2d::PYRotate)
	.def("rotateOrbit", &pyCQCGL2d::PYrotateOrbit)
#endif
	;

}
