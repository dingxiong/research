#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "CQCGL.hpp"
#include "myBoostPython.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class pyCQCGL : public CQCGL {
  
public:

    pyCQCGL(int N, double d,
	    double b, double c, double dr, double di,
	    int dimTan, int threadNum) :
	CQCGL(N, d, b, c, dr, di, dimTan, threadNum) {}

    /* K */
    bn::ndarray PYK(){
	return copy2bn(K);
    }

    /* L */
    bn::ndarray PYL(){
	return copy2bn<ArrayXXcd, std::complex<double>>(L);
    }

    /* wrap the velocity */
    bn::ndarray PYvelocity(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	ArrayXd tmpv = velocity(tmpa);
	return copy2bn(tmpv);
    }

    /* wrap the velSlice */
    bn::ndarray PYvelSlice(bn::ndarray aH){
	int m, n;
	getDims(aH, m, n);
	Map<VectorXd> tmpa((double*)aH.get_data(), m*n);
	VectorXd tmpv = velSlice(tmpa);
	return copy2bn(tmpv);
    }

    /* wrap the velPhase */
    bn::ndarray PYvelPhase(bn::ndarray aH){
	int m, n;
	getDims(aH, m, n);
	Map<VectorXd> tmpa((double*)aH.get_data(), m*n);
	VectorXd tmpv = velPhase(tmpa);
	return copy2bn(tmpv);
    }
    
    bn::ndarray PYrk4(bn::ndarray a0, const double dt, const int nstp, const int nq){
	int m, n;
	getDims(a0, m, n);
	Map<VectorXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(rk4(tmpa, dt, nstp, nq));
    }

    bp::tuple PYrk4j(bn::ndarray a0, double dt, int nstp, int nq, int nqr) {
	int m, n;
	getDims(a0, m, n);
	Map<VectorXd> tmpa((double*)a0.get_data(), m*n);
	auto tmp = rk4j(tmpa, dt, nstp, nq, nqr);
	return bp::make_tuple(copy2bn(tmp.first), copy2bn(tmp.second));
    }
    
    /* wrap the Lyap */
    bp::tuple PYLyap(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXcd lya = Lyap(tmpaa);
	return bp::make_tuple(copy2bn( lya.real() ),
			      copy2bn( lya.imag() )
			      );
    }
    
    /* wrap the LyapVel */
    bn::ndarray PYLyapVel(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXd lyavel = LyapVel(tmpaa);
	return copy2bn(lyavel);
    }
    
    /* wrap velocityReq */
    bn::ndarray PYvelocityReq(bn::ndarray a0, double th, double phi){
	int m, n;
	getDims(a0, m, n);	
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	ArrayXd tmpv = velocityReq(tmpa, th, phi);
	return copy2bn(tmpv);
    }


    /* pad */
    bn::ndarray PYpad(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	ArrayXXd tmppaa = pad(tmpaa);
	return copy2bn(pad(tmpaa));
    }

    /* generalPadding */
    bn::ndarray PYgeneralPadding(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( generalPadding(tmpaa) );
    }

    /* unpad */
    bn::ndarray PYunpad(bn::ndarray paa){
	int m, n;
	getDims(paa, m, n);
	Map<ArrayXXd> tmppaa((double*)paa.get_data(), n, m);
	return copy2bn(unpad(tmppaa));
    }

    
    /* wrap the integrator */
    bn::ndarray PYintg(bn::ndarray a0, int nstp, int np){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	return copy2bn(intg(tmpa, nstp, np));
    }
    

    /* wrap the integrator with Jacobian */
    bp::tuple PYintgj(bn::ndarray a0, size_t nstp, size_t np, size_t nqr){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	std::pair<ArrayXXd, ArrayXXd> tmp = intgj(tmpa, nstp, np, nqr);
	return bp::make_tuple(copy2bn(tmp.first), copy2bn(tmp.second));
    }

    /* wrap intgv */
    bn::ndarray PYintgv(const bn::ndarray &a0, const bn::ndarray &v,
		      size_t nstp){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	getDims(v, m, n);
	Map<ArrayXXd> tmpv((double*)v.get_data(), n, m);
	return copy2bn( intgv(tmpa, tmpv, nstp));
    }

    /* wrap intgvs */
   bp::tuple PYintgvs(const bn::ndarray &a0, const bn::ndarray &v,
		      int nstp, int np, int nqr){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa((double*)a0.get_data(), m*n);
	getDims(v, m, n);
	Map<ArrayXXd> tmpv((double*)v.get_data(), n, m);
	std::pair<ArrayXXd, ArrayXXd> tmp = intgvs(tmpa, tmpv, nstp, np, nqr);
	return bp::make_tuple(copy2bn(tmp.first), copy2bn(tmp.second));
    }
    
    /* wrap Fourier2Config */
    bn::ndarray PYFourier2Config(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( Fourier2Config(tmpaa) );
    }

    /* wrap Config2Fourier */
    bn::ndarray PYConfig2Fourier(bn::ndarray AA){
	int m, n;
	getDims(AA, m, n);
	Map<ArrayXXd> tmpAA((double*)AA.get_data(), n, m);
	return copy2bn( Config2Fourier(tmpAA) );
    }
    
    /* wrap Fourier2ConfigMag */
    bn::ndarray PYFourier2ConfigMag(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( Fourier2ConfigMag(tmpaa) );
    }

    /* wrap calPhase */
    bn::ndarray PYcalPhase(bn::ndarray AA){
	int m, n;
	getDims(AA, m, n);
	Map<ArrayXXd> tmpAA((double*)AA.get_data(), n, m);
	return copy2bn( calPhase(tmpAA) );
    }
    
    /* wrap Fourier2Phase */
    bn::ndarray PYFourier2Phase(bn::ndarray aa){
	int m, n;
	getDims(aa, m, n);
	Map<ArrayXXd> tmpaa((double*)aa.get_data(), n, m);
	return copy2bn( Fourier2Phase(tmpaa) );
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

    /* reduceIntg */
    bp::tuple PYreduceIntg(const bn::ndarray &a0, const size_t nstp,
			   const size_t np){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	std::tuple<ArrayXXd, ArrayXd, ArrayXd> tmp = 
	    reduceIntg(tmpa0, nstp, np);
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
    
    /* findReq */
    bp::tuple PYfindReq(bn::ndarray a0, double wth0, double wphi0,
			int MaxN, double tol, bool doesUseMyCG,
			bool doesPrint){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	std::tuple<ArrayXd, double, double, double> tmp = 
	    findReq(tmpa0, wth0, wphi0, MaxN, tol, doesUseMyCG, doesPrint);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)),  std::get<1>(tmp),
			      std::get<2>(tmp), std::get<3>(tmp));
    }
    
    /* optThPhi */
    bp::list PYoptThPhi(bn::ndarray a0){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	
	return toList(optThPhi(tmpa0));
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
	return copy2bn( transRotate(tmpaa, phi) );
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
    
    /* planeWave */
    bp::tuple PYplaneWave(int k, bool isPositve){
	auto tmp = planeWave(k, isPositve);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)),
			      std::get<1>(tmp),
			      std::get<2>(tmp));
    }
    
    /* powIt */
    bp::tuple PYpowIt(bn::ndarray a0, double th, double phi, bn::ndarray Q0, 
		      bool onlyLastQ, int nstp, int nqr, 
		      int maxit, double Qtol, bool Print, int PrintFreqency){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	getDims(Q0, m, n);	
	Map<MatrixXd> tmpQ0((double*)Q0.get_data(), n, m);
	auto tmp = powIt(tmpa0, th, phi, tmpQ0, onlyLastQ, nstp, nqr, maxit, Qtol, Print, PrintFreqency);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)),
			      copy2bn(std::get<1>(tmp)),
			      copy2bn(std::get<2>(tmp)),
			      toList(std::get<3>(tmp))
			      );
    }

    /* powEigE */
    bn::ndarray PYpowEigE(bn::ndarray a0, double th, double phi, bn::ndarray Q0, 
			  int nstp, int nqr, 
			  int maxit, double Qtol, bool Print, int PrintFreqency){
	int m, n;
	getDims(a0, m, n);
	Map<ArrayXd> tmpa0((double*)a0.get_data(), n*m);
	getDims(Q0, m, n);	
	Map<MatrixXd> tmpQ0((double*)Q0.get_data(), n, m);
	MatrixXd tmp = powEigE(tmpa0, th, phi, tmpQ0, nstp, nqr, maxit, Qtol, Print, PrintFreqency);
	return copy2bn(tmp);
    }
    
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



BOOST_PYTHON_MODULE(py_cqcgl1d_threads) {
    bn::initialize();

    // must provide the constructor
    bp::class_<CQCGL>("CQCGL", bp::init<
		      int, double, 
		      double, double, double, double, 
		      int, int>())
	;
    
    
    bp::class_<pyCQCGL, bp::bases<CQCGL> >("pyCQCGL", bp::init<
					   int, double, 
					   double, double, double, double,
					   int, int >())
	.def_readonly("N", &pyCQCGL::N)
	.def_readonly("d", &pyCQCGL::d)
	.def_readonly("Mu", &pyCQCGL::Mu)
	.def_readonly("Br", &pyCQCGL::Br)
	.def_readonly("Bi", &pyCQCGL::Bi)
	.def_readonly("Dr", &pyCQCGL::Dr)
	.def_readonly("Di", &pyCQCGL::Di)
	.def_readonly("Gr", &pyCQCGL::Gr)
	.def_readonly("Gi", &pyCQCGL::Gi)
	.def_readonly("Ndim", &pyCQCGL::Ndim)
	.def_readonly("Ne", &pyCQCGL::Ne)
	.def_readonly("b", &pyCQCGL::b)
	.def_readonly("c", &pyCQCGL::c)
	.def_readonly("dr", &pyCQCGL::dr)
	.def_readonly("di", &pyCQCGL::di)
	.def("K", &pyCQCGL::PYK)
	.def("L", &pyCQCGL::PYL)
	.def("velocity", &pyCQCGL::PYvelocity)
	.def("velSlice", &pyCQCGL::PYvelSlice)
	.def("velPhase", &pyCQCGL::PYvelPhase)
	.def("rk4", &pyCQCGL::PYrk4)
	.def("rk4j", &pyCQCGL::PYrk4j)
	.def("velocityReq", &pyCQCGL::PYvelocityReq)
	.def("Lyap", &pyCQCGL::PYLyap)
	.def("LyapVel", &pyCQCGL::PYLyapVel)
	.def("pad", &pyCQCGL::PYpad)
	.def("generalPadding", &pyCQCGL::PYgeneralPadding)
	.def("unpad", &pyCQCGL::PYunpad)
	.def("intg", &pyCQCGL::PYintg)
	.def("intgj", &pyCQCGL::PYintgj)
	.def("intgv", &pyCQCGL::PYintgv)
	.def("intgvs", &pyCQCGL::PYintgvs)
	.def("Fourier2Config", &pyCQCGL::PYFourier2Config)
	.def("Config2Fourier", &pyCQCGL::PYConfig2Fourier)
	.def("Fourier2ConfigMag", &pyCQCGL::PYFourier2ConfigMag)
	.def("calPhase", &pyCQCGL::PYcalPhase)
	.def("Fourier2Phase", &pyCQCGL::PYFourier2Phase)
	.def("orbit2sliceWrap", &pyCQCGL::PYorbit2sliceWrap)
	.def("orbit2slice", &pyCQCGL::PYorbit2slice)
	.def("stab", &pyCQCGL::PYstab)
	.def("stabReq", &pyCQCGL::PYstabReq)
	.def("reflect", &pyCQCGL::PYreflect)
	.def("reduceReflection", &pyCQCGL::PYreduceReflection)
	.def("refGradMat", &pyCQCGL::PYrefGradMat)
	.def("reflectVe", &pyCQCGL::PYreflectVe)
	.def("reflectVeAll", &pyCQCGL::PYreflectVeAll)
	.def("ve2slice", &pyCQCGL::PYve2slice)
	.def("reduceAllSymmetries", &pyCQCGL::PYreduceAllSymmetries)
	.def("reduceIntg", &pyCQCGL::PYreduceIntg)
	.def("reduceVe", &pyCQCGL::PYreduceVe)
	.def("findReq", &pyCQCGL::PYfindReq)
	.def("optThPhi", &pyCQCGL::PYoptThPhi)
	.def("transRotate", &pyCQCGL::PYtransRotate) 
	.def("transTangent", &pyCQCGL::PYtransTangent)
	.def("phaseRotate", &pyCQCGL::PYphaseRotate)
	.def("phaseTangent", &pyCQCGL::PYphaseTangent)
	.def("Rotate", &pyCQCGL::PYRotate)
	.def("rotateOrbit", &pyCQCGL::PYrotateOrbit)
	.def("planeWave", &pyCQCGL::PYplaneWave)
	.def("powIt", &pyCQCGL::PYpowIt)
	.def("powEigE", &pyCQCGL::PYpowEigE)
	;

}
