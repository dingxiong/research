#include "CQCGL1dRpo_arpack.hpp"
#include <arsnsym.h>
#include "myH5.hpp"

#define cee(x) (cout << (x) << endl << endl)

using namespace denseRoutines;
using namespace iterMethod;
using namespace Eigen;
using namespace std;
using namespace MyH5;


//////////////////////////////////////////////////////////////////////
//                      constructor                                 //
//////////////////////////////////////////////////////////////////////

// A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
CQCGL1dRpo_arpack::CQCGL1dRpo_arpack(int N, double d,
				     double Mu, double Dr, double Di, double Br, double Bi, 
				     double Gr, double Gi, int dimTan):
    CQCGL1dRpo(N, d, Mu, Dr, Di, Br, Bi, Gr, Gi, dimTan) {}

// A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
CQCGL1dRpo_arpack::CQCGL1dRpo_arpack(int N, double d, 
				     double b, double c, double dr, double di, 
				     int dimTan):
    CQCGL1dRpo(N, d, b, c, dr, di, dimTan){}
    
// iA_z + D A_{tt} + |A|^2 A + \nu |A|^4 A = i \delta A + i \beta A_{tt} + i |A|^2 A + i \mu |A|^4 A
CQCGL1dRpo_arpack::CQCGL1dRpo_arpack(int N, double d,
				     double delta, double beta, double D, double epsilon,
				     double mu, double nu, int dimTan) :
    CQCGL1dRpo(N, d, delta, beta, D, epsilon, mu, nu, dimTan) {}

CQCGL1dRpo_arpack::~CQCGL1dRpo_arpack(){}

CQCGL1dRpo_arpack & CQCGL1dRpo_arpack::operator=(const CQCGL1dRpo_arpack &x){
    return *this;
}

/////////////////////////////////////////////////////////////////////////
//                     member functions                                //
/////////////////////////////////////////////////////////////////////////

/// @param[in] ne  number of eigenvalues wanted
std::pair<VectorXcd, MatrixXd>
CQCGL1dRpo_arpack::evRpo(const ArrayXd &a0, double h, int nstp, 
			 double th, double phi, int ne){
    int n = Ndim;
    
    VectorXd er(ne+1), ei(ne+1);
    MatrixXd v((ne+1)*n, 1);
    double *p_er = er.data();
    double *p_ei = ei.data();
    double *p_v = v.data();

    Jdotx jdotx(this, a0, h, nstp, th, phi);
    ARNonSymStdEig<double, Jdotx> dprob(n, ne, &jdotx, &Jdotx::mul, "LM");
    dprob.ChangeTol(1e-9);

    int nconv = dprob.EigenValVectors(p_v, p_er, p_ei);
    if (nconv < ne) fprintf(stderr, "arpack not converged. nconv = %d\n", nconv);

    // truncate if necessary
    VectorXcd e(nconv);
    e.real() = er.head(nconv);
    e.imag() = ei.head(nconv);
    v.resize(n, ne+1);
    MatrixXcd vc = vr2vc(e, v.leftCols(nconv));
    
    // sort the eigenvalues by their magnitude
    std::vector<int> id = csort(e, 2);
    VectorXcd e2(nconv);
    MatrixXcd vc2(n, nconv);
    for (int i = 0; i < nconv; i++ ){
	e2(i) = e(id[i]);
	vc2.col(i) = vc.col(id[i]);
    }
    MatrixXd vr = vc2vr(vc2);
    
    return std::make_pair(e2, vr);
}

void 
CQCGL1dRpo_arpack::calEVParaSeq(std::string file, std::vector<double> Bis, 
				std::vector<double> Gis, int ne, bool saveV){
    double Bi0 = Bi;
    double Gi0 = Gi;

    ArrayXd x, a0;
    double T, th, phi, err;
    int nstp;
    VectorXcd e;
    MatrixXd v;

    for (int i = 0; i < Bis.size(); i++) {
	Bi = Bis[i];
	for (int j = 0; j < Gis.size(); j++){
	    Gi = Gis[j];
	    std::string g = toStr(Bi, Gi, 1);
	    if( checkGroup(file, g, false) ) {
		if ( !checkGroup(file, g + "/er", false) ) {
		    fprintf(stderr, "%g %g\n", Bi, Gi);
		    std::tie(x, T, nstp, th, phi, err) = readRpo(file, g);
		    a0 = x.head(Ndim);
		    std::tie(e, v) = evRpo(a0, T/nstp, nstp, th, phi, ne);
		    writeE(file, g, e);
		    if (saveV) writeV(file, g, v);
		}
	    }
	}
    }
    
    Bi = Bi0;
    Gi = Gi0;
}
