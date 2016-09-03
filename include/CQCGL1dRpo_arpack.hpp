#ifndef CQCGL1dRpo_arpack_H
#define CQCGL1dRpo_arpack_H

#include "CQCGL1dRpo.hpp"

class CQCGL1dRpo_arpack : public CQCGL1dRpo {

public:

    typedef std::complex<double> dcp;

    ////////////////////////////////////////////////////////////
    // A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
    CQCGL1dRpo_arpack(int N, double d,
		      double Mu, double Dr, double Di, double Br, double Bi, 
		      double Gr, double Gi, int dimTan);
    
    // A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
    CQCGL1dRpo_arpack(int N, double d, 
		      double b, double c, double dr, double di, 
		      int dimTan);
    
    // iA_z + D A_{tt} + |A|^2 A + \nu |A|^4 A = i \delta A + i \beta A_{tt} + i |A|^2 A + i \mu |A|^4 A
    CQCGL1dRpo_arpack(int N, double d,
		      double delta, double beta, double D, double epsilon,
		      double mu, double nu, int dimTan);
    ~CQCGL1dRpo_arpack();
    CQCGL1dRpo_arpack & operator=(const CQCGL1dRpo_arpack &x);

    ////////////////////////////////////////////////////////////
    std::pair<VectorXcd, MatrixXd>
    evRpo(const ArrayXd &a0, double h, int nstp, 
	  double th, double phi, int ne);
    
    void 
    calEVParaSeq(std::string file, std::vector<double> Bis, 
		 std::vector<double> Gis, int ne, bool saveV);

    static
    std::pair< std::vector<double>, std::vector<double> >
    getMissIds(std::string file, double Bi0, double Gi0, 
	       double incB, double incG, int nB, int nG);
};


struct Jdotx {
    CQCGL1dRpo_arpack *rpo;
    ArrayXd a0;
    double h, th, phi;
    int Nt;
    
    Jdotx(CQCGL1dRpo_arpack *rpo, const ArrayXd &a0, double h, int Nt,
	  double th, double phi) 
	: rpo(rpo), a0(a0), h(h), Nt(Nt), th(th), phi(phi){}
    
    void mul(double *v, double *w){
	int Ndim = rpo->Ndim;
	Map<const ArrayXd> mv(v, Ndim);
	Map<ArrayXd> mw(w, Ndim);
	ArrayXXd tmp = rpo->intgv(a0, mv, h, Nt);
	mw = rpo->Rotate(tmp.rightCols(1), th, phi);
    }
    
};


#endif	/* CQCGL1dRpo_arpack */
