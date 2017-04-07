#include <iostream>
#include <functional>
#include "CQCGL1dSubReq.hpp"
#include "iterMethod.hpp"


#define cee(x) (cout << (x) << endl << endl)

namespace ph = std::placeholders;

using namespace std;
using namespace Eigen;
using namespace iterMethod;
using namespace denseRoutines;
using namespace MyH5;

//////////////////////////////////////////////////////////////////////
//                      constructor                                 //
//////////////////////////////////////////////////////////////////////

// A_t = Mu A + (Dr + Di*i) A_{xx} + (Br + Bi*i) |A|^2 A + (Gr + Gi*i) |A|^4 A
CQCGL1dSubReq::CQCGL1dSubReq(int N, double d,
			     double Mu, double Dr, double Di, double Br, double Bi, 
			     double Gr, double Gi, int dimTan):
    CQCGL1dSub(N, d, Mu, Dr, Di, Br, Bi, Gr, Gi, dimTan) {}

// A_t = -A + (1 + b*i) A_{xx} + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
CQCGL1dSubReq::CQCGL1dSubReq(int N, double d, 
			     double b, double c, double dr, double di, 
			     int dimTan):
    CQCGL1dSub(N, d, b, c, dr, di, dimTan){}
    
// iA_z + D A_{tt} + |A|^2 A + \nu |A|^4 A = i \delta A + i \beta A_{tt} + i |A|^2 A + i \mu |A|^4 A
CQCGL1dSubReq::CQCGL1dSubReq(int N, double d,
			     double delta, double beta, double D, double epsilon,
			     double mu, double nu, int dimTan) :
    CQCGL1dSub(N, d, delta, beta, D, epsilon, mu, nu, dimTan) {}

CQCGL1dSubReq::~CQCGL1dSubReq(){}

CQCGL1dSubReq & CQCGL1dSubReq::operator=(const CQCGL1dSubReq &x){
    return *this;
}


//////////////////////////////////////////////////////////////////////
//                      member functions                            //
//////////////////////////////////////////////////////////////////////

std::string 
CQCGL1dSubReq::toStr(double Bi, double Gi, int id){
    //  avoid possibilty that 000000.000000 or -00000.000000
    if (fabs(Bi) < 1e-6) Bi = 0; 
    if (fabs(Gi) < 1e-6) Gi = 0;

    char g1[20], g2[20];
    sprintf(g1, "%013.6f", Bi);
    sprintf(g2, "%013.6f", Gi);
    
    string s1(g1);
    string s2(g2);
    string s = s1 + '/' + s2 + '/' + to_string(id);
    
    return s;
}

/**
 * @brief read req (relative equibrium) info from hdf5 file
 *
 */
std::tuple<VectorXd, double, double>
CQCGL1dSubReq::read(H5File &file, const std::string groupName){
    // H5File file(fileName, H5F_ACC_RDONLY);
    string DS = "/" + groupName + "/";

    return make_tuple(readMatrixXd(file, DS + "a").col(0),
		      readScalar<double>(file, DS + "wphi"),
		      readScalar<double>(file, DS + "err")
		      );
}



/**
 * @brief write [a, wth, wphi, err] of Req of cqcgl into a group
 * 
 * @note group should be a new group
 */
void 
CQCGL1dSubReq::write(H5File &file, const std::string groupName,
		     const ArrayXd &a, const double wphi, 
		     const double err){
    
    // H5File file(fileName, H5F_ACC_RDWR);
    checkGroup(file, groupName, true);
    string DS = "/" + groupName + "/";
	
    writeMatrixXd(file, DS + "a", a);
    writeScalar<double>(file, DS + "wphi", wphi);
    writeScalar<double>(file, DS + "err", err);
}

VectorXcd 
CQCGL1dSubReq::readE(H5File &file, const std::string groupName){
    string DS = "/" + groupName + "/";
    VectorXd er = readMatrixXd(file, DS + "er");
    VectorXd ei = readMatrixXd(file, DS + "ei");
    VectorXcd e(er.size());
    e.real() = er;
    e.imag() = ei;

    return e;
}

MatrixXcd 
CQCGL1dSubReq::readV(H5File &file, const std::string groupName){
    string DS = "/" + groupName + "/";
    MatrixXd vr = readMatrixXd(file, DS + "vr");
    MatrixXd vi = readMatrixXd(file, DS + "vi");
    MatrixXcd v(vr.rows(), vr.cols());
    v.real() = vr;
    v.imag() = vi;

    return v;
}

void 
CQCGL1dSubReq::writeE(H5File &file, const std::string groupName, 
		      const VectorXcd e){
    // H5File file(fileName, H5F_ACC_RDWR);
    checkGroup(file, groupName, true);
    string DS = "/" + groupName + "/";
    
    writeMatrixXd(file, DS + "er", e.real());
    writeMatrixXd(file, DS + "ei", e.imag());
}


void 
CQCGL1dSubReq::writeV(H5File &file, const std::string groupName, 
		      const MatrixXcd v){
    // H5File file(fileName, H5F_ACC_RDWR);
    checkGroup(file, groupName, true);
    string DS = "/" + groupName + "/";
    
    writeMatrixXd(file, DS + "vr", v.real());
    writeMatrixXd(file, DS + "vi", v.imag());
}

void 
CQCGL1dSubReq::move(H5File &fin, std::string gin, H5File &fout, std::string gout,
		    int flag){
    VectorXd a;
    VectorXcd e;
    MatrixXcd v;
    double wphi, err;

    std::tie(a, wphi, err) = read(fin, gin);
    if (flag == 1 || flag == 2) e = readE(fin, gin);
    if (flag == 2) v = readV(fin, gin);
    
    write(fout, gout, a, wphi, err);
    if (flag == 1 || flag == 2) writeE(fout, gout, e);
    if (flag == 2) writeV(fout, gout, v);
}

//====================================================================================================

VectorXd
CQCGL1dSubReq::Fx(const VectorXd &x){
    assert(x.size() == Ndim + 1);

    double wphi = x(Ndim); 
    VectorXd a = x.head(Ndim);	// use matrix type for resizing

    VectorXd F(Ndim+1);
    F << velocityReq(a, wphi), 0;
    
    return F;
}

std::tuple<MatrixXd, MatrixXd, VectorXd>
CQCGL1dSubReq::calJJF(const VectorXd &x){
    assert(x.size() == Ndim + 1);
    
    double wphi = x(Ndim); 
    VectorXd a0 = x.head(Ndim);
  
    MatrixXd DF(Ndim, Ndim+1); 
    ArrayXd tx_phase = phaseTangent(a0); 
    DF.topLeftCorner(Ndim, Ndim) = stabReq(a0, wphi); 
    DF.col(Ndim) = tx_phase;
    VectorXd F = velocity(a0) + wphi*tx_phase;


    MatrixXd JJ = DF.transpose() * DF;
    MatrixXd D  = JJ.diagonal().asDiagonal();
    VectorXd df = DF.transpose() * F; 

    return std::make_tuple(JJ, D, df);
}

std::tuple<VectorXd, double, double, int>
CQCGL1dSubReq::findReq_LM(const ArrayXd &a0, const double wphi0, 
			  const double tol,
			  const int maxit,
			  const int innerMaxit){
    
    VectorXd x(Ndim+1);
    x << a0, wphi0;
    
    auto fx = std::bind(&CQCGL1dSubReq::Fx, this, ph::_1);    
    CQCGL1dSubReqJJF<MatrixXd> jj(this);
    PartialPivLU<MatrixXd> solver;
    
    VectorXd xe;
    std::vector<double> res;
    int flag;
    std::tie(xe, res, flag) = LM0(fx, jj, solver, x, tol, maxit, innerMaxit);    
    if(flag != 0) fprintf(stderr, "Sub Req not converged ! \n");
    
    VectorXd a = xe.head(Ndim);
    double wphi = xe(Ndim);
    return std::make_tuple( a, wphi, res.back(), flag );
}


/** @brief find the optimal guess of wphi for a candidate req
 *  @return    [wth, wphi, err] such that velocityReq(a0, wth, wphi) minimal
 */
std::vector<double>
CQCGL1dSubReq::optThPhi(const ArrayXd &a0){ 
    VectorXd t = phaseTangent(a0);
    VectorXd v = velocity(a0);
    double c = t.dot(v) / t.dot(t);
    
    double err = (v - c * t).norm();
    
    std::vector<double> x{-c, err};
    return x;
}

#if 0
/**
 * @brief find req with a sequence of Bi or Gi
 */ 
void 
CQCGL1dSubReq::findReqParaSeq(H5File &file, int id, double step, int Ns, bool isBi){
    double Bi0 = Bi;
    double Gi0 = Gi;
    
    ArrayXd a0;
    double wth0, wphi0, err0;
    std::tie(a0, wth0, wphi0, err0) = read(file, toStr(Bi, Gi, id));
    
    ArrayXd a;
    double wth, wphi, err;
    int flag;
    
    int Nfail = 0;

    for (int i = 0; i < Ns; i++){
	if (isBi) Bi += step;
	else Gi += step;
	
	// if exist, use it as seed, otherwise find req
	if ( checkGroup(file, toStr(Bi, Gi, id), false) ){ 
	    std::tie(a0, wth0, wphi0, err0) = read(file, toStr(Bi, Gi, id));
	}
	else {
	    fprintf(stderr, "%d %g %g \n", id, Bi, Gi);
	    std::tie(a, wth, wphi, err, flag) = findReq_LM(a0, wth0, wphi0, 1e-10, 100, 1000);
	    if (flag == 0){
		write(file, toStr(Bi, Gi, id), a, wth, wphi, err);
		a0 = a;
		wth0 = wth;
		wphi0 = wphi;
	    }
	    else {
		if(++Nfail == 3) break;
	    }
	}
    }
    
    Bi = Bi0; 			// restore Bi, Gi
    Gi = Gi0;
}
#endif

/// @brief calculate the eigenvalues and eigenvectors of req in certain range
/// gs is obtained by scanGroup()
void 
CQCGL1dSubReq::calEVParaSeq(H5File &file, vector<vector<string>> gs, bool saveV){
    double Bi0 = Bi;
    double Gi0 = Gi;
    int id;

    ArrayXd a0;
    double wth0, wphi0, err0;
    VectorXcd e;
    MatrixXcd v;
    
    for (auto entry : gs){
	Bi = stod(entry[0]);
	Gi = stod(entry[1]);
	id = stoi(entry[2]);
	if( !checkGroup(file, toStr(Bi, Gi, id) + "/er", false) ){
	    fprintf(stderr, "%d %g %g \n", id, Bi, Gi);
	    std::tie(a0, wphi0, err0) = read(file, toStr(Bi, Gi, id));
	    std::tie(e, v) = evReq(a0, wphi0); 
	    writeE(file, toStr(Bi, Gi, id), e);
	    if(saveV) writeV(file, toStr(Bi, Gi, id), v.leftCols(10));
	}
    }

    Bi = Bi0; 			// restore Bi, Gi
    Gi = Gi0;
}

