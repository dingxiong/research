#include <iostream>
#include "CQCGL2dDislin.hpp"

using namespace denseRoutines;
using namespace Eigen;
using namespace std;

//////////////////////////////////////////////////////////////////////

/**
 * Constructor of cubic quintic complex Ginzburg-Landau equation
 * A_t = -A + (1 + b*i) (A_{xx}+A_{yy}) + (1 + c*i) |A|^2 A - (dr + di*i) |A|^4 A
 */
CQCGL2dDislin::CQCGL2dDislin(int N, int M, double dx, double dy,
			     double b, double c, double dr, double di,
			     int threadNum)
    : CQCGL2d(N, M, dx, dy, b, c, dr, di, threadNum)
{				
}

CQCGL2dDislin::CQCGL2dDislin(int N, double dx,
		 double b, double c, double dr, double di,
		 int threadNum)
    : CQCGL2d(N, dx, b, c, dr, di, threadNum)
{				
}


CQCGL2dDislin::~CQCGL2dDislin(){}

CQCGL2dDislin & CQCGL2dDislin::operator=(const CQCGL2dDislin &x){
    return *this;
}


//////////////////////////////////////////////////////////////////////
//                              member functions                    //
//////////////////////////////////////////////////////////////////////

/* should resize the field to make it insize the Z-coordinate */
void 
CQCGL2dDislin::initPlot(Dislin &g, const double Lx, const double Ly, 
			const double Lz){
    
    g.scrmod ("revers");	/* background color */
    g.metafl ("cons");		/* file format */
    g.winsiz(600, 425);		/* window size */
    g.clrmod("full");		/* 256 colors */
  
    g.disini ();		/* initialize */

    g.winmod ("none");		/* not holding */
    g.pagera ();		/* border around page */
    g.hwfont ();		/* set hardware font if available */

    g.name   ("X", "x");	/* axis titles */
    g.name   ("Y", "y");
    g.name   ("|A|", "z");

    // g.autres (N, N);		/* number of data points */
    g.axspos (500, 1900);		/* position of axis */
    g.ax3len (1600, 1600, 1600);	/* axis length in plot coordinates */
    g.widbar(50);			/* color bar width */

    // plot 3d axis system
    g.graf3  (0.0, Lx, 0.0, Lx, /* x : lower, upper, 1st label, label step */
	      0.0, Ly, 0.0, Ly,
	      0.0, Lz, 0.0, Lz);

    // g.titlin ("|A|", 4);		/* title up to 4 */
    g.height (50);		/* character height */
    g.title  ();			/* plot title */
}

/**
 * Note, can not use function Fourier2Config() because it will destroy the current state.
 * 
 */
void 
CQCGL2dDislin::plot(Dislin &g, const double t, const double err){
    F[0].ifft();
    ArrayXXd absA = F[0].v2.abs();
    double m = absA.maxCoeff();
    absA = absA / m;

    char buffer[30];
    sprintf(buffer, "%e", err);
    std::string s = "t: " + to_string(t) + "; max|A|: " + std::to_string(m) + 
	"; LTE: " + std::string(buffer);

    g.erase();
    g.titlin(s.c_str(), 4);
    g.title();
    g.pagera ();
    g.crvmat((double *)absA.data(), M, N, 1, 1); 
    g.sendbf();
}

void
CQCGL2dDislin::endPlot(Dislin &g){
    g.disfin();
}

/**
 * @brief Constant time step integrator
 */
void
CQCGL2dDislin::constAnim(const ArrayXXcd &a0, const double h, const int skip_rate){
    F[0].v1 = pad(a0);
    calCoe(h);
    
    Dislin g;
    initPlot(g, dx, dy, 1);
    plot(g, 0, 0);

    double t = 0;
    double du;
    int i = 0;
    while (true){
	oneStep(du, true);
	t += h;
	F[0].v1 = F[4].v1;	// update state
	if ( (i+1)%skip_rate == 0 ) {
	    plot(g, t, du);
	}
	i = (i+1) % skip_rate;
    }	
    
    endPlot(g);
}



/**
 * @brief time step adaptive integrator
 */
void
CQCGL2dDislin::adaptAnim(const ArrayXXcd &a0, const double h0, const int skip_rate){
    double h = h0; 
    calCoe(h);
    F[0].v1 = pad(a0);

    Dislin g;
    initPlot(g, dx, dy, 1);
    plot(g, 0, 0);

    double t = 0;
    double du = 0;
    int i = 0;
    bool doChange, doAccept;

    bool TimeEnds = false;
    while( true ){ 
	
	oneStep(du, true);
	double s = nu * std::pow(rtol/du, 1.0/4);
	double mu = adaptTs(doChange, doAccept, s);
	
	if (doAccept){
	    t += h;
	    F[0].v1 = F[4].v1;
	    if ( (i+1) % skip_rate == 0) plot(g, t, du);
	}
	
	if (doChange) {
	    h *= mu;
	    calCoe(h);
	}

	i = (i+1) % skip_rate;
    }

    endPlot(g);
}
