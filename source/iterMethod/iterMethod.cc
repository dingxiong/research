#include "iterMethod.hpp"

namespace iterMethod {

    /**
     * @brief obtain the cos and sin from coordinate (x, y)
     */
    void rotmat(const double &x, const double &y,
		double *c, double *s){
	if(y == 0){
	    *c = 1;
	    *s = 0;
	}
	else if ( fabs(y) > fabs(x) ){
	    double tmp = x / y;
	    *s = 1.0 / sqrt(1.0 + tmp*tmp);
	    *c = tmp * (*s);
	}
	else {
	    double tmp = y / x;
	    *c = 1.0 / sqrt(1.0 + tmp*tmp);
	    *s = tmp * (*c);
	}
    }

}
