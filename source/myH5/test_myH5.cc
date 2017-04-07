// how to compile
// h5c++ -O3 test_myH5.cc -L../../lib -I../../include -I$EIGEN -std=c++11 -lmyH5 -lm

#include "myH5.hpp"

using namespace H5;
using namespace std;
using namespace Eigen;
using namespace MyH5;

#define N10

int main(){
    
#ifdef N10
    //======================================================================
    string s = "/usr/local/home/xiong/00git/research/data/cgl/reqsubBiGi.h5";
    vector<string> x = scanGroup(s);
    for(auto s : x)
	cout << s << endl;
    
#endif
    return 0;
}
