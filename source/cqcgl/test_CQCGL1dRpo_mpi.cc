// to compile
//
// first use
// mpicxx --showme -O3 test_CQCGL1dRpo_mpi.cc  -L../../lib -I../../include -I$EIGEN -std=c++11 -lCQCGL1dRpo -lCQCGL1d -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -lmyfft -lfftw3 -lm
//
// then change g++ to h5c++
// 
// h5c++ -O3 test_CQCGL1dRpo_mpi.cc -L../../lib -I../../include -I/usr/local/home/xiong/apps/eigen/include/eigen3 -std=c++11 -lCQCGL1dRpo -lCQCGL1d -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -lmyfft -lfftw3 -lm -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi -pthread -L/usr//lib -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi -ldl -lhwloc 


#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include <ctime>
#include <mpi.h>

#include "CQCGL1dReq.hpp"
#include "CQCGL1dRpo.hpp"

using namespace std; 
using namespace Eigen;
using namespace iterMethod;

#define cee(x) (cout << (x) << endl << endl)

#define CASE_10

int main(int argc, char **argv){
    
    cout.precision(15);
    GMRES_IN_PRINT_FREQUENCE = 50;
    HOOK_PRINT_FREQUENCE = 1;
    GMRES_OUT_PRINT = false;
    
#ifdef CASE_10
    //====================================================================== 
    // find limit cycles by varying Bi and Gi
    const int N = 1024;
    const int L = 50;
    double Bi = 1.9;
    double Gi = -5.6;

    int id = 1;
    CQCGL1dRpo cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
	
    string file = "../../data/cgl/rpoBiGi2.h5";
    double stepB = 0.1;
    int NsB = 39;

    ////////////////////////////////////////////////////////////
    // mpi part 
    MPI_Init(&argc, &argv);
    int rank, num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    int inc = NsB / num;
    int rem = NsB - inc * num;
    int p_size = inc + (rank < rem ? 1 : 0);
    int p_start = inc*rank + (rank < rem ? rank : rem);
    int p_end = p_start + p_size;
    fprintf(stderr, "MPI : %d / %d; range : %d - %d \n", rank, num, p_start, p_end);
    ////////////////////////////////////////////////////////////

    for (int i = p_start; i < p_end; i++){
	cgl.Bi = Bi + i*stepB;
	cgl.findRpoParaSeq(file, 1, 0.1, 55, false);
    }
    

    ////////////////////////////////////////////////////////////
    MPI_Finalize();
    ////////////////////////////////////////////////////////////

#endif

    return 0;
}
