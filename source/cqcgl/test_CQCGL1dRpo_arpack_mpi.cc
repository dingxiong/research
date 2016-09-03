// to compile
//
// add -I$XDAPPS/arpackpp/include -llapack -larpack -lsuperlu -lopenblas
//
// first use
// mpicxx --showme -O3 test_CQCGL1dRpo_mpi.cc  -L../../lib -I../../include -I$EIGEN -std=c++11 -lCQCGL1dRpo -lCQCGL1d -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -lmyfft -lfftw3 -lm
//
// then change g++ to h5c++
// 
// h5c++ -O3 test_CQCGL1dRpo_arpack_mpi.cc -std=c++11 -L../../lib -I$RESH/include -I$EIGEN  -I$XDAPPS/arpackpp/include -lCQCGL1dRpo_arpack -lCQCGL1dRpo -lCQCGL1d -lsparseRoutines -ldenseRoutines -literMethod -lmyH5 -lmyfft -lfftw3 -lm -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi -pthread -L/usr//lib -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi -ldl -lhwloc -llapack -larpack -lsuperlu -lopenblas


#include <iostream>
#include <arsnsym.h>
#include <Eigen/Dense>
#include "CQCGL1dRpo_arpack.hpp"
#include "myH5.hpp"
#include <mpi.h>

using namespace std;
using namespace Eigen;
using namespace denseRoutines;
using namespace MyH5;

#define cee(x) (cout << (x) << endl << endl)

#define CASE_10


int main(int argc, char **argv){

#ifdef CASE_10
    //======================================================================
    // to visulize the limit cycle first 
    const int N = 1024;
    const double L = 50;
    double Bi = 1.9;
    double Gi = -5.6;
    
    CQCGL1dRpo_arpack cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
    string file = "../../data/cgl/rpoBiGiEV.h5";
    
    std::vector<double> Bis, Gis;
    std::tie(Bis, Gis) =  CQCGL1dRpo_arpack::getMissIds(file, Bi, Gi, 0.1, 0.1, 39, 55);
    int Ns = Bis.size();
    cout << Ns << endl;

    ////////////////////////////////////////////////////////////
    // mpi part 
    MPI_Init(&argc, &argv);
    int rank, num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    int inc = Ns / num;
    int rem = Ns - inc * num;
    int p_size = inc + (rank < rem ? 1 : 0);
    int p_start = inc*rank + (rank < rem ? rank : rem);
    int p_end = p_start + p_size;
    std::vector<double> p_Bis, p_Gis;
    for(int i = p_start; i < p_end; i++){
	p_Bis.push_back(Bis[i]);
	p_Gis.push_back(Gis[i]);
    }
    
    fprintf(stderr, "MPI : %d / %d; range : %d - %d \n", rank, num, p_start, p_end);
    ////////////////////////////////////////////////////////////

    cgl.calEVParaSeq(file, p_Bis, p_Gis, 16, true);
    

    ////////////////////////////////////////////////////////////
    MPI_Finalize();
    ////////////////////////////////////////////////////////////


#endif
#ifdef CASE_20
    //======================================================================
    // Calculate E and V along one Gi line for a specific Bi
    const int N = 1024;
    const int L = 50;
    double Bi = 2.3;
    double Gi = -5.6;

    CQCGL1dRpo_arpack cgl(N, L, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 1);
    string file = "../../data/cgl/rpoBiGiEV.h5";    
    std::vector<double> Bis, Gis;
    Bis.push_back(Bi);
    
    int NsG = 16;
    ////////////////////////////////////////////////////////////
    // mpi part 
    MPI_Init(&argc, &argv);
    int rank, num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    int inc = NsG / num;
    int rem = NsG - inc * num;
    int p_size = inc + (rank < rem ? 1 : 0);
    int p_start = inc*rank + (rank < rem ? rank : rem);
    int p_end = p_start + p_size;
    for (int i = p_start; i < p_end; i++) Gis.push_back(Gi+0.1*i);
    fprintf(stderr, "MPI : %d / %d; range : %d - %d \n", rank, num, p_start, p_end);
    ////////////////////////////////////////////////////////////

    cgl.calEVParaSeq(file, Bis, Gis, 10, true);

    ////////////////////////////////////////////////////////////
    MPI_Finalize();
    ////////////////////////////////////////////////////////////


#endif

    return 0;
}

