#include "grid3D.h"
#include "gpu.hpp"


Grid3D::Grid3D( int N_val ){

  N = N_val;
  n_ghost = 1; 
  N_cells_real = N * N * N;
  N_cells_total = (N + 2*n_ghost) * (N + 2*n_ghost) * (N + 2*n_ghost);
  

}

void Grid3D::Print_size(){
  std::cout << "Grid3D:  N=" << N << "   N_cells=" << N_cells_total << std::endl;
}

void Grid3D::Initialize(){

  std::cout << "Grid3D: Initializing Memory..." << std::endl;  
  Allocate_CPU_Memory();
  Allocate_GPU_Memory();
  
}

void Grid3D::Allocate_CPU_Memory(){

  CHECK( hipHostMalloc( &h_spins, N_cells_total*sizeof(double), hipHostMallocDefault ) );
  std::cout <<  " Allocated CPU Memory." << std::endl;

}


void Grid3D::Allocate_GPU_Memory(){

CHECK( hipMalloc( (void**)&d_spins, N_cells_total*sizeof(double) ) );
CHECK( hipMalloc( (void**)&d_spins_out, N_cells_total*sizeof(double) ) );
CHECK( hipMalloc( (void**)&d_random_field, N_cells_total*sizeof(float) ) );
std::cout <<  " Allocated GPU Memory." << std::endl;

}

void Grid3D::Print_Timers(){
  std::cout << "Time Random: " << Time.Gen_Rand.elapsed_time << std::endl;
  std::cout << "Time Ising : " << Time.Ising_Update.elapsed_time << std::endl;

}

void Grid3D::Reset( ){

  CHECK( hipHostFree( h_spins ) );
  CHECK( hipFree( d_spins ) );
  CHECK( hipFree( d_spins_out ) );
  CHECK( hipFree( d_random_field ) );

  hiprandDestroyGenerator( RND.generator );


  std::cout << "Grid3D Reset Complete." << std::endl; 


}