#include "global.h"
#include "grid3D.h"



void Grid3D::Initialize_Random_Generator( int seed ){

  std::cout << "Initializing Random Generator..." << std::endl;
  hiprandCreateGenerator( &RND.generator,  HIPRAND_RNG_PSEUDO_MTGP32 );

  std::cout << " Using seed: " << seed << std::endl;
  hiprandSetPseudoRandomGeneratorSeed( RND.generator, seed );

}

void Grid3D::Generate_Random_Field(){

  Time.Gen_Rand.Start();
  hiprandGenerateUniform( RND.generator, d_random_field, N_cells_total );
  hipDeviceSynchronize();
  Time.Gen_Rand.End();

}


__global__ void Ising_kernel( int N, int n_ghost, int N_total, float *random_field, double *spins_in, double *spins_out ){

  int id, xid, yid, zid;  
  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (N*N);
  yid = (id - zid*N*N) / N;
  xid = id - zid*N*N - yid*N;
  if ( id >= N_total ) return;

  float rnd_number;
  rnd_number = random_field[id];

  spins_out[id] = (double) rnd_number;


}


void Grid3D::Apply_Ising_Update(){

  Time.Ising_Update.Start();

  int ngrid = (N_cells_real - 1) / TPB + 1;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);                                          
  
  hipLaunchKernelGGL(Ising_kernel, dim1dGrid, dim1dBlock, 0, 0, N, n_ghost, N_cells_total, d_random_field, d_spins, d_spins_out );
  hipDeviceSynchronize();
  Time.Ising_Update.End();




}