#ifndef GRID3D_H
#define GRID3D_H

#include "global.h"
#include "timing.h"

#ifdef USE_HIP
#include<hiprand/hiprand.h>
#endif

class Grid3D{
public:

  int N;
  int N_cells_real;
  int N_cells_total;
  int n_ghost;

  double *h_spins;
  double *d_spins;
  double *d_spins_out;
  float *d_random_field;

  struct RANDOM{
    hiprandGenerator_t generator;

  } RND;

  struct Timers{
    Timer Gen_Rand;
    Timer Ising_Update;

  } Time;



Grid3D( int N_val );
void Print_size();
void Initialize();
void Allocate_CPU_Memory();
void Allocate_GPU_Memory();
void Initialize_Random_Generator( int seed );
void Generate_Random_Field();
void Apply_Ising_Update();
void Print_Timers();

void Reset();
};


#endif