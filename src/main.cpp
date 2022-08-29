#include <iostream>
#include "grid3D.h"

int main(int argc, char *argv[]){

  int N = 1024;
  int n_iter = 20;

  Grid3D Grid(N);
  Grid.Print_size();

  Grid.Initialize();

  Grid.Initialize_Random_Generator( 123456789 );

  for ( int iter=0; iter<=n_iter; iter++ ){
    std::cout << "\nIter: " << iter << std::endl;
    Grid.Generate_Random_Field();
    Grid.Apply_Ising_Update();
    Grid.Print_Timers();

  }


  Grid.Reset();



}

