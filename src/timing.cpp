
#include "timing.h"
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double get_time(void){

  struct timeval timer;
  gettimeofday(&timer,NULL);
  return timer.tv_sec + 1.0e-6*timer.tv_usec;
}

void Timer::Start(){
  time_start = get_time();
}

void Timer::End(){
  double time_end = get_time();
  elapsed_time = (time_end - time_start)*1000;
}