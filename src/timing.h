#ifndef TIMING_H
#define TIMING_H


class Timer
{
 public:
  const char* name;
  int n_steps = 0;
  double time_start;
  double elapsed_time;
  
  Timer(){}
  
  Timer(const char* input_name){
    name = input_name;
  }

  void Start();
  void End();
  void RecordTime( float time );
};

#endif