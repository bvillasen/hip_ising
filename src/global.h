#ifndef GLOBAL_H
#define GLOBAL_H

#include <iostream>
#include <hip/hip_runtime.h>

#define TPB 1024

#define HIP_CHECK(command) {   \
  hipError_t status = command; \
  if (status!=hipSuccess) {    \
    std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
    std::abort(); }}
  


#endif