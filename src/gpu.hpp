#pragma once

#ifdef USE_HIP

#define TPB 1024

#include <hip/hip_runtime.h>

#define CHECK(command) {   \
  hipError_t status = command; \
  if (status!=hipSuccess) {    \
    std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
    std::abort(); }}
  

#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError hipError_t
#define cudaError_t hipError_t
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemset hipMemset
#define cudaReadModeElementType hipReadModeElementType
#define cudaSetDevice hipSetDevice
#define cudaSuccess hipSuccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties

#define cufftDestroy hipfftDestroy
#define cufftDoubleComplex hipfftDoubleComplex
#define cufftDoubleReal hipfftDoubleReal
#define cufftExecD2Z hipfftExecD2Z
#define cufftExecZ2D hipfftExecZ2D
#define cufftExecZ2Z hipfftExecZ2Z
#define cufftHandle hipfftHandle
#define cufftPlan3d hipfftPlan3d
#define cufftPlanMany hipfftPlanMany

#define curandCreateGenerator hiprandCreateGenerator
#define CURAND_RNG_PSEUDO_MTGP32 HIPRAND_RNG_PSEUDO_MTGP32
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandGenerateUniform hiprandGenerateUniform

static void __attribute__((unused)) check(const hipError_t err, const char *const file, const int line)
{
  if (err == hipSuccess) return;
  fprintf(stderr,"HIP ERROR AT LINE %d OF FILE '%s': %s %s\n",line,file,hipGetErrorName(err),hipGetErrorString(err));
  fflush(stderr);
  exit(err);
}

#endif

#ifdef USE_CUDA

#define TPB 1024

#include <cuda_runtime.h>

#define hipLaunchKernelGGL(F,G,B,M,S,...) F<<<G,B,M,S>>>(__VA_ARGS__)


#define CHECK(command) {   \
  cudaError_t status = command; \
  if (status!=cudaSuccess) {    \
    std::cerr << "Error: CUDA reports " << cudaGetErrorString(status) << std::endl; \
    std::abort(); }}

#endif