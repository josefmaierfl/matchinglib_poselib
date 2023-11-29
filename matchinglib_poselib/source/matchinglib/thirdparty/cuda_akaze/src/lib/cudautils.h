#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cstdio>
#include <iostream>
// #include "cusolverDn.h"

#ifdef WIN32
#include <intrin.h>
#endif

#define safeCall(err)       __safeCall(err, __FILE__, __LINE__)
#define checkMsg(msg)       __checkMsg(msg, __FILE__, __LINE__)
// #define safeSolverCall(err) __safeSolverCall(err, __FILE__, __LINE__)
#define printMsgLine_0()           __printMsgLine(1, nullptr, __LINE__)
#define printMsgLine_1(code)       __printMsgLine(code, nullptr, __LINE__)
#define printMsgLine_2(code, msg)  __printMsgLine(code, msg, __LINE__)

#define printMsgLine_X(x, A, B, FUNC, ...) FUNC

#define printMsgLine(...) printMsgLine_X(, ##__VA_ARGS__,             \
                                         printMsgLine_2(__VA_ARGS__), \
                                         printMsgLine_1(__VA_ARGS__), \
                                         printMsgLine_0(__VA_ARGS__))

inline void __safeCall(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err) {
    fprintf(stderr, "safeCall() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

// inline void __safeSolverCall(cusolverStatus_t err, const char *file, const int line)
// {
//   if (CUSOLVER_STATUS_SUCCESS != err)
//   {
//     fprintf(stderr, "__safeSloverCall() Runtime API error in file <%s>, line %i : %i.\n", file, line, err);
//     exit(-1);
//   }
// }

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

inline void __printMsgLine(const int code, const char *errorMessage, const int line)
{
  if (code != 0)
  {
    if (errorMessage){
      fprintf(stderr, "printMsgLine() line %i : %s.\n", line, errorMessage);
    }else{
      fprintf(stderr, "Call to printMsgLine() in line %i.\n", line);
    }
  }
}

inline bool deviceInit(int dev)
{
  int deviceCount;
  safeCall(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
    return false;
  }
  if (dev < 0) dev = 0;						
  if (dev > deviceCount-1) dev = deviceCount - 1;
  cudaDeviceProp deviceProp;
  safeCall(cudaGetDeviceProperties(&deviceProp, dev));
  if (deviceProp.major < 1) {
    fprintf(stderr, "error: device does not support CUDA.\n");
    return false;					
  }
  safeCall(cudaSetDevice(dev));
  return true;
}

class TimerCPU
{
  static const int bits = 10;
public:
  long long beg_clock;
  float freq;
  TimerCPU(float freq_) : freq(freq_) {   // freq = clock frequency in MHz
    beg_clock = getTSC(bits);
  }
  long long getTSC(int bits) {
#ifdef WIN32
    return __rdtsc()/(1LL<<bits);
#else
    unsigned int low, high;
    __asm__(".byte 0x0f, 0x31" :"=a" (low), "=d" (high));
    return ((long long)high<<(32-bits)) | ((long long)low>>bits);
#endif
  }
  float read() {
    long long end_clock = getTSC(bits);
    long long Kcycles = end_clock - beg_clock;
    float time = (float)(1<<bits)*Kcycles/freq/1e3f;
    return time;
  }
};


#endif

