//********************************************************//
// CUDA extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <cstdio>

#include "cudautils.h"
#include "cudaImage.h"

int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
int iDivDown(int a, int b) { return a/b; }
int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }
int iAlignDown(int a, int b) { return a - a%b; }

__host__ __device__ CudaImage::CudaImage() : width(0), height(0), d_data(nullptr), h_data(nullptr), t_data(nullptr), d_internalAlloc(false), h_internalAlloc(false){}

__host__ void CudaImage::Download(size_t &src_pitch, cudaStream_t &stream)
{
  int p = pitch * sizeof(float);
  if (d_data != nullptr && h_data != nullptr)
  {
    safeCall(cudaMemcpy2DAsync(d_data, p, h_data, src_pitch, sizeof(float) * width, height, cudaMemcpyHostToDevice, stream));
  }
}