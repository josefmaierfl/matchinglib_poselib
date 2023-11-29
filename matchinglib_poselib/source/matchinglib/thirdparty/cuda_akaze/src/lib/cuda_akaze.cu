#include <opencv2/features2d/features2d.hpp>
#include "cuda_akaze.h"
// #include "cublas_v2.h"
// #include "cusolverDn.h"
#include <thread>

// #define VERBOSE

#define CONVROW_W 160
#define CONVCOL_W 32
#define CONVCOL_H 40
#define CONVCOL_S 8

#define SCHARR_W 32
#define SCHARR_H 16

#define NLDSTEP_W 32
#define NLDSTEP_H 13

#define ORIENT_S (13 * 16)
#define EXTRACT_S 64

template <typename T>
void allocCudaVar(T *&data, const size_t &length, cudaStream_t &stream)
{
  // cudaDeviceSynchronize();
  safeCall(cudaMalloc((void **)&data, length * sizeof(T)));
  safeCall(cudaMemsetAsync(data, 0, length * sizeof(T), stream));
}

template void allocCudaVar(int *&, const size_t &, cudaStream_t &);
template void allocCudaVar(unsigned int *&, const size_t &, cudaStream_t &);
template void allocCudaVar(float *&, const size_t &, cudaStream_t &);

void freeCudaVar(void *data)
{
  safeCall(cudaFree(data));
}

void WaitCuda(cudaStream_t &stream)
{
  safeCall(cudaStreamSynchronize(stream));
}

void DestroyStream(cudaStream_t &stream)
{
  safeCall(cudaStreamDestroy(stream));
}

void createStream(cudaStream_t &stream)
{
  safeCall(cudaStreamCreate(&stream));
}

struct Conv_t {
  float *d_Result;
  float *d_Data;
  int width;
  int pitch;
  int height;
};

template <int RADIUS>
__global__ void ConvRowGPU(struct Conv_t s, float *d_Kernel)
{
  __shared__ float data[CONVROW_W + 2 * RADIUS];
  const int tx = threadIdx.x;
  const int minx = blockIdx.x * CONVROW_W;
  const int maxx = min(minx + CONVROW_W, s.width);
  const int yptr = blockIdx.y * s.pitch;
  const int loadPos = minx + tx - RADIUS;
  const int writePos = minx + tx;

  if (loadPos < 0)
    data[tx] = s.d_Data[yptr];
  else if (loadPos >= s.width)
    data[tx] = s.d_Data[yptr + s.width - 1];
  else
    data[tx] = s.d_Data[yptr + loadPos];
  __syncthreads();
  if (writePos < maxx && tx < CONVROW_W) {
    float sum = 0.0f;
    for (int i = 0; i <= (2 * RADIUS); i++)
      sum += data[tx + i] * d_Kernel[i];
    s.d_Result[yptr + writePos] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Column convolution filter
///////////////////////////////////////////////////////////////////////////////
template <int RADIUS>
__global__ void ConvColGPU(struct Conv_t s, float *d_Kernel)
{
  __shared__ float data[CONVCOL_W * (CONVCOL_H + 2 * RADIUS)];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int miny = blockIdx.y * CONVCOL_H;
  const int maxy = min(miny + CONVCOL_H, s.height) - 1;
  const int totStart = miny - RADIUS;
  const int totEnd = maxy + RADIUS;
  const int colStart = blockIdx.x * CONVCOL_W + tx;
  const int colEnd = colStart + (s.height - 1) * s.pitch;
  const int smemStep = CONVCOL_W * CONVCOL_S;
  const int gmemStep = s.pitch * CONVCOL_S;

  if (colStart < s.width) {
    int smemPos = ty * CONVCOL_W + tx;
    int gmemPos = colStart + (totStart + ty) * s.pitch;
    for (int y = totStart + ty; y <= totEnd; y += CONVCOL_S)
    {
      if (y < 0)
        data[smemPos] = s.d_Data[colStart];
      else if (y >= s.height)
        data[smemPos] = s.d_Data[colEnd];
      else
        data[smemPos] = s.d_Data[gmemPos];
      smemPos += smemStep;
      gmemPos += gmemStep;
    }
  }
  __syncthreads();
  if (colStart < s.width) {
    int smemPos = ty * CONVCOL_W + tx;
    int gmemPos = colStart + (miny + ty) * s.pitch;
    for (int y = miny + ty; y <= maxy; y += CONVCOL_S)
    {
      float sum = 0.0f;
      for (int i = 0; i <= 2 * RADIUS; i++)
        sum += data[smemPos + i * CONVCOL_W] * d_Kernel[i];
      s.d_Result[gmemPos] = sum;
      smemPos += smemStep;
      gmemPos += gmemStep;
    }
  }
}

template <int RADIUS>
void SeparableFilter(CudaImage &inimg, CudaImage &outimg, CudaImage &temp,
                     float *h_Kernel, float *d_Kernel, cudaStream_t &stream, const std::string *info)
{
  int width = inimg.width;
  int pitch = inimg.pitch;
  int height = inimg.height;
  float *d_DataA = inimg.d_data;

  float *d_DataB = outimg.d_data;
  float *d_Temp = temp.d_data;
  if (d_DataA == nullptr || d_DataB == nullptr || d_Temp == nullptr)
  {
    printf("SeparableFilter: missing data\n");
    return;
  }
  const unsigned int kernelSize = (2 * RADIUS + 1) * sizeof(float);
  safeCall(cudaMemcpyAsync(d_Kernel, h_Kernel, kernelSize, cudaMemcpyHostToDevice, stream));

  dim3 blockGridRows(iDivUp(width, CONVROW_W), height);
  dim3 threadBlockRows(CONVROW_W + 2 * RADIUS);
  struct Conv_t s;
  s.d_Result = d_Temp;
  s.d_Data = d_DataA;
  s.width = width;
  s.pitch = pitch;
  s.height = height;
  ConvRowGPU<RADIUS><<<blockGridRows, threadBlockRows, 0, stream>>>(s, d_Kernel);
  checkMsg("Cuda error at kernel ConvRowGPU");

  dim3 blockGridColumns(iDivUp(width, CONVCOL_W), iDivUp(height, CONVCOL_H));
  dim3 threadBlockColumns(CONVCOL_W, CONVCOL_S);
  s.d_Result = d_DataB;
  s.d_Data = d_Temp;
  ConvColGPU<RADIUS><<<blockGridColumns, threadBlockColumns, 0, stream>>>(s, d_Kernel);
  checkMsg("Cuda error at kernel ConvColGPU");
}

template <int RADIUS>
void LowPassT(CudaImage &inimg, CudaImage &outimg, CudaImage &temp,
              double var, float *d_Kernel, cudaStream_t &stream, const std::string *info)
{
  float kernel[2 * RADIUS + 1];
  float kernelSum = 0.0f;
  for (int j = -RADIUS; j <= RADIUS; j++) {
    kernel[j + RADIUS] = expf(-1.f * ((float)j * (float)j / (float)var) / 2.f);
    kernelSum += kernel[j + RADIUS];
  }
  for (int j = -RADIUS; j <= RADIUS; j++) kernel[j + RADIUS] /= kernelSum;
  SeparableFilter<RADIUS>(inimg, outimg, temp, kernel, d_Kernel, stream, info);
}

void LowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var,
             int kernsize, cudaStream_t &stream, const std::string *info)
{
  float *d_Kernel = nullptr;
  safeCall(cudaMalloc((void **)&d_Kernel, 21 * sizeof(float)));
  safeCall(cudaMemsetAsync(d_Kernel, 0, 21 * sizeof(float), stream));
  if (kernsize <= 5)
    LowPassT<2>(inimg, outimg, temp, var, d_Kernel, stream, info);
  else if (kernsize <= 7)
    LowPassT<3>(inimg, outimg, temp, var, d_Kernel, stream, info);
  else if (kernsize <= 9)
    LowPassT<4>(inimg, outimg, temp, var, d_Kernel, stream, info);
  else {
    if (kernsize > 11)
      std::cerr << "Kernels larger than 11 not implemented" << std::endl;
    LowPassT<5>(inimg, outimg, temp, var, d_Kernel, stream, info);
  }
  safeCall(cudaStreamSynchronize(stream));
  safeCall(cudaFree(d_Kernel));
}

__global__ void ScharrT(float *imgd, float *lxd, float *lyd, int width,
                       int pitch, int height) {
#define BW (SCHARR_W + 2)
  __shared__ float buffer[BW * (SCHARR_H + 2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * SCHARR_W + tx;
  int y = blockIdx.y * SCHARR_H + ty;
  int xp = (x == 0 ? 1 : (x > width ? width - 2 : x - 1));
  int yp = (y == 0 ? 1 : (y > height ? height - 2 : y - 1));
  buffer[ty * BW + tx] = imgd[yp * pitch + xp];
  __syncthreads();
  if (x < width && y < height && tx < SCHARR_W && ty < SCHARR_H) {
    float *b = buffer + (ty + 1) * BW + (tx + 1);
    float ul = b[-BW - 1];
    float ur = b[-BW + 1];
    float ll = b[+BW - 1];
    float lr = b[+BW + 1];
    lxd[y * pitch + x] = 3.0f * (lr - ll + ur - ul) + 10.0f * (b[+1] - b[-1]);
    lyd[y * pitch + x] = 3.0f * (lr + ll - ur - ul) + 10.0f * (b[BW] - b[-BW]);
  }
}

void Scharr(CudaImage &img, CudaImage &lx, CudaImage &ly, cudaStream_t &stream)
{
  dim3 blocks(iDivUp(img.width, SCHARR_W), iDivUp(img.height, SCHARR_H));
  dim3 threads(SCHARR_W + 2, SCHARR_H + 2);
  ScharrT<<<blocks, threads, 0, stream>>>(img.d_data, lx.d_data, ly.d_data, img.width, img.pitch, img.height);
  checkMsg("Cuda error at kernel ScharrT");
}

__global__ void FlowT(float *imgd, float *flowd, int width, int pitch,
                     int height, DIFFUSIVITY_TYPE type, float invk) {
#define BW (SCHARR_W + 2)
  __shared__ float buffer[BW * (SCHARR_H + 2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * SCHARR_W + tx;
  int y = blockIdx.y * SCHARR_H + ty;
  int xp = (x == 0 ? 1 : (x > width ? width - 2 : x - 1));
  int yp = (y == 0 ? 1 : (y > height ? height - 2 : y - 1));
  buffer[ty * BW + tx] = imgd[yp * pitch + xp];
  __syncthreads();
  if (x < width && y < height && tx < SCHARR_W && ty < SCHARR_H) {
    float *b = buffer + (ty + 1) * BW + (tx + 1);
    float ul = b[-BW - 1];
    float ur = b[-BW + 1];
    float ll = b[+BW - 1];
    float lr = b[+BW + 1];
    float lx = 3.0f * (lr - ll + ur - ul) + 10.0f * (b[+1] - b[-1]);
    float ly = 3.0f * (lr + ll - ur - ul) + 10.0f * (b[BW] - b[-BW]);
    float dif2 = invk * (lx * lx + ly * ly);
    if (type == PM_G1)
      flowd[y * pitch + x] = exp(-dif2);
    else if (type == PM_G2)
      flowd[y * pitch + x] = 1.0f / (1.0f + dif2);
    else if (type == WEICKERT)
      flowd[y * pitch + x] = 1.0f - exp(-3.315 / (dif2 * dif2 * dif2 * dif2));
    else
      flowd[y * pitch + x] = 1.0f / sqrt(1.0f + dif2);
  }
}

void Flow(CudaImage &img, CudaImage &flow, DIFFUSIVITY_TYPE type,
          float kcontrast, cudaStream_t &stream, const std::string *info)
{
  dim3 blocks(iDivUp(img.width, SCHARR_W), iDivUp(img.height, SCHARR_H));
  dim3 threads(SCHARR_W + 2, SCHARR_H + 2);
  FlowT<<<blocks, threads, 0, stream>>>(img.d_data, flow.d_data, img.width, img.pitch,
                                       img.height, type,
                                       1.0f / (kcontrast * kcontrast));
  checkMsg("Cuda error at kernel FlowT");
}

struct NLDStep_t {
  float *imgd;
  float *flod;
  float *temd;
  int width;
  int pitch;
  int height;
  float stepsize;
};

__global__ void NLDStepT(NLDStep_t s) {
#undef BW
#define BW (NLDSTEP_W + 2)
  __shared__ float ibuff[BW * (NLDSTEP_H + 2)];
  __shared__ float fbuff[BW * (NLDSTEP_H + 2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * NLDSTEP_W + tx;
  int y = blockIdx.y * NLDSTEP_H + ty;
  int xp = (x == 0 ? 0 : (x > s.width ? s.width - 1 : x - 1));
  int yp = (y == 0 ? 0 : (y > s.height ? s.height - 1 : y - 1));
  ibuff[ty * BW + tx] = s.imgd[yp * s.pitch + xp];
  fbuff[ty * BW + tx] = s.flod[yp * s.pitch + xp];
  __syncthreads();
  if (tx < NLDSTEP_W && ty < NLDSTEP_H && x < s.width && y < s.height) {
    float *ib = ibuff + (ty + 1) * BW + (tx + 1);
    float *fb = fbuff + (ty + 1) * BW + (tx + 1);
    float ib0 = ib[0];
    float fb0 = fb[0];
    float xpos = (fb0 + fb[+1]) * (ib[+1] - ib0);
    float xneg = (fb0 + fb[-1]) * (ib0 - ib[-1]);
    float ypos = (fb0 + fb[+BW]) * (ib[+BW] - ib0);
    float yneg = (fb0 + fb[-BW]) * (ib0 - ib[-BW]);
    s.temd[y * s.pitch + x] = s.stepsize * (xpos - xneg + ypos - yneg);
  }
}

struct NLDUpdate_t {
  float *imgd;
  float *temd;
  int width;
  int pitch;
  int height;
};

__global__ void NLDUpdate(NLDUpdate_t s) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x < s.width && y < s.height) {
    int p = y * s.pitch + x;
    s.imgd[p] = s.imgd[p] + s.temd[p];
  }
}

void NLDStep(CudaImage &img, CudaImage &flow, CudaImage &temp,
             float stepsize, cudaStream_t &stream, const std::string *info)
{
  dim3 blocks0(iDivUp(img.width, NLDSTEP_W), iDivUp(img.height, NLDSTEP_H));
  dim3 threads0(NLDSTEP_W + 2, NLDSTEP_H + 2);
  NLDStep_t s;
  s.imgd = img.d_data;
  s.flod = flow.d_data;
  s.temd = temp.d_data;
  s.width = img.width;
  s.pitch = img.pitch;
  s.height = img.height;
  s.stepsize = 0.5 * stepsize;
  NLDStepT<<<blocks0, threads0, 0, stream>>>(s);
  checkMsg("Cuda error at kernel NLDStepT");
  dim3 blocks1(iDivUp(img.width, 32), iDivUp(img.height, 16));
  dim3 threads1(32, 16);
  NLDUpdate_t su;
  su.imgd = img.d_data;
  su.temd = temp.d_data;
  su.width = img.width;
  su.height = img.height;
  su.pitch = img.pitch;
  NLDUpdate<<<blocks1, threads1, 0, stream>>>(su);
  checkMsg("Cuda error at kernel NLDUpdate");
}

__global__ void HalfSampleT(float *iimd, float *oimd, int iwidth, int iheight,
                           int ipitch, int owidth, int oheight, int opitch) {
  __shared__ float buffer[16 * 33];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * 16 + tx;
  int y = blockIdx.y * 16 + ty;
  if (x >= owidth || y >= oheight) return;
  float *ptri = iimd + (2 * y) * ipitch + (2 * x);
  if (2 * owidth == iwidth) {
    buffer[ty * 32 + tx] = owidth * (ptri[0] + ptri[1]);
    ptri += ipitch;
    buffer[ty * 32 + tx + 16] = owidth * (ptri[0] + ptri[1]);
    if (ty == 15) {
      ptri += ipitch;
      buffer[tx + 32 * 16] = owidth * (ptri[0] + ptri[1]);
    } else if (y * 2 + 3 == iheight) {
      ptri += ipitch;
      buffer[tx + 32 * (ty + 1)] = owidth * (ptri[0] + ptri[1]);
    }
  } else {
    float f0 = owidth - x;
    float f2 = 1 + x;
    buffer[ty * 32 + tx] = f0 * ptri[0] + owidth * ptri[1] + f2 * ptri[2];
    ptri += ipitch;
    buffer[ty * 32 + tx + 16] = f0 * ptri[0] + owidth * ptri[1] + f2 * ptri[2];
    if (ty == 15 && 2 * oheight != iheight) {
      ptri += ipitch;
      buffer[tx + 32 * 16] = f0 * ptri[0] + owidth * ptri[1] + f2 * ptri[1];
    } else if (y * 2 + 3 == iheight && 2 * oheight != iheight) {
      ptri += ipitch;
      buffer[tx + 32 * (ty + 1)] =
          f0 * ptri[0] + owidth * ptri[1] + f2 * ptri[2];
    }
  }
  __syncthreads();
  float *buff = buffer + 32 * ty + tx;
  if (2 * oheight == iheight)
    oimd[y * opitch + x] = oheight * (buff[0] + buff[16]) / (iwidth * iheight);
  else {
    float f0 = oheight - y;
    float f2 = 1 + y;
    oimd[y * opitch + x] = (f0 * buff[0] + oheight * buff[16] + f2 * buff[32]) /
                           (iwidth * iheight);
  }
}

__global__ void HalfSample2(float *iimd, float *oimd, int ipitch, int owidth,
                            int oheight, int opitch) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x >= owidth || y >= oheight) return;
  float *ptr = iimd + (2 * y) * ipitch + (2 * x);
  oimd[y * opitch + x] =
      0.25f * (ptr[0] + ptr[1] + ptr[ipitch + 0] + ptr[ipitch + 1]);
}

void HalfSample(CudaImage &inimg, CudaImage &outimg, cudaStream_t &stream, const std::string *info)
{
  if (inimg.width == 2 * outimg.width && inimg.height == 2 * outimg.height) {
    dim3 blocks(iDivUp(outimg.width, 32), iDivUp(outimg.height, 16));
    dim3 threads(32, 16);
    HalfSample2<<<blocks, threads, 0, stream>>>(inimg.d_data, outimg.d_data,
                                                inimg.pitch, outimg.width,
                                                outimg.height, outimg.pitch);
    checkMsg("Cuda error at kernel HalfSample2");
  } else {
    dim3 blocks(iDivUp(outimg.width, 16), iDivUp(outimg.height, 16));
    dim3 threads(16, 16);
    HalfSampleT<<<blocks, threads, 0, stream>>>(inimg.d_data, outimg.d_data, inimg.width,
                                               inimg.height, inimg.pitch, outimg.width,
                                               outimg.height, outimg.pitch);
    checkMsg("Cuda error at kernel HalfSampleT");
  }
}

void Copy(CudaImage &inimg, CudaImage &outimg, cudaStream_t &stream, const std::string *info)
{
  safeCall(cudaMemcpy2DAsync(outimg.d_data, sizeof(float) * outimg.pitch,
                             inimg.d_data, sizeof(float) * outimg.pitch,
                             sizeof(float) * inimg.width, inimg.height,
                             cudaMemcpyDeviceToDevice, stream));
}

Buffer_Ptrs AllocBuffers(int width, int height, int num, int omax, int &maxpts,
                         std::vector<CudaImage> &buffers, cv::KeyPoint *&pts, cv::KeyPoint *&ptsbuffer, int *&ptindices, unsigned char *&desc, float *&descbuffer, CudaImage *&ims, int *comp_idx_1, int *comp_idx_2, cudaStream_t &stream)
{

  maxpts = 4 * ((maxpts+3)/4);

  buffers.resize(omax * num);
  safeCall(cudaHostRegister(buffers.data(), omax * num * sizeof(CudaImage), cudaHostRegisterDefault));
  int w = width;
  int h = height;
  const int alignTo = 128;
  int p = iAlignUp(w * sizeof(float), alignTo);
  Buffer_Ptrs b_ptrs;
  for (int i = 0; i < omax; i++) {
    for (int j = 0; j < num; j++) {
      CudaImage &buf = buffers[i * num + j];
      buf.width = w;
      buf.height = h;
      buf.pitch = p / sizeof(float);

      size_t p2;
      safeCall(cudaMallocPitch((void **)&(buf.d_data), &p2, p, h));
      b_ptrs.img_buffers.emplace_back(buf.d_data);
      if(static_cast<int>(p2) != p){
        int alignTo2 = alignTo;
        int nrTries = 0;
        while (p2 % sizeof(float) && nrTries < 4)
        {
          alignTo2 *= 2;
          safeCall(cudaFree(buf.d_data));
          int p3 = iAlignUp(w * sizeof(float), alignTo2);
          safeCall(cudaMallocPitch((void **)&(buf.d_data), &p2, p, h));
          b_ptrs.img_buffers.back() = buf.d_data;
          nrTries++;
        }
        if (nrTries >= 4 && p2 % sizeof(float)){
          safeCall(cudaFree(buf.d_data));
          int p3 = iAlignUp(w, alignTo);
          int h2 = sizeof(float) * h * (p + 4095) / 4096;
          safeCall(cudaMallocPitch((void **)&(buf.d_data), &p2, 4096, h2));
          b_ptrs.img_buffers.back() = buf.d_data;
          p2 = static_cast<size_t>(p3 * sizeof(float));
        }
        buf.pitch = static_cast<int>(p2) / sizeof(float);
      }
      cudaMemset2DAsync(buf.d_data, p2, 0, p, h, stream);
    }
    w /= 2;
    h /= 2;
    p = iAlignUp(w * sizeof(float), alignTo);
  }

  void *memory = nullptr;
  safeCall(cudaMalloc(&memory, sizeof(cv::KeyPoint) * maxpts));
  safeCall(cudaMemsetAsync(memory, 0, sizeof(cv::KeyPoint) * maxpts, stream));
  b_ptrs.kp_buffers.emplace_back(memory);
  pts = (cv::KeyPoint *)memory;

  memory = nullptr;
  safeCall(cudaMalloc(&memory, sizeof(cv::KeyPoint) * maxpts));
  safeCall(cudaMemsetAsync(memory, 0, sizeof(cv::KeyPoint) * maxpts, stream));
  b_ptrs.kp_buffers.emplace_back(memory);
  ptsbuffer = (cv::KeyPoint *)memory;

  memory = nullptr;
  safeCall(cudaMalloc(&memory, sizeof(unsigned char) * maxpts * 61));
  safeCall(cudaMemsetAsync(memory, 0, sizeof(unsigned char) * maxpts * 61, stream));
  b_ptrs.descr_buffers.emplace_back(memory);
  desc = (unsigned char *)memory;

  memory = nullptr;
  safeCall(cudaMalloc(&memory, sizeof(float) * 3 * 29 * maxpts));
  safeCall(cudaMemsetAsync(memory, 0, sizeof(float) * 3 * 29 * maxpts, stream));
  b_ptrs.descr_buffers.emplace_back(memory);
  descbuffer = (float *)memory;

  safeCall(cudaMalloc((void **)&ptindices, 21 * 21 * sizeof(int) * maxpts));
  safeCall(cudaMemsetAsync(ptindices, 0, 21 * 21 * sizeof(int) * maxpts, stream));
  b_ptrs.indices_buffer = ptindices;

  memory = nullptr;
  safeCall(cudaMalloc(&memory, sizeof(CudaImage) * num * omax));
  safeCall(cudaMemsetAsync(memory, 0, sizeof(CudaImage) * num * omax, stream));
  b_ptrs.struct_buffer = memory;
  ims = (CudaImage *)memory;

  safeCall(cudaStreamSynchronize(stream));

  InitCompareIndices(comp_idx_1, comp_idx_2, stream);

  return b_ptrs;
}

void FreeBuffers(Buffer_Ptrs &buffers)
{
  for(auto &imb : buffers.img_buffers){
    safeCall(cudaFree(imb));
  }
  buffers.img_buffers.clear();
  for (auto &kpb : buffers.kp_buffers)
  {
    safeCall(cudaFree(kpb));
  }
  buffers.kp_buffers.clear();
  for (auto &descrb : buffers.descr_buffers)
  {
    safeCall(cudaFree(descrb));
  }
  buffers.descr_buffers.clear();
  safeCall(cudaFree(buffers.indices_buffer));
  buffers.indices_buffer = nullptr;
  safeCall(cudaFree(buffers.struct_buffer));
  buffers.struct_buffer = nullptr;
}

#define CONTRAST_W 64
#define CONTRAST_H 7
#define HISTCONT_W 64
#define HISTCONT_H 8
#define HISTCONT_R 4

__global__ void MaxContrast(float *imgd, float *cond, int width, int pitch,
                            int height, unsigned int *d_Maxval)
{
#define WID (CONTRAST_W + 2)
  __shared__ float buffer[WID * (CONTRAST_H + 2)];
  __shared__ unsigned int maxval[32];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  if (tx < 32 && !ty) maxval[tx] = 0.0f;
  __syncthreads();
  int x = blockIdx.x * CONTRAST_W + tx;
  int y = blockIdx.y * CONTRAST_H + ty;
  if (x >= width || y >= height) return;
  float *b = buffer + ty * WID + tx;
  b[0] = imgd[y * pitch + x];
  __syncthreads();
  if (tx < CONTRAST_W && ty < CONTRAST_H && x < width - 2 && y < height - 2) {
    float dx = 3.0f * (b[0] - b[2] + b[2 * WID] - b[2 * WID + 2]) +
               10.0f * (b[WID] - b[WID + 2]);
    float dy = 3.0f * (b[0] + b[2] - b[2 * WID] - b[2 * WID + 2]) +
               10.0f * (b[1] - b[2 * WID + 1]);
    float grad = sqrt(dx * dx + dy * dy);
    cond[(y + 1) * pitch + (x + 1)] = grad;
    unsigned int *gradi = (unsigned int *)&grad;
    atomicMax(maxval + (tx & 31), *gradi);
  }
  __syncthreads();
  if (tx < 32 && !ty) atomicMax(d_Maxval, maxval[tx]);
}

__global__ void HistContrast(float *cond, int width, int pitch, int height,
                             float imaxval, int nbins, int *d_Histogram)
{
  __shared__ int hist[512];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int i = ty * HISTCONT_W + tx;
  if (i < nbins) hist[i] = 0;
  __syncthreads();
  int x = blockIdx.x * HISTCONT_W + tx;
  int y = blockIdx.y * HISTCONT_H * HISTCONT_R + ty;
  if (x > 0 && x < width - 1) {
    for (int i = 0; i < HISTCONT_R; i++) {
      if (y > 0 && y < height - 1) {
        int idx = min((int)((float)nbins * cond[y * pitch + x] * imaxval), nbins - 1);
        atomicAdd(hist + idx, 1);
      }
      y += HISTCONT_H;
    }
  }
  __syncthreads();
  if (i < nbins && hist[i] > 0) atomicAdd(d_Histogram + i, hist[i]);
}

void ContrastPercentile(CudaImage &img, CudaImage &temp, CudaImage &blur,
                        float perc, int nbins, float &contrast, cudaStream_t &stream, const std::string *info)
{
  std::string info1, *info_ptr1 = nullptr;
  if (info)
  {
    info1 = *info + "_c";
    info_ptr1 = &info1;
  }
  LowPass(img, blur, temp, 1.0f, 5, stream, info_ptr1);

  float h_Maxval = 0.0f;
  unsigned int *d_Maxval = nullptr;
  safeCall(cudaMalloc((void **)&d_Maxval, sizeof(int)));
  safeCall(cudaMemcpyAsync(d_Maxval, &h_Maxval, sizeof(float), cudaMemcpyHostToDevice, stream));
  dim3 blocks1(iDivUp(img.width, CONTRAST_W), iDivUp(img.height, CONTRAST_H));
  dim3 threads1(CONTRAST_W + 2, CONTRAST_H + 2);
  MaxContrast<<<blocks1, threads1, 0, stream>>>(blur.d_data, temp.d_data, blur.width, blur.pitch, blur.height, d_Maxval);
  checkMsg("Cuda error at kernel MaxContrast");
  safeCall(cudaMemcpyAsync(&h_Maxval, d_Maxval, sizeof(float), cudaMemcpyDeviceToHost, stream));
  safeCall(cudaStreamSynchronize(stream));
  safeCall(cudaFree(d_Maxval));

  if (nbins > 512) {
    printf(
        "Warning: Largest number of possible bins in ContrastPercentile() is "
        "512\n");
    nbins = 512;
  }
  int h_Histogram[512];
  memset(h_Histogram, 0, nbins * sizeof(int));
  int *d_Histogram;
  safeCall(cudaMalloc((void **)&d_Histogram, 512 * sizeof(int)));
  safeCall(cudaMemsetAsync(d_Histogram, 0, 512 * sizeof(int), stream));
  safeCall(
      cudaMemcpyAsync(d_Histogram, h_Histogram, nbins * sizeof(int), cudaMemcpyHostToDevice, stream));
  dim3 blocks2(iDivUp(temp.width, HISTCONT_W),
               iDivUp(temp.height, HISTCONT_H * HISTCONT_R));
  dim3 threads2(HISTCONT_W, HISTCONT_H);
  HistContrast<<<blocks2, threads2, 0, stream>>>(temp.d_data, temp.width, temp.pitch,
                                                 temp.height, 1.0f / h_Maxval, nbins, d_Histogram);
  checkMsg("Cuda error at kernel HistContrast");
  safeCall(
      cudaMemcpyAsync(h_Histogram, d_Histogram, nbins * sizeof(int), cudaMemcpyDeviceToHost, stream));
  safeCall(cudaStreamSynchronize(stream));
  safeCall(cudaFree(d_Histogram));

  int npoints = (temp.width - 2) * (temp.height - 2);
  int nthreshold = (int)(npoints * perc);
  int k = 0, nelements = 0;
  for (k = 0; nelements < nthreshold && k < nbins; k++)
    nelements += h_Histogram[k];
  contrast = (nelements < nthreshold ? 0.03f : h_Maxval * ((float)k / (float)nbins));
}

__global__ void Derivate(float *imd, float *lxd, float *lyd, int width,
                         int pitch, int height, int step, float fac1,
                         float fac2) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x >= width || y >= height) return;
  int xl = (x < step ? step - x : x - step);
  int xh = (x >= width - step ? 2 * width - x - step - 2 : x + step);
  int yl = (y < step ? step - y : y - step);
  int yh = (y >= height - step ? 2 * height - y - step - 2 : y + step);
  float ul = imd[yl * pitch + xl];
  float ur = imd[yl * pitch + xh];
  float ll = imd[yh * pitch + xl];
  float lr = imd[yh * pitch + xh];
  float cl = imd[y * pitch + xl];
  float cr = imd[y * pitch + xh];
  lxd[y * pitch + x] = fac1 * (ur + lr - ul - ll) + fac2 * (cr - cl);
  float uc = imd[yl * pitch + x];
  float lc = imd[yh * pitch + x];
  lyd[y * pitch + x] = fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc);
}

__global__ void HessianDeterminantT(float *lxd, float *lyd, float *detd,
                                   int width, int pitch, int height, int step,
                                   float fac1, float fac2) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x >= width || y >= height) return;
  int xl = (x < step ? step - x : x - step);
  int xh = (x >= width - step ? 2 * width - x - step - 2 : x + step);
  int yl = (y < step ? step - y : y - step);
  int yh = (y >= height - step ? 2 * height - y - step - 2 : y + step);
  float ul = lxd[yl * pitch + xl];
  float ur = lxd[yl * pitch + xh];
  float ll = lxd[yh * pitch + xl];
  float lr = lxd[yh * pitch + xh];
  float cl = lxd[y * pitch + xl];
  float cr = lxd[y * pitch + xh];
  float lxx = fac1 * (ur + lr - ul - ll) + fac2 * (cr - cl);
  float uc = lxd[yl * pitch + x];
  float lc = lxd[yh * pitch + x];
  float lyx = fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc);
  ul = lyd[yl * pitch + xl];
  ur = lyd[yl * pitch + xh];
  ll = lyd[yh * pitch + xl];
  lr = lyd[yh * pitch + xh];
  uc = lyd[yl * pitch + x];
  lc = lyd[yh * pitch + x];
  float lyy = fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc);
  detd[y * pitch + x] = lxx * lyy - lyx * lyx;
}

void HessianDeterminant(CudaImage &img, CudaImage &lx, CudaImage &ly,
                        int step, cudaStream_t &stream, const std::string *info)
{
  float w = 10.0 / 3.0;
  float fac1 = 1.0 / (2.0 * (w + 2.0));
  float fac2 = w * fac1;
  dim3 blocks(iDivUp(img.width, 32), iDivUp(img.height, 16));
  dim3 threads(32, 16);
  Derivate<<<blocks, threads, 0, stream>>>(img.d_data, lx.d_data, ly.d_data, img.width,
                                           img.pitch, img.height, step, fac1, fac2);
  checkMsg("Cuda error at kernel Derivate");
  HessianDeterminantT<<<blocks, threads, 0, stream>>>(lx.d_data, ly.d_data, img.d_data,
                                                     img.width, img.pitch, img.height,
                                                     step, fac1, fac2);
  checkMsg("Cuda error at kernel HessianDeterminantT");
}

struct sortstruct_response_t
{
  int idx;
  float response;
  int x;
  int y;
};

__forceinline__ __device__ bool atomicCompare(const sortstruct_response_t &i,
                                              const sortstruct_response_t &j)
{
  if (i.response < 0 && j.response < 0)
  {
    return false;
  }
  if (i.response < 0 && j.response >= 0)
  {
    return false;
  }
  if (j.response < 0 && i.response >= 0)
  {
    return true;
  }

  if (i.response > j.response)
  {
    return true;
  }
  if (i.response < j.response)
  {
    return false;
  }

  if (j.y < i.y)
  {
    return false;
  }
  if (j.y > i.y)
  {
    return true;
  }
  if (j.x < i.x)
  {
    return false;
  }
  return true;
}

__forceinline__ __device__ void atomicSort(sortstruct_response_t *pts, int shmidx, int offset,
                                           int sortdir)
{
  sortstruct_response_t &p0 = pts[shmidx + sortdir];
  sortstruct_response_t &p1 = pts[shmidx + (offset - sortdir)];

  if (atomicCompare(p0, p1))
  {
    int idxt = p0.idx;
    float respt = p0.response;
    int xt = p0.x;
    int yt = p0.y;
    p0.idx = p1.idx;
    p0.response = p1.response;
    p0.x = p1.x;
    p0.y = p1.y;
    p1.idx = idxt;
    p1.response = respt;
    p1.x = xt;
    p1.y = yt;
  }
}

#define FIND_EXTREMA_USE_INTERNAL_MEM 1
#define FIND_EXTREMA_THREADS_X 32
#define FIND_EXTREMA_THREADS_Y 16
#define FIND_EXTREMA_THREADS (FIND_EXTREMA_THREADS_X * FIND_EXTREMA_THREADS_Y)
#define FIND_EXTREMA_THREADS_MIN (FIND_EXTREMA_THREADS_X < FIND_EXTREMA_THREADS_Y ? FIND_EXTREMA_THREADS_X : FIND_EXTREMA_THREADS_Y)
#define FIND_EXTREMA_MAX_PTS (FIND_EXTREMA_THREADS / 4)
#if FIND_EXTREMA_USE_INTERNAL_MEM
__global__ void FindExtremaT(float *imd, float *imp, float *imn, int maxx,
                             int pitch, int maxy, float border, float dthreshold,
                             int scale, int octave, float size,
                             cv::KeyPoint *pts_blocks, int maxpts, unsigned int *d_PointCounter_blocks)
#else
__global__ void FindExtremaT(float *imd, float *imp, float *imn, int maxx,
                             int pitch, int maxy, float border, float dthreshold,
                             int scale, int octave, float size,
                             cv::KeyPoint *pts_blocks, int maxpts, unsigned int *d_PointCounter_blocks, cv::KeyPoint *pts_blocks_large, sortstruct_response_t *resp_sort_large)
#endif
{
  __shared__ unsigned int d_PointCounter[1];
#if FIND_EXTREMA_USE_INTERNAL_MEM
  __shared__ char pts_all[sizeof(cv::KeyPoint) * FIND_EXTREMA_MAX_PTS];
  __shared__ sortstruct_response_t resp_sort[FIND_EXTREMA_MAX_PTS];
#endif

  int block_idx = blockIdx.y * gridDim.x + blockIdx.x;

  cv::KeyPoint *pts = &(pts_blocks[block_idx * maxpts]);
#if FIND_EXTREMA_USE_INTERNAL_MEM == 0
  cv::KeyPoint *pts_all = &(pts_blocks_large[block_idx * FIND_EXTREMA_MAX_PTS]);
  sortstruct_response_t *resp_sort = &(resp_sort_large[block_idx * FIND_EXTREMA_MAX_PTS]);
#endif
  if (threadIdx.x == 0 && threadIdx.y == 0){
    d_PointCounter[0] = 0;
  }

  if (threadIdx.x < FIND_EXTREMA_THREADS_X / 2 && threadIdx.y < FIND_EXTREMA_THREADS_Y / 2)
  {
    int idx_xy = threadIdx.y * (FIND_EXTREMA_THREADS_X / 2) + threadIdx.x;
    resp_sort[idx_xy].idx = 0;
    resp_sort[idx_xy].response = -1.f;
  }
  __syncthreads();

  int x = blockIdx.x * FIND_EXTREMA_THREADS_X + threadIdx.x;
  int y = blockIdx.y * FIND_EXTREMA_THREADS_Y + threadIdx.y;

  int left_x = (int)(x - border + 0.5f) - 1;
  int right_x = (int)(x + border + 0.5f) + 1;
  int up_y = (int)(y - border + 0.5f) - 1;
  int down_y = (int)(y + border + 0.5f) + 1;
  if (left_x < 0 || right_x >= maxx || up_y < 0 || down_y >= maxy) return;
  int p = y * pitch + x;
  float v = imd[p];
  if (v > dthreshold && v > imd[p - pitch - 1] && v > imd[p + pitch + 1] &&
      v > imd[p + pitch - 1] && v > imd[p - pitch + 1] && v > imd[p - 1] &&
      v > imd[p + 1] && v > imd[p + pitch] && v > imd[p - pitch]) {
    float dx = 0.5f * (imd[p + 1] - imd[p - 1]);
    float dy = 0.5f * (imd[p + pitch] - imd[p - pitch]);
    float dxx = imd[p + 1] + imd[p - 1] - 2.0f * v;
    float dyy = imd[p + pitch] + imd[p - pitch] - 2.0f * v;
    float dxy = 0.25f * (imd[p + pitch + 1] + imd[p - pitch - 1] -
                         imd[p + pitch - 1] - imd[p - pitch + 1]);
    float det = dxx * dyy - dxy * dxy;
    float idet = (det != 0.0f ? 1.0f / det : 0.0f);
    float dst0 = idet * (dxy * dy - dyy * dx);
    float dst1 = idet * (dxy * dx - dxx * dy);
    if (dst0 >= -1.0f && dst0 <= 1.0f && dst1 >= -1.0f && dst1 <= 1.0f) {
      unsigned int idx = atomicAdd(d_PointCounter, 1);
      if (idx < FIND_EXTREMA_MAX_PTS)
      {
#if FIND_EXTREMA_USE_INTERNAL_MEM
        cv::KeyPoint *pts_ptr = (cv::KeyPoint *)pts_all;
        cv::KeyPoint &point = pts_ptr[idx];
#else
        cv::KeyPoint &point = pts_all[idx];
#endif
        resp_sort[idx].idx = idx;
        resp_sort[idx].response = v;
        resp_sort[idx].x = x;
        resp_sort[idx].y = y;
        point.response = v;
        point.size = 2.f * size;
        float octsub = (dst0 < 0 ? -1 : 1) * (octave + fabs(dst0));
        *(float *)(&point.octave) = octsub;
        point.class_id = scale;
        int ratio = (1 << octave);
        point.pt.x = (float)(ratio * x);
        point.pt.y = (float)(ratio * y);
        point.angle = dst1;
      }
      else
      {
        atomicSub(d_PointCounter, 1);
      }
    }
  }
  __syncthreads();

  if (d_PointCounter[0] == 0)
  {
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
      d_PointCounter_blocks[block_idx] = 0;
    }
    return;
  }

  if (d_PointCounter[0] == 1)
  {
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
      {
#if FIND_EXTREMA_USE_INTERNAL_MEM
        const cv::KeyPoint *pts_ptr = (cv::KeyPoint *)pts_all;
        const cv::KeyPoint &to_copy = pts_ptr[0];
#else
        cv::KeyPoint &to_copy = pts_all[0];
#endif
        pts[0].angle = to_copy.angle;
        pts[0].class_id = to_copy.class_id;
        pts[0].octave = to_copy.octave;
        pts[0].pt.y = to_copy.pt.y;
        pts[0].pt.x = to_copy.pt.x;
        pts[0].response = to_copy.response;
        pts[0].size = to_copy.size;
      }
    }
  }
  else
  {
    //Sort keypoints based on response
    const int thread_idx_xy = threadIdx.y * FIND_EXTREMA_THREADS_X + threadIdx.x;
    if (d_PointCounter[0] == 2){
      if (threadIdx.x == 0 && threadIdx.y == 0){
        if (resp_sort[0].response < resp_sort[1].response)
        {
          int idxt = resp_sort[0].idx;
          float respt = resp_sort[0].response;
          int xt = resp_sort[0].x;
          int yt = resp_sort[0].y;
          resp_sort[0].idx = resp_sort[1].idx;
          resp_sort[0].response = resp_sort[1].response;
          resp_sort[0].x = resp_sort[1].x;
          resp_sort[0].y = resp_sort[1].y;
          resp_sort[1].idx = idxt;
          resp_sort[1].response = respt;
          resp_sort[1].x = xt;
          resp_sort[1].y = yt;
        }
      }
    }else{
      for (int i = 1; i < FIND_EXTREMA_MAX_PTS; i <<= 1)
      {
        for (int j = i; j > 0; j >>= 1)
        {
          int mask = 0x0fffffff * j;
          if (thread_idx_xy < FIND_EXTREMA_MAX_PTS / 2)
          {
            int tx = thread_idx_xy;
            int sortdir = (tx & i) > 0 ? 0 : 1;
            int tidx = ((tx & mask) << 1) + (tx & ~mask);
            atomicSort(resp_sort, tidx, j, j * sortdir);
          }
          __syncthreads();
        }
      }
    }
    __syncthreads();

    for (int i = thread_idx_xy; i < min(maxpts, d_PointCounter[0]); i += FIND_EXTREMA_THREADS)
    {
#if FIND_EXTREMA_USE_INTERNAL_MEM
      const cv::KeyPoint *pts_ptr = (cv::KeyPoint *)pts_all;
      const cv::KeyPoint &to_copy = pts_ptr[resp_sort[i].idx];
#else
      cv::KeyPoint &to_copy = pts_all[resp_sort[i].idx];
#endif
      pts[i].angle = to_copy.angle;
      pts[i].class_id = to_copy.class_id;
      pts[i].octave = to_copy.octave;
      pts[i].pt.y = to_copy.pt.y;
      pts[i].pt.x = to_copy.pt.x;
      pts[i].response = to_copy.response;
      pts[i].size = to_copy.size;
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
      d_PointCounter[0] = min(maxpts, d_PointCounter[0]);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    d_PointCounter_blocks[block_idx] = d_PointCounter[0];
  }
}

__global__ void CopyKeypointsFromBlocksSimple(cv::KeyPoint *pts_blocks, cv::KeyPoint *pts, unsigned int *d_PointCounter_blocks, unsigned int blockSize, unsigned int nrBlocks, unsigned int maxpts, unsigned int *d_PointCounter) {
  if (threadIdx.x)
  {
    return;
  }
#define USE_MEMCOPY 0
#if USE_MEMCOPY == 0
  unsigned int idx = d_PointCounter[0];
#else
  char *in_ptr = (char *)pts_blocks;
  char *out_ptr = (char *)(&(pts[d_PointCounter[0]]));
#endif
  for (unsigned int i = 0; i < nrBlocks; i++)
  {
    if (d_PointCounter_blocks[i] > 0){
#if USE_MEMCOPY
      unsigned int block_length = d_PointCounter_blocks[i];
      memcpy(out_ptr, &(in_ptr[i * blockSize * sizeof(cv::KeyPoint)]), sizeof(cv::KeyPoint) * block_length);
      out_ptr += sizeof(cv::KeyPoint) * block_length;
#else
      for (unsigned int j = 0; j < d_PointCounter_blocks[i]; j++)
      {
        cv::KeyPoint &kp = pts_blocks[i * blockSize + j];
        if (idx >= maxpts)
        {
          return;
        }
        pts[idx].angle = kp.angle;
        pts[idx].class_id = kp.class_id;
        pts[idx].octave = kp.octave;
        pts[idx].pt.y = kp.pt.y;
        pts[idx].pt.x = kp.pt.x;
        pts[idx].response = kp.response;
        pts[idx].size = kp.size;
        idx++;
      }
#endif
    }
  }
}

#define COPY_KEYPOINTS_THREADS 512
__global__ void CopyKeypointsFromBlocks(cv::KeyPoint *pts_blocks, cv::KeyPoint *pts, unsigned int *d_PointCounter_blocks, sortstruct_response_t *srt, unsigned int blockSize, unsigned int nrBlocks, unsigned int nrKpCeil, unsigned int maxpts, unsigned int *d_PointCounter)
{
#define ARR_DIVIDE 8
#define ARR_SIZE1 (COPY_KEYPOINTS_THREADS / ARR_DIVIDE)
  __shared__ float mean_resp2[COPY_KEYPOINTS_THREADS], mean_resp1[ARR_SIZE1], mean_resp0[1];
  __shared__ unsigned int pts_idx[1];

  //Get mean response value
  mean_resp2[threadIdx.x] = 0;
  unsigned int cnt = 0;
  for (int i = threadIdx.x; i < (int)nrBlocks; i += COPY_KEYPOINTS_THREADS)
  {
    if (d_PointCounter_blocks[i] > 0)
    {
      for (unsigned int j = 0; j < d_PointCounter_blocks[i]; j++)
      {
        cv::KeyPoint &kp = pts_blocks[(unsigned int)i * blockSize + j];
        mean_resp2[threadIdx.x] += kp.response;
        cnt++;
      }
    }
  }
  if (cnt > 0)
  {
    mean_resp2[threadIdx.x] /= (float)cnt;
  }else{
    mean_resp2[threadIdx.x] = -1.f;
  }
  __syncthreads();
  if (threadIdx.x < ARR_SIZE1)
  {
    mean_resp1[threadIdx.x] = 0;
    cnt = 0;
    for (int j = 0; j < ARR_DIVIDE; j++)
    {
      int idx = threadIdx.x * ARR_DIVIDE + j;
      if (mean_resp2[idx] > 0)
      {
        mean_resp1[threadIdx.x] += mean_resp2[idx];
        cnt++;
      }
    }
    if (cnt > 0)
    {
      mean_resp1[threadIdx.x] /= (float)cnt;
    }
    else
    {
      mean_resp1[threadIdx.x] = -1.f;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    pts_idx[0] = 0;
    mean_resp0[0] = 0;
    cnt = 0;
    for (int i = 0; i < ARR_SIZE1; i++)
    {
      if (mean_resp1[i] > 0)
      {
        mean_resp0[0] += mean_resp1[i];
        cnt++;
      }
    }
    if (cnt > 0)
    {
      mean_resp0[0] /= (float)cnt;
      mean_resp0[0] *= (float)blockSize;
    }else{
      mean_resp0[0] = 1.f;
    }
  }
  __syncthreads();

  //Clear sort struct
  for (unsigned int i = (unsigned int)threadIdx.x; i < nrKpCeil; i += COPY_KEYPOINTS_THREADS)
  {
      srt[i].idx = 0;
      srt[i].response = -1.f;
      srt[i].x = 0;
      srt[i].y = 0;
  }
  __syncthreads();

  //Lower response values of blocks having too many keypoints with high response values
  //and create sort structures using these response values
  for (int i = threadIdx.x; i < (int)nrBlocks; i += COPY_KEYPOINTS_THREADS)
  {
    float multiplier = 1.f;
    const unsigned int idx_start = (unsigned int)i * blockSize;
    if (d_PointCounter_blocks[i] == blockSize)
    {
      float mean_resp_block = 0;
      for (unsigned int j = 0; j < blockSize; j++)
      {
        cv::KeyPoint &kp = pts_blocks[idx_start + j];
        mean_resp_block += kp.response;
      }
      if (mean_resp_block > mean_resp0[0])
      {
        multiplier = max(mean_resp0[0] / mean_resp_block, 0.33f);
      }
    }
    if (d_PointCounter_blocks[i] > 0)
    {
      for (unsigned int j = 0; j < d_PointCounter_blocks[i]; j++)
      {
        const unsigned int idx_b = idx_start + j;
        cv::KeyPoint &kp = pts_blocks[idx_b];
        const unsigned int idx_g = atomicAdd(pts_idx, 1);
        srt[idx_g].idx = idx_b;
        srt[idx_g].response = multiplier * kp.response;
        srt[idx_g].x = (int)(kp.pt.x + 0.5f);
        srt[idx_g].y = (int)(kp.pt.y + 0.5f);
      }
    }
  }
  __syncthreads();

  //Sort with high response values first
  if (pts_idx[0] > 2){
    for (int i = 1; i < (int)nrKpCeil; i <<= 1)
    {
      for (int j = i; j > 0; j >>= 1)
      {
        int mask = 0x0fffffff * j;
        for (int tx = threadIdx.x; tx < (int)nrKpCeil / 2; tx += COPY_KEYPOINTS_THREADS)
        {
          int sortdir = (tx & i) > 0 ? 0 : 1;
          int tidx = ((tx & mask) << 1) + (tx & ~mask);
          atomicSort(srt, tidx, j, j * sortdir);
        }
        __syncthreads();
      }
    }
  }
  else if (pts_idx[0] == 2)
  {
    if (threadIdx.x == 0){
      atomicSort(srt, 0, 1, 1);
    }
  }
  __syncthreads();

  //Copy keypoints with highest responses to final keypoint buffer
  const int maxpts2 = (int)maxpts - (int)d_PointCounter[0];
  for (int i = threadIdx.x; i < (int)maxpts2 && i < (int)pts_idx[0]; i += COPY_KEYPOINTS_THREADS)
  {
    const cv::KeyPoint &to_copy = pts_blocks[srt[i].idx];
    const int idx_g = i + (int)d_PointCounter[0];
    pts[idx_g].angle = to_copy.angle;
    pts[idx_g].class_id = to_copy.class_id;
    pts[idx_g].octave = to_copy.octave;
    pts[idx_g].pt.y = to_copy.pt.y;
    pts[idx_g].pt.x = to_copy.pt.x;
    pts[idx_g].response = to_copy.response;
    pts[idx_g].size = to_copy.size;
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    d_PointCounter[0] = min(maxpts, pts_idx[0] + d_PointCounter[0]);
  }
}

__global__ void CopyIdxArray(int scale, unsigned int *d_PointCounter, unsigned int *d_ExtremaIdx)
{
  d_ExtremaIdx[scale] = d_PointCounter[0];
}

__global__ void SumArray(unsigned int *d_array, unsigned int *d_sum, unsigned int size, bool addToInput)
{
  if (threadIdx.x)
  {
    return;
  }
  unsigned int sum = 0;
  for (unsigned int i = 0; i < size; i++)
  {
    sum += d_array[i];
  }
  if (addToInput){
    d_sum[0] += sum;
  }
  else
  {
    d_sum[0] = sum;
  }
}

bool FindExtrema(CudaImage &img, CudaImage &imgp, CudaImage &imgn,
                 float border, float dthreshold, int scale, int octave,
                 float size, cv::KeyPoint *pts, int maxpts, unsigned int *d_PointCounter, unsigned int *d_ExtremaIdx, cudaStream_t &stream, const std::string *info)
{
  unsigned int totPts = 0;
  safeCall(cudaMemcpyAsync(&totPts, d_PointCounter, sizeof(int), cudaMemcpyDeviceToHost, stream));
  safeCall(cudaStreamSynchronize(stream));
  if (static_cast<int>(totPts) >= maxpts)
  {
    CopyIdxArray<<<1, 1, 0, stream>>>(scale, d_PointCounter, d_ExtremaIdx);
    return false;
  }

  int nr_x_blocks = iDivUp(img.width, FIND_EXTREMA_THREADS_X);
  int nr_y_blocks = iDivUp(img.height, FIND_EXTREMA_THREADS_Y);
  int nr_blocks = nr_y_blocks * nr_x_blocks;

  int pts_remaining = maxpts - static_cast<int>(totPts);
  int maxpts_blocks_up = pts_remaining / nr_blocks + 3;

  unsigned int *d_PointCounter_blocks = nullptr;
  safeCall(cudaMalloc((void **)&d_PointCounter_blocks, nr_blocks * sizeof(unsigned int)));
  safeCall(cudaMemsetAsync(d_PointCounter_blocks, 0, nr_blocks * sizeof(unsigned int), stream));

  cv::KeyPoint *pts_blocks = nullptr;
  safeCall(cudaMalloc((void **)&pts_blocks, nr_blocks * maxpts_blocks_up * sizeof(cv::KeyPoint)));
  safeCall(cudaMemsetAsync(pts_blocks, 0, nr_blocks * maxpts_blocks_up * sizeof(cv::KeyPoint), stream));
  safeCall(cudaStreamSynchronize(stream));

#if FIND_EXTREMA_USE_INTERNAL_MEM == 0
  cv::KeyPoint *pts_blocks_large = nullptr;
  safeCall(cudaMalloc((void **)&pts_blocks_large, nr_blocks * FIND_EXTREMA_MAX_PTS * sizeof(cv::KeyPoint)));
  safeCall(cudaMemsetAsync(pts_blocks_large, 0, nr_blocks * FIND_EXTREMA_MAX_PTS * sizeof(cv::KeyPoint), stream));
  safeCall(cudaStreamSynchronize(stream));

  sortstruct_response_t *resp_sort = nullptr;
  safeCall(cudaMalloc((void **)&resp_sort, nr_blocks * FIND_EXTREMA_MAX_PTS * sizeof(sortstruct_response_t)));
  safeCall(cudaMemsetAsync(resp_sort, 0, nr_blocks * FIND_EXTREMA_MAX_PTS * sizeof(sortstruct_response_t), stream));
  safeCall(cudaStreamSynchronize(stream));
#endif

  dim3 blocks(nr_x_blocks, nr_y_blocks);
  dim3 threads(FIND_EXTREMA_THREADS_X, FIND_EXTREMA_THREADS_Y);
  float b = border;
#if FIND_EXTREMA_USE_INTERNAL_MEM
  FindExtremaT<<<blocks, threads, 0, stream>>>(img.d_data, imgp.d_data, imgn.d_data, img.width, img.pitch, img.height,
                                               b, dthreshold, scale, octave, size, pts_blocks, maxpts_blocks_up, d_PointCounter_blocks);
#else
  FindExtremaT<<<blocks, threads, 0, stream>>>(img.d_data, imgp.d_data, imgn.d_data, img.width, img.pitch, img.height,
                                               b, dthreshold, scale, octave, size, pts_blocks, maxpts_blocks_up, d_PointCounter_blocks, pts_blocks_large, resp_sort);
#endif
  checkMsg("Cuda error at kernel FindExtremaT");
  safeCall(cudaStreamSynchronize(stream));

#if FIND_EXTREMA_USE_INTERNAL_MEM == 0
  safeCall(cudaFree(pts_blocks_large));
  safeCall(cudaFree(resp_sort));
#endif

  unsigned int *d_PointCounter_internal = nullptr;
  safeCall(cudaMalloc((void **)&d_PointCounter_internal, sizeof(unsigned int)));
  safeCall(cudaMemsetAsync(d_PointCounter_internal, 0, sizeof(unsigned int), stream));
  SumArray<<<1, 1, 0, stream>>>(d_PointCounter_blocks, d_PointCounter_internal, static_cast<unsigned int>(nr_blocks), false);
  checkMsg("Cuda error at kernel SumArray");
  safeCall(cudaMemcpyAsync(&totPts, d_PointCounter_internal, sizeof(int), cudaMemcpyDeviceToHost, stream));
  safeCall(cudaStreamSynchronize(stream));
  if(totPts == 0){
    CopyIdxArray<<<1, 1, 0, stream>>>(scale, d_PointCounter, d_ExtremaIdx);
    checkMsg("Cuda error at kernel CopyIdxArray");
    safeCall(cudaFree(d_PointCounter_blocks));
    safeCall(cudaFree(pts_blocks));
    safeCall(cudaFree(d_PointCounter_internal));
    return true;
  }

  int maxpts_blocks_diff = static_cast<int>(totPts) - pts_remaining;
  if (maxpts_blocks_diff <= 0)
  {
    CopyKeypointsFromBlocksSimple<<<1, 1, 0, stream>>>(pts_blocks, pts, d_PointCounter_blocks, static_cast<unsigned int>(maxpts_blocks_up), static_cast<unsigned int>(nr_blocks), static_cast<unsigned int>(maxpts), d_PointCounter);
    checkMsg("Cuda error at kernel CopyKeypointsFromBlocksSimple");
    SumArray<<<1, 1, 0, stream>>>(d_PointCounter_blocks, d_PointCounter, static_cast<unsigned int>(nr_blocks), true);
    checkMsg("Cuda error at kernel SumArray");
  }
  else
  {
    unsigned int nrKpCeil = 1;
    while (nrKpCeil < totPts)
    {
      nrKpCeil *= 2;
    }
    sortstruct_response_t *srt = nullptr;
    safeCall(cudaMalloc((void **)&srt, nrKpCeil * sizeof(sortstruct_response_t)));
    dim3 threads_copy(COPY_KEYPOINTS_THREADS, 1, 1);
    CopyKeypointsFromBlocks<<<1, threads_copy, 0, stream>>>(pts_blocks, pts, d_PointCounter_blocks, srt, static_cast<unsigned int>(maxpts_blocks_up), static_cast<unsigned int>(nr_blocks), nrKpCeil, static_cast<unsigned int>(maxpts), d_PointCounter);
    checkMsg("Cuda error at kernel CopyKeypointsFromBlocks");
    safeCall(cudaFree(srt));
  }

  CopyIdxArray<<<1, 1, 0, stream>>>(scale, d_PointCounter, d_ExtremaIdx);
  checkMsg("Cuda error at kernel CopyIdxArray");

  safeCall(cudaFree(d_PointCounter_blocks));
  safeCall(cudaFree(pts_blocks));
  safeCall(cudaFree(d_PointCounter_internal));

  return true;
}

void ClearPoints(unsigned int *d_PointCounter, cudaStream_t &stream)
{
  safeCall(cudaMemsetAsync(d_PointCounter, 0, sizeof(int), stream));
}

__forceinline__ __device__ void atomicSort(int *pts, int shmidx, int offset,
                                           int sortdir) {
  int &p0 = pts[shmidx + sortdir];
  int &p1 = pts[shmidx + (offset - sortdir)];

  if (p0 < p1) {
    int t = p0;
    p0 = p1;
    p1 = t;
  }
}

__forceinline__ __device__ bool atomicCompare(const cv::KeyPoint &i,
                                              const cv::KeyPoint &j) {
  float t = i.pt.x * j.pt.x;
  if (t == 0) {
    if (j.pt.x != 0) {
      return false;
    } else {
      return true;
    }
  }

  if (i.pt.y < j.pt.y) return true;
  if (i.pt.y == j.pt.y && i.pt.x < j.pt.x) return true;

  return false;
}

struct sortstruct_t {
    int idx;
    short x;
    short y;
};

__forceinline__ __device__ bool atomicCompare(const sortstruct_t &i,
                                              const sortstruct_t &j)
{
  long int t = (long int)i.x * (long int)j.x;
  if (t == 0)
  {
    if(i.idx < 0 && j.idx >= 0){
      return true;
    }
    else if (i.idx < 0 || j.idx < 0)
    {
      return false;
    }
    if (j.x != 0)
    {
      return false;
    }
    else
    {
      if (i.y >= j.y){
        return false;
      }
      return true;
    }
  }

  if (i.y < j.y)
    return true;

  if (i.y == j.y && i.x < j.x)
    return true;

  return false;
}

__forceinline__ __device__ void atomicSort(sortstruct_t *pts, int shmidx,
                                           int offset, int sortdir) {
    sortstruct_t &p0 = pts[(shmidx + sortdir)];
    sortstruct_t &p1 = pts[(shmidx + (offset - sortdir))];

  if (atomicCompare(p0, p1)) {
      int idx = p0.idx;
      short ptx = p0.x;
      short pty = p0.y;
      p0.idx = p1.idx;
      p0.x = p1.x;
      p0.y = p1.y;
      p1.idx = idx;
      p1.x = ptx;
      p1.y = pty;
  }
}

#define USE_SORT_SHARED_MEM 0
#define BitonicSortThreads 1024
#if USE_SORT_SHARED_MEM
__global__ void bitonicSort_global(const cv::KeyPoint *pts, cv::KeyPoint *newpts, unsigned int *d_ExtremaIdx, int nrThreads)
#else
__global__ void bitonicSort_global(const cv::KeyPoint *pts, cv::KeyPoint *newpts, sortstruct_t *_shm, int _sz, unsigned int *d_ExtremaIdx, int nrThreads)
#endif
{
  __shared__ int first[1], last[1], nkpts[1], nkpts_ceil[1], add_i[1];
  if (threadIdx.x == 0){
    first[0] = (blockIdx.x == 0) ? 0 : (int)d_ExtremaIdx[blockIdx.x - 1];
    last[0] = (int)d_ExtremaIdx[blockIdx.x];
    nkpts[0] = last[0] - first[0];
    nkpts_ceil[0] = 1;
    while (nkpts_ceil[0] < nkpts[0])
    {
      nkpts_ceil[0] *= 2;
    }
    add_i[0] = nkpts_ceil[0] - nkpts[0];
  }
  __syncthreads();

  sortstruct_t *shm = &(_shm[_sz * blockIdx.x]);
  if (nkpts[0] > 2)
  {
    const cv::KeyPoint *tmpg = &pts[first[0]];
    for (int i = threadIdx.x; i < nkpts_ceil[0]; i += nrThreads)
    {
      if (i < nkpts[0])
      {
        shm[i].idx = i;
        shm[i].y = (short)(tmpg[i].pt.y);
        shm[i].x = (short)(tmpg[i].pt.x);
      }
      else
      {
        shm[i].idx = -1;
        shm[i].y = 0;
        shm[i].x = 0;
      }
    }
    __syncthreads();

    for (int i = 1; i < nkpts_ceil[0]; i <<= 1)
    {
      for (int j = i; j > 0; j >>= 1)
      {
        int mask = 0x0fffffff * j;
        for (int tx = threadIdx.x; tx < nkpts_ceil[0] / 2; tx += nrThreads)
        {
          int sortdir = (tx & i) > 0 ? 0 : 1;
          int tidx = ((tx & mask) << 1) + (tx & ~mask);
          atomicSort(shm, tidx, j, j * sortdir);
        }
        __syncthreads();
      }
    }
    __syncthreads();

    cv::KeyPoint *tmpnewg = &newpts[first[0]];
    for (int i = threadIdx.x; i < nkpts[0]; i += nrThreads)
    {
      const cv::KeyPoint &to_copy = tmpg[shm[add_i[0] + i].idx];
      tmpnewg[i].angle = to_copy.angle;
      tmpnewg[i].class_id = to_copy.class_id;
      tmpnewg[i].octave = to_copy.octave;
      tmpnewg[i].pt.y = to_copy.pt.y;
      tmpnewg[i].pt.x = to_copy.pt.x;
      tmpnewg[i].response = to_copy.response;
      tmpnewg[i].size = to_copy.size;
    }
  }
  else if (nkpts[0] == 2)
  {
    if (threadIdx.x == 0){
      const cv::KeyPoint *tmpg = &pts[first[0]];
      cv::KeyPoint *tmpnewg = &newpts[first[0]];
      shm[0].idx = 0;
      shm[0].y = (short)tmpg[0].pt.y;
      shm[0].x = (short)tmpg[0].pt.x;
      shm[1].idx = 1;
      shm[1].y = (short)tmpg[1].pt.y;
      shm[1].x = (short)tmpg[1].pt.x;
      if (atomicCompare(shm[0], shm[1]))
      {
        tmpnewg[0].angle = tmpg[0].angle;
        tmpnewg[0].class_id = tmpg[0].class_id;
        tmpnewg[0].octave = tmpg[0].octave;
        tmpnewg[0].pt.y = tmpg[0].pt.y;
        tmpnewg[0].pt.x = tmpg[0].pt.x;
        tmpnewg[0].response = tmpg[0].response;
        tmpnewg[0].size = tmpg[0].size;

        tmpnewg[1].angle = tmpg[1].angle;
        tmpnewg[1].class_id = tmpg[1].class_id;
        tmpnewg[1].octave = tmpg[1].octave;
        tmpnewg[1].pt.y = tmpg[1].pt.y;
        tmpnewg[1].pt.x = tmpg[1].pt.x;
        tmpnewg[1].response = tmpg[1].response;
        tmpnewg[1].size = tmpg[1].size;
      }else{
        tmpnewg[1].angle = tmpg[0].angle;
        tmpnewg[1].class_id = tmpg[0].class_id;
        tmpnewg[1].octave = tmpg[0].octave;
        tmpnewg[1].pt.y = tmpg[0].pt.y;
        tmpnewg[1].pt.x = tmpg[0].pt.x;
        tmpnewg[1].response = tmpg[0].response;
        tmpnewg[1].size = tmpg[0].size;

        tmpnewg[0].angle = tmpg[1].angle;
        tmpnewg[0].class_id = tmpg[1].class_id;
        tmpnewg[0].octave = tmpg[1].octave;
        tmpnewg[0].pt.y = tmpg[1].pt.y;
        tmpnewg[0].pt.x = tmpg[1].pt.x;
        tmpnewg[0].response = tmpg[1].response;
        tmpnewg[0].size = tmpg[1].size;
      }
    }
  }
  else if (nkpts[0] == 1)
  {
    if (threadIdx.x == 0){
      const cv::KeyPoint *tmpg = &pts[first[0]];
      cv::KeyPoint *tmpnewg = &newpts[first[0]];
      tmpnewg[0].angle = tmpg[0].angle;
      tmpnewg[0].class_id = tmpg[0].class_id;
      tmpnewg[0].octave = tmpg[0].octave;
      tmpnewg[0].pt.y = tmpg[0].pt.y;
      tmpnewg[0].pt.x = tmpg[0].pt.x;
      tmpnewg[0].response = tmpg[0].response;
      tmpnewg[0].size = tmpg[0].size;
    }
  }
}

#define FindNeighborsThreads 32
__global__ void FindNeighbors(cv::KeyPoint *pts, int *kptindices, int width, unsigned int *d_ExtremaIdx)
{
  __shared__ int gidx[1];

  // which scale?
  int scale = pts[blockIdx.x].class_id;

  int cmpIdx = scale < 1 ? 0 : (int)d_ExtremaIdx[scale - 1];

  const float size = pts[blockIdx.x].size;
  const float sizeth = 0.5f * size;
  const float size2th = sizeth * sizeth;

  if (threadIdx.x == 0){
    gidx[0] = 1;
  }

  __syncthreads();

  // One keypoint per block.
  const cv::KeyPoint &kpt = pts[blockIdx.x];

  if (threadIdx.x == 0 && (size < 0 || size > 21.f || kpt.pt.y < 0 || kpt.pt.x < 0 || kpt.response < 0))
  {
    gidx[0] = -1;
    kptindices[blockIdx.x * width] = 1;
  }
  __syncthreads();
  if (gidx[0] == -1){
    return;
  }

  // Key point to compare. Only compare with smaller than current
  // Iterate backwards instead and break as soon as possible!
  for (int i = blockIdx.x - threadIdx.x - 1; i >= cmpIdx; i -= FindNeighborsThreads)
  {

    const cv::KeyPoint &kpt_cmp = pts[i];

    if (kpt.pt.y - kpt_cmp.pt.y > sizeth)
      break;

    const float dx = kpt.pt.x - kpt_cmp.pt.x;
    const float dy = kpt.pt.y - kpt_cmp.pt.y;
    const float dist = dx * dx + dy * dy;

    if (dist < size2th)
    {
      int idx = atomicAdd(gidx, 1);
      kptindices[blockIdx.x * width + idx] = i;
    }
  }

  if (scale > 0)
  {
    int startidx = (int)d_ExtremaIdx[scale - 1];
    cmpIdx = scale < 2 ? 0 : (int)d_ExtremaIdx[scale - 2];
    for (int i = startidx - threadIdx.x - 1; i >= cmpIdx; i -= FindNeighborsThreads)
    {
      const cv::KeyPoint &kpt_cmp = pts[i];

      if (kpt_cmp.pt.y - kpt.pt.y > sizeth)
        continue;

      if (kpt.pt.y - kpt_cmp.pt.y > sizeth)
        break;

      const float dx = kpt.pt.x - kpt_cmp.pt.x;
      const float dy = kpt.pt.y - kpt_cmp.pt.y;
      const float dist = dx * dx + dy * dy;

      if (dist < size2th)
      {
        int idx = atomicAdd(gidx, 1);
        kptindices[blockIdx.x * width + idx] = i;
      }
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    kptindices[blockIdx.x * width] = gidx[0];
  }
}

// TODO Intermediate storage of memberarray and minneighbor
#define FilterExtremaThreads 1024
__global__ void FilterExtrema_kernel(cv::KeyPoint *kpts, cv::KeyPoint *newkpts,
                                     int *kptindices, int width,
                                     int *memberarray,
                                     int *minneighbor,
                                     unsigned char *shouldAdd,
                                     unsigned int *d_PointCounter)
{
  // -1  means not processed
  // -2  means added but replaced
  // >=0 means added
  __shared__ bool shouldBreak[1];

  int nump = (int)d_PointCounter[0];

  // Initially all points are unprocessed
  for (int i = threadIdx.x; i < nump; i += FilterExtremaThreads)
  {
    memberarray[i] = -1;
  }

  if (threadIdx.x == 0) {
    shouldBreak[0] = true;
  }

  __syncthreads();

  // Loop until there are no more points to process
  for (int xx=0; xx<10000; ++xx) {
    // Mark all points for addition and no minimum neighbor
    for (int i = threadIdx.x; i < nump; i += FilterExtremaThreads)
    {
      minneighbor[i] = nump + 1;
      shouldAdd[i] = 255;
    }
    __syncthreads();

    // Look through all points. If there are points that have not been processed,
    // disable breaking and check if it has no processed neighbors (add), has all processed
    // neighbors (compare with neighbors) or has some unprocessed neighbor (wait)
    for (int i = threadIdx.x; i < nump; i += FilterExtremaThreads) {
      int neighborsSize = kptindices[i * width] - 1;
      int *neighbors = &(kptindices[i * width + 1]);

      // Only do if we didn't process the point before
      if (memberarray[i] == -1) {
        // If we process at least one point we shouldn't break
        // No need to sync. Only want to know if at least one thread wants to
        // continue
        shouldBreak[0] = false;
        // Sort neighbors according to the order of currently added points
        // (often very few)
        // If the neighbor has been replaced, stick it to the back
        // If any neighbor has not been processed, break;
        bool shouldProcess = true;
        for (int k = 0; k < neighborsSize; ++k) {
          // If the point has one or more unprocessed neighbors, skip
          if (memberarray[neighbors[k]] == -1) {
            shouldProcess = false;
            shouldAdd[i] = 0;
            break;
          }
          // If it has a neighbor that is in the list, we don't add, but process
          if (memberarray[neighbors[k]] >= 0) {
            shouldAdd[i] = 0;
          }
        }

        // We should process and potentially replace the neighbor
        if (shouldProcess && (shouldAdd[i] < 127 || shouldAdd[i] == 128) && (neighborsSize > 0))
        {
          // Find the smallest neighbor. Often only one or two, so no ned for fancy algorithm
          for (int k = 0; k < neighborsSize; ++k) {
            for (int j = k + 1; j < neighborsSize; ++j) {
              if (memberarray[neighbors[k]] == -2 ||
                  (memberarray[neighbors[j]] != -2 &&
                   memberarray[neighbors[j]] < memberarray[neighbors[k]])) {
                int t = neighbors[k];
                neighbors[k] = neighbors[j];
                neighbors[j] = t;
              }
            }
          }
          // Pick the first neighbor
          // We need to make sure, in case more than one point has this
          // neighbor,
          // That the point with lowest memberarrayindex processes it first
          // Here minneighbor[i] is the target and i the neighbor
          const int nidx = neighbors[0];
          atomicMin(&(minneighbor[nidx]), i);
        }
      }
    }
    __syncthreads();

    // Check which points we can add
    for (size_t i = threadIdx.x; i < nump; i += FilterExtremaThreads) {
      if (memberarray[i] == -1) {
        if (shouldAdd[i] > 128 || shouldAdd[i] == 127)//if true
        {
          memberarray[i] = i;
        }
      }
    }
    __syncthreads();

    // Look at the neighbors. If the response is higher, replace
    for (size_t i = threadIdx.x; i < nump; i += FilterExtremaThreads) {
      if (minneighbor[i] < nump) {
        if (memberarray[minneighbor[i]] == -1) {
          if (shouldAdd[minneighbor[i]] < 127 || shouldAdd[minneighbor[i]] == 128) // if false
          {
            const cv::KeyPoint &p0 = kpts[minneighbor[i]];
            const cv::KeyPoint &p1 = kpts[i];
            if (p0.response > p1.response) {
              int val_old = atomicExch(&(memberarray[minneighbor[i]]), i);
              if (val_old == -1){
                atomicExch(&(memberarray[i]), -2);
              }else{
                atomicExch(&(memberarray[minneighbor[i]]), val_old);
              }
            } else {
              int val_old = atomicExch(&(memberarray[minneighbor[i]]), -2);
              if (val_old != -1)
              {
                atomicExch(&(memberarray[minneighbor[i]]), val_old);
              }
            }
          }
        }
      }
    }
    __syncthreads();

    // Are we done?
    if (shouldBreak[0]) break;

    __syncthreads();

    if (threadIdx.x == 0) {
      shouldBreak[0] = true;
    }
    __syncthreads();
  }

  __syncthreads();
}

__global__ void sortFiltered_kernel(cv::KeyPoint *kpts, cv::KeyPoint *newkpts,
                                    int *memberarray, unsigned int *d_PointCounter)
{

  __shared__ int minneighbor[2048];
  __shared__ unsigned int curridx[1];

  int nump = (int)d_PointCounter[0];

  if (threadIdx.x == 0) {
    curridx[0] = 0;
  }
  __syncthreads();
  minneighbor[threadIdx.x] = nump + 1;
  minneighbor[threadIdx.x + 1024] = nump + 1;

  // Sort array
  const int upper = (nump + 2047) & (0xfffff800);

  for (int i = threadIdx.x; i < upper; i += 2048) {

    minneighbor[threadIdx.x] =
        i >= nump ? nump+1 : (memberarray[i] < 0 ? nump+1 : (kpts[memberarray[i]].size < 0 ? nump+1 : memberarray[i]));
    minneighbor[threadIdx.x + 1024] =
        i + 1024 >= nump ? nump+1
                         : (memberarray[i + 1024] < 0 ? nump+1 : (kpts[memberarray[i+1024]].size < 0 ? nump+1 : memberarray[i+1024]));

    __syncthreads();

#if 0
    for (int i1 = 1; i1 < 2048; i1 <<= 1)
    {
      for (int j = i1; j > 0; j >>= 1)
      {
        int mask = 0x0fffffff * j;
        for (int tx = threadIdx.x; tx < 1024; tx += 1024)
        {
          int sortdir = (tx & i1) > 0 ? 0 : 1;
          int tidx = ((tx & mask) << 1) + (tx & ~mask);
          atomicSort(minneighbor, tidx, j, j * sortdir);
        }
        __syncthreads();
      }
    }
#else
    // Sort and store keypoints
    #pragma unroll 1
        for (int k = 1; k < 2048; k <<= 1) {
          int sortdir = (threadIdx.x & k) > 0 ? 0 : 1;

    #pragma unroll 1
          for (int j = k; j > 0; j >>= 1) {
            int mask = 0x0fffffff * j;
            int tidx = ((threadIdx.x & mask) << 1) + (threadIdx.x & ~mask);
            atomicSort(minneighbor, tidx, j, j * sortdir);
            __syncthreads();
          }
        }
#endif

    __syncthreads();

#pragma unroll 1
    for (int k = threadIdx.x; k < 2048; k += 1024)
    {
      if (minneighbor[k] < nump)
      {
        // Restore subpixel component
        cv::KeyPoint &okpt = kpts[minneighbor[k]];
        float octsub = fabs(*(float *)(&kpts[minneighbor[k]].octave));
        int octave = (int)octsub;
        float subp = (*(float *)(&kpts[minneighbor[k]].octave) < 0 ? -1 : 1) * (octsub - (float)octave);
        float ratio = 1 << octave;
        cv::KeyPoint &tkpt = newkpts[(unsigned int)k + curridx[0]];
        tkpt.pt.y = ratio * ((int)(0.5f + okpt.pt.y / ratio) + okpt.angle);
        tkpt.pt.x = ratio * ((int)(0.5f + okpt.pt.x / ratio) + subp);
        // newkpts[(unsigned int)k + curridx[0] + (unsigned int)threadIdx.x].angle = 0; // This will be set elsewhere
        tkpt.class_id = okpt.class_id;
        tkpt.octave = octave;
        tkpt.response = okpt.response;
        tkpt.size = okpt.size;
      }
    }
    __syncthreads();


    // How many did we add?
    if (minneighbor[2047] < nump) {
      if (threadIdx.x == 0){
        curridx[0] += 2048U;
      }
    } else {
      if (minneighbor[1024] < nump) {
        if (threadIdx.x < 1023 && minneighbor[1024 + threadIdx.x] < nump &&
            minneighbor[1024 + threadIdx.x + 1] == nump+1) {
          curridx[0] += 1024U + (unsigned int)threadIdx.x + 1U;
        }
      } else {
        if (minneighbor[threadIdx.x] < nump &&
            minneighbor[threadIdx.x + 1] == nump+1) {
          curridx[0] += (unsigned int)threadIdx.x + 1U;
        }
      }
      __syncthreads();
    }
    __syncthreads();
    minneighbor[threadIdx.x] = nump + 1;
    minneighbor[threadIdx.x + 1024] = nump + 1;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    d_PointCounter[0] = curridx[0];
  }
}

void FilterExtrema(cv::KeyPoint *pts, cv::KeyPoint *newpts, int *kptindices, int &nump, unsigned int *d_PointCounter, unsigned int *d_ExtremaIdx, int nrLevelImgs, cudaStream_t &stream, const std::string *info)
{
  if(nrLevelImgs < 2){
    return;
  }
  safeCall(cudaMemcpyAsync(&nump, d_PointCounter, sizeof(int), cudaMemcpyDeviceToHost, stream));
  safeCall(cudaStreamSynchronize(stream));
  if(!nump){
    return;
  }

  unsigned int *extremaidx_h = new unsigned int[nrLevelImgs];
  safeCall(cudaMemcpyAsync(extremaidx_h, d_ExtremaIdx, nrLevelImgs * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  safeCall(cudaStreamSynchronize(stream));
  int maxnump = static_cast<int>(extremaidx_h[0]);
  for (int i = 1; i < nrLevelImgs; ++i)
  {
    maxnump = max(maxnump, static_cast<int>(extremaidx_h[i]) - static_cast<int>(extremaidx_h[i - 1]));
  }

  int width = 21 * 21;

  // Sort the list of points
  int nump_ceil = 1;
  while (nump_ceil < maxnump)
  {
    nump_ceil <<= 1;
  }
  int nrSortThreads = min(BitonicSortThreads, nump_ceil);
  dim3 blocks(nrLevelImgs, 1, 1);
  dim3 threads(nrSortThreads, 1, 1);

#if USE_SORT_SHARED_MEM
  bitonicSort_global<<<blocks, threads, nump_ceil * sizeof(sortstruct_t), stream>>>(pts, newpts, d_ExtremaIdx, nrSortThreads);
#else
  sortstruct_t *sortstruct = nullptr;
  safeCall(cudaMalloc((void **)&sortstruct, nump_ceil * nrLevelImgs * sizeof(sortstruct_t)));
  safeCall(cudaMemsetAsync(sortstruct, 0, nump_ceil * nrLevelImgs * sizeof(sortstruct_t), stream));
  bitonicSort_global<<<blocks, threads, 0, stream>>>(pts, newpts, sortstruct, nump_ceil, d_ExtremaIdx, nrSortThreads);
#endif
  checkMsg("Cuda error at kernel bitonicSort_global");

#if !USE_SORT_SHARED_MEM
  safeCall(cudaFree(sortstruct));
#endif

  // Find all neighbors
  blocks.x = nump;
  threads.x = FindNeighborsThreads;
  FindNeighbors<<<blocks, threads, 0, stream>>>(newpts, kptindices, width, d_ExtremaIdx);

  blocks.x = 1;
  threads.x = FilterExtremaThreads;
  int *buffer1, *buffer2;
  safeCall(cudaMalloc((void **)&buffer1, nump * sizeof(int)));
  safeCall(cudaMalloc((void **)&buffer2, nump * sizeof(int)));
  unsigned char *buffer3;
  safeCall(cudaMalloc((void **)&buffer3, nump * sizeof(char)));
  safeCall(cudaMemsetAsync(buffer1, -1, nump * sizeof(int), stream));
  safeCall(cudaMemsetAsync(buffer2, nump + 1, nump * sizeof(int), stream));
  safeCall(cudaMemsetAsync(buffer3, -127, nump * sizeof(unsigned char), stream));
  FilterExtrema_kernel<<<blocks, threads, 0, stream>>>(newpts, pts, kptindices, width,
                                                       buffer1, buffer2, buffer3, d_PointCounter);
  checkMsg("Cuda error at kernel FilterExtrema_kernel");

  threads.x = 1024;
  sortFiltered_kernel<<<blocks, threads, 0, stream>>>(newpts, pts, buffer1, d_PointCounter);
  checkMsg("Cuda error at kernel sortFiltered_kernel");
  safeCall(cudaStreamSynchronize(stream));
  safeCall(cudaFree(buffer1));
  safeCall(cudaFree(buffer2));
  safeCall(cudaFree(buffer3));
  safeCall(cudaMemcpyAsync(&nump, d_PointCounter, sizeof(int), cudaMemcpyDeviceToHost, stream));
  safeCall(cudaStreamSynchronize(stream));
  delete[] extremaidx_h;
}

int GetPoints(std::vector<cv::KeyPoint> &h_pts, cv::KeyPoint *d_pts, int numPts, cudaStream_t &stream)
{
  if(!numPts){
    return 0;
  }
  h_pts.resize(numPts);
  safeCall(cudaMemcpyAsync((float *)&h_pts[0], d_pts,
                           sizeof(cv::KeyPoint) * numPts,
                           cudaMemcpyDeviceToHost, stream));
  safeCall(cudaStreamSynchronize(stream));
  return numPts;
}

void GetDescriptors(cv::Mat &h_desc, unsigned char *d_desc, int numPts, cudaStream_t &stream)
{
  if (!numPts)
  {
    return;
  }
  h_desc = cv::Mat(numPts, 61, CV_8U);
  safeCall(cudaMemcpyAsync(h_desc.data, d_desc, numPts * 61, cudaMemcpyDeviceToHost, stream));
  safeCall(cudaStreamSynchronize(stream));
}

__global__ void ExtractDescriptorsT(cv::KeyPoint *d_pts, CudaImage *d_imgs,
                                   float *_vals, int size2, int size3,
                                   int size4) {
  __shared__ float acc_vals[3 * 30 * EXTRACT_S];

  float *acc_vals_im = &acc_vals[0];
  float *acc_vals_dx = &acc_vals[30 * EXTRACT_S];
  float *acc_vals_dy = &acc_vals[2 * 30 * EXTRACT_S];

  int p = blockIdx.x;

  float *vals = &_vals[p * 3 * 29];

  float iratio = 1.0f / (1 << d_pts[p].octave);
  int scale = (int)(0.5f * d_pts[p].size * iratio + 0.5f);
  float xf = d_pts[p].pt.x * iratio;
  float yf = d_pts[p].pt.y * iratio;
  float ang = d_pts[p].angle;
  float co = cos(ang);
  float si = sin(ang);
  int tx = threadIdx.x;
  int lev = d_pts[p].class_id;
  float *imd = d_imgs[4 * lev + 0].d_data;
  float *dxd = d_imgs[4 * lev + 2].d_data;
  float *dyd = d_imgs[4 * lev + 3].d_data;
  int pitch = d_imgs[4 * lev + 0].pitch;
  int winsize = max(3 * size3, 4 * size4);

  for (int i = 0; i < 30; ++i) {
    acc_vals_im[i * EXTRACT_S + tx] = 0.f;
    acc_vals_dx[i * EXTRACT_S + tx] = 0.f;
    acc_vals_dy[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float im = imd[pos];
    float dx = dxd[pos];
    float dy = dyd[pos];
    float rx = -dx * si + dy * co;
    float ry = dx * co + dy * si;

    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // Add 2x2
      acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx] += im;
      acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx + 1] += rx;
      acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx + 2] += ry;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // Add 3x3
      acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx] += im;
      acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx + 1] += rx;
      acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx + 2] += ry;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // Add 4x4
      acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx] += im;
      acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx + 1] += rx;
      acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx + 2] += ry;
    }
  }

  __syncthreads();

// Reduce stuff
  float acc_reg;
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    for (int d = 0; d < 90; d += 30) {
      if (tx_d < 32) {
        acc_reg = acc_vals[3 * 30 * tx_d + offset + d] +
                  acc_vals[3 * 30 * (tx_d + 32) + offset + d];
        acc_reg += __shfl_down_sync(0xFFFFFFFF, acc_reg, 1);
        acc_reg += __shfl_down_sync(0xFFFFFFFF, acc_reg, 2);
        acc_reg += __shfl_down_sync(0xFFFFFFFF, acc_reg, 4);
        acc_reg += __shfl_down_sync(0xFFFFFFFF, acc_reg, 8);
        acc_reg += __shfl_down_sync(0xFFFFFFFF, acc_reg, 16);
      }
      if (tx_d == 0) {
        acc_vals[offset + d] = acc_reg;
      }
    }
  }

  __syncthreads();

  // Have 29*3 values to store
  // They are in acc_vals[0..28,64*30..64*30+28,64*60..64*60+28]
  if (tx < 29) {
    vals[tx] = acc_vals[tx];
    vals[29 + tx] = acc_vals[29 + tx];
    vals[2 * 29 + tx] = acc_vals[2 * 29 + tx];
  }
}

__global__ void ExtractDescriptors_serial(cv::KeyPoint *d_pts,
                                          CudaImage *d_imgs, float *_vals,
                                          int size2, int size3, int size4) {
  __shared__ float acc_vals[30 * EXTRACT_S];
  __shared__ float final_vals[3 * 30];

  int p = blockIdx.x;

  float *vals = &_vals[p * 3 * 29];

  float iratio = 1.0f / (1 << d_pts[p].octave);
  int scale = (int)(0.5f * d_pts[p].size * iratio + 0.5f);
  float xf = d_pts[p].pt.x * iratio;
  float yf = d_pts[p].pt.y * iratio;
  float ang = d_pts[p].angle;
  float co = cos(ang);
  float si = sin(ang);
  int tx = threadIdx.x;
  int lev = d_pts[p].class_id;
  float *imd = d_imgs[4 * lev + 0].d_data;
  float *dxd = d_imgs[4 * lev + 2].d_data;
  float *dyd = d_imgs[4 * lev + 3].d_data;
  int pitch = d_imgs[4 * lev + 0].pitch;
  int winsize = max(3 * size3, 4 * size4);

  // IM
  for (int i = 0; i < 30; ++i) {
    acc_vals[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float im = imd[pos];
    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // Add 2x2
      acc_vals[(y2 * 2 + x2) + 30 * tx] += im;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // Add 3x3
      acc_vals[(4 + y3 * 3 + x3) + 30 * tx] += im;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // Add 4x4
      acc_vals[(4 + 9 + y4 * 4 + x4) + 30 * tx] += im;
    }
  }

  __syncthreads();

// Reduce stuff
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    int acc_idx = 30 * tx_d + offset;
    if (tx_d < 32) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 32];
    }
    if (tx_d < 16) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 16];
    }
    if (tx_d < 8) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 8];
    }
    if (tx_d < 4) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 4];
    }
    if (tx_d < 2) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 2];
    }
    if (tx_d < 1) {
      final_vals[3 * offset] = acc_vals[acc_idx] + acc_vals[offset + 30];
    }
  }

  // DX
  for (int i = 0; i < 30; ++i) {
    acc_vals[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float dx = dxd[pos];
    float dy = dyd[pos];
    float rx = -dx * si + dy * co;
    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // Add 2x2
      acc_vals[(y2 * 2 + x2) + 30 * tx] += rx;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // Add 3x3
      acc_vals[(4 + y3 * 3 + x3) + 30 * tx] += rx;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // Add 4x4
      acc_vals[(4 + 9 + y4 * 4 + x4) + 30 * tx] += rx;
    }
  }

  __syncthreads();

// Reduce stuff
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    int acc_idx = 30 * tx_d + offset;
    if (tx_d < 32) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 32];
    }
    if (tx_d < 16) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 16];
    }
    if (tx_d < 8) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 8];
    }
    if (tx_d < 4) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 4];
    }
    if (tx_d < 2) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 2];
    }
    if (tx_d < 1) {
      final_vals[3 * offset] = acc_vals[acc_idx] + acc_vals[offset + 30];
    }
  }

  // DY
  for (int i = 0; i < 30; ++i) {
    acc_vals[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float dx = dxd[pos];
    float dy = dyd[pos];
    float ry = dx * co + dy * si;
    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // Add 2x2
      acc_vals[(y2 * 2 + x2) + 30 * tx] += ry;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // Add 3x3
      acc_vals[(4 + y3 * 3 + x3) + 30 * tx] += ry;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // Add 4x4
      acc_vals[(4 + 9 + y4 * 4 + x4) + 30 * tx] += ry;
    }
  }

  __syncthreads();

// Reduce stuff
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    int acc_idx = 30 * tx_d + offset;
    if (tx_d < 32) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 32];
    }
    if (tx_d < 16) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 16];
    }
    if (tx_d < 8) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 8];
    }
    if (tx_d < 4) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 4];
    }
    if (tx_d < 2) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 2];
    }
    if (tx_d < 1) {
      final_vals[3 * offset] = acc_vals[acc_idx] + acc_vals[offset + 30];
    }
  }

  __syncthreads();

  // Have 29*3 values to store
  // They are in acc_vals[0..28,64*30..64*30+28,64*60..64*60+28]
  if (tx < 29) {
    vals[tx] = final_vals[tx];
    vals[29 + tx] = final_vals[29 + tx];
    vals[2 * 29 + tx] = final_vals[2 * 29 + tx];
  }
}

__global__ void BuildDescriptor(float *_valsim, unsigned char *_desc, const int *comp_idx_1, const int *comp_idx_2)
{
  int p = blockIdx.x;
  size_t idx = threadIdx.x;

  if (idx < 61) {
    float *valsim = &_valsim[3 * 29 * p];
    unsigned char *desc = &_desc[61 * p];
    unsigned char desc_r = 0;

#pragma unroll
    for (int i = 0; i < (idx == 60 ? 6 : 8); ++i) {
      int idx1 = comp_idx_1[idx * 8 + i];
      int idx2 = comp_idx_2[idx * 8 + i];
      desc_r |= (valsim[idx1] > valsim[idx2] ? 1 : 0) << i;
    }

    desc[idx] = desc_r;
  }
}

void ExtractDescriptors(cv::KeyPoint *d_pts, CudaImage *d_imgs,
                        unsigned char *desc_d, float *vals_d, int patsize, int numPts, int *comp_idx_1, int *comp_idx_2, cudaStream_t &stream, const std::string *info)
{
  if (!numPts)
  {
    return;
  }
  int size2 = patsize;
  int size3 = ceil(2.0f * patsize / 3.0f);
  int size4 = ceil(0.5f * patsize);

  dim3 blocks(numPts);
  dim3 threads(EXTRACT_S);

  ExtractDescriptorsT<<<blocks, threads, 0, stream>>>(d_pts, d_imgs, vals_d, size2, size3, size4);
  checkMsg("Cuda error at kernel ExtractDescriptorsT");

  safeCall(cudaMemsetAsync(desc_d, 0, numPts * 61, stream));
  BuildDescriptor<<<blocks, 64, 0, stream>>>(vals_d, desc_d, comp_idx_1, comp_idx_2);
  checkMsg("Cuda error at kernel BuildDescriptor");
}

#define NTHREADS_MATCH 32
__global__ void MatchDescriptors(unsigned char *d1, unsigned char *d2,
                                 int pitch, int nkpts_2, cv::DMatch *matches) {
  int p = blockIdx.x;

  int x = threadIdx.x;

  __shared__ int idxBest[NTHREADS_MATCH];
  __shared__ int idxSecondBest[NTHREADS_MATCH];
  __shared__ int scoreBest[NTHREADS_MATCH];
  __shared__ int scoreSecondBest[NTHREADS_MATCH];

  idxBest[x] = 0;
  idxSecondBest[x] = 0;
  scoreBest[x] = 512;
  scoreSecondBest[x] = 512;

  __syncthreads();

  // curent version fixed with popc, still not convinced
  unsigned long long *d1i = (unsigned long long *)(d1 + pitch * p);

  for (int i = 0; i < nkpts_2; i += NTHREADS_MATCH) {
    unsigned long long *d2i = (unsigned long long *)(d2 + pitch * (x + i));
    if (i + x < nkpts_2) {
      // Check d1[p] with d2[i]
      int score = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        score += __popcll(d1i[j] ^ d2i[j]);
      }
      if (score < scoreBest[x]) {
        scoreSecondBest[x] = scoreBest[x];
        scoreBest[x] = score;
        idxSecondBest[x] = idxBest[x];
        idxBest[x] = i + x;
      } else if (score < scoreSecondBest[x]) {
        scoreSecondBest[x] = score;
        idxSecondBest[x] = i + x;
      }
    }
  }

  __syncthreads();

  for (int i = NTHREADS_MATCH / 2; i >= 1; i /= 2) {
    if (x < i) {
      if (scoreBest[x + i] < scoreBest[x]) {
        scoreSecondBest[x] = scoreBest[x];
        scoreBest[x] = scoreBest[x + i];
        idxSecondBest[x] = idxBest[x];
        idxBest[x] = idxBest[x + i];
      } else if (scoreBest[x + i] < scoreSecondBest[x]) {
        scoreSecondBest[x] = scoreBest[x + i];
        idxSecondBest[x] = idxBest[x + i];
      }
      if (scoreSecondBest[x + i] < scoreSecondBest[x]) {
        scoreSecondBest[x] = scoreSecondBest[x + i];
        idxSecondBest[x] = idxSecondBest[x + i];
      }
    }
  }

  if (x == 0) {
    matches[2 * p].queryIdx = p;
    matches[2 * p].trainIdx = idxBest[x];
    matches[2 * p].distance = scoreBest[x];
    matches[2 * p + 1].queryIdx = p;
    matches[2 * p + 1].trainIdx = idxSecondBest[x];
    matches[2 * p + 1].distance = scoreSecondBest[x];
  }
}

void MatchDescriptors(cv::Mat &desc_query, cv::Mat &desc_train,
                      std::vector<std::vector<cv::DMatch>> &dmatches,
                      size_t pitch,
                      unsigned char *descq_d, unsigned char *desct_d, cv::DMatch *dmatches_d, cv::DMatch *dmatches_h)
{

  dim3 block(desc_query.rows);

  MatchDescriptors<<<block, NTHREADS_MATCH>>>(descq_d, desct_d, pitch, desc_train.rows, dmatches_d);

  safeCall(cudaMemcpy(dmatches_h, dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch),
                      cudaMemcpyDeviceToHost));

  for (int i = 0; i < desc_query.rows; ++i)
  {
    std::vector<cv::DMatch> tdmatch;
    tdmatch.push_back(dmatches_h[2 * i]);
    tdmatch.push_back(dmatches_h[2 * i + 1]);
    dmatches.push_back(tdmatch);
  }
}

void MatchDescriptors(cv::Mat &desc_query, cv::Mat &desc_train,
                      std::vector<std::vector<cv::DMatch> > &dmatches) {
  size_t pitch1, pitch2;
  unsigned char *descq_d;
  safeCall(cudaMallocPitch(&descq_d, &pitch1, 64, desc_query.rows));
  safeCall(cudaMemset2D(descq_d, pitch1, 0, 64, desc_query.rows));
  safeCall(cudaMemcpy2D(descq_d, pitch1, desc_query.data, desc_query.cols,
               desc_query.cols, desc_query.rows, cudaMemcpyHostToDevice));
  unsigned char *desct_d;
  safeCall(cudaMallocPitch(&desct_d, &pitch2, 64, desc_train.rows));
  safeCall(cudaMemset2D(desct_d, pitch2, 0, 64, desc_train.rows));
  safeCall(cudaMemcpy2D(desct_d, pitch2, desc_train.data, desc_train.cols,
               desc_train.cols, desc_train.rows, cudaMemcpyHostToDevice));

  dim3 block(desc_query.rows);

  cv::DMatch *dmatches_d;
  safeCall(cudaMalloc(&dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch)));

  MatchDescriptors << <block, NTHREADS_MATCH>>>(descq_d, desct_d, pitch1, desc_train.rows, dmatches_d);

  cv::DMatch *dmatches_h = new cv::DMatch[2 * desc_query.rows];
  safeCall(cudaMemcpy(dmatches_h, dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch),
             cudaMemcpyDeviceToHost));

  for (int i = 0; i < desc_query.rows; ++i) {
    std::vector<cv::DMatch> tdmatch;
    tdmatch.push_back(dmatches_h[2 * i]);
    tdmatch.push_back(dmatches_h[2 * i + 1]);
    dmatches.push_back(tdmatch);
  }

  safeCall(cudaFree(descq_d));
  safeCall(cudaFree(desct_d));
  safeCall(cudaFree(dmatches_d));

  delete[] dmatches_h;
}

void InitCompareIndices(int *comp_idx_1, int *comp_idx_2, cudaStream_t &stream)
{
  int comp_idx_1_h[61 * 8] = {0};
  int comp_idx_2_h[61 * 8] = {0};
  safeCall(cudaHostRegister(comp_idx_1_h, 61 * 8 * sizeof(int), cudaHostRegisterDefault));
  safeCall(cudaHostRegister(comp_idx_2_h, 61 * 8 * sizeof(int), cudaHostRegisterDefault));

  int cntr = 0;
  for (int j = 0; j < 4; ++j)
  {
    for (int i = j + 1; i < 4; ++i)
    {
      comp_idx_1_h[cntr] = 3 * j;
      comp_idx_2_h[cntr] = 3 * i;
      cntr++;
    }
  }
  for (int j = 0; j < 3; ++j)
  {
    for (int i = j + 1; i < 4; ++i)
    {
      comp_idx_1_h[cntr] = 3 * j + 1;
      comp_idx_2_h[cntr] = 3 * i + 1;
      cntr++;
    }
  }
  for (int j = 0; j < 3; ++j)
  {
    for (int i = j + 1; i < 4; ++i)
    {
      comp_idx_1_h[cntr] = 3 * j + 2;
      comp_idx_2_h[cntr] = 3 * i + 2;
      cntr++;
    }
  }

  // 3x3
  for (int j = 4; j < 12; ++j)
  {
    for (int i = j + 1; i < 13; ++i)
    {
      comp_idx_1_h[cntr] = 3 * j;
      comp_idx_2_h[cntr] = 3 * i;
      cntr++;
    }
  }
  for (int j = 4; j < 12; ++j)
  {
    for (int i = j + 1; i < 13; ++i)
    {
      comp_idx_1_h[cntr] = 3 * j + 1;
      comp_idx_2_h[cntr] = 3 * i + 1;
      cntr++;
    }
  }
  for (int j = 4; j < 12; ++j)
  {
    for (int i = j + 1; i < 13; ++i)
    {
      comp_idx_1_h[cntr] = 3 * j + 2;
      comp_idx_2_h[cntr] = 3 * i + 2;
      cntr++;
    }
  }

  // 4x4
  for (int j = 13; j < 28; ++j)
  {
    for (int i = j + 1; i < 29; ++i)
    {
      comp_idx_1_h[cntr] = 3 * j;
      comp_idx_2_h[cntr] = 3 * i;
      cntr++;
    }
  }
  for (int j = 13; j < 28; ++j)
  {
    for (int i = j + 1; i < 29; ++i)
    {
      comp_idx_1_h[cntr] = 3 * j + 1;
      comp_idx_2_h[cntr] = 3 * i + 1;
      cntr++;
    }
  }
  for (int j = 13; j < 28; ++j)
  {
    for (int i = j + 1; i < 29; ++i)
    {
      comp_idx_1_h[cntr] = 3 * j + 2;
      comp_idx_2_h[cntr] = 3 * i + 2;
      cntr++;
    }
  }

  safeCall(cudaMemcpyAsync(comp_idx_1, comp_idx_1_h, 8 * 61 * sizeof(int), cudaMemcpyHostToDevice, stream));
  safeCall(cudaMemcpyAsync(comp_idx_2, comp_idx_2_h, 8 * 61 * sizeof(int), cudaMemcpyHostToDevice, stream));
  safeCall(cudaStreamSynchronize(stream));
  safeCall(cudaHostUnregister(comp_idx_1_h));
  safeCall(cudaHostUnregister(comp_idx_2_h));
}

__global__ void FindOrientationT(cv::KeyPoint *d_pts, CudaImage *d_imgs)
{
  __shared__ float resx_threads[ORIENT_S], resy_threads[ORIENT_S];
  __shared__ int angle_bin_threads[ORIENT_S];
  __shared__ float resx[42], resy[42];
  __shared__ float re8x[42], re8y[42];

  int p = blockIdx.x;
  int tx = threadIdx.x;

  if (tx < 42){
    resx[tx] = resy[tx] = 0.0f;
  }

  resx_threads[tx] = 0;
  resy_threads[tx] = 0;
  angle_bin_threads[tx] = -1;

  __syncthreads();
  int lev = d_pts[p].class_id;
  int img_idx0 = 4 * lev;
  float *dxd = d_imgs[img_idx0 + 2].d_data;
  float *dyd = d_imgs[img_idx0 + 3].d_data;
  int pitch = d_imgs[img_idx0].pitch;
  int width = d_imgs[img_idx0].width;
  int height = d_imgs[img_idx0].height;
  int octave = d_pts[p].octave;
  int step = (int)(0.5f * d_pts[p].size + 0.5f) >> octave;
  int x = (int)(d_pts[p].pt.x + 0.5f) >> octave;
  int y = (int)(d_pts[p].pt.y + 0.5f) >> octave;
  int i = (tx & 15) - 6;
  int j = (tx / 16) - 6;
  int r2 = i * i + j * j;
  if (r2 < 36) {
    //Get histogram (adding dx and dy for every bin) of angles
    float gweight = exp(-r2 / (2.5f * 2.5f * 2.0f));
    int pos_y = y + step * j;
    if (pos_y >= 0 && pos_y < height)
    {
      pos_y *= pitch;
      int pos_x = x + step * i;
      if (pos_x >= 0 && pos_x < width)
      {
        int pos = pos_y + pos_x;
        float dx = gweight * dxd[pos];
        float dy = gweight * dyd[pos];
        if (fabs(dy) > 1e-6 || fabs(dx) > 1e-6)
        {
          float angle = atan2(dy, dx);
          int a = max(min((int)(angle * (21.f / CV_PI)) + 21, 41), 0);
          resx_threads[tx] = dx;
          resy_threads[tx] = dy;
          angle_bin_threads[tx] = a;
        }
      }
    }
  }
  __syncthreads();
  if(tx == 1){
#pragma unroll
    for (int i = 0; i < ORIENT_S; i++)
    {
      const int &a = angle_bin_threads[i];
      if (a >= 0)
      {
        resx[a] += resx_threads[i];
        resy[a] += resy_threads[i];
      }
    }
  }
  __syncthreads();
  //Next: Running avg with window size of 7
  if (tx < 42) {
    re8x[tx] = resx[tx];
    re8y[tx] = resy[tx];
    for (int k = tx + 1; k < tx + 7; k++) {
      re8x[tx] += resx[k < 42 ? k : k - 42];
      re8y[tx] += resy[k < 42 ? k : k - 42];
    }
  }
  __syncthreads();
  //Get index (bin) in histogram of max length for dx and dy
  if (tx == 0) {
    float maxr = 0.0f;
    int maxk = 0;
    for (int k = 0; k < 42; k++) {
      float r = re8x[k] * re8x[k] + re8y[k] * re8y[k];
      if (r > maxr) {
        maxr = r;
        maxk = k;
      }
    }
    float angle = 0;
    if (fabs(re8y[maxk]) > 1e-5 || fabs(re8x[maxk]) > 1e-5)
    {
      angle = atan2(re8y[maxk], re8x[maxk]);
    }
    d_pts[p].angle = (angle < 0.0f ? angle + 2.0f * CV_PI : angle);
  }
}

void FindOrientation(cv::KeyPoint *d_pts, std::vector<CudaImage> &h_imgs, CudaImage *d_imgs, int numPts, cudaStream_t &stream, const std::string *info)
{
  if(!numPts){
    return;
  }

  safeCall(cudaMemcpyAsync(d_imgs, (float *)h_imgs.data(),
                           sizeof(CudaImage) * h_imgs.size(),
                           cudaMemcpyHostToDevice, stream));

  dim3 blocks(numPts);
  dim3 threads(ORIENT_S);
  FindOrientationT<<<blocks, threads, 0, stream>>>(d_pts, d_imgs);
  checkMsg("Cuda error at kernel FindOrientationT");
}

#define REF_THREADS 1024
__global__ void computeDxy_PtRefine(cv::KeyPoint *d_pts, CudaImage *d_imgs, float *dxy, int nump)
{
  int elem = blockIdx.x;

  for (int i = threadIdx.x; i < nump; i += REF_THREADS)
  {
    int lev = d_pts[i].class_id;
    float *ldet = d_imgs[4 * lev + 1].d_data;
    int octave = d_pts[i].octave;
    int pitch = d_imgs[4 * lev + 0].pitch;
    int width1 = d_imgs[4 * lev + 0].width - 1;
    int height1 = d_imgs[4 * lev + 0].height - 1;
    int x = (int)(d_pts[i].pt.x + 0.5f) >> octave;
    int y = (int)(d_pts[i].pt.y + 0.5f) >> octave;
    if (elem == 0)
    {
      // Compute the gradient in x
      int pos = y * pitch + x;
      float Dx = 0;
      if (x > 0 && x < width1)
      {
        Dx = 0.5f * (ldet[pos + 1] - ldet[pos - 1]);
      }
      else if (x == 0)
      {
        Dx = 0.5f * (ldet[pos + 2] - ldet[pos]);
      }
      else if (x == width1)
      {
        Dx = 0.5f * (ldet[pos] - ldet[pos - 2]);
      }
      dxy[i * 5] = Dx;
    }
    else if(elem == 1)
    {
      // Compute the gradient in y
      float Dy = 0;
      if (y > 0 && y < height1)
      {
        Dy = 0.5f * (ldet[(y + 1) * pitch + x] - ldet[(y - 1) * pitch + x]);
      }
      else if (y == 0)
      {
        Dy = 0.5f * (ldet[(y + 2) * pitch + x] - ldet[y * pitch + x]);
      }
      else if (y == height1)
      {
        Dy = 0.5f * (ldet[y * pitch + x] - ldet[(y - 2) * pitch + x]);
      }
      dxy[i * 5 + 1] = Dy;
    }
    else if (elem == 2)
    {
      // Compute the Hessian in x
      int pos = y * pitch + x;
      float Dxx = 0;
      if (x > 0 && x < width1)
      {
        Dxx = ldet[pos + 1] + ldet[pos - 1] - 2.0f * ldet[pos];
      }
      else if (x == 0)
      {
        Dxx = ldet[pos + 2] + ldet[pos] - 2.0f * ldet[pos + 1];
      }
      else if (x == width1)
      {
        Dxx = ldet[pos] + ldet[pos - 2] - 2.0f * ldet[pos - 1];
      }
      dxy[i * 5 + 2] = Dxx;
    }
    else if (elem == 3)
    {
      // Compute the Hessian in y
      float Dyy = 0;
      if (y > 0 && y < height1)
      {
        Dyy = ldet[(y + 1) * pitch + x] + ldet[(y - 1) * pitch + x] - 2.0f * ldet[y * pitch + x];
      }
      else if (y == 0)
      {
        Dyy = ldet[(y + 2) * pitch + x] + ldet[y * pitch + x] - 2.0f * ldet[(y + 1) * pitch + x];
      }
      else if (y == height1)
      {
        Dyy = ldet[y * pitch + x] + ldet[(y - 2) * pitch + x] - 2.0f * ldet[(y - 1) * pitch + x];
      }
      dxy[i * 5 + 3] = Dyy;
    }
    else if (elem == 4)
    {
      // Compute the Hessian in xy
      float Dxy = 0;
      if (x > 0 && x < width1 && y > 0 && y < height1)
      {
        Dxy = 0.25f * (ldet[(y + 1) * pitch + x + 1] + ldet[(y - 1) * pitch + x - 1] -
                       ldet[(y - 1) * pitch + x + 1] - ldet[(y + 1) * pitch + x - 1]);
      }
      else
      {
        int xp = x + 1;
        int xn = x - 1;
        if (x == 0)
        {
          xp++;
          xn++;
        }
        else if (x == width1)
        {
          xp--;
          xn--;
        }
        int yp = y + 1;
        int yn = y - 1;
        if (y == 0)
        {
          yp++;
          yn++;
        }
        else if (y == height1)
        {
          yp--;
          yn--;
        }
        Dxy = 0.25f * (ldet[yp * pitch + xp] + ldet[yn * pitch + xn] -
                       ldet[yn * pitch + xp] - ldet[yp * pitch + xn]);
      }
      dxy[i * 5 + 4] = Dxy;
    }
  }
}

__global__ void computePtDiff(const float *dxy5, float *dxy, int nump)
{
  for (int i = threadIdx.x; i < nump; i += REF_THREADS)
  {
    int ii = i * 5;
    float d1 = dxy5[ii + 2];
    float d2 = dxy5[ii + 4];
    float d3 = d2;
    float d4 = dxy5[ii + 3];

    float b1 = -dxy5[ii];
    float b2 = -dxy5[ii + 1];

    float nu = b1 * d3 - b2 * d1;
    float de = d2 * d3 - d4 * d1;

    float x1 = 0;
    float x2 = 0;
    if(fabsf(de) > 1e-4){
      x2 = nu / de;
      if (fabsf(x2) <= 1.f){
        if (fabsf(d1) > fabsf(d3)){
          nu = b1 - d2 * x2;
          de = d1;
        }else{
          nu = b2 - d4 * x2;
          de = d3;
        }
        if (fabsf(de) > 1e-4){
          x1 = nu / de;
          if (fabsf(x1) > 1.f){
            x1 = 0;
            x2 = 0;
          }
        }
        else
        {
          x1 = 0;
          x2 = 0;
        }
      }else{
        x2 = 0;
      }
    }
    dxy[i * 2] = x1;
    dxy[i * 2 + 1] = x2;
  }
}

__global__ void refinePts(cv::KeyPoint *d_pts, const float *dxy, int nump)
{
  for (int i = threadIdx.x; i < nump; i += REF_THREADS)
  {
    int octave = d_pts[i].octave;
    float ratio = 1 << octave;
    float dx = dxy[i * 2];
    float dy = dxy[i * 2 + 1];

    // Refine the coordinates
    d_pts[i].pt.x += dx * ratio + .5f * (ratio - 1.f);
    d_pts[i].pt.y += dy * ratio + .5f * (ratio - 1.f);
  }
}

// Subpixel refinement is already applied in function sortFiltered_kernel
// using subpixel refinement calculated in FindExtremaT (dx is stored in kp.angle and dy in kp.octave)
void subpixel_refinement(cv::KeyPoint *d_pts, CudaImage *d_imgs, int &nrLevels, const int &nump, cudaStream_t &stream)
{
  float *dxy5 = nullptr;
  safeCall(cudaMalloc((void **)&dxy5, nump * 5 * sizeof(float)));
  dim3 blocks(5, 1, 1);
  dim3 threads(REF_THREADS, 1, 1);
  computeDxy_PtRefine<<<blocks, threads, 0, stream>>>(d_pts, d_imgs, dxy5, nump);
  checkMsg("Cuda error at kernel computeDxy_PtRefine");

  float *dxy = nullptr;
  safeCall(cudaMalloc((void **)&dxy, nump * 2 * sizeof(float)));
  blocks.x = 1;
  computePtDiff<<<blocks, threads, 0, stream>>>(dxy5, dxy, nump);
  checkMsg("Cuda error at kernel computePtDiff");

  refinePts<<<blocks, threads, 0, stream>>>(d_pts, dxy, nump);
  checkMsg("Cuda error at kernel refinePts");

  safeCall(cudaFree(dxy5));
  safeCall(cudaFree(dxy));
}
