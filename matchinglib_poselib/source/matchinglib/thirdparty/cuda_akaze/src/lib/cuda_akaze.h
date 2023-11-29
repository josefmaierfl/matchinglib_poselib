#pragma once
#include "AKAZEConfig.h"
#include "cudaImage.h"
#include "cudautils.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

void MatchDescriptors(cv::Mat &desc_query, cv::Mat &desc_train,
		      std::vector<std::vector<cv::DMatch> > &dmatches,
		      size_t pitch, 
		      unsigned char* descq_d, unsigned char* desct_d, cv::DMatch* dmatches_d, cv::DMatch* dmatches_h);
void MatchDescriptors(cv::Mat& desc_query, cv::Mat& desc_train, std::vector<std::vector<cv::DMatch> >& dmatches);

struct Buffer_Ptrs{
	std::vector<float *> img_buffers;
	std::vector<void *> kp_buffers;
	std::vector<void *> descr_buffers;
	int *indices_buffer;
	void *struct_buffer;
};

Buffer_Ptrs AllocBuffers(int width, int height, int num, int omax, int &maxpts, std::vector<CudaImage> &buffers, cv::KeyPoint *&pts, cv::KeyPoint *&ptsbuffer, int *&ptindices, unsigned char *&desc, float *&descbuffer, CudaImage *&ims, int *comp_idx_1, int *comp_idx_2, cudaStream_t &stream);
void InitCompareIndices(int *comp_idx_1, int *comp_idx_2, cudaStream_t &stream);
void FreeBuffers(Buffer_Ptrs &buffers);
void LowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var, int kernsize, cudaStream_t &stream, const std::string *info = nullptr);
void Scharr(CudaImage &img, CudaImage &lx, CudaImage &ly, cudaStream_t &stream);
void Flow(CudaImage &img, CudaImage &flow, DIFFUSIVITY_TYPE type, float kcontrast, cudaStream_t &stream, const std::string *info = nullptr);
void NLDStep(CudaImage &img, CudaImage &flow, CudaImage &temp, float stepsize, cudaStream_t &stream, const std::string *info = nullptr);
void HalfSample(CudaImage &inimg, CudaImage &outimg, cudaStream_t &stream, const std::string *info = nullptr);
void Copy(CudaImage &inimg, CudaImage &outimg, cudaStream_t &stream, const std::string *info = nullptr);
void ContrastPercentile(CudaImage &img, CudaImage &temp, CudaImage &blur, float perc, int nbins, float &contrast, cudaStream_t &stream, const std::string *info = nullptr);
void HessianDeterminant(CudaImage &img, CudaImage &lx, CudaImage &ly, int step, cudaStream_t &stream, const std::string *info = nullptr);
bool FindExtrema(CudaImage &img, CudaImage &imgp, CudaImage &imgn, float border, float dthreshold, int scale, int octave, float size, cv::KeyPoint *pts, int maxpts, unsigned int *d_PointCounter, unsigned int *d_ExtremaIdx, cudaStream_t &stream, const std::string *info = nullptr);
void FilterExtrema(cv::KeyPoint *pts, cv::KeyPoint *newpts, int *kptindices, int &nump, unsigned int *d_PointCounter, unsigned int *d_ExtremaIdx, int nrLevelImgs, cudaStream_t &stream, const std::string *info = nullptr);
void ClearPoints(unsigned int *d_PointCounter, cudaStream_t &stream);
int GetPoints(std::vector<cv::KeyPoint> &h_pts, cv::KeyPoint *d_pts, int numPts, cudaStream_t &stream);
void WaitCuda(cudaStream_t &stream);
void DestroyStream(cudaStream_t &stream);
void createStream(cudaStream_t &stream);
void GetDescriptors(cv::Mat &h_desc, unsigned char *d_desc, int numPts, cudaStream_t &stream);
void FindOrientation(cv::KeyPoint *d_pts, std::vector<CudaImage> &h_imgs, CudaImage *d_imgs, int numPts, cudaStream_t &stream, const std::string *info = nullptr);
void ExtractDescriptors(cv::KeyPoint *d_pts, CudaImage *cuda_images, unsigned char *desc_d, float *vals_d, int patsize, int numPts, int *comp_idx_1, int *comp_idx_2, cudaStream_t &stream, const std::string *info = nullptr);
void subpixel_refinement(cv::KeyPoint *d_pts, CudaImage *d_imgs, int &nrLevels, const int &nump, cudaStream_t &stream);

template <typename T>
void allocCudaVar(T *&data, const size_t &length, cudaStream_t &stream);
void freeCudaVar(void *data);