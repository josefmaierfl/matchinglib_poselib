/**
 * @file AKAZE.h
 * @brief Main class for detecting and computing binary descriptors in an
 * accelerated nonlinear scale space
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#pragma once

/* ************************************************************************* */
// #define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include "AKAZEConfig.h"
#include "fed.h"
#include "utils.h"
#include "nldiffusion_functions.h"
#include "cudaImage.h"
#include "cuda_akaze.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <random>

// OpenCV
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/core/cuda.hpp"

/* ************************************************************************* */
namespace libAKAZECU {

  class Matcher
  {
  private:
    int maxnquery;
    unsigned char *descq_d;

    int maxntrain;
    unsigned char *desct_d;

    cv::DMatch *dmatches_d;
    cv::DMatch *dmatches_h;

    size_t pitch;

  public:
    Matcher() : maxnquery(0), descq_d(nullptr), maxntrain(0), desct_d(nullptr),
                dmatches_d(0), dmatches_h(0), pitch(0) {}

    ~Matcher();

    // python
    cv::Mat bfmatch_(cv::Mat desc_query, cv::Mat desc_train);

    void bfmatch(cv::Mat &desc_query, cv::Mat &desc_train,
                 std::vector<std::vector<cv::DMatch>> &dmatches);
  };

  class AKAZE
  {
  private:
    AKAZEOptions options_;              ///< Configuration options for AKAZE
    std::vector<TEvolution> evolution_; ///< Vector of nonlinear diffusion evolution
    cudaStream_t &stream;
    unsigned int *d_PointCounter = nullptr, *d_ExtremaIdx = nullptr;
    int *comp_idx_1 = nullptr, *comp_idx_2 = nullptr;
    cv::cuda::HostMem h_img;
    std::mt19937 &mt;

    /// FED parameters
    int ncycles_;                            ///< Number of cycles
    bool reordering_;                        ///< Flag for reordering time steps
    std::vector<std::vector<float>> tsteps_; ///< Vector of FED dynamic time steps
    std::vector<int> nsteps_;                ///< Vector of number of steps per cycle

    /// Matrices for the M-LDB descriptor computation
    cv::Mat descriptorSamples_;
    cv::Mat descriptorBits_;
    cv::Mat bitMask_;

    /// Computation times variables in ms
    AKAZETiming timing_;

    /// CUDA memory buffers
    Buffer_Ptrs b_ptrs;
    cv::KeyPoint *cuda_points;
    cv::KeyPoint *cuda_bufferpoints;
    unsigned char *cuda_desc;
    float *cuda_descbuffer;
    int *cuda_ptindices;
    CudaImage *cuda_images = nullptr;
    std::vector<CudaImage> cuda_buffers;
    int nump;
    double maskRatio = 1.0;

  public:
    /// AKAZE constructor with input options
    /// @param options AKAZE configuration options
    /// @note This constructor allocates memory for the nonlinear scale space
    AKAZE(const AKAZEOptions &options, cudaStream_t &stream_, std::mt19937 &mt_, cv::InputArray mask = cv::noArray());

    /// Destructor
    ~AKAZE();

    /// Allocate the memory for the nonlinear scale space
    void Allocate_Memory_Evolution();

    /// This method creates the nonlinear scale space for a given image
    /// @param img Input image for which the nonlinear scale space needs to be created
    /// @return 0 if the nonlinear scale space was created successfully, -1 otherwise
    int Create_Nonlinear_Scale_Space(const cv::Mat &img);

    /// @brief This method selects interesting keypoints through the nonlinear scale space
    /// @param kpts Vector of detected keypoints
    cv::Mat Feature_Detection_();
    void Feature_Detection(std::vector<cv::KeyPoint> *kpts = nullptr);

    /// This method performs subpixel refinement of the detected keypoints fitting a quadratic
    // void Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts);

    /// Feature description methods
    void Compute_Descriptors(std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);

    /// Return the computation times
    AKAZETiming Get_Computation_Times() const
    {
      return timing_;
    }
  };

  /* ************************************************************************* */

  /// This function sets default parameters for the A-KAZE detector
  void setDefaultAKAZEOptions(AKAZEOptions& options);

  /// This function computes a (quasi-random) list of bits to be taken
  /// from the full descriptor. To speed the extraction, the function creates
  /// a list of the samples that are involved in generating at least a bit (sampleList)
  /// and a list of the comparisons between those samples (comparisons)
  /// @param sampleList
  /// @param comparisons The matrix with the binary comparisons
  /// @param nbits The number of bits of the descriptor
  /// @param pattern_size The pattern size for the binary descriptor
  /// @param nchannels Number of channels to consider in the descriptor (1-3)
  /// @note The function keeps the 18 bits (3-channels by 6 comparisons) of the
  /// coarser grid, since it provides the most robust estimations
  void generateDescriptorSubsample(cv::Mat& sampleList, cv::Mat& comparisons,
                                   int nbits, int pattern_size, int nchannels);
  void generateDescriptorSubsample(cv::Mat &sampleList, cv::Mat &comparisons,
                                   int nbits, int pattern_size, int nchannels, std::mt19937 &mt);

  /// This function checks descriptor limits for a given keypoint
  inline void check_descriptor_limits(int& x, int& y, int width, int height);

  /// This function computes the value of a 2D Gaussian function
  inline float gaussian(float x, float y, float sigma) {
    return expf(-(x*x+y*y)/(2.0f*sigma*sigma));
  }

  /// This funtion rounds float to nearest integer
  inline int fRound(float flt) {
    return (int)(flt+0.5f);
  }
}
