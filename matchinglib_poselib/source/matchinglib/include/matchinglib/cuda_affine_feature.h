//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2021 Josef Maier
//
//Permission is hereby granted, free of charge, to any person obtaining
//a copy of this software and associated documentation files (the "Software"),
//to deal in the Software without restriction, including without limitation
//the rights to use, copy, modify, merge, publish, distribute, sublicense,
//and/or sell copies of the Software, and to permit persons to whom the
//Software is furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included
//in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//Author: Josef Maier (josefjohann-dot-maier-at-gmail-dot-at)

#pragma once

#include "matchinglib/glob_includes.h"
//#include <vector>
//#include <string>
#include <random>

#include "opencv2/core.hpp"
#include <opencv2/features2d.hpp>
// #define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include "opencv2/core/cuda.hpp"

#include "matchinglib/matchinglib_api.h"

namespace matchinglib
{
    namespace cuda
    {
        int MATCHINGLIB_API getMaxNrGpuThreadsFromMemoryUsage(const cv::Mat &image, const size_t &byteMuliplier, const size_t &byteAdd, const size_t &bytesReserve = 0);
        size_t MATCHINGLIB_API getDefaultByteMultiplierGpu(const int nlevels = 4);
        size_t MATCHINGLIB_API getDefaultByteAddGpu();

        class MATCHINGLIB_API Feature2DAsync : public cv::Feature2D
        {
        public:
            CV_WRAP virtual ~Feature2DAsync();
            CV_WRAP virtual void detectAsync(cv::InputArray image,
                                             std::vector<cv::KeyPoint> &keypoints,
                                             cv::InputArray mask = cv::noArray(),
                                             cv::cuda::Stream &stream = cv::cuda::Stream::Null());
            CV_WRAP virtual void detectAsync(cv::InputArray image,
                                             std::vector<cv::KeyPoint> &keypoints,
                                             std::mt19937 &mt,
                                             cv::InputArray mask = cv::noArray(),
                                             cv::cuda::Stream &stream = cv::cuda::Stream::Null());
            CV_WRAP virtual void computeAsync(cv::InputArray image,
                                              std::vector<cv::KeyPoint> &keypoints,
                                              cv::OutputArray descriptors,
                                              cv::cuda::Stream &stream = cv::cuda::Stream::Null());
            CV_WRAP virtual void computeAsync(cv::InputArray image,
                                              std::vector<cv::KeyPoint> &keypoints,
                                              cv::OutputArray descriptors,
                                              std::mt19937 &mt,
                                              cv::cuda::Stream &stream = cv::cuda::Stream::Null());
            CV_WRAP virtual void detectAndComputeAsync(cv::InputArray image,
                                                       cv::InputArray mask,
                                                       std::vector<cv::KeyPoint> &keypoints,
                                                       cv::OutputArray descriptors,
                                                       bool useProvidedKeypoints = false,
                                                       cv::cuda::Stream &stream = cv::cuda::Stream::Null());
            CV_WRAP virtual void detectAndComputeAsync(cv::InputArray image,
                                                       cv::InputArray mask,
                                                       std::vector<cv::KeyPoint> &keypoints,
                                                       cv::OutputArray descriptors,
                                                       std::mt19937 &mt,
                                                       bool useProvidedKeypoints = false,
                                                       cv::cuda::Stream &stream = cv::cuda::Stream::Null());
            // CV_WRAP virtual void convert(cv::InputArray gpu_keypoints,
            //                              CV_OUT std::vector<cv::KeyPoint> &keypoints) = 0;

            // Get byte multiplier for image area which states allocated GPU memory
            CV_WRAP virtual size_t getByteMultiplierGPU() const = 0;
            // Get additional number of bytes of allocated GPU memory
            CV_WRAP virtual size_t getByteAddGPU() const = 0;
        };

        class MATCHINGLIB_API AffineFeature : public Feature2DAsync
        {
        public:
            CV_WRAP static cv::Ptr<AffineFeature> create(const cv::Ptr<cv::Feature2D> &backend,
                                                         int maxTilt = 5, int minTilt = 0, float tiltStep = 1.4142135623730951f, float rotateStepBase = 72, bool parallelize = true, int outerThreadCnt = 1);

            CV_WRAP virtual void setViewParams(const std::vector<float> &tilts, const std::vector<float> &rolls) = 0;
            CV_WRAP virtual void getViewParams(std::vector<float> &tilts, std::vector<float> &rolls) const = 0;
            CV_WRAP virtual cv::String getDefaultName() const CV_OVERRIDE;
            CV_WRAP virtual const cv::Ptr<cv::Feature2D> getBackendPtr() = 0;
            CV_WRAP virtual void setImgInfoStr(const std::string &img_info) = 0;
        };

        typedef MATCHINGLIB_API AffineFeature AffineFeatureDetector;
        typedef MATCHINGLIB_API AffineFeature AffineDescriptorExtractor;

        class MATCHINGLIB_API ORB : public Feature2DAsync
        {
        public:
            enum ScoreType
            {
                HARRIS_SCORE = 0,
                FAST_SCORE = 1
            };
            CV_WRAP static cv::Ptr<ORB> create(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31,
                                               int firstLevel = 0, int WTA_K = 2, ORB::ScoreType scoreType = ORB::HARRIS_SCORE, int patchSize = 31, int fastThreshold = 20, bool blurForDescriptor = false);

            CV_WRAP virtual int getMaxFeatures() const = 0;
            CV_WRAP virtual double getScaleFactor() const = 0;
            CV_WRAP virtual int getNLevels() const = 0;
            CV_WRAP virtual int getEdgeThreshold() const = 0;
            CV_WRAP virtual int getFirstLevel() const = 0;
            CV_WRAP virtual int getWTA_K() const = 0;
            CV_WRAP virtual ORB::ScoreType getScoreType() const = 0;
            CV_WRAP virtual int getPatchSize() const = 0;

            CV_WRAP virtual void setBlurForDescriptor(bool blurForDescriptor) = 0;
            CV_WRAP virtual bool getBlurForDescriptor() const = 0;

            CV_WRAP virtual void setFastThreshold(int fastThreshold) = 0;
            CV_WRAP virtual int getFastThreshold() const = 0;
            CV_WRAP virtual cv::String getDefaultName() const CV_OVERRIDE;
        };

#ifdef WITH_AKAZE_CUDA
        class MATCHINGLIB_API AKAZE : public Feature2DAsync
        {
        public:
            enum AKAZEDIFFUSIVITY_TYPE
            {
                PM_G1 = 0,
                PM_G2 = 1,
                WEICKERT = 2,
                CHARBONNIER = 3
            };
            CV_WRAP static cv::Ptr<AKAZE> create(int imgWidth, int imgHeight, int omax = 4, int nsublevels = 4, float soffset = 1.6f, float derivative_factor = 1.5f, float sderivatives = 1.f, AKAZE::AKAZEDIFFUSIVITY_TYPE diffusivity = AKAZE::AKAZEDIFFUSIVITY_TYPE::PM_G2, float dthreshold = 0.001f, float min_dthreshold = 0.00001f, int descriptor_size = 0, int descriptor_channels = 3, int descriptor_pattern_size = 10, float kcontrast = 0.001f, float kcontrast_percentile = 0.7f, size_t kcontrast_nbins = 300, int ncudaimages = 4, int maxkeypoints = 16 * 8192, bool verbosity = false);

            CV_WRAP virtual cv::String getDefaultName() const CV_OVERRIDE;

            // Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
            CV_WRAP virtual void setMaxOctave(int omax) = 0;
            CV_WRAP virtual int getMaxOctave() const = 0;

            // Default number of sublevels per scale level
            CV_WRAP virtual void setNrSublevels(int nsublevels) = 0;
            CV_WRAP virtual int getNrSublevels() const = 0;

            // Width and height of the input image
            CV_WRAP virtual void setImgDimensions(int img_width, int img_height) = 0;
            CV_WRAP virtual int getImgWidth() const = 0;
            CV_WRAP virtual int getImgHeight() const = 0;

            // Base scale offset (sigma units)
            CV_WRAP virtual void setScaleOffset(float soffset) = 0;
            CV_WRAP virtual float getScaleOffset() const = 0;

            // Factor for the multiscale derivatives
            CV_WRAP virtual void setDerivativeFactor(float derivative_factor) = 0;
            CV_WRAP virtual float getDerivativeFactor() const = 0;

            // Smoothing factor for the derivatives
            CV_WRAP virtual void setDerivatSmooth(float sderivatives) = 0;
            CV_WRAP virtual float getDerivatSmooth() const = 0;

            // Diffusivity type
            CV_WRAP virtual void setDiffusivityType(AKAZE::AKAZEDIFFUSIVITY_TYPE diffusivity) = 0;
            CV_WRAP virtual AKAZE::AKAZEDIFFUSIVITY_TYPE getDiffusivityType() const = 0;

            // Detector response threshold to accept point
            CV_WRAP virtual void setDetectResponseTh(float dthreshold) = 0;
            CV_WRAP virtual float getDetectResponseTh() const = 0;

            // Minimum detector threshold to accept a point
            CV_WRAP virtual void setMinDetectResponseTh(float min_dthreshold) = 0;
            CV_WRAP virtual float getMinDetectResponseTh() const = 0;

            // Size of the descriptor in bits. 0->Full size
            CV_WRAP virtual void setDescriptorSize(int descriptor_size) = 0;
            CV_WRAP virtual int getDescriptorSize() const = 0;

            // Number of channels in the descriptor (1, 2, 3)
            CV_WRAP virtual void setNrChannels(int descriptor_channels) = 0;
            CV_WRAP virtual int getNrChannels() const = 0;

            // Actual patch size is 2*pattern_size*point.scale
            CV_WRAP virtual void setPatternSize(int descriptor_pattern_size) = 0;
            CV_WRAP virtual int getPatternSize() const = 0;

            // The contrast factor parameter
            CV_WRAP virtual void setContrastFactor(float kcontrast) = 0;
            CV_WRAP virtual float getContrastFactor() const = 0;

            // Percentile level for the contrast factor
            CV_WRAP virtual void setPercentileLevel(float kcontrast_percentile) = 0;
            CV_WRAP virtual float getPercentileLevel() const = 0;

            // Number of bins for the contrast factor histogram
            CV_WRAP virtual void setNrBins(size_t kcontrast_nbins) = 0;
            CV_WRAP virtual size_t getNrBins() const = 0;

            // Set to true for displaying verbosity information
            CV_WRAP virtual void setVerbosity(bool verbosity) = 0;
            CV_WRAP virtual bool getVerbosity() const = 0;

            // Number of CUDA images allocated per octave
            CV_WRAP virtual void setNrCudaImages(int ncudaimages) = 0;
            CV_WRAP virtual int getNrCudaImages() const = 0;

            // Maximum number of keypoints allocated
            CV_WRAP virtual void setMaxNrKeypoints(int maxkeypoints, const bool adaptThreshold = true) = 0;
            CV_WRAP virtual int getMaxNrKeypoints() const = 0;

            // Info about image
            CV_WRAP virtual void setImageInfo(const std::string &info, const bool append = true) = 0;
            CV_WRAP virtual std::string getImageInfo() const = 0;

            // // Get byte multiplier for image area which states allocated GPU memory
            // CV_WRAP virtual size_t getByteMultiplierGPU() const = 0;
            // // Get additional number of bytes of allocated GPU memory
            // CV_WRAP virtual size_t getByteAddGPU() const = 0;
        };
#endif
    }
}
