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

// #define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_affine_feature.h>
// #include <utils_common.h>

// #include <iostream>
#include <thread>
#include <mutex>

#include "opencv2/cudafilters.hpp"
#include "opencv2/core/utils/trace.hpp"
#include "opencv2/cudawarping.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#ifdef WITH_AKAZE_CUDA
#include <AKAZE.h>
#endif

// #include "cuda.h"
// #include "cuda_runtime_api.h"

//Checks, if determinants, etc. are too close to 0
template <class T>
inline bool nearZero(const T d, const double EPSILON = 1e-4)
{
    return (static_cast<double>(d) < EPSILON) && (static_cast<double>(d) > -EPSILON);
}

#define AFFINE_USE_MULTIPLE_THREADS 1

/* values for DEBUG_SHOW_KEYPOINTS_WARPED:
 *   1 ... Show in window
 *   2 ... Save warped image with keypoints to disk
 *   3 ... Save warped image without keypoints to disk
 */
#define DEBUG_SHOW_KEYPOINTS_WARPED 0

#define WRITE1_OR_READ2_DEBUG_VALUES_AFFINE 0
#if WRITE1_OR_READ2_DEBUG_VALUES_AFFINE || DEBUG_SHOW_KEYPOINTS_WARPED > 1
#include "dbg_helpers.h"
#endif

using namespace std;

namespace matchinglib
{
    void visualizeKeypoints(const cv::Mat &img, const std::vector<cv::KeyPoint> &keypoints, const double &tilt, const double &roll, const std::string *img_name = nullptr);

#if WRITE1_OR_READ2_DEBUG_VALUES_AFFINE && WRITE1_OR_READ2_DEBUG_VALUES
    double getMatSum(const cv::Mat &m){
        return cv::sum(m)[0];
    }
#endif

    namespace cuda
    {
        Feature2DAsync::~Feature2DAsync()
        {
        }

        void Feature2DAsync::detectAsync(cv::InputArray image,
                                         std::vector<cv::KeyPoint> &keypoints,
                                         cv::InputArray mask,
                                         cv::cuda::Stream &stream)
        {
            if (image.empty())
            {
                keypoints.clear();
                return;
            }

            detectAndComputeAsync(image, mask, keypoints, cv::noArray(), false, stream);
        }

        void Feature2DAsync::detectAsync(cv::InputArray image,
                                         std::vector<cv::KeyPoint> &keypoints,
                                         std::mt19937 &mt,
                                         cv::InputArray mask,
                                         cv::cuda::Stream &stream)
        {
            if (image.empty())
            {
                keypoints.clear();
                return;
            }

            detectAndComputeAsync(image, mask, keypoints, cv::noArray(), mt, false, stream);
        }

        void Feature2DAsync::computeAsync(cv::InputArray image,
                                          std::vector<cv::KeyPoint> &keypoints,
                                          cv::OutputArray descriptors,
                                          cv::cuda::Stream &stream)
        {
            if (image.empty())
            {
                descriptors.release();
                return;
            }

            detectAndComputeAsync(image, cv::noArray(), keypoints, descriptors, true, stream);
        }

        void Feature2DAsync::computeAsync(cv::InputArray image,
                                          std::vector<cv::KeyPoint> &keypoints,
                                          cv::OutputArray descriptors,
                                          std::mt19937 &mt,
                                          cv::cuda::Stream &stream)
        {
            if (image.empty())
            {
                descriptors.release();
                return;
            }

            detectAndComputeAsync(image, cv::noArray(), keypoints, descriptors, mt, true, stream);
        }

        void Feature2DAsync::detectAndComputeAsync(cv::InputArray /*image*/,
                                                   cv::InputArray /*mask*/,
                                                   std::vector<cv::KeyPoint> & /*keypoints*/,
                                                   cv::OutputArray /*descriptors*/,
                                                   bool /*useProvidedKeypoints*/,
                                                   cv::cuda::Stream & /*stream*/)
        {
            CV_Error(cv::Error::StsNotImplemented, "");
        }

        void Feature2DAsync::detectAndComputeAsync(cv::InputArray /*image*/,
                                                   cv::InputArray /*mask*/,
                                                   std::vector<cv::KeyPoint> & /*keypoints*/,
                                                   cv::OutputArray /*descriptors*/,
                                                   std::mt19937 & /*mt*/,
                                                   bool /*useProvidedKeypoints*/,
                                                   cv::cuda::Stream & /*stream*/)
        {
            CV_Error(cv::Error::StsNotImplemented, "");
        }

        class AffineFeature_Impl CV_FINAL : public AffineFeature
        {
        public:
            explicit AffineFeature_Impl(const cv::Ptr<Feature2DAsync> &backend,
                                        int maxTilt, int minTilt, float tiltStep, float rotateStepBase, bool parallelize, int outerThreadCnt);

            int descriptorSize() const CV_OVERRIDE
            {
                return backend_->descriptorSize();
            }

            int descriptorType() const CV_OVERRIDE
            {
                return backend_->descriptorType();
            }

            int defaultNorm() const CV_OVERRIDE
            {
                return backend_->defaultNorm();
            }

            void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                  cv::OutputArray descriptors, bool useProvidedKeypoints = false) CV_OVERRIDE;
            void detectAndComputeAsync(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                       cv::OutputArray descriptors, bool useProvidedKeypoints, cv::cuda::Stream &stream) CV_OVERRIDE;
            void detectAndComputeAsync(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                       cv::OutputArray descriptors, std::mt19937 &mt, bool useProvidedKeypoints, cv::cuda::Stream &stream) CV_OVERRIDE;

            void setViewParams(const std::vector<float> &tilts, const std::vector<float> &rolls) CV_OVERRIDE;
            void getViewParams(std::vector<float> &tilts, std::vector<float> &rolls) const CV_OVERRIDE;

            const cv::Ptr<cv::Feature2D> getBackendPtr() CV_OVERRIDE;

            void setImgInfoStr(const std::string &img_info) CV_OVERRIDE;

            virtual size_t getByteMultiplierGPU() const CV_OVERRIDE { return byte_multiplier; }
            virtual size_t getByteAddGPU() const CV_OVERRIDE { return byte_add; }

        protected:
            void splitKeypointsByView(const std::vector<cv::KeyPoint> &keypoints_,
                                      std::vector<std::vector<cv::KeyPoint>> &keypointsByView) const;

            const cv::Ptr<Feature2DAsync> backend_;
            int maxTilt_;
            int minTilt_;
            float tiltStep_;
            float rotateStepBase_;
            bool parallelize_;
            int outerThreadCnt_;

            // Tilt factors.
            std::vector<float> tilts_;
            // Roll factors.
            std::vector<float> rolls_;

        private:
            AffineFeature_Impl(const AffineFeature_Impl &);            // copy disabled
            AffineFeature_Impl &operator=(const AffineFeature_Impl &); // assign disabled
            void calcByteMultiplier();
            void calcByteAdd();
            size_t byte_multiplier = 1;
            size_t byte_add = 0;
            std::string img_dbg_info;
        };

        AffineFeature_Impl::AffineFeature_Impl(const cv::Ptr<Feature2DAsync> &backend,
                                               int maxTilt, int minTilt, float tiltStep, float rotateStepBase, bool parallelize, int outerThreadCnt)
            : backend_(backend), maxTilt_(maxTilt), minTilt_(minTilt), tiltStep_(tiltStep), rotateStepBase_(rotateStepBase), parallelize_(parallelize), outerThreadCnt_(outerThreadCnt)
        {
            int i = minTilt_;
            if (i == 0)
            {
                tilts_.push_back(1);
                rolls_.push_back(0);
                i++;
            }
            float tilt = 1;
            for (; i <= maxTilt_; i++)
            {
                tilt *= tiltStep_;
                float rotateStep = rotateStepBase_ / tilt;
                int rollN = cvFloor(180.0f / rotateStep);
                if (nearZero(rollN * rotateStep - 180.0f))
                    rollN--;
                for (int j = 0; j <= rollN; j++)
                {
                    tilts_.push_back(tilt);
                    rolls_.push_back(rotateStep * j);
                }
            }
            calcByteMultiplier();
            calcByteAdd();
        }

        void AffineFeature_Impl::setViewParams(const std::vector<float> &tilts,
                                               const std::vector<float> &rolls)
        {
            CV_Assert(tilts.size() == rolls.size());
            tilts_ = tilts;
            rolls_ = rolls;
        }

        void AffineFeature_Impl::getViewParams(std::vector<float> &tilts,
                                               std::vector<float> &rolls) const
        {
            tilts = tilts_;
            rolls = rolls_;
        }

        const cv::Ptr<cv::Feature2D> AffineFeature_Impl::getBackendPtr()
        {
            return backend_;
        }

        void AffineFeature_Impl::setImgInfoStr(const std::string &img_info)
        {
            img_dbg_info = img_info;
        }

        void AffineFeature_Impl::splitKeypointsByView(const std::vector<cv::KeyPoint> &keypoints_,
                                                      std::vector<std::vector<cv::KeyPoint>> &keypointsByView) const
        {
            for (size_t i = 0; i < keypoints_.size(); i++)
            {
                const cv::KeyPoint &kp = keypoints_[i];
                CV_Assert(kp.class_id >= 0 && kp.class_id < (int)tilts_.size());
                keypointsByView[kp.class_id].push_back(kp);
            }
        }

        void AffineFeature_Impl::calcByteMultiplier()
        {
            double s = static_cast<double>(tilts_.size()) * 1.5 * 1.5 * static_cast<double>(backend_->getByteMultiplierGPU());
            byte_multiplier = static_cast<size_t>(std::ceil(s));
        }

        void AffineFeature_Impl::calcByteAdd()
        {
            byte_add = tilts_.size() * backend_->getByteAddGPU();
        }

#if WRITE1_OR_READ2_DEBUG_VALUES_AFFINE && WRITE1_OR_READ2_DEBUG_VALUES
        mutex m_sum_skew, m_nrMaskPix, m_imgSizeHash;
#endif
        class skewedDetectAndCompute
        {
        public:
            skewedDetectAndCompute(
                const std::vector<float> &_tilts,
                const std::vector<float> &_rolls,
                std::vector<std::vector<cv::KeyPoint>> &_keypointsCollection,
                std::vector<cv::Mat> &_descriptorCollection,
                std::shared_ptr<std::vector<cv::cuda::HostMem>> _image,
                std::shared_ptr<std::vector<cv::cuda::HostMem>> _mask,
                const bool _do_keypoints,
                const bool _do_descriptors,
                const cv::Ptr<Feature2DAsync> &_backend,
                const bool _isNotParallel)
                : tilts(_tilts),
                  rolls(_rolls),
                  keypointsCollection(_keypointsCollection),
                  descriptorCollection(_descriptorCollection),
                  image(_image),
                  mask(_mask),
                  do_keypoints(_do_keypoints),
                  do_descriptors(_do_descriptors),
                  backend(_backend),
                  isNotParallel(_isNotParallel)
            {
            }

            // void operator()(const cv::Range &range) const
            // {
            //     std::random_device rd;
            //     std::mt19937 g(rd());
            //     operator()(range, g);
            // }

            void operator()(const cv::Range &range, std::mt19937 &mt) const
            {
                CV_TRACE_FUNCTION();

                const int begin = range.start;
                const int end = range.end;
                cv::cuda::Stream streamA;

                for (int a = begin; a < end; a++)
                {
                    cv::Mat warpedImage, warpedMask;
                    cv::Matx23f pose, invPose;
                    int streamIdx = isNotParallel ? 0 : a;
                    cv::cuda::HostMem maskMat;
                    if(!mask->empty()){
                        maskMat = mask->at(streamIdx);
                    }

                    cv::Size warped_size = getWarpedImgSize(tilts[a], rolls[a], image->at(streamIdx).size());
                    warpedImage = cv::Mat(warped_size, image->at(streamIdx).type());
                    if (!nearZero(rolls[a]) || !nearZero(tilts[a] - 1.f))
                    {
                        warpedMask = cv::Mat(warped_size, CV_8UC1);
                    }
                    else if (!mask->empty())
                    {
                        maskMat.createMatHeader().copyTo(warpedMask);
                    }

                    affineSkew(tilts[a], rolls[a], image->at(streamIdx), maskMat, warpedImage, warpedMask, pose, streamA);
                    streamA.waitForCompletion();
                    cv::invertAffineTransform(pose, invPose);

                    std::vector<cv::KeyPoint> wKeypoints;
                    cv::Mat wDescriptors;
                    if (!do_keypoints)
                    {
                        const std::vector<cv::KeyPoint> &keypointsInView = keypointsCollection[a];
                        if (keypointsInView.size() == 0) // when there are no keypoints in this affine view
                            continue;

                        std::vector<cv::Point2f> pts_, pts;
                        cv::KeyPoint::convert(keypointsInView, pts_);
                        cv::transform(pts_, pts, pose);
                        wKeypoints.resize(keypointsInView.size());
                        for (size_t wi = 0; wi < wKeypoints.size(); wi++)
                        {
                            wKeypoints[wi] = keypointsInView[wi];
                            wKeypoints[wi].pt = pts[wi];
                        }
                    }

#if WRITE1_OR_READ2_DEBUG_VALUES_AFFINE && WRITE1_OR_READ2_DEBUG_VALUES
                    cv::Ptr<matchinglib::cuda::AKAZE> akaze_ptr = backend.dynamicCast<matchinglib::cuda::AKAZE>();
                    std::string info_affine, *info_affine_ptr = nullptr;
                    if (akaze_ptr)
                    {
                        info_affine = akaze_ptr->getImageInfo();
                        info_affine += "_" + std::to_string(static_cast<int>(std::round(tilts[a] * 100.f)));
                        info_affine += "_" + std::to_string(static_cast<int>(std::round(rolls[a] * 100.f)));
                        info_affine_ptr = &info_affine;
                    }

                    double imgsum = getMatSum(warpedImage);
                    std::string msg_out = readWritePrintDbgVal("sum_skew", m_sum_skew, imgsum, info_affine_ptr);
                    if (!msg_out.empty())
                    {
                        printDbgMsgLine(1, msg_out.c_str());
                    }

                    int nrMaskPix = warpedMask.empty() ? 0 : cv::countNonZero(warpedMask);
                    msg_out = readWritePrintDbgVal("nrMaskPix", m_nrMaskPix, nrMaskPix, info_affine_ptr);
                    if (!msg_out.empty())
                    {
                        printDbgMsgLine(1, msg_out.c_str());
                    }

                    size_t hash_seed = 0;
                    hash_combine(hash_seed, warpedImage.rows);
                    hash_combine(hash_seed, warpedImage.cols);
                    msg_out = readWritePrintDbgVal("imgSizeHash", m_imgSizeHash, hash_seed, info_affine_ptr);
                    if (!msg_out.empty())
                    {
                        printDbgMsgLine(1, msg_out.c_str());
                    }
#endif

                    //Only use with single thread
                    std::string *info_old_ptr = nullptr;
#if AFFINE_USE_MULTIPLE_THREADS == 0 || DEBUG_SHOW_KEYPOINTS_WARPED > 1
                    cv::Ptr<matchinglib::cuda::AKAZE> akaze_ptr = backend.dynamicCast<matchinglib::cuda::AKAZE>();
                    std::string info_old;
                    if (akaze_ptr){
                        info_old = akaze_ptr->getImageInfo();
                        info_old_ptr = &info_old;
                        std::string info = "_" + std::to_string(static_cast<int>(std::round(tilts[a] * 100.f)));
                        info += "_" + std::to_string(static_cast<int>(std::round(rolls[a] * 100.f)));
                        akaze_ptr->setImageInfo(info, true);
                    }
#endif

                    backend->detectAndComputeAsync(warpedImage, warpedMask, wKeypoints, wDescriptors, mt, !do_keypoints, streamA);
                    streamA.waitForCompletion();

#if AFFINE_USE_MULTIPLE_THREADS == 0 || DEBUG_SHOW_KEYPOINTS_WARPED > 1
                    if (akaze_ptr)
                    {
                        akaze_ptr->setImageInfo(info_old, false);
                    }
#endif

#if DEBUG_SHOW_KEYPOINTS_WARPED
                    visualizeKeypoints(warpedImage, wKeypoints, tilts[a], rolls[a], info_old_ptr);
#endif
                    if (do_keypoints)
                    {
                        // KeyPointsFilter::runByPixelsMask( wKeypoints, warpedMask );
                        if (wKeypoints.size() == 0)
                        {
                            keypointsCollection[a].clear();
                            continue;
                        }
                        std::vector<cv::Point2f> pts_, pts;
                        cv::KeyPoint::convert(wKeypoints, pts_);
                        cv::transform(pts_, pts, invPose);

                        keypointsCollection[a].resize(wKeypoints.size());
                        for (size_t wi = 0; wi < wKeypoints.size(); wi++)
                        {
                            keypointsCollection[a][wi] = wKeypoints[wi];
                            keypointsCollection[a][wi].pt = pts[wi];
                            keypointsCollection[a][wi].class_id = a;
                        }
                    }
                    if (do_descriptors && !wDescriptors.empty())
                    {
                        wDescriptors.copyTo(descriptorCollection[a]);
                    }
                }
            }

        private:
            void affineSkew(float tilt, float phi, cv::cuda::HostMem &image_in, cv::cuda::HostMem &mask_in, cv::Mat &warpedImage, cv::Mat &warpedMask, cv::Matx23f &pose, cv::cuda::Stream &stream) const
            {
                std::scoped_lock lock(m_skew);
                int h = image_in.size().height;
                int w = image_in.size().width;
                cv::cuda::GpuMat gpu_image, gpu_rotImage, gpu_mask;
#define USE_IMG_HOSTMEM 0
#if USE_IMG_HOSTMEM
                cv::cuda::HostMem warpedImage_host;
#endif

                cv::cuda::GpuMat mask0;
                cv::Mat mask0_;
                cv::cuda::HostMem mask0_host;
                if (mask_in.empty()){
                    mask0_ = cv::Mat(h, w, CV_8UC1, 255);
                    mask0_host = cv::cuda::HostMem(mask0_, cv::cuda::HostMem::PAGE_LOCKED);
                    cv::cuda::ensureSizeIsEnough(mask0_.size(), mask0_.type(), mask0);
                    mask0.upload(mask0_host, stream);
                }
                else{
                    mask0_host = cv::cuda::HostMem(mask_in, cv::cuda::HostMem::PAGE_LOCKED);
                    cv::cuda::ensureSizeIsEnough(mask_in.size(), mask_in.type(), mask0);
                    mask0.upload(mask0_host, stream);
                }
                pose = cv::Matx23f(1.f, 0, 0,
                                   0, 1.f, 0);

                cv::cuda::ensureSizeIsEnough(image_in.size(), image_in.type(), gpu_image);
                gpu_image.upload(image_in, stream); // RAM => GPU

                if (nearZero(phi)){
                    cv::cuda::ensureSizeIsEnough(gpu_image.size(), gpu_image.type(), gpu_rotImage);
                    gpu_image.copyTo(gpu_rotImage, stream);
                }
                else
                {
                    phi = phi * (float)CV_PI / 180.f;
                    float s = std::sin(phi);
                    float c = std::cos(phi);
                    cv::Matx22f A(c, -s, s, c);
                    cv::Matx<float, 4, 2> corners(0, 0, (float)w, 0, (float)w, (float)h, 0, (float)h);
                    cv::Mat tf(corners * A.t());
                    cv::Mat tcorners;
                    tf.convertTo(tcorners, CV_32S);
                    cv::Rect rect = cv::boundingRect(tcorners);
                    h = rect.height;
                    w = rect.width;
                    pose = cv::Matx23f(c, -s, -(float)rect.x,
                                       s, c, -(float)rect.y);
                    cv::cuda::ensureSizeIsEnough(cv::Size(w, h), gpu_image.type(), gpu_rotImage);
                    cv::cuda::warpAffine(gpu_image, gpu_rotImage, pose, cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_REPLICATE, cv::Scalar(), stream);
                }
                if (nearZero(tilt - 1.f)){
#if USE_IMG_HOSTMEM
                    warpedImage_host = cv::cuda::HostMem(gpu_rotImage.size(), gpu_rotImage.type(), cv::cuda::HostMem::PAGE_LOCKED);
                    gpu_rotImage.download(warpedImage_host, stream);
                    stream.waitForCompletion();
                    warpedImage_host.createMatHeader().copyTo(warpedImage);
#else
                    cv::cuda::registerPageLocked(warpedImage);
                    gpu_rotImage.download(warpedImage, stream);
                    stream.waitForCompletion();
                    cv::cuda::unregisterPageLocked(warpedImage);
#endif
                    
                }
                else
                {
                    float s = 0.8f * sqrt(tilt * tilt - 1.f);
                    cv::cuda::GpuMat gpu_im_warp;
                    cv::Size kSize(cvRound(s * 6.f + 1.f) | 1, cvRound(0.01f * 6.f + 1.f) | 1);
                    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(gpu_rotImage.type(), gpu_rotImage.type(), kSize, s, 0.01);
                    filter->apply(gpu_rotImage, gpu_rotImage, stream);
                    cv::Size destSi = gpu_rotImage.size();
                    destSi.width = static_cast<int>(std::round(static_cast<float>(destSi.width) / tilt));
                    cv::cuda::ensureSizeIsEnough(destSi, gpu_rotImage.type(), gpu_im_warp);
                    cv::cuda::resize(gpu_rotImage, gpu_im_warp, cv::Size(0, 0), 1.0 / tilt, 1.0, cv::INTER_NEAREST, stream);
#if USE_IMG_HOSTMEM
                    warpedImage_host = cv::cuda::HostMem(gpu_im_warp.size(), gpu_im_warp.type(), cv::cuda::HostMem::PAGE_LOCKED);
                    gpu_im_warp.download(warpedImage_host, stream); // GPU => RAM
                    stream.waitForCompletion();
                    warpedImage_host.createMatHeader().copyTo(warpedImage);
#else
                    cv::cuda::registerPageLocked(warpedImage);
                    gpu_im_warp.download(warpedImage, stream); // GPU => RAM
                    stream.waitForCompletion();
                    cv::cuda::unregisterPageLocked(warpedImage);
#endif
                    pose(0, 0) /= tilt;
                    pose(0, 1) /= tilt;
                    pose(0, 2) /= tilt;
                }
                if (!nearZero(phi) || !nearZero(tilt - 1.f))
                {
                    cv::cuda::GpuMat gpu_warpedMask;
                    cv::cuda::ensureSizeIsEnough(warpedImage.size(), mask0.type(), gpu_warpedMask);
                    cv::cuda::warpAffine(mask0, gpu_warpedMask, pose, warpedImage.size(), cv::INTER_NEAREST, 0, cv::Scalar(), stream);
                    mask0_host = cv::cuda::HostMem(warpedImage.size(), mask0.type(), cv::cuda::HostMem::PAGE_LOCKED);
                    gpu_warpedMask.download(mask0_host, stream); // GPU => RAM
                    stream.waitForCompletion();
                    mask0_host.createMatHeader().copyTo(warpedMask);
                }
            }

            cv::Size getWarpedImgSize(const float &tilt, float phi, const cv::Size &inp_size) const
            {
                cv::Size warped_size = inp_size;
                if (!nearZero(phi))
                {
                    phi = phi * (float)CV_PI / 180.f;
                    float s = std::sin(phi);
                    float c = std::cos(phi);
                    cv::Matx22f A(c, -s, s, c);
                    cv::Matx<float, 4, 2> corners(0, 0, (float)inp_size.width, 0, (float)inp_size.width, (float)inp_size.height, 0, (float)inp_size.height);
                    cv::Mat tf(corners * A.t());
                    cv::Mat tcorners;
                    tf.convertTo(tcorners, CV_32S);
                    cv::Rect rect = cv::boundingRect(tcorners);
                    warped_size.height = rect.height;
                    warped_size.width = rect.width;
                }
                if (!nearZero(tilt - 1.f))
                {
                    warped_size.width = static_cast<int>(std::round(static_cast<float>(warped_size.width) / tilt));
                }
                return warped_size;
            }

            const std::vector<float> &tilts;
            const std::vector<float> &rolls;
            std::vector<std::vector<cv::KeyPoint>> &keypointsCollection;
            std::vector<cv::Mat> &descriptorCollection;
            std::shared_ptr<std::vector<cv::cuda::HostMem>> image;
            std::shared_ptr<std::vector<cv::cuda::HostMem>> mask;
            const bool do_keypoints;
            const bool do_descriptors;
            const cv::Ptr<Feature2DAsync> &backend;
            bool isNotParallel = false;
            mutable std::mutex m_skew;
        };

        void AffineFeature_Impl::detectAndCompute(cv::InputArray _image, cv::InputArray _mask,
                                                  std::vector<cv::KeyPoint> &keypoints,
                                                  cv::OutputArray _descriptors,
                                                  bool useProvidedKeypoints)
        {
            CV_TRACE_FUNCTION();
            std::mt19937 g(0);

            detectAndComputeAsync(_image, _mask, keypoints, _descriptors, g, useProvidedKeypoints, cv::cuda::Stream::Null());
        }

        void AffineFeature_Impl::detectAndComputeAsync(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint> &keypoints,
                                                       cv::OutputArray _descriptors, bool useProvidedKeypoints, cv::cuda::Stream &stream)
        {
            std::random_device rd;
            std::mt19937 g(rd());
            detectAndComputeAsync(_image, _mask, keypoints, _descriptors, g, useProvidedKeypoints, stream);
        }

#if WRITE1_OR_READ2_DEBUG_VALUES_AFFINE && WRITE1_OR_READ2_DEBUG_VALUES
        std::mutex m_hash_sum_affine_kp_xy;
#endif
        void AffineFeature_Impl::detectAndComputeAsync(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint> &keypoints,
                                                       cv::OutputArray _descriptors, std::mt19937 &mt, bool useProvidedKeypoints, cv::cuda::Stream &stream)
        {
            bool do_keypoints = !useProvidedKeypoints;
            bool do_descriptors = _descriptors.needed();
            const cv::Mat image = _image.getMat(), mask = _mask.getMat();
            std::shared_ptr<std::vector<cv::cuda::HostMem>> srcMemArray = std::make_shared<std::vector<cv::cuda::HostMem>>();
            std::shared_ptr<std::vector<cv::cuda::HostMem>> srcMemMaskArray = std::make_shared<std::vector<cv::cuda::HostMem>>();

            cv::Mat descriptors;

            if ((!do_keypoints && !do_descriptors) || _image.empty()) return;

            std::vector<std::vector<cv::KeyPoint>> keypointsCollection(tilts_.size());
            std::vector<cv::Mat> descriptorCollection(tilts_.size());

            if (do_keypoints)
                keypoints.clear();
            else
                splitKeypointsByView(keypoints, keypointsCollection);

#if AFFINE_USE_MULTIPLE_THREADS
            int maxParallel = std::max(getMaxNrGpuThreadsFromMemoryUsage(image, backend_->getByteMultiplierGPU(), backend_->getByteAddGPU()) / outerThreadCnt_, 1);
#else
            int maxParallel = 1;
#endif            
            if(maxParallel == 1){
                parallelize_ = false;
            }
#if DEBUG_SHOW_KEYPOINTS_WARPED
            parallelize_ = false;
#endif

            if (parallelize_)
            {
                for (size_t i = 0; i < tilts_.size(); i++)
                {
                    //Initialize Pinned Memory with input image
                    cv::cuda::HostMem srcHostMem = cv::cuda::HostMem(image, cv::cuda::HostMem::PAGE_LOCKED);
                    srcMemArray->push_back(srcHostMem);
                    if (!mask.empty()){
                        cv::cuda::HostMem srcMaskHostMem = cv::cuda::HostMem(mask, cv::cuda::HostMem::PAGE_LOCKED);
                        srcMemMaskArray->push_back(srcMaskHostMem);
                    }
                }
                int nrTilts = static_cast<int>(tilts_.size());
                std::vector<std::thread> threads;
                const auto sdc = skewedDetectAndCompute(tilts_, rolls_,
                                                        keypointsCollection, descriptorCollection,
                                                        srcMemArray, srcMemMaskArray,
                                                        do_keypoints, do_descriptors, backend_, false);
                int batchSize = static_cast<int>(std::ceil(static_cast<float>(nrTilts) / static_cast<float>(maxParallel)));
                getThreadBatchSize(nrTilts, maxParallel, batchSize);
                //Initialize random devices
                std::vector<std::mt19937> rand_devices;
                for (int i = 0; i < maxParallel; ++i)
                {
                    rand_devices.push_back(mt);
                }
                for (int i = 0; i < maxParallel; ++i)
                {
                    const int startIdx = i * batchSize;
                    const int endIdx = std::min((i + 1) * batchSize, nrTilts);
                    threads.push_back(std::thread(std::bind(&skewedDetectAndCompute::operator(), std::ref(sdc), cv::Range(startIdx, endIdx), std::ref(rand_devices.at(i)))));
                }

                for (auto &t : threads)
                {
                    if (t.joinable())
                    {
                        t.join();
                    }
                }
            }
            else
            {
                cv::cuda::HostMem srcHostMem = cv::cuda::HostMem(image, cv::cuda::HostMem::PAGE_LOCKED);
                srcMemArray->push_back(srcHostMem);
                if (!mask.empty())
                {
                    cv::cuda::HostMem srcMaskHostMem = cv::cuda::HostMem(mask, cv::cuda::HostMem::PAGE_LOCKED);
                    srcMemMaskArray->push_back(srcMaskHostMem);
                }
                const auto sdc = skewedDetectAndCompute(tilts_, rolls_,
                                                        keypointsCollection, descriptorCollection,
                                                        srcMemArray, srcMemMaskArray,
                                                        do_keypoints, do_descriptors, backend_, true);
                sdc(cv::Range(0, (int)tilts_.size()), mt);
            }

            if (do_keypoints)
            {
#if WRITE1_OR_READ2_DEBUG_VALUES_AFFINE && WRITE1_OR_READ2_DEBUG_VALUES
                size_t hash_sum_xy = 0;
#endif
                if(!img_dbg_info.empty()){
                    
                }
                // std::cout << "sum_x_aff: ";
                for (size_t i = 0; i < keypointsCollection.size(); i++)
                {
                    const std::vector<cv::KeyPoint> &keys = keypointsCollection[i];
                    keypoints.insert(keypoints.end(), keys.begin(), keys.end());
#if WRITE1_OR_READ2_DEBUG_VALUES_AFFINE && WRITE1_OR_READ2_DEBUG_VALUES
                    if (!keys.empty()){
                        size_t sum_x = 0, sum_y = 0;
                        for (const auto &kpi : keys)
                        {
                            sum_x += static_cast<size_t>(std::round(kpi.pt.x));
                            sum_y += static_cast<size_t>(std::round(kpi.pt.y));
                        }
                        poselib::hash_combine(hash_sum_xy, sum_x);
                        poselib::hash_combine(hash_sum_xy, sum_y);
                    }
#endif
                }
#if WRITE1_OR_READ2_DEBUG_VALUES_AFFINE && WRITE1_OR_READ2_DEBUG_VALUES
                std::string info_affine, *info_affine_ptr = nullptr;
                if (!img_dbg_info.empty())
                {
                    info_affine = img_dbg_info;
                    info_affine_ptr = &info_affine;
                }
                std::string msg_out = readWritePrintDbgVal("hash_sum_affine_kp_xy", m_hash_sum_affine_kp_xy, hash_sum_xy, info_affine_ptr);
                if (!msg_out.empty())
                {
                    printDbgMsgLine(1, msg_out.c_str());
                }
#endif
            }

            if (do_descriptors)
            {
                _descriptors.create((int)keypoints.size(), backend_->descriptorSize(), backend_->descriptorType());
                descriptors = _descriptors.getMat();
                int iter = 0;
                for (size_t i = 0; i < descriptorCollection.size(); i++)
                {
                    const cv::Mat &descs = descriptorCollection[i];
                    if (descs.empty())
                        continue;
                    cv::Mat roi(descriptors, cv::Rect(0, iter, descriptors.cols, descs.rows));
                    descs.copyTo(roi);
                    iter += descs.rows;
                }
            }
        }

        cv::Ptr<AffineFeature> AffineFeature::create(const cv::Ptr<cv::Feature2D> &backend,
                                                     int maxTilt, int minTilt, float tiltStep, float rotateStepBase, bool parallelize, int outerThreadCnt)
        {
            CV_Assert(minTilt < maxTilt);
            CV_Assert(tiltStep > 0);
            CV_Assert(rotateStepBase > 0);
            return cv::makePtr<AffineFeature_Impl>(backend.dynamicCast<Feature2DAsync>(), maxTilt, minTilt, tiltStep, rotateStepBase, parallelize, outerThreadCnt);
        }

        cv::String AffineFeature::getDefaultName() const
        {
            return (cv::Feature2D::getDefaultName() + ".CudaAffineFeature");
        }

        // ---------------------- ORB ----------------------
        class ORB_Impl CV_FINAL : public ORB
        {
        public:
            explicit ORB_Impl(int _nfeatures, float _scaleFactor, int _nlevels, int _edgeThreshold,
                              int _firstLevel, int _WTA_K, ORB::ScoreType _scoreType, int _patchSize, int _fastThreshold, bool _blurForDescriptor) : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels), edgeThreshold(_edgeThreshold), firstLevel(_firstLevel), wta_k(_WTA_K), scoreType(_scoreType), patchSize(_patchSize), fastThreshold(_fastThreshold), blurForDescriptor(_blurForDescriptor)
            {
                calcByteMultiplier();
                calcByteAdd();
            }

            int getMaxFeatures() const CV_OVERRIDE { return nfeatures; }

            double getScaleFactor() const CV_OVERRIDE { return scaleFactor; }

            int getNLevels() const CV_OVERRIDE { return nlevels; }

            int getEdgeThreshold() const CV_OVERRIDE { return edgeThreshold; }

            int getFirstLevel() const CV_OVERRIDE { return firstLevel; }

            int getWTA_K() const CV_OVERRIDE { return wta_k; }

            ORB::ScoreType getScoreType() const CV_OVERRIDE { return scoreType; }

            int getPatchSize() const CV_OVERRIDE { return patchSize; }

            void setBlurForDescriptor(bool blurForDescriptor_) CV_OVERRIDE { 
                blurForDescriptor = blurForDescriptor_;
            }
            bool getBlurForDescriptor() const CV_OVERRIDE { return blurForDescriptor; }

            void setFastThreshold(int fastThreshold_) CV_OVERRIDE {
                fastThreshold = fastThreshold_;
            }

            int getFastThreshold() const CV_OVERRIDE { return fastThreshold; }

            // returns the descriptor size in bytes
            int descriptorSize() const CV_OVERRIDE {
                return cv::ORB::kBytes;
            }
            // returns the descriptor type
            int descriptorType() const CV_OVERRIDE {
                return CV_8U;
            }
            // returns the default norm type
            int defaultNorm() const CV_OVERRIDE {
                return cv::NORM_HAMMING;
            }

            // Compute the ORB_Impl features and descriptors on an image
            void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                  cv::OutputArray descriptors, bool useProvidedKeypoints = false) CV_OVERRIDE;
            void detectAndComputeAsync(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                       cv::OutputArray descriptors, bool useProvidedKeypoints, cv::cuda::Stream &stream) CV_OVERRIDE;
            void detectAndComputeAsync(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                       cv::OutputArray descriptors, std::mt19937 &mt, bool useProvidedKeypoints, cv::cuda::Stream &stream) CV_OVERRIDE;
            virtual size_t getByteMultiplierGPU() const CV_OVERRIDE { return byte_multiplier; }
            virtual size_t getByteAddGPU() const CV_OVERRIDE { return byte_add; }
            void calcByteMultiplier();
            void calcByteAdd();

        protected:
            int nfeatures;
            double scaleFactor;
            int nlevels;
            int edgeThreshold;
            int firstLevel;
            int wta_k;
            ORB::ScoreType scoreType;
            int patchSize;
            int fastThreshold;
            bool blurForDescriptor;
            size_t byte_multiplier = 1;
            size_t byte_add = 0;
        };

        void ORB_Impl::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                      cv::OutputArray descriptors, bool useProvidedKeypoints)
        {
            cv::cuda::Stream stream;
            detectAndComputeAsync(image, mask, keypoints, descriptors, useProvidedKeypoints, stream);
        }

        void ORB_Impl::detectAndComputeAsync(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                             cv::OutputArray descriptors, bool useProvidedKeypoints, cv::cuda::Stream &stream)
        {
            std::random_device rd;
            std::mt19937 g(rd());
            detectAndComputeAsync(image, mask, keypoints, descriptors, g, useProvidedKeypoints, stream);
        }

        void ORB_Impl::detectAndComputeAsync(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                             cv::OutputArray descriptors, std::mt19937 &mt, bool useProvidedKeypoints, cv::cuda::Stream &stream)
        {
            CV_Assert(!useProvidedKeypoints);
            const cv::Mat img = image.getMat();
            const cv::Mat mask_ = mask.getMat();
            cv::cuda::GpuMat gpu_img, gpu_mask, gpu_descr;
            cv::cuda::ensureSizeIsEnough(img.size(), img.type(), gpu_img);
            gpu_img.upload(img, stream);
            if (!mask_.empty())
            {
                cv::cuda::ensureSizeIsEnough(mask_.size(), mask_.type(), gpu_mask);
                gpu_mask.upload(mask_, stream);
            }

            cv::cuda::GpuMat keypoints_gpu;
            cv::Mat descr_out;
            cv::Ptr<cv::cuda::ORB> orb_cuda_ptr = cv::cuda::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, wta_k, scoreType, patchSize, fastThreshold, blurForDescriptor);
            try {
                orb_cuda_ptr->detectAndComputeAsync(gpu_img, gpu_mask, keypoints_gpu, gpu_descr, useProvidedKeypoints, stream);
                gpu_descr.download(descriptors, stream);
                if (!useProvidedKeypoints)
                {
                    orb_cuda_ptr->convert(keypoints_gpu, keypoints);
                }
                stream.waitForCompletion();
            }
            catch (cv::Exception &e) {
                std::cerr << "Exception during ORB feature detection: " << e.what() << std::endl;
            }
        }

        cv::Ptr<ORB> ORB::create(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold,
                                 int firstLevel, int wta_k, ORB::ScoreType scoreType, int patchSize, int fastThreshold, bool blurForDescriptor)
        {
            CV_Assert(firstLevel >= 0);
            return cv::makePtr<ORB_Impl>(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                         firstLevel, wta_k, scoreType, patchSize, fastThreshold, blurForDescriptor);
        }

        cv::String ORB::getDefaultName() const
        {
            return (cv::Feature2D::getDefaultName() + ".CudaORB");
        }

        void ORB_Impl::calcByteMultiplier()
        {
            double s2 = 0;
            for (int i = 0; i < nlevels; i++)
            {
                double s = 1. / std::pow(scaleFactor, i);
                s *= s;
                s2 += s;
            }
            double mult = s2;
            mult *= 4.;
            const double nbf = static_cast<double>(sizeof(float));
            mult += 2 * nbf;
            mult += s2 * 32;
            byte_multiplier = static_cast<size_t>(std::ceil(mult));
        }

        void ORB_Impl::calcByteAdd()
        {
            const size_t nbf = static_cast<size_t>(sizeof(float));
            if (wta_k == 2){
                byte_add = 1024 * nbf;
            }
            else
            {
                byte_add = static_cast<size_t>(wta_k) * 4 * 32 * nbf;
            }
        }

        // ---------------------- AKAZE ----------------------
#ifdef WITH_AKAZE_CUDA
        class AKAZE_Impl CV_FINAL : public AKAZE
        {
        public:
            explicit AKAZE_Impl(int imgWidth, int imgHeight, int _omax, int _nsublevels, float _soffset, float _derivative_factor, float _sderivatives, AKAZE::AKAZEDIFFUSIVITY_TYPE _diffusivity, float _dthreshold, float _min_dthreshold, int _descriptor_size, int _descriptor_channels, int _descriptor_pattern_size, float _kcontrast, float _kcontrast_percentile, size_t _kcontrast_nbins, int _ncudaimages, int _maxkeypoints, bool _verbosity) : diffusivity(_diffusivity)
            {
                options.omax = _omax;
                options.nsublevels = _nsublevels;
                options.soffset = _soffset;
                options.derivative_factor = _derivative_factor;
                options.sderivatives = _sderivatives;
                switch(diffusivity){
                    case AKAZE::AKAZEDIFFUSIVITY_TYPE::PM_G1:
                        options.diffusivity = DIFFUSIVITY_TYPE::PM_G1;
                        break;
                    case AKAZE::AKAZEDIFFUSIVITY_TYPE::PM_G2:
                        options.diffusivity = DIFFUSIVITY_TYPE::PM_G2;
                        break;
                    case AKAZE::AKAZEDIFFUSIVITY_TYPE::WEICKERT:
                        options.diffusivity = DIFFUSIVITY_TYPE::WEICKERT;
                        break;
                    case AKAZE::AKAZEDIFFUSIVITY_TYPE::CHARBONNIER:
                        options.diffusivity = DIFFUSIVITY_TYPE::CHARBONNIER;
                        break;
                    default:
                        options.diffusivity = DIFFUSIVITY_TYPE::PM_G1;
                        break;
                }
                options.dthreshold = _dthreshold;
                options.min_dthreshold = _min_dthreshold;
                options.descriptor_size = _descriptor_size;
                options.descriptor_channels = _descriptor_channels;
                options.descriptor_pattern_size = _descriptor_pattern_size;
                options.kcontrast = _kcontrast;
                options.kcontrast_percentile = _kcontrast_percentile;
                options.kcontrast_nbins = _kcontrast_nbins;
                options.ncudaimages = _ncudaimages;
                options.maxkeypoints = _maxkeypoints;
                options.verbosity = _verbosity;
                options.img_height = imgHeight;
                options.img_width = imgWidth;
                calcByteMultiplier();
                calcByteAdd();
            }

            void setMaxOctave(int omax_) CV_OVERRIDE { 
                options.omax = omax_;
                calcByteMultiplier();
                calcByteAdd();
            }
            int getMaxOctave() const CV_OVERRIDE { return options.omax; }
            void setNrSublevels(int nsublevels_) CV_OVERRIDE { 
                options.nsublevels = nsublevels_;
                calcByteMultiplier();
                calcByteAdd();
            }
            int getNrSublevels() const CV_OVERRIDE { return options.nsublevels; }
            int getImgWidth() const CV_OVERRIDE { return options.img_width; }
            void setImgDimensions(int img_width_, int img_height_) CV_OVERRIDE
            {
                options.img_height = img_height_;
                options.img_width = img_width_;
            }
            int getImgHeight() const CV_OVERRIDE { return options.img_height; }
            void setScaleOffset(float soffset_) CV_OVERRIDE { 
                options.soffset = soffset_;
            }
            float getScaleOffset() const CV_OVERRIDE { return options.soffset; }
            void setDerivativeFactor(float derivative_factor_) CV_OVERRIDE { 
                options.derivative_factor = derivative_factor_;
            }
            float getDerivativeFactor() const CV_OVERRIDE { return options.derivative_factor; }
            void setDerivatSmooth(float sderivatives_) CV_OVERRIDE { 
                options.sderivatives = sderivatives_;
            }
            float getDerivatSmooth() const CV_OVERRIDE { return options.sderivatives; }
            void setDiffusivityType(AKAZE::AKAZEDIFFUSIVITY_TYPE diffusivity_) CV_OVERRIDE{
                diffusivity = diffusivity_;
                switch (diffusivity)
                {
                case AKAZE::AKAZEDIFFUSIVITY_TYPE::PM_G1:
                    options.diffusivity = DIFFUSIVITY_TYPE::PM_G1;
                    break;
                case AKAZE::AKAZEDIFFUSIVITY_TYPE::PM_G2:
                    options.diffusivity = DIFFUSIVITY_TYPE::PM_G2;
                    break;
                case AKAZE::AKAZEDIFFUSIVITY_TYPE::WEICKERT:
                    options.diffusivity = DIFFUSIVITY_TYPE::WEICKERT;
                    break;
                case AKAZE::AKAZEDIFFUSIVITY_TYPE::CHARBONNIER:
                    options.diffusivity = DIFFUSIVITY_TYPE::CHARBONNIER;
                    break;
                default:
                    options.diffusivity = DIFFUSIVITY_TYPE::PM_G1;
                    break;
                }
            }
            AKAZE::AKAZEDIFFUSIVITY_TYPE getDiffusivityType() const CV_OVERRIDE { return diffusivity; }
            void setDetectResponseTh(float dthreshold_) CV_OVERRIDE { 
                options.dthreshold = dthreshold_;
            }
            float getDetectResponseTh() const CV_OVERRIDE { return options.dthreshold; }
            void setMinDetectResponseTh(float min_dthreshold_) CV_OVERRIDE { 
                options.min_dthreshold = min_dthreshold_;
            }
            float getMinDetectResponseTh() const CV_OVERRIDE { return options.min_dthreshold; }
            void setDescriptorSize(int descriptor_size_) CV_OVERRIDE { 
                options.descriptor_size = descriptor_size_;
            }
            int getDescriptorSize() const CV_OVERRIDE { return options.descriptor_size; }
            void setNrChannels(int descriptor_channels_) CV_OVERRIDE { 
                options.descriptor_channels = descriptor_channels_;
            }
            int getNrChannels() const CV_OVERRIDE { return options.descriptor_channels; }
            void setPatternSize(int descriptor_pattern_size_) CV_OVERRIDE { 
                options.descriptor_pattern_size = descriptor_pattern_size_;
            }
            int getPatternSize() const CV_OVERRIDE { return options.descriptor_pattern_size; }
            void setContrastFactor(float kcontrast_) CV_OVERRIDE { 
                options.kcontrast = kcontrast_;
            }
            float getContrastFactor() const CV_OVERRIDE { return options.kcontrast; }
            void setPercentileLevel(float kcontrast_percentile_) CV_OVERRIDE { 
                options.kcontrast_percentile = kcontrast_percentile_;
            }
            float getPercentileLevel() const CV_OVERRIDE { return options.kcontrast_percentile; }
            void setNrBins(size_t kcontrast_nbins_) CV_OVERRIDE { 
                options.kcontrast_nbins = kcontrast_nbins_;
                calcByteAdd();
            }
            size_t getNrBins() const CV_OVERRIDE { return options.kcontrast_nbins; }
            void setVerbosity(bool verbosity_) CV_OVERRIDE { 
                options.verbosity = verbosity_;
            }
            bool getVerbosity() const CV_OVERRIDE { return options.verbosity; }
            void setNrCudaImages(int ncudaimages_) CV_OVERRIDE { 
                options.ncudaimages = ncudaimages_;
            }
            int getNrCudaImages() const CV_OVERRIDE { return options.ncudaimages; }
            void setMaxNrKeypoints(int maxkeypoints_, const bool adaptThreshold = true) CV_OVERRIDE
            {
                options.maxkeypoints = maxkeypoints_;
                if (adaptThreshold && maxkeypoints_ < 100000)
                {
                    options.dthreshold = 10.f / static_cast<float>(maxkeypoints_);
                }
                calcByteAdd();
            }
            int getMaxNrKeypoints() const CV_OVERRIDE { return options.maxkeypoints; }

            // Info about image
            void setImageInfo(const std::string &info, const bool append = true) CV_OVERRIDE 
            {
                if(append){
                    options.info += info;
                }else
                {
                    options.info = info;
                }
            }
            std::string getImageInfo() const CV_OVERRIDE { return options.info; }

            // Compute the AKAZE_Impl features and descriptors on an image
            void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                  cv::OutputArray descriptors, bool useProvidedKeypoints = false) CV_OVERRIDE;
            void detectAndComputeAsync(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                       cv::OutputArray descriptors, bool useProvidedKeypoints, cv::cuda::Stream &stream) CV_OVERRIDE;
            void detectAndComputeAsync(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                       cv::OutputArray descriptors, std::mt19937 &mt, bool useProvidedKeypoints, cv::cuda::Stream &stream) CV_OVERRIDE;

            virtual int descriptorSize() const {
                if (options.descriptor < DESCRIPTOR_TYPE::MLDB_UPRIGHT)
                {
                    return 64;
                }
                else
                {
                    // We use the full length binary descriptor -> 486 bits
                    if (options.descriptor_size == 0)
                    {
                        int t = (6 + 36 + 120) * options.descriptor_channels;
                        return ceil(t / 8.);
                    }
                    else
                    {
                        // We use the random bit selection length binary descriptor
                        return ceil(options.descriptor_size / 8.);
                    }
                }
            }

            virtual int descriptorType() const {
                if (options.descriptor < DESCRIPTOR_TYPE::MLDB_UPRIGHT)
                {
                    return CV_32FC1;
                }
                else
                {
                    return CV_8UC1;
                }
            }

            virtual int defaultNorm() const {
                if (options.descriptor < DESCRIPTOR_TYPE::MLDB_UPRIGHT)
                {
                    return cv::NORM_L2;
                }
                else
                {
                    // We use the full length binary descriptor -> 486 bits
                    return cv::NORM_HAMMING;
                }
            }

            virtual size_t getByteMultiplierGPU() const CV_OVERRIDE { return byte_multiplier; }
            virtual size_t getByteAddGPU() const CV_OVERRIDE { return byte_add; }
            void calcByteMultiplier();
            void calcByteAdd();

        protected:
            AKAZE::AKAZEDIFFUSIVITY_TYPE diffusivity;
            AKAZEOptions options;
            size_t byte_multiplier = 1;
            size_t byte_add = 0;
        };

        void AKAZE_Impl::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                          cv::OutputArray descriptors, bool useProvidedKeypoints)
        {
            // cv::cuda::Stream stream;
            std::mt19937 g(0);
            detectAndComputeAsync(image, mask, keypoints, descriptors, g, useProvidedKeypoints, cv::cuda::Stream::Null());
        }

        void AKAZE_Impl::detectAndComputeAsync(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                               cv::OutputArray descriptors, bool useProvidedKeypoints, cv::cuda::Stream &stream)
        {
            std::random_device rd;
            std::mt19937 g(rd());
            detectAndComputeAsync(image, mask, keypoints, descriptors, g, useProvidedKeypoints, stream);
        }

        void AKAZE_Impl::detectAndComputeAsync(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                                               cv::OutputArray descriptors, std::mt19937 &mt, bool useProvidedKeypoints, cv::cuda::Stream &stream)
        {
            if (useProvidedKeypoints){
                throw std::runtime_error("Calculating AKAZE descriptors from provided keypoints not implemented.");
            }
            stream.waitForCompletion();
            const cv::Mat mask_ = mask.getMat();
            const cv::Mat img_ = image.getMat();
            AKAZEOptions options_local = options;
            options_local.setHeight(img_.rows);
            options_local.setWidth(img_.cols);

            CV_Assert(!img_.empty() && (img_.type() == CV_32FC1 || img_.type() == CV_8UC1));

            cv::Mat img_32;
            if (img_.type() != CV_32F)
            {
                img_.convertTo(img_32, CV_32F, 1.0 / 255.0, 0);
            }
            else
            {
                img_32 = img_.clone();
            }
            cudaStream_t streamA = cv::cuda::StreamAccessor::getStream(stream);
            libAKAZECU::AKAZE evolution(options_local, streamA, mt, mask);
            evolution.Create_Nonlinear_Scale_Space(img_32);
            const bool calc_descr = descriptors.needed();
            cv::Mat descr;
            if (calc_descr){
                evolution.Feature_Detection();
                evolution.Compute_Descriptors(keypoints, descr);
                stream.waitForCompletion();

                CV_Assert(descr.rows == static_cast<int>(keypoints.size()));
                if (!mask_.empty() && !descr.empty())
                {
                    cv::Mat descr_mask;
                    std::vector<cv::KeyPoint> keypoints_mask;
                    for (int i = 0; i < descr.rows; i++)
                    {
                        const cv::KeyPoint &kp = keypoints.at(i);
                        cv::Point pt(static_cast<int>(std::round(kp.pt.x)), static_cast<int>(std::round(kp.pt.y)));
                        if (mask_.at<unsigned char>(pt))
                        {
                            keypoints_mask.emplace_back(kp);
                            descr_mask.push_back(descr.row(i));
                        }
                    }
                    descr_mask.copyTo(descr);
                    keypoints = keypoints_mask;
                }
                descriptors.create(descr.size(), descr.type());
                cv::Mat descr_out = descriptors.getMat();
                descr.copyTo(descr_out);
            }else{
                evolution.Feature_Detection(&keypoints);
                stream.waitForCompletion();

                if (!mask_.empty() && !keypoints.empty())
                {
                    std::vector<cv::KeyPoint> keypoints_mask;
                    for (size_t i = 0; i < keypoints.size(); i++)
                    {
                        const cv::KeyPoint &kp = keypoints.at(i);
                        cv::Point pt(static_cast<int>(std::round(kp.pt.x)), static_cast<int>(std::round(kp.pt.y)));
                        if (mask_.at<unsigned char>(pt))
                        {
                            keypoints_mask.emplace_back(kp);
                        }
                    }
                    keypoints = keypoints_mask;
                }
            }
        }

        cv::Ptr<AKAZE> AKAZE::create(int imgWidth, int imgHeight, int omax, int nsublevels, float soffset, float derivative_factor, float sderivatives, AKAZE::AKAZEDIFFUSIVITY_TYPE diffusivity, float dthreshold, float min_dthreshold, int descriptor_size, int descriptor_channels, int descriptor_pattern_size, float kcontrast, float kcontrast_percentile, size_t kcontrast_nbins, int ncudaimages, int maxkeypoints, bool verbosity)
        {
            CV_Assert(imgWidth > 0 &&imgHeight > 0 && omax == 4 && nsublevels == 4);
            return cv::makePtr<AKAZE_Impl>(imgWidth, imgHeight, omax, nsublevels, soffset, derivative_factor, sderivatives, diffusivity, dthreshold, min_dthreshold, descriptor_size, descriptor_channels, descriptor_pattern_size, kcontrast, kcontrast_percentile, kcontrast_nbins, ncudaimages, maxkeypoints, verbosity);
        }

        cv::String AKAZE::getDefaultName() const
        {
            return (cv::Feature2D::getDefaultName() + ".CudaAKAZE");
        }

        void AKAZE_Impl::calcByteMultiplier()
        {
            double cu_imgs = 0;
            for (int i = 0; i < options.omax; i++)
            {
                double s = 1. / static_cast<double>(std::pow(2, i));
                s *= s;
                cu_imgs += s;
            }
            const double ns = 4 * static_cast<double>(options.nsublevels);
            const double nbf = static_cast<double>(sizeof(float));
            cu_imgs *= ns * nbf;
            cu_imgs *= 1.1;//Account for image padding

            // From function FindExtrema
            const double b = 2.0 / (32.0 * 16.0);
            cu_imgs += 2.0 * b * static_cast<double>(sizeof(int));
            cu_imgs += (128 + 3) * b * static_cast<double>(sizeof(cv::KeyPoint));
            cu_imgs += b * static_cast<double>(3 * sizeof(int) + sizeof(float));

            byte_multiplier = static_cast<size_t>(std::ceil(cu_imgs));
        }

        void AKAZE_Impl::calcByteAdd(){
            const size_t ns = 4 * static_cast<size_t>(options.nsublevels);
            const size_t nbf = static_cast<size_t>(sizeof(float));
            const size_t nbi = static_cast<size_t>(sizeof(int));
            const size_t nbkp = static_cast<size_t>(sizeof(cv::KeyPoint));
            const size_t ptsMax = static_cast<size_t>(options.maxkeypoints);

            byte_add = (2 * nbkp + 61 + 3 * 29 + 21 * 21 * nbi) * ptsMax + (3 * nbi + 3 * nbf + 2) * (ns * static_cast<size_t>(options.omax) + nbf);
            byte_add += 3 * nbi * ptsMax * ns * static_cast<size_t>(options.omax);
            byte_add *= nbf;
            byte_add += 2 * 8 * 61 * nbi;
            byte_add += (2 * 5 + 1) * nbf;
            byte_add += static_cast<size_t>(options.kcontrast_nbins) * nbi * 2;
            byte_add += ptsMax * nbi * 3;
            byte_add += 512 * nbi;

            size_t nceil = static_cast<size_t>(std::ceil(std::log10(static_cast<double>(ptsMax)) / std::log10(2.)));
            byte_add += std::pow(2, nceil) * 16 * (nbi + 2);
            byte_add += 21 * nbf;
            byte_add += (17 + 2 * 61 * 8 + 1 + 512) * nbi;

            byte_add += nbi + std::pow(2, nceil) * (nbf + 3 * nbi);
            byte_add += ptsMax * nbkp;
        }
#endif

        int getMaxNrGpuThreadsFromMemoryUsage(const cv::Mat &image, const size_t &byteMuliplier, const size_t &byteAdd, const size_t &bytesReserve){
            size_t freeMem, totalMem;
            cudaSetDevice(0);
            cudaMemGetInfo(&freeMem, &totalMem);
            if (bytesReserve > freeMem){
                return 1;
            }
            size_t nrImgBytes = image.total() * image.elemSize();
            //Account for mask and intermediate mats
            nrImgBytes += image.total();
            nrImgBytes += byteMuliplier * image.total();
            nrImgBytes += byteAdd;
            size_t nrThreadsMax = (freeMem - bytesReserve) / nrImgBytes;
            return static_cast<int>(nrThreadsMax);
        }

        size_t getDefaultByteMultiplierGpu(const int nlevels)
        {
            double s2 = 0;
            for (int i = 0; i < nlevels; i++)
            {
                double s = 1. / std::pow(2., i);
                s *= s;
                s2 += s;
            }
            double mult = s2;
            mult *= 4.;
            const double nbf = static_cast<double>(sizeof(float));
            mult += 2 * nbf;
            mult += s2 * 32;
            return static_cast<size_t>(std::ceil(mult));
        }

        size_t getDefaultByteAddGpu()
        {
            const size_t nbf = static_cast<size_t>(sizeof(float));
            return 1024 * 1024 * nbf;
        }
    }

    void visualizeKeypoints(const cv::Mat &img, const std::vector<cv::KeyPoint> &keypoints, const double &tilt, const double &roll, const std::string *img_name)
    {
#if DEBUG_SHOW_KEYPOINTS_WARPED > 1
        std::string info = std::to_string(static_cast<int>(std::round(tilt * 100.f)));
        info += "_" + std::to_string(static_cast<int>(std::round(roll * 100.f)));
        if(img_name)
        {
            info = *img_name + "_" + info;
        }
#endif
#if DEBUG_SHOW_KEYPOINTS_WARPED == 1
        if(keypoints.empty()){
            std::cout << "No keypoints found for tilt " << tilt << " and roll " << roll << std::endl;
        }
        else{
            std::cout << keypoints.size() << " keypoints found for tilt " << tilt << " and roll " << roll << std::endl;
        }
#endif
#if DEBUG_SHOW_KEYPOINTS_WARPED < 3
        cv::Mat color;
        cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);
        // const cv::Vec3b pix_val(0, 0, 255);
        for (const auto &kp : keypoints)
        {
            const cv::Point pt(static_cast<int>(round(kp.pt.x)), static_cast<int>(round(kp.pt.y)));
            cv::circle(color, pt, 2, cv::Scalar(0, 0, 255), cv::FILLED);
            // color.at<cv::Vec3b>(pt) = pix_val;
        }
#else
        const cv::Mat color = img;
#endif
#if DEBUG_SHOW_KEYPOINTS_WARPED == 1
        const int tilt_i = static_cast<int>(std::round(10. * tilt));
        const int roll_i = static_cast<int>(std::round(10. * roll));
        const std::string window_name = "keypoints_" + std::to_string(keypoints.size()) + "_10tilt_" + std::to_string(tilt_i) + "_10roll_" + std::to_string(roll_i);
        cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::imshow(window_name, color);
        cv::waitKey(0);
        cv::destroyWindow(window_name);
#elif DEBUG_SHOW_KEYPOINTS_WARPED > 1
            storeDebugImgToDisk(color, "dbg_affine_warps", DEBUG_IMG_STOREPATH, "affine_warp", info, false);
#endif
    }
}