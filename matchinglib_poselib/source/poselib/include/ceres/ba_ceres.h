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

// #include "utils_common.h"
#include "ceres/ba_types.h"
#include <opencv2/highgui.hpp>
#include <unordered_set>

#include "poselib/pose_helper.h"

namespace poselib
{
    struct CamMatDamping
    {
        enum DampApply
        {
            NO_DAMPING = 0x0,
            DAMP_F_CHANGE = 0x1,
            DAMP_FX_FY_RATIO = 0x2,
            DAMP_CX_CY = 0x4
        };
        double imgDiag_2 = 0.;
        double focalMin = 0.;
        double focalMax = 0.;
        double focalMid = 0.;
        double focalRange2 = 0.;
        double fChangeMax = 0.5;
        double fxfyRatioDiffMax = 0.05;
        double cxcyDistMidRatioMax = 0.15;
        int damping = NO_DAMPING;

        CamMatDamping(const cv::Size &img_Size, const double &fChangeMax_ = 0., const double &fxfyRatioDiffMax_ = 0., const double &cxcyDistMidRatioMax_ = 0.);
        CamMatDamping(const cv::Size &img_Size, const bool dampF = false, const bool dampFxFy = false, const bool dampCxCy = false);
        void setFChangeMax(const double &fChangeMax_);
        void setFxfyRatioDiffMax(const double &fxfyRatioDiffMax_);
        void setCxcyDistMidRatioMax(const double &cxcyDistMidRatioMax_);
        bool useDampingF() const;
        bool useDampingFxFy() const;
        bool useDampingCxCy() const;
        void disableDampingF();
        void disableDampingFxFy();
        void disableDampingCxCy();
        void getFMinMax(const cv::Size &img_Size, const double &ang_min_deg = 22.5, const double &ang_max_deg = 120.);
    };

    bool stereo_bundleAdjustment(cv::InputArray p1,
                                 cv::InputArray p2,
                                 cv::InputOutputArray R,
                                 cv::InputOutputArray t,
                                 cv::InputOutputArray K1,
                                 cv::InputOutputArray K2,
                                 const cv::Size &img_Size,
                                 cv::InputOutputArray Q = cv::noArray(),
                                 bool pointsInImgCoords = false,
                                 cv::InputArray mask = cv::noArray(),
                                 const double angleThresh = 1.25,
                                 const double t_norm_tresh = 0.05,
                                 const LossFunctionToUse loss = LossFunctionToUse::CAUCHY,
                                 const MethodToUse method = MethodToUse::DEFAULT,
                                 const double toleranceMultiplier = 1.0,
                                 const double th = -1.0,
                                 const bool fixFocal = false,
                                 const bool fixPrincipalPt = false,
                                 const bool fixDistortion = false,
                                 const bool distortion_damping = false,
                                 const bool optimCalibrationOnly = false,
                                 const bool fxEqFy = false,
                                 const bool normalize3Dpts = true,
                                 const CamMatDamping *K_damping = nullptr,
                                 cv::InputOutputArray dist1 = cv::noArray(),
                                 cv::InputOutputArray dist2 = cv::noArray(),
                                 const std::vector<size_t> *constant_3d_point_idx = nullptr,
                                 int cpu_count = -1,
                                 int verbose = 0);

    bool refineMultCamBA(cv::InputArray ps,
                         cv::InputArray map3D,
                         cv::InputOutputArray Rs,
                         cv::InputOutputArray ts,
                         cv::InputOutputArray Qs,
                         cv::InputOutputArray Ks,
                         const std::vector<size_t> &img_2_cam_idx,
                         const cv::Size &img_Size,
                         cv::InputArray masks_corr = cv::noArray(),
                         cv::InputArray mask_Q = cv::noArray(),
                         const double angleThresh = 1.25,
                         const double t_norm_tresh = 0.05,
                         const LossFunctionToUse loss = LossFunctionToUse::CAUCHY,
                         const MethodToUse method = MethodToUse::DEFAULT,
                         const double toleranceMultiplier = 1.0,
                         const double th = -1.0,
                         const bool fixFocal = false,
                         const bool fixPrincipalPt = false,
                         const bool fixDistortion = false,
                         const bool distortion_damping = false,
                         const bool optimCalibrationOnly = false,
                         const bool fxEqFy = false,
                         const bool keepScaling = true,
                         const bool normalize3Dpts = true,
                         const CamMatDamping *K_damping = nullptr,
                         cv::InputOutputArray dists = cv::noArray(),
                         const std::vector<size_t> *constant_3d_point_idx = nullptr,
                         const std::vector<size_t> *constant_cam_idx = nullptr,
                         const std::vector<size_t> *constant_pose_idx = nullptr,
                         int cpu_count = -1,
                         int verbose = 0);

    bool refineMultCamBA(const std::unordered_map<std::pair<int, int>, std::vector<cv::Point2f>, pair_hash, pair_EqualTo> &corrs,
                         const std::unordered_map<std::pair<int, int>, std::vector<size_t>, pair_hash, pair_EqualTo> &map3D,
                         std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &Rs,
                         std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &ts,
                         cv::InputOutputArray Qs,
                         cv::InputOutputArray Ks,
                         const std::unordered_map<std::pair<int, int>, size_t, pair_hash, pair_EqualTo> &cimg_2_cam_idx,
                         const cv::Size &img_Size,
                         const double angleThresh = 1.25,
                         const double t_norm_tresh = 0.05,
                         const LossFunctionToUse loss = LossFunctionToUse::CAUCHY,
                         const MethodToUse method = MethodToUse::DEFAULT,
                         const double toleranceMultiplier = 1.0,
                         const double th = -1.0,
                         const bool fixFocal = false,
                         const bool fixPrincipalPt = false,
                         const bool fixDistortion = false,
                         const bool distortion_damping = false,
                         const bool optimCalibrationOnly = false,
                         const bool fxEqFy = false,
                         const bool keepScaling = true,
                         const bool normalize3Dpts = true,
                         const CamMatDamping *K_damping = nullptr,
                         const std::unordered_map<std::pair<int, int>, std::vector<bool>, pair_hash, pair_EqualTo> *masks_corr = nullptr,
                         cv::InputArray mask_Q = cv::noArray(),
                         cv::InputOutputArray dists = cv::noArray(),
                         const std::vector<size_t> *constant_3d_point_idx = nullptr,
                         const std::vector<size_t> *constant_cam_idx = nullptr,
                         const std::unordered_set<std::pair<int, int>, pair_hash, pair_EqualTo> *constant_pose_idx = nullptr,
                         int cpu_count = -1,
                         int verbose = 0,
                         std::string *dbg_info = nullptr);

    bool refineCamsSampsonBA(const std::unordered_map<std::pair<int, int>, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>, pair_hash, pair_EqualTo> &corrs,
                             std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &Rs,
                             std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &ts,
                             std::unordered_map<int, cv::Mat> &Ks,
                             const cv::Size &img_Size,
                             const double angleThresh = 15.,
                             const double t_norm_tresh = 0.2,
                             const LossFunctionToUse loss = LossFunctionToUse::CAUCHY,
                             const MethodToUse method = MethodToUse::DEFAULT,
                             const double toleranceMultiplier = 1.0,
                             const double th = 2.5,
                             const bool fixFocal = false,
                             const bool fixPrincipalPt = true,
                             const bool fixDistortion = true,
                             const bool distortion_damping = false,
                             const bool fxEqFy = true,
                             const CamMatDamping *K_damping = nullptr,
                             const std::unordered_map<std::pair<int, int>, std::vector<bool>, pair_hash, pair_EqualTo> *masks_corr = nullptr,
                             std::unordered_map<int, cv::Mat> *dists = nullptr,
                             const std::vector<int> *constant_cam_idx = nullptr,
                             const std::unordered_set<std::pair<int, int>, pair_hash, pair_EqualTo> *constant_pose_idx = nullptr,
                             int cpu_count = -1,
                             int verbose = 0);

    bool refineFixedDepthBA(const std::unordered_map<int, std::vector<cv::Point2f>> &corrs,
                            const std::unordered_map<int, std::vector<size_t>> &map3D,
                            std::unordered_map<int, cv::Mat> &Rs,
                            std::unordered_map<int, cv::Mat> &ts,
                            cv::InputOutputArray Qs,
                            cv::InputOutputArray Ks,
                            std::unordered_map<int, std::pair<double, double>> &depth_scales, // key = refers to depth_vals first val, val = (depth scale, depth add)
                            const std::vector<std::pair<int, double>> &depth_vals,
                            const std::unordered_map<int, size_t> &cimg_2_cam_idx,
                            const cv::Size &img_Size,
                            const double angleThresh = 45.0,
                            const double t_norm_tresh = 2.0,
                            const LossFunctionToUse loss = LossFunctionToUse::CAUCHY,
                            const MethodToUse method = MethodToUse::DEFAULT,
                            const double toleranceMultiplier = 1.0,
                            const double th = -1.0,
                            const bool fixFocal = false,
                            const bool fixPrincipalPt = true,
                            const bool fixDistortion = true,
                            const bool distortion_damping = false,
                            const bool optimCalibrationOnly = false,
                            const bool fxEqFy = true,
                            const bool keepScaling = true,
                            const CamMatDamping *K_damping = nullptr,
                            const std::unordered_map<int, std::vector<bool>> *masks_corr = nullptr,
                            cv::InputArray mask_Q = cv::noArray(),
                            cv::InputOutputArray dists = cv::noArray(),
                            const std::vector<size_t> *constant_cam_idx = nullptr,
                            const std::unordered_set<int> *constant_pose_idx = nullptr,
                            int cpu_count = -1,
                            int verbose = 0);
}
