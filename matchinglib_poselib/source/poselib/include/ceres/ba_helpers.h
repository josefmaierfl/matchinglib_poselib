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
#include <opencv2/highgui.hpp>
#include <unordered_set>
#include "ceres/ba_cost_functions.h"
#include "ceres/ba_types.h"
#include "poselib/pose_helper.h"

#include <math.h>
#include <numeric>
// #include <opencv2/core/eigen.hpp>

#include "poselib/pose_helper.h"

namespace poselib
{
    inline void getScaleAndShiftFromMatrix(cv::InputArray M, cv::Mat &shift, double &scale)
    {
        const cv::Mat M_ = M.getMat();
        scale = M_.at<double>(0, 0);
        shift = M_.col(3).clone();
        shift.resize(3);
    }

    inline void shiftScale3DPoint(cv::Mat &Q, const cv::Mat &shift, const double &scale)
    {
        CV_Assert(Q.rows == 3);
        Q *= scale;
        Q += shift;
    }

    inline void undoShiftScale3DPoint_T(cv::Mat &Q, const cv::Mat &shift_T, const double &scale)
    {
        CV_Assert(Q.cols == 3);
        Q -= shift_T;
        Q /= scale;
    }

    inline void undoShiftScale3DPoint(cv::Mat &Q, const cv::Mat &shift, const double &scale)
    {
        CV_Assert(Q.rows == 3);
        Q -= shift;
        Q /= scale;
    }

    inline void shiftScaleTranslationVec(cv::Mat &t, const cv::Mat &R, const cv::Mat &shift, const double &scale)
    {
        t *= scale;
        t -= R * shift;
    }

    inline void undoShiftScaleTranslationVec(cv::Mat &t, const cv::Mat &R, const cv::Mat &shift, const double &scale)
    {
        t += R * shift;//R should be the actual estimated R, not the one used before BA
        t /= scale;
    }

    inline void undoShiftScaleTranslationVec(cv::InputOutputArray t, cv::InputArray R, const cv::Mat &shift, const double &scale)
    {
        cv::Mat t_ = t.getMat();
        const cv::Mat R_ = R.getMat();
        undoShiftScaleTranslationVec(t_, R, shift, scale);
    }

    cv::Mat normalize3Dpts_GetShiftScale(cv::InputOutputArray Qs);

    void shiftScaleTranslationVecs(std::vector<cv::Mat> &ts, const std::vector<cv::Mat> &Rs, cv::InputArray M);

    void shiftScaleTranslationVecs(std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &ts, const std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &Rs, cv::InputArray M);

    void undoNormalizeQs(cv::InputOutputArray Qs, cv::InputArray M);

    void undoShiftScaleTranslationVecs(std::vector<cv::Mat> &ts, const std::vector<cv::Mat> &Rs, cv::InputArray M);

    void undoShiftScaleTranslationVecs(std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &ts, const std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &Rs, cv::InputArray M);

    template <typename T>
    bool checkPoseDifferenceRescale(T &ba_data, std::vector<Eigen::Vector3d> &t_normed_init, std::vector<double> &t_norms_init, std::vector<Eigen::Vector4d> &quats_initial, const double &angleThresh, const double &t_norm_tresh, const bool keepScaling = false, double *scaling_med = nullptr)
    {
        const size_t nrPoses = quats_initial.size();
        std::vector<Eigen::Vector3d> t_normed;
        std::vector<double> t_norm_mult;
        for (size_t i = 0; i < nrPoses; ++i)
        {
            if (ba_data.constant_poses[i])
            {
                continue;
            }
            const Eigen::Vector3d &t_ref = ba_data.ts.at(i);
            const double t_norm = t_ref.norm();
            if (abs(t_norm) < 1e-3)
            {
                std::cout << "Detected a too small translation vector norm after BA!" << std::endl;
                return false;
            }
            t_norm_mult.emplace_back(t_norms_init.at(i) / t_norm);
            Eigen::Vector3d t_normed = t_ref / t_norm;

            double r_diff, t_diff;
            poselib::getRTQuality(quats_initial.at(i), ba_data.quats.at(i), t_normed_init.at(i), t_normed, &r_diff, &t_diff);
            r_diff = 180.0 * r_diff / M_PI;
            if ((abs(r_diff) > angleThresh) || (t_diff > t_norm_tresh))
            {
                std::cout << "Difference of translation vectors and/or rotation before and after BA above threshold. Discarding refined values." << std::endl;
                return false;
            }
        }

        double scaling_med_ = 1.;
        if (!t_norm_mult.empty())
        {
            std::sort(t_norm_mult.begin(), t_norm_mult.end());
            if (t_norm_mult.size() % 2)
            {
                scaling_med_ = t_norm_mult.at((t_norm_mult.size() - 1) / 2);
            }
            else
            {
                const size_t mid_el = t_norm_mult.size() / 2;
                scaling_med_ = (t_norm_mult.at(mid_el - 1) + t_norm_mult.at(mid_el)) / 2.0;
            }
            for (const auto &sm : t_norm_mult)
            {
                const double scale_mult = sm / scaling_med_;
                if (scale_mult < 0.66 || scale_mult > 1.33)
                {
                    std::cout << "Detected too large scale variations in translation vector norms after BA!" << std::endl;
                    return false;
                }
            }
            if (scaling_med && keepScaling)
            {
                for (size_t i = 0; i < nrPoses; ++i)
                {
                    ba_data.ts.at(i) *= scaling_med_;
                }
                *scaling_med = scaling_med_;
            }
        }
        return true;
    }

    bool getCameraParametersColmap(const cv::Mat &Kx, const int &max_img_size, const cv::Size &img_Size, const bool fxEqFy, colmap::Camera &cam, double &focal_length_ratio, bool &fxEqFy_all_i, cv::InputArray distortion = cv::noArray());

    template <typename T>
    bool checkIntrinsics(const T &ba_data, const bool fixFocal, const bool fixPrincipalPt, const bool fixDistortion, const bool haveDists, const std::vector<double> &focal_length_ratios)
    {
        if (!fixFocal || !fixPrincipalPt || (!fixDistortion && haveDists))
        {
            const double maxDistortionCoeff = 3.5;
            size_t idx = 0;
            for (const auto &cam : ba_data.cams)
            {
                if (!ba_data.constant_cams[idx])
                {
                    const double &f_rat = focal_length_ratios.at(idx);
                    const double minFrat = std::max(std::min(0.1, 0.33 * f_rat), 0.075);
                    const double maxFrat = std::min(std::max(5., 3.0 * f_rat), 10.0);
                    if (cam.HasBogusParams(minFrat, maxFrat, maxDistortionCoeff))
                    {
                        std::cout << "Intrinsics of camera " << idx << " out of range after BA! Discarding refined values." << std::endl;
                        return false;
                    }
                }
                idx++;
            }
        }
        return true;
    }

    template <typename T>
    bool checkIntrinsicsMaps(const T &ba_data, const bool fixFocal, const bool fixPrincipalPt, const bool fixDistortion, const bool haveDists, const std::unordered_map<int, double> &focal_length_ratios)
    {
        if (!fixFocal || !fixPrincipalPt || (!fixDistortion && haveDists))
        {
            const double maxDistortionCoeff = 3.5;
            for (const auto &cam : ba_data.cams)
            {
                if (!ba_data.constant_cams.at(cam.first))
                {
                    const double &f_rat = focal_length_ratios.at(cam.first);
                    const double minFrat = std::max(std::min(0.1, 0.33 * f_rat), 0.075);
                    const double maxFrat = std::min(std::max(5., 3.0 * f_rat), 10.0);
                    if (cam.second.HasBogusParams(minFrat, maxFrat, maxDistortionCoeff))
                    {
                        std::cout << "Intrinsics of camera " << cam.first << " out of range after BA! Discarding refined values." << std::endl;
                        return false;
                    }
                }
            }
        }
        return true;
    }

    template <typename T>
    void updateIntrinsics(T &ba_data, const bool fixFocal, const bool fixPrincipalPt, const bool fixDistortion, const bool haveDists, const size_t &camSi, const std::vector<bool> &fxEqFy_all, std::vector<cv::Mat> &K_vec, std::vector<cv::Mat> &dist_mvec, const double &precision = 1.0)
    {
        if (!fixFocal)
        {
            for (size_t i = 0; i < camSi; ++i)
            {
                if (ba_data.constant_cams[i])
                {
                    continue;
                }
                cv::Mat &Kx = K_vec.at(i);
                colmap::Camera &cam = ba_data.cams.at(i);
                if (fxEqFy_all[i])
                {
                    Kx.at<double>(0, 0) = convertPrecisionRet(cam.FocalLength(), precision);
                    Kx.at<double>(1, 1) = convertPrecisionRet(cam.FocalLength(), precision);
                }
                else
                {
                    Kx.at<double>(0, 0) = convertPrecisionRet(cam.FocalLengthX(), precision);
                    Kx.at<double>(1, 1) = convertPrecisionRet(cam.FocalLengthY(), precision);
                }
            }
        }

        if (!fixPrincipalPt)
        {
            for (size_t i = 0; i < camSi; ++i)
            {
                if (ba_data.constant_cams[i])
                {
                    continue;
                }
                cv::Mat &Kx = K_vec.at(i);
                colmap::Camera &cam = ba_data.cams.at(i);
                Kx.at<double>(0, 2) = convertPrecisionRet(cam.PrincipalPointX(), precision);
                Kx.at<double>(1, 2) = convertPrecisionRet(cam.PrincipalPointY(), precision);
            }
        }

        if (!fixDistortion && haveDists)
        {
            for (size_t i = 0; i < camSi; ++i)
            {
                if (ba_data.constant_cams[i])
                {
                    continue;
                }
                cv::Mat &dist_x = dist_mvec.at(i);
                colmap::Camera &cam = ba_data.cams.at(i);
                const std::vector<size_t> &cam_dist_pars_idx = cam.ExtraParamsIdxs();
                int idx_i = 0;
                for (const auto &idx_p : cam_dist_pars_idx)
                {
                    dist_x.at<double>(idx_i++) = convertPrecisionRet(cam.Params(idx_p), precision);
                }
                // std::cout << "dist_2(" << i << ") " << dist_x.at<double>(0) << std::endl;
            }
        }
    }

    template <typename T>
    void updateIntrinsics(T &ba_data, const bool fixFocal, const bool fixPrincipalPt, const bool fixDistortion, const bool haveDists, const std::unordered_map<int, bool> &fxEqFy_all, std::unordered_map<int, cv::Mat> &Ks, std::unordered_map<int, cv::Mat> *dists, const double &precision = 1.0)
    {
        if (!fixFocal)
        {
            for (const auto &cam : ba_data.cams)
            {
                if (ba_data.constant_cams.at(cam.first))
                {
                    continue;
                }
                cv::Mat &Kx = Ks.at(cam.first);
                if (fxEqFy_all.at(cam.first))
                {
                    Kx.at<double>(0, 0) = convertPrecisionRet(cam.second.FocalLength(), precision);
                    Kx.at<double>(1, 1) = convertPrecisionRet(cam.second.FocalLength(), precision);
                }
                else
                {
                    Kx.at<double>(0, 0) = convertPrecisionRet(cam.second.FocalLengthX(), precision);
                    Kx.at<double>(1, 1) = convertPrecisionRet(cam.second.FocalLengthY(), precision);
                }
            }
        }

        if (!fixPrincipalPt)
        {
            for (const auto &cam : ba_data.cams)
            {
                if (ba_data.constant_cams.at(cam.first))
                {
                    continue;
                }
                cv::Mat &Kx = Ks.at(cam.first);
                Kx.at<double>(0, 2) = convertPrecisionRet(cam.second.PrincipalPointX(), precision);
                Kx.at<double>(1, 2) = convertPrecisionRet(cam.second.PrincipalPointY(), precision);
            }
        }

        if (!fixDistortion && haveDists)
        {
            for (const auto &cam : ba_data.cams)
            {
                if (ba_data.constant_cams.at(cam.first))
                {
                    continue;
                }
                cv::Mat &dist_x = dists->at(cam.first);
                const std::vector<size_t> &cam_dist_pars_idx = cam.second.ExtraParamsIdxs();
                int idx_i = 0;
                for (const auto &idx_p : cam_dist_pars_idx)
                {
                    dist_x.at<double>(idx_i++) = convertPrecisionRet(cam.second.Params(idx_p), precision);
                }
            }
        }
    }

    template <typename T>
    void update3Dpoints(const T &ba_data, const bool optimCalibrationOnly, const bool haveQMasks, const int &nr_Qs, const cv::Mat &mask_Q, cv::Mat &Q, cv::InputArray shiftScaleMat = cv::noArray(), const bool keepScaling = false, const double *scaling_med = nullptr, const double &precision = 1.0)
    {
        bool shiftScale = !shiftScaleMat.empty();
        double scale = 1.0;
        cv::Mat shift;
        if(shiftScale){
            cv::Mat M = shiftScaleMat.getMat();
            getScaleAndShiftFromMatrix(M, shift, scale);
            shift = shift.t();
        }
        if (!optimCalibrationOnly)
        {
            std::vector<size_t>::const_iterator it = ba_data.constant_points3d.begin();
            if (!haveQMasks)
            {
                for (int i = 0; i < nr_Qs; ++i)
                {
                    if (it != ba_data.constant_points3d.end() && i == static_cast<int>(*it))
                    {
                        it++;
                        continue;
                    }
                    if (scaling_med && keepScaling)
                    {
                        Q.at<double>(i, 0) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(i).x(), precision);
                        Q.at<double>(i, 1) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(i).y(), precision);
                        Q.at<double>(i, 2) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(i).z(), precision);
                        if (shiftScale)
                        {
                            cv::Mat qt = Q.row(i);
                            undoShiftScale3DPoint_T(qt, shift, scale);
                        }
                    }
                    else
                    {
                        Q.at<double>(i, 0) = convertPrecisionRet(ba_data.points3d.at(i).x(), precision);
                        Q.at<double>(i, 1) = convertPrecisionRet(ba_data.points3d.at(i).y(), precision);
                        Q.at<double>(i, 2) = convertPrecisionRet(ba_data.points3d.at(i).z(), precision);
                        if (shiftScale)
                        {
                            cv::Mat qt = Q.row(i);
                            undoShiftScale3DPoint_T(qt, shift, scale);
                        }
                    }
                }
            }
            else
            {
                size_t idx = 0;
                for (int i = 0; i < nr_Qs; ++i)
                {
                    if (mask_Q.at<unsigned char>(i))
                    {
                        if (it != ba_data.constant_points3d.end() && idx == *it)
                        {
                            it++;
                        }
                        else
                        {
                            if (scaling_med && keepScaling)
                            {
                                Q.at<double>(i, 0) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(idx).x(), precision);
                                Q.at<double>(i, 1) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(idx).y(), precision);
                                Q.at<double>(i, 2) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(idx).z(), precision);
                                if (shiftScale)
                                {
                                    cv::Mat qt = Q.row(i);
                                    undoShiftScale3DPoint_T(qt, shift, scale);
                                }
                            }
                            else
                            {
                                Q.at<double>(i, 0) = convertPrecisionRet(ba_data.points3d.at(idx).x(), precision);
                                Q.at<double>(i, 1) = convertPrecisionRet(ba_data.points3d.at(idx).y(), precision);
                                Q.at<double>(i, 2) = convertPrecisionRet(ba_data.points3d.at(idx).z(), precision);
                                if (shiftScale)
                                {
                                    cv::Mat qt = Q.row(i);
                                    undoShiftScale3DPoint_T(qt, shift, scale);
                                }
                            }
                        }
                        idx++;
                    }
                }
            }
        }
    }

    template <typename T>
    void update3DpointsNoConstPts(const T &ba_data, const bool optimCalibrationOnly, const bool haveQMasks, const int &nr_Qs, const cv::Mat &mask_Q, cv::Mat &Q, const bool keepScaling = false, const double *scaling_med = nullptr, const double &precision = 1.0)
    {
        if (!optimCalibrationOnly)
        {
            if (!haveQMasks)
            {
                for (int i = 0; i < nr_Qs; ++i)
                {
                    if (scaling_med && keepScaling)
                    {
                        Q.at<double>(i, 0) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(i).x(), precision);
                        Q.at<double>(i, 1) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(i).y(), precision);
                        Q.at<double>(i, 2) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(i).z(), precision);
                    }
                    else
                    {
                        Q.at<double>(i, 0) = convertPrecisionRet(ba_data.points3d.at(i).x(), precision);
                        Q.at<double>(i, 1) = convertPrecisionRet(ba_data.points3d.at(i).y(), precision);
                        Q.at<double>(i, 2) = convertPrecisionRet(ba_data.points3d.at(i).z(), precision);
                    }
                }
            }
            else
            {
                size_t idx = 0;
                for (int i = 0; i < nr_Qs; ++i)
                {
                    if (mask_Q.at<unsigned char>(i))
                    {
                        if (scaling_med && keepScaling)
                        {
                            Q.at<double>(i, 0) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(idx).x(), precision);
                            Q.at<double>(i, 1) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(idx).y(), precision);
                            Q.at<double>(i, 2) = convertPrecisionRet(*scaling_med * ba_data.points3d.at(idx).z(), precision);
                        }
                        else
                        {
                            Q.at<double>(i, 0) = convertPrecisionRet(ba_data.points3d.at(idx).x(), precision);
                            Q.at<double>(i, 1) = convertPrecisionRet(ba_data.points3d.at(idx).y(), precision);
                            Q.at<double>(i, 2) = convertPrecisionRet(ba_data.points3d.at(idx).z(), precision);
                        }
                        idx++;
                    }
                }
            }
        }
    }

    void setOptionsCostfunction(BundleAdjustmentOptions &options, const LossFunctionToUse &loss, const double &th = 0);

    void setOptionsSolveMethod(BundleAdjustmentOptions &options, const MethodToUse &method);
}