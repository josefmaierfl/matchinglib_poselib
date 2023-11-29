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

#include "ceres/ba_ceres.h"
#include "ceres/ba_cost_functions.h"

#include <math.h>
#include <numeric>
#include <map>
#include <opencv2/core/eigen.hpp>

#include "poselib/pose_helper.h"
#include "ceres/ba_helpers.h"
// #include "utils_cv.h"

#define TAKE_NOT_BA_OWNERSHIP true
#define CERES_RESULT_CONVERT_PRECISION FLT_EPSILON

using namespace cv;
using namespace std;

namespace poselib
{
    CamMatDamping::CamMatDamping(const cv::Size &img_Size, const double &fChangeMax_, const double &fxfyRatioDiffMax_, const double &cxcyDistMidRatioMax_)
    {
        imgDiag_2 = (img_Size.width * img_Size.width + img_Size.height * img_Size.height) / 4.0;
        getFMinMax(img_Size);
        setFChangeMax(fChangeMax_);
        setFxfyRatioDiffMax(fxfyRatioDiffMax_);
        setCxcyDistMidRatioMax(cxcyDistMidRatioMax_);
    }

    CamMatDamping::CamMatDamping(const cv::Size &img_Size, const bool dampF, const bool dampFxFy, const bool dampCxCy)
    {
        imgDiag_2 = (img_Size.width * img_Size.width + img_Size.height * img_Size.height) / 4.0;
        getFMinMax(img_Size);
        if (dampF)
        {
            damping |= DAMP_F_CHANGE;
        }
        if (dampFxFy)
        {
            damping |= DAMP_FX_FY_RATIO;
        }
        if (dampCxCy)
        {
            damping |= DAMP_CX_CY;
        }
    }

    void CamMatDamping::setFChangeMax(const double &fChangeMax_)
    {
        if (!nearZero(fChangeMax_) && fChangeMax_ > 0.)
        {
            damping |= DAMP_F_CHANGE;
            fChangeMax = fChangeMax_;
        }
    }

    void CamMatDamping::setFxfyRatioDiffMax(const double &fxfyRatioDiffMax_)
    {
        if (!nearZero(fxfyRatioDiffMax_) && fxfyRatioDiffMax_ > 0.)
        {
            damping |= DAMP_FX_FY_RATIO;
            fxfyRatioDiffMax = fxfyRatioDiffMax_;
        }
    }

    void CamMatDamping::setCxcyDistMidRatioMax(const double &cxcyDistMidRatioMax_)
    {
        if (!nearZero(cxcyDistMidRatioMax_) && cxcyDistMidRatioMax_ > 0.)
        {
            damping |= DAMP_CX_CY;
            cxcyDistMidRatioMax = cxcyDistMidRatioMax_;
        }
    }

    bool CamMatDamping::useDampingF() const
    {
        return damping & DAMP_F_CHANGE;
    }

    bool CamMatDamping::useDampingFxFy() const
    {
        return damping & DAMP_FX_FY_RATIO;
    }

    bool CamMatDamping::useDampingCxCy() const
    {
        return damping & DAMP_CX_CY;
    }

    void CamMatDamping::disableDampingF()
    {
        damping &= ~((int)DAMP_F_CHANGE);
    }

    void CamMatDamping::disableDampingFxFy()
    {
        damping &= ~((int)DAMP_FX_FY_RATIO);
    }

    void CamMatDamping::disableDampingCxCy()
    {
        damping &= ~((int)DAMP_CX_CY);
    }

    void CamMatDamping::getFMinMax(const cv::Size &img_Size, const double &ang_min_deg, const double &ang_max_deg)
    {
        const int dim_max = std::max(img_Size.width, img_Size.height);
        const double w2 = static_cast<double>(dim_max) / 2.0;
        const double ang_min2 = ang_min_deg * M_PI / 360.; //((ang_min_deg / 2) * M_PI / 180.)
        const double ang_max2 = ang_max_deg * M_PI / 360.; //((ang_max_deg / 2) * M_PI / 180.)
        focalMax = w2 / std::tan(ang_min2);
        focalMin = w2 / std::tan(ang_max2);

        focalMid = (focalMin + focalMax) / 2.0;
        focalRange2 = focalMid - focalMin;
    }

    bool stereo_bundleAdjustment(cv::InputArray p1,
                                 cv::InputArray p2,
                                 cv::InputOutputArray R,
                                 cv::InputOutputArray t,
                                 cv::InputOutputArray K1,
                                 cv::InputOutputArray K2,
                                 const cv::Size &img_Size,
                                 cv::InputOutputArray Q,
                                 bool pointsInImgCoords,
                                 cv::InputArray mask,
                                 const double angleThresh,
                                 const double t_norm_tresh,
                                 const LossFunctionToUse loss,
                                 const MethodToUse method,
                                 const double toleranceMultiplier,
                                 const double th,
                                 const bool fixFocal,
                                 const bool fixPrincipalPt,
                                 const bool fixDistortion,
                                 const bool distortion_damping,
                                 const bool optimCalibrationOnly,
                                 const bool fxEqFy,
                                 const bool normalize3Dpts,
                                 const CamMatDamping *K_damping,
                                 cv::InputOutputArray dist1,
                                 cv::InputOutputArray dist2,
                                 const std::vector<size_t> *constant_3d_point_idx,
                                 int cpu_count,
                                 int verbose)
    {
        CV_Assert(p1.type() == CV_64FC1);
        CV_Assert(p2.type() == CV_64FC1);
        CV_Assert(R.type() == CV_64FC1);
        CV_Assert(t.type() == CV_64FC1);
        CV_Assert(K1.type() == CV_64FC1);
        CV_Assert(K2.type() == CV_64FC1);
        CV_Assert(Q.empty() || Q.type() == CV_64FC1);
        CV_Assert(mask.empty() || mask.type() == CV_8UC1);
        CV_Assert(dist1.empty() || dist1.type() == CV_64FC1);
        CV_Assert(dist2.empty() || dist2.type() == CV_64FC1);
        if (p1.empty() || p2.empty())
        {
            cerr << "Image projections must be provided." << endl;
            return false;
        }
        if (Q.empty() && pointsInImgCoords)
        {
            cerr << "When no 3D points are provided, correspondences must be in the camera coordinate system." << endl;
            return false;
        }
        if (!Q.empty() && (p1.rows() != Q.rows())){
            cerr << "The number of image projections and 3D points does not match." << endl;
            return false;
        }
        if (p1.rows() != p2.rows())
        {
            cerr << "The same number of image projections must be provided." << endl;
            return false;
        }
        if (K1.empty() || K2.empty()){
            cerr << "Camera matrices must be provided." << endl;
            return false;
        }
        if (!mask.empty())
        {
            int mask_elems = 0;
            if (mask.cols() > mask.rows())
            {
                mask_elems = mask.cols();
            }else{
                mask_elems = mask.rows();
            }
            if (mask_elems != p1.rows()){
                cerr << "Size of provided mask does not correspond to number of correspondences." << endl;
                return false;
            }
        }

        Mat K1_ = K1.getMat();
        Mat K2_ = K2.getMat();
        const double mean_f1 = (K1_.at<double>(0, 0) + K1_.at<double>(1, 1)) / 2.;
        const double mean_f2 = (K2_.at<double>(0, 0) + K2_.at<double>(1, 1)) / 2.;
        bool fxEqFy_ = fxEqFy;
        const int max_img_size = std::max(img_Size.width, img_Size.height);
        const double focal_length_ratio = (mean_f1 + mean_f2) / static_cast<double>(2 * max_img_size);

        Mat R_ = R.getMat();
        Mat t_ = t.getMat();
        if (t_.cols > t_.rows)
        {
            t_ = t_.t();
        }
        Mat t_initial = t_.clone();
        const double t_norm = cv::norm(t_);
        if (abs(t_norm) < 1e-3)
        {
            cout << "Detected a too small translation vector norm before BA!" << endl;
            return false;
        }
        // t_ /= t_norm;

        Mat Q_normalized, t_normalized, M;
        if(!Q.empty()){
            Q.getMat().copyTo(Q_normalized);
            t_.copyTo(t_normalized);
        }

        double scale = 1.0;
        cv::Mat shift;
        if (normalize3Dpts && !Q_normalized.empty())
        {
            M = normalize3Dpts_GetShiftScale(Q_normalized);
            getScaleAndShiftFromMatrix(M, shift, scale);
            shiftScaleTranslationVec(t_normalized, R_, shift, scale);
            const double t_norm2 = cv::norm(t_normalized);
            if (abs(t_norm2) < 1e-3)
            {
                cout << "Detected a too small translation vector norm before BA using normalization!" << endl;
                return false;
            }
        }

        BundleAdjustmentOptions options;
        setOptionsCostfunction(options, loss, th);
        setOptionsSolveMethod(options, method);
        options.solver_options.parameter_tolerance = 1e-8;
        options.solver_options.gradient_tolerance = 1e-8;
        if(pointsInImgCoords){
            options.solver_options.function_tolerance = 1e-6;
        }else{
            options.solver_options.function_tolerance = 1e-9;
        }
        if (toleranceMultiplier >= 1e-4 && toleranceMultiplier <= 1e5)
        {
            options.solver_options.parameter_tolerance *= toleranceMultiplier;
            options.solver_options.gradient_tolerance *= toleranceMultiplier;
            options.solver_options.function_tolerance *= toleranceMultiplier;
            if (toleranceMultiplier > 9.0)
            {
                options.solver_options.min_trust_region_radius = 1e-9;
                options.solver_options.max_trust_region_radius = 1e10;
            }
        }
        if(!verbose){
            options.solver_options.minimizer_progress_to_stdout = false;
            options.solver_options.logging_type = ceres::SILENT;
            options.print_summary = false;
        }else if(verbose == 1){
            options.solver_options.minimizer_progress_to_stdout = false;
            options.solver_options.logging_type = ceres::SILENT;
            options.print_summary = true;
        }else{
            options.solver_options.minimizer_progress_to_stdout = true;
            options.solver_options.logging_type = ceres::PER_MINIMIZER_ITERATION;
            options.print_summary = true;
        }
        options.CeresCPUcnt = cpu_count;
        options.refine_focal_length = !fixFocal;
        options.refine_principal_point = !fixPrincipalPt;
        options.refine_extra_params = !fixDistortion;

        StereoBAData stereo_data;

        if(!dist1.empty() && !dist2.empty()){
            Mat dist1_ = dist1.getMat();
            Mat dist2_ = dist2.getMat();
            CV_Assert(dist1_.size() == dist2_.size());
            if(dist1_.rows > dist1_.cols){
                dist1_ = dist1_.t();
                dist2_ = dist2_.t();
            }
            if (dist1_.cols == 1)
            {
                stereo_data.cam1.InitializeWithName("SIMPLE_RADIAL", mean_f1, img_Size.width, img_Size.height);
                std::vector<double> params = {mean_f1,
                                              K1_.at<double>(0, 2),
                                              K1_.at<double>(1, 2),
                                              dist1_.at<double>(0)};
                stereo_data.cam1.SetParams(params);
                stereo_data.cam2.InitializeWithName("SIMPLE_RADIAL", mean_f2, img_Size.width, img_Size.height);
                std::vector<double> params2 = {mean_f2,
                                               K2_.at<double>(0, 2),
                                               K2_.at<double>(1, 2),
                                               dist2_.at<double>(0)};
                stereo_data.cam2.SetParams(params2);
                fxEqFy_ = true;
            }
            else if (dist1_.cols == 2)
            {
                stereo_data.cam1.InitializeWithName("RADIAL", mean_f1, img_Size.width, img_Size.height);
                std::vector<double> params = {mean_f1,
                                              K1_.at<double>(0, 2),
                                              K1_.at<double>(1, 2),
                                              dist1_.at<double>(0),
                                              dist1_.at<double>(1)};
                stereo_data.cam1.SetParams(params);
                stereo_data.cam2.InitializeWithName("RADIAL", mean_f2, img_Size.width, img_Size.height);
                std::vector<double> params2 = {mean_f2,
                                               K2_.at<double>(0, 2),
                                               K2_.at<double>(1, 2),
                                               dist2_.at<double>(0),
                                               dist2_.at<double>(1)};
                stereo_data.cam2.SetParams(params2);
                fxEqFy_ = true;
            }
            else if (dist1_.cols == 4)
            {
                stereo_data.cam1.InitializeWithName("OPENCV", mean_f1, img_Size.width, img_Size.height);
                std::vector<double> params = {K1_.at<double>(0, 0),
                                              K1_.at<double>(1, 1),
                                              K1_.at<double>(0, 2),
                                              K1_.at<double>(1, 2),
                                              dist1_.at<double>(0),
                                              dist1_.at<double>(1),
                                              dist1_.at<double>(2),
                                              dist1_.at<double>(3)};
                stereo_data.cam1.SetParams(params);
                stereo_data.cam2.InitializeWithName("OPENCV", mean_f2, img_Size.width, img_Size.height);
                std::vector<double> params2 = {K2_.at<double>(0, 0),
                                               K2_.at<double>(1, 1),
                                               K2_.at<double>(0, 2),
                                               K2_.at<double>(1, 2),
                                               dist2_.at<double>(0),
                                               dist2_.at<double>(1),
                                               dist2_.at<double>(2),
                                               dist2_.at<double>(3)};
                stereo_data.cam2.SetParams(params2);
                fxEqFy_ = false;
            }
            else if (dist1_.cols == 5)
            {
                stereo_data.cam1.InitializeWithName("OPENCV_RADIAL3", mean_f1, img_Size.width, img_Size.height);
                std::vector<double> params = {K1_.at<double>(0, 0),
                                              K1_.at<double>(1, 1),
                                              K1_.at<double>(0, 2),
                                              K1_.at<double>(1, 2),
                                              dist1_.at<double>(0),
                                              dist1_.at<double>(1),
                                              dist1_.at<double>(2),
                                              dist1_.at<double>(3),
                                              dist1_.at<double>(4)};
                stereo_data.cam1.SetParams(params);
                stereo_data.cam2.InitializeWithName("OPENCV_RADIAL3", mean_f2, img_Size.width, img_Size.height);
                std::vector<double> params2 = {K2_.at<double>(0, 0),
                                               K2_.at<double>(1, 1),
                                               K2_.at<double>(0, 2),
                                               K2_.at<double>(1, 2),
                                               dist2_.at<double>(0),
                                               dist2_.at<double>(1),
                                               dist2_.at<double>(2),
                                               dist2_.at<double>(3),
                                               dist2_.at<double>(4)};
                stereo_data.cam2.SetParams(params2);
                fxEqFy_ = false;
            }
            else if (dist1_.cols == 8)
            {
                stereo_data.cam1.InitializeWithName("FULL_OPENCV", mean_f1, img_Size.width, img_Size.height);
                std::vector<double> params = {K1_.at<double>(0, 0),
                                              K1_.at<double>(1, 1),
                                              K1_.at<double>(0, 2),
                                              K1_.at<double>(1, 2),
                                              dist1_.at<double>(0),
                                              dist1_.at<double>(1),
                                              dist1_.at<double>(2),
                                              dist1_.at<double>(3),
                                              dist1_.at<double>(4),
                                              dist1_.at<double>(5),
                                              dist1_.at<double>(6),
                                              dist1_.at<double>(7)};
                stereo_data.cam1.SetParams(params);
                stereo_data.cam2.InitializeWithName("FULL_OPENCV", mean_f2, img_Size.width, img_Size.height);
                std::vector<double> params2 = {K2_.at<double>(0, 0),
                                               K2_.at<double>(1, 1),
                                               K2_.at<double>(0, 2),
                                               K2_.at<double>(1, 2),
                                               dist2_.at<double>(0),
                                               dist2_.at<double>(1),
                                               dist2_.at<double>(2),
                                               dist2_.at<double>(3),
                                               dist2_.at<double>(4),
                                               dist2_.at<double>(5),
                                               dist2_.at<double>(6),
                                               dist2_.at<double>(7)};
                stereo_data.cam2.SetParams(params2);
                fxEqFy_ = false;
            }else{
                cerr << "Distortion model not supported!" << endl;
                return false;
            }
        }
        else if(fxEqFy){
            stereo_data.cam1.InitializeWithName("SIMPLE_PINHOLE", mean_f1, img_Size.width, img_Size.height);
            std::vector<double> params = {mean_f1,
                                          K1_.at<double>(0, 2),
                                          K1_.at<double>(1, 2)};
            stereo_data.cam1.SetParams(params);
            stereo_data.cam2.InitializeWithName("SIMPLE_PINHOLE", mean_f2, img_Size.width, img_Size.height);
            std::vector<double> params2 = {mean_f2,
                                           K2_.at<double>(0, 2),
                                           K2_.at<double>(1, 2)};
            stereo_data.cam2.SetParams(params2);
        }
        else
        {
            stereo_data.cam1.InitializeWithName("PINHOLE", mean_f1, img_Size.width, img_Size.height);
            std::vector<double> params = {K1_.at<double>(0, 0),
                                          K1_.at<double>(1, 1),
                                          K1_.at<double>(0, 2),
                                          K1_.at<double>(1, 2)};
            stereo_data.cam1.SetParams(params);
            stereo_data.cam2.InitializeWithName("PINHOLE", mean_f2, img_Size.width, img_Size.height);
            std::vector<double> params2 = {K2_.at<double>(0, 0),
                                           K2_.at<double>(1, 1),
                                           K2_.at<double>(0, 2),
                                           K2_.at<double>(1, 2)};
            stereo_data.cam2.SetParams(params2);
            fxEqFy_ = false;
        }

        if (optimCalibrationOnly)
        {
            int nr_elems = p1.rows();
            if(!mask.empty()){
                nr_elems = cv::countNonZero(mask.getMat());
            }
            stereo_data.constant_points3d = vector<size_t>(nr_elems);
            iota(stereo_data.constant_points3d.begin(), stereo_data.constant_points3d.end(), 0);
        }
        else if (constant_3d_point_idx && !constant_3d_point_idx->empty())
        {
            std::vector<size_t> constant_3d_point_idx_tmp;
            if(mask.empty()){
                constant_3d_point_idx_tmp = *constant_3d_point_idx;
            }else{
                std::vector<size_t> constant_3d_point_idx_tmp2 = *constant_3d_point_idx;
                std::sort(constant_3d_point_idx_tmp2.begin(), constant_3d_point_idx_tmp2.end());
                std::vector<size_t>::iterator it = constant_3d_point_idx_tmp2.begin();
                Mat mask_ = mask.getMat();
                int miss_cnt = 0;
                const int nr_elems = p1.rows();
                for (int i = 0; i < nr_elems; ++i){
                    int idx = -1;
                    if (it != constant_3d_point_idx_tmp2.end() && i == static_cast<int>(*it))
                    {
                        idx = i - miss_cnt;
                        it++;
                    }
                    if (mask_.at<unsigned char>(i))
                    {
                        if(idx >= 0){
                            constant_3d_point_idx_tmp.emplace_back(static_cast<size_t>(idx));
                        }
                    }else{
                        miss_cnt++;
                    }
                }
            }
            stereo_data.constant_points3d = constant_3d_point_idx_tmp;
        }

        stereo_data.inCamCoordinates = !pointsInImgCoords;

        cv::cv2eigen(R_, stereo_data.R_rel);
        stereo_data.getRelQuaternions();
        Eigen::Vector4d quat_rel_initial = stereo_data.quat_rel;
        cv::cv2eigen(t_normalized, stereo_data.t_rel);
        Eigen::Vector3d te_initial = stereo_data.t_rel;

        Mat p1_ = p1.getMat();
        Mat p2_ = p2.getMat();
        const int nr_meas = p1_.rows;

        Mat mask_tmp;
        if (!mask.empty()){
            mask_tmp = mask.getMat();
        }else{
            mask_tmp = Mat::ones(1, nr_meas, CV_8UC1);
        }

        if (!Q_normalized.empty())
        {
            for (int i = 0; i < nr_meas; ++i){
                if(mask_tmp.at<unsigned char>(i)){
                    Eigen::Vector3d q;
                    cv2eigen(Q_normalized.row(i).t(), q);
                    stereo_data.points3d.push_back(q);
                }
            }
        }

        for (int i = 0; i < nr_meas; ++i){
            if (mask_tmp.at<unsigned char>(i)){
                Eigen::Vector2d x1, x2;
                cv2eigen(p1_.row(i).t(), x1);
                cv2eigen(p2_.row(i).t(), x2);
                stereo_data.corrs.emplace_back(make_pair(x1, x2));
            }
        }
        
        stereo_data.dist_damp = distortion_damping && !fixDistortion && !dist1.empty() && !dist2.empty();

        if (K_damping && K_damping->damping)
        {
            const double cx_init = static_cast<double>(img_Size.width) / 2.;
            const double cy_init = static_cast<double>(img_Size.height) / 2.;
            CamMatDamping damp_tmp = *K_damping;
            if (fixFocal && K_damping->useDampingF()){
                damp_tmp.disableDampingF();
            }
            if ((fxEqFy || fixFocal) && K_damping->useDampingFxFy())
            {
                damp_tmp.disableDampingFxFy();
            }
            if (fixPrincipalPt && K_damping->useDampingCxCy())
            {
                damp_tmp.disableDampingCxCy();
            }
            if (damp_tmp.damping){
                stereo_data.camMat_damp = damp_tmp.damping;
                stereo_data.camMatDampPars.emplace_back(colmap::CamMatDampingSingle(damp_tmp.imgDiag_2,
                                                                                    damp_tmp.focalMid,
                                                                                    damp_tmp.focalRange2,
                                                                                    stereo_data.cam1.MeanFocalLength(),
                                                                                    cx_init,
                                                                                    cy_init,
                                                                                    damp_tmp.damping,
                                                                                    damp_tmp.useDampingF() ? damp_tmp.fChangeMax : 0.,
                                                                                    damp_tmp.useDampingFxFy() ? damp_tmp.fxfyRatioDiffMax : 0.,
                                                                                    damp_tmp.useDampingCxCy() ? damp_tmp.cxcyDistMidRatioMax : 0.));
                stereo_data.camMatDampPars.emplace_back(colmap::CamMatDampingSingle(damp_tmp.imgDiag_2,
                                                                                    damp_tmp.focalMid,
                                                                                    damp_tmp.focalRange2,
                                                                                    stereo_data.cam2.MeanFocalLength(),
                                                                                    cx_init,
                                                                                    cy_init,
                                                                                    damp_tmp.damping,
                                                                                    damp_tmp.useDampingF() ? damp_tmp.fChangeMax : 0.,
                                                                                    damp_tmp.useDampingFxFy() ? damp_tmp.fxfyRatioDiffMax : 0.,
                                                                                    damp_tmp.useDampingCxCy() ? damp_tmp.cxcyDistMidRatioMax : 0.));
            }
        }

        StereoBundleAdjuster ba(options, TAKE_NOT_BA_OWNERSHIP);
        if (!ba.Solve(&stereo_data)){
            t_initial.copyTo(t.getMat());
            return false;
        }

        //Normalize the translation vector
        const double t_norm2 = stereo_data.t_rel.norm();
        if(abs(t_norm2) < 1e-3){
            cout << "Detected a too small translation vector norm after BA!" << endl;
            t_initial.copyTo(t.getMat());
            return false;
        }
        stereo_data.t_rel /= t_norm2;

        double r_diff, t_diff;
        poselib::getRTQuality(quat_rel_initial, stereo_data.quat_rel, te_initial, stereo_data.t_rel, &r_diff, &t_diff);
        r_diff = 180.0 * r_diff / M_PI;
        if ((abs(r_diff) > angleThresh) || (t_diff > t_norm_tresh))
        {
            cout << "Difference of translation vectors and/or rotation before and after BA above threshold. Discarding refined values." << endl;
            t_initial.copyTo(t.getMat());
            return false;
        }

        if ((!fixFocal || !fixPrincipalPt || (!fixDistortion && !dist1.empty() && !dist2.empty())) && pointsInImgCoords)
        {
            const double minFrat = max(min(0.5, 0.66 * focal_length_ratio), 0.2);
            const double maxFrat = min(max(1.5, 1.33 * focal_length_ratio), 5.0);
            const double maxDistortionCoeff = 1.3;
            if (stereo_data.cam1.HasBogusParams(minFrat, maxFrat, maxDistortionCoeff))
            {
                cout << "Intrinsics of cam 1 out of range after BA! Discarding refined values." << endl;
                t_initial.copyTo(t.getMat());
                return false;
            }
            if (stereo_data.cam2.HasBogusParams(minFrat, maxFrat, maxDistortionCoeff))
            {
                cout << "Intrinsics of cam 2 out of range after BA! Discarding refined values." << endl;
                t_initial.copyTo(t.getMat());
                return false;
            }
        }

        // stereo_data.t_rel *= t_norm;
        eigen2cv(stereo_data.t_rel, t.getMat());
        stereo_data.QuaternionToRotMat();
        eigen2cv(stereo_data.R_rel, R.getMat());

        convertPrecisionMat(t, CERES_RESULT_CONVERT_PRECISION);
        convertPrecisionMat(R, CERES_RESULT_CONVERT_PRECISION);

        if (normalize3Dpts && !Q_normalized.empty())
        {
            undoShiftScaleTranslationVec(t, R, shift, scale);
        }

        if (!fixFocal && pointsInImgCoords)
        {
            if (fxEqFy_)
            {
                K1_.at<double>(0, 0) = convertPrecisionRet(stereo_data.cam1.FocalLength(), CERES_RESULT_CONVERT_PRECISION);
                K1_.at<double>(1, 1) = convertPrecisionRet(stereo_data.cam1.FocalLength(), CERES_RESULT_CONVERT_PRECISION);
                K2_.at<double>(0, 0) = convertPrecisionRet(stereo_data.cam2.FocalLength(), CERES_RESULT_CONVERT_PRECISION);
                K2_.at<double>(1, 1) = convertPrecisionRet(stereo_data.cam2.FocalLength(), CERES_RESULT_CONVERT_PRECISION);
            }
            else
            {
                K1_.at<double>(0, 0) = convertPrecisionRet(stereo_data.cam1.FocalLengthX(), CERES_RESULT_CONVERT_PRECISION);
                K1_.at<double>(1, 1) = convertPrecisionRet(stereo_data.cam1.FocalLengthY(), CERES_RESULT_CONVERT_PRECISION);
                K2_.at<double>(0, 0) = convertPrecisionRet(stereo_data.cam2.FocalLengthX(), CERES_RESULT_CONVERT_PRECISION);
                K2_.at<double>(1, 1) = convertPrecisionRet(stereo_data.cam2.FocalLengthY(), CERES_RESULT_CONVERT_PRECISION);
            }
        }

        if (!fixPrincipalPt && pointsInImgCoords)
        {
            K1_.at<double>(0, 2) = convertPrecisionRet(stereo_data.cam1.PrincipalPointX(), CERES_RESULT_CONVERT_PRECISION);
            K1_.at<double>(1, 2) = convertPrecisionRet(stereo_data.cam1.PrincipalPointY(), CERES_RESULT_CONVERT_PRECISION);
            K2_.at<double>(0, 2) = convertPrecisionRet(stereo_data.cam2.PrincipalPointX(), CERES_RESULT_CONVERT_PRECISION);
            K2_.at<double>(1, 2) = convertPrecisionRet(stereo_data.cam2.PrincipalPointY(), CERES_RESULT_CONVERT_PRECISION);
        }

        if (!fixDistortion && pointsInImgCoords && !dist1.empty() && !dist2.empty()){
            Mat dist1_o = dist1.getMat();
            Mat dist2_o = dist2.getMat();
            const vector<size_t> &cam1_dist_pars_idx = stereo_data.cam1.ExtraParamsIdxs();
            const vector<size_t> &cam2_dist_pars_idx = stereo_data.cam2.ExtraParamsIdxs();
            int idx = 0;
            for (const auto &idx_p : cam1_dist_pars_idx)
            {
                dist1_o.at<double>(idx++) = convertPrecisionRet(stereo_data.cam1.Params(idx_p), CERES_RESULT_CONVERT_PRECISION);
            }
            idx = 0;
            for (const auto &idx_p : cam2_dist_pars_idx)
            {
                dist2_o.at<double>(idx++) = convertPrecisionRet(stereo_data.cam2.Params(idx_p), CERES_RESULT_CONVERT_PRECISION);
            }
        }

        if (Q.needed() && !optimCalibrationOnly)
        {
            Mat Q_ = Q.getMat();
            double scale = 1.0;
            cv::Mat shift;
            if (normalize3Dpts)
            {
                getScaleAndShiftFromMatrix(M, shift, scale);
                shift = shift.t();
            }
            const int nr_Q = Q_.rows;
            if(mask.empty()){                
                for (int i = 0; i < nr_Q; ++i){
                    Q_.at<double>(i, 0) = convertPrecisionRet(stereo_data.points3d.at(i).x() / t_norm2, CERES_RESULT_CONVERT_PRECISION);
                    Q_.at<double>(i, 1) = convertPrecisionRet(stereo_data.points3d.at(i).y() / t_norm2, CERES_RESULT_CONVERT_PRECISION);
                    Q_.at<double>(i, 2) = convertPrecisionRet(stereo_data.points3d.at(i).z() / t_norm2, CERES_RESULT_CONVERT_PRECISION);
                    if (normalize3Dpts)
                    {
                        cv::Mat qt = Q_.row(i);
                        undoShiftScale3DPoint_T(qt, shift, scale);
                    }
                }
            }else{
                int idx = 0;
                for (int i = 0; i < nr_Q; ++i){
                    if(mask_tmp.at<unsigned char>(i)){
                        Q_.at<double>(i, 0) = convertPrecisionRet(stereo_data.points3d.at(idx).x() / t_norm2, CERES_RESULT_CONVERT_PRECISION);
                        Q_.at<double>(i, 1) = convertPrecisionRet(stereo_data.points3d.at(idx).y() / t_norm2, CERES_RESULT_CONVERT_PRECISION);
                        Q_.at<double>(i, 2) = convertPrecisionRet(stereo_data.points3d.at(idx).z() / t_norm2, CERES_RESULT_CONVERT_PRECISION);
                        if (normalize3Dpts)
                        {
                            cv::Mat qt = Q_.row(i);
                            undoShiftScale3DPoint_T(qt, shift, scale);
                        }
                        idx++;
                    }
                }
            }
        }

        return true;
    }

    bool refineMultCamBA(cv::InputArray ps,
                         cv::InputArray map3D,
                         cv::InputOutputArray Rs,
                         cv::InputOutputArray ts,
                         cv::InputOutputArray Qs,
                         cv::InputOutputArray Ks,
                         const std::vector<size_t> &img_2_cam_idx,
                         const cv::Size &img_Size,
                         cv::InputArray masks_corr,
                         cv::InputArray mask_Q,
                         const double angleThresh,
                         const double t_norm_tresh,
                         const LossFunctionToUse loss,
                         const MethodToUse method,
                         const double toleranceMultiplier,
                         const double th,
                         const bool fixFocal,
                         const bool fixPrincipalPt,
                         const bool fixDistortion,
                         const bool distortion_damping,
                         const bool optimCalibrationOnly,
                         const bool fxEqFy,
                         const bool keepScaling,
                         const bool normalize3Dpts,
                         const CamMatDamping *K_damping,
                         cv::InputOutputArray dists,
                         const std::vector<size_t> *constant_3d_point_idx,
                         const std::vector<size_t> *constant_cam_idx,
                         const std::vector<size_t> *constant_pose_idx,
                         int cpu_count,
                         int verbose)
    {
        if(ps.empty() || Qs.empty() || map3D.empty() || Rs.empty() || ts.empty() || Ks.empty())
        {
            cerr << "Some input variables to BA are empty. Skipping!" << endl;
            return false;
        }
        if (!ps.isMatVector() || !map3D.isMatVector() || !Rs.isMatVector() || !ts.isMatVector() || !Ks.isMatVector() || (!masks_corr.empty() && !masks_corr.isMatVector()) || (!dists.empty() && !dists.isMatVector()))
        {
            cerr << "Inputs must be of type vector<Mat>. Skipping!" << endl;
            return false;
        }
        if (!mask_Q.empty() && !mask_Q.isMat())
        {
            cerr << "Input mask_Q must be of type Mat. Skipping!" << endl;
            return false;
        }
        if (!Qs.isMat())
        {
            cerr << "Input Qs must be of type Mat. Skipping!" << endl;
            return false;
        }
        CV_Assert(Qs.type() == CV_64FC1);
        Mat Q = Qs.getMat();
        int nr_Qs = Q.size().height;
        if (nr_Qs < 15)
        {
            cerr << "Too less 3D points. Skipping!" << endl;
            return false;
        }

        Mat mask_Q_;
        bool haveQMasks = false;
        if (!mask_Q.empty() && mask_Q.size().area() != nr_Qs)
        {
            cerr << "Input mask mask_Q of type Mat must be of same size as Qs. Skipping!" << endl;
            return false;
        }
        if (!mask_Q.empty()){
            mask_Q_ = mask_Q.getMat();
            haveQMasks = true;
        }
        else
        {
            mask_Q_ = Mat::ones(1, nr_Qs, CV_8UC1);
        }

        vector<Mat> p2D_vec, map3D_vec, R_mvec, t_mvec, K_vec;
        ps.getMatVector(p2D_vec);
        map3D.getMatVector(map3D_vec);
        Rs.getMatVector(R_mvec);
        ts.getMatVector(t_mvec);
        Ks.getMatVector(K_vec);
        const size_t vecSi = p2D_vec.size();
        if (map3D_vec.size() != vecSi || R_mvec.size() != vecSi || t_mvec.size() != vecSi || img_2_cam_idx.size() != vecSi)
        {
            cerr << "Image inputs of type vector<Mat> must be of same size. Skipping!" << endl;
            return false;
        }
        const size_t camSi = K_vec.size();
        bool haveMasks = false;
        vector<Mat> kpMask_vec;
        if (!masks_corr.empty())
        {
            masks_corr.getMatVector(kpMask_vec);
            if (kpMask_vec.size() != vecSi)
            {
                cerr << "Input masks of type vector<Mat> must be of same size as other inputs. Skipping!" << endl;
                return false;
            }
            for (size_t i = 0; i < kpMask_vec.size(); ++i)
            {
                const int mSi = kpMask_vec[i].size().area();
                const int pSi = p2D_vec[i].size().height;
                if (pSi != mSi)
                {
                    cerr << "Input mask " << i << " does not have the same size as corresponding 2D features. Skipping BA!" << endl;
                    return false;
                }
            }
            haveMasks = true;
        }
        
        for (size_t i = 0; i < map3D_vec.size(); ++i)
        {
            const int mapSi = map3D_vec[i].size().area();
            const int ones = cv::countNonZero(map3D_vec[i]);
            int pSi = p2D_vec[i].size().height;
            CV_Assert(p2D_vec[i].type() == CV_64FC1);
            if (haveMasks)
            {
                pSi = kpMask_vec[i].size().area();
            }else{
                kpMask_vec.emplace_back(Mat::ones(1, pSi, CV_8UC1));
            }
            if (nr_Qs != mapSi)
            {
                cerr << "3D map " << i << " does not have the same size as number of available 3D points. Skipping BA!" << endl;
                return false;
            }
            if (ones != pSi)
            {
                cerr << "3D map " << i << " does not contain the same number of non-zero values as provided 2D correspondences. Skipping BA!" << endl;
                return false;
            }
            if (map3D_vec[i].type() != CV_8SC1)
            {
                cerr << "3D map " << i << " must be of type char. Skipping BA!" << endl;
                return false;
            }
        }
        vector<Mat> dist_mvec;
        bool haveDists = false;
        if (!dists.empty())
        {
            dists.getMatVector(dist_mvec);
            if (dist_mvec.size() != camSi)
            {
                cerr << "Input dists of type vector<Mat> must be of same size as other camera inputs. Skipping!" << endl;
                return false;
            }
            haveDists = true;
        }

        Mat Q_normalized, M;
        vector<Mat> t_mvec_normalized;
        double scale = 1.0;
        cv::Mat shift;
        if (normalize3Dpts)
        {
            Q.copyTo(Q_normalized);
            for(const auto &t : t_mvec){
                t_mvec_normalized.emplace_back(t.clone());
            }
            M = normalize3Dpts_GetShiftScale(Q_normalized);
            getScaleAndShiftFromMatrix(M, shift, scale);
            shiftScaleTranslationVecs(t_mvec_normalized, R_mvec, M);
        }else{
            t_mvec_normalized = t_mvec;
            Q_normalized = Q;
        }

        BundleAdjustmentOptions options;
        setOptionsCostfunction(options, loss, th);
        setOptionsSolveMethod(options, method);
        options.solver_options.parameter_tolerance = 1e-8;
        options.solver_options.gradient_tolerance = 1e-8;
        options.solver_options.function_tolerance = 1e-6;
        if (toleranceMultiplier >= 1e-3 && toleranceMultiplier <= 1e4)
        {
            options.solver_options.parameter_tolerance *= toleranceMultiplier;
            options.solver_options.gradient_tolerance *= toleranceMultiplier;
            options.solver_options.function_tolerance *= toleranceMultiplier;
            if (toleranceMultiplier > 9.0)
            {
                options.solver_options.min_trust_region_radius = 1e-9;
                options.solver_options.max_trust_region_radius = 1e10;
            }
        }
        if (!verbose)
        {
            options.solver_options.minimizer_progress_to_stdout = false;
            options.solver_options.logging_type = ceres::SILENT;
            options.print_summary = false;
        }
        else if (verbose == 1)
        {
            options.solver_options.minimizer_progress_to_stdout = false;
            options.solver_options.logging_type = ceres::SILENT;
            options.print_summary = true;
        }
        else
        {
            options.solver_options.minimizer_progress_to_stdout = true;
            options.solver_options.logging_type = ceres::PER_MINIMIZER_ITERATION;
            options.print_summary = true;
        }
        options.CeresCPUcnt = cpu_count;
        options.refine_focal_length = !fixFocal;
        options.refine_principal_point = !fixPrincipalPt;
        options.refine_extra_params = !fixDistortion;

        GlobalBAData ba_data;        
        vector<bool> fxEqFy_all(camSi, fxEqFy);
        const int max_img_size = std::max(img_Size.width, img_Size.height);
        vector<double> focal_length_ratios;

        for (size_t i = 0; i < camSi; ++i){
            Mat &Kx = K_vec[i];
            colmap::Camera cam;
            double focal_length_ratio;
            bool modelSupported = false;
            bool fxEqFy_all_i = fxEqFy;
            if (haveDists)
            {
                modelSupported = getCameraParametersColmap(Kx, max_img_size, img_Size, fxEqFy, cam, focal_length_ratio, fxEqFy_all_i, dist_mvec[i]);
            }else{
                modelSupported = getCameraParametersColmap(Kx, max_img_size, img_Size, fxEqFy, cam, focal_length_ratio, fxEqFy_all_i);
            }
            if (!modelSupported)
            {
                return false;
            }
            focal_length_ratios.emplace_back(focal_length_ratio);
            fxEqFy_all[i] = fxEqFy_all_i;
            ba_data.cams.emplace_back(move(cam));
        }        
        ba_data.constant_cams = vector<bool>(camSi, false);
        if (constant_cam_idx && !constant_cam_idx->empty())
        {
            for(const auto &i : *constant_cam_idx){
                ba_data.constant_cams[i] = true;
            }
        }

        if (optimCalibrationOnly)
        {
            int nr_elems = nr_Qs;
            if (haveQMasks)
            {
                nr_elems = cv::countNonZero(mask_Q_);
            }
            ba_data.constant_points3d = vector<size_t>(nr_elems);
            iota(ba_data.constant_points3d.begin(), ba_data.constant_points3d.end(), 0);
        }
        else if (constant_3d_point_idx && !constant_3d_point_idx->empty())
        {
            std::vector<size_t> constant_3d_point_idx_tmp;
            if (!haveQMasks)
            {
                constant_3d_point_idx_tmp = *constant_3d_point_idx;
            }
            else
            {
                std::vector<size_t> constant_3d_point_idx_tmp2 = *constant_3d_point_idx;
                std::sort(constant_3d_point_idx_tmp2.begin(), constant_3d_point_idx_tmp2.end());
                std::vector<size_t>::iterator it = constant_3d_point_idx_tmp2.begin();                
                int miss_cnt = 0;
                for (int i = 0; i < nr_Qs; ++i)
                {
                    int idx = -1;
                    if (it != constant_3d_point_idx_tmp2.end() && i == static_cast<int>(*it))
                    {
                        idx = i - miss_cnt;
                        it++;
                    }
                    if (mask_Q_.at<unsigned char>(i))
                    {
                        if (idx >= 0)
                        {
                            constant_3d_point_idx_tmp.emplace_back(static_cast<size_t>(idx));
                        }
                    }
                    else
                    {
                        miss_cnt++;
                    }
                }
            }
            ba_data.constant_points3d = move(constant_3d_point_idx_tmp);
        }

        vector<int> q_diff_idx(nr_Qs, 0);
        if(haveQMasks){
            int diff_idx = 0;
            for (int i = 0; i < nr_Qs; ++i)
            {
                if (mask_Q_.at<unsigned char>(i)){
                    q_diff_idx[i] = diff_idx;
                    Eigen::Vector3d point3d;
                    cv2eigen(Q_normalized.row(i).t(), point3d);
                    ba_data.points3d.emplace_back(move(point3d));
                }
                else
                {
                    q_diff_idx[i] = -1;
                    diff_idx++;
                }
            }
        }else{
            for (int i = 0; i < nr_Qs; ++i){
                Eigen::Vector3d point3d;
                cv2eigen(Q_normalized.row(i).t(), point3d);
                ba_data.points3d.emplace_back(move(point3d));
            }
        }

        vector<bool> img_has_3d_pts(vecSi, true);
        for (size_t i = 0; i < vecSi; ++i)
        {
            Mat &qmap = map3D_vec.at(i);
            Mat &projections = p2D_vec.at(i);
            Mat &proj_mask = kpMask_vec.at(i);
            std::vector<Eigen::Vector2d> corrs;
            std::vector<size_t> points3d_idx;
            int x_idx = 0;
            for (int j = 0; j < nr_Qs; ++j)
            {
                if(qmap.at<char>(j)){
                    int &qd_idx = q_diff_idx.at(j);
                    if (qd_idx >= 0 && proj_mask.at<unsigned char>(x_idx))
                    {
                        corrs.emplace_back(Eigen::Vector2d(projections.at<double>(x_idx, 0), projections.at<double>(x_idx, 1)));
                        points3d_idx.push_back(static_cast<size_t>(j - qd_idx));
                    }
                    x_idx++;
                }
            }
            if (corrs.empty()){
                img_has_3d_pts[i] = false;
            }
            else{
                ba_data.corrs.emplace_back(move(corrs));
                ba_data.points3d_idx.emplace_back(move(points3d_idx));
            }                    
        }

        std::vector<Eigen::Vector3d> t_normed_init;
        std::vector<double> t_norms_init;
        for (size_t i = 0; i < vecSi; ++i)
        {
            if (img_has_3d_pts[i]){
                Eigen::Matrix3d R;
                cv::cv2eigen(R_mvec[i], R);
                ba_data.Rs.emplace_back(move(R));
                Eigen::Vector3d t;
                if (t_mvec_normalized[i].cols > t_mvec_normalized[i].rows)
                {
                    t_mvec_normalized[i] = t_mvec_normalized[i].t();
                }
                cv::cv2eigen(t_mvec_normalized[i], t);
                ba_data.ts.push_back(t);
                double t_norm = t.norm();
                if(abs(t_norm) < 1e-4){
                    t_norm = 1.0;
                }
                t_norms_init.push_back(t_norm);
                t_normed_init.emplace_back(t / t_norm);
                ba_data.imgs_to_cam_idx.push_back(img_2_cam_idx[i]);
                ba_data.constant_poses.emplace_back(false);
            }
        }

        if (constant_pose_idx && !constant_pose_idx->empty()){
            size_t miss_idx = 0;
            std::vector<size_t> constant_pose_idx_tmp = *constant_pose_idx;
            sort(constant_pose_idx_tmp.begin(), constant_pose_idx_tmp.end());
            std::vector<size_t>::iterator it = constant_pose_idx_tmp.begin();
            for (size_t i = 0; i < vecSi; ++i)
            {
                if (it != constant_pose_idx_tmp.end() && i == *it)
                {
                    ba_data.constant_poses[i - miss_idx] = true;
                    it++;
                }
                if (!img_has_3d_pts[i]){
                    miss_idx++;
                }
            }
        }

        ba_data.getRelQuaternions();
        std::vector<Eigen::Vector4d> quats_initial = ba_data.quats;

        ba_data.dist_damp = distortion_damping && !fixDistortion && haveDists;

        if (K_damping && K_damping->damping)
        {
            const double cx_init = static_cast<double>(img_Size.width) / 2.;
            const double cy_init = static_cast<double>(img_Size.height) / 2.;
            CamMatDamping damp_tmp = *K_damping;
            if (fixFocal && K_damping->useDampingF())
            {
                damp_tmp.disableDampingF();
            }
            if ((fxEqFy || fixFocal) && K_damping->useDampingFxFy())
            {
                damp_tmp.disableDampingFxFy();
            }
            if (fixPrincipalPt && K_damping->useDampingCxCy())
            {
                damp_tmp.disableDampingCxCy();
            }
            if (damp_tmp.damping){
                ba_data.camMat_damp = damp_tmp.damping;
                for(const auto &ci : ba_data.cams){
                    ba_data.camMatDampPars.emplace_back(colmap::CamMatDampingSingle(damp_tmp.imgDiag_2,
                                                                                    damp_tmp.focalMid,
                                                                                    damp_tmp.focalRange2,
                                                                                    ci.MeanFocalLength(),
                                                                                    cx_init,
                                                                                    cy_init,
                                                                                    damp_tmp.damping,
                                                                                    damp_tmp.useDampingF() ? damp_tmp.fChangeMax : 0.,
                                                                                    damp_tmp.useDampingFxFy() ? damp_tmp.fxfyRatioDiffMax : 0.,
                                                                                    damp_tmp.useDampingCxCy() ? damp_tmp.cxcyDistMidRatioMax : 0.));
                }
            }
        }

        GlobalBundleAdjuster ba(options, TAKE_NOT_BA_OWNERSHIP);
        if (!ba.Solve(&ba_data))
        {
            return false;
        }

        //Normalize the translation vectors
        double scaling_med = 1.;
        if (!checkPoseDifferenceRescale(ba_data, t_normed_init, t_norms_init, quats_initial, angleThresh, t_norm_tresh, keepScaling, &scaling_med))
        {
            return false;
        }

        if(!checkIntrinsics(ba_data, fixFocal, fixPrincipalPt, fixDistortion, haveDists, focal_length_ratios)){
            return false;
        }

        ba_data.QuaternionToRotMat();
        size_t idx = 0;
        for (size_t i = 0; i < vecSi; ++i){
            if (img_has_3d_pts[i]){
                if(!ba_data.constant_poses[idx]){
                    eigen2cv(ba_data.ts[idx], t_mvec[i]);
                    eigen2cv(ba_data.Rs[idx], R_mvec[i]);
                    convertPrecisionMat(t_mvec[i], CERES_RESULT_CONVERT_PRECISION);
                    convertPrecisionMat(R_mvec[i], CERES_RESULT_CONVERT_PRECISION);
                    if (normalize3Dpts)
                    {
                        undoShiftScaleTranslationVec(t_mvec[i], R_mvec[i], shift, scale);
                    }
                }
                idx++;
            }
        }

        updateIntrinsics(ba_data, fixFocal, fixPrincipalPt, fixDistortion, haveDists, camSi, fxEqFy_all, K_vec, dist_mvec, CERES_RESULT_CONVERT_PRECISION);
        update3Dpoints(ba_data, optimCalibrationOnly, haveQMasks, nr_Qs, mask_Q_, Q, M, keepScaling, &scaling_med, CERES_RESULT_CONVERT_PRECISION);

        return true;
    }

    bool refineMultCamBA(const std::unordered_map<std::pair<int, int>, std::vector<cv::Point2f>, pair_hash, pair_EqualTo> &corrs,
                         const std::unordered_map<std::pair<int, int>, std::vector<size_t>, pair_hash, pair_EqualTo> &map3D,
                         std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &Rs,
                         std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &ts,
                         cv::InputOutputArray Qs,
                         cv::InputOutputArray Ks,
                         const std::unordered_map<std::pair<int, int>, size_t, pair_hash, pair_EqualTo> &cimg_2_cam_idx,
                         const cv::Size &img_Size,
                         const double angleThresh,
                         const double t_norm_tresh,
                         const LossFunctionToUse loss,
                         const MethodToUse method,
                         const double toleranceMultiplier,
                         const double th,
                         const bool fixFocal,
                         const bool fixPrincipalPt,
                         const bool fixDistortion,
                         const bool distortion_damping,
                         const bool optimCalibrationOnly,
                         const bool fxEqFy,
                         const bool keepScaling,
                         const bool normalize3Dpts,
                         const CamMatDamping *K_damping,
                         const std::unordered_map<std::pair<int, int>, std::vector<bool>, pair_hash, pair_EqualTo> *masks_corr,
                         cv::InputArray mask_Q,
                         cv::InputOutputArray dists,
                         const std::vector<size_t> *constant_3d_point_idx,
                         const std::vector<size_t> *constant_cam_idx,
                         const std::unordered_set<std::pair<int, int>, pair_hash, pair_EqualTo> *constant_pose_idx,
                         int cpu_count,
                         int verbose,
                         std::string *dbg_info)
    {
        if (corrs.empty() || Qs.empty() || map3D.empty() || Rs.empty() || ts.empty() || Ks.empty())
        {
            cerr << "Some input variables to BA are empty. Skipping!" << endl;
            return false;
        }
        if (!Ks.isMatVector() || (!dists.empty() && !dists.isMatVector()))
        {
            cerr << "Inputs must be of type vector<Mat>. Skipping!" << endl;
            return false;
        }
        if (!mask_Q.empty() && !mask_Q.isMat())
        {
            cerr << "Input mask_Q must be of type Mat. Skipping!" << endl;
            return false;
        }
        if (!Qs.isMat())
        {
            cerr << "Input Qs must be of type Mat. Skipping!" << endl;
            return false;
        }
        CV_Assert(Qs.type() == CV_64FC1);
        Mat Q = Qs.getMat();
        int nr_Qs = Q.size().height;
        if (nr_Qs < 15)
        {
            cerr << "Too less 3D points. Skipping!" << endl;
            return false;
        }

        Mat mask_Q_;
        bool haveQMasks = false;
        if (!mask_Q.empty() && mask_Q.size().area() != nr_Qs)
        {
            cerr << "Input mask mask_Q of type Mat must be of same size as Qs. Skipping!" << endl;
            return false;
        }
        if (!mask_Q.empty())
        {
            mask_Q_ = mask_Q.getMat();
            haveQMasks = true;
        }
        else
        {
            mask_Q_ = Mat::ones(1, nr_Qs, CV_8UC1);
        }

        vector<Mat> K_vec;
        Ks.getMatVector(K_vec);
        const size_t vecSi = corrs.size();
        if (map3D.size() != vecSi)
        {
            cerr << "Size of map3D must be equal to corrs. Skipping!" << endl;
            return false;
        }
        if (Rs.size() != vecSi || ts.size() != vecSi || cimg_2_cam_idx.size() != vecSi)
        {
            cerr << "Camera parameters and poses of real cams must be of same size. Skipping!" << endl;
            return false;
        }
        const size_t camSi = K_vec.size();
        bool haveMasks = false;
        if (masks_corr)
        {
            for (const auto &i : *masks_corr)
            {
                const int mSi = i.second.size();
                const int pSi = corrs.at(i.first).size();
                if (pSi != mSi)
                {
                    cerr << "Input mask (c" << i.first.first << ", i" << i.first.second << ") does not have the same size as corresponding 2D features. Skipping BA!" << endl;
                    return false;
                }
            }
            haveMasks = true;
        }

        for (const auto &i : map3D)
        {
            const size_t idx_max = *std::max_element(i.second.begin(), i.second.end());
            if (idx_max >= static_cast<size_t>(nr_Qs))
            {
                cerr << "Index of 3D map (c" << i.first.first << ", i" << i.first.second << ") out of bounds. Skipping BA!" << endl;
                return false;
            }
        }
        vector<Mat> dist_mvec;
        bool haveDists = false;
        if (!dists.empty())
        {
            dists.getMatVector(dist_mvec);
            if (dist_mvec.size() != camSi)
            {
                cerr << "Input dists of type vector<Mat> must be of same size as other camera inputs. Skipping!" << endl;
                return false;
            }
            haveDists = true;
        }

        Mat Q_normalized, M;
        std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> ts_normalized;
        double scale = 1.0;
        cv::Mat shift;
        if (normalize3Dpts)
        {
            Q.copyTo(Q_normalized);
            for (const auto &t : ts)
            {
                ts_normalized.emplace(t.first, t.second.clone());
            }
            M = normalize3Dpts_GetShiftScale(Q_normalized);
            getScaleAndShiftFromMatrix(M, shift, scale);
            shiftScaleTranslationVecs(ts_normalized, Rs, M);
        }
        else
        {
            ts_normalized = ts;
            Q_normalized = Q;
        }

        BundleAdjustmentOptions options;
        setOptionsCostfunction(options, loss, th);
        setOptionsSolveMethod(options, method);
        options.solver_options.parameter_tolerance = 1e-8;
        options.solver_options.gradient_tolerance = 1e-8;
        options.solver_options.function_tolerance = 1e-6;
        if (toleranceMultiplier >= 1e-3 && toleranceMultiplier <= 1e4)
        {
            options.solver_options.parameter_tolerance *= toleranceMultiplier;
            options.solver_options.gradient_tolerance *= toleranceMultiplier;
            options.solver_options.function_tolerance *= toleranceMultiplier;
            if (toleranceMultiplier > 9.0)
            {
                options.solver_options.min_trust_region_radius = 1e-9;
                options.solver_options.max_trust_region_radius = 1e10;
            }
        }
        if (!verbose)
        {
            options.solver_options.minimizer_progress_to_stdout = false;
            options.solver_options.logging_type = ceres::SILENT;
            options.print_summary = false;
        }
        else if (verbose == 1)
        {
            options.solver_options.minimizer_progress_to_stdout = false;
            options.solver_options.logging_type = ceres::SILENT;
            options.print_summary = true;
        }
        else
        {
            options.solver_options.minimizer_progress_to_stdout = true;
            options.solver_options.logging_type = ceres::PER_MINIMIZER_ITERATION;
            options.print_summary = true;
        }
        options.CeresCPUcnt = cpu_count;
        options.refine_focal_length = !fixFocal;
        options.refine_principal_point = !fixPrincipalPt;
        options.refine_extra_params = !fixDistortion;

        GlobalBAData ba_data;
        vector<bool> fxEqFy_all(camSi, fxEqFy);
        const int max_img_size = std::max(img_Size.width, img_Size.height);
        vector<double> focal_length_ratios;

        for (size_t i = 0; i < camSi; ++i)
        {
            Mat &Kx = K_vec[i];
            colmap::Camera cam;
            double focal_length_ratio;
            bool modelSupported = false;
            bool fxEqFy_all_i = fxEqFy;
            if (haveDists)
            {
                modelSupported = getCameraParametersColmap(Kx, max_img_size, img_Size, fxEqFy, cam, focal_length_ratio, fxEqFy_all_i, dist_mvec[i]);
            }
            else
            {
                modelSupported = getCameraParametersColmap(Kx, max_img_size, img_Size, fxEqFy, cam, focal_length_ratio, fxEqFy_all_i);
            }
            if (!modelSupported)
            {
                return false;
            }
            focal_length_ratios.emplace_back(focal_length_ratio);
            fxEqFy_all[i] = fxEqFy_all_i;
            ba_data.cams.emplace_back(move(cam));
        }
        ba_data.constant_cams = vector<bool>(camSi, false);
        if (constant_cam_idx && !constant_cam_idx->empty())
        {
            for (const auto &i : *constant_cam_idx)
            {
                ba_data.constant_cams[i] = true;
            }
        }

        if (optimCalibrationOnly)
        {
            int nr_elems = nr_Qs;
            if (haveQMasks)
            {
                nr_elems = cv::countNonZero(mask_Q_);
            }
            ba_data.constant_points3d = vector<size_t>(nr_elems);
            iota(ba_data.constant_points3d.begin(), ba_data.constant_points3d.end(), 0);
        }
        else if (constant_3d_point_idx && !constant_3d_point_idx->empty())
        {
            std::vector<size_t> constant_3d_point_idx_tmp;
            if (!haveQMasks)
            {
                constant_3d_point_idx_tmp = *constant_3d_point_idx;
            }
            else
            {
                std::vector<size_t> constant_3d_point_idx_tmp2 = *constant_3d_point_idx;
                std::sort(constant_3d_point_idx_tmp2.begin(), constant_3d_point_idx_tmp2.end());
                std::vector<size_t>::iterator it = constant_3d_point_idx_tmp2.begin();
                int miss_cnt = 0;
                for (int i = 0; i < nr_Qs; ++i)
                {
                    int idx = -1;
                    if (it != constant_3d_point_idx_tmp2.end() && i == static_cast<int>(*it))
                    {
                        idx = i - miss_cnt;
                        it++;
                    }
                    if (mask_Q_.at<unsigned char>(i))
                    {
                        if (idx >= 0)
                        {
                            constant_3d_point_idx_tmp.emplace_back(static_cast<size_t>(idx));
                        }
                    }
                    else
                    {
                        miss_cnt++;
                    }
                }
            }
            ba_data.constant_points3d = move(constant_3d_point_idx_tmp);
        }

        vector<int> q_diff_idx(nr_Qs, 0);
        if (haveQMasks)
        {
            int diff_idx = 0;
            for (int i = 0; i < nr_Qs; ++i)
            {
                if (mask_Q_.at<unsigned char>(i))
                {
                    q_diff_idx[i] = diff_idx;
                    Eigen::Vector3d point3d;
                    cv2eigen(Q_normalized.row(i).t(), point3d);
                    ba_data.points3d.emplace_back(move(point3d));
                }
                else
                {
                    q_diff_idx[i] = -1;
                    diff_idx++;
                }
            }
        }
        else
        {
            for (int i = 0; i < nr_Qs; ++i)
            {
                Eigen::Vector3d point3d;
                cv2eigen(Q_normalized.row(i).t(), point3d);
                ba_data.points3d.emplace_back(move(point3d));
            }
        }

        vector<bool> img_has_3d_pts(vecSi, true);
        size_t cnt = 0;
        vector<Mat> R_mvec, t_mvec;
        std::vector<size_t> img_2_cam_idx;
        std::vector<size_t> constant_pose_idx2;
        std::unordered_map<size_t, std::pair<int, int>> ci_idx_save;
        for (const auto &i : corrs)
        {
            vector<bool> kpMask;
            if (haveMasks && (masks_corr->find(i.first) != masks_corr->end()))
            {
                kpMask = masks_corr->at(i.first);
            }
            else
            {
                kpMask = vector<bool>(i.second.size(), true);
            }
            std::vector<Eigen::Vector2d> corrs_eigen;
            std::vector<size_t> points3d_idx;
            int x_idx = 0;
            for (const auto &q_idx : map3D.at(i.first))
            {
                if (kpMask.at(x_idx) && mask_Q_.at<unsigned char>(static_cast<int>(q_idx)))
                {
                    int &qd_idx = q_diff_idx.at(q_idx);
                    if (qd_idx >= 0)
                    {
                        corrs_eigen.emplace_back(Eigen::Vector2d(static_cast<double>(i.second.at(x_idx).x), static_cast<double>(i.second.at(x_idx).y)));
                        points3d_idx.push_back(q_idx - static_cast<size_t>(qd_idx));
                    }
                }
                x_idx++;
            }
            if (!corrs_eigen.empty())
            {
                ba_data.corrs.emplace_back(move(corrs_eigen));
                ba_data.points3d_idx.emplace_back(move(points3d_idx));
                ci_idx_save.emplace(cnt, i.first);
                if (constant_pose_idx && !constant_pose_idx->empty() && (constant_pose_idx->find(i.first) != constant_pose_idx->end()))
                {
                    ba_data.constant_poses.emplace_back(true);
                }else{
                    ba_data.constant_poses.emplace_back(false);
                }
            }
            else
            {
                cout << "No match available for camera (c" << i.first.first << ", i" << i.first.second << ")." << endl;
                img_has_3d_pts[cnt] = false;
            }
            R_mvec.emplace_back(Rs.at(i.first));
            t_mvec.emplace_back(ts_normalized.at(i.first));
            img_2_cam_idx.emplace_back(cimg_2_cam_idx.at(i.first));
            cnt++;
        }

        std::vector<Eigen::Vector3d> t_normed_init;
        std::vector<double> t_norms_init;
        for (size_t i = 0; i < vecSi; ++i)
        {
            if (img_has_3d_pts[i]){
                Eigen::Matrix3d R;
                cv::cv2eigen(R_mvec.at(i), R);
                ba_data.Rs.emplace_back(move(R));
                Eigen::Vector3d t;
                if (t_mvec.at(i).cols > t_mvec.at(i).rows)
                {
                    t_mvec.at(i) = t_mvec.at(i).t();
                }
                cv::cv2eigen(t_mvec.at(i), t);
                ba_data.ts.push_back(t);
                double t_norm = t.norm();
                if (abs(t_norm) < 1e-4)
                {
                    t_norm = 1.0;
                }
                t_norms_init.push_back(t_norm);
                t_normed_init.emplace_back(t / t_norm);
                ba_data.imgs_to_cam_idx.push_back(img_2_cam_idx[i]);
            }
        }
        
        ba_data.getRelQuaternions();
        std::vector<Eigen::Vector4d> quats_initial = ba_data.quats;

        ba_data.dist_damp = distortion_damping && !fixDistortion && haveDists;

        if (K_damping && K_damping->damping)
        {
            const double cx_init = static_cast<double>(img_Size.width) / 2.;
            const double cy_init = static_cast<double>(img_Size.height) / 2.;
            CamMatDamping damp_tmp = *K_damping;
            if (fixFocal && K_damping->useDampingF())
            {
                damp_tmp.disableDampingF();
            }
            if ((fxEqFy || fixFocal) && K_damping->useDampingFxFy())
            {
                damp_tmp.disableDampingFxFy();
            }
            if (fixPrincipalPt && K_damping->useDampingCxCy())
            {
                damp_tmp.disableDampingCxCy();
            }
            if (damp_tmp.damping)
            {
                ba_data.camMat_damp = damp_tmp.damping;
                for (const auto &ci : ba_data.cams)
                {
                    ba_data.camMatDampPars.emplace_back(colmap::CamMatDampingSingle(damp_tmp.imgDiag_2,
                                                                                    damp_tmp.focalMid,
                                                                                    damp_tmp.focalRange2,
                                                                                    ci.MeanFocalLength(),
                                                                                    cx_init,
                                                                                    cy_init,
                                                                                    damp_tmp.damping,
                                                                                    damp_tmp.useDampingF() ? damp_tmp.fChangeMax : 0.,
                                                                                    damp_tmp.useDampingFxFy() ? damp_tmp.fxfyRatioDiffMax : 0.,
                                                                                    damp_tmp.useDampingCxCy() ? damp_tmp.cxcyDistMidRatioMax : 0.));
                }
            }
        }

        GlobalBundleAdjuster ba(options, TAKE_NOT_BA_OWNERSHIP);
        if (!ba.Solve(&ba_data, dbg_info))
        {
            cerr << "Bundle adustment failed (bad input data)." << endl;
            return false;
        }

        //Normalize the translation vectors
        double scaling_med = 1.;
        if (!checkPoseDifferenceRescale(ba_data, t_normed_init, t_norms_init, quats_initial, angleThresh, t_norm_tresh, keepScaling, &scaling_med))
        {
            return false;
        }

        if (!checkIntrinsics(ba_data, fixFocal, fixPrincipalPt, fixDistortion, haveDists, focal_length_ratios))
        {
            return false;
        }

        ba_data.QuaternionToRotMat();
        size_t idx = 0;
        for (size_t i = 0; i < vecSi; ++i)
        {
            if (img_has_3d_pts[i])
            {
                if (!ba_data.constant_poses[idx])
                {
                    cv::Mat R_tmp, t_tmp;
                    const std::pair<int, int> &ci = ci_idx_save.at(i);
                    eigen2cv(ba_data.ts.at(idx), t_tmp);
                    eigen2cv(ba_data.Rs.at(idx), R_tmp);
                    convertPrecisionMat(t_tmp, CERES_RESULT_CONVERT_PRECISION);
                    convertPrecisionMat(R_tmp, CERES_RESULT_CONVERT_PRECISION);
                    if (normalize3Dpts)
                    {
                        undoShiftScaleTranslationVec(t_tmp, R_tmp, shift, scale);
                    }
                    R_tmp.copyTo(Rs.at(ci));
                    t_tmp.copyTo(ts.at(ci));
                }
                idx++;
            }
        }

        updateIntrinsics(ba_data, fixFocal, fixPrincipalPt, fixDistortion, haveDists, camSi, fxEqFy_all, K_vec, dist_mvec, CERES_RESULT_CONVERT_PRECISION);
        // vector<Mat> &dist_mvec_ = *(std::vector<Mat>*)dists.getObj();
        update3Dpoints(ba_data, optimCalibrationOnly, haveQMasks, nr_Qs, mask_Q_, Q, M, keepScaling, &scaling_med, CERES_RESULT_CONVERT_PRECISION);

        return true;
    }

    bool refineCamsSampsonBA(const std::unordered_map<std::pair<int, int>, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>, pair_hash, pair_EqualTo> &corrs,
                             std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &Rs,
                             std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &ts,
                             std::unordered_map<int, cv::Mat> &Ks,
                             const cv::Size &img_Size,
                             const double angleThresh,
                             const double t_norm_tresh,
                             const LossFunctionToUse loss,
                             const MethodToUse method,
                             const double toleranceMultiplier,
                             const double th,
                             const bool fixFocal,
                             const bool fixPrincipalPt,
                             const bool fixDistortion,
                             const bool distortion_damping,
                             const bool fxEqFy,
                             const CamMatDamping *K_damping,
                             const std::unordered_map<std::pair<int, int>, std::vector<bool>, pair_hash, pair_EqualTo> *masks_corr,
                             std::unordered_map<int, cv::Mat> *dists,
                             const std::vector<int> *constant_cam_idx,
                             const std::unordered_set<std::pair<int, int>, pair_hash, pair_EqualTo> *constant_pose_idx,
                             int cpu_count,
                             int verbose)
    {
        if (corrs.empty() || Rs.empty() || ts.empty() || Ks.empty())
        {
            cerr << "Some input variables to BA are empty. Skipping!" << endl;
            return false;
        }

        const size_t vecSi = corrs.size();
        if (Rs.size() != vecSi || ts.size() != vecSi)
        {
            cerr << "Camera parameters and poses of real cams must be of same size. Skipping!" << endl;
            return false;
        }

        const size_t camSi = Ks.size();
        bool haveMasks = false;
        if (masks_corr)
        {
            for (const auto &i : *masks_corr)
            {
                const int mSi = i.second.size();
                const int pSi = corrs.at(i.first).first.size();
                if (pSi != mSi)
                {
                    cerr << "Input mask (c" << i.first.first << ", i" << i.first.second << ") does not have the same size as corresponding 2D features. Skipping BA!" << endl;
                    return false;
                }
            }
            haveMasks = true;
        }

        bool haveDists = false;
        if (dists)
        {
            if (dists->size() != camSi)
            {
                cerr << "Number of camera matrices must be equal to number of distortion parameter sets. Skipping!" << endl;
                return false;
            }
            haveDists = true;
        }

        BundleAdjustmentOptions options;
        setOptionsCostfunction(options, loss);
        setOptionsSolveMethod(options, method);
        options.solver_options.parameter_tolerance = 1e-8;
        options.solver_options.gradient_tolerance = 1e-8;
        options.solver_options.function_tolerance = 1e-6;
        if (toleranceMultiplier >= 1e-3 && toleranceMultiplier <= 1e4)
        {
            options.solver_options.parameter_tolerance *= toleranceMultiplier;
            options.solver_options.gradient_tolerance *= toleranceMultiplier;
            options.solver_options.function_tolerance *= toleranceMultiplier;
            if (toleranceMultiplier > 9.0)
            {
                options.solver_options.min_trust_region_radius = 1e-9;
                options.solver_options.max_trust_region_radius = 1e10;
            }
        }
        if (!verbose)
        {
            options.solver_options.minimizer_progress_to_stdout = false;
            options.solver_options.logging_type = ceres::SILENT;
            options.print_summary = false;
        }
        else if (verbose == 1)
        {
            options.solver_options.minimizer_progress_to_stdout = false;
            options.solver_options.logging_type = ceres::SILENT;
            options.print_summary = true;
        }
        else
        {
            options.solver_options.minimizer_progress_to_stdout = true;
            options.solver_options.logging_type = ceres::PER_MINIMIZER_ITERATION;
            options.print_summary = true;
        }
        options.CeresCPUcnt = cpu_count;
        options.refine_focal_length = !fixFocal;
        options.refine_principal_point = !fixPrincipalPt;
        options.refine_extra_params = !fixDistortion;

        std::unordered_map<int, colmap::Camera> cams;
        std::unordered_map<int, bool> fxEqFy_all;
        const int max_img_size = std::max(img_Size.width, img_Size.height);
        std::unordered_map<int, double> focal_length_ratios;

        double focal_sum = 0.;
        for (auto &Kx : Ks)
        {
            colmap::Camera cam;
            double focal_length_ratio;
            bool modelSupported = false;
            bool fxEqFy_all_i = fxEqFy;
            if (haveDists)
            {
                modelSupported = getCameraParametersColmap(Kx.second, max_img_size, img_Size, fxEqFy, cam, focal_length_ratio, fxEqFy_all_i, dists->at(Kx.first));
            }
            else
            {
                modelSupported = getCameraParametersColmap(Kx.second, max_img_size, img_Size, fxEqFy, cam, focal_length_ratio, fxEqFy_all_i);
            }
            if (!modelSupported)
            {
                return false;
            }
            focal_length_ratios.emplace(Kx.first, focal_length_ratio);
            fxEqFy_all.emplace(Kx.first, fxEqFy_all_i);
            focal_sum += cam.MeanFocalLength();
            cams.emplace(Kx.first, move(cam));
        }

        focal_sum /= static_cast<double>(cams.size());
        if (th > DBL_EPSILON)
        {
            double th_cam = th / focal_sum;
            options.loss_function_scale = th_cam;
        }
        std::unordered_map<int, bool> constant_cams;
        for (const auto &K : Ks)
        {
            constant_cams.emplace(K.first, false);
        }
        if (constant_cam_idx && !constant_cam_idx->empty())
        {
            for (const auto &i : *constant_cam_idx)
            {
                constant_cams.at(i) = true;
            }
        }
        
        std::unordered_map<std::pair<int, int>, std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>, pair_hash, pair_EqualTo> corrs_eigen;
        std::unordered_map<std::pair<int, int>, bool, pair_hash, pair_EqualTo> constant_poses;
        // std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> Rs_tmp, ts_tmp;
        for (const auto &ms : corrs)
        {
            vector<bool> kpMask;
            if (haveMasks && (masks_corr->find(ms.first) != masks_corr->end()))
            {
                kpMask = masks_corr->at(ms.first);
            }
            else
            {
                kpMask = vector<bool>(ms.second.first.size(), true);
            }
            std::vector<Eigen::Vector2d> corrs_eigen1, corrs_eigen2;
            for (size_t i = 0; i < ms.second.first.size(); i++)
            {
                if (kpMask.at(i))
                {
                    const cv::Point2f &pt1 = ms.second.first.at(i);
                    const cv::Point2f &pt2 = ms.second.second.at(i);
                    corrs_eigen1.emplace_back(Eigen::Vector2d(static_cast<double>(pt1.x), static_cast<double>(pt1.y)));
                    corrs_eigen2.emplace_back(Eigen::Vector2d(static_cast<double>(pt2.x), static_cast<double>(pt2.y)));
                }
            }
            if (!corrs_eigen1.empty())
            {
                corrs_eigen.emplace(ms.first, make_pair(move(corrs_eigen1), move(corrs_eigen2)));
                if (constant_pose_idx && !constant_pose_idx->empty() && (constant_pose_idx->find(ms.first) != constant_pose_idx->end()))
                {
                    constant_poses.emplace(ms.first, true);
                }
                else
                {
                    constant_poses.emplace(ms.first, false);
                }
            }
            else
            {
                cout << "No match available for camera pair " << ms.first.first << "-" << ms.first.second << "." << endl;
            }
        }

        std::unordered_map<std::pair<int, int>, Eigen::Matrix3d, pair_hash, pair_EqualTo> Rs_eigen;
        std::unordered_map<std::pair<int, int>, Eigen::Vector3d, pair_hash, pair_EqualTo> ts_eigen;
        std::unordered_map<int, int> cams_multiple;
        bool allImgsNeeded = true;
        for (const auto &R_cv : Rs)
        {
            Eigen::Matrix3d R;
            cv::cv2eigen(R_cv.second, R);
            Rs_eigen.emplace(R_cv.first, move(R));
            Eigen::Vector3d t;
            if (ts.at(R_cv.first).cols > ts.at(R_cv.first).rows)
            {
                ts.at(R_cv.first) = ts.at(R_cv.first).t();
            }
            ts.at(R_cv.first) /= cv::norm(ts.at(R_cv.first));
            cv::cv2eigen(ts.at(R_cv.first), t);
            ts_eigen.emplace(R_cv.first, move(t));
            if (cams_multiple.find(R_cv.first.first) == cams_multiple.end())
            {
                cams_multiple.emplace(R_cv.first.first, 1);
            }
            else
            {
                cams_multiple.at(R_cv.first.first)++;
            }
            if (cams_multiple.find(R_cv.first.second) == cams_multiple.end())
            {
                cams_multiple.emplace(R_cv.first.second, 1);
            }
            else
            {
                cams_multiple.at(R_cv.first.second)++;
            }
        }

        for(const auto &mult_cnt : cams_multiple){
            if (mult_cnt.second > 2){
                allImgsNeeded = false;
                break;
            }
        }

        bool dist_damp = distortion_damping && !fixDistortion && haveDists;

        RelativePoseBAData ba_data(cams, Rs_eigen, ts_eigen, corrs_eigen, dist_damp);
        std::unordered_map<std::pair<int, int>, Eigen::Vector4d, pair_hash, pair_EqualTo> quats_initial = ba_data.quats;
        ba_data.constant_cams = constant_cams;
        ba_data.constant_poses = constant_poses;

        if (K_damping && K_damping->damping)
        {
            const double cx_init = static_cast<double>(img_Size.width) / 2.;
            const double cy_init = static_cast<double>(img_Size.height) / 2.;
            CamMatDamping damp_tmp = *K_damping;
            if (fixFocal && K_damping->useDampingF())
            {
                damp_tmp.disableDampingF();
            }
            if ((fxEqFy || fixFocal) && K_damping->useDampingFxFy())
            {
                damp_tmp.disableDampingFxFy();
            }
            if (fixPrincipalPt && K_damping->useDampingCxCy())
            {
                damp_tmp.disableDampingCxCy();
            }
            if (damp_tmp.damping)
            {
                ba_data.camMat_damp = damp_tmp.damping;
                for (const auto &ci : ba_data.cams)
                {
                    ba_data.camMatDampPars.emplace(ci.first, colmap::CamMatDampingSingle(damp_tmp.imgDiag_2,
                                                                                         damp_tmp.focalMid,
                                                                                         damp_tmp.focalRange2,
                                                                                         ci.second.MeanFocalLength(),
                                                                                         cx_init,
                                                                                         cy_init,
                                                                                         damp_tmp.damping,
                                                                                         damp_tmp.useDampingF() ? damp_tmp.fChangeMax : 0.,
                                                                                         damp_tmp.useDampingFxFy() ? damp_tmp.fxfyRatioDiffMax : 0.,
                                                                                         damp_tmp.useDampingCxCy() ? damp_tmp.cxcyDistMidRatioMax : 0.));
                }
            }
        }

        RelativePoseBundleAdjuster ba(options, TAKE_NOT_BA_OWNERSHIP, allImgsNeeded);
        if (!ba.Solve(&ba_data))
        {
            cerr << "Bundle adustment failed (bad input data)." << endl;
            return false;
        }

        if (!allImgsNeeded)
        {
            std::unordered_map<std::pair<int, int>, std::vector<std::vector<std::pair<int, int>>>, pair_hash, pair_EqualTo> missing_restore_sequences;
            ba.getNotRefinedImgs(missing_restore_sequences);
            if (!missing_restore_sequences.empty()){
                ba_data.QuaternionToRotMat();
                for (const auto &mrs : missing_restore_sequences){
                    const std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>> &corrs_missed = corrs_eigen.at(mrs.first);
                    std::vector<Eigen::Matrix3d> Es;
                    for(const auto &track : mrs.second){
                        if (track.empty() || track.size() > 3)
                        {
                            continue;
                        }
                        Eigen::Matrix3d E12, R1;
                        if (!ba_data.getRE(track[0], E12, R1)){
                            continue;
                        }
                        if (track.size() == 1)
                        {
                            Es.emplace_back(E12);
                        }
                        else if (track.size() == 2)
                        {
                            Eigen::Matrix3d E23, R2;
                            if (!ba_data.getRE(track[1], E23, R2))
                            {
                                continue;
                            }
                            /* Derived from:
                               E_12 = [t_1]_x R_1 with skew-symmetric matrix [t_1]_x representing cross product
                               E_23 = [t_2]_x R_2
                               E_13 y = [s R_2 t_1 + t2]_x R_2 R_1 y = (s R_2 t_1 + t_2) x (R_2 R_1 y) =
                                      = (s R_2 t_1) x (R_2 R_1 y) + t_2 x R_2 R_1 y =
                                      = R_2 (s t_1 x R_1 y) + t_2 x (R_2 R_1 y) =
                                      = s R_2 [t_1]_x R_1 y + [t_2]_x R_2 R_1 y = 
                                      = s R_2 E_12 y + E_23 R_1 y =
                                      = (s R_2 E_12 + E_23 R_1) y
                               Scale estimation:
                               y'^T E_13 y = y'^T (s R_2 E_12 + E_23 R_1) y = 0
                               0 = y'^T (s R_2 E_12 y + E_23 R_1 y) = s y'^T R_2 E_12 y + y'^T E_23 R_1 y
                               y'^T E_23 R_1 y = -s y'^T R_2 E_12 y 
                               s = -1 * (y'^T E_23 R_1 y) / (y'^T R_2 E_12 y)
                            */
                            Eigen::Matrix3d R2E12 = R2 * E12;
                            Eigen::Matrix3d E23R1 = E23 * R1;
                            Eigen::Matrix3d K1i = ba_data.cams.at(mrs.first.first).CalibrationMatrix().inverse();
                            Eigen::Matrix3d K2i = ba_data.cams.at(mrs.first.second).CalibrationMatrix().inverse();
                            std::vector<double> scales;
                            for (size_t i = 0; i < corrs_missed.first.size(); i++)
                            {
                                Eigen::Vector3d x1h(corrs_missed.first.at(i).x(), corrs_missed.first.at(i).y(), 1.0);
                                Eigen::Vector3d x2h(corrs_missed.second.at(i).x(), corrs_missed.second.at(i).y(), 1.0);
                                Eigen::Vector3d x1 = K1i * x1h;
                                Eigen::Vector3d x2 = K2i * x2h;
                                x1 /= x1.z();
                                x2 /= x2.z();
                                double e12 = calculateEpipolarError(x1, x2, R2E12);
                                if(nearZero(e12, 1e-10)){
                                    continue;
                                }
                                double e23 = calculateEpipolarError(x1, x2, E23R1);
                                double s = -1. * e23 / e12;
                                if(std::abs(s) < 1e-2 || std::abs(s) > 100.){
                                    continue;
                                }
                                scales.emplace_back(s);
                            }
                            double scale_use = 1.0;
                            if(scales.size() > 10){
                                const double s = getMedian(scales);
                                if (!nearZero(s - 1.)){
                                    scale_use = s;
                                }
                            }
                            Eigen::Matrix3d E13 = scale_use * R2E12 + E23R1;
                            Es.emplace_back(E13);
                        }
                        else if (track.size() == 3)
                        {
                            Eigen::Matrix3d E23, R2;
                            if (!ba_data.getRE(track[1], E23, R2))
                            {
                                continue;
                            }
                            Eigen::Matrix3d E34, R4;
                            if (!ba_data.getRE(track[2], E34, R4))
                            {
                                continue;
                            }
                            /* Derived from:
                               E_12 = [t_1]_x R_1 with skew-symmetric matrix [t_1]_x representing cross product
                               E_23 = [t_2]_x R_2
                               E_34 = [t_4]_x R_4
                               R_3 = R_1 R_2
                               E_14 y = (s_1 R_4 E_13 + E_34 R_3) y = s_1 R_4 (s R_2 E_12 + E_23 R_1) y + E_34 R_3 y
                               y'^T E_14 y = s s_1 y'^T R_4 R_2 E_12 y + s_1 y'^T R_4 E_23 R_1 y + y'^T E_34 R_3 y = 0
                               s_2 = s s_1
                               a = y'^T R_4 R_2 E_12 y
                               b = y'^T R_4 E_23 R_1 y
                               c = y'^T E_34 R_3 y
                               a_i s_2 + b_i s_1 + c_i = 0 for each correspondence i
                               s_1 = (c_i - a_i c_(i+1) / a_(i+1)) / (b_i - a_i b_(i+1) / a_(i+1))
                               s_2 = -1 * (b_i s_1 + c_i) / a_i
                            */
                            Eigen::Matrix3d R3 = R1 * R2;
                            Eigen::Matrix3d R4R2E12 = R4 * R2 * E12;
                            Eigen::Matrix3d R4E23R1 = R4 * E23 * R1;
                            Eigen::Matrix3d E34R3 = E34 * R3;
                            Eigen::Matrix3d K1i = ba_data.cams.at(mrs.first.first).CalibrationMatrix().inverse();
                            Eigen::Matrix3d K2i = ba_data.cams.at(mrs.first.second).CalibrationMatrix().inverse();
                            std::vector<std::tuple<double, double, double>> coeffs;
                            for (size_t i = 0; i < corrs_missed.first.size(); i++)
                            {
                                Eigen::Vector3d x1h(corrs_missed.first.at(i).x(), corrs_missed.first.at(i).y(), 1.0);
                                Eigen::Vector3d x2h(corrs_missed.second.at(i).x(), corrs_missed.second.at(i).y(), 1.0);
                                Eigen::Vector3d x1 = K1i * x1h;
                                Eigen::Vector3d x2 = K2i * x2h;
                                x1 /= x1.z();
                                x2 /= x2.z();
                                
                                double e12 = calculateEpipolarError(x1, x2, R4R2E12);
                                if (nearZero(e12, 1e-10))
                                {
                                    continue;
                                }
                                double e23 = calculateEpipolarError(x1, x2, R4E23R1);
                                double e34 = calculateEpipolarError(x1, x2, E34R3);
                                coeffs.emplace_back(std::make_tuple(e12, e23, e34));
                            }
                            double scale_use1 = 1.0, scale_use2 = 1.0;
                            size_t s_cnt = 1;
                            size_t i_plus = 1, j_plus = 1;
                            if (coeffs.size() > 10)
                            {
                                std::vector<double> s1s, s2s;
                                for (size_t i = 0; i < coeffs.size() - 1; i += i_plus)
                                {
                                    for (size_t j = i; j < coeffs.size(); j += j_plus)
                                    {
                                        const double d = std::get<0>(coeffs.at(i)) / std::get<0>(coeffs.at(j));
                                        const double den = std::get<1>(coeffs.at(i)) - d * std::get<1>(coeffs.at(j));
                                        if (nearZero(den, 1e-10))
                                        {
                                            continue;
                                        }
                                        const double num = std::get<2>(coeffs.at(i)) - d * std::get<2>(coeffs.at(j));
                                        const double s1 = num / den;
                                        if (std::abs(s1) < 1e-2 || std::abs(s1) > 100.)
                                        {
                                            continue;
                                        }
                                        const double s2 = -1. * (s1 * std::get<1>(coeffs.at(i)) + std::get<2>(coeffs.at(i))) / std::get<0>(coeffs.at(i));
                                        if (std::abs(s2) < 1e-2 || std::abs(s2) > 100.)
                                        {
                                            continue;
                                        }
                                        s1s.emplace_back(s1);
                                        s2s.emplace_back(s2);
                                        s_cnt++;
                                        if(s_cnt % 1000 == 0){
                                            if ((coeffs.size() - j) / 4 > j_plus){
                                                j_plus *= 2;
                                            }
                                        }
                                    }
                                    j_plus = 1;
                                    if (s_cnt % 10000 == 0)
                                    {
                                        if ((coeffs.size() - i) / 4 > i_plus)
                                        {
                                            i_plus *= 2;
                                        }
                                    }
                                    if(s_cnt > 10000000){
                                        break;
                                    }
                                }
                                if (s1s.size() > 10){
                                    double s1_mean, s2_mean, s1_sd, s2_sd;
                                    getMeanStandardDeviation(s1s, s1_mean, s1_sd);
                                    getMeanStandardDeviation(s2s, s2_mean, s2_sd);
                                    const double s1_th_lo = s1_mean - 1.5 * s1_sd;
                                    const double s1_th_hi = s1_mean + 1.5 * s1_sd;
                                    const double s2_th_lo = s2_mean - 1.5 * s2_sd;
                                    const double s2_th_hi = s2_mean + 1.5 * s2_sd;
                                    std::vector<double> s1s2, s2s2;
                                    for (size_t i = 0; i < s1s.size(); i++)
                                    {
                                        if (s1s.at(i) > s1_th_lo && s1s.at(i) < s1_th_hi && s2s.at(i) > s2_th_lo && s2s.at(i) < s2_th_hi){
                                            s1s2.emplace_back(s1s.at(i));
                                            s2s2.emplace_back(s2s.at(i));
                                        }
                                    }
                                    if (s1s2.size() > 10){
                                        const double s1 = getMean(s1s2);
                                        const double s2 = getMean(s2s2);
                                        if (std::abs(s1) > 1e-2 && std::abs(s1) < 100. && std::abs(s2) > 1e-2 && std::abs(s2) < 100.)
                                        {
                                            if(!nearZero(s1 - 1.)){
                                                scale_use1 = s1;
                                            }
                                            if (!nearZero(s2 - 1.)){
                                                scale_use2 = s2;
                                            }
                                        }
                                    }
                                }
                            }
                            Eigen::Matrix3d E14 = scale_use2 * R4R2E12 + scale_use1 * R4E23R1 + E34R3;
                            Es.emplace_back(E14);
                        }
                    }
                    if (!Es.empty()){
                        if(Es.size() == 1){
                            cv::Mat E12_cv;
                            cv::eigen2cv(Es[0], E12_cv);
                            cv::Mat R12, t12;
                            getNearestRTfromEDecomposition(E12_cv, Rs.at(mrs.first), ts.at(mrs.first), R12, t12);
                            Eigen::Matrix3d R12_eigen;
                            Eigen::Vector3d t12_eigen;
                            cv::cv2eigen(t12, t12_eigen);
                            cv::cv2eigen(R12, R12_eigen);
                            ba_data.overwritePose(mrs.first, R12_eigen, t12_eigen);
                        }else{
                            std::map<double, size_t> sampsons;
                            Eigen::Matrix3d K1i = ba_data.cams.at(mrs.first.first).CalibrationMatrix().inverse();
                            Eigen::Matrix3d K2i = ba_data.cams.at(mrs.first.second).CalibrationMatrix().inverse();
                            cv::Mat K1i_cv, K2i_cv;
                            cv::eigen2cv(K1i, K1i_cv);
                            cv::eigen2cv(K2i, K2i_cv);
                            for (size_t i = 0; i < Es.size(); i++)
                            {
                                cv::Mat E12_cv;
                                cv::eigen2cv(Es[i], E12_cv);
                                cv::Mat F = K2i_cv.t() * E12_cv * K1i_cv;
                                std::vector<double> s_errs;
                                for (size_t j = 0; j < corrs_missed.first.size(); j++){
                                    const cv::Point2f p1(static_cast<float>(corrs_missed.first.at(j).x()), static_cast<float>(corrs_missed.first.at(j).y()));
                                    const cv::Point2f p2(static_cast<float>(corrs_missed.second.at(j).x()), static_cast<float>(corrs_missed.second.at(j).y()));
                                    s_errs.emplace_back(calculateSampsonError(p1, p2, F));
                                }
                                sampsons.emplace(getMedian(s_errs), i);
                            }
                            const size_t E_use_idx = sampsons.begin()->second;
                            cv::Mat E12_cv;
                            cv::eigen2cv(Es[E_use_idx], E12_cv);
                            cv::Mat R12, t12;
                            getNearestRTfromEDecomposition(E12_cv, Rs.at(mrs.first), ts.at(mrs.first), R12, t12);
                            Eigen::Matrix3d R12_eigen;
                            Eigen::Vector3d t12_eigen;
                            cv::cv2eigen(t12, t12_eigen);
                            cv::cv2eigen(R12, R12_eigen);
                            ba_data.overwritePose(mrs.first, R12_eigen, t12_eigen);
                        }
                    }
                }
            }
        }

        //Check pose difference
        for (auto &q : ba_data.quats)
        {
            if (ba_data.constant_poses.at(q.first))
            {
                continue;
            }
            Eigen::Vector3d &t_ref = ba_data.ts.at(q.first);
            const double t_norm = t_ref.norm();
            if (abs(t_norm) < 1e-3)
            {
                std::cout << "Detected a too small translation vector norm after BA!" << std::endl;
                return false;
            }
            t_ref /= t_norm;

            double r_diff, t_diff;
            poselib::getRTQuality(quats_initial.at(q.first), q.second, ts_eigen.at(q.first), t_ref, &r_diff, &t_diff);
            r_diff = 180.0 * r_diff / M_PI;
            if ((abs(r_diff) > angleThresh) || (t_diff > t_norm_tresh))
            {
                std::cout << "Difference of translation vectors and/or rotation before and after BA above threshold. Discarding refined values." << std::endl;
                return false;
            }
        }

        if (!checkIntrinsicsMaps(ba_data, fixFocal, fixPrincipalPt, fixDistortion, haveDists, focal_length_ratios))
        {
            return false;
        }

        ba_data.QuaternionToRotMat();
        for (auto &R : ba_data.Rs)
        {
            if (!ba_data.constant_poses.at(R.first))
            {
                cv::Mat R_tmp, t_tmp;
                eigen2cv(ba_data.ts.at(R.first), t_tmp);
                eigen2cv(R.second, R_tmp);
                convertPrecisionMat(t_tmp, CERES_RESULT_CONVERT_PRECISION);
                convertPrecisionMat(R_tmp, CERES_RESULT_CONVERT_PRECISION);
                R_tmp.copyTo(Rs.at(R.first));
                t_tmp.copyTo(ts.at(R.first));
            }
        }

        updateIntrinsics(ba_data, fixFocal, fixPrincipalPt, fixDistortion, haveDists, fxEqFy_all, Ks, dists, CERES_RESULT_CONVERT_PRECISION);

        return true;
    }

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
                            const double angleThresh,
                            const double t_norm_tresh,
                            const LossFunctionToUse loss,
                            const MethodToUse method,
                            const double toleranceMultiplier,
                            const double th,
                            const bool fixFocal,
                            const bool fixPrincipalPt,
                            const bool fixDistortion,
                            const bool distortion_damping,
                            const bool optimCalibrationOnly,
                            const bool fxEqFy,
                            const bool keepScaling,
                            const CamMatDamping *K_damping,
                            const std::unordered_map<int, std::vector<bool>> *masks_corr,
                            cv::InputArray mask_Q,
                            cv::InputOutputArray dists,
                            const std::vector<size_t> *constant_cam_idx,
                            const std::unordered_set<int> *constant_pose_idx,
                            int cpu_count,
                            int verbose)
    {
        if (corrs.empty() || Qs.empty() || map3D.empty() || Rs.empty() || ts.empty() || Ks.empty() || depth_scales.empty() || depth_vals.empty())
        {
            cerr << "Some input variables to BA are empty. Skipping!" << endl;
            return false;
        }
        if (!Ks.isMatVector() || (!dists.empty() && !dists.isMatVector()))
        {
            cerr << "Inputs must be of type vector<Mat>. Skipping!" << endl;
            return false;
        }
        if (!mask_Q.empty() && !mask_Q.isMat())
        {
            cerr << "Input mask_Q must be of type Mat. Skipping!" << endl;
            return false;
        }
        if (!Qs.isMat())
        {
            cerr << "Input Qs must be of type Mat. Skipping!" << endl;
            return false;
        }
        CV_Assert(Qs.type() == CV_64FC1);
        Mat Q = Qs.getMat();
        int nr_Qs = Q.size().height;
        if (nr_Qs < 15)
        {
            cerr << "Too less 3D points. Skipping!" << endl;
            return false;
        }

        Mat mask_Q_;
        bool haveQMasks = false;
        if (!mask_Q.empty() && mask_Q.size().area() != nr_Qs)
        {
            cerr << "Input mask mask_Q of type Mat must be of same size as Qs. Skipping!" << endl;
            return false;
        }
        if (!mask_Q.empty())
        {
            mask_Q_ = mask_Q.getMat();
            haveQMasks = true;
        }
        else
        {
            mask_Q_ = Mat::ones(1, nr_Qs, CV_8UC1);
        }

        vector<Mat> K_vec;
        Ks.getMatVector(K_vec);
        const size_t vecSi = corrs.size();
        if (map3D.size() != vecSi)
        {
            cerr << "Size of map3D must be equal to corrs. Skipping!" << endl;
            return false;
        }
        if (depth_vals.size() != static_cast<size_t>(nr_Qs))
        {
            cerr << "Size of depth_vals must be equal to nr of 3D points. Skipping!" << endl;
            return false;
        }
        for(const auto &d : depth_vals){
            if (depth_scales.find(d.first) == depth_scales.end()){
                cerr << "Missing scaling parameter for depths of img " << d.first << ". Skipping!" << endl;
                return false;
            }
        }
        if (Rs.size() != vecSi || ts.size() != vecSi || cimg_2_cam_idx.size() != vecSi)
        {
            cerr << "Camera parameters and poses of real cams must be of same size. Skipping!" << endl;
            return false;
        }
        const size_t camSi = K_vec.size();
        bool haveMasks = false;
        if (masks_corr)
        {
            for (const auto &i : *masks_corr)
            {
                const int mSi = i.second.size();
                const int pSi = corrs.at(i.first).size();
                if (pSi != mSi)
                {
                    cerr << "Input mask " << i.first << " does not have the same size as corresponding 2D features. Skipping BA!" << endl;
                    return false;
                }
            }
            haveMasks = true;
        }

        for (const auto &i : map3D)
        {
            const size_t idx_max = *std::max_element(i.second.begin(), i.second.end());
            if (idx_max >= static_cast<size_t>(nr_Qs))
            {
                cerr << "Index of 3D map " << i.first << " out of bounds. Skipping BA!" << endl;
                return false;
            }
        }

        vector<Mat> dist_mvec;
        bool haveDists = false;
        if (!dists.empty())
        {
            dists.getMatVector(dist_mvec);
            if (dist_mvec.size() != camSi)
            {
                cerr << "Input dists of type vector<Mat> must be of same size as other camera inputs. Skipping!" << endl;
                return false;
            }
            haveDists = true;
        }

        BundleAdjustmentOptions options;
        setOptionsCostfunction(options, loss, th);
        setOptionsSolveMethod(options, method);
        options.solver_options.parameter_tolerance = 1e-8;
        options.solver_options.gradient_tolerance = 1e-8;
        options.solver_options.function_tolerance = 1e-6;
        if (toleranceMultiplier >= 1e-3 && toleranceMultiplier <= 1e4)
        {
            options.solver_options.parameter_tolerance *= toleranceMultiplier;
            options.solver_options.gradient_tolerance *= toleranceMultiplier;
            options.solver_options.function_tolerance *= toleranceMultiplier;
            if (toleranceMultiplier > 9.0)
            {
                options.solver_options.min_trust_region_radius = 1e-9;
                options.solver_options.max_trust_region_radius = 1e10;
            }
        }
        // options.solver_options.max_num_iterations = 200;
        if (!verbose)
        {
            options.solver_options.minimizer_progress_to_stdout = false;
            options.solver_options.logging_type = ceres::SILENT;
            options.print_summary = false;
        }
        else if (verbose == 1)
        {
            options.solver_options.minimizer_progress_to_stdout = false;
            options.solver_options.logging_type = ceres::SILENT;
            options.print_summary = true;
        }
        else
        {
            options.solver_options.minimizer_progress_to_stdout = true;
            options.solver_options.logging_type = ceres::PER_MINIMIZER_ITERATION;
            options.print_summary = true;
        }
        options.CeresCPUcnt = cpu_count;
        options.refine_focal_length = !fixFocal;
        options.refine_principal_point = !fixPrincipalPt;
        options.refine_extra_params = !fixDistortion;

        vector<bool> fxEqFy_all(camSi, fxEqFy);
        const int max_img_size = std::max(img_Size.width, img_Size.height);
        vector<double> focal_length_ratios;

        std::vector<colmap::Camera> cams;
        for (size_t i = 0; i < camSi; ++i)
        {
            Mat &Kx = K_vec[i];
            colmap::Camera cam;
            double focal_length_ratio;
            bool modelSupported = false;
            bool fxEqFy_all_i = fxEqFy;
            if (haveDists)
            {
                modelSupported = getCameraParametersColmap(Kx, max_img_size, img_Size, fxEqFy, cam, focal_length_ratio, fxEqFy_all_i, dist_mvec[i]);
            }
            else
            {
                modelSupported = getCameraParametersColmap(Kx, max_img_size, img_Size, fxEqFy, cam, focal_length_ratio, fxEqFy_all_i);
            }
            if (!modelSupported)
            {
                return false;
            }
            focal_length_ratios.emplace_back(focal_length_ratio);
            fxEqFy_all[i] = fxEqFy_all_i;
            cams.emplace_back(move(cam));
        }
        std::vector<bool> constant_cams = vector<bool>(camSi, false);
        if (constant_cam_idx && !constant_cam_idx->empty())
        {
            for (const auto &i : *constant_cam_idx)
            {
                constant_cams.at(i) = true;
            }
        }
        
        vector<int> q_diff_idx(nr_Qs, 0);
        std::vector<Eigen::Vector3d> points3d;
        if (haveQMasks)
        {
            int diff_idx = 0;
            for (int i = 0; i < nr_Qs; ++i)
            {
                if (mask_Q_.at<unsigned char>(i))
                {
                    q_diff_idx[i] = diff_idx;
                    Eigen::Vector3d point3d;
                    cv2eigen(Q.row(i).t(), point3d);
                    points3d.emplace_back(move(point3d));
                }
                else
                {
                    q_diff_idx[i] = -1;
                    diff_idx++;
                }
            }
        }
        else
        {
            for (int i = 0; i < nr_Qs; ++i)
            {
                Eigen::Vector3d point3d;
                cv2eigen(Q.row(i).t(), point3d);
                points3d.emplace_back(move(point3d));
            }
        }

        std::unordered_map<int, Eigen::Vector2d> depth_scales_imgs;
        for(const auto &sa : depth_scales){
            depth_scales_imgs.emplace(sa.first, Eigen::Vector2d(sa.second.first, sa.second.second));
        }

        vector<bool> img_has_3d_pts(vecSi, true);
        size_t cnt = 0;
        vector<Mat> R_mvec, t_mvec;
        std::vector<size_t> img_2_cam_idx;
        std::vector<size_t> constant_pose_idx2;
        std::unordered_map<size_t, int> ci_idx_save;
        std::vector<std::vector<Eigen::Vector2d>> corrs_eigen_all;
        std::vector<std::vector<size_t>> points3d_idx_all;
        std::vector<bool> constant_poses;
        for (const auto &i : corrs)
        {
            vector<bool> kpMask;
            if (haveMasks && (masks_corr->find(i.first) != masks_corr->end()))
            {
                kpMask = masks_corr->at(i.first);
            }
            else
            {
                kpMask = vector<bool>(i.second.size(), true);
            }
            std::vector<Eigen::Vector2d> corrs_eigen;
            std::vector<size_t> points3d_idx;
            int x_idx = 0;
            for (const auto &q_idx : map3D.at(i.first))
            {
                if (kpMask.at(x_idx) && mask_Q_.at<unsigned char>(static_cast<int>(q_idx)))
                {
                    int &qd_idx = q_diff_idx.at(q_idx);
                    if (qd_idx >= 0)
                    {
                        corrs_eigen.emplace_back(Eigen::Vector2d(static_cast<double>(i.second.at(x_idx).x), static_cast<double>(i.second.at(x_idx).y)));
                        points3d_idx.emplace_back(q_idx - static_cast<size_t>(qd_idx));
                    }
                }
                x_idx++;
            }
            if (!corrs_eigen.empty())
            {
                corrs_eigen_all.emplace_back(move(corrs_eigen));
                points3d_idx_all.emplace_back(move(points3d_idx));
                ci_idx_save.emplace(cnt, i.first);
                if (constant_pose_idx && !constant_pose_idx->empty() && (constant_pose_idx->find(i.first) != constant_pose_idx->end()))
                {
                    constant_poses.emplace_back(true);
                }
                else
                {
                    constant_poses.emplace_back(false);
                }
            }
            else
            {
                cout << "No match available for camera " << i.first << "." << endl;
                img_has_3d_pts[cnt] = false;
            }
            R_mvec.emplace_back(Rs.at(i.first));
            t_mvec.emplace_back(ts.at(i.first));
            img_2_cam_idx.emplace_back(cimg_2_cam_idx.at(i.first));
            cnt++;
        }

        std::vector<Eigen::Vector3d> t_normed_init;
        std::vector<double> t_norms_init;
        std::vector<Eigen::Matrix3d> Rs_eigen;
        std::vector<Eigen::Vector3d> ts_eigen;
        std::vector<size_t> imgs_to_cam_idx2;
        for (size_t i = 0; i < vecSi; ++i)
        {
            if (img_has_3d_pts[i])
            {
                Eigen::Matrix3d R;
                cv::cv2eigen(R_mvec.at(i), R);
                Rs_eigen.emplace_back(move(R));
                Eigen::Vector3d t;
                if (t_mvec.at(i).cols > t_mvec.at(i).rows)
                {
                    t_mvec.at(i) = t_mvec.at(i).t();
                }
                cv::cv2eigen(t_mvec.at(i), t);
                ts_eigen.push_back(t);
                double t_norm = t.norm();
                if (abs(t_norm) < 1e-4)
                {
                    t_norm = 1.0;
                }
                t_norms_init.push_back(t_norm);
                t_normed_init.emplace_back(t / t_norm);
                imgs_to_cam_idx2.push_back(img_2_cam_idx[i]);
            }
        }

        bool dist_damp = distortion_damping && !fixDistortion && haveDists;

        FixedDepthBAData ba_data(cams, imgs_to_cam_idx2, Rs_eigen, ts_eigen, corrs_eigen_all, points3d, depth_vals, depth_scales_imgs, points3d_idx_all, dist_damp);

        ba_data.constant_cams = move(constant_cams);
        ba_data.constant_poses = move(constant_poses);

        std::vector<Eigen::Vector4d> quats_initial = ba_data.quats;

        if (K_damping && K_damping->damping)
        {
            const double cx_init = static_cast<double>(img_Size.width) / 2.;
            const double cy_init = static_cast<double>(img_Size.height) / 2.;
            CamMatDamping damp_tmp = *K_damping;
            if (fixFocal && K_damping->useDampingF())
            {
                damp_tmp.disableDampingF();
            }
            if ((fxEqFy || fixFocal) && K_damping->useDampingFxFy())
            {
                damp_tmp.disableDampingFxFy();
            }
            if (fixPrincipalPt && K_damping->useDampingCxCy())
            {
                damp_tmp.disableDampingCxCy();
            }
            if (damp_tmp.damping)
            {
                ba_data.camMat_damp = damp_tmp.damping;
                for (const auto &ci : ba_data.cams)
                {
                    ba_data.camMatDampPars.emplace_back(colmap::CamMatDampingSingle(damp_tmp.imgDiag_2,
                                                                                    damp_tmp.focalMid,
                                                                                    damp_tmp.focalRange2,
                                                                                    ci.MeanFocalLength(),
                                                                                    cx_init,
                                                                                    cy_init,
                                                                                    damp_tmp.damping,
                                                                                    damp_tmp.useDampingF() ? damp_tmp.fChangeMax : 0.,
                                                                                    damp_tmp.useDampingFxFy() ? damp_tmp.fxfyRatioDiffMax : 0.,
                                                                                    damp_tmp.useDampingCxCy() ? damp_tmp.cxcyDistMidRatioMax : 0.));
                }
            }
        }

        FixedDepthBundleAdjuster ba(options, TAKE_NOT_BA_OWNERSHIP);
        if (!ba.Solve(&ba_data))
        {
            cerr << "Bundle adustment failed (bad input data)." << endl;
            return false;
        }

        //Normalize the translation vectors
        double scaling_med = 1.;
        if (!checkPoseDifferenceRescale(ba_data, t_normed_init, t_norms_init, quats_initial, angleThresh, t_norm_tresh, keepScaling, &scaling_med))
        {
            return false;
        }

        if (!checkIntrinsics(ba_data, fixFocal, fixPrincipalPt, fixDistortion, haveDists, focal_length_ratios))
        {
            return false;
        }

        ba_data.QuaternionToRotMat();
        size_t idx = 0;
        for (size_t i = 0; i < vecSi; ++i)
        {
            if (img_has_3d_pts[i])
            {
                if (!ba_data.constant_poses[idx])
                {
                    cv::Mat R_tmp, t_tmp;
                    const int &ci = ci_idx_save.at(i);
                    eigen2cv(ba_data.ts.at(idx), t_tmp);
                    eigen2cv(ba_data.Rs.at(idx), R_tmp);
                    convertPrecisionMat(t_tmp, CERES_RESULT_CONVERT_PRECISION);
                    convertPrecisionMat(R_tmp, CERES_RESULT_CONVERT_PRECISION);
                    R_tmp.copyTo(Rs.at(ci));
                    t_tmp.copyTo(ts.at(ci));
                }
                idx++;
            }
        }

        updateIntrinsics(ba_data, fixFocal, fixPrincipalPt, fixDistortion, haveDists, camSi, fxEqFy_all, K_vec, dist_mvec, CERES_RESULT_CONVERT_PRECISION);
        update3DpointsNoConstPts(ba_data, optimCalibrationOnly, haveQMasks, nr_Qs, mask_Q_, Q, keepScaling, &scaling_med, CERES_RESULT_CONVERT_PRECISION);

        for (const auto &sa : ba_data.depth_scales_imgs)
        {
            depth_scales.at(sa.first) = make_pair(convertPrecisionRet(sa.second.x(), CERES_RESULT_CONVERT_PRECISION), convertPrecisionRet(sa.second.y(), CERES_RESULT_CONVERT_PRECISION));
        }

        return true;
    }
}