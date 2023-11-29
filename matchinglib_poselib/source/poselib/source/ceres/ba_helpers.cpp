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

#include "ceres/ba_helpers.h"

namespace poselib
{
    bool getCameraParametersColmap(const cv::Mat &Kx, const int &max_img_size, const cv::Size &img_Size, const bool fxEqFy, colmap::Camera &cam, double &focal_length_ratio, bool &fxEqFy_all_i, cv::InputArray distortion)
    {
        const double mean_f = (Kx.at<double>(0, 0) + Kx.at<double>(1, 1)) / 2.;
        focal_length_ratio = mean_f / static_cast<double>(max_img_size);
        if (!distortion.empty())
        {
            cv::Mat dist_c = distortion.getMat();
            if (dist_c.rows > dist_c.cols)
            {
                dist_c = dist_c.t();
            }
            if (dist_c.cols == 1)
            {
                cam.InitializeWithName("SIMPLE_RADIAL", mean_f, img_Size.width, img_Size.height);
                std::vector<double> params = {mean_f,
                                              Kx.at<double>(0, 2),
                                              Kx.at<double>(1, 2),
                                              dist_c.at<double>(0)};
                cam.SetParams(params);
                fxEqFy_all_i = true;
            }
            else if (dist_c.cols == 2)
            {
                cam.InitializeWithName("RADIAL", mean_f, img_Size.width, img_Size.height);
                std::vector<double> params = {mean_f,
                                              Kx.at<double>(0, 2),
                                              Kx.at<double>(1, 2),
                                              dist_c.at<double>(0),
                                              dist_c.at<double>(1)};
                cam.SetParams(params);
                fxEqFy_all_i = true;
            }
            else if (dist_c.cols == 4)
            {
                cam.InitializeWithName("OPENCV", mean_f, img_Size.width, img_Size.height);
                std::vector<double> params = {Kx.at<double>(0, 0),
                                              Kx.at<double>(1, 1),
                                              Kx.at<double>(0, 2),
                                              Kx.at<double>(1, 2),
                                              dist_c.at<double>(0),
                                              dist_c.at<double>(1),
                                              dist_c.at<double>(2),
                                              dist_c.at<double>(3)};
                cam.SetParams(params);
                fxEqFy_all_i = false;
            }
            else if (dist_c.cols == 5)
            {
                cam.InitializeWithName("OPENCV_RADIAL3", mean_f, img_Size.width, img_Size.height);
                std::vector<double> params = {Kx.at<double>(0, 0),
                                              Kx.at<double>(1, 1),
                                              Kx.at<double>(0, 2),
                                              Kx.at<double>(1, 2),
                                              dist_c.at<double>(0),
                                              dist_c.at<double>(1),
                                              dist_c.at<double>(2),
                                              dist_c.at<double>(3),
                                              dist_c.at<double>(4)};
                cam.SetParams(params);
                fxEqFy_all_i = false;
            }
            else if (dist_c.cols == 8)
            {
                cam.InitializeWithName("FULL_OPENCV", mean_f, img_Size.width, img_Size.height);
                std::vector<double> params = {Kx.at<double>(0, 0),
                                              Kx.at<double>(1, 1),
                                              Kx.at<double>(0, 2),
                                              Kx.at<double>(1, 2),
                                              dist_c.at<double>(0),
                                              dist_c.at<double>(1),
                                              dist_c.at<double>(2),
                                              dist_c.at<double>(3),
                                              dist_c.at<double>(4),
                                              dist_c.at<double>(5),
                                              dist_c.at<double>(6),
                                              dist_c.at<double>(7)};
                cam.SetParams(params);
                fxEqFy_all_i = false;
            }
            else if (dist_c.cols == 12)
            {
                cam.InitializeWithName("FULL_OPENCV2", mean_f, img_Size.width, img_Size.height);
                std::vector<double> params = {Kx.at<double>(0, 0),
                                              Kx.at<double>(1, 1),
                                              Kx.at<double>(0, 2),
                                              Kx.at<double>(1, 2),
                                              dist_c.at<double>(0),
                                              dist_c.at<double>(1),
                                              dist_c.at<double>(2),
                                              dist_c.at<double>(3),
                                              dist_c.at<double>(4),
                                              dist_c.at<double>(5),
                                              dist_c.at<double>(6),
                                              dist_c.at<double>(7),
                                              dist_c.at<double>(8),
                                              dist_c.at<double>(9),
                                              dist_c.at<double>(10),
                                              dist_c.at<double>(11)};
                cam.SetParams(params);
                fxEqFy_all_i = false;
            }
            else if (dist_c.cols == 14)
            {
                cam.InitializeWithName("FULL_OPENCV3", mean_f, img_Size.width, img_Size.height);
                std::vector<double> params = {Kx.at<double>(0, 0),
                                              Kx.at<double>(1, 1),
                                              Kx.at<double>(0, 2),
                                              Kx.at<double>(1, 2),
                                              dist_c.at<double>(0),
                                              dist_c.at<double>(1),
                                              dist_c.at<double>(2),
                                              dist_c.at<double>(3),
                                              dist_c.at<double>(4),
                                              dist_c.at<double>(5),
                                              dist_c.at<double>(6),
                                              dist_c.at<double>(7),
                                              dist_c.at<double>(8),
                                              dist_c.at<double>(9),
                                              dist_c.at<double>(10),
                                              dist_c.at<double>(11),
                                              dist_c.at<double>(12),
                                              dist_c.at<double>(13)};
                cam.SetParams(params);
                fxEqFy_all_i = false;
            }
            else
            {
                std::cerr << "Distortion model not supported!" << std::endl;
                return false;
            }
        }
        else if (fxEqFy)
        {
            cam.InitializeWithName("SIMPLE_PINHOLE", mean_f, img_Size.width, img_Size.height);
            std::vector<double> params = {mean_f,
                                          Kx.at<double>(0, 2),
                                          Kx.at<double>(1, 2)};
            cam.SetParams(params);
        }
        else
        {
            cam.InitializeWithName("PINHOLE", mean_f, img_Size.width, img_Size.height);
            std::vector<double> params = {Kx.at<double>(0, 0),
                                          Kx.at<double>(1, 1),
                                          Kx.at<double>(0, 2),
                                          Kx.at<double>(1, 2)};
            cam.SetParams(params);
            fxEqFy_all_i = false;
        }
        return true;
    }

    void setOptionsCostfunction(BundleAdjustmentOptions &options, const LossFunctionToUse &loss, const double &th)
    {
        switch (loss)
        {
        case LossFunctionToUse::SQUARED:
            options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
            break;
        case LossFunctionToUse::SOFT_L1:
            options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
            if (th > DBL_EPSILON)
            {
                options.loss_function_scale = th;
            }
            break;
        case LossFunctionToUse::CAUCHY:
            options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::CAUCHY;
            if (th > DBL_EPSILON)
            {
                options.loss_function_scale = th;
            }
            break;
        case LossFunctionToUse::HUBER:
            options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::HUBER;
            if (th > DBL_EPSILON)
            {
                options.loss_function_scale = th;
            }
            break;
        default:
            break;
        }
    }

    void setOptionsSolveMethod(BundleAdjustmentOptions &options, const MethodToUse &method)
    {
        if (method == MethodToUse::DEFAULT)
        {
            options.useDefaultSolver = true;
            return;
        }
        else
        {
            options.useDefaultSolver = false;
            options.solver_options.minimizer_type = ceres::TRUST_REGION;
        }
        switch (method)
        {
        case MethodToUse::DEFAULT:
            break;
        case MethodToUse::LM_SparseNormalChol:
            options.solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            break;
        case MethodToUse::LM_SparseNormalChol_NonMonotonic:
            options.solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.solver_options.use_nonmonotonic_steps = true;
            options.solver_options.max_consecutive_nonmonotonic_steps = 5;
            break;
        case MethodToUse::LM_DenseSchur:
            options.solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.solver_options.linear_solver_type = ceres::DENSE_SCHUR;
            break;
        case MethodToUse::LM_DenseSchur_NonMonotonic:
            options.solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.solver_options.linear_solver_type = ceres::DENSE_SCHUR;
            options.solver_options.use_nonmonotonic_steps = true;
            options.solver_options.max_consecutive_nonmonotonic_steps = 5;
            break;
        case MethodToUse::LM_IterativeSchur_PreconJacobi:
            options.solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            options.solver_options.min_linear_solver_iterations = 1;
            options.solver_options.preconditioner_type = ceres::JACOBI;
            break;
        case MethodToUse::LM_IterativeSchur_Implicit_PreconSchurJacobi:
            options.solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            options.solver_options.min_linear_solver_iterations = 1;
            options.solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
            options.solver_options.use_explicit_schur_complement = false;
            break;
        case MethodToUse::LM_IterativeSchur_Explicit_PreconSchurJacobi:
            options.solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            options.solver_options.min_linear_solver_iterations = 1;
            options.solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
            options.solver_options.use_explicit_schur_complement = true;
            break;
        case MethodToUse::LM_IterativeSchur_PreconJacobi_NonMonotonic:
            options.solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            options.solver_options.min_linear_solver_iterations = 1;
            options.solver_options.preconditioner_type = ceres::JACOBI;
            options.solver_options.use_nonmonotonic_steps = true;
            options.solver_options.max_consecutive_nonmonotonic_steps = 5;
            break;
        case MethodToUse::LM_IterativeSchur_Implicit_PreconSchurJacobi_NonMonotonic:
            options.solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            options.solver_options.min_linear_solver_iterations = 1;
            options.solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
            options.solver_options.use_explicit_schur_complement = false;
            options.solver_options.use_nonmonotonic_steps = true;
            options.solver_options.max_consecutive_nonmonotonic_steps = 5;
            break;
        case MethodToUse::LM_IterativeSchur_Explicit_PreconSchurJacobi_NonMonotonic:
            options.solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            options.solver_options.min_linear_solver_iterations = 1;
            options.solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
            options.solver_options.use_explicit_schur_complement = true;
            options.solver_options.use_nonmonotonic_steps = true;
            options.solver_options.max_consecutive_nonmonotonic_steps = 5;
            break;
        case MethodToUse::Dogleg_SparseNormalChol:
            options.solver_options.trust_region_strategy_type = ceres::DOGLEG;
            options.solver_options.dogleg_type = ceres::SUBSPACE_DOGLEG;
            options.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            break;
        case MethodToUse::Dogleg_SparseNormalChol_NonMonotonic:
            options.solver_options.trust_region_strategy_type = ceres::DOGLEG;
            options.solver_options.dogleg_type = ceres::SUBSPACE_DOGLEG;
            options.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.solver_options.use_nonmonotonic_steps = true;
            options.solver_options.max_consecutive_nonmonotonic_steps = 5;
            break;
        case MethodToUse::Dogleg_DenseSchur:
            options.solver_options.trust_region_strategy_type = ceres::DOGLEG;
            options.solver_options.dogleg_type = ceres::SUBSPACE_DOGLEG;
            options.solver_options.linear_solver_type = ceres::DENSE_SCHUR;
            break;
        case MethodToUse::Dogleg_DenseSchur_NonMonotonic:
            options.solver_options.trust_region_strategy_type = ceres::DOGLEG;
            options.solver_options.dogleg_type = ceres::SUBSPACE_DOGLEG;
            options.solver_options.linear_solver_type = ceres::DENSE_SCHUR;
            options.solver_options.use_nonmonotonic_steps = true;
            options.solver_options.max_consecutive_nonmonotonic_steps = 5;
            break;
        default:
            options.useDefaultSolver = true;
            break;
        }
    }

    cv::Mat normalize3Dpts_GetShiftScale(cv::InputOutputArray Qs)
    {
        cv::Mat Qs_ = Qs.getMat();
        double shift_x = 0, shift_y = 0, shift_z = 0;
        const int nrQ = Qs_.rows;
        const double nrQ_dbl = static_cast<double>(nrQ);
        for (int row = 0; row < nrQ; row++)
        {
            const cv::Mat &qt = Qs_.row(row);
            shift_x += qt.at<double>(0);
            shift_y += qt.at<double>(1);
            shift_z += qt.at<double>(2);
        }
        shift_x /= nrQ_dbl;
        shift_y /= nrQ_dbl;
        shift_z /= nrQ_dbl;
        shift_x *= -1.0;
        shift_y *= -1.0;
        shift_z *= -1.0;

        double scale_x = 0, scale_y = 0, scale_z = 0;
        for (int row = 0; row < nrQ; row++)
        {
            cv::Mat qt = Qs_.row(row);
            qt.at<double>(0) += shift_x;
            qt.at<double>(1) += shift_y;
            qt.at<double>(2) += shift_z;
            scale_x += std::abs(qt.at<double>(0));
            scale_y += std::abs(qt.at<double>(1));
            scale_z += std::abs(qt.at<double>(2));
        }
        scale_x /= nrQ_dbl;
        scale_y /= nrQ_dbl;
        scale_z /= nrQ_dbl;
        const double scale = 1.0 / std::sqrt(scale_x * scale_x + scale_y * scale_y + scale_z * scale_z);
        Qs_ *= scale;
        shift_x *= scale;
        shift_y *= scale;
        shift_z *= scale;

        cv::Mat M = cv::Mat::eye(4, 4, CV_64FC1);
        M.at<double>(0, 0) = scale;
        M.at<double>(1, 1) = scale;
        M.at<double>(2, 2) = scale;
        M.at<double>(0, 3) = shift_x;
        M.at<double>(1, 3) = shift_y;
        M.at<double>(2, 3) = shift_z;

        // std::cout << "Scale: " << scale << ", Shift: [" << shift_x << ", " << shift_y << ", " << shift_z << "]" << std::endl; 

        return M;
    }

    void shiftScaleTranslationVecs(std::vector<cv::Mat> &ts, const std::vector<cv::Mat> &Rs, cv::InputArray M)
    {
        double scale = 1.0;
        cv::Mat shift;
        getScaleAndShiftFromMatrix(M, shift, scale);
        for (size_t i = 0; i < ts.size(); i++)
        {
            shiftScaleTranslationVec(ts[i], Rs.at(i), shift, scale);
        }
    }

    void shiftScaleTranslationVecs(std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &ts, const std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &Rs, cv::InputArray M)
    {
        double scale = 1.0;
        cv::Mat shift;
        getScaleAndShiftFromMatrix(M, shift, scale);
        for (auto &t : ts)
        {
            shiftScaleTranslationVec(t.second, Rs.at(t.first), shift, scale);
        }
    }

    void undoShiftScaleTranslationVecs(std::vector<cv::Mat> &ts, const std::vector<cv::Mat> &Rs, cv::InputArray M)
    {
        double scale = 1.0;
        cv::Mat shift;
        getScaleAndShiftFromMatrix(M, shift, scale);
        for (size_t i = 0; i < ts.size(); i++)
        {
            undoShiftScaleTranslationVec(ts[i], Rs.at(i), shift, scale);
        }
    }

    void undoShiftScaleTranslationVecs(std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &ts, const std::unordered_map<std::pair<int, int>, cv::Mat, pair_hash, pair_EqualTo> &Rs, cv::InputArray M)
    {
        double scale = 1.0;
        cv::Mat shift;
        getScaleAndShiftFromMatrix(M, shift, scale);
        for (auto &t : ts)
        {
            undoShiftScaleTranslationVec(t.second, Rs.at(t.first), shift, scale);
        }
    }

    void undoNormalizeQs(cv::InputOutputArray Qs, cv::InputArray M)
    {
        cv::Mat Qs_ = Qs.getMat();
        double scale = 1.0;
        cv::Mat shift;
        getScaleAndShiftFromMatrix(M, shift, scale);
        shift = shift.t();

        // Remove scale and shift
        for (int row = 0; row < Qs_.rows; row++)
        {
            cv::Mat qt = Qs_.row(row);
            undoShiftScale3DPoint_T(qt, shift, scale);
        }
    }
}