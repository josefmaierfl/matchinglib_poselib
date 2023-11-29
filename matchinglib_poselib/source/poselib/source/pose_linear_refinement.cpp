//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2019 AIT Austrian Institute of Technology GmbH
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
/**********************************************************************************************************
FILE: pose_linear_refinement.cpp

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: October 2017

LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functions for linear refinement of Essential matrices
**********************************************************************************************************/

#include "poselib/pose_linear_refinement.h"
#include "poselib/pose_helper.h"
#include <memory>
#include <numeric>      // std::accumulate
#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeWeightingAdapter.hpp>
#include "usac/utils/PoseFunctions.h"
#include "usac/utils/weightingEssential.h"
#include "usac/config/ConfigParamsEssentialMat.h"
#include "opencv2/core/eigen.hpp"


using namespace std;
using namespace cv;

namespace poselib
{

    /* --------------------------- Defines --------------------------- */


    /* --------------------- Function prototypes --------------------- */
    void findRefinementWeights(const opengv::relative_pose::CentralRelativeAdapter adapter, const opengv::essential_t & model, const std::vector<int>& inliers,
        size_t numInliers, double* weights, int refineMethod, double th, double pseudoHuberThreshold_multiplier = 0.1);
    bool refineModel(opengv::relative_pose::CentralRelativeAdapter &adapter,
                     std::vector<int> &indices,
                     const size_t numPoints,
                     opengv::essential_t &model,
                     bool weighted,
                     double *weights,
                     int refineMethod, // a combination of poselib::RefinePostAlg
                     std::mt19937 &mt,
                     opengv::rotation_t *R_inout = NULL,
                     opengv::translation_t *t_out = NULL);
    size_t evaluateModelE(const opengv::relative_pose::CentralRelativeAdapter adapter,
        const opengv::essential_t modelE,
        std::vector<double> & errors,
        std::vector<int> & inliers,
        double th);


    /* --------------------- Functions --------------------- */

    bool refineEssentialLinear(cv::InputArray p1,
                               cv::InputArray p2,
                               cv::InputOutputArray E,
                               cv::InputOutputArray mask,
                               int refineMethod, // a combination of poselib::RefinePostAlg
                               size_t &nr_inliers,
                               cv::InputOutputArray R,
                               cv::OutputArray t,
                               double th,
                               size_t num_iterative_steps,
                               double threshold_multiplier,
                               double pseudoHuberThreshold_multiplier,
                               double maxRelativeInlierCntLoss)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        return refineEssentialLinear( p1, p2, E, g, mask, refineMethod, nr_inliers, R, t, th, num_iterative_steps, threshold_multiplier,  pseudoHuberThreshold_multiplier, maxRelativeInlierCntLoss);
    }

    bool refineEssentialLinear(cv::InputArray p1,
                               cv::InputArray p2,
                               cv::InputOutputArray E,
                               std::mt19937 &mt,
                               cv::InputOutputArray mask,
                               int refineMethod, // a combination of poselib::RefinePostAlg
                               size_t &nr_inliers,
                               cv::InputOutputArray R,
                               cv::OutputArray t,
                               double th,
                               size_t num_iterative_steps,
                               double threshold_multiplier,
                               double pseudoHuberThreshold_multiplier,
                               double maxRelativeInlierCntLoss)
    {
        CV_Assert((p1.rows() == p2.rows()) &&
            (p1.cols() == 2) &&
            (p1.cols() == p2.cols()) &&
            (p1.rows() == mask.cols()) &&
            (p1.type() == CV_64FC1) &&
            (p1.type() == p2.type()) &&
            (E.type() == CV_64FC1) &&
            (mask.type() == CV_8U));

        cv::Mat mask_, p1_, p2_, E_;
        opengv::rotation_t R_inout = opengv::rotation_t::Zero();
        opengv::rotation_t R_refined = opengv::rotation_t::Zero();
        opengv::translation_t t_out = opengv::translation_t::Zero();
        opengv::translation_t t_refined = opengv::translation_t::Zero();
        mask_ = mask.getMat();
        p1_ = p1.getMat();
        p2_ = p2.getMat();
        E_ = E.getMat();
        opengv::essential_t model;
        cv::cv2eigen(E_, model);

        if (!R.empty())
        {
            cv::cv2eigen(R.getMat(), R_inout);
        }

        int num_inliers = cv::countNonZero(mask_);
        if(num_inliers < 6){
            return false;
        }
        int num_data_points = p1_.rows;
        double *weights = new double[num_data_points];
        double th2 = th * th;
        double threshold_step_size = (threshold_multiplier * th2 - th2)
            / num_iterative_steps;
        std::vector<int> inliers(num_data_points);

        //convert data into format used in USAC
        //std::vector<double> pointData(6 * num_data_points);
        opengv::bearingVectors_t bearingVectors1;
        opengv::bearingVectors_t bearingVectors2;
        for (int i = 0, cnt = 0; i < num_data_points; ++i)
        {
            /*pointData[6 * i] = p1_.at<double>(i, 0);
            pointData[6 * i + 1] = p1_.at<double>(i, 1);
            pointData[6 * i + 2] = 1.0;

            pointData[6 * i + 3] = p2_.at<double>(i, 0);
            pointData[6 * i + 4] = p2_.at<double>(i, 1);
            pointData[6 * i + 5] = 1.0;*/

            opengv::point_t bodyPoint1;
            opengv::point_t bodyPoint2;
            bodyPoint1 << p1_.at<double>(i, 0), p1_.at<double>(i, 1), 1.0;
            bodyPoint2 << p2_.at<double>(i, 0), p2_.at<double>(i, 1), 1.0;
            bodyPoint1 = bodyPoint1 / bodyPoint1.norm();
            bodyPoint2 = bodyPoint2 / bodyPoint2.norm();
            bearingVectors1.push_back(bodyPoint1);
            bearingVectors2.push_back(bodyPoint2);

            if (mask_.at<bool>(i))
            {
                inliers[cnt++] = i;
            }
        }

        std::unique_ptr<opengv::relative_pose::CentralRelativeAdapter> adapter; //Pointer to adapter for OpenGV
        //create a central relative adapter
        //opengv inverts the input inside the function (as Nister and Stewenius deliver a inverted pose), so invert the input here as well!!!!!!!!!!!
        adapter.reset(new opengv::relative_pose::CentralRelativeAdapter(
            bearingVectors2,
            bearingVectors1));

        // iterative (reweighted) refinement - reduce threshold in steps, find new inliers and refit essential matrix
        // using weighting
        bool poseRefined = false;
        opengv::essential_t model_new = model;
        for (size_t j = 0; j < num_iterative_steps; ++j)
        {
            std::vector<double> errors;
            std::vector<int> inlierVec_tmp;
            size_t temp_inliers = 0;
            if((refineMethod & 0xF0) != poselib::RefinePostAlg::PR_NO_WEIGHTS)
                findRefinementWeights(*adapter, model_new, inliers, num_inliers, weights, refineMethod, th, pseudoHuberThreshold_multiplier);
            if (poseRefined)
            {
                if (!refineModel(*adapter, inliers, num_inliers, model_new, (refineMethod & 0xF0) != poselib::RefinePostAlg::PR_NO_WEIGHTS, weights, refineMethod, mt, &R_refined, &t_refined))
                {
                    //model_new = model;
                    //poseRefined = false;
                    break;
                }
            }
            else
            {
                if (((refineMethod & 0xF) == poselib::RefinePostAlg::PR_KNEIP) && !poselib::isMatRoationMat(R_inout))
                {
                    /*std::vector<double> error_sums;
                    opengv::essentials_t kneip_essentials;
                    opengv::rotations_t kneip_rotations;
                    opengv::translations_t kneip_translations;*/
                    for (size_t i = 0; i < MAX_SOLS_KNEIP; i++)
                    {
                        R_refined = R_inout;
                        if (!refineModel(*adapter, inliers, num_inliers, model_new, (refineMethod & 0xF0) != poselib::RefinePostAlg::PR_NO_WEIGHTS, weights, refineMethod, mt, &R_refined, &t_refined))
                        {
                            //model_new = model;
                            //R_refined = R_inout;
                            poseRefined = false;
                            continue;
                        }

                        temp_inliers = evaluateModelE(*adapter, model_new, errors, inlierVec_tmp, th2);
                        if ((double)temp_inliers < (1.0 - maxRelativeInlierCntLoss) * (double)num_inliers)
                        {
                            poseRefined = false;
                            continue;
                        }

                        poseRefined = true;
                        break;

                        /*error_sums.push_back(std::accumulate(errors.begin(), errors.end(), 0.0));
                        kneip_essentials.push_back(model_new);
                        kneip_rotations.push_back(R_refined);
                        kneip_translations.push_back(t_refined);*/
                    }
                    if (!poseRefined)
                        break;

                    /*if (!error_sums.empty())
                    {
                        poseRefined = true;
                        int min_elem = std::distance(error_sums.begin(), std::min_element(error_sums.begin(), error_sums.end()));
                        model_new = kneip_essentials[min_elem];
                        R_refined = kneip_rotations[min_elem];
                        t_refined = kneip_translations[min_elem];
                    }
                    else
                        break;*/
                }
                else
                {
                    R_refined = R_inout;
                    if (!refineModel(*adapter, inliers, num_inliers, model_new, (refineMethod & 0xF0) != poselib::RefinePostAlg::PR_NO_WEIGHTS, weights, refineMethod, mt, &R_refined, &t_refined))
                    {
                        //model_new = model;
                        //R_refined = R_inout;
                        //poseRefined = false;
                        break;
                    }
                    poseRefined = true;
                }
            }

            temp_inliers = evaluateModelE(*adapter, model_new, errors, inlierVec_tmp, (threshold_multiplier * th2) - (j + 1)*threshold_step_size);

            if ((double)temp_inliers >= (1.0 - maxRelativeInlierCntLoss) * (double)num_inliers)
            {
                model = model_new;
                R_inout = R_refined;
                t_out = t_refined;
                num_inliers = temp_inliers;
                inliers = inlierVec_tmp;
            }
            else if (j == 0)
            {
                return false;
            }
            else
                break;
        }

        if (R.needed() && t.needed() && poselib::isMatRoationMat(R_inout) && !t_out.isZero(1e-3))
        {
            cv::Mat R_(3, 3, CV_64FC1);
            if (R.empty())
                R.create(3, 3, CV_64F);
            cv::eigen2cv(R_inout, R_);
            R_.copyTo(R.getMat());

            cv::Mat t_(3, 1, CV_64FC1);
            if (t.empty())
                t.create(3, 1, CV_64F);
            cv::eigen2cv(t_out, t_);
            t_.copyTo(t.getMat());
        }
        else if (R.needed())
        {
            R.clear();
        }
        else if (t.needed())
        {
            t.clear();
        }

        mask_.setTo(0);
        for (int i = 0; i < num_inliers; ++i)
        {
            mask_.at<bool>(inliers[i]) = true;
        }
        mask_.copyTo(mask.getMat());//Is this necessary?

        cv::eigen2cv(model, E_);
        E_.copyTo(E.getMat());//Is this necessary?

        nr_inliers = num_inliers;

        delete[] weights;
        return true;
    }

    // ============================================================================================
    // findWeights: given model and points, compute weights to be used in refinement
    // ============================================================================================
    void findRefinementWeights(const opengv::relative_pose::CentralRelativeAdapter adapter, const opengv::essential_t & model, const std::vector<int>& inliers,
        size_t numInliers, double* weights, int refineMethod, double th, double pseudoHuberThreshold_multiplier)
    {
        double rx, ry, ryc, rxc;
        double* pt;
        size_t pt_index;
        opengv::bearingVector_t f, fprime;

        if (((refineMethod & 0xF0) == RefinePostAlg::PR_TORR_WEIGHTS))
        {
            for (size_t i = 0; i < numInliers; ++i)
            {
                f = adapter.getBearingVector2(inliers[i]);
                fprime = adapter.getBearingVector1(inliers[i]);
                weights[i] = computeTorrWeight(f, fprime, model);
            }
        }
        else if (((refineMethod & 0xF0) == RefinePostAlg::PR_PSEUDOHUBER_WEIGHTS))
        {
            double pseudohuberth = th * pseudoHuberThreshold_multiplier;
            for (size_t i = 0; i < numInliers; ++i)
            {
                f = adapter.getBearingVector2(inliers[i]);
                fprime = adapter.getBearingVector1(inliers[i]);
                weights[i] = computePseudoHuberWeight(f, fprime, model, pseudohuberth);
            }
        }
    }

    // ============================================================================================
    // generateRefinedModel: compute model using non-minimal set of samples
    // default operation is to use a weight of 1 for every data point
    // ============================================================================================
    bool refineModel(opengv::relative_pose::CentralRelativeAdapter &adapter,
                     std::vector<int> &indices,
                     const size_t numPoints,
                     opengv::essential_t &model,
                     bool weighted,
                     double *weights,
                     int refineMethod, // a combination of poselib::RefinePostAlg
                     std::mt19937 &mt,
                     opengv::rotation_t *R_inout,
                     opengv::translation_t *t_out)
    {
        opengv::bearingVectors_t bearingVectors1;
        opengv::bearingVectors_t bearingVectors2;
        std::vector<double> weights_vec;
        std::unique_ptr<opengv::relative_pose::CentralRelativeWeightingAdapter> adapter_weights; //Pointer to adapter for OpenGV if Stewenius with weights is used

        if (weighted)
        {
            bearingVectors1.reserve(numPoints);
            bearingVectors1.reserve(numPoints);
            weights_vec.reserve(numPoints);
            for (size_t i = 0; i < numPoints; i++)
            {
                bearingVectors1.push_back(adapter.getBearingVector1(indices[i]));
                bearingVectors2.push_back(adapter.getBearingVector2(indices[i]));
                weights_vec.push_back((weights[i]));
            }
            adapter_weights.reset(new opengv::relative_pose::CentralRelativeWeightingAdapter(
                bearingVectors1,
                bearingVectors2,
                weights_vec));
        }

        if ((refineMethod & 0xF) == poselib::RefinePostAlg::PR_8PT)
        {
            if (weighted)
            {
                model = eightpt_weight(*adapter_weights);
            }
            else
            {
                model = opengv::relative_pose::eightpt(adapter, indices);
            }
        }
        else if ((refineMethod & 0xF) == poselib::RefinePostAlg::PR_STEWENIUS)
        {
            opengv::complexEssentials_t comlexEs;
            opengv::essentials_t Es_eigen;
            opengv::essential_t E_singele;

            if ((((refineMethod & 0xF0) == poselib::RefinePostAlg::PR_TORR_WEIGHTS) || ((refineMethod & 0xF0) == poselib::RefinePostAlg::PR_PSEUDOHUBER_WEIGHTS)) && weighted)
            {
                comlexEs = fivept_stewenius_weight(*adapter_weights);
            }
            else
            {
                comlexEs = opengv::relative_pose::fivept_stewenius(adapter, indices);
            }

            for (size_t i = 0; i < comlexEs.size(); i++)
            {
                bool is_Imag = false;
                for (int r = 0; r < 3; r++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        if (!poselib::nearZero(100 * comlexEs.at(i)(r, c).imag()))
                        {
                            is_Imag = true;
                            break;
                        }
                    }
                    if (is_Imag)
                        break;
                }
                if (!is_Imag)
                {
                    opengv::essential_t E_eigen;
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++)
                            E_eigen(r, c) = comlexEs.at(i)(r, c).real();
                    Es_eigen.push_back(E_eigen);
                }
            }

            size_t nsols = Es_eigen.size();
            if (nsols > 1)
            {
                std::vector<double> errSums(nsols, 0);
                std::vector<std::vector<double>> possible_models(nsols);
                for (size_t j = 0; j < nsols; j++)
                {
                    possible_models[j].resize(9);
                    for (size_t r = 0; r < 3; r++)
                        for (size_t c = 0; c < 3; c++)
                            possible_models[j][r * 3 + c] = Es_eigen[j](r, c);
                }
                for (size_t i = 0; i < numPoints; ++i)
                {
                    opengv::bearingVector_t p_tmp1 = adapter.getBearingVector2(indices[i]);
                    opengv::bearingVector_t p_tmp2 = adapter.getBearingVector1(indices[i]);
                    p_tmp1 /= p_tmp1[2];
                    p_tmp2 /= p_tmp2[2];
                    VectorXd vec_joined(6,1);
                    vec_joined << p_tmp1, p_tmp2;

                    for (size_t j = 0; j < nsols; j++)
                    {
                        errSums[j] += PoseTools::getSampsonError(possible_models[j], vec_joined.data(), 0);
                    }
                    if ((i > 3) && (i % 4 == 0))
                    {
                        std::vector<double> errSums_tmp = errSums;
                        std::partial_sort(errSums_tmp.begin(), errSums_tmp.begin() + 2, errSums_tmp.end());
                        if (errSums_tmp[0] < 0.66 * errSums_tmp[1])
                        {
                            break;
                        }
                    }
                }
                int min_elem = std::distance(errSums.begin(), std::min_element(errSums.begin(), errSums.end()));
                E_singele = Es_eigen[min_elem];
            }
            else if (nsols == 1)
                E_singele = Es_eigen[0];
            else
                return false;

            model = E_singele;
        }
        else if ((refineMethod & 0xF) == poselib::RefinePostAlg::PR_NISTER)
        {
            opengv::essentials_t Es_eigen;
            opengv::essential_t E_singele;

            if ((((refineMethod & 0xF0) == poselib::RefinePostAlg::PR_TORR_WEIGHTS) || ((refineMethod & 0xF0) == poselib::RefinePostAlg::PR_PSEUDOHUBER_WEIGHTS)) && weighted)
            {
                Es_eigen = fivept_nister_weight(*adapter_weights);
            }
            else
            {
                Es_eigen = opengv::relative_pose::fivept_nister(adapter, indices);
            }

            size_t nsols = Es_eigen.size();
            if (nsols > 1)
            {
                std::vector<double> errSums(nsols, 0);
                std::vector<std::vector<double>> possible_models(nsols);
                for (size_t j = 0; j < nsols; j++)
                {
                    possible_models[j].resize(9);
                    for (size_t r = 0; r < 3; r++)
                        for (size_t c = 0; c < 3; c++)
                            possible_models[j][r * 3 + c] = Es_eigen[j](r, c);
                }
                for (size_t i = 0; i < numPoints; ++i)
                {
                    opengv::bearingVector_t p_tmp1 = adapter.getBearingVector2(indices[i]);
                    opengv::bearingVector_t p_tmp2 = adapter.getBearingVector1(indices[i]);
                    p_tmp1 /= p_tmp1[2];
                    p_tmp2 /= p_tmp2[2];
                    VectorXd vec_joined(6, 1);
                    vec_joined << p_tmp1, p_tmp2;

                    for (size_t j = 0; j < nsols; j++)
                    {
                        errSums[j] += PoseTools::getSampsonError(possible_models[j], vec_joined.data(), 0);
                    }
                    if ((i > 3) && (i % 4 == 0))
                    {
                        std::vector<double> errSums_tmp = errSums;
                        std::partial_sort(errSums_tmp.begin(), errSums_tmp.begin() + 2, errSums_tmp.end());
                        if (errSums_tmp[0] < 0.66 * errSums_tmp[1])
                        {
                            break;
                        }
                    }
                }
                int min_elem = std::distance(errSums.begin(), std::min_element(errSums.begin(), errSums.end()));
                E_singele = Es_eigen[min_elem];
            }
            else if (nsols == 1)
                E_singele = Es_eigen[0];
            else
                return false;

            model = E_singele;
        }
        else if (((refineMethod & 0xF) == poselib::RefinePostAlg::PR_KNEIP) && ((R_inout != NULL) && (t_out != NULL)))
        {
            opengv::eigensolverOutput_t eig_out;
            cv::Mat R_tmp, t_tmp, E_tmp;
            //Variation of R as init for eigen-solver
            opengv::rotation_t R_init, R_eigen_new;

            //Check if R_inout is a rotation matrix
            bool is_rot_mat = poselib::isMatRoationMat(*R_inout);

            if (is_rot_mat)
            {
                adapter.setR12(*R_inout);
                eig_out.rotation = *R_inout;
            }
            else
            {
                R_init = Eigen::Matrix3d::Identity();
                PoseTools::getPerturbedRotation(R_init, mt, RAND_ROTATION_AMPLITUDE); //Check if the amplitude is too large or too small!
                adapter.setR12(R_init);
                eig_out.rotation = R_init;
            }

            if ((((refineMethod & 0xF0) == poselib::RefinePostAlg::PR_TORR_WEIGHTS) || ((refineMethod & 0xF0) == poselib::RefinePostAlg::PR_PSEUDOHUBER_WEIGHTS)) && weighted)
            {
                adapter_weights->setR12(eig_out.rotation);

                R_eigen_new = opengv::relative_pose::eigensolver(*adapter_weights, eig_out);
                *t_out = eig_out.translation;
                *t_out /= t_out->norm();
            }
            else
            {
                R_eigen_new = opengv::relative_pose::eigensolver(adapter, indices, eig_out);
                *t_out = eig_out.translation;
                *t_out /= t_out->norm();
                /*R_eigen_new.transposeInPlace();
                t_eigen_new = -1.0 * R_eigen_new * t_eigen_new;*/
            }
            for (size_t r = 0; r < 3; r++)
                for (size_t c = 0; c < 3; c++)
                    if (isnan(((long double)R_eigen_new(r, c))))
                    {
                        return false;
                    }
            if (!poselib::isMatRoationMat(R_eigen_new))
                return false;
            if (t_out->isZero(1e-3))
                return false;

            *R_inout = R_eigen_new;
            cv::eigen2cv(R_eigen_new, R_tmp);
            cv::eigen2cv(*t_out, t_tmp);
            E_tmp = poselib::getEfromRT(R_tmp, t_tmp);
            cv::cv2eigen(E_tmp, model);
        }
        else if (((refineMethod & 0xF) == poselib::RefinePostAlg::PR_KNEIP) && ((R_inout == NULL) || (t_out == NULL)))
        {
            cout << "Rotation and/or translation not set in function call! Skipping refinement!" << endl;
            return false;
        }
        else
        {
            std::cout << "Refinement algorithm not supported! Skipping!" << std::endl;
            return false;
        }

        return true;
    }

    // ============================================================================================
    // evaluateModel: test model against all data points
    // ============================================================================================
    size_t evaluateModelE(const opengv::relative_pose::CentralRelativeAdapter adapter,
        const opengv::essential_t modelE,
        std::vector<double> & errors,
        std::vector<int> & inliers,
        double th)
    {
        double rx, ry, rwc, ryc, rxc, r, temp_err;
        size_t n = adapter.getNumberCorrespondences();
        if (errors.size() != n)
        {
            errors.clear();
            errors = std::vector<double>(n);
        }
        inliers.clear();
        inliers.reserve(n);

        size_t numInliers = 0;
        size_t pt_index;

        for (int i = 0; i < n; ++i)
        {
            //compute sampson L2 error
            Eigen::Vector3d x1 = adapter.getBearingVector2(i);
            Eigen::Vector3d x2 = adapter.getBearingVector1(i);
            errors[i] = poselib::getSampsonL2Error(modelE, x1, x2);

            if (errors[i] < th)
            {
                ++numInliers;
                inliers.push_back(i);
            }
        }
        return numInliers;
    }

}
