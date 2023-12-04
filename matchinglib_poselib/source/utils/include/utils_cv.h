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

#include <vector>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/highgui.hpp>

#include <trees.h>
#include <utils_common.h>

#include "utilslib/utilslib_api.h"

namespace utilslib
{
    //Computes the Sampson distance (first-order geometric error) for the provided point correspondence
    double UTILSLIB_API calculateSampsonError(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &F);
    double UTILSLIB_API getUsedCorrespAreaRatio(const std::vector<cv::Point2f> &pts, const cv::Size &imgSize);
    
    cv::Mat_<double> UTILSLIB_API calculatePlane(const cv::Vec3d &Q1, const cv::Vec3d &Q2, const cv::Vec3d &Q3);
    double UTILSLIB_API distanceToPlane(const cv::Vec3d &Q1, const cv::Mat &plane, const bool &absDist = true);
    cv::Mat_<double> UTILSLIB_API calculatePlaneKnownNormal(const cv::Vec3d &Q, const cv::Vec3d &planeNormal);
    double UTILSLIB_API Circle_Sigma(const std::vector<cv::Point2d> &data, const cv::Point2d &mid, const double &radius);
    double UTILSLIB_API fitCircle(const std::vector<cv::Point2d> &data, cv::Point2d &mid, double &radius);
    int UTILSLIB_API geometric_circle_fit(const std::vector<cv::Point2d> &data, const cv::Point2d &mid_init, const double &radius_init, cv::Point2d &mid, double &radius, const double &lambda_init = 0.001);
    bool UTILSLIB_API estimateCircle(const std::vector<cv::Point2d> &data2D, const double &th, std::mt19937 &mt, cv::Point2d &center, double &radius, std::vector<size_t> *inliers = nullptr);

    cv::Point2d UTILSLIB_API projectToPlane(const cv::Mat &Q3, const cv::Mat &basisX, const cv::Mat &basisY, const cv::Mat &origin);
    void UTILSLIB_API calculatePlaneBasisVectors(const cv::Mat &planeNormal, cv::Mat &axis1, cv::Mat &axis2);
    cv::Mat_<double> UTILSLIB_API CalculatePt3DMean(const cv::Mat_<double> &points);
    cv::Mat_<double> UTILSLIB_API estimateRigid3DTansformation(const cv::Mat_<double> &points1, const cv::Mat_<double> &points2, double *scaling = nullptr);
    double UTILSLIB_API getRigidTransformPt3DError(const cv::Mat_<double> &P, const cv::Mat_<double> &X1, const cv::Mat_<double> &X2);

    double UTILSLIB_API getDescriptorDist(const cv::Mat &descr1, const cv::Mat &descr2);

    int UTILSLIB_API getVectorMainDirIdx(const cv::Mat vec);
    double UTILSLIB_API getAngleBetwVecs(const cv::Mat vec1, const cv::Mat vec2);
    bool UTILSLIB_API angleBetwVecsBelowTh(const cv::Mat vec1, const cv::Mat vec2, const double &th_deg, const bool ignoreMirror = true);

    // Image Shadow / Highlight Correction
    cv::Mat UTILSLIB_API shadowHighlightCorrection(cv::InputArray img, 
                                                   const float &shadow_amount_percent, 
                                                   const float &shadow_tone_percent, 
                                                   const int &shadow_radius, 
                                                   const float &highlight_amount_percent, 
                                                   const float &highlight_tone_percent, 
                                                   const int &highlight_radius, 
                                                   const bool histEqual = true);
}