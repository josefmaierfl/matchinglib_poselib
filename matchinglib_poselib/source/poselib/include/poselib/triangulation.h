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

#include <string>
#include <opencv2/highgui.hpp>
#include "poselib/poselib_api.h"

namespace poselib
{
    cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point2d u,  //image point (u,v)
                                                    cv::Matx34d P,  //projection 1 matrix
                                                    cv::Point2d u1, //image point in 2nd camera
                                                    cv::Matx34d P1, //projection 2 matrix
                                                    cv::Mat K1,
                                                    cv::Mat K2);

    cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point2d u,  //homogenous image point (u,v,1)
                                                    cv::Matx34d P,  //camera 1 matrix
                                                    cv::Point2d u1, //homogenous image point in 2nd camera
                                                    cv::Matx34d P1  //camera 2 matrix
    );
    cv::Mat_<double> LinearLSTriangulation(cv::Point2d u,  //homogenous image point (u,v,1)
                                           cv::Matx34d P,  //camera 1 matrix
                                           cv::Point2d u1, //homogenous image point in 2nd camera
                                           cv::Matx34d P1  //camera 2 matrix
    );

    //Recovers the rotation and translation from an essential matrix and triangulates the given correspondences to form 3D coordinates.
    int POSELIB_API getPoseTriangPts(cv::InputArray E,
                        cv::InputArray p1,
                        cv::InputArray p2,
                        cv::OutputArray R,
                        cv::OutputArray t,
                        cv::OutputArray Q,
                        cv::InputOutputArray mask = cv::noArray(),
                        const double dist = 50.0,
                        bool translatE = false);
    //Triangulates 3D-points from correspondences with provided R and t
    int POSELIB_API triangPts3D(cv::InputArray R, cv::InputArray t, cv::InputArray _points1, cv::InputArray _points2, cv::OutputArray Q3D, cv::InputOutputArray mask = cv::noArray(), const double dist = 50.0);

    //Transform into camera coordinates and 1 measurement per row
    void POSELIB_API imgToCamCordinatesAndMeasPerRow(cv::InputArray p1, cv::InputArray p2, cv::InputArray K1, cv::InputArray K2, cv::OutputArray p1c, cv::OutputArray p2c);
    //For correspondence in the image coordinate sytem
    cv::Mat_<double> POSELIB_API triangulatePoints_img(const cv::Mat &R, const cv::Mat &t, const cv::Mat &K1, const cv::Mat &K2, const cv::Mat &p1, const cv::Mat &p2, const double &th = -1.0, cv::InputOutputArray mask = cv::noArray());
    //For correspondence in camera coordinates
    cv::Mat_<double> POSELIB_API triangulatePoints_cam(const cv::Mat &R, const cv::Mat &t, const cv::Mat &p1, const cv::Mat &p2, const double &th = -1.0, cv::InputOutputArray mask = cv::noArray());
    //For correspondence in the image coordinate sytem
    cv::Mat_<double> POSELIB_API triangulatePoints_img(const cv::Mat &R1, const cv::Mat &t1, const cv::Mat &R2, const cv::Mat &t2, const cv::Mat &K1, const cv::Mat &K2, const cv::Mat &p1, const cv::Mat &p2, const double &th = -1.0, cv::InputOutputArray mask = cv::noArray());
    //For correspondence in camera coordinates
    cv::Mat_<double> POSELIB_API triangulatePoints_cam(const cv::Mat &R1, const cv::Mat &t1, const cv::Mat &R2, const cv::Mat &t2, const cv::Mat &p1, const cv::Mat &p2, const double &th = -1.0, cv::InputOutputArray mask = cv::noArray());
    cv::Mat_<double> POSELIB_API triangulatePoints(const cv::Matx34d &P1, const cv::Matx34d &P2, const cv::Mat &p1, const cv::Mat &p2, const double &th = -1.0, cv::InputOutputArray mask = cv::noArray());
    cv::Mat_<double> POSELIB_API getPose_triangulatePoints(cv::InputArray E, cv::InputArray x1, cv::InputArray x2, cv::OutputArray R, cv::OutputArray t, cv::InputOutputArray mask = cv::noArray(), cv::InputArray K1 = cv::noArray(), cv::InputArray K2 = cv::noArray(), const bool noTriangulation = false, const double &depth_th = 140., cv::InputArray R_hint = cv::noArray(), cv::InputArray t_hint = cv::noArray(), std::vector<cv::Mat> *Rs_possible = nullptr, std::vector<cv::Mat> *ts_possible = nullptr, const double &th = -1.0);
    double POSELIB_API CalculateDepth(const cv::Matx34d &P, const cv::Mat &point3Dh);
    bool POSELIB_API isInlier(const cv::Matx34d &P, const cv::Mat &point3Dh, const cv::Mat &point2D, const double &th);
    bool POSELIB_API isInlier(const cv::Matx34d &P, const cv::Mat &point3Dh, const cv::Point2d &point2D, const double &th);

    bool POSELIB_API triangulateMultProjections(const int &imgId, const std::vector<cv::Point2d> &ps_ud, const std::vector<int> &camIdxs, const std::vector<cv::Mat> &Pi, const std::vector<cv::Mat> &Ri, const std::vector<cv::Mat> &ti, const std::vector<cv::Mat> &Ki, const std::vector<cv::Point2d> &ps, cv::Mat &Qs, std::vector<std::pair<int, int>> &projectionInvalids, const double &th = 10.0, const bool invalid_return = true);

    bool POSELIB_API triangulateMultProjections(const std::vector<cv::Point2d> &ps_ud, const std::vector<std::pair<int, int>> &ciIdxs, const std::vector<cv::Mat> &Pi, const std::vector<cv::Mat> &Ki, const std::vector<cv::Point2d> &ps, cv::Mat &Qs, std::vector<std::pair<int, int>> &projectionInvalids, const double &th = 10.0, const bool invalid_return = true);

    void POSELIB_API triangulateMultProjectionsArray(const std::vector<std::unordered_map<int, cv::Point2f>> &pts2D, const std::vector<cv::Mat> &R, const std::vector<cv::Mat> &t, const std::vector<cv::Mat> &K, const std::vector<cv::Mat> &distortion, std::vector<bool> &Qs_mask, cv::Mat &Qs, const double &th);

    size_t POSELIB_API reTriangulateMultProjections(const std::vector<cv::Point2d> &ps_ud, const std::vector<cv::Mat> &Pi, const std::vector<cv::Mat> &Ri, const std::vector<cv::Mat> &ti, const std::vector<cv::Mat> &Ki, const std::vector<cv::Point2d> &ps, const std::vector<size_t> &valids, cv::Mat &Q_new, std::vector<size_t> &iv_new, double &errs_mean_new, const double &th2 = 100.0, size_t tries = 0);
}
