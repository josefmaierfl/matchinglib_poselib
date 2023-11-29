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
 FILE: pose_homography.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: June 2016

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionality to robustly estimate a pose out of multiple homographys
 which in turn are generated using point correspondences and a homography estimation algorithm embedded
 in the ARRSAC algorithm.
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "poselib/glob_includes.h"
#include <random>

#include "poselib/poselib_api.h"

namespace poselib
{

/* --------------------------- Defines --------------------------- */

#define MIN_PTS_PLANE 15//25 //Minimum number of inliers to consider a plane as valid

#define MAX_PLANES_PER_PAIR 50//12 //Maximum number of planes that are extracted from one image pair (0 means infinity)

/* --------------------- Function prototypes --------------------- */
//Estimates an essential matrix as well as rotation and translation using homography alignment.
int POSELIB_API estimatePoseHomographies(cv::InputArray p1,
										 cv::InputArray p2,
										 cv::OutputArray R,
										 cv::OutputArray t,
										 cv::OutputArray E,
										 double th,
										 std::mt19937 &mt,
										 int &inliers,
										 cv::InputOutputArray mask = cv::noArray(),
										 bool checkPlaneStrength = false,
										 bool varTh = false,
										 std::vector<std::pair<cv::Mat, cv::Mat>> *inlier_points = NULL,
										 std::vector<unsigned int> *numbersHinliers = NULL,
										 std::vector<cv::Mat> *homographies = NULL,
										 std::vector<double> *planeStrengths = NULL);
//Estimates n planes and their homographies in the given set of point correspondences.
int POSELIB_API estimateMultHomographys(cv::InputArray p1,
										cv::InputArray p2,
										double th,
										std::mt19937 &mt,
										std::vector<cv::Mat> *inl_mask = NULL,
										std::vector<std::pair<cv::Mat, cv::Mat>> *inl_points = NULL,
										std::vector<cv::Mat> *Hs = NULL,
										std::vector<unsigned int> *num_inl = NULL,
										bool varTh = true,
										std::vector<double> *planeStrength = NULL,
										unsigned int maxPlanes = 0,
										unsigned int minPtsPerPlane = 4);
//Robust homography estimation using ARRSAC.
bool POSELIB_API computeHomographyArrsac(cv::InputArray points1,
										 cv::InputArray points2,
										 cv::OutputArray H,
										 double th,
										 std::mt19937 &mt,
										 cv::OutputArray mask = cv::noArray(),
										 cv::OutputArray p_filtered1 = cv::noArray(),
										 cv::OutputArray p_filtered2 = cv::noArray());
}
