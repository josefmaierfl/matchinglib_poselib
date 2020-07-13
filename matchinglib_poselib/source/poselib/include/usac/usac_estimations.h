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
FILE: usac_estimations.h

PLATFORM: Windows 7, MS Visual Studio 2014, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: May 2017

LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides interfaces to the USAC framework for robust parameter estimations.
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "poselib/glob_includes.h"
#include "usac/config/ConfigParamsEssentialMat.h"


/* --------------------------- Defines --------------------------- */



/* --------------------- Function prototypes --------------------- */

//Robustly estimates a fundamental matrix using the USAC framework and checks for degenerate configurations.
int estimateFundMatrixUsac(cv::InputArray p1,
	cv::InputArray p2,
	cv::OutputArray F,
	double & sprt_delta_result,
	double & sprt_epsilon_result,
	double th = 0.8,
	double prosac_beta = 0.09,
	double sprt_delta = 0.05,
	double sprt_epsilon = 0.15,
	cv::OutputArray inliers = cv::noArray(),
	cv::OutputArray H = cv::noArray(),
	cv::OutputArray inliers_degenerate = cv::noArray(),
	double *fraction_degen_inliers = nullptr,
	std::vector<unsigned int> sortedMatchIdx = {},
	bool verbose_ = false);

//Robustly estimates an essential matrix using the USAC framework and checks for degenerate configurations.
int estimateEssentialMatUsac(const cv::Mat & p1,
	const cv::Mat & p2,
	cv::OutputArray E,
	double & sprt_delta_result,
	double & sprt_epsilon_result,
	double th,
	double focalLength,
	double th_pixels = 0.8,
	double prosac_beta = 0.09,
	double sprt_delta = 0.05,
	double sprt_epsilon = 0.15,
	bool checkDegeneracy = true,
	USACConfig::EssentialMatEstimatorUsed used_estimator = USACConfig::ESTIM_STEWENIUS,
	USACConfig::RefineAlgorithm	refineMethod = USACConfig::REFINE_8PT_PSEUDOHUBER,
	cv::OutputArray inliers = cv::noArray(),
	unsigned int *nr_inliers = nullptr,
	cv::OutputArray H = cv::noArray(),
	cv::OutputArray inliers_degenerate_H = cv::noArray(),
	cv::OutputArray R = cv::noArray(),
	cv::OutputArray inliers_degenerate_R = cv::noArray(),
	cv::OutputArray t = cv::noArray(),
	cv::OutputArray inliers_degenerate_t = cv::noArray(),
	cv::OutputArray inliers_degenerate_noMotion = cv::noArray(),
	double *fraction_degen_inliers_H = nullptr,
	double *fraction_degen_inliers_R = nullptr,
	double *fraction_degen_inliers_t = nullptr,
	double *fraction_degen_inliers_noMot = nullptr,
	std::vector<unsigned int> sortedMatchIdx = {},
	cv::OutputArray R_E = cv::noArray(),
	cv::OutputArray t_E = cv::noArray(),
	bool verbose_ = false);

//Robustly estimates a rotation matrix using the USAC framework
int estimateRotationMatUsac(const cv::Mat & p1,
	const cv::Mat & p2,
	cv::OutputArray R,
	double & sprt_delta_result,
	double & sprt_epsilon_result,
	double focalLength,
	double th_pixels = 0.8,
	double prosac_beta = 0.09,
	double sprt_delta = 0.05,
	double sprt_epsilon = 0.15,
	cv::OutputArray inliers = cv::noArray(),
	unsigned int *nr_inliers = nullptr,
	std::vector<unsigned int> sortedMatchIdx = {},
	bool verbose_ = false);

//If a degenerate model (rotation only) was detected and it is quasi-degenerate in reality, this function tries to upgrade to an essential matrix
int upgradeEssentialMatDegenUsac(const cv::Mat & p1,
	const cv::Mat & p2,
	cv::InputArray inliers_degen,
	cv::OutputArray E,
	double & sprt_delta_result,
	double & sprt_epsilon_result,
	double th,
	double focalLength,
	double th_pixels = 0.8,
	double sprt_delta = 0.05,
	double sprt_epsilon = 0.15,
	USACConfig::EssentialMatEstimatorUsed used_estimator = USACConfig::ESTIM_STEWENIUS,
	USACConfig::RefineAlgorithm	refineMethod = USACConfig::REFINE_8PT_PSEUDOHUBER,
	cv::OutputArray inliers = cv::noArray(),
	unsigned int *nr_inliers = nullptr,
	cv::OutputArray R = cv::noArray(),
	cv::OutputArray t = cv::noArray(),
	bool verbose_ = false);

//Estimation of the Essential matrix or rotation matrix (degenerate case) using QDEGSAC and multiple USACs. Degeneracy is detected robustly.
int estimateEssentialQDEGSAC(const cv::Mat & p1,
	const cv::Mat & p2,
	cv::OutputArray E,
	double & sprt_delta_result,
	double & sprt_epsilon_result,
	double th,
	double focalLength,
	double th_pixels = 0.8,
	double prosac_beta = 0.09,
	double sprt_delta = 0.05,
	double sprt_epsilon = 0.15,
	double t_red = 0.5,//Inlier ratio threshold for detecting degeneracy
	USACConfig::EssentialMatEstimatorUsed used_estimator = USACConfig::ESTIM_STEWENIUS,
	USACConfig::RefineAlgorithm	refineMethod = USACConfig::REFINE_8PT_PSEUDOHUBER,
	cv::OutputArray inliers = cv::noArray(),
	unsigned int *nr_inliers = nullptr,
	cv::OutputArray R = cv::noArray(),
	cv::OutputArray inliers_degenerate_R = cv::noArray(),
	double *fraction_degen_inliers_R = nullptr,
	std::vector<unsigned int> sortedMatchIdx = {},
	cv::OutputArray R_E = cv::noArray(),
	cv::OutputArray t_E = cv::noArray(),
	bool verbose_ = false);
