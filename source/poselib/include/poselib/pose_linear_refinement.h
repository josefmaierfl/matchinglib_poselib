/**********************************************************************************************************
FILE: pose_linear_refinement.cpp

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: October 2017

LOCATION: TechGate Vienna, Donau-City-Stra?e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functions for linear refinement of Essential matrices
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "poselib/glob_includes.h"
#include "poselib/pose_estim.h"
//#include <Eigen/Core>

#include "poselib/poselib_api.h"

namespace poselib
{

	/* --------------------------- Defines --------------------------- */



	/* --------------------- Function prototypes --------------------- */

	bool POSELIB_API refineEssentialLinear(cv::InputArray p1,
		cv::InputArray p2,
		cv::InputOutputArray E,
		cv::InputOutputArray mask,
		int refineMethod,//a combination of poselib::RefinePostAlg
		unsigned int & nr_inliers,
		cv::InputOutputArray R = cv::noArray(),
		cv::OutputArray t = cv::noArray(),
		double th = 0.8,
		unsigned int num_iterative_steps = 4,
		double threshold_multiplier = 2.0,
		double pseudoHuberThreshold_multiplier = 0.1,
		double maxRelativeInlierCntLoss = 0.15);
}