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
		size_t & nr_inliers,
		cv::InputOutputArray R = cv::noArray(),
		cv::OutputArray t = cv::noArray(),
		double th = 0.8,
		size_t num_iterative_steps = 4,
		double threshold_multiplier = 2.0,
		double pseudoHuberThreshold_multiplier = 0.1,
		double maxRelativeInlierCntLoss = 0.15);
}