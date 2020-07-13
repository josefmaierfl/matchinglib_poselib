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

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "poselib/glob_includes.h"

namespace poselib
{
	/* --------------------------- Defines --------------------------- */

	typedef struct CoordinateProps
	{
		CoordinateProps() : pt1(0, 0),
			pt2(0, 0),
			ptIdx(0),
			poolIdx(0),
			Q(0, 0, 0),
			Q_tooFar(false),
			age(0),
			descrDist(0),
			keyPResponses{ 0,0 },
			nrFound(1),
			meanSampsonError(DBL_MAX),
			SampsonErrors(std::vector<double>())
		{}

		cv::Point2f pt1;//Keypoint position in first/left image
		cv::Point2f pt2;//Keypoint position in second/right image
		size_t ptIdx;//Index which points to the corresponding points in the camera coordinate system
		size_t poolIdx;//Key of the map that holds the iterators to the list entries of this data structure
		cv::Point3d Q;//Triangulated 3D point
		bool Q_tooFar;//If the z-coordinate of Q is too large, this falue is true. Too far Q's should be excluded in BA to be more stable.
		size_t age;//For how many estimation iterations is this correspondence alive
		float descrDist;//Descriptor distance
		float keyPResponses[2];//Response of corresponding keypoints in the first and second image
		size_t nrFound;//How often the correspondence was found in different image pairs
		double meanSampsonError;//Mean sampson Error over all available E within age
		std::vector<double> SampsonErrors;//Sampson Error for every E this correspondence was used to calculate it
	} CoordinateProps;
}