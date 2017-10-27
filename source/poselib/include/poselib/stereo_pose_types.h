/**********************************************************************************************************
FILE: nanoflannInterface.h

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: October 2017

LOCATION: TechGate Vienna, Donau-City-Stra?e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides datatypes used within stereo_pose_refinement.h and nanoflannInterface.h
**********************************************************************************************************/

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