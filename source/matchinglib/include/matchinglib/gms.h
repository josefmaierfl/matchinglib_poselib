/**********************************************************************************************************
FILE: gms.h

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.0

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: November 2016

LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: Interface for filtering matches with the GMS algorithm
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "matchinglib/glob_includes.h"

#include "matchinglib/matchinglib_api.h"

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

int MATCHINGLIB_API filterMatchesGMS(const std::vector<cv::KeyPoint>& keypoints1, const cv::Size imageSize1,
	const std::vector<cv::KeyPoint>& keypoints2, const cv::Size imageSize2,
	const std::vector<cv::DMatch>& matches,
	std::vector<bool>& inlierMask, const bool useScale = false, const bool useRotation = false);

int MATCHINGLIB_API filterMatchesGMS(const std::vector<cv::KeyPoint>& keypoints1, const cv::Size imageSize1,
	const std::vector<cv::KeyPoint>& keypoints2, const cv::Size imageSize2,
	const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& matches_filtered,
	const bool useScale = false, const bool useRotation = false);
