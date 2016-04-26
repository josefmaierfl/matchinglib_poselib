/**********************************************************************************************************
 FILE: match_statOptFlow.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for matching features
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "matchinglib/matchinglib_api.h"

#include <memory>

namespace matchinglib
{

/* --------------------------- Defines --------------------------- */


/* --------------------- Function prototypes --------------------- */

//Matches 2 feature sets with an user selectable matching algorithm.
int MATCHINGLIB_API getMatches(std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2,
			   cv::Mat descriptors1, cv::Mat descriptors2, cv::Size imgSi, std::vector<cv::DMatch> & finalMatches,
			   std::string matcher_name = "GMBSOF", bool VFCrefine = false, bool ratioTest = true);
//This function calculates the subpixel-position of matched keypoints by template matching
int MATCHINGLIB_API getSubPixMatches(cv::Mat img1, cv::Mat img2, std::vector<cv::KeyPoint> *keypoints1, std::vector<cv::KeyPoint> *keypoints2, std::vector<bool> *inliers = NULL);

}
