/**********************************************************************************************************
 FILE: correspondences.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for generating matched feature sets out of image 
			  information.
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "glob_includes.h"

#include "matchinglib\matchinglib_api.h"

namespace matchinglib
{

/* --------------------------- Defines --------------------------- */


/* --------------------- Function prototypes --------------------- */

//Generation of features followed by matching, filtering, and subpixel-refinement.
int MATCHINGLIB_API getCorrespondences(cv::Mat img1, 
					   cv::Mat img2,
					   std::vector<cv::DMatch> & finalMatches,
					   std::string featuretype = "FAST", 
					   std::string extractortype = "FREAK", 
					   std::string matchertype = "GMBSOF", 
					   bool dynamicKeypDet = true,
					   int limitNrfeatures = 8000, 
					   bool VFCrefine = false, 
					   bool ratioTest = true,
					   bool SOFrefine = false,
					   bool subPixRefine = false);

}