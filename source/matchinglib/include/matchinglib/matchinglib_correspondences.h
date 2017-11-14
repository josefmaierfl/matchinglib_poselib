/**********************************************************************************************************
 FILE: correspondences.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for generating matched feature sets out of image
              information.
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "matchinglib/glob_includes.h"

#include "matchinglib/matchinglib_api.h"
#include <string>

namespace matchinglib
{

/* --------------------------- Defines --------------------------- */


/* --------------------- Function prototypes --------------------- */

//Generation of features followed by matching, filtering, and subpixel-refinement.
int MATCHINGLIB_API getCorrespondences(cv::Mat& img1,
                       cv::Mat& img2,
                       std::vector<cv::DMatch> & finalMatches,
                       std::vector<cv::KeyPoint> & kp1,
                       std::vector<cv::KeyPoint> & kp2,
                       std::string featuretype = "FAST",
                       std::string extractortype = "FREAK",
                       std::string matchertype = "GMBSOF",
                       bool dynamicKeypDet = true,
                       int limitNrfeatures = 8000,
                       bool VFCrefine = false,
					   bool GMSrefine = false,
                       bool ratioTest = true,
                       bool SOFrefine = false,
                       int subPixRefine = 0,
                       int verbose = 0,
					   std::string idxPars_NMSLIB = "",
					   std::string queryPars_NMSLIB = "");

bool MATCHINGLIB_API IsKeypointTypeSupported(std::string const& type);
std::vector<std::string> MATCHINGLIB_API GetSupportedKeypointTypes();

bool MATCHINGLIB_API IsDescriptorTypeSupported(std::string const& type);
std::vector<std::string> MATCHINGLIB_API GetSupportedDescriptorTypes();

} // namepace matchinglib
