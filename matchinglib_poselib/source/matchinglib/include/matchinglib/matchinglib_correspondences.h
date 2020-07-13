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
 FILE: correspondences.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

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

void MATCHINGLIB_API filterMatchesSOF(const std::vector<cv::KeyPoint> &keypoints1,
                          const std::vector<cv::KeyPoint> &keypoints2,
                          const cv::Size &imgSi,
                          std::vector<cv::DMatch> &matches);

bool MATCHINGLIB_API IsKeypointTypeSupported(std::string const& type);
std::vector<std::string> MATCHINGLIB_API GetSupportedKeypointTypes();

bool MATCHINGLIB_API IsDescriptorTypeSupported(std::string const& type);
std::vector<std::string> MATCHINGLIB_API GetSupportedDescriptorTypes();

} // namepace matchinglib
