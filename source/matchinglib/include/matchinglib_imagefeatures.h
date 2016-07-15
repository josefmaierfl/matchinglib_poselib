/**********************************************************************************************************
 FILE: features.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for extracting keypoints and generating descriptors as
        well as for sub-pixel refinement
**********************************************************************************************************/

#pragma once

#include "opencv2/core/core.hpp"
#include "glob_includes.h"

#include "matchinglib/matchinglib_api.h"

namespace matchinglib
{

  /* --------------------------- Defines --------------------------- */


  /* --------------------- Function prototypes --------------------- */
//Find the keypoints in the image
  int MATCHINGLIB_API getKeypoints(cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, std::string& keypointtype, bool dynamicKeypDet = true,
                                   int limitNrfeatures = 8000);
//Extraction of descriptors at given keypoint locations
  int MATCHINGLIB_API getDescriptors(cv::Mat &img,
                                     std::vector<cv::KeyPoint> & keypoints,
                                     std::string& descriptortype,
                                     cv::Mat & descriptors);

  bool MATCHINGLIB_API IsKeypointTypeSupported(std::string const& type);
  std::vector<std::string> MATCHINGLIB_API GetSupportedKeypointTypes();

  bool MATCHINGLIB_API IsDescriptorTypeSupported(std::string const& type);
  std::vector<std::string> MATCHINGLIB_API GetSupportedDescriptorTypes();

} // namepace matchinglib

