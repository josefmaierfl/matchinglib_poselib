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
 FILE: features.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for extracting keypoints and generating descriptors as
        well as for sub-pixel refinement
**********************************************************************************************************/

#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "matchinglib/glob_includes.h"

#include <unordered_map>
#include <unordered_set>

#include "matchinglib/matchinglib_api.h"

namespace matchinglib
{

  /* --------------------------- Defines --------------------------- */


  /* --------------------- Function prototypes --------------------- */
//Find the keypoints in the image
  int MATCHINGLIB_API getKeypoints(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, const std::string &keypointtype, const bool dynamicKeypDet = true, int limitNrfeatures = 8000);
  int MATCHINGLIB_API getKeypoints(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, const std::string &keypointtype, cv::Ptr<cv::FeatureDetector> &detector, const bool dynamicKeypDet = true, const int limitNrfeatures = 8000);
  //Extraction of descriptors at given keypoint locations
  int MATCHINGLIB_API getDescriptors(const cv::Mat &img,
                                     std::vector<cv::KeyPoint> &keypoints,
                                     std::string const &descriptortype,
                                     cv::Mat &descriptors,
                                     std::string const &keypointtype = "",
                                     const bool affineInvariant = false);
  int MATCHINGLIB_API getDescriptors(const cv::Mat &img,
                                     std::vector<cv::KeyPoint> &keypoints,
                                     std::string const &descriptortype,
                                     cv::Mat &descriptors,
                                     cv::Ptr<cv::DescriptorExtractor> &extractor,
                                     std::string const &keypointtype = "",
                                     const bool affineInvariant = false);

  //Filters keypoints with a small response value within multiple grids.
  void MATCHINGLIB_API responseFilterGridBased(std::vector<cv::KeyPoint> &keys, cv::Size imgSi, int number);

  bool MATCHINGLIB_API IsKeypointTypeSupported(std::string const& type);
  std::vector<std::string> MATCHINGLIB_API GetSupportedKeypointTypes();

  bool MATCHINGLIB_API IsDescriptorTypeSupported(std::string const& type);
  std::vector<std::string> MATCHINGLIB_API GetSupportedDescriptorTypes();

  bool MATCHINGLIB_API IsKeypointTypeToDescriptorTypeCompatible(const std::string &keypointType, const std::string &descriptorType);
  const MATCHINGLIB_API std::unordered_map<std::string, std::unordered_set<std::string>> &GetUnsupportedKeypointDescriptorCombs();

  bool MATCHINGLIB_API IsBinaryDescriptor(const std::string &type);
  std::vector<std::string> MATCHINGLIB_API GetBinaryDescriptorTypes();

} // namepace matchinglib
