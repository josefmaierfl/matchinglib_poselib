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
 FILE: match_statOptFlow.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for matching features
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "matchinglib/matchinglib_api.h"

#include <memory>
#include <string>
#include <random>

namespace matchinglib
{

  /* --------------------------- Defines --------------------------- */


  /* --------------------- Function prototypes --------------------- */

//Matches 2 feature sets with an user selectable matching algorithm.
  int MATCHINGLIB_API getMatches(const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                                 cv::Mat const &descriptors1, cv::Mat const &descriptors2, cv::Size imgSi, std::vector<cv::DMatch> &finalMatches,
                                 std::string const &matcher_name = "GMBSOF", bool VFCrefine = false, bool ratioTest = true,
                                 std::string const &descriptor_name = "", std::string idxPars_NMSLIB = "", std::string queryPars_NMSLIB = "", const size_t nr_threads = 0);
  
  int MATCHINGLIB_API getMatches(const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                                 cv::Mat const &descriptors1, cv::Mat const &descriptors2, cv::Size imgSi, std::vector<cv::DMatch> &finalMatches, std::mt19937 &mt,
                                 std::string const &matcher_name = "GMBSOF", bool VFCrefine = false, bool ratioTest = true,
                                 std::string const &descriptor_name = "", std::string idxPars_NMSLIB = "", std::string queryPars_NMSLIB = "", const size_t nr_threads = 0);

  int MATCHINGLIB_API getMatches_OpticalFlow(const std::vector<cv::KeyPoint> &keypoints_prev, const std::vector<cv::KeyPoint> &keypoints_next,
      cv::Mat &img_prev, cv::Mat const& img_next, std::vector<cv::DMatch> & finalMatches,
      bool const buildpyr = false, bool drawRes = true, cv::Size winSize = cv::Size(31,31), float searchRadius_px = 10.0f);

  int MATCHINGLIB_API getMatches_OpticalFlowAdvanced(const std::vector<cv::KeyPoint> &keypoints_prev,
      const std::vector<cv::KeyPoint> &keypoints_next,
      cv::Mat const& descriptors1, cv::Mat const& descriptors2,
      cv::Mat &img_prev, cv::Mat const& img_next, std::vector<cv::DMatch> & finalMatches, std::string const& matcher_name = "ALKOF",
      bool const buildpyr = false, bool drawRes = true, cv::Size winSize = cv::Size(31,31), float searchRadius_px = 10.0f,
      unsigned const numNeighbors = 3, const float maxHammDist = 50.0f);

  int MATCHINGLIB_API getMatches_OpticalFlowTracker(std::vector<cv::KeyPoint> & keypoints_prev,
      std::vector<cv::KeyPoint> & keypoints_predicted,
      cv::Mat const& descriptors1,
      cv::Mat &img_prev, cv::Mat &img_next,
      std::vector<cv::DMatch> & finalMatches,
      std::string const& matcher_name = "LKOFT", std::string const& desciptor_type = "ORB",
      bool const buildpyr = false, bool drawRes = true, cv::Size winSize = cv::Size(31,31), const float maxHammDist = 50.0f);

//This function calculates the subpixel-position of matched keypoints by template matching
  int MATCHINGLIB_API getSubPixMatches(cv::Mat& img1, cv::Mat& img2, std::vector<cv::KeyPoint> *keypoints1,
                                       std::vector<cv::KeyPoint> *keypoints2, std::vector<bool> *inliers = NULL);
  
  //This function calculates the subpixel-position of matched keypoints by using the OpenCV function cv::cornerSubPix()
  int MATCHINGLIB_API getSubPixMatches_seperate_Imgs(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> *keypoints1, std::vector<cv::KeyPoint> *keypoints2,
	  std::vector<bool> *inliers);

  bool MATCHINGLIB_API IsMatcherSupported(std::string const& type);
  std::vector<std::string> MATCHINGLIB_API GetSupportedMatcher();



}
