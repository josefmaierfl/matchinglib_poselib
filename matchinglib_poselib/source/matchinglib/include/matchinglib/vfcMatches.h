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
 FILE: vfcMatches.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: February 2016

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: Interface for filtering matches with the VFC algorithm
**********************************************************************************************************/
#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "matchinglib/glob_includes.h"
#include <random>

#include "matchinglib/matchinglib_api.h"

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

//Filters matches using the VFC algorithm
int MATCHINGLIB_API filterWithVFC(std::vector<cv::KeyPoint> const &keypL, std::vector<cv::KeyPoint> const &keypR, std::vector<cv::DMatch> const &matches_in, std::vector<cv::DMatch> &matches_out);

int MATCHINGLIB_API filterWithVFC(std::vector<cv::KeyPoint> const &keypL, std::vector<cv::KeyPoint> const &keypR, std::vector<cv::DMatch> const &matches_in, std::vector<cv::DMatch> &matches_out, std::mt19937 &mt);
