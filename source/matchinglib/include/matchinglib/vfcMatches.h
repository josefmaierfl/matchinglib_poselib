/**********************************************************************************************************
 FILE: vfcMatches.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: February 2016

 LOCATION: TechGate Vienna, Donau-City-Straße 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: Interface for filtering matches with the VFC algorithm
**********************************************************************************************************/
#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "matchinglib//glob_includes.h"

#include "matchinglib/matchinglib_api.h"

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

//Filters matches using the VFC algorithm
int MATCHINGLIB_API filterWithVFC(std::vector<cv::KeyPoint> const& keypL, std::vector<cv::KeyPoint> const& keypR, std::vector<cv::DMatch> const& matches_in, std::vector<cv::DMatch> & matches_out);
