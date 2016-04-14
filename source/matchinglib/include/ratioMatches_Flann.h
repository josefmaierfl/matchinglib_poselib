/**********************************************************************************************************
 FILE: ratioMatches_Flann.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: October 2015

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: Workaround for using the flann lib with OpenCV lib using the cv namespace in other files
**********************************************************************************************************/
#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "..\include\glob_includes.h"

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

//This function performs the rario test and returns only matches with the best ratio
bool ratioTestFlannMatches(const cv::Mat descriptors1, const cv::Mat descriptors2,
                         std::vector<cv::DMatch>& filteredMatches12, double *estim_inlRatio = NULL, bool onlyRatTestMatches = false);