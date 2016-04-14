/**********************************************************************************************************
 FILE: vfcMatches.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: February 2016

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: Interface for filtering matches with the VFC algorithm
**********************************************************************************************************/
#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "..\include\glob_includes.h"

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

//Filters matches using the VFC algorithm
int filterWithVFC(std::vector<cv::KeyPoint> keypL, std::vector<cv::KeyPoint> keypR, std::vector<cv::DMatch> matches_in, std::vector<cv::DMatch> & matches_out);