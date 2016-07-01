/**********************************************************************************************************
 FILE: pose_homography.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: June 2016

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionality to robustly estimate a pose out of multiple homographys
 which in turn are generated using point correspondences and a homography estimation algorithm embedded
 in the ARRSAC algorithm.
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "glob_includes.h"

#include "poselib/poselib_api.h"

namespace poselib
{

/* --------------------------- Defines --------------------------- */

#define MIN_PTS_PLANE 15//25 //Minimum number of inliers to consider a plane as valid

#define MAX_PLANES_PER_PAIR 50//12 //Maximum number of planes that are extracted from one image pair (0 means infinity)

/* --------------------- Function prototypes --------------------- */
//Estimates an essential matrix as well as rotation and translation using homography alignment.
int POSELIB_API estimatePoseHomographies(cv::InputArray p1, 
							 cv::InputArray p2, 
							 cv::OutputArray R,
							 cv::OutputArray t,
							 cv::OutputArray E,
							 double th, 
							 int & inliers,
							 cv::InputOutputArray mask = cv::noArray(), 
							 bool checkPlaneStrength = false,
							 bool varTh = false,
							 std::vector<std::pair<cv::Mat,cv::Mat>>* inlier_points = NULL, 
							 std::vector<unsigned int>* numbersHinliers = NULL,
							 std::vector<cv::Mat>* homographies = NULL,
							 std::vector<double>* planeStrengths = NULL);
//Estimates n planes and their homographies in the given set of point correspondences.
int POSELIB_API estimateMultHomographys(cv::InputArray p1, 
							cv::InputArray p2, 
							double th, 
							std::vector<cv::Mat> *inl_mask = NULL, 
							std::vector<std::pair<cv::Mat,cv::Mat>> *inl_points = NULL,
							std::vector<cv::Mat> *Hs = NULL,
							std::vector<unsigned int> *num_inl = NULL,
							bool varTh = true,
							std::vector<double> *planeStrength = NULL,
							unsigned int maxPlanes = 0,
							unsigned int minPtsPerPlane = 4);
//Robust homography estimation using ARRSAC.
bool POSELIB_API computeHomographyArrsac(cv::InputArray points1, 
							 cv::InputArray points2, 
							 cv::OutputArray H, 
							 double th, 
							 cv::OutputArray mask = cv::noArray(), 
							 cv::OutputArray p_filtered1 = cv::noArray(), 
							 cv::OutputArray p_filtered2 = cv::noArray());

}