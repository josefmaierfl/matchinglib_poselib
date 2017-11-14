/**********************************************************************************************************
FILE: gms.cpp

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.0

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: November 2016

LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: Interface for filtering matches with the GMS algorithm
**********************************************************************************************************/

#include "../include/matchinglib/gms.h"
#include "MatchGMS.h"

using namespace cv;
using namespace std;

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

/* --------------------- Functions --------------------- */

int filterMatchesGMS(const std::vector<cv::KeyPoint>& keypoints1, const cv::Size imageSize1,
	const std::vector<cv::KeyPoint>& keypoints2, const cv::Size imageSize2,
	const std::vector<cv::DMatch>& matches,
	std::vector<bool>& inlierMask, const bool useScale, const bool useRotation)
{
	MatchGMS gms(keypoints1, imageSize1, keypoints2, imageSize2, matches);
	return gms.getInlierMask(inlierMask, useScale, useRotation);
}

int filterMatchesGMS(const std::vector<cv::KeyPoint>& keypoints1, const cv::Size imageSize1,
	const std::vector<cv::KeyPoint>& keypoints2, const cv::Size imageSize2,
	const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& matches_filtered,
	const bool useScale, const bool useRotation)
{
	std::vector<bool> inlierMask;
	int n = 0;
	MatchGMS gms(keypoints1, imageSize1, keypoints2, imageSize2, matches);
	n = gms.getInlierMask(inlierMask, useScale, useRotation);

	matches_filtered.clear();
	matches_filtered.resize(n);
	if (n)
	{
		for (size_t i = 0, j = 0; i < inlierMask.size(); i++)
		{
			if (inlierMask[i])
				matches_filtered[j++] = matches[i];
		}
	}

	return n;
}