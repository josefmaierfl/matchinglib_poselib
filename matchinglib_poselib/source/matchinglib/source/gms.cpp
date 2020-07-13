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
FILE: gms.cpp

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.0

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: November 2016

LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

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