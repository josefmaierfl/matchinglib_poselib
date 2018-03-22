/**********************************************************************************************************
FILE: readGTM.h

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: May 2017

LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for reading the GTMs from a given file.
**********************************************************************************************************/

#pragma once

#include "glob_includes.h"
#include <fstream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

/* Reads a line from the stream, checks if the first word corresponds to the given keyword and if true, reads the value after the keyword.
*
* ifstream gtFromFile			Input  -> Input stream
* string keyWord				Input  -> Keyword that is compared to the first word of the stream
* double *value				Output -> Value from the stream after the keyword
*
* Return value:				true:	Success
*								false:	Failed
*/
bool read1DoubleVal(std::ifstream & gtFromFile, std::string keyWord, double *value)
{
	std::string singleLine;
	size_t strpos;

	std::getline(gtFromFile, singleLine);
	if (singleLine.empty()) return false;
	strpos = singleLine.find(keyWord);
	if (strpos == std::string::npos)
	{
		gtFromFile.close();
		return false;
	}
	singleLine = singleLine.substr(strpos + keyWord.size());
	*value = strtod(singleLine.c_str(), NULL);

	return true;
}

/* Checks, if ground truth information for the given image pair, inlier ratio and feature detector type is already available on the harddisk.
*
* string filenameGT			Input  -> The path and filename of the ground truth file
*
* Return value:				true:	Reading GT file successful
*								false:	Reading GT file failed
*/
bool readGTMatchesDisk(std::string filenameGT, 
	std::vector<bool> & leftInlier,
	std::vector<cv::DMatch> & matchesGT,
	std::vector<cv::KeyPoint> & keypL,
	std::vector<cv::KeyPoint> & keypR,
	double *inlRatioL = NULL,
	double *inlRatioR = NULL,
	double *inlRatioO = NULL,
	double *positivesGT = NULL,
	double *negativesGTl = NULL,
	double *negativesGTr = NULL,
	double *usedMatchTH = NULL)
{
	double inlRatioL_;
	double inlRatioR_;
	double inlRatioO_;
	double positivesGT_;
	double negativesGTl_;
	double negativesGTr_;
	double usedMatchTH_;

	if (filenameGT.empty())
		return false;

	std::ifstream gtFromFile(filenameGT.c_str());
	if (!gtFromFile.good())
	{
		gtFromFile.close();
		return false;
	}

	if (!read1DoubleVal(gtFromFile, "irl ", &inlRatioL_)) return false;
	if (!read1DoubleVal(gtFromFile, "irr ", &inlRatioR_)) return false;
	if (!read1DoubleVal(gtFromFile, "iro ", &inlRatioO_)) return false;
	if (!read1DoubleVal(gtFromFile, "posGT ", &positivesGT_)) return false;
	if (!read1DoubleVal(gtFromFile, "negGTl ", &negativesGTl_)) return false;
	if (!read1DoubleVal(gtFromFile, "negGTr ", &negativesGTr_)) return false;
	if (!read1DoubleVal(gtFromFile, "th ", &usedMatchTH_)) return false;

	if (usedMatchTH_ == 0)
	{
		if (!(int)floor(100000.0 * (inlRatioL_ - 0.00034) + 0.5) &&
			!(int)floor(100000.0 * (inlRatioR_ - 0.00021) + 0.5) && (positivesGT_ == 0) && (negativesGTl_ == 0) && (negativesGTr_ == 0))
			return true;
		else
			return false;
	}
	if (std::abs(positivesGT_ / (positivesGT_ + negativesGTl_) - inlRatioL_) > 0.005) return false;
	if (std::abs(positivesGT_ / (positivesGT_ + negativesGTr_) - inlRatioR_) > 0.005) return false;
	if (2.0 * positivesGT_ / (double)(negativesGTr_ + 2 * positivesGT_ + negativesGTl_) - inlRatioO_ > 0.005) return false;

	bool isInlier;
	int newIntVal;
	float newFloatVal;
	cv::DMatch singleMatch;
	cv::KeyPoint singleKeyPoint;

	//Get inlier vector
	{
		std::string line, word;
		std::istringstream is;
		std::getline(gtFromFile, line);
		if (line.empty())
		{
			gtFromFile.close();
			return false;
		}
		is.str(line);
		is >> word;
		if (word.compare(0, 7, "inliers") != 0)
		{
			gtFromFile.close();
			return false;
		}
		leftInlier.clear();
		while (is >> std::boolalpha >> isInlier)
		{
			leftInlier.push_back(isInlier);
		}
		if (leftInlier.empty())
		{
			gtFromFile.close();
			return false;
		}
	}

	//get matches
	{
		std::string line, word;
		std::istringstream is;
		std::getline(gtFromFile, line);
		if (line.empty())
		{	
			gtFromFile.close();
			return false;
		}
		is.str(line);
		is >> word;
		if (word.compare(0, 7, "matches") != 0)
		{			
			gtFromFile.close();
			return false;
		}
		matchesGT.clear();
		while (is >> newIntVal)
		{
			singleMatch.queryIdx = newIntVal;
			if (is >> newIntVal)
				singleMatch.trainIdx = newIntVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			if (is >> newFloatVal)
				singleMatch.distance = newFloatVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			matchesGT.push_back(singleMatch);
		}
		if (matchesGT.empty())
		{			
			gtFromFile.close();
			return false;
		}
	}

	//get left keypoints
	{
		std::string line, word;
		std::istringstream is;
		std::getline(gtFromFile, line);
		if (line.empty())
		{			
			gtFromFile.close();
			return false;
		}
		is.str(line);
		is >> word;
		if (word.compare(0, 5, "keypl") != 0)
		{			
			gtFromFile.close();
			return false;
		}
		keypL.clear();
		while (is >> newFloatVal)
		{
			singleKeyPoint.pt.x = newFloatVal;
			if (is >> newFloatVal)
				singleKeyPoint.pt.y = newFloatVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			if (is >> newFloatVal)
				singleKeyPoint.response = newFloatVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			if (is >> newFloatVal)
				singleKeyPoint.angle = newFloatVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			if (is >> newFloatVal)
				singleKeyPoint.size = newFloatVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			if (is >> newIntVal)
				singleKeyPoint.octave = newIntVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			if (is >> newIntVal)
				singleKeyPoint.class_id = newIntVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			keypL.push_back(singleKeyPoint);
		}
		if (keypL.empty())
		{			
			gtFromFile.close();
			return false;
		}
	}

	//get right keypoints
	{
		std::string line, word;
		std::istringstream is;
		std::getline(gtFromFile, line);
		if (line.empty())
		{			
			gtFromFile.close();
			return false;
		}
		is.str(line);
		is >> word;
		if (word.compare(0, 5, "keypr") != 0)
		{			
			gtFromFile.close();
			return false;
		}
		keypR.clear();
		while (is >> newFloatVal)
		{
			singleKeyPoint.pt.x = newFloatVal;
			if (is >> newFloatVal)
				singleKeyPoint.pt.y = newFloatVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			if (is >> newFloatVal)
				singleKeyPoint.response = newFloatVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			if (is >> newFloatVal)
				singleKeyPoint.angle = newFloatVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			if (is >> newFloatVal)
				singleKeyPoint.size = newFloatVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			if (is >> newIntVal)
				singleKeyPoint.octave = newIntVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			if (is >> newIntVal)
				singleKeyPoint.class_id = newIntVal;
			else
			{				
				gtFromFile.close();
				return false;
			}
			keypR.push_back(singleKeyPoint);
		}
		if (keypR.empty())
		{			
			gtFromFile.close();
			return false;
		}
	}

	if (keypL.size() != leftInlier.size()) return false;
	if ((double)keypR.size() - (positivesGT_ + negativesGTr_) != 0) return false;
	if ((double)keypL.size() - (positivesGT_ + negativesGTl_) != 0) return false;
	if ((double)matchesGT.size() - positivesGT_ != 0) return false;

	if (inlRatioL)
		*inlRatioL = inlRatioL_;
	if (inlRatioL)
		*inlRatioL = inlRatioL_;
	if (inlRatioR)
		*inlRatioR = inlRatioR_;
	if (inlRatioO)
		*inlRatioO = inlRatioO_;
	if (positivesGT)
		*positivesGT = positivesGT_;
	if (negativesGTl)
		*negativesGTl = negativesGTl_;
	if (negativesGTr)
		*negativesGTr = negativesGTr_;
	if (usedMatchTH)
		*usedMatchTH = usedMatchTH_;

	return true;
}
