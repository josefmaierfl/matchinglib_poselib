/**********************************************************************************************************
FILE: nanoflannInterface.h

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: October 2017

LOCATION: TechGate Vienna, Donau-City-Stra?e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functions to search keypoint positions within a tree built using nanoflann
**********************************************************************************************************/

#pragma once

#include "poselib/glob_includes.h"
#include "poselib/stereo_pose_types.h"
#include <list>
#include <unordered_map>

namespace poselib
{

	/* --------------------------- Defines --------------------------- */


	/* --------------------- Function prototypes --------------------- */


	/* ---------------------- Classes & Structs ---------------------- */
	
	

	class keyPointTreeInterface
	{
	private:
		void *treePtr;
	public:
		keyPointTreeInterface(std::list<CoordinateProps> *correspondencePool_,
			std::unordered_map<size_t, std::list<CoordinateProps>::iterator> *poolIdxIt_);

		~keyPointTreeInterface();

		int buildInitialTree();

		int resetTree(std::list<CoordinateProps> *correspondencePool_,
			std::unordered_map<size_t, std::list<CoordinateProps>::iterator> *poolIdxIt_);

		void killTree();

		int addElements(size_t firstIdx, size_t length);

		void removeElements(size_t idx);

		size_t knnSearch(cv::Point2f queryPt, size_t knn, std::vector<std::pair<size_t, float>> & result);

		size_t radiusSearch(cv::Point2f queryPt, float radius, std::vector<std::pair<size_t, float>> & result);
	};


	/* -------------------------- Functions -------------------------- */

}