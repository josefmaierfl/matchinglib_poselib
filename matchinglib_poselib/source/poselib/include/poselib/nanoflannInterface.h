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
FILE: nanoflannInterface.h

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: October 2017

LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

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

        virtual ~keyPointTreeInterface();

        int buildInitialTree();

        int resetTree(std::list<CoordinateProps> *correspondencePool_,
            std::unordered_map<size_t, std::list<CoordinateProps>::iterator> *poolIdxIt_);

        void killTree();

        int addElements(size_t firstIdx, size_t length);

        void removeElements(size_t idx);

        size_t knnSearch(const cv::Point2f &queryPt, size_t knn, std::vector<std::pair<size_t, float>> & result);

        size_t radiusSearch(const cv::Point2f &queryPt, float radius, std::vector<std::pair<size_t, float>> & result);
    };


    /* -------------------------- Functions -------------------------- */

}
