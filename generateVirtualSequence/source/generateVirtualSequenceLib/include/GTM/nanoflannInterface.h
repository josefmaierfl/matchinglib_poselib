//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2020 Josef Maier
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

#pragma once

#include "glob_includes.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <list>
#include <unordered_map>

using namespace cv;

    /* --------------------------- Defines --------------------------- */


    /* --------------------- Function prototypes --------------------- */


    /* ---------------------- Classes & Structs ---------------------- */



class keyPointTreeInterface
{
private:
    void *treePtr;
    bool delWhenDestruct = true;
public:
    keyPointTreeInterface(const std::vector<cv::KeyPoint> *featuresPtr_, bool delWhenDestruct_ = true);

    keyPointTreeInterface(): treePtr(nullptr), delWhenDestruct(true){}

    keyPointTreeInterface& operator=(const keyPointTreeInterface& gss){
        treePtr = gss.treePtr;
        delWhenDestruct = true;
//        gss.treePtr = nullptr;
    }

    keyPointTreeInterface(keyPointTreeInterface& gss): treePtr(gss.treePtr){
        gss.treePtr = nullptr;
    }

    virtual ~keyPointTreeInterface();

    int buildInitialTree();

    int resetTree(const std::vector<cv::KeyPoint> *featuresPtr_);

    void killTree();

    size_t knnSearch(const cv::Point2f &queryPt, size_t knn, std::vector<std::pair<size_t, float>> & result);

    size_t radiusSearch(const cv::Point2f &queryPt, float radius, std::vector<std::pair<size_t, float>> & result);

    int addElements(size_t firstIdx, size_t length);

    void removeElements(size_t idx);
};


/* -------------------------- Functions -------------------------- */

