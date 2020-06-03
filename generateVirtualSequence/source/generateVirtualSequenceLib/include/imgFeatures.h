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

//
// Created by maierj on 06.03.19.
//

#pragma once

#include "opencv2/core/core.hpp"
#include <opencv2/core/types.hpp>
#include "glob_includes.h"

namespace matchinglib
{

    /* --------------------------- Defines --------------------------- */


    /* --------------------- Function prototypes --------------------- */
//Find the keypoints in the image
    int getKeypoints(cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, std::string& keypointtype, bool dynamicKeypDet = true,
    int limitNrfeatures = 8000);
//Extraction of descriptors at given keypoint locations
    int getDescriptors(cv::Mat &img,
    std::vector<cv::KeyPoint> &keypoints,
            std::string& descriptortype,
    cv::Mat & descriptors,
            std::string const& keypointtype = "");

    bool IsKeypointTypeSupported(std::string const& type);
    std::vector<std::string> GetSupportedKeypointTypes();

    bool IsDescriptorTypeSupported(std::string const& type);
    std::vector<std::string> GetSupportedDescriptorTypes();

} // namepace matchinglib
