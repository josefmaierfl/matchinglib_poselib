//
// Created by maierj on 06.03.19.
//
/**********************************************************************************************************
 FILE: features.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for extracting keypoints and generating descriptors as
        well as for sub-pixel refinement
**********************************************************************************************************/

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
