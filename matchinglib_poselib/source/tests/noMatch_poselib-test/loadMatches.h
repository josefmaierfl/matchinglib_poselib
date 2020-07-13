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
// Created by maierj on 08.05.19.
//

#ifndef GENERATEVIRTUALSEQUENCE_LOADMATCHES_H
#define GENERATEVIRTUALSEQUENCE_LOADMATCHES_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

struct sequMatches{
    /* Keypoints for the first (left or top) stereo cam (there is no 1:1 correspondence between
     * frameKeypoints1 and frameKeypoints2 as they are shuffled but the keypoint order of each
     * of them is the same compared to their corresponding descriptor Mat (rows))*/
    std::vector<cv::KeyPoint> frameKeypoints1;
    /* Keypoints for the second (right or bottom) stereo cam (there is no 1:1 correspondence between
     * frameKeypoints1 and frameKeypoints2 as they are shuffled but the keypoint order of each
     * of them is the same compared to their corresponding descriptor Mat (rows))*/
    std::vector<cv::KeyPoint> frameKeypoints2;
    /* Descriptors for first (left or top) stereo cam (there is no 1:1 correspondence between
     * frameDescriptors1 and frameDescriptors2 as they are shuffled but the descriptor order
     * is the same compared to its corresponding keypoint vector frameKeypoints1).
     * Descriptors corresponding to the same static 3D point (not for moving objects) in different
     * stereo frames are equal*/
    cv::Mat frameDescriptors1;
    /* Descriptors for second (right or bottom) stereo cam (there is no 1:1 correspondence between
     * frameDescriptors1 and frameDescriptors2 as they are shuffled but the descriptor order
     * the same compared to its corresponding keypoint vector frameKeypoints2).
     * Descriptors corresponding to the same static 3D point (not for moving objects) in different
     * stereo frames are similar*/
    cv::Mat frameDescriptors2;
    /* Matches between features of a single stereo frame. They are sorted based on the descriptor
     * distance (smallest first)*/
    std::vector<cv::DMatch> frameMatches;
    /* Indicates if a feature (frameKeypoints1 and corresponding frameDescriptors1) is an inlier.*/
    std::vector<bool> frameInliers;
    /* Keypoints in the second stereo image without a positioning error (in general, keypoints
     * in the first stereo image are without errors)*/
    std::vector<cv::KeyPoint> frameKeypoints2NoErr;
    /* Holds the homographies for all patches arround keypoints for warping the patch which is
     * then used to calculate the matching descriptor. Homographies corresponding to the same
     * static 3D point (not for moving objects) in different stereo frames are similar*/
    std::vector<cv::Mat> frameHomographies;
    /* Holds homographies for all patches arround keypoints in the first camera (for tracked features)
     * for warping the patch which is then used to calculate the matching descriptor.
     * Homographies corresponding to the same static 3D point in different stereo frames are similar
     */
    std::vector<cv::Mat> frameHomographiesCam1;
    /* Holds the keypoints from the images used to extract patches (image indices for keypoints
     * are stored in srcImgPatchKpImgIdx1)*/
    std::vector<cv::KeyPoint> srcImgPatchKp1;
    /* Holds the image indices of the images used to extract patches for every keypoint in srcImgPatchKp1 (same order) */
    std::vector<size_t> srcImgPatchKpImgIdx1;
    /* Holds the keypoints from the images used to extract patches for the second keypoint of a match.
     * (image indices for keypoints are stored in srcImgPatchKpImgIdx2) */
    std::vector<cv::KeyPoint> srcImgPatchKp2;
    /* Holds the image indices of the images used to extract patches for every keypoint in srcImgPatchKp2 (same order) */
    std::vector<size_t> srcImgPatchKpImgIdx2;
    /* Specifies the type of a correspondence (TN from static (=4) or TN from moving (=5) object,
     * or TP from a new static (=0), a new moving (=1), an old static (=2), or an old moving (=3)
     * object (old means, that the corresponding 3D point emerged before this stereo frame and
     * also has one or more correspondences in a different stereo frame))*/
    std::vector<int> corrType;
    /* Frame number*/
    size_t actFrameCnt;
    /* Actual rotation matrix of the stereo rig: x2 = actR * x1 + actT*/
    cv::Mat actR;
    /* Actual translation vector of the stereo rig: x2 = actR * x1 + actT*/
    cv::Mat actT;
    /* Actual correct camera matrix of camera 1*/
    cv::Mat K1;
    /* Actual correct camera matrix of camera 2*/
    cv::Mat K2;
    /* Actual distorted camera matrix of camera 1*/
    cv::Mat actKd1;
    /* Actual distorted camera matrix of camera 1*/
    cv::Mat actKd2;
};

/*void operator >> (const cv::FileNode& n, bool& value)
{
    int bVal;
    n >> bVal;
    value = false;
    if(bVal){
        value = true;
    }
}*/

bool readMatchesFromDisk(const std::string &filename,
                         sequMatches &sm){
    FileStorage fs = FileStorage(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    fs["frameKeypoints1"] >> sm.frameKeypoints1;
    fs["frameKeypoints2"] >> sm.frameKeypoints2;
    fs["frameDescriptors1"] >> sm.frameDescriptors1;
    fs["frameDescriptors2"] >> sm.frameDescriptors2;
    fs["frameMatches"] >> sm.frameMatches;

    FileNode n = fs["frameInliers"];
    if (n.type() != FileNode::SEQ) {
        cerr << "frameInliers is not a sequence! FAIL" << endl;
        return false;
    }
    sm.frameInliers.clear();
    FileNodeIterator it = n.begin(), it_end = n.end();
    while ( it != it_end) {
        bool inli = false;
        it >> inli;
        sm.frameInliers.push_back(inli);
    }

    fs["frameKeypoints2NoErr"] >> sm.frameKeypoints2NoErr;

    n = fs["frameHomographies"];
    if (n.type() != FileNode::SEQ) {
        cerr << "frameHomographies is not a sequence! FAIL" << endl;
        return false;
    }
    sm.frameHomographies.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        Mat m;
        it >> m;
        sm.frameHomographies.emplace_back(m.clone());
    }

    n = fs["frameHomographiesCam1"];
    if (n.type() != FileNode::SEQ) {
        cerr << "frameHomographiesCam1 is not a sequence! FAIL" << endl;
        return false;
    }
    sm.frameHomographiesCam1.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        Mat m;
        it >> m;
        sm.frameHomographiesCam1.emplace_back(m.clone());
    }

    fs["srcImgPatchKp1"] >> sm.srcImgPatchKp1;

    n = fs["srcImgPatchKpImgIdx1"];
    if (n.type() != FileNode::SEQ) {
        cerr << "srcImgPatchKpImgIdx1 is not a sequence! FAIL" << endl;
        return false;
    }
    sm.srcImgPatchKpImgIdx1.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        int tmp = 0;
        it >> tmp;
        sm.srcImgPatchKpImgIdx1.push_back(static_cast<size_t>(tmp));
    }

    fs["srcImgPatchKp2"] >> sm.srcImgPatchKp2;

    n = fs["srcImgPatchKpImgIdx2"];
    if (n.type() != FileNode::SEQ) {
        cerr << "srcImgPatchKpImgIdx2 is not a sequence! FAIL" << endl;
        return false;
    }
    sm.srcImgPatchKpImgIdx2.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        int tmp = 0;
        it >> tmp;
        sm.srcImgPatchKpImgIdx2.push_back(static_cast<size_t>(tmp));
    }

    n = fs["corrType"];
    if (n.type() != FileNode::SEQ) {
        cerr << "corrType is not a sequence! FAIL" << endl;
        return false;
    }
    sm.corrType.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        int tmp = 0;
        it >> tmp;
        sm.corrType.push_back(tmp);
    }

    fs.release();

    return true;
}

bool readCamParsFromDisk(const std::string &filename,
                         sequMatches &sm){
    FileStorage fs(filename, FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    int tmp = 0;

    fs["actFrameCnt"] >> tmp;
    sm.actFrameCnt = (size_t) tmp;
    fs["actR"] >> sm.actR;
    fs["actT"] >> sm.actT;
    fs["K1"] >> sm.K1;
    fs["K2"] >> sm.K2;
    fs["actKd1"] >> sm.actKd1;
    fs["actKd2"] >> sm.actKd2;

    fs.release();

    return true;
}



#endif //GENERATEVIRTUALSEQUENCE_LOADMATCHES_H
