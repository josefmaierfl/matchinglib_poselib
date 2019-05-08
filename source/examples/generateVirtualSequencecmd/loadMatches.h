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
    /* Holds the keypoints from the images used to extract patches (image indices for keypoints
     * are stored in srcImgPatchKpImgIdx)*/
    std::vector<cv::KeyPoint> srcImgPatchKp;
    /* Holds the image indices of the images used to extract patches for every keypoint in srcImgPatchKp (same order)*/
    std::vector<int> srcImgPatchKpImgIdx;
    /* Specifies the type of a correspondence (TN from static (=4) or TN from moving (=5) object,
     * or TP from a new static (=0), a new moving (=1), an old static (=2), or an old moving (=3)
     * object (old means, that the corresponding 3D point emerged before this stereo frame and
     * also has one or more correspondences in a different stereo frame))*/
    std::vector<int> corrType;
};

void operator >> (const cv::FileNode& n, bool& value)
{
    int bVal;
    n >> bVal;
    value = false;
    if(bVal){
        value = true;
    }
}

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
    for (; it != it_end; ++it) {
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

    fs["srcImgPatchKp"] >> sm.srcImgPatchKp;

    n = fs["srcImgPatchKpImgIdx"];
    if (n.type() != FileNode::SEQ) {
        cerr << "srcImgPatchKpImgIdx is not a sequence! FAIL" << endl;
        return false;
    }
    sm.srcImgPatchKpImgIdx.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        int tmp = 0;
        it >> tmp;
        sm.srcImgPatchKpImgIdx.push_back(tmp);
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



#endif //GENERATEVIRTUALSEQUENCE_LOADMATCHES_H
