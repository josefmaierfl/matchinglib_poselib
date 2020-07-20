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
//
// Created by maierj on 7/20/20.
//

#ifndef GENERATEVIRTUALSEQUENCE_READGTM_H
#define GENERATEVIRTUALSEQUENCE_READGTM_H

#include <string>
#include <vector>
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

//Parameters of every image pair from the annotation tool for testing the GT (automatic and/or manual annotation)
struct annotImgPars{
    /* false GT matching coordinates within the GT matches dataset */
    std::vector<std::pair<cv::Point2f,cv::Point2f>> falseGT;
    /* Histogram of the distances from the matching positions to annotated positions */
    std::vector<std::pair<double,int>> distanceHisto;
    /* distances from the matching positions to annotated positions */
    std::vector<double> distances;
    /* Number of matches from the GT that are not matchable in reality */
    int notMatchable;
//	int truePosArr[5];//Number of true positives after matching of the annotated matches
//	int falsePosArr[5];//Number of false positives after matching of the annotated matches
//	int falseNegArr[5];//Number of false negatives after matching of the annotated matches
    /* vector from the matching positions to annotated positions */
    std::vector<cv::Point2f> errvecs;
    /* The resulting annotated matches */
    std::vector<std::pair<cv::Point2f,cv::Point2f>> perfectMatches;
    /* Indices for perfectMatches pointing to corresponding match in matchesGT */
    std::vector<int> matchesGT_idx;
    /* The fundamental matrix or homography calculted from the annotated matches */
    cv::Mat HE;
    /* Validity level of false matches for the filled KITTI GT (1=original GT, 2= filled GT) */
    std::vector<int> validityValFalseGT;
    /* Vectors from the dataset GT to the annotated positions */
    std::vector<cv::Point2f> errvecsGT;
    /* Distances from the dataset GT to the annotated positions */
    std::vector<double> distancesGT;
    /* Validity level of all annoted matches for the filled disparity/flow GT (1=original GT, 2= filled GT, -1= not annotated) */
    std::vector<int> validityValGT;
    /* Distances of the annotated positions to an estimated model (HE) from the annotated positions */
    std::vector<double> distancesEstModel;
//	int> selectedSamples;//Number of samples per image pair
//	int> nrTotalFails;//Number of total fails from the first annotated image pair until the actual image pair
    /* The ID of the image pair in the form "datasetName-datasetPart-leftImageName-rightImageName" or "firstImageName-secondImageName" */
    std::string id;
    /* Holds a 'M' for a manual, an 'A' for an automatic annotated match, and 'U' for matches without refinement */
    std::vector<char> autoManualAnnot;

    annotImgPars(){
        notMatchable = 0;
    }

    void clear(){
        falseGT.clear();
        distanceHisto.clear();
        distances.clear();
        notMatchable = 0;
        errvecs.clear();
        perfectMatches.clear();
        matchesGT_idx.clear();
        HE = cv::Mat();
        validityValFalseGT.clear();
        errvecsGT.clear();
        distancesGT.clear();
        validityValGT.clear();
        distancesEstModel.clear();
        id = "";
        autoManualAnnot.clear();
    }
};

//Ground Truth Matches (GTM) and additional information
struct gtmData{
    /* Feature type like FAST, SIFT, ... that are defined in the OpenCV (FeatureDetector::create) */
    std::string featuretype;
    /* Specifies the descriptor type like FREAK, SIFT, ... for filtering ground truth matches. */
    std::string GTfilterExtractor;
    /* Threshold within a match is considered as true match */
    double usedMatchTH = 0;
    /* Inlier ratio over all keypoints in both images */
    double inlRatio = 0;
    /* Number of inliers (from ground truth) */
    size_t positivesGT = 0;
    /* Number of outliers in left image (from ground truth) */
    size_t negativesGT = 0;
    /* Keypoints in first camera */
    std::vector<cv::KeyPoint> keypL;
    /* Keypoints in second camera */
    std::vector<cv::KeyPoint> keypR;
    /* Ground truth matches (ordered increasing query index), distance parameter does NOT correspond to descriptor
     * distance but to squared distance of the keypoint in the second image to the calculated
     * position from ground truth (homography, disparity, or flow) */
    std::vector<cv::DMatch> matchesGT;
    /* Specifies if an detected feature of the left image has a true match in the right image (true). Corresponds to the index of keypL */
    std::vector<bool> leftInlier;
    /* Specifies if an detected feature of the right image has a true match in the left image (true). Corresponds to the index of keypR */
    std::vector<bool> rightInlier;
    /* True, if refined data was read into "quality" */
    bool refinedGTMAvailable = false;
    /* Parameters of every image pair from the annotation tool for testing the GT (automatic and/or manual annotation) */
    annotImgPars quality;
};

/* Read GTM from disk
 *
 * string filenameGT			Input  -> The path and filename of the ground truth file
 */
int readGTMatchesDisk(const std::string &filenameGT, gtmData &data)
{
    FileStorage fs(filenameGT, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filenameGT << endl;
        return -1;
    }
    FileNode n = fs["noDataAvailable"];
    if(!n.empty()){
        return -2;
    }
    fs["keypointType"] >> data.featuretype;
    fs["descriptorType"] >> data.GTfilterExtractor;
    fs["usedMatchTH"] >> data.usedMatchTH;
    fs["inlRatio"] >> data.inlRatio;
    int tmp = 0;
    fs["positivesGT"] >> tmp;
    data.positivesGT = static_cast<size_t>(tmp);
    fs["negativesGT"] >> tmp;
    data.negativesGT = static_cast<size_t>(tmp);
    fs["keypL"] >> data.keypL;
    fs["keypR"] >> data.keypR;
    fs["matchesGT"] >> data.matchesGT;
    n = fs["leftInlier"];
    if (n.type() != FileNode::SEQ) {
        cerr << "leftInlier is not a sequence! FAIL" << endl;
        return -1;
    }
    data.leftInlier.clear();
    FileNodeIterator it = n.begin(), it_end = n.end();
    while (it != it_end) {
        bool inl = false;
        it >> inl;
        data.leftInlier.push_back(inl);
    }
    n = fs["rightInlier"];
    if (n.type() != FileNode::SEQ) {
        cerr << "rightInlier is not a sequence! FAIL" << endl;
        return -1;
    }
    data.rightInlier.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        bool inl = false;
        it >> inl;
        data.rightInlier.push_back(inl);
    }

    data.quality.clear();
    n = fs["falseGT"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "falseGT is not a sequence! FAIL" << endl;
            return -1;
        }
        it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it) {
            FileNode n1 = *it;
            cv::Point2f p1, p2;
            n1["first"] >> p1;
            n1["second"] >> p2;
            data.quality.falseGT.emplace_back(p1, p2);
        }
    }
    n = fs["distanceHisto"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "distanceHisto is not a sequence! FAIL" << endl;
            return -1;
        }
        it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it) {
            FileNode n1 = *it;
            int v2;
            double v1;
            n1["first"] >> v1;
            n1["second"] >> v2;
            data.quality.distanceHisto.emplace_back(v1, v2);
        }
    }
    n = fs["distances"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "distances is not a sequence! FAIL" << endl;
            return -1;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            double v = 0;
            it >> v;
            data.quality.distances.push_back(v);
        }
    }
    n = fs["notMatchable"];
    if(!n.empty()) {
        n >> data.quality.notMatchable;
    }
    n = fs["errvecs"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "errvecs is not a sequence! FAIL" << endl;
            return -1;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            cv::Point2f p1;
            it >> p1;
            data.quality.errvecs.push_back(p1);
        }
    }
    n = fs["perfectMatches"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "perfectMatches is not a sequence! FAIL" << endl;
            return -1;
        }
        data.refinedGTMAvailable = true;
        it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it) {
            FileNode n1 = *it;
            cv::Point2f p1, p2;
            n1["first"] >> p1;
            n1["second"] >> p2;
            data.quality.perfectMatches.emplace_back(p1, p2);
        }
    }else{
        data.refinedGTMAvailable = false;
    }
    n = fs["matchesGT_idx"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "matchesGT_idx is not a sequence! FAIL" << endl;
            return -1;
        }
        data.refinedGTMAvailable &= true;
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            int v = 0;
            it >> v;
            data.quality.matchesGT_idx.push_back(v);
        }
    }else{
        data.refinedGTMAvailable = false;
    }
    n = fs["HE"];
    if(!n.empty()) {
        n >> data.quality.HE;
    }
    n = fs["validityValFalseGT"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "validityValFalseGT is not a sequence! FAIL" << endl;
            return -1;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            int v = 0;
            it >> v;
            data.quality.validityValFalseGT.push_back(v);
        }
    }
    n = fs["errvecsGT"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "errvecsGT is not a sequence! FAIL" << endl;
            return -1;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            cv::Point2f p1;
            it >> p1;
            data.quality.errvecsGT.push_back(p1);
        }
    }
    n = fs["distancesGT"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "distancesGT is not a sequence! FAIL" << endl;
            return -1;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            double v = 0;
            it >> v;
            data.quality.distancesGT.push_back(v);
        }
    }
    n = fs["validityValGT"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "validityValGT is not a sequence! FAIL" << endl;
            return -1;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            int v = 0;
            it >> v;
            data.quality.validityValGT.push_back(v);
        }
    }
    n = fs["distancesEstModel"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "distancesEstModel is not a sequence! FAIL" << endl;
            return -1;
        }
        data.refinedGTMAvailable &= true;
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            double v = 0;
            it >> v;
            data.quality.distancesEstModel.push_back(v);
        }
    }else{
        data.refinedGTMAvailable = false;
    }
    n = fs["id"];
    if(!n.empty()) {
        n >> data.quality.id;
    }
    n = fs["autoManualAnnot"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "autoManualAnnot is not a sequence! FAIL" << endl;
            return -1;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            string v;
            it >> v;
            data.quality.autoManualAnnot.push_back(*v.c_str());
        }
    }

    return 0;
}

#endif //GENERATEVIRTUALSEQUENCE_READGTM_H
