//
// Created by maierj on 06.03.19.
//

#ifndef GENERATEVIRTUALSEQUENCE_GENERATEMATCHES_H
#define GENERATEVIRTUALSEQUENCE_GENERATEMATCHES_H

#include "generateSequence.h"
#include <opencv2/core/types.hpp>

struct GENERATEVIRTUALSEQUENCELIB_API GenMatchSequParameters{
    std::string mainStorePath;//Path for storing results
    std::string imgPath;//Path containing the images for producing keypoint patches
    std::string imgPrePostFix;//image pre- and/or postfix (supports wildcards & subfolders) for images within imgPath
    std::string keyPointType;//Name of keypoint detector
    std::string descriptorType;//Name of descriptor extractor
    bool keypPosErrType;//Keypoint detector error (true) or error normal distribution (false)
    std::pair<double, double> keypErrDistr;//Keypoint error distribution (mean, std)
    std::pair<double, double> imgIntNoise;//Noise (mean, std) on the image intensity for descriptor calculation
    double lostCorrPor;//Portion (0 to 0.9) of lost correspondences from frame to frame. It corresponds to the portion of backprojected 3D-points that should not be visible in the frame.
    bool storePtClouds;//If true, all PCL point clouds and necessary information to load a cam sequence with correspondences are stored to disk
    bool takeLessFramesIfLessKeyP;//If true and too less images images are provided (resulting in too less keypoints), only as many frames with GT matches are provided as keypoints are available. Otherwise (false), the point clouds and other information are stored to disk and the generation of GT matches is aborted.

    GenMatchSequParameters(std::string mainStorePath_,
                           std::string imgPath_,
                           std::string imgPrePostFix_,
                           std::string keyPointType_,
                           std::string descriptorType_,
                           bool keypPosErrType_ = false,
                           std::pair<double, double> keypErrDistr_ = std::make_pair(0, 0.5),
                           std::pair<double, double> imgIntNoise_ = std::make_pair(0, 5.0),
                           double lostCorrPor_ = 0,
                           bool storePtClouds_ = false,
                           bool takeLessFramesIfLessKeyP_ = false):
            mainStorePath(std::move(mainStorePath_)),
            imgPath(std::move(imgPath_)),
            imgPrePostFix(std::move(imgPrePostFix_)),
            keyPointType(std::move(keyPointType_)),
            descriptorType(std::move(descriptorType_)),
            keypPosErrType(keypPosErrType_),
            keypErrDistr(std::move(keypErrDistr_)),
            imgIntNoise(std::move(imgIntNoise_)),
            lostCorrPor(lostCorrPor_),
            storePtClouds(storePtClouds_),
            takeLessFramesIfLessKeyP(takeLessFramesIfLessKeyP_){
        CV_Assert(keypPosErrType || (!keypPosErrType && (keypErrDistr.first > -5.0) && (keypErrDistr.first < 5.0) && (keypErrDistr.second > -5.0) && (keypErrDistr.second < 5.0)));
        CV_Assert((imgIntNoise.first > -25.0) && (imgIntNoise.first < 25.0) && (imgIntNoise.second > -25.0) && (imgIntNoise.second < 25.0));
        CV_Assert((lostCorrPor >= 0) && (lostCorrPor <= 0.9));
        CV_Assert(!mainStorePath.empty());
        CV_Assert(!imgPath.empty());

    }
};

class GENERATEVIRTUALSEQUENCELIB_API genMatchSequ : genStereoSequ {
public:
    genMatchSequ(cv::Size &imgSize_,
                 cv::Mat &K1_,
                 cv::Mat &K2_,
                 std::vector<cv::Mat> &R_,
                 std::vector<cv::Mat> &t_,
                 StereoSequParameters &pars3D_,
                 GenMatchSequParameters &parsMtch_,
                 uint32_t verboseMatch_ = 0,
                 uint32_t verbose3D = 0) :
                 genStereoSequ(imgSize_, K1_, K2_, R_, t_, pars3D_, verbose3D),
                 parsMtch(parsMtch_),
                 verboseMatch(verboseMatch_)
                 {

    };

    void startCalc() override;

private:
    bool getImageList();
    void createParsHash();
    size_t hashFromSequPars();
    size_t hashFromMtchPars();
    void totalNrCorrs();
    bool getFeatures();
    bool checkMatchability();

public:
    GenMatchSequParameters parsMtch;
    uint32_t verboseMatch = 0;

private:
    size_t minNrFramesMatch = 10;//Minimum number of required frames that should be generated if there are too less keypoints available
    std::vector<std::string> imageList;
    size_t nrCorrsFullSequ;
    std::vector<cv::Mat> imgs;
    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors1;
    size_t nrFramesGenMatches;
    size_t hash_Sequ, hash_Matches;
    std::string hashResult;

};

#endif //GENERATEVIRTUALSEQUENCE_GENERATEMATCHES_H
