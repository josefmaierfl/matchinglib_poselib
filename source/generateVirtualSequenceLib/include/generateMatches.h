//
// Created by maierj on 06.03.19.
//

#ifndef GENERATEVIRTUALSEQUENCE_GENERATEMATCHES_H
#define GENERATEVIRTUALSEQUENCE_GENERATEMATCHES_H

#include "generateSequence.h"

struct GENERATEVIRTUALSEQUENCELIB_API GenMatchSequParameters{
    std::string mainStorePath;
    std::string imgPath;
    std::string keyPointType;
    std::string descriptorType;
    bool keypPosErrType;//Keypoint detector error (true) or error normal distribution (false)
    std::pair<double, double> keypErrDistr;//Keypoint error distribution (mean, std)
    std::pair<double, double> imgIntNoise;//Noise (mean, std) on the image intensity for descriptor calculation
    double lostCorrPor;//Portion of lost correspondences from frame to frame. It corresponds to the portion of 3D-points that would be visible in the next frame.

    GenMatchSequParameters(std::string mainStorePath_,
                           std::string imgPath_,
                           std::string keyPointType_,
                           std::string descriptorType_,
                           bool keypPosErrType_ = false,
                           std::pair<double, double> keypErrDistr_ = std::make_pair(0, 0.5),
                           std::pair<double, double> imgIntNoise_ = std::make_pair(0, 5.0),
                           double lostCorrPor_ = 0):
            mainStorePath(std::move(mainStorePath_)),
            imgPath(std::move(imgPath_)),
            keyPointType(std::move(keyPointType_)),
            descriptorType(std::move(descriptorType_)),
            keypPosErrType(keypPosErrType_),
            keypErrDistr(std::move(keypErrDistr_)),
            imgIntNoise(std::move(imgIntNoise_)),
            lostCorrPor(lostCorrPor_){
        CV_Assert(keypPosErrType || (!keypPosErrType && (keypErrDistr.first > -5.0) && (keypErrDistr.first < 5.0) && (keypErrDistr.second > -5.0) && (keypErrDistr.second < 5.0)));
        CV_Assert((imgIntNoise.first > -25.0) && (imgIntNoise.first < 25.0) && (imgIntNoise.second > -25.0) && (imgIntNoise.second < 25.0));
    }
};

class GENERATEVIRTUALSEQUENCELIB_API genMatchSequ : genStereoSequ {
public:
    genMatchSequ(cv::Size &imgSize_,
                 cv::Mat &K1_,
                 cv::Mat &K2_,
                 std::vector<cv::Mat> &R_,
                 std::vector<cv::Mat> &t_,
                 StereoSequParameters &pars_,
                 uint32_t verboseMatch = 0,
                 uint32_t verbose3D = 0) : genStereoSequ(imgSize_, K1_, K2_, R_, t_, pars_, verbose3D) {

    };

    void startCalc() override;
};

#endif //GENERATEVIRTUALSEQUENCE_GENERATEMATCHES_H
