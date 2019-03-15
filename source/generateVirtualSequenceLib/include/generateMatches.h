//
// Created by maierj on 06.03.19.
//

#ifndef GENERATEVIRTUALSEQUENCE_GENERATEMATCHES_H
#define GENERATEVIRTUALSEQUENCE_GENERATEMATCHES_H

#include "generateSequence.h"
#include <opencv2/core/types.hpp>
#include <map>

struct GENERATEVIRTUALSEQUENCELIB_API GenMatchSequParameters {
    std::string mainStorePath;//Path for storing results. If empty and the 3D correspondences are loaded from file, the path for loading these correspondences is also used for storing the matches
    std::string imgPath;//Path containing the images for producing keypoint patches
    std::string imgPrePostFix;//image pre- and/or postfix (supports wildcards & subfolders) for images within imgPath
    std::string keyPointType;//Name of keypoint detector
    std::string descriptorType;//Name of descriptor extractor
    bool keypPosErrType;//Keypoint detector error (true) or error normal distribution (false)
    std::pair<double, double> keypErrDistr;//Keypoint error distribution (mean, std)
    std::pair<double, double> imgIntNoise;//Noise (mean, std) on the image intensity for descriptor calculation
    double lostCorrPor;//Portion (0 to 0.9) of lost correspondences from frame to frame. It corresponds to the portion of backprojected 3D-points that should not be visible in the frame.
    bool storePtClouds;//If true, all PCL point clouds and necessary information to load a cam sequence with correspondences are stored to disk
    bool rwXMLinfo;//If true, the parameters and information are stored and read in XML format. Otherwise it is stored or read in YAML format
    bool compressedWrittenInfo;//If true, the stored information and parameters are compressed (appends .gz) and it is assumed that paramter files to be read are aslo compressed
    bool takeLessFramesIfLessKeyP;//If true and too less images images are provided (resulting in too less keypoints), only as many frames with GT matches are provided as keypoints are available.

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
                           bool rwXMLinfo_ = false,
                           bool compressedWrittenInfo_ = false,
                           bool takeLessFramesIfLessKeyP_ = false) :
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
            rwXMLinfo(rwXMLinfo_),
            compressedWrittenInfo(compressedWrittenInfo_),
            takeLessFramesIfLessKeyP(takeLessFramesIfLessKeyP_) {
        CV_Assert(keypPosErrType || (!keypPosErrType && (keypErrDistr.first > -5.0) && (keypErrDistr.first < 5.0) &&
                                     (keypErrDistr.second > -5.0) && (keypErrDistr.second < 5.0)));
        CV_Assert((imgIntNoise.first > -25.0) && (imgIntNoise.first < 25.0) && (imgIntNoise.second > -25.0) &&
                  (imgIntNoise.second < 25.0));
        CV_Assert((lostCorrPor >= 0) && (lostCorrPor <= 0.9));
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
                 StereoSequParameters pars3D_,
                 GenMatchSequParameters &parsMtch_,
                 uint32_t verbose_ = 0) :
            genStereoSequ(imgSize_, K1_, K2_, R_, t_, pars3D_, verbose_),
            parsMtch(parsMtch_),
            pars3D(pars3D_),
            imgSize(imgSize_),
            K1(K1_),
            K2(K2_),
            sequParsLoaded(false) {
        CV_Assert(!parsMtch.mainStorePath.empty());
        genSequenceParsFileName();
        K1i = K1.inv();
        K2i = K2.inv();
    };

    genMatchSequ(const std::string &sequLoadFolder,
                 GenMatchSequParameters &parsMtch_,
                 uint32_t verboseMatch_ = 0);

    bool generateMatches();

//    void startCalc() override;

private:
    bool getImageList();

    size_t hashFromSequPars();

    size_t hashFromMtchPars();

    void totalNrCorrs();

    bool getFeatures();

    bool checkMatchability();

    bool writeSequenceParameters(const std::string &filename);

    void writeSomeSequenceParameters(cv::FileStorage &fs);

    bool writeSequenceOverviewPars();

    bool readSequenceParameters(const std::string &filename);

    bool write3DInfoSingleFrame(const std::string &filename);

    bool read3DInfoSingleFrame(const std::string &filename);

    bool writePointClouds(const std::string &path, const std::string &basename, bool &overwrite);

    bool readPointClouds(const std::string &path, const std::string &basename);

    void genSequenceParsFileName();

    bool genSequenceParsStorePath();

    bool genMatchDataStorePath();

    bool writeMatchingParameters();

    std::string genSequFileExtension(const std::string &basename);

    //Rotates a line 'b' about a line 'a' (only direction vector) using the given angle
    cv::Mat rotateAboutLine(const cv::Mat &a, const double &angle, const cv::Mat &b);

    //Calculates a homography by rotating a plane in 3D (which was generated using a 3D point and its projections into camera 1 & 2) and backprojection of corresponding points on that plane into the second image
    cv::Mat getHomographyForDistortion(const cv::Mat &X,
                                       const cv::Mat &x1,
                                       const cv::Mat &x2,
                                       int64_t idx3D,
                                       size_t keyPIdx,
                                       cv::InputArray planeNVec,
                                       bool visualize);

    cv::Mat getHomographyForDistortionChkOld(const cv::Mat& X,
                                             const cv::Mat& x1,
                                             const cv::Mat& x2,
                                             int64_t idx3D,
                                             size_t keyPIdx,
                                             bool visualize);

    //Create a homography for a TN correspondence
    cv::Mat getHomographyForDistortionTN(const cv::Mat& x1,
                                         bool visualize);

    void visualizePlanes(std::vector<cv::Mat> &pts3D,
                         const cv::Mat &plane1,
                         const cv::Mat &plane2);

    void addImgNoiseGauss(const cv::Mat &patchIn, cv::Mat &patchOut, bool visualize = false);
    void addImgNoiseSaltAndPepper(const cv::Mat &patchIn,
            cv::Mat &patchOut,
            int minTH = 30,
            int maxTH = 225,
            bool visualize = false);
    void generateCorrespondingFeatures();
    void generateCorrespondingFeaturesTP(size_t featureIdxBegin);
    void generateCorrespondingFeaturesTN(size_t featureIdxBegin);
    bool getRectFitsInEllipse(const cv::Mat &H,
                              const cv::KeyPoint &kp,
                              cv::Rect &patchROIimg1,
                              cv::Rect &patchROIimg2,
                              cv::Point2d &ellipseCenter,
                              double &ellipseRot,
                              cv::Size2d &axes);

public:
    GenMatchSequParameters parsMtch;

private:
    const int minPatchSize2 = 65;//Corresponds to the minimal patch size (must be an odd number) we want after warping. It is also used to define the maximum patch size by multiplying it with maxPatchSizeMult2
    const int maxPatchSizeMult2 = 3;//Multiplication factor for minPatchSize2 to define the maximum allowed patch size of the warped image
    const size_t maxImgLoad = 100;//Defines the maximum number of images that are loaded and saved in a vector
    size_t minNrFramesMatch = 10;//Minimum number of required frames that should be generated if there are too less keypoints available
    std::vector<cv::Mat> imgs;//If less than maxImgLoad images are in the specified folder, they are loaded into this vector. Otherwise, this vector holds only images for the current frame
    std::vector<std::string> imageList;//Holds the filenames of all images to extract keypoints
    size_t nrCorrsFullSequ;//Number of predicted overall correspondences (TP+TN) for all frames
    std::vector<cv::KeyPoint> keypoints1;//Keypoints from all used images
    cv::Mat descriptors1;//Descriptors from all used images
    size_t nrFramesGenMatches;//Number of frames used to calculate matches. If a smaller number of keypoints was found than necessary for the full sequence, this number corresponds to the number of frames for which enough features are available. Otherwise, it equals to totalNrFrames.
    size_t hash_Sequ, hash_Matches;//Hash values for the generated 3D sequence and the matches based on their input parameters.
    StereoSequParameters pars3D;//Holds all parameters for calculating a 3D sequence. Is manly used to load existing 3D sequences.
    cv::Size imgSize;//Size of the images
    cv::Mat K1, K1i;//Camera matrix 1 & its inverse
    cv::Mat K2, K2i;//Camera matrix 2 & its inverse
    size_t nrMovObjAllFrames;//Sum over the number of moving objects in every frame
    std::string sequParFileName;//File name for storing and loading parameters of 3D sequences
    std::string sequParPath;//Path for storing and loading parameters of 3D sequences
    std::string matchDataPath;//Path for storing parameters for generating matches
    bool sequParsLoaded = false;//Indicates if the 3D sequence was/will be loaded or generated during execution of the program
    const std::string pclBaseFName = "pclCloud";//Base name for storing PCL point clouds. A specialization is added at the end of this string
    const std::string sequSingleFrameBaseFName = "sequSingleFrameData";//Base name for storing data of generated frames (correspondences)
    std::string sequLoadPath = "";//Holds the path for loading a 3D sequence
    std::vector<size_t> featureImgIdx;//Contains an index to the corresponding image for every keypoint and descriptor
    cv::Mat actTransGlobWorld;//Transformation for the actual frame to transform 3D camera coordinates to world coordinates
    cv::Mat actTransGlobWorldit;//Inverse and translated Transformation for the actual frame to transform 3D camera coordinates to world coordinates
    std::map<int64_t,std::pair<cv::Mat,size_t>> planeTo3DIdx;//Holds the plane coefficients and keypoint index for every used keypoint in correspondence to the index of the 3D point in the point cloud
    double actNormT;//Norm of the actual translation vector between the stereo cameras
    std::vector<std::pair<std::map<size_t,size_t>,std::vector<size_t>>> imgFrameIdxMap;//If more than maxImgLoad images to generate features are used, every map contains to most maxImgLoad used images (key = img idx, value = position in the vector holding the images) for keypoints per frame. The vector inside the pair holds a consecutive order of image indices for loading the images
    bool loadImgsEveryFrame = false;//Indicates if there are more than maxImgLoad images in the folder and the images used to extract patches must be loaded for every frame
};

#endif //GENERATEVIRTUALSEQUENCE_GENERATEMATCHES_H
