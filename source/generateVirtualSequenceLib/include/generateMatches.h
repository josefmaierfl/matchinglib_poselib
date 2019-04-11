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
    std::pair<double, double> keypErrDistr;//Keypoint error distribution (mean, std) for the matching keypoint location
    std::pair<double, double> imgIntNoise;//Noise (mean, std) on the image intensity for descriptor calculation
//    double lostCorrPor;//Portion (0 to 0.9) of lost correspondences from frame to frame. It corresponds to the portion of backprojected 3D-points that should not be visible in the frame.
    bool storePtClouds;//If true, all PCL point clouds and necessary information to load a cam sequence with correspondences are stored to disk
    bool rwXMLinfo;//If true, the parameters and information are stored and read in XML format. Otherwise it is stored or read in YAML format
    bool compressedWrittenInfo;//If true, the stored information and parameters are compressed (appends .gz) and it is assumed that paramter files to be read are aslo compressed
    bool takeLessFramesIfLessKeyP;//If true and too less images images are provided (resulting in too less keypoints), only as many frames with GT matches are provided as keypoints are available.
    bool parsValid;//Specifies, if the stored values within this struct are valid

    GenMatchSequParameters(std::string mainStorePath_,
                           std::string imgPath_,
                           std::string imgPrePostFix_,
                           std::string keyPointType_,
                           std::string descriptorType_,
                           bool keypPosErrType_ = false,
                           std::pair<double, double> keypErrDistr_ = std::make_pair(0, 0.5),
                           std::pair<double, double> imgIntNoise_ = std::make_pair(0, 5.0),
//                           double lostCorrPor_ = 0,
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
//            lostCorrPor(lostCorrPor_),
            storePtClouds(storePtClouds_),
            rwXMLinfo(rwXMLinfo_),
            compressedWrittenInfo(compressedWrittenInfo_),
            takeLessFramesIfLessKeyP(takeLessFramesIfLessKeyP_),
            parsValid(true){
        keypErrDistr.first = abs(keypErrDistr.first);
        keypErrDistr.second = abs(keypErrDistr.second);
        CV_Assert(keypPosErrType || (!keypPosErrType && (keypErrDistr.first < 5.0) &&
                                     (keypErrDistr.second < 5.0)
                                     && (keypErrDistr.first + 3.0 * keypErrDistr.second < 10.0)));
        imgIntNoise.second = abs(imgIntNoise.second);
        CV_Assert((imgIntNoise.first > -25.0) && (imgIntNoise.first < 25.0) && (imgIntNoise.second < 25.0));
//        CV_Assert((lostCorrPor >= 0) && (lostCorrPor <= 0.9));
        CV_Assert(!imgPath.empty());
    }

    GenMatchSequParameters():
            mainStorePath(""),
            imgPath(""),
            imgPrePostFix(""),
            keyPointType(""),
            descriptorType(""),
            keypPosErrType(false),
            keypErrDistr(std::make_pair(0, 0.5)),
            imgIntNoise(std::make_pair(0, 5.0)),
//            lostCorrPor(lostCorrPor_),
            storePtClouds(false),
            rwXMLinfo(false),
            compressedWrittenInfo(false),
            takeLessFramesIfLessKeyP(false),
            parsValid(false){}
};

struct stats{
    double median;
    double mean;
    double standardDev;
    double minVal;
    double maxVal;
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
                 bool filter_occluded_points_,
                 uint32_t verbose_ = 0) :
            genStereoSequ(imgSize_, K1_, K2_, R_, t_, pars3D_, filter_occluded_points_, verbose_),
            parsMtch(parsMtch_),
            pars3D(pars3D_),
            imgSize(imgSize_),
            K1(K1_),
            K2(K2_),
            sequParsLoaded(false) {
        CV_Assert(!parsMtch.mainStorePath.empty());
        CV_Assert(parsMtch.parsValid);
        genSequenceParsFileName();
        K1i = K1.inv();
        K2i = K2.inv();
        kpErrors.clear();
    };

    genMatchSequ(const std::string &sequLoadFolder,
                 GenMatchSequParameters &parsMtch_,
                 uint32_t verboseMatch_ = 0);

    bool generateMatches();

private:
    //Loads the image names (including folders) of all specified images (used to generate matches) within a given folder
    bool getImageList();
    //Generates a hash value from the parameters used to generate a scene and 3D correspondences, respectively
    size_t hashFromSequPars();
    //Generates a hash value from the parameters used to generate matches from 3D correspondences
    size_t hashFromMtchPars();
    //Calculates the total number of correspondences (Tp+TN) within a whole scene
    void totalNrCorrs();
    //Loads images and extracts keypoints and descriptors from them to generate matches later on
    bool getFeatures();
    //Calculates the descriptor distance between 2 descriptors
    double getDescriptorDistance(const cv::Mat &descriptor1, const cv::Mat &descriptor2);
    //Writes the parameters used to generate 3D scenes to disk
    bool writeSequenceParameters(const std::string &filename);
    //Writes a subset of parameters used to generate a 3D scene to disk (used within an overview file which holds basic information about all sub-folders that contain parameters and different 3D scenes)
    void writeSomeSequenceParameters(cv::FileStorage &fs);
    //Generates a file or appends information to a file which holds an overview with basic information about all sub-folders that contain parameters and different 3D scenes (within a base-path)
    bool writeSequenceOverviewPars();
    //Reads parameters from a generated file that where used to generate a 3D sequence
    bool readSequenceParameters(const std::string &filename);
    //Writes information and correspondences for a single stereo frame to disk
    bool write3DInfoSingleFrame(const std::string &filename);
    //Reads information and correspondences for a single stereo frame from disk
    bool read3DInfoSingleFrame(const std::string &filename);
    //Writes PCL point clouds for the static scene and PCL point clouds for all moving objects to disk
    bool writePointClouds(const std::string &path, const std::string &basename, bool &overwrite);
    //Reads PCL point clouds for the static scene and PCL point clouds for all moving objects from disk
    bool readPointClouds(const std::string &path, const std::string &basename);
    //Generates the filename used to store/load parameters for generating a 3D sequence
    void genSequenceParsFileName();
    //Generates a path for storing results (matches (in sub-folder(s) of generated path) and if desired, the generated 3D scene. The last folder of this path might be a hash value if a dedicated storing-path is provided or corresponds to a given path used to load a 3D scene.
    bool genSequenceParsStorePath();
    //Generates a folder inside the folder of the 3D scene for storing matches
    bool genMatchDataStorePath();
    //Generates a YAML/XML file containing parameters for generating matches from 3D scenes for every sub-folder (For every run with the same 3D scene, the parameter set for the matches is appended at the end of the file)
    bool writeMatchingParameters();
    //Generates a new file inside the folder of the matches which holds the mean and standard deviation of keypoint position errors of the whole scene in addition to a list of images (including their folder structure) that were used to extract patches for calculating descriptors and keypoints
    bool writeKeyPointErrorAndSrcImgs(double &meanErr, double &sdErr);
    //Writes matches and features in addition to other information for every frame to disk
    bool writeMatchesToDisk();
    //Generates the user specified file extension (xml and yaml in combination with gz)
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
    //Checks if the 3D point of the given correspondence was already used before in a different stereo frame to calculate a homography (in this case the same 3D plane is used to calculate a homography) and if not, calculates a new homography
    cv::Mat getHomographyForDistortionChkOld(const cv::Mat& X,
                                             const cv::Mat& x1,
                                             const cv::Mat& x2,
                                             int64_t idx3D,
                                             size_t keyPIdx,
                                             bool visualize);

    //Create a homography for a TN correspondence
    cv::Mat getHomographyForDistortionTN(const cv::Mat& x1,
                                         bool visualize);
    //Visualizes the planes in 3D used to calculate a homography
    void visualizePlanes(std::vector<cv::Mat> &pts3D,
                         const cv::Mat &plane1,
                         const cv::Mat &plane2);
    //Adds gaussian noise to an image patch
    void addImgNoiseGauss(const cv::Mat &patchIn,
            cv::Mat &patchOut,
                          double meanNoise,
                          double stdNoise,
                          bool visualize = false);

    //Adds salt and pepper noise to an image patch
    void addImgNoiseSaltAndPepper(const cv::Mat &patchIn,
            cv::Mat &patchOut,
            int minTH = 30,
            int maxTH = 225,
            bool visualize = false);

    //Generates features and matches for correspondences of a given stereo frame (TN and TP) and stores them to disk
    bool generateCorrespondingFeatures();
    //Generates features and matches based on image patches and calculated homographies for either TN or TP
    void generateCorrespondingFeaturesTPTN(size_t featureIdxBegin,
                                           bool useTN,
                                           std::vector<cv::KeyPoint> &frameKPs1,
                                           std::vector<cv::KeyPoint> &frameKPs2,
                                           cv::Mat &frameDescr1,
                                           cv::Mat &frameDescr2,
                                           std::vector<cv::DMatch> &frameMatches,
                                           std::vector<cv::Mat> &homo,
                                           std::vector<std::pair<size_t,cv::KeyPoint>> &srcImgIdxAndKp);
    //Calculates the size of a patch that should be extracted from the source image to get a minimum square patch size after warping with the given homography based on the shape of the ellipse which emerges after warping a circle with the given keypoint diameter
    bool getRectFitsInEllipse(const cv::Mat &H,
                              const cv::KeyPoint &kp,
                              cv::Rect &patchROIimg1,
                              cv::Rect &patchROIimg2,
                              cv::Rect &patchROIimg21,
                              cv::Point2d &ellipseCenter,
                              double &ellipseRot,
                              cv::Size2d &axes,
                              bool &reflectionX,
                              bool &reflectionY,
                              cv::Size &imgFeatureSi);
    //Calculate statistics on descriptor distances for not matching descriptors
    void calcGoodBadDescriptorTH();
    //Distorts the keypoint position
    void distortKeyPointPosition(cv::KeyPoint &kp2,
                                 const cv::Rect &roi,
                                 std::normal_distribution<double> &distr);
    //Generates keypoints without a position error (Order of correspondences from generating the 3D scene must be the same as for the keypoints)
    void getErrorFreeKeypoints(const std::vector<cv::KeyPoint> &kpWithErr,
                               std::vector<cv::KeyPoint> &kpNoErr);

public:
    GenMatchSequParameters parsMtch;

private:
    const int minPatchSize2 = 85;//Corresponds to the minimal patch size (must be an odd number) we want after warping. It is also used to define the maximum patch size by multiplying it with maxPatchSizeMult2
    const int maxPatchSizeMult2 = 3;//Multiplication factor for minPatchSize2 to define the maximum allowed patch size of the warped image
    const size_t maxImgLoad = 100;//Defines the maximum number of images that are loaded and saved in a vector
    size_t minNrFramesMatch = 10;//Minimum number of required frames that should be generated if there are too less keypoints available
    std::vector<cv::Mat> imgs;//If less than maxImgLoad images are in the specified folder, they are loaded into this vector. Otherwise, this vector holds only images for the current frame
    std::vector<std::string> imageList;//Holds the filenames of all images to extract keypoints
    size_t nrCorrsFullSequ = 0;//Number of predicted overall correspondences (TP+TN) for all frames
    std::vector<cv::KeyPoint> keypoints1;//Keypoints from all used images
    cv::Mat descriptors1;//Descriptors from all used images
    size_t nrFramesGenMatches = 0;//Number of frames used to calculate matches. If a smaller number of keypoints was found than necessary for the full sequence, this number corresponds to the number of frames for which enough features are available. Otherwise, it equals to totalNrFrames.
    size_t hash_Sequ = 0, hash_Matches = 0;//Hash values for the generated 3D sequence and the matches based on their input parameters.
    StereoSequParameters pars3D;//Holds all parameters for calculating a 3D sequence. Is manly used to load existing 3D sequences.
    cv::Size imgSize;//Size of the images
    cv::Mat K1, K1i;//Camera matrix 1 & its inverse
    cv::Mat K2, K2i;//Camera matrix 2 & its inverse
    size_t nrMovObjAllFrames = 0;//Sum over the number of moving objects in every frame
    std::string sequParFileName = "";//File name for storing and loading parameters of 3D sequences
    std::string sequParPath = "";//Path for storing and loading parameters of 3D sequences
    std::string matchDataPath = "";//Path for storing parameters for generating matches
    bool sequParsLoaded = false;//Indicates if the 3D sequence was/will be loaded or generated during execution of the program
    const std::string pclBaseFName = "pclCloud";//Base name for storing PCL point clouds. A specialization is added at the end of this string
    const std::string sequSingleFrameBaseFName = "sequSingleFrameData";//Base name for storing data of generated frames (correspondences)
    const std::string matchSingleFrameBaseFName = "matchSingleFrameData";//Base name for storing data of generated matches
    std::string sequLoadPath = "";//Holds the path for loading a 3D sequence
    std::vector<size_t> featureImgIdx;//Contains an index to the corresponding image for every keypoint and descriptor
    cv::Mat actTransGlobWorld;//Transformation for the actual frame to transform 3D camera coordinates to world coordinates
    cv::Mat actTransGlobWorldit;//Inverse and translated Transformation for the actual frame to transform 3D camera coordinates to world coordinates
    std::map<int64_t,std::pair<cv::Mat,size_t>> planeTo3DIdx;//Holds the plane coefficients and keypoint index for every used keypoint in correspondence to the index of the 3D point in the point cloud
    double actNormT = 0;//Norm of the actual translation vector between the stereo cameras
    std::vector<std::pair<std::map<size_t,size_t>,std::vector<size_t>>> imgFrameIdxMap;//If more than maxImgLoad images to generate features are used, every map contains to most maxImgLoad used images (key = img idx, value = position in the vector holding the images) for keypoints per frame. The vector inside the pair holds a consecutive order of image indices for loading the images
    bool loadImgsEveryFrame = false;//Indicates if there are more than maxImgLoad images in the folder and the images used to extract patches must be loaded for every frame
    stats badDescrTH = {0,0,0,0,0};//Descriptor distance statistics for not matching descriptors. E.g. a descriptor distance larger the median could be considered as not matching descriptors
    std::vector<cv::KeyPoint> frameKeypoints1, frameKeypoints2;//Keypoints for the actual stereo frame (there is no 1:1 correspondence between these 2 as they are shuffled but the keypoint order of each of them is the same as in their corresponding descriptor Mat (rows))
    std::vector<cv::KeyPoint> frameKeypoints2NoErr;//Keypoints in the second stereo image without a positioning error (in general, keypoints in the first stereo image are without errors)
    cv::Mat frameDescriptors1, frameDescriptors2;//Descriptors for the actual stereo frame (there is no 1:1 correspondence between these 2 as they are shuffled but the descriptor order of each of them is the same as in their corresponding keypoint vector). Descriptors corresponding to the same static 3D point (not for moving objects) in different stereo frames are similar
    std::vector<cv::DMatch> frameMatches;//Matches between features of a single stereo frame. They are sorted based on the descriptor distance (smallest first)
    std::vector<bool> frameInliers;//Indicates if a feature (frameKeypoints1 and corresponding frameDescriptors1) is an inlier.
    std::vector<cv::Mat> frameHomographies;//Holds the homographies for all patches arround keypoints for warping the patch which is then used to calculate the matching descriptor. Homographies corresponding to the same static 3D point (not for moving objects) in different stereo frames are similar
    std::vector<std::pair<size_t,cv::KeyPoint>> srcImgPatchIdxAndKp; //Holds the keypoint and image index of the image used to extract patches
    std::vector<int> corrType;//Specifies the type of a correspondence (TN from static (=4) or TN from moving (=5) object, or TP from a new static (=0), a new moving (=1), an old static (=2), or an old moving (=3) object (old means,that the corresponding 3D point emerged before this stereo frame and also has one or more correspondences in a different stereo frame))
    std::vector<double> kpErrors;//Holds distances from the original to the distorted keypoint locations for every correspondence of the whole sequence
    //std::string matchParsFileName = "";//File name (including path) of the parameter file to create matches
    bool overwriteMatchingFiles = false;//If true (after asking the user), data files holding matches (from a different call to this software) in the same folder (should not happen - only if the user manually copies data) are replaced by the new data files
    std::vector<std::pair<double,double>> timePerFrameMatch;//Holds time measurements for every frame in microseconds. The first value corresponds to the time for loading or calculating 3D information of one stereo frame. The second value holds the time for calculating the matches.
    qualityParm timeMatchStats = qualityParm();//Statistics for the execution time in microseconds for calculating matches based on all frames
    qualityParm time3DStats = qualityParm();//Statistics for the execution time in microseconds for calculating 3D correspondences based on all frames
};

#endif //GENERATEVIRTUALSEQUENCE_GENERATEMATCHES_H
