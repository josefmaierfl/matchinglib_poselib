/**********************************************************************************************************
 FILE: test_GMbSOF.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: September 2015

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for testing different matching algorithms. The class
 baseMatcher provides all functionalities necessary for before and after matching, like feature and 
 descriptor extraction, quality measurement on the final matches as well as refinement of the found
 matches. The matching algorithms themself must be implemented as a child class of this base class.
**********************************************************************************************************/

#pragma once

#include "glob_includes.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <tuple>
#include <utility>
#include "GTM/prepareMegaDepth.h"
//#include "generateVirtualSequenceLib/generateVirtualSequenceLib_api.h"

/* --------------------------- Defines --------------------------- */

#define INITMATCHDISTANCETH_GT 10 //Initial threshold within a match is considered as true match (true value is estimated in filterInitFeaturesGT())

//Quality parameters
//typedef struct matchQualParams{
//	unsigned int trueNeg;//True negatives
//	unsigned int truePos;//True positives
//	unsigned int falseNeg;//False negatives
//	unsigned int falsePos;//False positives
//	double ppv;//Precision or positive predictive value ppv=truePos/(truePos+falsePos)
//	double tpr;//Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)
//	double fpr;//Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)
//	double acc;//Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)
//} matchQualParams;

//Parameters of every image pair from the annotation tool for testing the GT
struct annotImgPars{
	std::vector<std::pair<cv::Point2f,cv::Point2f>> falseGT;//false GT matching coordinates within the GT matches dataset
	std::vector<std::pair<double,int>> distanceHisto;//Histogram of the distances from the matching positions to annotated positions
	std::vector<double> distances;//distances from the matching positions to annotated positions
	int notMatchable;//Number of matches from the GT that are not matchable in reality
//	int truePosArr[5];//Number of true positives after matching of the annotated matches
//	int falsePosArr[5];//Number of false positives after matching of the annotated matches
//	int falseNegArr[5];//Number of false negatives after matching of the annotated matches
	std::vector<cv::Point2f> errvecs;//vector from the matching positions to annotated positions
	std::vector<std::pair<cv::Point2f,cv::Point2f>> perfectMatches;//The resulting annotated matches
    std::vector<int> matchesGT_idx;//Indices for perfectMatches pointing to corresponding match in matchesGT
	cv::Mat HE;//The fundamental matrix or homography calculted from the annotated matches
	std::vector<int> validityValFalseGT;//Validity level of false matches for the filled KITTI GT (1=original GT, 2= filled GT)
	std::vector<cv::Point2f> errvecsGT;//Vectors from the dataset GT to the annotated positions
	std::vector<double> distancesGT;//Distances from the dataset GT to the annotated positions
	std::vector<int> validityValGT;//Validity level of all annoted matches for the filled KITTI GT (1=original GT, 2= filled GT, -1= not annotated)
	std::vector<double> distancesEstModel;//Distances of the annotated positions to an estimated model (HE) from the annotated positions
//	int> selectedSamples;//Number of samples per image pair
//	int> nrTotalFails;//Number of total fails from the first annotated image pair until the actual image pair
	std::string id;//The ID of the image pair in the form "datasetName-datasetPart-leftImageName-rightImageName" or "firstImageName-secondImageName"
	std::vector<char> autoManualAnnot;//Holds a 'M' for a manual, an 'A' for an automatic annotated match, and 'U' for matches without refinement

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

struct GTMdata{
    size_t sum_TP = 0;//Number of GT TP loaded/computed so far over all datasets
    size_t sum_TN = 0;//Number of GT TN loaded/computed so far over all datasets
    size_t sum_TP_Oxford = 0;//Number of GT TP loaded/computed so far over the Oxford dataset
    size_t sum_TN_Oxford = 0;//Number of GT TN loaded/computed so far over the Oxford dataset
    size_t sum_TP_KITTI = 0;//Number of GT TP loaded/computed so far over the KITTI dataset
    size_t sum_TN_KITTI = 0;//Number of GT TN loaded/computed so far over the KITTI dataset
    size_t sum_TP_MegaDepth = 0;//Number of GT TP loaded/computed so far over the MegaDepth dataset
    size_t sum_TN_MegaDepth = 0;//Number of GT TN loaded/computed so far over the MegaDepth dataset
    std::vector<std::vector<cv::KeyPoint>> keypLAll;//Left GT keypoints loaded/computed so far over the whole dataset
    std::vector<std::vector<cv::KeyPoint>> keypRAll;//Right GT keypoints loaded/computed so far over the whole dataset
    std::vector<std::vector<bool>> leftInlierAll;// Left inlier/outlier mask loaded/computed so far over the whole dataset
    std::vector<std::vector<bool>> rightInlierAll;//Right inlier/outlier mask loaded/computed so far over the whole dataset
    std::vector<std::vector<cv::DMatch>> matchesGTAll;//Ground truth matches loaded/computed so far over the whole dataset
    std::vector<std::vector<std::pair<std::string, std::string>>> imgNamesAll;//Image names including paths of first and second source images corresponding to indices of keypLAll, keypRAll, leftInlierAll, rightInlierAll, matchesGTAll
    std::vector<char> sourceGT;//Indicates from which dataset GTM were calculated (O ... Oxford, K ... KITTI, M ... MegaDepth)

    GTMdata(){
        sum_TP = 0;
        sum_TN = 0;
        sum_TP_Oxford = 0;
        sum_TN_Oxford = 0;
        sum_TP_KITTI = 0;
        sum_TN_KITTI = 0;
        sum_TP_MegaDepth = 0;
        sum_TN_MegaDepth = 0;
    }

    void clear(){
        sum_TP = 0;
        sum_TN = 0;
        sum_TP_Oxford = 0;
        sum_TN_Oxford = 0;
        sum_TP_KITTI = 0;
        sum_TN_KITTI = 0;
        sum_TP_MegaDepth = 0;
        sum_TN_MegaDepth = 0;
        keypLAll.clear();
        keypRAll.clear();
        leftInlierAll.clear();
        rightInlierAll.clear();
        matchesGTAll.clear();
        imgNamesAll.clear();
    }
};

struct kittiFolders{
    struct folderPostF{
        std::string sub_folder;
        std::string postfix;

        folderPostF(std::string sub_folder_, std::string postfix_):sub_folder(std::move(sub_folder_)), postfix(std::move(postfix_)){}
        folderPostF(std::string &&sub_folder_, std::string &&postfix_):sub_folder(move(sub_folder_)), postfix(move(postfix_)){}
        folderPostF() = default;
        folderPostF(folderPostF &&kf) noexcept :sub_folder(move(kf.sub_folder)), postfix(move(kf.postfix)){}
        folderPostF(std::initializer_list<std::string> kf):sub_folder(*kf.begin()), postfix(*(kf.begin() + 1)){}
        folderPostF(const folderPostF &kf) = default;
        folderPostF& operator=(const folderPostF & kf){
            sub_folder = kf.sub_folder;
            postfix = kf.postfix;
        }
    };
    folderPostF img1;
    folderPostF img2;
    folderPostF gt12;
    bool isFlow;

    kittiFolders(){
        isFlow = false;
    }

    kittiFolders(const kittiFolders &kf){
        img1 = kf.img1;
        img2 = kf.img2;
        gt12 = kf.gt12;
        isFlow = kf.isFlow;
    }

    kittiFolders(const std::string &sf1, const std::string &pf1,
                 const std::string &sf2, const std::string &pf2,
                 const std::string &sf3, const std::string &pf3,
                 bool if1){
        img1.sub_folder = sf1;
        img1.postfix = pf1;
        img2.sub_folder = sf2;
        img2.postfix = pf2;
        gt12.sub_folder = sf3;
        gt12.postfix = pf3;
        isFlow = if1;
    }

    kittiFolders(std::initializer_list<std::string> const img1_,
                 std::initializer_list<std::string> const img2_,
                 std::initializer_list<std::string> const gt12_,
                 bool if1):img1(img1_), img2(img2_), gt12(gt12_), isFlow(if1){}
//                 {
//        std::copy(img1_.begin(), img1_.end(), img1);
//        std::copy(img2_.begin(), img2_.end(), img2);
//        std::copy(gt12_.begin(), gt12_.end(), gt12);
//        isFlow = if1;
//    }

//    kittiFolders(std::initializer_list<std::string> &&img1_,
//                 std::initializer_list<std::string> &&img2_,
//                 std::initializer_list<std::string> &&gt12_,
//                 bool &&if1):img1(img1_), img2(img2_), gt12(gt12_), isFlow(if1){}
};

/* --------------------------- Classes --------------------------- */

class baseMatcher {
public:
	//VARIABLES --------------------------------------------
    GTMdata gtmdata;//Holds GTM over all loaded/calculated datasets

	//FUNCTION PROTOTYPES ----------------------------------------

	//Constructor
    baseMatcher(std::string _featuretype, std::string _imgsPath, std::string _descriptortype, bool refineGTM_ = true);

    //Prepare GTM from Oxford dataset
    bool calcGTM_Oxford(size_t &min_nrTP);
    //Prepare GTM from KITTI dataset
    bool calcGTM_KITTI(size_t &min_nrTP);
    //Prepare GTM from MegaDepth dataset
    bool calcGTM_MegaDepth(size_t &min_nrTP);

private:
    const std::string base_url_oxford = "http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/";
    const std::string gtm_sub_folder = "GTM";
    const std::string gtm_ending = ".yaml.gz";
    const struct megaDepthFStruct{
        std::string mainFolder = "MegaDepth";//MegaDepth
        std::string depthImgSubF = "MegaDepth_v1";//MegaDepth_v1 -> followed by numbered (zero padded) folder
        std::string depthImgPart = "dense";//dense* -> * corresponds to number
        std::string depthSubF = "depths";//depths
        std::string depthExt = ".h5";//*.h5
        std::string imgSubF = "imgs";//imgs
        std::string sfmSubF = "SfM";//SfM -> followed by numbered (zero padded) folder
        std::string sfmSubSub = "sparse/manhattan";//sparse/manhattan -> followed by numbered folder corresponding to number in depthImgPart
        std::string sfmImgF = "images";
        std::string flowSub = "flow";//flow -> in depthImgPart
    } mdFolders;
    bool refineGTM = true;
    annotImgPars quality;
    bool refinedGTMAvailable = false;

    //Quality parameters after feature extraction
    double inlRatio, inlRatio_refined;//Inlier ratio over all keypoints in both images
    size_t positivesGT, positivesGT_refined;//Number of inliers (from ground truth)
    size_t negativesGT, negativesGT_refined;//Number of outliers in left image (from ground truth)

    //Images
    cv::Mat imgs[2];//First and second image (must be 8bit in depth)
    std::string imgsPath;//Main path containing images and subfolders with 3rd party datasets

    //Ground truth
    std::vector<bool> leftInlier, leftInlier_refined;//Specifies if an detected feature of the left image has a true match in the right image (true). Corresponds to the index of keypL
    std::vector<bool> rightInlier, rightInlier_refined;//Specifies if an detected feature of the right image has a true match in the left image (true). Corresponds to the index of keypR
    std::vector<cv::DMatch> matchesGT, matchesGT_refined;//Ground truth matches (ordered increasing query index)
    cv::Mat flowGT;//Ground truth flow file (if available, otherwise the ground truth homography should be used -> different constructors)
    cv::Mat homoGT;//Ground truth homography (if available, otherwise the ground truth flow should be used -> different constructors)
    bool flowGtIsUsed;//Specifies if the flow or the homography is used as ground truth
    double usedMatchTH;//Threshold within a match is considered as true match
    std::string GTfilterExtractor;//Specifies the descriptor type like FREAK, SIFT, ... for filtering ground truth matches. DEFAULT=FREAK

    //Features
    std::vector<cv::KeyPoint> keypL, keypL_refined;//Left keypoints
    std::vector<cv::KeyPoint> keypR, keypR_refined;//Right keypoints
    std::string featuretype;//Feature type like FAST, SIFT, ... that are defined in the OpenCV (FeatureDetector::create)

    //Check if GT data is available for the Oxford dataset. If not, download it
    bool getOxfordDatasets(const std::string &path);
    //Check if GT data is available for an Oxford sub-set. If not, download it
    bool getOxfordDataset(const std::string &path, const std::string &datasetName);
    //Check if an Oxford dataset folder is complete
    static bool checkOxfordSubDataset(const std::string &path);
    //Loads Oxford images and calculates GTM using GT homographies
    bool calculateGTM_Oxford(const std::pair<std::string, std::string> &imageNames, const cv::Mat &H);
    //Refines keypoint positions of found GTM. Optionally, manual annotation can be performed
    bool refineFoundGTM(int remainingImgs);
    //Filters refined GTM positions based on a new estimated model (H or F) and stores refined GTM into matchesGT_refined, keypR_refined, and keypL_refined
    void getRefinedGTM();
    //Stores refined GTM into GTM variables
    void switchToRefinedGTM();
    //Calculates, refines, and stores GTM of an Oxford image pair
    bool calcRefineStoreGTM_Oxford(const std::pair<std::string, std::string> &imageNames, const cv::Mat &H,
                                   const std::string &gtm_path, const std::string &sub,
                                   const int &remainingImgs, bool save_it = true);
    //Check if GTM are available on disk and load them
    bool loadGTM(const std::string &gtm_path, const std::pair<std::string, std::string> &imageNames);
    //Extract image names from GTM file name given a list of image file names of the folder containing the images
    bool getImgNamesFromGTM(const std::string &file, const std::vector<std::string> &imgNames,
                            std::string &imgName1, std::string &imgName2);
    //Adds GTM from a single image pair to gtmdata (gtmdata.imgNamesAll is not added)
    void addGTMdataToPool();
    //Checks if GTM for the Oxford dataset are available and if not calculates them
    bool getOxfordGTM(const std::string &path, size_t &min_nrTP);
    //Load image names of corresponding images and their homographies
    static bool loadOxfordImagesHomographies(const std::string &path,
                                             std::vector<std::pair<std::string, std::string>> &imgNames,
                                             std::vector<cv::Mat> &homographies);
    //Load corresponding image and GT file names for a KITTI sub-dataset
    static bool loadKittiImageGtFnames(const std::string &mainPath, kittiFolders &info,
                                std::vector<std::tuple<std::string, std::string, std::string>> &fileNames);
    //Loads KITTI GT, interpolates flow/disparity, and calculates GTM
    bool getKitti_MD_GTM(const std::string &img1f, const std::string &img2f, const std::string &gt, bool is_flow,
                         cv::InputArray flow = cv::noArray());
    //Calculates, refines, and stores GTM of an KITTI image pair
    bool calcRefineStoreGTM_KITTI_MD(const std::tuple<std::string, std::string, std::string> &fileNames,
                                     bool is_flow, const std::string &gtm_path, const std::string &sub,
                                     const int &remainingImgs, bool save_it, const std::string &gt_type,
                                     cv::InputArray flow = cv::noArray());
    //Holds sub-directory names for the Oxford dataset
    static std::vector<std::string> GetOxfordSubDirs();

    static std::vector<kittiFolders> GetKITTISubDirs();

    std::vector<megaDepthFolders> GetMegaDepthSubDirs(const std::string &path);

    //Initial detection of all features without filtering
    bool detectFeatures();

    bool testGTmatches(int & samples, std::vector<std::pair<cv::Point2f,cv::Point2f>> & falseGT, int & usedSamples,
                       std::vector<std::pair<double,int>> & distanceHisto, std::vector<double> & distances, int remainingImgs, int & notMatchable,
                       std::vector<cv::Point2f> & errvecs, std::vector<std::pair<cv::Point2f,cv::Point2f>> & perfectMatches,
                       std::vector<int> &matchesGT_idx, cv::Mat & HE, std::vector<int> & validityValFalseGT,
                       std::vector<cv::Point2f> & errvecsGT, std::vector<double> & distancesGT, std::vector<int> & validityValGT, std::vector<double> & distancesEstModel,
                       std::vector<char> & autoManualAnno, const std::string &featureType = "SIFT", double threshhTh = 64.0, const int *fullN = nullptr, const int *fullSamples = nullptr, const int *fullFails = nullptr);
	
	//Filter initial features with a ground truth flow file if available and calculate ground truth quality parameters
	int filterInitFeaturesGT();

	bool readGTMatchesDisk(const std::string &filenameGT);

	bool writeGTMatchesDisk(const std::string &filenameGT, bool writeQualityPars = false);

	void clearGTvars();

	std::string prepareFileNameGT(const std::pair<std::string, std::string> &filenamesImg, const std::string &GTM_path);
    std::string getGTMbasename() const;
    static std::string concatImgNames(const std::pair<std::string, std::string> &filenamesImg);
};

/* --------------------- Function prototypes --------------------- */

//Estimates the minimum sample size for a given population
void getMinSampleSize(int N, double p, double & e, double & minSampleSize);
