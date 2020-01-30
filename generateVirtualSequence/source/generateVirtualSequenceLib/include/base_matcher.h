///**********************************************************************************************************
// FILE: test_GMbSOF.h
//
// PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2
//
// CODE: C++
// 
// AUTOR: Josef Maier, AIT Austrian Institute of Technology
//
// DATE: September 2015
//
// LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna
//
// VERSION: 1.0
//
// DISCRIPTION: This file provides functionalities for testing different matching algorithms. The class
// baseMatcher provides all functionalities necessary for before and after matching, like feature and 
// descriptor extraction, quality measurement on the final matches as well as refinement of the found
// matches. The matching algorithms themself must be implemented as a child class of this base class.
//**********************************************************************************************************/
//
//#pragma once
//
//#include "glob_includes.h"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/features2d/features2d.hpp"
//
///* --------------------------- Defines --------------------------- */
//
//#define INITMATCHQUALEVAL_O 0 //If set to 1, the GMbSOF algorithm is aborted after generation of the SOF and the initial and filtered matches are given back for analysis (ensure that the method AdvancedMatching is configured the right way for this measurement (The define INITMATCHQUALEVAL in match_statOptFlow.h must also be 1.))
//#define COSTDISTRATIOEVAL 0 //If set to 1, the ratios for the discriptor distances and spatial distances (from found keypoint to estimated position) to their local medians can be eveluated
//
//#define INITMATCHDISTANCETH_GT 10 //Initial threshold within a match is considered as true match (true value is estimated in filterInitFeaturesGT())
//#define TIMEMEASITERS 100 //Iterations performed for time measurement
//
//#define CORRECT_MATCHING_RESULT 0 //Only used for correcting the final results of GT testing due to a bug in the code(for the first run on the dataset use 1 and for the second 2). 
//								  //For coorrecting (due to a sign failure) the automatically annotated matches 3 is used. Otherwise this should be 0.
//
////Quality parameters
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
//
////Parameters of every image pair from the annotation tool for testing the GT
//typedef struct annotImgPars{
//	std::vector<std::vector<std::pair<cv::Point2f,cv::Point2f>>> falseGT;//false GT matching coordinates within the GT matches dataset
//	std::vector<std::vector<std::pair<double,int>>> distanceHisto;//Histogram of the distances from the matching positions to annotated positions
//	std::vector<std::vector<double>> distances;//distances from the matching positions to annotated positions
//	std::vector<int> notMatchable;//Number of matches from the GT that are not matchable in reality
//	std::vector<int> truePosArr[5];//Number of true positives after matching of the annotated matches
//	std::vector<int> falsePosArr[5];//Number of false positives after matching of the annotated matches
//	std::vector<int> falseNegArr[5];//Number of false negatives after matching of the annotated matches
//	std::vector<std::vector<cv::Point2f>> errvecs;//vector from the matching positions to annotated positions
//	std::vector<std::vector<std::pair<cv::Point2f,cv::Point2f>>> perfectMatches;//The resulting annotated matches
//	std::vector<cv::Mat> HE;//The fundamental matrix or homography calculted from the annotated matches
//	std::vector<std::vector<int>> validityValFalseGT;//Validity level of false matches for the filled KITTI GT (1=original GT, 2= filled GT)
//	std::vector<std::vector<cv::Point2f>> errvecsGT;//Vectors from the dataset GT to the annotated positions
//	std::vector<std::vector<double>> distancesGT;//Distances from the dataset GT to the annotated positions
//	std::vector<std::vector<int>> validityValGT;//Validity level of all annoted matches for the filled KITTI GT (1=original GT, 2= filled GT)
//	std::vector<std::vector<double>> distancesEstModel;//Distances of the annotated positions to an estimated model (HE) from the annotated positions
//	std::vector<int> selectedSamples;//Number of samples per image pair
//	std::vector<int> nrTotalFails;//Number of total fails from the first annoptated image pair until the actual image pair
//	std::vector<std::string> id;//The ID of each image pair in the form "left-image_right-image" or "first-image_second-image"
//	std::vector<std::vector<char>> autoManualAnnot;//Holds a 'M' for a manual and an 'A' for an automatic annotated match
//} annotImgPars;
//
///* --------------------------- Classes --------------------------- */
//
//class baseMatcher {
//public:
//	//VARIABLES --------------------------------------------
//
//	//Variables for storing time measurements
//	double tf;//Time needed for feature extraction
//	double td;//Time needed for descriptor extraction
//	double tm;//Time needed for matching
//	double to;//Overall runtime to=tf+td+tm
//	double tkm;//Average matching time per keypoint (number of left keypoints) tkm=tm/(positivesGT+negativesGTl)
//	double tko;//Average overall runtime per keypoint (number of left keypoints) tkm=tm/(positivesGT+negativesGTl)
//	double tr;//Runtime for refinement with VFC
//	double tmeanD;//Mean runtime of the descriptor extractor per keypoint
//
//	//Variables for performing time measurements
//	int cyklesTM; //Number of cycles for which a operation is repeated and the minimum time is extracted (feature & descriptor extraction, matching)
//	bool measureT; //Enables time measurement (if true) over cyklesTM computations
//	bool measureTd; //Enables time measurement (if true) for descriptor runtime evaluations
//
//	//Quality parameters after feature extraction
//	double specifiedInlRatio;//User input for specifying a desirable inlier ratio for both images
//	double inlRatioL;//Inlier ratio in left image
//	double inlRatioR;//Inlier ratio in right image
//	double inlRatioO;//Inlier ratio over all keypoints in both images
//	double positivesGT;//Number of inliers (from ground truth)
//	double negativesGTl;//Number of outliers in left image (from ground truth)
//	double negativesGTr;//Number of outliers in right image (from ground truth)
//
//	//Quality parameters
//	matchQualParams qpm;//Quality parameters from matching
//	matchQualParams qpr;//Quality parameters from refinement with VFC (if INITMATCHQUALEVAL_O = 1, this variable is used to store the results after fitering the init matches with the SOF -> only for GMbSOF)
//	std::vector<bool> falseNegMatches;//Specifies which true match wasnt found by the matching algorithm
//	std::vector<bool> falsePosMatches;//Specifies which by the matching algorithm found match is incorrect
//	std::vector<bool> truePosMatches;//Specifies which by the matching algorithm found match is correct
//	std::vector<bool> falseNegMatchesRef;//Specifies which true match wasnt found by the matching algorithm
//	std::vector<bool> falsePosMatchesRef;//Specifies which by the matching algorithm found match is incorrect
//	std::vector<bool> truePosMatchesRef;//Specifies which by the matching algorithm found match is correct
//
//	//Images
//	cv::Mat imgs[2];//First and second image (must be 8bit in depth)
//	std::string imgsPath;//Path to the folder containing the images to write the GTmatches into a subfolder
//	std::string fileNameImgL;//File name of the first (or left) image or any other string that is used to generate the filename of the ground truth matches
//
//	//Ground truth
//	std::vector<bool> leftInlier;//Specifies if an detected feature of the left image has a true match in the right image (true). Corresponds to the index of keypL
//	std::vector<cv::DMatch> matchesGT;//Ground truth matches (ordered increasing query index)
//	cv::Mat flowGT;//Ground truth flow file (if available, otherwise the ground truth homography should be used -> different constructors)
//	cv::Mat homoGT;//Ground truth homography (if available, otherwise the ground truth flow should be used -> different constructors)
//	bool flowGtIsUsed;//Specifies if the flow or the homography is used as ground truth
//	double usedMatchTH;//Threshold within a match is considered as true match
//	bool useSameKeypSiVarInl;//If true, the maximum number of keypoints for all inlier ratios corresponds to the number of true positives at an inlier ratio of 1.0
//	std::string GTfilterExtractor;//Specifies the descriptor type like FREAK, SIFT, ... for filtering ground truth matches. DEFAULT=FREAK
//
//	//Features
//	std::vector<cv::KeyPoint> keypL;//Left keypoints
//	std::vector<cv::KeyPoint> keypR;//Right keypoints
//	std::string featuretype;//Feature type like FAST, SIFT, ... that are defined in the OpenCV (FeatureDetector::create)
//
//	//Descriptors
//	cv::Mat descriptorsL;//Left descriptors
//	cv::Mat descriptorsR;//Right descriptors
//	std::string descriptortype;//Descriptor type like SIFT, FREAK, ... that are defined in the OpenCV (DescriptorExtractor::create)
//
//	//Matches
//	std::vector<cv::DMatch> matches;//Matches calculated from the specific matching algorithm
//	std::vector<cv::DMatch> matchesRefined;//Refined matches after refinement with VFC (if INITMATCHQUALEVAL_O = 1, this variable is used to store the Initial matches after filtering with the SOF -> only for GMbSOF)
//#if INITMATCHQUALEVAL_O
//	double initEstiInlRatio; //Estimated inlier ratio from the GMbSOF algorithm after initial matching
//#endif
//#if COSTDISTRATIOEVAL
//	std::vector<float> costRatios;
//	std::vector<float> distRatios;
//	std::vector<bool> tpfp;//True if the match is a true positive, false if its a false positive
//#endif
//
//	//Variable to check the code configuration
//	bool specialGMbSOFtest;
//
//	bool generateOnlyInitialGTM;
//
//	//FUNCTION PROTOTYPES ----------------------------------------
//
//	//Constructor
//	baseMatcher(cv::Mat leftImg, cv::Mat rightImg, std::string _featuretype, std::string _descriptortype, cv::Mat flowOrHomoGT, bool _flowGtIsUsed, std::string _imgsPath, std::string _fileNameImgL);
//
//	//Refine the calculated matches using Vector Field Consensus (VFC)
//	int refineMatches();
//	//Run the whole feature extraction and matching process
//	int performMatching(double UsrInlRatio = 0, bool _measureT = false, unsigned int repEvals = 20);
//	//Show final matches with or without refinement and optionally store the output to disk
//	int showMatches(int drawflags = 1, bool refinedMatches = false, std::string path = "", std::string file = "", bool storeOnly = false);
//
//	//Initial detection of all features without filtering
//	int detectFeatures();
//
//	int checkForGT();
//
//	//void setTestGTMode();
//
//	int testGTmatches(int & samples, std::vector<std::pair<cv::Point2f,cv::Point2f>> & falseGT, int & usedSamples, 
//					  std::vector<std::pair<double,int>> & distanceHisto, std::vector<double> & distances, int remainingImgs, int & notMatchable,
//					  int *truePosArr, int *falsePosArr, int *falseNegArr, //Arrays truePosArr, falsePosArr, and falseNegArr must be of size 4 and initialized
//					  std::vector<cv::Point2f> & errvecs, std::vector<std::pair<cv::Point2f,cv::Point2f>> & perfectMatches, cv::Mat & HE, std::vector<int> & validityValFalseGT,
//					  std::vector<cv::Point2f> & errvecsGT, std::vector<double> & distancesGT, std::vector<int> & validityValGT, std::vector<double> & distancesEstModel,
//					  annotImgPars & annotationData, std::vector<char> & autoManualAnno, double threshhTh = 64.0, int imgNr = 0, int *fullN = NULL, int *fullSamples = NULL, int *fullFails = NULL);
//	int helpOldCodeBug(std::vector<std::pair<int,double>> wrongGTidxDist, std::vector<int> used_matches, 
//					   int *truePosArr, int *falsePosArr, int *falseNegArr);
//
//private:
//	std::string gtsubfolder;
//	bool noGTgenBefore;
//	std::string filenameGT_initial, imgsPath_initial, fileNameImgL_initial;
//	//Detect features
//	int getValidFeaturesDescriptors();
//	
//	//Filter initial features with a ground truth flow file if available and calculate ground truth quality parameters
//	int filterInitFeaturesGT();
//	//Matching of the found keypoints by different algorithms (the functionality must be implemented in a child class)
//	virtual int matchValidKeypoints() = 0;
//	//Eveluate the quality parameters of the matched keypoints
//	int evalMatches(bool refinedMatches);
//
//	bool readGTMatchesDisk(std::string filenameGT);
//
//	int writeGTMatchesDisk(std::string filenameGT, bool writeEmptyFile = false);
//
//	void clearGTvars();
//
//	void clearMatchingResult();
//
//	void prepareFileNameGT(std::string& filenameGT, std::string& imgsPath_tmp, std::string& fileNameImgL_tmp, bool noInlRatFilter = false);
//
//	//bool testGT;
//};
//
///* --------------------- Function prototypes --------------------- */
//
////Estimates the minimum sample size for a given population
//void getMinSampleSize(int N, double p, double & e, double & minSampleSize);
