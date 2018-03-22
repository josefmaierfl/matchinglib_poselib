///**********************************************************************************************************
// FILE: eval_start.cpp
//
// PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2
//
// CODE: C++
// 
// AUTOR: Josef Maier, AIT Austrian Institute of Technology
//
// DATE: October 2015
//
// LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna
//
// VERSION: 1.0
//
// DISCRIPTION: This file provides functionalities for 
//**********************************************************************************************************/
//#define NOMINMAX
//
//#include "eval_start.h"
//
//#include <direct.h>
//#include <fstream>
//#include <iomanip>
//
//#include "opencv2\calib3d\calib3d.hpp"
//#include <opencv2\imgproc\imgproc.hpp>
//
//#include "base_matcher.h"
//
//#include "io_data.h"
//
//using namespace std;
//
///* --------------------------- Defines --------------------------- */
//
///*
// * medErr ...	median of the reprojection errors masked as inliers
// * arithErr ... arithmetic mean value of the reprojection errors masked as inliers
// * arithStd	... standard deviation of the reprojection errors masked as inliers
// * medStd ... standard deviation of the reprojection errors masked as inliers using the median instead of the mean value
// * lowerQuart ... lower quartile
// * upperQuart ... upper quartile
//*/
//typedef struct qualityParm1 {
//		double medErr, arithErr, arithStd, medStd, lowerQuart, upperQuart;
//} qualityParm1;
//
//
///* --------------------- Function prototypes --------------------- */
//void getStatisticfromVec2(const std::vector<double> vals, qualityParm1 *stats, bool rejQuartiles);
//bool dirExists(const std::string& dirName_in);
//bool file_exists(const std::string& name);
//void getSmoothComulDist(std::vector<double> precision, std::vector<std::pair<double,double>> & cumuldistPrecision, 
//						std::vector<std::pair<double,double>> & cumuldistPrecisionBins, double distrval, double bindist);
//void writeAddDataToFile(string pathFile, string matcherType, string description, qualityParm1 *param, std::vector<double> vals, bool boxplot);
//void normalizeCumulDistBins(std::vector<std::vector<double>> & io3Dvec = std::vector<std::vector<double>>(), 
//							std::vector<double> & io2Dvec = std::vector<double>(),
//							int nrinlrats = 0, std::vector<double> *outBinHeights = NULL, double *vec2DmaxBinVal = NULL, double *takevec2DmaxBinVal = NULL);
//void calcMultCumulDistCD(std::vector<std::vector<float>> costRatios,
//						 std::vector<std::vector<float>> distRatios,
//						 std::vector<std::vector<bool>> tpfps,
//						 std::vector<std::vector<double>> & costCumulDistrTP,
//						 std::vector<std::vector<double>> & distCumulDistrTP,
//						 std::vector<std::vector<double>> & costCumulDistrFP,
//						 std::vector<std::vector<double>> & distCumulDistrFP,
//						 std::vector<double> & costCumulDallInlTP,
//						 std::vector<double> & distCumulDallInlTP,
//						 std::vector<double> & costCumulDallInlFP,
//						 std::vector<double> & distCumulDallInlFP,
//						 std::vector<double> & costOutBinHeightsTP,
//						 std::vector<double> & distOutBinHeightsTP,
//						 std::vector<double> & costOutBinHeightsFP,
//						 std::vector<double> & distOutBinHeightsFP,
//						 std::vector<float> & cumulDistXvalues,
//						 std::vector<float> & cumulDistXborders,
//						 std::vector<double> & addedCDcumulDistTP, 
//						 std::vector<double> & addedCDcumulDistFP, 
//						 std::vector<double> & add2threshCDcumulDistTP, 
//						 std::vector<double> & add2threshCDcumulDistFP, 
//						 std::vector<double> & add1threshCDcumulDistTP, 
//						 std::vector<double> & add1threshCDcumulDistFP,
//						 int nrinlrats,
//						 float granularity,
//						 int & maxelementsCostsAll, 
//						 int & maxelementsDistsAll,
//						 int & maxelementsCosts,
//						 int & maxelementsDists
//						 );
//void cutFarElemts(std::vector<double> & vec1, std::vector<double> & vec2, float granularity, double minBinHeight, int *maxElements = NULL);
//void getSameVectorSizes(std::vector<std::vector<std::pair<size_t, double>>> & vec, std::vector<double> inlratsin, std::vector<double> & inlratsout);
////Check if evalation was done before
//void checkPrevEval(std::string outpath, std::string outfilename, std::vector<string> filenamesl, bool & datafileexists, int & nrGTmatchesDataset);
//void readParametersTested(int & starttestimgnr, int & nrTotalFails, int & nrSamplesTillNow, std::vector<std::string> filenamesl, std::ifstream *evalsToFile, std::string path, 
//						  std::string outfilename, int & missingSamples, std::vector<std::vector<double>> & StaticSamplesizes, double & maxmaxSampleSize, double & sampleRatio, 
//						  double & minminSampleSize, int & nrGTmatchesDataset, int flowDispH, std::vector<double> & distances, int *truePosArr, int *falsePosArr, int *falseNegArr, 
//						  int & notMatchable, std::vector<cv::Point2f> & errvecs, std::vector<std::pair<cv::Point2f,cv::Point2f>> & perfectMatches, 
//						  std::vector<cv::Point2f> & errvecsGT, std::vector<double> & distancesGT, std::vector<int> & validityValGT, std::vector<double> & distancesEstModel, 
//						  annotImgPars & annotationData);
//int testImagePairs(std::auto_ptr<baseMatcher> mymatcher, std::string descriptorExtractorGT, double & sampleRatio, int & missingSamples, std::ofstream *evalsToFile, int & nrSamplesTillNow, int & nrTotalFails,
//				   double & minminSampleSize, std::vector<std::vector<double>> StaticSamplesizes, int nrGTmatchesDataset, std::vector<double> & distances, int remainingImgs,
//				   int *truePosArr, int *falsePosArr, int *falseNegArr, int & notMatchable, std::vector<cv::Point2f> & errvecs, std::vector<std::pair<cv::Point2f,cv::Point2f>> & perfectMatches,
//				   std::vector<cv::Point2f> & errvecsGT, std::vector<double> & distancesGT, std::vector<int> & validityValGT, std::vector<double> & distancesEstModel, annotImgPars & annotationData,
//				   std::string fileNamesLR, double threshhTh = 64.0, int imgNr = 0, std::string path="", std::string outfilename="");
//// Returns value of Binomial Coefficient C(n, k)
//int binomialCoeff(int n, int k);
//void reEstimateSampleSize(int nrSamplesTillNow, double & minminSampleSize, int nrTotalFails, std::vector<std::vector<double>> StaticSamplesizes, int nrGTmatchesDataset, double & sampleRatio);
//int writeStats(std::string path, std::string filename, std::vector<double> distances, std::vector<double> distancesGT, std::vector<double> distancesEstModel, annotImgPars annotationData);
//void correctMatchingResult(int entrynumber, std::string path, std::string outfilename, std::auto_ptr<baseMatcher> mymatcher);
//int writeErrIndividualImg(std::string path, std::string filename, annotImgPars annotationData);
//int countAutoManualAnnotations(std::string path, std::string filename, annotImgPars annotationData);
//int writeStatsV2(std::string path, std::string filename, annotImgPars annotationData);
//
///* --------------------- Functions --------------------- */
//
///* Starts the time measurement for different matching algorithms and szenes.
// *
// * string imgsPath				Input  -> Path which includes both left and right images
// * string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
// * int flowDispH				Input  -> Indicates which type of ground truth data is used:
// *										  0: flow files from KITTI database
// *										  1: disparity files from KITTI database
// *										  2: homography files (Please note that a homography always relates
// *											 to the first image (e.g. 1->2, 1->3, ...))
// * string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images 
// *									      (after prefix only comes the image number)
// * string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
// *									      (after prefix only comes the image number). For testing with homographies,
// *										  this string can be empty.
// * string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
// *									      (after prefix only comes the image number)
// * string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
// *										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
// *										  detectors are possible.
// * string descriptorExtractor	Input  -> The used descriptor extractor. Possible inputs should only be FREAK
// *										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
// * string matcherType			Input  -> The matcher type under test. Possible inputs are:
// *											CASCHASH: Cascade Hashing matcher
// *											GEOMAWARE: Geometry-aware Feature matching algorithm
// *											GMBSOF: Guided matching based on statistical optical flow
// *											HIRCLUIDX: Hirarchical Clustering Index Matching from the FLANN library
// *											HIRKMEANS: hierarchical k-means tree matcher from the FLANN library
// *											VFCKNN: Vector field consensus (VFC) algorithm with k nearest neighbor 
// *													matches provided from the Hirarchical Clustering Index Matching 
// *													algorithm from the FLANN library
// *											LIBVISO: matcher from the libviso2 library
// *											LINEAR: linear Matching algorithm (Brute force) from the FLANN library
// *											LSHIDX: LSH Index Matching algorithm from the FLANN library
// *											RANDKDTREE: randomized KD-trees matcher from the FLANN library
// * bool useRatioTest			Input  -> Specifies if a ratio test should be performed on the results of a matching
// *										  algorithm. The ratio test is only possible for the following algorithms:
// *										  HIRCLUIDX, HIRKMEANS, LINEAR, LSHIDX, RANDKDTREE
// * string storeResultPath		Input  -> Path were the resulting measurements should be stored
// * bool refine					Input  -> If true [DEFAULT = false], the results from the matching algorithm are
// *										  refined using VFC.
// * double inlRatio				Input  -> The inlier ratio which should be generated using the ground truth data
// *										  [DEFAULT = 1.0]
// * int showResult				Input  -> If >= 0, the result from the matching algorithm is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * int showRefinedResult		Input  -> If >= 0, the result after refinement with VFC is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * string storeImgResPath		Input  -> Optional path for storing the resulting matches drawn into the images,
// *										  where the options of which results should be drawn are specified in
// *										  "showResult". If this path is set, the images are NOT displayed.
// * string storeRefResPath		Input  -> Optional path for storing the resulting matches after refinement 
// *										  drawn into the images, where the options of which results should be 
// *										  drawn are specified in "showRefinedResult". If this path is set, the images 
// *										  are NOT displayed.
// * 
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Failed
// */
//int startTimeMeasurement(std::string imgsPath, std::string flowDispHPath, int flowDispH, 
//						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//						 std::string featureDetector, std::string descriptorExtractor, std::string matcherType,
//						 bool useRatioTest, std::string storeResultPath, bool refine, double inlRatio, 
//						 int showResult, int showRefinedResult, std::string storeImgResPath, 
//						 std::string storeRefResPath, std::string idxPars_NMSLIB, std::string queryPars_NMSLIB)
//{
//	int err;
//	cv::Mat src[2];
//	double minFeatureExtrT = DBL_MAX;
//	vector<double> avgDescrTperKeyp;
//	double avgDescrTperKeypVal = 0;
//	double minDescrExtrT = DBL_MAX;
//	vector<pair<int,double>> matchTinliersGT;
//	vector<pair<int,double>> overallTinliersGT;
//	pair<int, double> medOverallTinliersGT;
//	vector<pair<int,double>> refineTinliersGT;
//	vector<pair<int,double>> overallRefTinliersGT;
//	pair<int, double> medOverallRefTinliersGT;
//	std::auto_ptr<baseMatcher> mymatcher;
//	string outpath, outfilename;
//
//	if(!storeImgResPath.empty() && (showResult == -1))
//	{
//		cout << "If you want to store the resulting images you must specify the showResult flag! Exiting." << endl;
//		exit(0);
//	}
//	if(!storeRefResPath.empty() && (showRefinedResult == -1))
//	{
//		cout << "If you want to store the resulting images with refined matches you must specify the showRefinedResult flag! Exiting." << endl;
//		exit(0);
//	}
//	if(!refine && (showRefinedResult >= 0))
//	{
//		cout << "Cant show refined results if refinement is disabled" << endl;
//	}
//
//	if(flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int i = 0; i < (int)filenamesl.size(); i++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageFlowFile(flowDispHPath, filenamesflow[i], &flowimg);
//			if(err)
//			{
//				cout << "Could not open flow file with index " << i << endl;
//				continue;
//			}
//			if(!matcherType.compare("CASCHASH"))
//			{
//				mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i]));
//			}
//			else if(!matcherType.compare("GMBSOF"))
//			{
//				mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], 0.3));
//			}
//			else if(!matcherType.compare("HIRCLUIDX"))
//			{
//				mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("HIRKMEANS"))
//			{
//				mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("VFCKNN"))
//			{
//				mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], 2, useRatioTest));
//			}
//			else if(!matcherType.compare("LINEAR"))
//			{
//				mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("LSHIDX"))
//			{
//				mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("RANDKDTREE"))
//			{
//				mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if (!matcherType.compare("SWGRAPH"))
//			{
//				mymatcher = std::auto_ptr<swgraph_matcher>(new swgraph_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("HNSW"))
//			{
//				mymatcher = std::auto_ptr<hnsw_matcher>(new hnsw_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("VPTREE"))
//			{
//				mymatcher = std::auto_ptr<vptree_matcher>(new vptree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("ANNOY"))
//			{
//				mymatcher = std::auto_ptr<annoy_matcher>(new annoy_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else
//			{
//				cout << "No valid matcher specified! Exiting." << endl;
//				exit(1);
//			}
//
//			if(mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			cout << "Testing image " << i << " of " << filenamesl.size() << " with inlier ratio " << inlRatio << endl;
//			err = mymatcher->performMatching(inlRatio, true);
//			if(err)
//			{
//				continue;
//			}
//			if(minFeatureExtrT > mymatcher->tf)
//				minFeatureExtrT = mymatcher->tf; //Get minimum feature extraction time
//
//			avgDescrTperKeyp.push_back(mymatcher->tmeanD); //Get mean runtime of the descriptor extractor per 2 keypoints for one image pair (for extraction of left AND right descriptor)
//			if(minDescrExtrT > mymatcher->td)
//				minDescrExtrT = mymatcher->td; //Get time for extraction of all descriptors (left and right keypoints)
//
//			matchTinliersGT.push_back(std::make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->tm)); //Get matching time and number of ground truth inliers
//			overallTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->to)); //Get overall time (feature & descriptor extraction + matching) and number of ground truth inliers
//
//			if(refine)
//			{
//				err = mymatcher->refineMatches();
//				if(!err)
//				{
//					refineTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->tr)); //Get refinement time and number of ground truth inliers
//					overallRefTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->to + mymatcher->tr)); //Get overall time (feature & descriptor extraction + matching) and number of ground truth inliers
//				}
//			}
//
//			if(showResult >= 0)
//			{
//				if(storeImgResPath.empty())
//				{
//					mymatcher->showMatches(showResult);
//				}
//				else
//				{
//					if(dirExists(storeImgResPath)) //Check if output directory existis
//					{
//						outpath = storeImgResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "img_time_flow_idx" + std::to_string((ULONGLONG)i) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//				}
//			}
//			if((showRefinedResult >= 0) && refine && !err)
//			{
//				if(storeRefResPath.empty())
//				{
//					mymatcher->showMatches(showRefinedResult);
//				}
//				else
//				{
//					if(dirExists(storeRefResPath)) //Check if output directory existis
//					{
//						outpath = storeRefResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "imgRef_time_flow_idx" + std::to_string((ULONGLONG)i) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//				}
//			}
//		}
//		if(avgDescrTperKeyp.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//		for(size_t i = 0; i < avgDescrTperKeyp.size(); i++)
//		{
//			avgDescrTperKeypVal += avgDescrTperKeyp[i];
//		}
//		avgDescrTperKeypVal /= (double)avgDescrTperKeyp.size(); //Get mean runtime of the descriptor extractor per 2 keypoints for all image pairs of the szenes (for extraction of left AND right descriptor)
//
//		sort(matchTinliersGT.begin(), matchTinliersGT.end(),
//				[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//		sort(overallTinliersGT.begin(), overallTinliersGT.end(),
//				[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//		if(refine)
//		{
//			if(refineTinliersGT.empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			sort(refineTinliersGT.begin(), refineTinliersGT.end(),
//					[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//			sort(overallRefTinliersGT.begin(), overallRefTinliersGT.end(),
//					[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//			if(overallRefTinliersGT.size() % 2) //Get median of GT inliers and its corresponding overall matching time
//			{
//				medOverallRefTinliersGT.first = overallRefTinliersGT[(overallRefTinliersGT.size()-1)/2].first;
//				medOverallRefTinliersGT.second = overallRefTinliersGT[(overallRefTinliersGT.size()-1)/2].second;
//			}
//			else
//			{
//				medOverallRefTinliersGT.first = (overallRefTinliersGT[overallRefTinliersGT.size() / 2].first + overallRefTinliersGT[overallRefTinliersGT.size() / 2 - 1].first) / 2;
//				medOverallRefTinliersGT.second = (overallRefTinliersGT[overallRefTinliersGT.size() / 2].second + overallRefTinliersGT[overallRefTinliersGT.size() / 2 - 1].second) / 2;
//			}
//		}
//
//		if(overallTinliersGT.size() % 2) //Get median of GT inliers and its corresponding overall matching time
//		{
//			medOverallTinliersGT.first = overallTinliersGT[(overallTinliersGT.size()-1)/2].first;
//			medOverallTinliersGT.second = overallTinliersGT[(overallTinliersGT.size()-1)/2].second;
//		}
//		else
//		{
//			medOverallTinliersGT.first = (overallTinliersGT[overallTinliersGT.size() / 2].first + overallTinliersGT[overallTinliersGT.size() / 2 - 1].first) / 2;
//			medOverallTinliersGT.second = (overallTinliersGT[overallTinliersGT.size() / 2].second + overallTinliersGT[overallTinliersGT.size() / 2 - 1].second) / 2;
//		}		
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_time_flow_featDescrMinVals_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".txt";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "Minimum feature extraction time (for first and second imgs) over all image pairs:" << endl;
//			evalsToFile << minFeatureExtrT << endl << endl;
//			evalsToFile << "Minimum time for extraction of all descriptors (for first and second imgs) over all image pairs:" << endl;
//			evalsToFile << minDescrExtrT << endl << endl;
//			evalsToFile << "Average descriptor extraction time per 2 descriptors (for first and second imgs) over all image pairs:" << endl;
//			evalsToFile << avgDescrTperKeypVal << endl << endl;
//			evalsToFile << "Overall runtime (feature & descriptor extraction + matching time) for the median of GT inliers over all image pairs:" << endl;
//			evalsToFile << "GT inliers: " << medOverallTinliersGT.first << " t: " << medOverallTinliersGT.second << endl;
//			if(refine)
//			{
//				evalsToFile << endl;
//				evalsToFile << "Overall runtime with refinement (feature & descriptor extraction + matching time + refinement) for the median of GT inliers over all image pairs:" << endl;
//				evalsToFile << "GT inliers: " << medOverallRefTinliersGT.first << " t: " << medOverallRefTinliersGT.second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_time_flow_matchT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Number of ground truth inliers (100%) in relation to matching time" << endl;
//			for(size_t i = 0; i < matchTinliersGT.size(); i++)
//			{
//				evalsToFile << matchTinliersGT[i].first << " " << matchTinliersGT[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_time_flow_overallT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Number of ground truth inliers (100%) in relation to overall runtime (feature & descriptor extraction + matching time)" << endl;
//			for(size_t i = 0; i < overallTinliersGT.size(); i++)
//			{
//				evalsToFile << overallTinliersGT[i].first << " " << overallTinliersGT[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		if(refine)
//		{
//			{
//				outfilename = "tex_time_flow_refineT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Number of ground truth inliers (100%) in relation to refinement time" << endl;
//				for(size_t i = 0; i < refineTinliersGT.size(); i++)
//				{
//					evalsToFile << refineTinliersGT[i].first << " " << refineTinliersGT[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_time_flow_overallRefineT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Number of ground truth inliers (100%) in relation to overall runtime (feature & descriptor extraction + matching + refinement time)" << endl;
//				for(size_t i = 0; i < overallRefTinliersGT.size(); i++)
//				{
//					evalsToFile << overallRefTinliersGT[i].first << " " << overallRefTinliersGT[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else if(flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int i = 0; i < (int)filenamesl.size(); i++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageDisparityFile(flowDispHPath, filenamesflow[i], &flowimg);
//			if(err)
//			{
//				cout << "Could not open disparity file with index " << i << endl;
//				continue;
//			}
//			if(!matcherType.compare("CASCHASH"))
//			{
//				mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i]));
//			}
//			else if(!matcherType.compare("GMBSOF"))
//			{
//				mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], 0.3));
//			}
//			else if(!matcherType.compare("HIRCLUIDX"))
//			{
//				mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("HIRKMEANS"))
//			{
//				mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("VFCKNN"))
//			{
//				mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], 2, useRatioTest));
//			}
//			else if(!matcherType.compare("LINEAR"))
//			{
//				mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("LSHIDX"))
//			{
//				mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("RANDKDTREE"))
//			{
//				mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if (!matcherType.compare("SWGRAPH"))
//			{
//				mymatcher = std::auto_ptr<swgraph_matcher>(new swgraph_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("HNSW"))
//			{
//				mymatcher = std::auto_ptr<hnsw_matcher>(new hnsw_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("VPTREE"))
//			{
//				mymatcher = std::auto_ptr<vptree_matcher>(new vptree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("ANNOY"))
//			{
//				mymatcher = std::auto_ptr<annoy_matcher>(new annoy_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else
//			{
//				cout << "No valid matcher specified! Exiting." << endl;
//				exit(1);
//			}
//
//			if(mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			cout << "Testing image " << i << " of " << filenamesl.size() << " with inlier ratio " << inlRatio << endl;
//			err = mymatcher->performMatching(inlRatio, true);
//			if(err)
//			{
//				continue;
//			}
//			if(minFeatureExtrT > mymatcher->tf)
//				minFeatureExtrT = mymatcher->tf; //Get minimum feature extraction time
//
//			avgDescrTperKeyp.push_back(mymatcher->tmeanD); //Get mean runtime of the descriptor extractor per 2 keypoints for one image pair (for extraction of left AND right descriptor)
//			if(minDescrExtrT > mymatcher->td)
//				minDescrExtrT = mymatcher->td; //Get time for extraction of all descriptors (left and right keypoints)
//
//			matchTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->tm)); //Get matching time and number of ground truth inliers
//			overallTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->to)); //Get overall time (feature & descriptor extraction + matching) and number of ground truth inliers
//
//			if(refine)
//			{
//				err = mymatcher->refineMatches();
//				if(!err)
//				{
//					refineTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->tr)); //Get refinement time and number of ground truth inliers
//					overallRefTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->to + mymatcher->tr)); //Get overall time (feature & descriptor extraction + matching) and number of ground truth inliers
//				}
//			}
//
//			if(showResult >= 0)
//			{
//				if(storeImgResPath.empty())
//				{
//					mymatcher->showMatches(showResult);
//				}
//				else
//				{
//					if(dirExists(storeImgResPath)) //Check if output directory existis
//					{
//						outpath = storeImgResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "img_time_disp_idx" + std::to_string((ULONGLONG)i) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//				}
//			}
//			if((showRefinedResult >= 0) && refine && !err)
//			{
//				if(storeRefResPath.empty())
//				{
//					mymatcher->showMatches(showRefinedResult);
//				}
//				else
//				{
//					if(dirExists(storeRefResPath)) //Check if output directory existis
//					{
//						outpath = storeRefResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "imgRef_time_disp_idx" + std::to_string((ULONGLONG)i) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//				}
//			}
//		}
//		if(avgDescrTperKeyp.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//		for(size_t i = 0; i < avgDescrTperKeyp.size(); i++)
//		{
//			avgDescrTperKeypVal += avgDescrTperKeyp[i];
//		}
//		avgDescrTperKeypVal /= (double)avgDescrTperKeyp.size(); //Get mean runtime of the descriptor extractor per 2 keypoints for all image pairs of the szenes (for extraction of left AND right descriptor)
//
//		sort(matchTinliersGT.begin(), matchTinliersGT.end(),
//				[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//		sort(overallTinliersGT.begin(), overallTinliersGT.end(),
//				[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//		if(refine)
//		{
//			if(refineTinliersGT.empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			sort(refineTinliersGT.begin(), refineTinliersGT.end(),
//					[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//			sort(overallRefTinliersGT.begin(), overallRefTinliersGT.end(),
//					[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//			if(overallRefTinliersGT.size() % 2) //Get median of GT inliers and its corresponding overall matching time
//			{
//				medOverallRefTinliersGT.first = overallRefTinliersGT[(overallRefTinliersGT.size()-1)/2].first;
//				medOverallRefTinliersGT.second = overallRefTinliersGT[(overallRefTinliersGT.size()-1)/2].second;
//			}
//			else
//			{
//				medOverallRefTinliersGT.first = (overallRefTinliersGT[overallRefTinliersGT.size() / 2].first + overallRefTinliersGT[overallRefTinliersGT.size() / 2 - 1].first) / 2;
//				medOverallRefTinliersGT.second = (overallRefTinliersGT[overallRefTinliersGT.size() / 2].second + overallRefTinliersGT[overallRefTinliersGT.size() / 2 - 1].second) / 2;
//			}
//		}
//
//		if(overallTinliersGT.size() % 2) //Get median of GT inliers and its corresponding overall matching time
//		{
//			medOverallTinliersGT.first = overallTinliersGT[(overallTinliersGT.size()-1)/2].first;
//			medOverallTinliersGT.second = overallTinliersGT[(overallTinliersGT.size()-1)/2].second;
//		}
//		else
//		{
//			medOverallTinliersGT.first = (overallTinliersGT[overallTinliersGT.size() / 2].first + overallTinliersGT[overallTinliersGT.size() / 2 - 1].first) / 2;
//			medOverallTinliersGT.second = (overallTinliersGT[overallTinliersGT.size() / 2].second + overallTinliersGT[overallTinliersGT.size() / 2 - 1].second) / 2;
//		}		
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_time_disp_featDescrMinVals_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".txt";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "Minimum feature extraction time (for left and right imgs) over all image pairs:" << endl;
//			evalsToFile << minFeatureExtrT << endl << endl;
//			evalsToFile << "Minimum time for extraction of all descriptors (for left and right imgs) over all image pairs:" << endl;
//			evalsToFile << minDescrExtrT << endl << endl;
//			evalsToFile << "Average descriptor extraction time per 2 descriptors (for left and right imgs) over all image pairs:" << endl;
//			evalsToFile << avgDescrTperKeypVal << endl << endl;
//			evalsToFile << "Overall runtime (feature & descriptor extraction + matching time) for the median of GT inliers over all image pairs:" << endl;
//			evalsToFile << "GT inliers: " << medOverallTinliersGT.first << " t: " << medOverallTinliersGT.second << endl;
//			if(refine)
//			{
//				evalsToFile << endl;
//				evalsToFile << "Overall runtime with refinement (feature & descriptor extraction + matching time + refinement) for the median of GT inliers over all image pairs:" << endl;
//				evalsToFile << "GT inliers: " << medOverallRefTinliersGT.first << " t: " << medOverallRefTinliersGT.second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_time_disp_matchT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Number of ground truth inliers (100%) in relation to matching time" << endl;
//			for(size_t i = 0; i < matchTinliersGT.size(); i++)
//			{
//				evalsToFile << matchTinliersGT[i].first << " " << matchTinliersGT[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_time_disp_overallT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Number of ground truth inliers (100%) in relation to overall runtime (feature & descriptor extraction + matching time)" << endl;
//			for(size_t i = 0; i < overallTinliersGT.size(); i++)
//			{
//				evalsToFile << overallTinliersGT[i].first << " " << overallTinliersGT[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		if(refine)
//		{
//			{
//				outfilename = "tex_time_disp_refineT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Number of ground truth inliers (100%) in relation to refinement time" << endl;
//				for(size_t i = 0; i < refineTinliersGT.size(); i++)
//				{
//					evalsToFile << refineTinliersGT[i].first << " " << refineTinliersGT[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_time_disp_overallRefineT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Number of ground truth inliers (100%) in relation to overall runtime (feature & descriptor extraction + matching + refinement time)" << endl;
//				for(size_t i = 0; i < overallRefTinliersGT.size(); i++)
//				{
//					evalsToFile << overallRefTinliersGT[i].first << " " << overallRefTinliersGT[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else if(flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		//cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if(err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if(err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if(err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		if(fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//			for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//				
//				if(!matcherType.compare("CASCHASH"))
//				{
//					mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("GMBSOF"))
//				{
//					mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 0.3));
//				}
//				else if(!matcherType.compare("HIRCLUIDX"))
//				{
//					mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("HIRKMEANS"))
//				{
//					mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("VFCKNN"))
//				{
//					mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 2, useRatioTest));
//				}
//				else if(!matcherType.compare("LINEAR"))
//				{
//					mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("LSHIDX"))
//				{
//					mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("RANDKDTREE"))
//				{
//					mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if (!matcherType.compare("SWGRAPH"))
//				{
//					mymatcher = std::auto_ptr<swgraph_matcher>(new swgraph_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("HNSW"))
//				{
//					mymatcher = std::auto_ptr<hnsw_matcher>(new hnsw_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("VPTREE"))
//				{
//					mymatcher = std::auto_ptr<vptree_matcher>(new vptree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("ANNOY"))
//				{
//					mymatcher = std::auto_ptr<annoy_matcher>(new annoy_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else
//				{
//					cout << "No valid matcher specified! Exiting." << endl;
//					exit(1);
//				}
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				cout << "Testing image " << idx1 << " with inlier ratio " << inlRatio << endl;
//				err = mymatcher->performMatching(inlRatio, true);
//				if(err)
//				{
//					continue;
//				}
//				if(minFeatureExtrT > mymatcher->tf)
//					minFeatureExtrT = mymatcher->tf; //Get minimum feature extraction time
//
//				avgDescrTperKeyp.push_back(mymatcher->tmeanD); //Get mean runtime of the descriptor extractor per 2 keypoints for one image pair (for extraction of left AND right descriptor)
//				if(minDescrExtrT > mymatcher->td)
//					minDescrExtrT = mymatcher->td; //Get time for extraction of all descriptors (left and right keypoints)
//
//				matchTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->tm)); //Get matching time and number of ground truth inliers
//				overallTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->to)); //Get overall time (feature & descriptor extraction + matching) and number of ground truth inliers
//
//				if(refine)
//				{
//					err = mymatcher->refineMatches();
//					if(!err)
//					{
//						refineTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->tr)); //Get refinement time and number of ground truth inliers
//						overallRefTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->to + mymatcher->tr)); //Get overall time (feature & descriptor extraction + matching) and number of ground truth inliers
//					}
//				}
//
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_time_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//				if((showRefinedResult >= 0) && refine && !err)
//				{
//					if(storeRefResPath.empty())
//					{
//						mymatcher->showMatches(showRefinedResult);
//					}
//					else
//					{
//						if(dirExists(storeRefResPath)) //Check if output directory existis
//						{
//							outpath = storeRefResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "imgRef_time_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//					}
//				}
//			}
//			//Generate new homographys to evaluate all other possible configurations of the images to each other
//			int nr_evalsH = (int)fnames.size();
//			for(int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//			{
//				for(int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//				{
//					cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//					src[0] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx2 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//					
//					if(!matcherType.compare("CASCHASH"))
//					{
//						mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//					}
//					else if(!matcherType.compare("GMBSOF"))
//					{
//						mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], 0.3));
//					}
//					else if(!matcherType.compare("HIRCLUIDX"))
//					{
//						mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//					}
//					else if(!matcherType.compare("HIRKMEANS"))
//					{
//						mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//					}
//					else if(!matcherType.compare("VFCKNN"))
//					{
//						mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], 2, useRatioTest));
//					}
//					else if(!matcherType.compare("LINEAR"))
//					{
//						mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//					}
//					else if(!matcherType.compare("LSHIDX"))
//					{
//						mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//					}
//					else if(!matcherType.compare("RANDKDTREE"))
//					{
//						mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//					}
//					else if (!matcherType.compare("SWGRAPH"))
//					{
//						mymatcher = std::auto_ptr<swgraph_matcher>(new swgraph_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//					}
//					else if (!matcherType.compare("HNSW"))
//					{
//						mymatcher = std::auto_ptr<hnsw_matcher>(new hnsw_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//					}
//					else if (!matcherType.compare("VPTREE"))
//					{
//						mymatcher = std::auto_ptr<vptree_matcher>(new vptree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//					}
//					else if (!matcherType.compare("ANNOY"))
//					{
//						mymatcher = std::auto_ptr<annoy_matcher>(new annoy_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//					}
//					else
//					{
//						cout << "No valid matcher specified! Exiting." << endl;
//						exit(1);
//					}
//
//					if(mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					cout << "Testing image " << nr_evalsH << " with inlier ratio " << inlRatio << endl;
//					nr_evalsH++;
//					err = mymatcher->performMatching(inlRatio, true);
//					if(err)
//					{
//						continue;
//					}
//					if(minFeatureExtrT > mymatcher->tf)
//						minFeatureExtrT = mymatcher->tf; //Get minimum feature extraction time
//
//					avgDescrTperKeyp.push_back(mymatcher->tmeanD); //Get mean runtime of the descriptor extractor per 2 keypoints for one image pair (for extraction of left AND right descriptor)
//					if(minDescrExtrT > mymatcher->td)
//						minDescrExtrT = mymatcher->td; //Get time for extraction of all descriptors (left and right keypoints)
//
//					matchTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->tm)); //Get matching time and number of ground truth inliers
//					overallTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->to)); //Get overall time (feature & descriptor extraction + matching) and number of ground truth inliers
//
//					if(refine)
//					{
//						err = mymatcher->refineMatches();
//						if(!err)
//						{
//							refineTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->tr)); //Get refinement time and number of ground truth inliers
//							overallRefTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->to + mymatcher->tr)); //Get overall time (feature & descriptor extraction + matching) and number of ground truth inliers
//						}
//					}
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_time_H_idxs" + std::to_string((ULONGLONG)(idx1 + 1)) + "-" + std::to_string((ULONGLONG)(idx2 + 1)) + 
//										"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//										matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//					if((showRefinedResult >= 0) && refine && !err)
//					{
//						if(storeRefResPath.empty())
//						{
//							mymatcher->showMatches(showRefinedResult);
//						}
//						else
//						{
//							if(dirExists(storeRefResPath)) //Check if output directory existis
//							{
//								outpath = storeRefResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "imgRef_time_H_idxs" + std::to_string((ULONGLONG)(idx1 + 1)) + "-" + std::to_string((ULONGLONG)(idx2 + 1)) + 
//										"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//										matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//						}
//					}
//				}
//			}
//		}
//		else
//		{
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//			for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//				
//				if(!matcherType.compare("CASCHASH"))
//				{
//					mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("GMBSOF"))
//				{
//					mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 0.3));
//				}
//				else if(!matcherType.compare("HIRCLUIDX"))
//				{
//					mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("HIRKMEANS"))
//				{
//					mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("VFCKNN"))
//				{
//					mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 2, useRatioTest));
//				}
//				else if(!matcherType.compare("LINEAR"))
//				{
//					mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("LSHIDX"))
//				{
//					mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("RANDKDTREE"))
//				{
//					mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if (!matcherType.compare("SWGRAPH"))
//				{
//					mymatcher = std::auto_ptr<swgraph_matcher>(new swgraph_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("HNSW"))
//				{
//					mymatcher = std::auto_ptr<hnsw_matcher>(new hnsw_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("VPTREE"))
//				{
//					mymatcher = std::auto_ptr<vptree_matcher>(new vptree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("ANNOY"))
//				{
//					mymatcher = std::auto_ptr<annoy_matcher>(new annoy_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else
//				{
//					cout << "No valid matcher specified! Exiting." << endl;
//					exit(1);
//				}
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				cout << "Testing image " << idx1 << " with inlier ratio " << inlRatio << endl;
//				err = mymatcher->performMatching(inlRatio, true);
//				if(err)
//				{
//					continue;
//				}
//				if(minFeatureExtrT > mymatcher->tf)
//					minFeatureExtrT = mymatcher->tf; //Get minimum feature extraction time
//
//				avgDescrTperKeyp.push_back(mymatcher->tmeanD); //Get mean runtime of the descriptor extractor per 2 keypoints for one image pair (for extraction of left AND right descriptor)
//				if(minDescrExtrT > mymatcher->td)
//					minDescrExtrT = mymatcher->td; //Get time for extraction of all descriptors (left and right keypoints)
//
//				matchTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->tm)); //Get matching time and number of ground truth inliers
//				overallTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->to)); //Get overall time (feature & descriptor extraction + matching) and number of ground truth inliers
//
//				if(refine)
//				{
//					err = mymatcher->refineMatches();
//					if(!err)
//					{
//						refineTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->tr)); //Get refinement time and number of ground truth inliers
//						overallRefTinliersGT.push_back(make_pair((int)mymatcher->positivesGT + (int)mymatcher->negativesGTl, mymatcher->to + mymatcher->tr)); //Get overall time (feature & descriptor extraction + matching) and number of ground truth inliers
//					}
//				}
//
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_time_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//				if((showRefinedResult >= 0) && refine && !err)
//				{
//					if(storeRefResPath.empty())
//					{
//						mymatcher->showMatches(showRefinedResult);
//					}
//					else
//					{
//						if(dirExists(storeRefResPath)) //Check if output directory existis
//						{
//							outpath = storeRefResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "imgRef_time_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//					}
//				}
//			}
//		}
//
//		if(avgDescrTperKeyp.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//		for(size_t i = 0; i < avgDescrTperKeyp.size(); i++)
//		{
//			avgDescrTperKeypVal += avgDescrTperKeyp[i];
//		}
//		avgDescrTperKeypVal /= (double)avgDescrTperKeyp.size(); //Get mean runtime of the descriptor extractor per 2 keypoints for all image pairs of the szenes (for extraction of left AND right descriptor)
//
//		sort(matchTinliersGT.begin(), matchTinliersGT.end(),
//				[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//		sort(overallTinliersGT.begin(), overallTinliersGT.end(),
//				[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//		if(refine)
//		{
//			if(refineTinliersGT.empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			sort(refineTinliersGT.begin(), refineTinliersGT.end(),
//					[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//			sort(overallRefTinliersGT.begin(), overallRefTinliersGT.end(),
//					[](pair<int,double> first, pair<int,double> second){return first.first < second.first;});
//
//			if(overallRefTinliersGT.size() % 2) //Get median of GT inliers and its corresponding overall matching time
//			{
//				medOverallRefTinliersGT.first = overallRefTinliersGT[(overallRefTinliersGT.size()-1)/2].first;
//				medOverallRefTinliersGT.second = overallRefTinliersGT[(overallRefTinliersGT.size()-1)/2].second;
//			}
//			else
//			{
//				medOverallRefTinliersGT.first = (overallRefTinliersGT[overallRefTinliersGT.size() / 2].first + overallRefTinliersGT[overallRefTinliersGT.size() / 2 - 1].first) / 2;
//				medOverallRefTinliersGT.second = (overallRefTinliersGT[overallRefTinliersGT.size() / 2].second + overallRefTinliersGT[overallRefTinliersGT.size() / 2 - 1].second) / 2;
//			}
//		}
//
//		if(overallTinliersGT.size() % 2) //Get median of GT inliers and its corresponding overall matching time
//		{
//			medOverallTinliersGT.first = overallTinliersGT[(overallTinliersGT.size()-1)/2].first;
//			medOverallTinliersGT.second = overallTinliersGT[(overallTinliersGT.size()-1)/2].second;
//		}
//		else
//		{
//			medOverallTinliersGT.first = (overallTinliersGT[overallTinliersGT.size() / 2].first + overallTinliersGT[overallTinliersGT.size() / 2 - 1].first) / 2;
//			medOverallTinliersGT.second = (overallTinliersGT[overallTinliersGT.size() / 2].second + overallTinliersGT[overallTinliersGT.size() / 2 - 1].second) / 2;
//		}		
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_time_H_featDescrMinVals_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".txt";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "Minimum feature extraction time (for fisrt and second imgs) over all image pairs:" << endl;
//			evalsToFile << minFeatureExtrT << endl << endl;
//			evalsToFile << "Minimum time for extraction of all descriptors (for fisrt and second imgs) over all image pairs:" << endl;
//			evalsToFile << minDescrExtrT << endl << endl;
//			evalsToFile << "Average descriptor extraction time per 2 descriptors (for fisrt and second imgs) over all image pairs:" << endl;
//			evalsToFile << avgDescrTperKeypVal << endl << endl;
//			evalsToFile << "Overall runtime (feature & descriptor extraction + matching time) for the median of GT inliers over all image pairs:" << endl;
//			evalsToFile << "GT inliers: " << medOverallTinliersGT.first << " t: " << medOverallTinliersGT.second << endl;
//			if(refine)
//			{
//				evalsToFile << endl;
//				evalsToFile << "Overall runtime with refinement (feature & descriptor extraction + matching time + refinement) for the median of GT inliers over all image pairs:" << endl;
//				evalsToFile << "GT inliers: " << medOverallRefTinliersGT.first << " t: " << medOverallRefTinliersGT.second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_time_H_matchT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Number of ground truth inliers (100%) in relation to matching time" << endl;
//			for(size_t i = 0; i < matchTinliersGT.size(); i++)
//			{
//				evalsToFile << matchTinliersGT[i].first << " " << matchTinliersGT[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_time_H_overallT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Number of ground truth inliers (100%) in relation to overall runtime (feature & descriptor extraction + matching time)" << endl;
//			for(size_t i = 0; i < overallTinliersGT.size(); i++)
//			{
//				evalsToFile << overallTinliersGT[i].first << " " << overallTinliersGT[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		if(refine)
//		{
//			{
//				outfilename = "tex_time_H_refineT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Number of ground truth inliers (100%) in relation to refinement time" << endl;
//				for(size_t i = 0; i < refineTinliersGT.size(); i++)
//				{
//					evalsToFile << refineTinliersGT[i].first << " " << refineTinliersGT[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_time_H_overallRefineT_inliersGT_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Number of ground truth inliers (100%) in relation to overall runtime (feature & descriptor extraction + matching + refinement time)" << endl;
//				for(size_t i = 0; i < overallRefTinliersGT.size(); i++)
//				{
//					evalsToFile << overallRefTinliersGT[i].first << " " << overallRefTinliersGT[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//	return 0;
//}
//
///* Starts the test for different inlier ratios on one image pair for different matching algorithms and szenes.
// * The output is generated for precision, recall, fall-out and accuracy.
// *
// * string imgsPath				Input  -> Path which includes both left and right images
// * string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
// * int flowDispH				Input  -> Indicates which type of ground truth data is used:
// *										  0: flow files from KITTI database
// *										  1: disparity files from KITTI database
// *										  2: homography files (Please note that a homography always relates
// *											 to the first image (e.g. 1->2, 1->3, ...))
// * string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images 
// *									      (after prefix only comes the image number)
// * string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
// *									      (after prefix only comes the image number). For testing with homographies,
// *										  this string can be empty.
// * string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
// *									      (after prefix only comes the image number)
// * string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
// *										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
// *										  detectors are possible.
// * string descriptorExtractor	Input  -> The used descriptor extractor. Possible inputs should only be FREAK
// *										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
// * string matcherType			Input  -> The matcher type under test. Possible inputs are:
// *											CASCHASH: Cascade Hashing matcher
// *											GEOMAWARE: Geometry-aware Feature matching algorithm
// *											GMBSOF: Guided matching based on statistical optical flow
// *											HIRCLUIDX: Hirarchical Clustering Index Matching from the FLANN library
// *											HIRKMEANS: hierarchical k-means tree matcher from the FLANN library
// *											VFCKNN: Vector field consensus (VFC) algorithm with k nearest neighbor 
// *													matches provided from the Hirarchical Clustering Index Matching 
// *													algorithm from the FLANN library
// *											LIBVISO: matcher from the libviso2 library
// *											LINEAR: linear Matching algorithm (Brute force) from the FLANN library
// *											LSHIDX: LSH Index Matching algorithm from the FLANN library
// *											RANDKDTREE: randomized KD-trees matcher from the FLANN library
// * bool useRatioTest			Input  -> Specifies if a ratio test should be performed on the results of a matching
// *										  algorithm. The ratio test is only possible for the following algorithms:
// *										  HIRCLUIDX, HIRKMEANS, LINEAR, LSHIDX, RANDKDTREE
// * string storeResultPath		Input  -> Path were the resulting measurements should be stored
// * int imgidx1					Input  -> Index number (starting with 0) for the image pair for which the test
// *										  should be performed. Only this value is needed for flow or disparity image
// *										  data. For homography image data this index points to the first image and
// *										  imgidx2 points to the second image.
// * int imgidx2					Input  -> Only used for homography image data. It specifies the index of the second
// *										  image that should be used.
// * bool refine					Input  -> If true [DEFAULT = false], the results from the matching algorithm are
// *										  refined using VFC.
// * int showResult				Input  -> If >= 0, the result from the matching algorithm is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * int showRefinedResult		Input  -> If >= 0, the result after refinement with VFC is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * string storeImgResPath		Input  -> Optional path for storing the resulting matches drawn into the images,
// *										  where the options of which results should be drawn are specified in
// *										  "showResult". If this path is set, the images are NOT displayed.
// * string storeRefResPath		Input  -> Optional path for storing the resulting matches after refinement 
// *										  drawn into the images, where the options of which results should be 
// *										  drawn are specified in "showRefinedResult". If this path is set, the images 
// *										  are NOT displayed.
// * 
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Failed
// */
//int startInlierRatioMeasurement(std::string imgsPath, std::string flowDispHPath, int flowDispH, 
//								std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//								std::string featureDetector, std::string descriptorExtractor, std::string matcherType,
//								bool useRatioTest, std::string storeResultPath, int imgidx1, int imgidx2, bool refine, 
//								int showResult, int showRefinedResult, std::string storeImgResPath, 
//								std::string storeRefResPath)
//{
//	int err;
//	cv::Mat src[2];
//	vector<pair<double,double>> inlRatPrecision, inlRatRecall, inlRatFallOut, inlRatAccuracy;
//	vector<pair<double,double>> inlRatPrecisionRef, inlRatRecallRef, inlRatFallOutRef, inlRatAccuracyRef;
//	std::auto_ptr<baseMatcher> mymatcher;
//	string outpath, outfilename;
//	vector<double> inlierRatios;
//
//	if(!storeImgResPath.empty() && (showResult == -1))
//	{
//		cout << "If you want to store the resulting images you must specify the showResult flag! Exiting." << endl;
//		exit(0);
//	}
//	if(!storeRefResPath.empty() && (showRefinedResult == -1))
//	{
//		cout << "If you want to store the resulting images with refined matches you must specify the showRefinedResult flag! Exiting." << endl;
//		exit(0);
//	}
//	if(!refine && (showRefinedResult >= 0))
//	{
//		cout << "Cant show refined results if refinement is disabled" << endl;
//	}
//
//	//Generate inlier ratios
//	double startInlRatio = 1.0;
//	inlierRatios.push_back(startInlRatio);
//	while(startInlRatio > 0.2)
//	{
//		startInlRatio -= 0.05;
//		inlierRatios.push_back(startInlRatio);
//	}
//	while(startInlRatio > 0.1)
//	{
//		startInlRatio -= 0.02;
//		inlierRatios.push_back(startInlRatio);
//	}
//	while(startInlRatio > 0.03)
//	{
//		startInlRatio -= 0.01;
//		inlierRatios.push_back(startInlRatio);
//	}
//	while(startInlRatio > 0.005)
//	{
//		startInlRatio -= 0.005;
//		inlierRatios.push_back(startInlRatio);
//	}
//
//	if(flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//		if((imgidx1 < 0) || (imgidx1 >= filenamesl.size()))
//		{
//			cout << "Image pair index ouot of range! Exiting." << endl;
//			exit(0);
//		}
//
//		cv::Mat flowimg;
//		src[0] = cv::imread(imgsPath + "\\" + filenamesl[imgidx1],CV_LOAD_IMAGE_GRAYSCALE);
//		src[1] = cv::imread(imgsPath + "\\" + filenamesr[imgidx1],CV_LOAD_IMAGE_GRAYSCALE);
//		err = convertImageFlowFile(flowDispHPath, filenamesflow[imgidx1], &flowimg);
//		if(err)
//		{
//			cout << "Could not open flow file with index " << imgidx1 << ". Exiting." << endl;
//			exit(0);
//		}
//		if(!matcherType.compare("CASCHASH"))
//		{
//			mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1]));
//		}
//		else if(!matcherType.compare("GMBSOF"))
//		{
//			mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], 0.41));
//		}
//		else if(!matcherType.compare("HIRCLUIDX"))
//		{
//			mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], useRatioTest));
//		}
//		else if(!matcherType.compare("HIRKMEANS"))
//		{
//			mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], useRatioTest));
//		}
//		else if(!matcherType.compare("VFCKNN"))
//		{
//			mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], 2, useRatioTest));
//		}
//		else if(!matcherType.compare("LINEAR"))
//		{
//			mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], useRatioTest));
//		}
//		else if(!matcherType.compare("LSHIDX"))
//		{
//			mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], useRatioTest));
//		}
//		else if(!matcherType.compare("RANDKDTREE"))
//		{
//			mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], useRatioTest));
//		}
//		else
//		{
//			cout << "No valid matcher specified! Exiting." << endl;
//			exit(1);
//		}
//		if(mymatcher->specialGMbSOFtest)
//		{
//			cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//			exit(0);
//		}
//
//		for(size_t i = 0; i < inlierRatios.size(); i++)
//		{
//			err = mymatcher->performMatching(inlierRatios[i]);
//			if(err)
//			{
//				if(err == -2)
//				{
//					inlRatPrecision.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatRecall.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatFallOut.push_back(make_pair(mymatcher->inlRatioL, 1.0));
//					inlRatAccuracy.push_back(make_pair(mymatcher->inlRatioL, 0));
//					continue;
//				}
//				else
//				{
//					continue;
//				}
//			}
//			inlRatPrecision.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.ppv));
//			inlRatRecall.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.tpr));
//			inlRatFallOut.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.fpr));
//			inlRatAccuracy.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.acc));
//
//			if(refine)
//			{
//				err = mymatcher->refineMatches();
//				if(err)
//				{
//					inlRatPrecisionRef.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatRecallRef.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatFallOutRef.push_back(make_pair(mymatcher->inlRatioL, 1.0));
//					inlRatAccuracyRef.push_back(make_pair(mymatcher->inlRatioL, 0));
//				}
//				else
//				{
//					inlRatPrecisionRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.ppv));
//					inlRatRecallRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.tpr));
//					inlRatFallOutRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.fpr));
//					inlRatAccuracyRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.acc));
//				}
//			}
//
//			if(showResult >= 0)
//			{
//				if(storeImgResPath.empty())
//				{
//					mymatcher->showMatches(showResult);
//				}
//				else
//				{
//					if(dirExists(storeImgResPath)) //Check if output directory existis
//					{
//						outpath = storeImgResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "img_inlRatio_flow_idx" + std::to_string((ULONGLONG)imgidx1) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//				}
//			}
//			if((showRefinedResult >= 0) && refine && !err)
//			{
//				if(storeRefResPath.empty())
//				{
//					mymatcher->showMatches(showRefinedResult);
//				}
//				else
//				{
//					if(dirExists(storeRefResPath)) //Check if output directory existis
//					{
//						outpath = storeRefResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "imgRef_inlRatio_flow_idx" + std::to_string((ULONGLONG)imgidx1) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//				}
//			}
//
//		}
//		if(inlRatPrecision.empty())
//		{
//			cout << "Ground truth generation or algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		sort(inlRatPrecision.begin(), inlRatPrecision.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		sort(inlRatRecall.begin(), inlRatRecall.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		sort(inlRatFallOut.begin(), inlRatFallOut.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		sort(inlRatAccuracy.begin(), inlRatAccuracy.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//
//		if(refine)
//		{
//			if(inlRatPrecisionRef.empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			sort(inlRatPrecisionRef.begin(), inlRatPrecisionRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//			sort(inlRatRecallRef.begin(), inlRatRecallRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//			sort(inlRatFallOutRef.begin(), inlRatFallOutRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//			sort(inlRatAccuracyRef.begin(), inlRatAccuracyRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_inlRatio_flow_inliersGT_precision_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			for(size_t i = 0; i < inlRatPrecision.size(); i++)
//			{
//				evalsToFile << inlRatPrecision[i].first << " " << inlRatPrecision[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_flow_inliersGT_recall_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			for(size_t i = 0; i < inlRatRecall.size(); i++)
//			{
//				evalsToFile << inlRatRecall[i].first << " " << inlRatRecall[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_flow_inliersGT_fpr_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			for(size_t i = 0; i < inlRatFallOut.size(); i++)
//			{
//				evalsToFile << inlRatFallOut[i].first << " " << inlRatFallOut[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_flow_inliersGT_acc_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			for(size_t i = 0; i < inlRatAccuracy.size(); i++)
//			{
//				evalsToFile << inlRatAccuracy[i].first << " " << inlRatAccuracy[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		if(refine)
//		{
//			{
//				outfilename = "tex_inlRatio_flow_inliersGT_precisionRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				for(size_t i = 0; i < inlRatPrecisionRef.size(); i++)
//				{
//					evalsToFile << inlRatPrecisionRef[i].first << " " << inlRatPrecisionRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_flow_inliersGT_recallRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				for(size_t i = 0; i < inlRatRecallRef.size(); i++)
//				{
//					evalsToFile << inlRatRecallRef[i].first << " " << inlRatRecallRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_flow_inliersGT_fprRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				for(size_t i = 0; i < inlRatFallOutRef.size(); i++)
//				{
//					evalsToFile << inlRatFallOutRef[i].first << " " << inlRatFallOutRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_flow_inliersGT_accRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				for(size_t i = 0; i < inlRatAccuracyRef.size(); i++)
//				{
//					evalsToFile << inlRatAccuracyRef[i].first << " " << inlRatAccuracyRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else if(flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//		if((imgidx1 < 0) || (imgidx1 >= filenamesl.size()))
//		{
//			cout << "Image pair index ouot of range! Exiting." << endl;
//			exit(0);
//		}
//
//		cv::Mat flowimg;
//		src[0] = cv::imread(imgsPath + "\\" + filenamesl[imgidx1],CV_LOAD_IMAGE_GRAYSCALE);
//		src[1] = cv::imread(imgsPath + "\\" + filenamesr[imgidx1],CV_LOAD_IMAGE_GRAYSCALE);
//		err = convertImageDisparityFile(flowDispHPath, filenamesflow[imgidx1], &flowimg);
//		if(err)
//		{
//			cout << "Could not open disparity file with index " << imgidx1 << ". Exiting." << endl;
//			exit(0);
//		}
//		if(!matcherType.compare("CASCHASH"))
//		{
//			mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1]));
//		}
//		else if(!matcherType.compare("GMBSOF"))
//		{
//			mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], 0.26));
//		}
//		else if(!matcherType.compare("HIRCLUIDX"))
//		{
//			mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], useRatioTest));
//		}
//		else if(!matcherType.compare("HIRKMEANS"))
//		{
//			mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], useRatioTest));
//		}
//		else if(!matcherType.compare("VFCKNN"))
//		{
//			mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], 2, useRatioTest));
//		}
//		else if(!matcherType.compare("LINEAR"))
//		{
//			mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], useRatioTest));
//		}
//		else if(!matcherType.compare("LSHIDX"))
//		{
//			mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], useRatioTest));
//		}
//		else if(!matcherType.compare("RANDKDTREE"))
//		{
//			mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[imgidx1], useRatioTest));
//		}
//		else
//		{
//			cout << "No valid matcher specified! Exiting." << endl;
//			exit(1);
//		}
//
//		if(mymatcher->specialGMbSOFtest)
//		{
//			cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//			exit(0);
//		}
//
//		for(size_t i = 0; i < inlierRatios.size(); i++)
//		{
//			err = mymatcher->performMatching(inlierRatios[i]);
//			if(err)
//			{
//				if(err == -2)
//				{
//					inlRatPrecision.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatRecall.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatFallOut.push_back(make_pair(mymatcher->inlRatioL, 1.0));
//					inlRatAccuracy.push_back(make_pair(mymatcher->inlRatioL, 0));
//					continue;
//				}
//				else
//				{
//					continue;
//				}
//			}
//			inlRatPrecision.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.ppv));
//			inlRatRecall.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.tpr));
//			inlRatFallOut.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.fpr));
//			inlRatAccuracy.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.acc));
//
//			if(refine)
//			{
//				err = mymatcher->refineMatches();
//				if(err)
//				{
//					inlRatPrecisionRef.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatRecallRef.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatFallOutRef.push_back(make_pair(mymatcher->inlRatioL, 1.0));
//					inlRatAccuracyRef.push_back(make_pair(mymatcher->inlRatioL, 0));
//				}
//				else
//				{
//					inlRatPrecisionRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.ppv));
//					inlRatRecallRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.tpr));
//					inlRatFallOutRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.fpr));
//					inlRatAccuracyRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.acc));
//				}
//			}
//
//			if(showResult >= 0)
//			{
//				if(storeImgResPath.empty())
//				{
//					mymatcher->showMatches(showResult);
//				}
//				else
//				{
//					if(dirExists(storeImgResPath)) //Check if output directory existis
//					{
//						outpath = storeImgResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "img_inlRatio_disp_idx" + std::to_string((ULONGLONG)imgidx1) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//				}
//			}
//			if((showRefinedResult >= 0) && refine && !err)
//			{
//				if(storeRefResPath.empty())
//				{
//					mymatcher->showMatches(showRefinedResult);
//				}
//				else
//				{
//					if(dirExists(storeRefResPath)) //Check if output directory existis
//					{
//						outpath = storeRefResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "imgRef_inlRatio_disp_idx" + std::to_string((ULONGLONG)imgidx1) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//				}
//			}
//
//		}
//
//		if(inlRatPrecision.empty())
//		{
//			cout << "Ground truth generation or algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		sort(inlRatPrecision.begin(), inlRatPrecision.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		sort(inlRatRecall.begin(), inlRatRecall.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		sort(inlRatFallOut.begin(), inlRatFallOut.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		sort(inlRatAccuracy.begin(), inlRatAccuracy.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//
//		if(refine)
//		{
//			if(inlRatPrecisionRef.empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			sort(inlRatPrecisionRef.begin(), inlRatPrecisionRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//			sort(inlRatRecallRef.begin(), inlRatRecallRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//			sort(inlRatFallOutRef.begin(), inlRatFallOutRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//			sort(inlRatAccuracyRef.begin(), inlRatAccuracyRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_inlRatio_disp_inliersGT_precision_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			for(size_t i = 0; i < inlRatPrecision.size(); i++)
//			{
//				evalsToFile << inlRatPrecision[i].first << " " << inlRatPrecision[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_disp_inliersGT_recall_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			for(size_t i = 0; i < inlRatRecall.size(); i++)
//			{
//				evalsToFile << inlRatRecall[i].first << " " << inlRatRecall[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_disp_inliersGT_fpr_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			for(size_t i = 0; i < inlRatFallOut.size(); i++)
//			{
//				evalsToFile << inlRatFallOut[i].first << " " << inlRatFallOut[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_disp_inliersGT_acc_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			for(size_t i = 0; i < inlRatAccuracy.size(); i++)
//			{
//				evalsToFile << inlRatAccuracy[i].first << " " << inlRatAccuracy[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		if(refine)
//		{
//			{
//				outfilename = "tex_inlRatio_disp_inliersGT_precisionRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				for(size_t i = 0; i < inlRatPrecisionRef.size(); i++)
//				{
//					evalsToFile << inlRatPrecisionRef[i].first << " " << inlRatPrecisionRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_disp_inliersGT_recallRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				for(size_t i = 0; i < inlRatRecallRef.size(); i++)
//				{
//					evalsToFile << inlRatRecallRef[i].first << " " << inlRatRecallRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_disp_inliersGT_fprRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				for(size_t i = 0; i < inlRatFallOutRef.size(); i++)
//				{
//					evalsToFile << inlRatFallOutRef[i].first << " " << inlRatFallOutRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_disp_inliersGT_accRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				for(size_t i = 0; i < inlRatAccuracyRef.size(); i++)
//				{
//					evalsToFile << inlRatAccuracyRef[i].first << " " << inlRatAccuracyRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else if(flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if(err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if(err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if(err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//		if((imgidx1 < 0) || (imgidx1 > ((int)filenamesl.size() - 2)) || (imgidx2 < 0) || (imgidx2 > ((int)filenamesl.size() - 1)))
//		{
//			cout << "Homography image indexes out of range! Exiting." << endl;
//			exit(0);
//		}
//
//		if(imgidx1 == 0)
//		{
//			H = Hs[imgidx2-1];
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesl[imgidx2],CV_LOAD_IMAGE_GRAYSCALE);
//		}
//		else
//		{
//			H = Hs[imgidx2 - 1] * Hs[imgidx1 - 1].inv();//H = (Hs[imgidx2 - 1].inv() * Hs[imgidx1 - 1]).inv();
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[imgidx1],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesl[imgidx2],CV_LOAD_IMAGE_GRAYSCALE);
//		}
//
//		if(!matcherType.compare("CASCHASH"))
//		{
//			mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[imgidx1] + "_" + filenamesl[imgidx2]));
//		}
//		else if(!matcherType.compare("GMBSOF"))
//		{
//			mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[imgidx1] + "_" + filenamesl[imgidx2], 0.22));
//		}
//		else if(!matcherType.compare("HIRCLUIDX"))
//		{
//			mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[imgidx1] + "_" + filenamesl[imgidx2], useRatioTest));
//		}
//		else if(!matcherType.compare("HIRKMEANS"))
//		{
//			mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[imgidx1] + "_" + filenamesl[imgidx2], useRatioTest));
//		}
//		else if(!matcherType.compare("VFCKNN"))
//		{
//			mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[imgidx1] + "_" + filenamesl[imgidx2], 2, useRatioTest));
//		}
//		else if(!matcherType.compare("LINEAR"))
//		{
//			mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[imgidx1] + "_" + filenamesl[imgidx2], useRatioTest));
//		}
//		else if(!matcherType.compare("LSHIDX"))
//		{
//			mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[imgidx1] + "_" + filenamesl[imgidx2], useRatioTest));
//		}
//		else if(!matcherType.compare("RANDKDTREE"))
//		{
//			mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[imgidx1] + "_" + filenamesl[imgidx2], useRatioTest));
//		}
//		else
//		{
//			cout << "No valid matcher specified! Exiting." << endl;
//			exit(1);
//		}
//
//		if(mymatcher->specialGMbSOFtest)
//		{
//			cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//			exit(0);
//		}
//
//		for(size_t i = 0; i < inlierRatios.size(); i++)
//		{
//			err = mymatcher->performMatching(inlierRatios[i]);
//			if(err)
//			{
//				if(err == -2)
//				{
//					inlRatPrecision.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatRecall.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatFallOut.push_back(make_pair(mymatcher->inlRatioL, 1.0));
//					inlRatAccuracy.push_back(make_pair(mymatcher->inlRatioL, 0));
//					continue;
//				}
//				else
//				{
//					continue;
//				}
//			}
//			inlRatPrecision.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.ppv));
//			inlRatRecall.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.tpr));
//			inlRatFallOut.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.fpr));
//			inlRatAccuracy.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpm.acc));
//
//			if(refine)
//			{
//				err = mymatcher->refineMatches();
//				if(err)
//				{
//					inlRatPrecisionRef.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatRecallRef.push_back(make_pair(mymatcher->inlRatioL, 0));
//					inlRatFallOutRef.push_back(make_pair(mymatcher->inlRatioL, 1.0));
//					inlRatAccuracyRef.push_back(make_pair(mymatcher->inlRatioL, 0));
//				}
//				else
//				{
//					inlRatPrecisionRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.ppv));
//					inlRatRecallRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.tpr));
//					inlRatFallOutRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.fpr));
//					inlRatAccuracyRef.push_back(make_pair(mymatcher->inlRatioL, mymatcher->qpr.acc));
//				}
//			}
//
//			if(showResult >= 0)
//			{
//				if(storeImgResPath.empty())
//				{
//					mymatcher->showMatches(showResult);
//				}
//				else
//				{
//					if(dirExists(storeImgResPath)) //Check if output directory existis
//					{
//						outpath = storeImgResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "img_inlRatio_H_idx" + std::to_string((ULONGLONG)imgidx1) + "-" + std::to_string((ULONGLONG)imgidx2) 
//								+ "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//				}
//			}
//			if((showRefinedResult >= 0) && refine && !err)
//			{
//				if(storeRefResPath.empty())
//				{
//					mymatcher->showMatches(showRefinedResult);
//				}
//				else
//				{
//					if(dirExists(storeRefResPath)) //Check if output directory existis
//					{
//						outpath = storeRefResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "imgRef_inlRatio_H_idx" + std::to_string((ULONGLONG)imgidx1) + "-" + std::to_string((ULONGLONG)imgidx2) + 
//								"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//				}
//			}
//
//		}
//
//		if(inlRatPrecision.empty())
//		{
//			cout << "Ground truth generation or algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		sort(inlRatPrecision.begin(), inlRatPrecision.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		sort(inlRatRecall.begin(), inlRatRecall.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		sort(inlRatFallOut.begin(), inlRatFallOut.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		sort(inlRatAccuracy.begin(), inlRatAccuracy.end(),
//			[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//
//		if(refine)
//		{
//			if(inlRatPrecisionRef.empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			sort(inlRatPrecisionRef.begin(), inlRatPrecisionRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//			sort(inlRatRecallRef.begin(), inlRatRecallRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//			sort(inlRatFallOutRef.begin(), inlRatFallOutRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//			sort(inlRatAccuracyRef.begin(), inlRatAccuracyRef.end(),
//				[](pair<double,double> first, pair<double,double> second){return first.first < second.first;});
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_inlRatio_H_inliersGT_precision_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			for(size_t i = 0; i < inlRatPrecision.size(); i++)
//			{
//				evalsToFile << inlRatPrecision[i].first << " " << inlRatPrecision[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_H_inliersGT_recall_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			for(size_t i = 0; i < inlRatRecall.size(); i++)
//			{
//				evalsToFile << inlRatRecall[i].first << " " << inlRatRecall[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_H_inliersGT_fpr_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			for(size_t i = 0; i < inlRatFallOut.size(); i++)
//			{
//				evalsToFile << inlRatFallOut[i].first << " " << inlRatFallOut[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_H_inliersGT_acc_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			for(size_t i = 0; i < inlRatAccuracy.size(); i++)
//			{
//				evalsToFile << inlRatAccuracy[i].first << " " << inlRatAccuracy[i].second << endl;
//			}
//			evalsToFile.close();
//		}
//		if(refine)
//		{
//			{
//				outfilename = "tex_inlRatio_H_inliersGT_precisionRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				for(size_t i = 0; i < inlRatPrecisionRef.size(); i++)
//				{
//					evalsToFile << inlRatPrecisionRef[i].first << " " << inlRatPrecisionRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_H_inliersGT_recallRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				for(size_t i = 0; i < inlRatRecallRef.size(); i++)
//				{
//					evalsToFile << inlRatRecallRef[i].first << " " << inlRatRecallRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_H_inliersGT_fprRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				for(size_t i = 0; i < inlRatFallOutRef.size(); i++)
//				{
//					evalsToFile << inlRatFallOutRef[i].first << " " << inlRatFallOutRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_H_inliersGT_accRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				for(size_t i = 0; i < inlRatAccuracyRef.size(); i++)
//				{
//					evalsToFile << inlRatAccuracyRef[i].first << " " << inlRatAccuracyRef[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//	return 0;
//}
//
///* Starts the evaluation of precision, recall, fall-out and accuracy for different matching algorithms and szenes
// * on all input pairs.
// *
// * string imgsPath				Input  -> Path which includes both left and right images
// * string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
// * int flowDispH				Input  -> Indicates which type of ground truth data is used:
// *										  0: flow files from KITTI database
// *										  1: disparity files from KITTI database
// *										  2: homography files (Please note that a homography always relates
// *											 to the first image (e.g. 1->2, 1->3, ...))
// * string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images 
// *									      (after prefix only comes the image number)
// * string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
// *									      (after prefix only comes the image number). For testing with homographies,
// *										  this string can be empty.
// * string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
// *									      (after prefix only comes the image number)
// * string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
// *										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
// *										  detectors are possible.
// * string descriptorExtractor	Input  -> The used descriptor extractor. Possible inputs should only be FREAK
// *										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
// * string matcherType			Input  -> The matcher type under test. Possible inputs are:
// *											CASCHASH: Cascade Hashing matcher
// *											GEOMAWARE: Geometry-aware Feature matching algorithm
// *											GMBSOF: Guided matching based on statistical optical flow
// *											HIRCLUIDX: Hirarchical Clustering Index Matching from the FLANN library
// *											HIRKMEANS: hierarchical k-means tree matcher from the FLANN library
// *											VFCKNN: Vector field consensus (VFC) algorithm with k nearest neighbor 
// *													matches provided from the Hirarchical Clustering Index Matching 
// *													algorithm from the FLANN library
// *											LIBVISO: matcher from the libviso2 library
// *											LINEAR: linear Matching algorithm (Brute force) from the FLANN library
// *											LSHIDX: LSH Index Matching algorithm from the FLANN library
// *											RANDKDTREE: randomized KD-trees matcher from the FLANN library
// * bool useRatioTest			Input  -> Specifies if a ratio test should be performed on the results of a matching
// *										  algorithm. The ratio test is only possible for the following algorithms:
// *										  HIRCLUIDX, HIRKMEANS, LINEAR, LSHIDX, RANDKDTREE
// * string storeResultPath		Input  -> Path were the resulting measurements should be stored
// * bool refine					Input  -> If true [DEFAULT = false], the results from the matching algorithm are
// *										  refined using VFC.
// * double inlRatio				Input  -> The inlier ratio which should be generated using the ground truth data
// *										  [DEFAULT = 1.0]
// * int showResult				Input  -> If >= 0, the result from the matching algorithm is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * int showRefinedResult		Input  -> If >= 0, the result after refinement with VFC is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * string storeImgResPath		Input  -> Optional path for storing the resulting matches drawn into the images,
// *										  where the options of which results should be drawn are specified in
// *										  "showResult". If this path is set, the images are NOT displayed.
// * string storeRefResPath		Input  -> Optional path for storing the resulting matches after refinement 
// *										  drawn into the images, where the options of which results should be 
// *										  drawn are specified in "showRefinedResult". If this path is set, the images 
// *										  are NOT displayed.
// * 
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Failed
// */
//int startQualPMeasurement(std::string imgsPath, std::string flowDispHPath, int flowDispH, 
//						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//						 std::string featureDetector, std::string descriptorExtractor, std::string matcherType,
//						 bool useRatioTest, std::string storeResultPath, bool refine, double inlRatio, 
//						 int showResult, int showRefinedResult, std::string storeImgResPath, 
//						 std::string storeRefResPath)
//{
//	int err;
//	cv::Mat src[2];
//	vector<double> precision, recall, fallOut, accuracy;
//	vector<double> precisionRef, recallRef, fallOutRef, accuracyRef;
//	qualityParm1 precisionStat, recallStat, fallOutStat, accuracyStat;
//	qualityParm1 precisionRefStat, recallRefStat, fallOutRefStat, accuracyRefStat;
//	std::auto_ptr<baseMatcher> mymatcher;
//	string outpath, outfilename;
//
//	if(!storeImgResPath.empty() && (showResult == -1))
//	{
//		cout << "If you want to store the resulting images you must specify the showResult flag! Exiting." << endl;
//		exit(0);
//	}
//	if(!storeRefResPath.empty() && (showRefinedResult == -1))
//	{
//		cout << "If you want to store the resulting images with refined matches you must specify the showRefinedResult flag! Exiting." << endl;
//		exit(0);
//	}
//	if(!refine && (showRefinedResult >= 0))
//	{
//		cout << "Cant show refined results if refinement is disabled" << endl;
//	}
//
//	//Generate cumulative distribution bins
//	vector<pair<double,double>> cumuldistPrecision, cumuldistRecall, cumuldistFallOut, cumuldistAccuracy;
//	vector<pair<double,double>> cumuldistPrecisionRef, cumuldistRecallRef, cumuldistFallOutRef, cumuldistAccuracyRef;
//	vector<pair<double,double>> cumuldistPrecisionBins, cumuldistRecallBins, cumuldistFallOutBins, cumuldistAccuracyBins;
//	vector<pair<double,double>> cumuldistPrecisionBinsRef, cumuldistRecallBinsRef, cumuldistFallOutBinsRef, cumuldistAccuracyBinsRef;
//	double distrval = 0.02;
//	double bindist = distrval * 2.0;
//	for(int i = 0; i < 25; i++)
//	{
//		cumuldistPrecision.push_back(make_pair(distrval + (double)i * bindist, -1.0));
//		cumuldistRecall.push_back(make_pair(distrval + (double)i * bindist, -1.0));
//		cumuldistFallOut.push_back(make_pair(distrval + (double)i * bindist, -1.0));
//		cumuldistAccuracy.push_back(make_pair(distrval + (double)i * bindist, -1.0));
//		cumuldistPrecisionRef.push_back(make_pair(distrval + (double)i * bindist, -1.0));
//		cumuldistRecallRef.push_back(make_pair(distrval + (double)i * bindist, -1.0));
//		cumuldistFallOutRef.push_back(make_pair(distrval + (double)i * bindist, -1.0));
//		cumuldistAccuracyRef.push_back(make_pair(distrval + (double)i * bindist, -1.0));
//	}
//
//	if(flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int i = 0; i < (int)filenamesl.size(); i++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageFlowFile(flowDispHPath, filenamesflow[i], &flowimg);
//			if(err)
//			{
//				cout << "Could not open flow file with index " << i << endl;
//				continue;
//			}
//			if(!matcherType.compare("CASCHASH"))
//			{
//				mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i]));
//			}
//			else if(!matcherType.compare("GMBSOF"))
//			{
//				mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], 0.41));
//			}
//			else if(!matcherType.compare("HIRCLUIDX"))
//			{
//				mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("HIRKMEANS"))
//			{
//				mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("VFCKNN"))
//			{
//				mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], 2, useRatioTest));
//			}
//			else if(!matcherType.compare("LINEAR"))
//			{
//				mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("LSHIDX"))
//			{
//				mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("RANDKDTREE"))
//			{
//				mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else
//			{
//				cout << "No valid matcher specified! Exiting." << endl;
//				exit(1);
//			}
//
//			if(mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			err = mymatcher->performMatching(inlRatio);
//			if(err)
//			{
//				if( err == -2)
//				{
//					precision.push_back(0);
//					recall.push_back(0);
//					fallOut.push_back(1.0);
//					accuracy.push_back(0);
//				}
//				continue;
//			}
//			precision.push_back(mymatcher->qpm.ppv);
//			recall.push_back(mymatcher->qpm.tpr);
//			fallOut.push_back(mymatcher->qpm.fpr);
//			accuracy.push_back(mymatcher->qpm.acc);
//
//			
//			if(refine)
//			{
//				err = mymatcher->refineMatches();
//				if(!err)
//				{
//					precisionRef.push_back(mymatcher->qpr.ppv);
//					recallRef.push_back(mymatcher->qpr.tpr);
//					fallOutRef.push_back(mymatcher->qpr.fpr);
//					accuracyRef.push_back(mymatcher->qpr.acc);
//				}
//				else
//				{
//					precisionRef.push_back(0);
//					recallRef.push_back(0);
//					fallOutRef.push_back(1.0);
//					accuracyRef.push_back(0);
//				}
//			}
//
//			if(showResult >= 0)
//			{
//				if(storeImgResPath.empty())
//				{
//					mymatcher->showMatches(showResult);
//				}
//				else
//				{
//					if(dirExists(storeImgResPath)) //Check if output directory existis
//					{
//						outpath = storeImgResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "img_qualP_flow_idx" + std::to_string((ULONGLONG)i) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//				}
//			}
//			if((showRefinedResult >= 0) && refine && !err)
//			{
//				if(storeRefResPath.empty())
//				{
//					mymatcher->showMatches(showRefinedResult);
//				}
//				else
//				{
//					if(dirExists(storeRefResPath)) //Check if output directory existis
//					{
//						outpath = storeRefResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "imgRef_qualP_flow_idx" + std::to_string((ULONGLONG)i) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//				}
//			}
//		}
//		if(precision.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//		
//		sort(precision.begin(), precision.end(),
//					[](double first, double second){return first < second;});
//		sort(recall.begin(), recall.end(),
//					[](double first, double second){return first < second;});
//		sort(fallOut.begin(), fallOut.end(),
//					[](double first, double second){return first < second;});
//		sort(accuracy.begin(), accuracy.end(),
//					[](double first, double second){return first < second;});
//
//		getSmoothComulDist(precision, cumuldistPrecision, cumuldistPrecisionBins, distrval, bindist);
//		getSmoothComulDist(recall, cumuldistRecall, cumuldistRecallBins, distrval, bindist);
//		getSmoothComulDist(fallOut, cumuldistFallOut, cumuldistFallOutBins, distrval, bindist);
//		getSmoothComulDist(accuracy, cumuldistAccuracy, cumuldistAccuracyBins, distrval, bindist);
//		
//		getStatisticfromVec2(precision, &precisionStat, false);
//		getStatisticfromVec2(recall, &recallStat, false);
//		getStatisticfromVec2(fallOut, &fallOutStat, false);
//		getStatisticfromVec2(accuracy, &accuracyStat, false);
//
//		if(refine)
//		{
//			if(precisionRef.empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			sort(precisionRef.begin(), precisionRef.end(),
//					[](double first, double second){return first < second;});
//			sort(recallRef.begin(), recallRef.end(),
//						[](double first, double second){return first < second;});
//			sort(fallOutRef.begin(), fallOutRef.end(),
//						[](double first, double second){return first < second;});
//			sort(accuracyRef.begin(), accuracyRef.end(),
//						[](double first, double second){return first < second;});
//
//			getSmoothComulDist(precisionRef, cumuldistPrecisionRef, cumuldistPrecisionBinsRef, distrval, bindist);
//			getSmoothComulDist(recallRef, cumuldistRecallRef, cumuldistRecallBinsRef, distrval, bindist);
//			getSmoothComulDist(fallOutRef, cumuldistFallOutRef, cumuldistFallOutBinsRef, distrval, bindist);
//			getSmoothComulDist(accuracyRef, cumuldistAccuracyRef, cumuldistAccuracyBinsRef, distrval, bindist);
//
//			getStatisticfromVec2(precisionRef, &precisionRefStat, false);
//			getStatisticfromVec2(recallRef, &recallRefStat, false);
//			getStatisticfromVec2(fallOutRef, &fallOutRefStat, false);
//			getStatisticfromVec2(accuracyRef, &accuracyRefStat, false);
//		}
//
//		
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		
//		//Output for smoothed data
//		{
//			outfilename = "tex_qualP_flow_preci_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistPrecision.size(); i++)
//			{
//				evalsToFile << cumuldistPrecision[i].second << " " << cumuldistPrecision[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_flow_recall_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistRecall.size(); i++)
//			{
//				evalsToFile << cumuldistRecall[i].second << " " << cumuldistRecall[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_flow_fpr_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistFallOut.size(); i++)
//			{
//				evalsToFile << cumuldistFallOut[i].second << " " << cumuldistFallOut[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_flow_acc_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistAccuracy.size(); i++)
//			{
//				evalsToFile << cumuldistAccuracy[i].second << " " << cumuldistAccuracy[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//
//		//Output for original data
//		{
//			outfilename = "tex_qualP_flow_preciBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistPrecisionBins.size(); i++)
//			{
//				evalsToFile << cumuldistPrecisionBins[i].second << " " << cumuldistPrecisionBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_flow_recallBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistRecallBins.size(); i++)
//			{
//				evalsToFile << cumuldistRecallBins[i].second << " " << cumuldistRecallBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_flow_fprBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistFallOutBins.size(); i++)
//			{
//				evalsToFile << cumuldistFallOutBins[i].second << " " << cumuldistFallOutBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_flow_accBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistAccuracyBins.size(); i++)
//			{
//				evalsToFile << cumuldistAccuracyBins[i].second << " " << cumuldistAccuracyBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//
//		//Output statistics mean & variance
//		{
//			outfilename = "tex_qualPStatsMV_flow_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//						&precisionStat, precision, false);
//		}
//		{
//			outfilename = "tex_qualPStatsMV_flow_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//						&recallStat, recall, false);
//		}
//		{
//			outfilename = "tex_qualPStatsMV_flow_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//						&fallOutStat, fallOut, false);
//		}
//		{
//			outfilename = "tex_qualPStatsMV_flow_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//						&accuracyStat, accuracy, false);
//		}
//
//		//Output statistics box plot (median, stdmax, stdmin, max, min)
//		{
//			outfilename = "tex_qualPStatsBP_flow_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//						&precisionStat, precision, true);
//		}
//		{
//			outfilename = "tex_qualPStatsBP_flow_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//						&recallStat, recall, true);
//		}
//		{
//			outfilename = "tex_qualPStatsBP_flow_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//						&fallOutStat, fallOut, true);
//		}
//		{
//			outfilename = "tex_qualPStatsBP_flow_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//						&accuracyStat, accuracy, true);
//		}
//
//		if(refine)
//		{
//			//Output for smoothed data
//			{
//				outfilename = "tex_qualPRef_flow_preci_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistPrecisionRef.size(); i++)
//				{
//					evalsToFile << cumuldistPrecisionRef[i].second << " " << cumuldistPrecisionRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_flow_recall_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistRecallRef.size(); i++)
//				{
//					evalsToFile << cumuldistRecallRef[i].second << " " << cumuldistRecallRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_flow_fpr_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistFallOutRef.size(); i++)
//				{
//					evalsToFile << cumuldistFallOutRef[i].second << " " << cumuldistFallOutRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_flow_acc_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistAccuracyRef.size(); i++)
//				{
//					evalsToFile << cumuldistAccuracyRef[i].second << " " << cumuldistAccuracyRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//
//			//Output for original data
//			{
//				outfilename = "tex_qualPRef_flow_preciBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistPrecisionBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistPrecisionBinsRef[i].second << " " << cumuldistPrecisionBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_flow_recallBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistRecallBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistRecallBinsRef[i].second << " " << cumuldistRecallBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_flow_fprBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistFallOutBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistFallOutBinsRef[i].second << " " << cumuldistFallOutBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_flow_accBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistAccuracyBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistAccuracyBinsRef[i].second << " " << cumuldistAccuracyBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//
//			//Output statistics mean & variance
//			{
//				outfilename = "tex_qualPRefStatsMV_flow_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//							&precisionRefStat, precisionRef, false);
//			}
//			{
//				outfilename = "tex_qualPRefStatsMV_flow_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//							&recallRefStat, recallRef, false);
//			}
//			{
//				outfilename = "tex_qualPRefStatsMV_flow_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//							&fallOutRefStat, fallOutRef, false);
//			}
//			{
//				outfilename = "tex_qualPRefStatsMV_flow_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//							&accuracyRefStat, accuracyRef, false);
//			}
//
//			//Output statistics box plot (median, stdmax, stdmin, max, min)
//			{
//				outfilename = "tex_qualPRefStatsBP_flow_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//							&precisionRefStat, precisionRef, true);
//			}
//			{
//				outfilename = "tex_qualPRefStatsBP_flow_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//							&recallRefStat, recallRef, true);
//			}
//			{
//				outfilename = "tex_qualPRefStatsBP_flow_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//							&fallOutRefStat, fallOutRef, true);
//			}
//			{
//				outfilename = "tex_qualPRefStatsBP_flow_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//							&accuracyRefStat, accuracyRef, true);
//			}
//		}
//	}
//	else if(flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int i = 0; i < (int)filenamesl.size(); i++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageDisparityFile(flowDispHPath, filenamesflow[i], &flowimg);
//			if(err)
//			{
//				cout << "Could not open disparity file with index " << i << endl;
//				continue;
//			}
//			if(!matcherType.compare("CASCHASH"))
//			{
//				mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i]));
//			}
//			else if(!matcherType.compare("GMBSOF"))
//			{
//				mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], 0.26));
//			}
//			else if(!matcherType.compare("HIRCLUIDX"))
//			{
//				mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("HIRKMEANS"))
//			{
//				mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("VFCKNN"))
//			{
//				mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], 2, useRatioTest));
//			}
//			else if(!matcherType.compare("LINEAR"))
//			{
//				mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("LSHIDX"))
//			{
//				mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else if(!matcherType.compare("RANDKDTREE"))
//			{
//				mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], useRatioTest));
//			}
//			else
//			{
//				cout << "No valid matcher specified! Exiting." << endl;
//				exit(1);
//			}
//
//			if(mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			err = mymatcher->performMatching(inlRatio);
//			if(err)
//			{
//				if( err == -2)
//				{
//					precision.push_back(0);
//					recall.push_back(0);
//					fallOut.push_back(1.0);
//					accuracy.push_back(0);
//				}
//				continue;
//			}
//			precision.push_back(mymatcher->qpm.ppv);
//			recall.push_back(mymatcher->qpm.tpr);
//			fallOut.push_back(mymatcher->qpm.fpr);
//			accuracy.push_back(mymatcher->qpm.acc);
//
//			
//			if(refine)
//			{
//				err = mymatcher->refineMatches();
//				if(!err)
//				{
//					precisionRef.push_back(mymatcher->qpr.ppv);
//					recallRef.push_back(mymatcher->qpr.tpr);
//					fallOutRef.push_back(mymatcher->qpr.fpr);
//					accuracyRef.push_back(mymatcher->qpr.acc);
//				}
//				else
//				{
//					precisionRef.push_back(0);
//					recallRef.push_back(0);
//					fallOutRef.push_back(1.0);
//					accuracyRef.push_back(0);
//				}
//			}
//
//			if(showResult >= 0)
//			{
//				if(storeImgResPath.empty())
//				{
//					mymatcher->showMatches(showResult);
//				}
//				else
//				{
//					if(dirExists(storeImgResPath)) //Check if output directory existis
//					{
//						outpath = storeImgResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "img_qualP_disp_idx" + std::to_string((ULONGLONG)i) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//				}
//			}
//			if((showRefinedResult >= 0) && refine && !err)
//			{
//				if(storeRefResPath.empty())
//				{
//					mymatcher->showMatches(showRefinedResult);
//				}
//				else
//				{
//					if(dirExists(storeRefResPath)) //Check if output directory existis
//					{
//						outpath = storeRefResPath;
//					}
//					else
//					{
//						outpath = imgsPath + "\\evalImgs";
//						if(!dirExists(outpath))
//							_mkdir(outpath.c_str());
//					}
//					outfilename = "imgRef_qualP_disp_idx" + std::to_string((ULONGLONG)i) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//					mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//				}
//			}
//		}
//		if(precision.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//		
//		sort(precision.begin(), precision.end(),
//					[](double first, double second){return first < second;});
//		sort(recall.begin(), recall.end(),
//					[](double first, double second){return first < second;});
//		sort(fallOut.begin(), fallOut.end(),
//					[](double first, double second){return first < second;});
//		sort(accuracy.begin(), accuracy.end(),
//					[](double first, double second){return first < second;});
//
//		getSmoothComulDist(precision, cumuldistPrecision, cumuldistPrecisionBins, distrval, bindist);
//		getSmoothComulDist(recall, cumuldistRecall, cumuldistRecallBins, distrval, bindist);
//		getSmoothComulDist(fallOut, cumuldistFallOut, cumuldistFallOutBins, distrval, bindist);
//		getSmoothComulDist(accuracy, cumuldistAccuracy, cumuldistAccuracyBins, distrval, bindist);
//
//		getStatisticfromVec2(precision, &precisionStat, false);
//		getStatisticfromVec2(recall, &recallStat, false);
//		getStatisticfromVec2(fallOut, &fallOutStat, false);
//		getStatisticfromVec2(accuracy, &accuracyStat, false);
//
//		if(refine)
//		{
//			if(precisionRef.empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			sort(precisionRef.begin(), precisionRef.end(),
//					[](double first, double second){return first < second;});
//			sort(recallRef.begin(), recallRef.end(),
//						[](double first, double second){return first < second;});
//			sort(fallOutRef.begin(), fallOutRef.end(),
//						[](double first, double second){return first < second;});
//			sort(accuracyRef.begin(), accuracyRef.end(),
//						[](double first, double second){return first < second;});
//
//			getSmoothComulDist(precisionRef, cumuldistPrecisionRef, cumuldistPrecisionBinsRef, distrval, bindist);
//			getSmoothComulDist(recallRef, cumuldistRecallRef, cumuldistRecallBinsRef, distrval, bindist);
//			getSmoothComulDist(fallOutRef, cumuldistFallOutRef, cumuldistFallOutBinsRef, distrval, bindist);
//			getSmoothComulDist(accuracyRef, cumuldistAccuracyRef, cumuldistAccuracyBinsRef, distrval, bindist);
//
//			getStatisticfromVec2(precisionRef, &precisionRefStat, false);
//			getStatisticfromVec2(recallRef, &recallRefStat, false);
//			getStatisticfromVec2(fallOutRef, &fallOutRefStat, false);
//			getStatisticfromVec2(accuracyRef, &accuracyRefStat, false);
//		}
//
//		
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		
//		//Output for smoothed data
//		{
//			outfilename = "tex_qualP_disp_preci_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistPrecision.size(); i++)
//			{
//				evalsToFile << cumuldistPrecision[i].second << " " << cumuldistPrecision[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_disp_recall_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistRecall.size(); i++)
//			{
//				evalsToFile << cumuldistRecall[i].second << " " << cumuldistRecall[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_disp_fpr_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistFallOut.size(); i++)
//			{
//				evalsToFile << cumuldistFallOut[i].second << " " << cumuldistFallOut[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_disp_acc_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistAccuracy.size(); i++)
//			{
//				evalsToFile << cumuldistAccuracy[i].second << " " << cumuldistAccuracy[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//
//		//Output for original data
//		{
//			outfilename = "tex_qualP_disp_preciBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistPrecisionBins.size(); i++)
//			{
//				evalsToFile << cumuldistPrecisionBins[i].second << " " << cumuldistPrecisionBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_disp_recallBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistRecallBins.size(); i++)
//			{
//				evalsToFile << cumuldistRecallBins[i].second << " " << cumuldistRecallBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_disp_fprBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistFallOutBins.size(); i++)
//			{
//				evalsToFile << cumuldistFallOutBins[i].second << " " << cumuldistFallOutBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_disp_accBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistAccuracyBins.size(); i++)
//			{
//				evalsToFile << cumuldistAccuracyBins[i].second << " " << cumuldistAccuracyBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//
//		//Output statistics mean & variance
//		{
//			outfilename = "tex_qualPStatsMV_disp_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//						&precisionStat, precision, false);
//		}
//		{
//			outfilename = "tex_qualPStatsMV_disp_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//						&recallStat, recall, false);
//		}
//		{
//			outfilename = "tex_qualPStatsMV_disp_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//						&fallOutStat, fallOut, false);
//		}
//		{
//			outfilename = "tex_qualPStatsMV_disp_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//						&accuracyStat, accuracy, false);
//		}
//
//		//Output statistics box plot (median, stdmax, stdmin, max, min)
//		{
//			outfilename = "tex_qualPStatsBP_disp_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//						&precisionStat, precision, true);
//		}
//		{
//			outfilename = "tex_qualPStatsBP_disp_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//						&recallStat, recall, true);
//		}
//		{
//			outfilename = "tex_qualPStatsBP_disp_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//						&fallOutStat, fallOut, true);
//		}
//		{
//			outfilename = "tex_qualPStatsBP_disp_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//						&accuracyStat, accuracy, true);
//		}
//
//		if(refine)
//		{
//			//Output for smoothed data
//			{
//				outfilename = "tex_qualPRef_disp_preci_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistPrecisionRef.size(); i++)
//				{
//					evalsToFile << cumuldistPrecisionRef[i].second << " " << cumuldistPrecisionRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_disp_recall_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistRecallRef.size(); i++)
//				{
//					evalsToFile << cumuldistRecallRef[i].second << " " << cumuldistRecallRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_disp_fpr_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistFallOutRef.size(); i++)
//				{
//					evalsToFile << cumuldistFallOutRef[i].second << " " << cumuldistFallOutRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_disp_acc_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistAccuracyRef.size(); i++)
//				{
//					evalsToFile << cumuldistAccuracyRef[i].second << " " << cumuldistAccuracyRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//
//			//Output for original data
//			{
//				outfilename = "tex_qualPRef_disp_preciBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistPrecisionBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistPrecisionBinsRef[i].second << " " << cumuldistPrecisionBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_disp_recallBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistRecallBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistRecallBinsRef[i].second << " " << cumuldistRecallBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_disp_fprBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistFallOutBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistFallOutBinsRef[i].second << " " << cumuldistFallOutBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_disp_accBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistAccuracyBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistAccuracyBinsRef[i].second << " " << cumuldistAccuracyBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//
//			//Output statistics mean & variance
//			{
//				outfilename = "tex_qualPRefStatsMV_disp_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//							&precisionRefStat, precisionRef, false);
//			}
//			{
//				outfilename = "tex_qualPRefStatsMV_disp_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//							&recallRefStat, recallRef, false);
//			}
//			{
//				outfilename = "tex_qualPRefStatsMV_disp_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//							&fallOutRefStat, fallOutRef, false);
//			}
//			{
//				outfilename = "tex_qualPRefStatsMV_disp_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//							&accuracyRefStat, accuracyRef, false);
//			}
//
//			//Output statistics box plot (median, stdmax, stdmin, max, min)
//			{
//				outfilename = "tex_qualPRefStatsBP_disp_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//							&precisionRefStat, precisionRef, true);
//			}
//			{
//				outfilename = "tex_qualPRefStatsBP_disp_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//							&recallRefStat, recallRef, true);
//			}
//			{
//				outfilename = "tex_qualPRefStatsBP_disp_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//							&fallOutRefStat, fallOutRef, true);
//			}
//			{
//				outfilename = "tex_qualPRefStatsBP_disp_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//							&accuracyRefStat, accuracyRef, true);
//			}
//		}
//	}
//	else if(flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		//cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if(err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if(err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if(err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		if(fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//			for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//				
//				if(!matcherType.compare("CASCHASH"))
//				{
//					mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("GMBSOF"))
//				{
//					mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 0.22));
//				}
//				else if(!matcherType.compare("HIRCLUIDX"))
//				{
//					mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("HIRKMEANS"))
//				{
//					mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("VFCKNN"))
//				{
//					mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 2, useRatioTest));
//				}
//				else if(!matcherType.compare("LINEAR"))
//				{
//					mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("LSHIDX"))
//				{
//					mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("RANDKDTREE"))
//				{
//					mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else
//				{
//					cout << "No valid matcher specified! Exiting." << endl;
//					exit(1);
//				}
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				err = mymatcher->performMatching(inlRatio);
//				if(err)
//				{
//					if( err == -2)
//					{
//						precision.push_back(0);
//						recall.push_back(0);
//						fallOut.push_back(1.0);
//						accuracy.push_back(0);
//					}
//					continue;
//				}
//				precision.push_back(mymatcher->qpm.ppv);
//				recall.push_back(mymatcher->qpm.tpr);
//				fallOut.push_back(mymatcher->qpm.fpr);
//				accuracy.push_back(mymatcher->qpm.acc);
//
//			
//				if(refine)
//				{
//					err = mymatcher->refineMatches();
//					if(!err)
//					{
//						precisionRef.push_back(mymatcher->qpr.ppv);
//						recallRef.push_back(mymatcher->qpr.tpr);
//						fallOutRef.push_back(mymatcher->qpr.fpr);
//						accuracyRef.push_back(mymatcher->qpr.acc);
//					}
//					else
//					{
//						precisionRef.push_back(0);
//						recallRef.push_back(0);
//						fallOutRef.push_back(1.0);
//						accuracyRef.push_back(0);
//					}
//				}
//
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_qualP_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//				if((showRefinedResult >= 0) && refine && !err)
//				{
//					if(storeRefResPath.empty())
//					{
//						mymatcher->showMatches(showRefinedResult);
//					}
//					else
//					{
//						if(dirExists(storeRefResPath)) //Check if output directory existis
//						{
//							outpath = storeRefResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "imgRef_qualP_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//					}
//				}
//			}
//			//Generate new homographys to evaluate all other possible configurations of the images to each other
//			for(int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//			{
//				for(int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//				{
//					cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//					src[0] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx2 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//					
//					if(!matcherType.compare("CASCHASH"))
//					{
//						mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//					}
//					else if(!matcherType.compare("GMBSOF"))
//					{
//						mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], 0.22));
//					}
//					else if(!matcherType.compare("HIRCLUIDX"))
//					{
//						mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//					}
//					else if(!matcherType.compare("HIRKMEANS"))
//					{
//						mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//					}
//					else if(!matcherType.compare("VFCKNN"))
//					{
//						mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], 2, useRatioTest));
//					}
//					else if(!matcherType.compare("LINEAR"))
//					{
//						mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//					}
//					else if(!matcherType.compare("LSHIDX"))
//					{
//						mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//					}
//					else if(!matcherType.compare("RANDKDTREE"))
//					{
//						mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//					}
//					else
//					{
//						cout << "No valid matcher specified! Exiting." << endl;
//						exit(1);
//					}
//
//					if(mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					err = mymatcher->performMatching(inlRatio);
//					if(err)
//					{
//						if( err == -2)
//						{
//							precision.push_back(0);
//							recall.push_back(0);
//							fallOut.push_back(1.0);
//							accuracy.push_back(0);
//						}
//						continue;
//					}
//					precision.push_back(mymatcher->qpm.ppv);
//					recall.push_back(mymatcher->qpm.tpr);
//					fallOut.push_back(mymatcher->qpm.fpr);
//					accuracy.push_back(mymatcher->qpm.acc);
//
//			
//					if(refine)
//					{
//						err = mymatcher->refineMatches();
//						if(!err)
//						{
//							precisionRef.push_back(mymatcher->qpr.ppv);
//							recallRef.push_back(mymatcher->qpr.tpr);
//							fallOutRef.push_back(mymatcher->qpr.fpr);
//							accuracyRef.push_back(mymatcher->qpr.acc);
//						}
//						else
//						{
//							precisionRef.push_back(0);
//							recallRef.push_back(0);
//							fallOutRef.push_back(1.0);
//							accuracyRef.push_back(0);
//						}
//					}
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_qualP_H_idxs" + std::to_string((ULONGLONG)(idx1 + 1)) + "-" + std::to_string((ULONGLONG)(idx2 + 1)) + 
//										"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//										matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//					if((showRefinedResult >= 0) && refine && !err)
//					{
//						if(storeRefResPath.empty())
//						{
//							mymatcher->showMatches(showRefinedResult);
//						}
//						else
//						{
//							if(dirExists(storeRefResPath)) //Check if output directory existis
//							{
//								outpath = storeRefResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "imgRef_qualP_H_idxs" + std::to_string((ULONGLONG)(idx1 + 1)) + "-" + std::to_string((ULONGLONG)(idx2 + 1)) + 
//										"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//										matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//						}
//					}
//				}
//			}
//		}
//		else
//		{
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//			for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//				
//				if(!matcherType.compare("CASCHASH"))
//				{
//					mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("GMBSOF"))
//				{
//					mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 0.22));
//				}
//				else if(!matcherType.compare("HIRCLUIDX"))
//				{
//					mymatcher = std::auto_ptr<HirClustIdx_matcher>(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("HIRKMEANS"))
//				{
//					mymatcher = std::auto_ptr<HirarchKMeans_matcher>(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("VFCKNN"))
//				{
//					mymatcher = std::auto_ptr<VFCknn_matcher>(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 2, useRatioTest));
//				}
//				else if(!matcherType.compare("LINEAR"))
//				{
//					mymatcher = std::auto_ptr<Linear_matcher>(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("LSHIDX"))
//				{
//					mymatcher = std::auto_ptr<LSHidx_matcher>(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("RANDKDTREE"))
//				{
//					mymatcher = std::auto_ptr<RandKDTree_matcher>(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else
//				{
//					cout << "No valid matcher specified! Exiting." << endl;
//					exit(1);
//				}
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				err = mymatcher->performMatching(inlRatio);
//				if(err)
//				{
//					if( err == -2)
//					{
//						precision.push_back(0);
//						recall.push_back(0);
//						fallOut.push_back(1.0);
//						accuracy.push_back(0);
//					}
//					continue;
//				}
//				precision.push_back(mymatcher->qpm.ppv);
//				recall.push_back(mymatcher->qpm.tpr);
//				fallOut.push_back(mymatcher->qpm.fpr);
//				accuracy.push_back(mymatcher->qpm.acc);
//
//			
//				if(refine)
//				{
//					err = mymatcher->refineMatches();
//					if(!err)
//					{
//						precisionRef.push_back(mymatcher->qpr.ppv);
//						recallRef.push_back(mymatcher->qpr.tpr);
//						fallOutRef.push_back(mymatcher->qpr.fpr);
//						accuracyRef.push_back(mymatcher->qpr.acc);
//					}
//					else
//					{
//						precisionRef.push_back(0);
//						recallRef.push_back(0);
//						fallOutRef.push_back(1.0);
//						accuracyRef.push_back(0);
//					}
//				}
//
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_qualP_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//				if((showRefinedResult >= 0) && refine && !err)
//				{
//					if(storeRefResPath.empty())
//					{
//						mymatcher->showMatches(showRefinedResult);
//					}
//					else
//					{
//						if(dirExists(storeRefResPath)) //Check if output directory existis
//						{
//							outpath = storeRefResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "imgRef_qualP_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//					}
//				}
//			}
//		}
//
//		if(precision.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//		
//		sort(precision.begin(), precision.end(),
//					[](double first, double second){return first < second;});
//		sort(recall.begin(), recall.end(),
//					[](double first, double second){return first < second;});
//		sort(fallOut.begin(), fallOut.end(),
//					[](double first, double second){return first < second;});
//		sort(accuracy.begin(), accuracy.end(),
//					[](double first, double second){return first < second;});
//
//		getSmoothComulDist(precision, cumuldistPrecision, cumuldistPrecisionBins, distrval, bindist);
//		getSmoothComulDist(recall, cumuldistRecall, cumuldistRecallBins, distrval, bindist);
//		getSmoothComulDist(fallOut, cumuldistFallOut, cumuldistFallOutBins, distrval, bindist);
//		getSmoothComulDist(accuracy, cumuldistAccuracy, cumuldistAccuracyBins, distrval, bindist);
//
//		getStatisticfromVec2(precision, &precisionStat, false);
//		getStatisticfromVec2(recall, &recallStat, false);
//		getStatisticfromVec2(fallOut, &fallOutStat, false);
//		getStatisticfromVec2(accuracy, &accuracyStat, false);
//
//		if(refine)
//		{
//			if(precisionRef.empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			sort(precisionRef.begin(), precisionRef.end(),
//					[](double first, double second){return first < second;});
//			sort(recallRef.begin(), recallRef.end(),
//						[](double first, double second){return first < second;});
//			sort(fallOutRef.begin(), fallOutRef.end(),
//						[](double first, double second){return first < second;});
//			sort(accuracyRef.begin(), accuracyRef.end(),
//						[](double first, double second){return first < second;});
//
//			getSmoothComulDist(precisionRef, cumuldistPrecisionRef, cumuldistPrecisionBinsRef, distrval, bindist);
//			getSmoothComulDist(recallRef, cumuldistRecallRef, cumuldistRecallBinsRef, distrval, bindist);
//			getSmoothComulDist(fallOutRef, cumuldistFallOutRef, cumuldistFallOutBinsRef, distrval, bindist);
//			getSmoothComulDist(accuracyRef, cumuldistAccuracyRef, cumuldistAccuracyBinsRef, distrval, bindist);
//
//			getStatisticfromVec2(precisionRef, &precisionRefStat, false);
//			getStatisticfromVec2(recallRef, &recallRefStat, false);
//			getStatisticfromVec2(fallOutRef, &fallOutRefStat, false);
//			getStatisticfromVec2(accuracyRef, &accuracyRefStat, false);
//		}
//
//		
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		
//		//Output for smoothed data
//		{
//			outfilename = "tex_qualP_H_preci_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistPrecision.size(); i++)
//			{
//				evalsToFile << cumuldistPrecision[i].second << " " << cumuldistPrecision[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_H_recall_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistRecall.size(); i++)
//			{
//				evalsToFile << cumuldistRecall[i].second << " " << cumuldistRecall[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_H_fpr_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistFallOut.size(); i++)
//			{
//				evalsToFile << cumuldistFallOut[i].second << " " << cumuldistFallOut[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_H_acc_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//			for(size_t i = 0; i < cumuldistAccuracy.size(); i++)
//			{
//				evalsToFile << cumuldistAccuracy[i].second << " " << cumuldistAccuracy[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//
//		//Output for original data
//		{
//			outfilename = "tex_qualP_H_preciBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistPrecisionBins.size(); i++)
//			{
//				evalsToFile << cumuldistPrecisionBins[i].second << " " << cumuldistPrecisionBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_H_recallBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistRecallBins.size(); i++)
//			{
//				evalsToFile << cumuldistRecallBins[i].second << " " << cumuldistRecallBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_H_fprBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistFallOutBins.size(); i++)
//			{
//				evalsToFile << cumuldistFallOutBins[i].second << " " << cumuldistFallOutBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_qualP_H_accBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//			for(size_t i = 0; i < cumuldistAccuracyBins.size(); i++)
//			{
//				evalsToFile << cumuldistAccuracyBins[i].second << " " << cumuldistAccuracyBins[i].first << endl;
//			}
//			evalsToFile.close();
//		}
//
//		//Output statistics mean & variance
//		{
//			outfilename = "tex_qualPStatsMV_H_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//						&precisionStat, precision, false);
//		}
//		{
//			outfilename = "tex_qualPStatsMV_H_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//						&recallStat, recall, false);
//		}
//		{
//			outfilename = "tex_qualPStatsMV_H_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//						&fallOutStat, fallOut, false);
//		}
//		{
//			outfilename = "tex_qualPStatsMV_H_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Mean and standard deviation of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//						&accuracyStat, accuracy, false);
//		}
//
//		//Output statistics box plot (median, stdmax, stdmin, max, min)
//		{
//			outfilename = "tex_qualPStatsBP_H_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//						&precisionStat, precision, true);
//		}
//		{
//			outfilename = "tex_qualPStatsBP_H_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//						&recallStat, recall, true);
//		}
//		{
//			outfilename = "tex_qualPStatsBP_H_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//						&fallOutStat, fallOut, true);
//		}
//		{
//			outfilename = "tex_qualPStatsBP_H_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//							std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//			writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//						"# Box plot parameters of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//						&accuracyStat, accuracy, true);
//		}
//
//		if(refine)
//		{
//			//Output for smoothed data
//			{
//				outfilename = "tex_qualPRef_H_preci_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistPrecisionRef.size(); i++)
//				{
//					evalsToFile << cumuldistPrecisionRef[i].second << " " << cumuldistPrecisionRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_H_recall_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistRecallRef.size(); i++)
//				{
//					evalsToFile << cumuldistRecallRef[i].second << " " << cumuldistRecallRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_H_fpr_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistFallOutRef.size(); i++)
//				{
//					evalsToFile << cumuldistFallOutRef[i].second << " " << cumuldistFallOutRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_H_acc_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "# This is a smoothed version of the data and might be slightly different from the measurements" << endl;
//				for(size_t i = 0; i < cumuldistAccuracyRef.size(); i++)
//				{
//					evalsToFile << cumuldistAccuracyRef[i].second << " " << cumuldistAccuracyRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//
//			//Output for original data
//			{
//				outfilename = "tex_qualPRef_H_preciBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistPrecisionBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistPrecisionBinsRef[i].second << " " << cumuldistPrecisionBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_H_recallBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistRecallBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistRecallBinsRef[i].second << " " << cumuldistRecallBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_H_fprBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistFallOutBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistFallOutBinsRef[i].second << " " << cumuldistFallOutBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_qualPRef_H_accBins_cumul_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Cumulative distribution in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "# This is the original version of the data where empty bins were removed" << endl;
//				for(size_t i = 0; i < cumuldistAccuracyBinsRef.size(); i++)
//				{
//					evalsToFile << cumuldistAccuracyBinsRef[i].second << " " << cumuldistAccuracyBinsRef[i].first << endl;
//				}
//				evalsToFile.close();
//			}
//
//			//Output statistics mean & variance
//			{
//				outfilename = "tex_qualPRefStatsMV_H_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//							&precisionRefStat, precisionRef, false);
//			}
//			{
//				outfilename = "tex_qualPRefStatsMV_H_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//							&recallRefStat, recallRef, false);
//			}
//			{
//				outfilename = "tex_qualPRefStatsMV_H_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//							&fallOutRefStat, fallOutRef, false);
//			}
//			{
//				outfilename = "tex_qualPRefStatsMV_H_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Mean and standard deviation of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//							&accuracyRefStat, accuracyRef, false);
//			}
//
//			//Output statistics box plot (median, stdmax, stdmin, max, min)
//			{
//				outfilename = "tex_qualPRefStatsBP_H_preci_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Precision or positive predictive value ppv=truePos/(truePos+falsePos)", 
//							&precisionRefStat, precisionRef, true);
//			}
//			{
//				outfilename = "tex_qualPRefStatsBP_H_recall_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)", 
//							&recallRefStat, recallRef, true);
//			}
//			{
//				outfilename = "tex_qualPRefStatsBP_H_fpr_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)", 
//							&fallOutRefStat, fallOutRef, true);
//			}
//			{
//				outfilename = "tex_qualPRefStatsBP_H_acc_" + featureDetector + "_" + descriptorExtractor + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + ".dat";
//				writeAddDataToFile(outpath + "\\" + outfilename, matcherType, 
//							"# Box plot parameters of Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)", 
//							&accuracyRefStat, accuracyRef, true);
//			}
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//	return 0;
//}
//
///* --------------------- Functions --------------------- */
//
///* Starts testing different thresholds for the statistical flow verification step.
// *
// * string imgsPath				Input  -> Path which includes both left and right images
// * string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
// * int flowDispH				Input  -> Indicates which type of ground truth data is used:
// *										  0: flow files from KITTI database
// *										  1: disparity files from KITTI database
// *										  2: homography files (Please note that a homography always relates
// *											 to the first image (e.g. 1->2, 1->3, ...))
// * string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images 
// *									      (after prefix only comes the image number)
// * string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
// *									      (after prefix only comes the image number). For testing with homographies,
// *										  this string can be empty.
// * string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
// *									      (after prefix only comes the image number)
// * string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
// *										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
// *										  detectors are possible.
// * string descriptorExtractor	Input  -> The used descriptor extractor. Possible inputs should only be FREAK
// *										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
// * string storeResultPath		Input  -> Path were the resulting measurements should be stored
// * int showResult				Input  -> If >= 0, the result from the matching algorithm is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * string storeImgResPath		Input  -> Optional path for storing the resulting matches drawn into the images,
// *										  where the options of which results should be drawn are specified in
// *										  "showResult". If this path is set, the images are NOT displayed.
// * 
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Failed
// */
//int testGMbSOFthreshold(std::string imgsPath, std::string flowDispHPath, int flowDispH, 
//						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//						 std::string featureDetector, std::string descriptorExtractor,
//						 std::string storeResultPath, int showResult, std::string storeImgResPath)
//{
//	int err;
//	cv::Mat src[2];
//	int numthdivs = 12;
//	double minth = 0.1, thchange = 0.08;
//	const bool useMedian = false;
//	double setth;
//	vector<pair<double,vector<pair<double,vector<double>>>>> thPreci, thReca, thFpr, thAcc; //{[inl1, (th1, (q1, ..., qn)), ..., (thn, (q1, ..., qn))], ..., [inln, ...]}
//	vector<pair<double,vector<pair<double, double>>>> thPreciMed, thRecaMed, thFprMed, thAccMed;
//	//vector<double> inlierRatios;
//	std::auto_ptr<GMbSOF_matcher> mymatcher;
//	string outpath, outfilename;
//
//	if(!storeImgResPath.empty() && (showResult == -1))
//	{
//		cout << "If you want to store the resulting images you must specify the showResult flag! Exiting." << endl;
//		exit(0);
//	}
//
//	//Generate inlier ratios
//	double startInlRatio = 1.0;
//	//inlierRatios.push_back(startInlRatio);
//	thPreci.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//	thReca.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//	thFpr.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//	thAcc.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//	thPreciMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//	thRecaMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//	thFprMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//	thAccMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//	//init thresholds
//	for(int j = 0; j < numthdivs; j++)
//	{
//		setth = minth + (double)j * thchange;
//		thPreci.back().second.push_back(make_pair(setth, vector<double>()));
//		thReca.back().second.push_back(make_pair(setth, vector<double>()));
//		thFpr.back().second.push_back(make_pair(setth, vector<double>()));
//		thAcc.back().second.push_back(make_pair(setth, vector<double>()));
//		thPreciMed.back().second.push_back(make_pair(setth, 0));
//		thRecaMed.back().second.push_back(make_pair(setth, 0));
//		thFprMed.back().second.push_back(make_pair(setth, 1.0));
//		thAccMed.back().second.push_back(make_pair(setth, 0));
//	}
//	while(startInlRatio > 0.2)
//	{
//		startInlRatio -= 0.05;
//		//inlierRatios.push_back(startInlRatio);
//		thPreci.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thReca.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thFpr.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thAcc.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thPreciMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		thRecaMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		thFprMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		thAccMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		//init thresholds
//		for(int j = 0; j < numthdivs; j++)
//		{
//			setth = minth + (double)j * thchange;
//			thPreci.back().second.push_back(make_pair(setth, vector<double>()));
//			thReca.back().second.push_back(make_pair(setth, vector<double>()));
//			thFpr.back().second.push_back(make_pair(setth, vector<double>()));
//			thAcc.back().second.push_back(make_pair(setth, vector<double>()));
//			thPreciMed.back().second.push_back(make_pair(setth, 0));
//			thRecaMed.back().second.push_back(make_pair(setth, 0));
//			thFprMed.back().second.push_back(make_pair(setth, 1.0));
//			thAccMed.back().second.push_back(make_pair(setth, 0));
//		}
//	}
//	while(startInlRatio > 0.1)
//	{
//		startInlRatio -= 0.02;
//		//inlierRatios.push_back(startInlRatio);
//		thPreci.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thReca.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thFpr.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thAcc.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thPreciMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		thRecaMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		thFprMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		thAccMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		//init thresholds
//		for(int j = 0; j < numthdivs; j++)
//		{
//			setth = minth + (double)j * thchange;
//			thPreci.back().second.push_back(make_pair(setth, vector<double>()));
//			thReca.back().second.push_back(make_pair(setth, vector<double>()));
//			thFpr.back().second.push_back(make_pair(setth, vector<double>()));
//			thAcc.back().second.push_back(make_pair(setth, vector<double>()));
//			thPreciMed.back().second.push_back(make_pair(setth, 0));
//			thRecaMed.back().second.push_back(make_pair(setth, 0));
//			thFprMed.back().second.push_back(make_pair(setth, 1.0));
//			thAccMed.back().second.push_back(make_pair(setth, 0));
//		}
//	}
//	while(startInlRatio > 0.01)
//	{
//		startInlRatio -= 0.01;
//		//inlierRatios.push_back(startInlRatio);
//		thPreci.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thReca.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thFpr.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thAcc.push_back(make_pair(startInlRatio,vector<pair<double,vector<double>>>()));
//		thPreciMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		thRecaMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		thFprMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		thAccMed.push_back(make_pair(startInlRatio,vector<pair<double, double>>()));
//		//init thresholds
//		for(int j = 0; j < numthdivs; j++)
//		{
//			setth = minth + (double)j * thchange;
//			thPreci.back().second.push_back(make_pair(setth, vector<double>()));
//			thReca.back().second.push_back(make_pair(setth, vector<double>()));
//			thFpr.back().second.push_back(make_pair(setth, vector<double>()));
//			thAcc.back().second.push_back(make_pair(setth, vector<double>()));
//			thPreciMed.back().second.push_back(make_pair(setth, 0));
//			thRecaMed.back().second.push_back(make_pair(setth, 0));
//			thFprMed.back().second.push_back(make_pair(setth, 1.0));
//			thAccMed.back().second.push_back(make_pair(setth, 0));
//		}
//	}
//
//	if(flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int k = 0; k < (int)thPreci.size(); k++)
//		{
//			for(int i = 0; i < (int)filenamesl.size(); i++)
//			{
//				cv::Mat flowimg;
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//				src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//				err = convertImageFlowFile(flowDispHPath, filenamesflow[i], &flowimg);
//				if(err)
//				{
//					cout << "Could not open flow file with index " << i << endl;
//					continue;
//				}
//				for(int j = 0; j < numthdivs; j++)
//				{
//					//mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], thPreci[j].first));
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], thPreci[k].second[j].first));
//
//					if(mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					err = mymatcher->performMatching(thPreci[k].first);
//					if(err)
//					{
//						if( err == -2)
//						{
//							thPreci[k].second[j].second.push_back(0);
//							thReca[k].second[j].second.push_back(0);
//							thFpr[k].second[j].second.push_back(1.0);
//							thAcc[k].second[j].second.push_back(0);
//						}
//						continue;
//					}
//
//					thPreci[k].second[j].second.push_back(mymatcher->qpm.ppv);
//					thReca[k].second[j].second.push_back(mymatcher->qpm.tpr);
//					thFpr[k].second[j].second.push_back(mymatcher->qpm.fpr);
//					thAcc[k].second[j].second.push_back(mymatcher->qpm.acc);			
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_th_flow_idxs" + std::to_string((ULONGLONG)i) + "_th" + 
//										std::to_string((ULONGLONG)(floor(thPreci[k].second[j].first * 100.0 +0.5))) + "_" + 
//										featureDetector + "_" + descriptorExtractor + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(thPreci[k].first*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//				}
//			}
//			thPreci[k].first = mymatcher->inlRatioO;
//			thReca[k].first = mymatcher->inlRatioO;
//			thFpr[k].first = mymatcher->inlRatioO;
//			thAcc[k].first = mymatcher->inlRatioO;
//		}
//		/*if(thPreci.back().second.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}*/
//		
//		for(int k = 0; k < thPreci.size(); k++)
//		{
//			if(useMedian)
//			{
//				for(int i = 0; i < thPreci[k].second.size(); i++)
//				{
//					if(thPreci[k].second[i].second.size() > 2)
//					{
//						sort(thPreci[k].second[i].second.begin(), thPreci[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//						sort(thReca[k].second[i].second.begin(), thReca[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//						sort(thFpr[k].second[i].second.begin(), thFpr[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//						sort(thAcc[k].second[i].second.begin(), thAcc[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//					}
//
//					if(!thPreci[k].second[i].second.empty())
//					{
//						if(thPreci[i].second.size() % 2) //Get median of GT inliers and its corresponding overall matching time
//						{
//							thPreciMed[k].second[i].second = thPreci[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//							thRecaMed[k].second[i].second = thReca[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//							thFprMed[k].second[i].second = thFpr[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//							thAccMed[k].second[i].second = thAcc[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//						}
//						else
//						{
//							thPreciMed[k].second[i].second = (thPreci[k].second[i].second[thPreci[k].second[i].second.size()/2] + thPreci[k].second[i].second[thPreci[k].second[i].second.size()/2 - 1]/2);
//							thRecaMed[k].second[i].second = (thReca[k].second[i].second[thPreci[k].second[i].second.size()/2] + thReca[k].second[i].second[thPreci[k].second[i].second.size()/2-1])/2;
//							thFprMed[k].second[i].second = (thFpr[k].second[i].second[thPreci[k].second[i].second.size()/2] + thFpr[k].second[i].second[thPreci[k].second[i].second.size()/2-1])/2;
//							thAccMed[k].second[i].second = (thAcc[k].second[i].second[thPreci[k].second[i].second.size()/2] + thAcc[k].second[i].second[thPreci[k].second[i].second.size()/2-1])/2;
//						}
//					}
//				}
//			}
//			else
//			{
//				for(int i = 0; i < thPreci[k].second.size(); i++)
//				{
//					if(thPreci[k].second[i].second.empty())
//						continue;
//					thPreciMed[k].second[i].second = 0;
//					thRecaMed[k].second[i].second = 0;
//					thFprMed[k].second[i].second = 0;
//					thAccMed[k].second[i].second = 0;
//					for(int j = 0; j < thPreci[k].second[i].second.size(); j++)
//					{
//						thPreciMed[k].second[i].second += thPreci[k].second[i].second[j];
//						thRecaMed[k].second[i].second += thReca[k].second[i].second[j];
//						thFprMed[k].second[i].second += thFpr[k].second[i].second[j];
//						thAccMed[k].second[i].second += thAcc[k].second[i].second[j];
//					}
//					thPreciMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//					thRecaMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//					thFprMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//					thAccMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//				}
//			}
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		
//		//Output of 2D figures with different inlier ratios
//		for(int j = 0; j < 17; j = j + 4)
//		{
//			{
//				outfilename = "tex_th_2D_flow_preci_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thPreciMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thPreciMed[j].second.size(); i++)
//				{
//					evalsToFile << thPreciMed[j].second[i].first << " " << thPreciMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//
//			{
//				outfilename = "tex_th_2D_flow_recall_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thRecaMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thRecaMed[j].second.size(); i++)
//				{
//					evalsToFile << thRecaMed[j].second[i].first << " " << thRecaMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_th_2D_flow_fpr_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thFprMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thFprMed[j].second.size(); i++)
//				{
//					evalsToFile << thFprMed[j].second[i].first << " " << thFprMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_th_2D_flow_acc_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thAccMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thAccMed[j].second.size(); i++)
//				{
//					evalsToFile << thAccMed[j].second[i].first << " " << thAccMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//
//		//Output of 3D plots
//		{
//			outfilename = "tex_th_3D_flow_preci_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thPreciMed.size(); k++)
//			{
//				for(size_t i = 0; i < thPreciMed[k].second.size(); i++)
//				{
//					evalsToFile << thPreciMed[k].first << " " << thPreciMed[k].second[i].first << " " << thPreciMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_th_3D_flow_recall_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thRecaMed.size(); k++)
//			{
//				for(size_t i = 0; i < thRecaMed[k].second.size(); i++)
//				{
//					evalsToFile << thRecaMed[k].first << " " << thRecaMed[k].second[i].first << " " << thRecaMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_th_3D_flow_fpr_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thFprMed.size(); k++)
//			{
//				for(size_t i = 0; i < thFprMed[k].second.size(); i++)
//				{
//					evalsToFile << thFprMed[k].first << " " << thFprMed[k].second[i].first << " " << thFprMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_th_3D_flow_acc_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thAccMed.size(); k++)
//			{
//				for(size_t i = 0; i < thAccMed[k].second.size(); i++)
//				{
//					evalsToFile << thAccMed[k].first << " " << thAccMed[k].second[i].first << " " << thAccMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//	}
//	else if(flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int k = 0; k < (int)thPreci.size(); k++)
//		{
//			for(int i = 0; i < (int)filenamesl.size(); i++)
//			{
//				cv::Mat flowimg;
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//				src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//				err = convertImageDisparityFile(flowDispHPath, filenamesflow[i], &flowimg);
//				if(err)
//				{
//					cout << "Could not open disparity file with index " << i << endl;
//					continue;
//				}
//				for(int j = 0; j < numthdivs; j++)
//				{
//					//mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], thPreci[j].first));
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], thPreci[k].second[j].first));
//
//					if(mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					err = mymatcher->performMatching(thPreci[k].first);
//					if(err)
//					{
//						if( err == -2)
//						{
//							thPreci[k].second[j].second.push_back(0);
//							thReca[k].second[j].second.push_back(0);
//							thFpr[k].second[j].second.push_back(1.0);
//							thAcc[k].second[j].second.push_back(0);
//						}
//						continue;
//					}
//
//					thPreci[k].second[j].second.push_back(mymatcher->qpm.ppv);
//					thReca[k].second[j].second.push_back(mymatcher->qpm.tpr);
//					thFpr[k].second[j].second.push_back(mymatcher->qpm.fpr);
//					thAcc[k].second[j].second.push_back(mymatcher->qpm.acc);			
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_th_disp_idxs" + std::to_string((ULONGLONG)i) + "_th" + 
//										std::to_string((ULONGLONG)(floor(thPreci[k].second[j].first * 100.0 +0.5))) + "_" + 
//										featureDetector + "_" + descriptorExtractor + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(thPreci[k].first*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//				}
//			}
//			thPreci[k].first = mymatcher->inlRatioO;
//			thReca[k].first = mymatcher->inlRatioO;
//			thFpr[k].first = mymatcher->inlRatioO;
//			thAcc[k].first = mymatcher->inlRatioO;
//		}
//		/*if(thPreci.back().second.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}*/
//		
//		for(int k = 0; k < thPreci.size(); k++)
//		{
//			if(useMedian)
//			{
//				for(int i = 0; i < thPreci[k].second.size(); i++)
//				{
//					if(thPreci[k].second[i].second.size() > 2)
//					{
//						sort(thPreci[k].second[i].second.begin(), thPreci[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//						sort(thReca[k].second[i].second.begin(), thReca[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//						sort(thFpr[k].second[i].second.begin(), thFpr[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//						sort(thAcc[k].second[i].second.begin(), thAcc[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//					}
//
//					if(!thPreci[k].second[i].second.empty())
//					{
//						if(thPreci[i].second.size() % 2) //Get median of GT inliers and its corresponding overall matching time
//						{
//							thPreciMed[k].second[i].second = thPreci[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//							thRecaMed[k].second[i].second = thReca[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//							thFprMed[k].second[i].second = thFpr[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//							thAccMed[k].second[i].second = thAcc[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//						}
//						else
//						{
//							thPreciMed[k].second[i].second = (thPreci[k].second[i].second[thPreci[k].second[i].second.size()/2] + thPreci[k].second[i].second[thPreci[k].second[i].second.size()/2 - 1]/2);
//							thRecaMed[k].second[i].second = (thReca[k].second[i].second[thPreci[k].second[i].second.size()/2] + thReca[k].second[i].second[thPreci[k].second[i].second.size()/2-1])/2;
//							thFprMed[k].second[i].second = (thFpr[k].second[i].second[thPreci[k].second[i].second.size()/2] + thFpr[k].second[i].second[thPreci[k].second[i].second.size()/2-1])/2;
//							thAccMed[k].second[i].second = (thAcc[k].second[i].second[thPreci[k].second[i].second.size()/2] + thAcc[k].second[i].second[thPreci[k].second[i].second.size()/2-1])/2;
//						}
//					}
//				}
//			}
//			else
//			{
//				for(int i = 0; i < thPreci[k].second.size(); i++)
//				{
//					if(thPreci[k].second[i].second.empty())
//						continue;
//					thPreciMed[k].second[i].second = 0;
//					thRecaMed[k].second[i].second = 0;
//					thFprMed[k].second[i].second = 0;
//					thAccMed[k].second[i].second = 0;
//					for(int j = 0; j < thPreci[k].second[i].second.size(); j++)
//					{
//						thPreciMed[k].second[i].second += thPreci[k].second[i].second[j];
//						thRecaMed[k].second[i].second += thReca[k].second[i].second[j];
//						thFprMed[k].second[i].second += thFpr[k].second[i].second[j];
//						thAccMed[k].second[i].second += thAcc[k].second[i].second[j];
//					}
//					thPreciMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//					thRecaMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//					thFprMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//					thAccMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//				}
//			}
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//
//		//Output of 2D figures with different inlier ratios
//		for(int j = 0; j < 17; j = j + 4)
//		{
//			{
//				outfilename = "tex_th_2D_disp_preci_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thPreciMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thPreciMed[j].second.size(); i++)
//				{
//					evalsToFile << thPreciMed[j].second[i].first << " " << thPreciMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//
//			{
//				outfilename = "tex_th_2D_disp_recall_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thRecaMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thRecaMed[j].second.size(); i++)
//				{
//					evalsToFile << thRecaMed[j].second[i].first << " " << thRecaMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_th_2D_disp_fpr_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thFprMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thFprMed[j].second.size(); i++)
//				{
//					evalsToFile << thFprMed[j].second[i].first << " " << thFprMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_th_2D_disp_acc_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thAccMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thAccMed[j].second.size(); i++)
//				{
//					evalsToFile << thAccMed[j].second[i].first << " " << thAccMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//
//		//Output of 3D plots
//		{
//			outfilename = "tex_th_3D_disp_preci_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thPreciMed.size(); k++)
//			{
//				for(size_t i = 0; i < thPreciMed[k].second.size(); i++)
//				{
//					evalsToFile << thPreciMed[k].first << " " << thPreciMed[k].second[i].first << " " << thPreciMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_th_3D_disp_recall_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thRecaMed.size(); k++)
//			{
//				for(size_t i = 0; i < thRecaMed[k].second.size(); i++)
//				{
//					evalsToFile << thRecaMed[k].first << " " << thRecaMed[k].second[i].first << " " << thRecaMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_th_3D_disp_fpr_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thFprMed.size(); k++)
//			{
//				for(size_t i = 0; i < thFprMed[k].second.size(); i++)
//				{
//					evalsToFile << thFprMed[k].first << " " << thFprMed[k].second[i].first << " " << thFprMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_th_3D_disp_acc_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thAccMed.size(); k++)
//			{
//				for(size_t i = 0; i < thAccMed[k].second.size(); i++)
//				{
//					evalsToFile << thAccMed[k].first << " " << thAccMed[k].second[i].first << " " << thAccMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//	}
//	else if(flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		//cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if(err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if(err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if(err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		if(fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//		{
//			for(int k = 0; k < (int)thPreci.size(); k++)
//			{
//				//Take the stored homographys and perform evaluation
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//				for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//				{
//					cv::Mat H = Hs[idx1];
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//					for(int j = 0; j < numthdivs; j++)
//					{
//						//mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], thPreci[k].second[j].first));
//						mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], thPreci[k].second[j].first));
//
//						if(mymatcher->specialGMbSOFtest)
//						{
//							cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//							exit(0);
//						}
//
//						err = mymatcher->performMatching(thPreci[k].first);
//						if(err)
//						{
//							if( err == -2)
//							{
//								thPreci[k].second[j].second.push_back(0);
//								thReca[k].second[j].second.push_back(0);
//								thFpr[k].second[j].second.push_back(1.0);
//								thAcc[k].second[j].second.push_back(0);
//							}
//							continue;
//						}
//
//						thPreci[k].second[j].second.push_back(mymatcher->qpm.ppv);
//						thReca[k].second[j].second.push_back(mymatcher->qpm.tpr);
//						thFpr[k].second[j].second.push_back(mymatcher->qpm.fpr);
//						thAcc[k].second[j].second.push_back(mymatcher->qpm.acc);			
//
//						if(showResult >= 0)
//						{
//							if(storeImgResPath.empty())
//							{
//								mymatcher->showMatches(showResult);
//							}
//							else
//							{
//								if(dirExists(storeImgResPath)) //Check if output directory existis
//								{
//									outpath = storeImgResPath;
//								}
//								else
//								{
//									outpath = imgsPath + "\\evalImgs";
//									if(!dirExists(outpath))
//										_mkdir(outpath.c_str());
//								}
//								outfilename = "img_th_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_th" + 
//											std::to_string((ULONGLONG)(floor(thPreci[k].second[j].first * 100.0 +0.5))) + "_" + 
//											featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//											std::to_string((ULONGLONG)std::floor(thPreci[k].first*100.0 + 0.5)) + ".bmp";
//								mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//							}
//						}
//					}
//				}
//				//Generate new homographys to evaluate all other possible configurations of the images to each other
//				for(int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//				{
//					for(int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//					{
//						//H = (Hs[idx2].inv() * Hs[idx1]).inv();
//						cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//						src[0] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//						src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx2 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//						for(int j = 0; j < numthdivs; j++)
//						{
//							//mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], thPreci[j].first));
//							mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], thPreci[k].second[j].first));
//
//							if(mymatcher->specialGMbSOFtest)
//							{
//								cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//								exit(0);
//							}
//
//							err = mymatcher->performMatching(thPreci[k].first);
//							if(err)
//							{
//								if( err == -2)
//								{
//									thPreci[k].second[j].second.push_back(0);
//									thReca[k].second[j].second.push_back(0);
//									thFpr[k].second[j].second.push_back(1.0);
//									thAcc[k].second[j].second.push_back(0);
//								}
//								continue;
//							}
//
//							thPreci[k].second[j].second.push_back(mymatcher->qpm.ppv);
//							thReca[k].second[j].second.push_back(mymatcher->qpm.tpr);
//							thFpr[k].second[j].second.push_back(mymatcher->qpm.fpr);
//							thAcc[k].second[j].second.push_back(mymatcher->qpm.acc);			
//
//							if(showResult >= 0)
//							{
//								if(storeImgResPath.empty())
//								{
//									mymatcher->showMatches(showResult);
//								}
//								else
//								{
//									if(dirExists(storeImgResPath)) //Check if output directory existis
//									{
//										outpath = storeImgResPath;
//									}
//									else
//									{
//										outpath = imgsPath + "\\evalImgs";
//										if(!dirExists(outpath))
//											_mkdir(outpath.c_str());
//									}
//									outfilename = "img_th_H_idxs" + std::to_string((ULONGLONG)(idx1 + 1)) + "-" + std::to_string((ULONGLONG)(idx2 + 1)) + "_th" + 
//												std::to_string((ULONGLONG)(floor(thPreci[k].second[j].first * 100.0 +0.5))) + "_" + 
//												featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//												std::to_string((ULONGLONG)std::floor(thPreci[k].first*100.0 + 0.5)) + ".bmp";
//									mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//		else
//		{
//			for(int k = 0; k < (int)thPreci.size(); k++)
//			{
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//				for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//				{
//					cv::Mat H = Hs[idx1];
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//				
//					for(int j = 0; j < numthdivs; j++)
//					{
//						//mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], thPreci[k].second[j].first));
//						mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], thPreci[k].second[j].first));
//
//						if(mymatcher->specialGMbSOFtest)
//						{
//							cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//							exit(0);
//						}
//
//						err = mymatcher->performMatching(thPreci[k].first);
//						if(err)
//						{
//							if( err == -2)
//							{
//								thPreci[k].second[j].second.push_back(0);
//								thReca[k].second[j].second.push_back(0);
//								thFpr[k].second[j].second.push_back(1.0);
//								thAcc[k].second[j].second.push_back(0);
//							}
//							continue;
//						}
//
//						thPreci[k].second[j].second.push_back(mymatcher->qpm.ppv);
//						thReca[k].second[j].second.push_back(mymatcher->qpm.tpr);
//						thFpr[k].second[j].second.push_back(mymatcher->qpm.fpr);
//						thAcc[k].second[j].second.push_back(mymatcher->qpm.acc);			
//
//						if(showResult >= 0)
//						{
//							if(storeImgResPath.empty())
//							{
//								mymatcher->showMatches(showResult);
//							}
//							else
//							{
//								if(dirExists(storeImgResPath)) //Check if output directory existis
//								{
//									outpath = storeImgResPath;
//								}
//								else
//								{
//									outpath = imgsPath + "\\evalImgs";
//									if(!dirExists(outpath))
//										_mkdir(outpath.c_str());
//								}
//								outfilename = "img_th_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_th" + 
//											std::to_string((ULONGLONG)(floor(thPreci[k].second[j].first * 100.0 +0.5))) + "_" + 
//											featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//											std::to_string((ULONGLONG)std::floor(thPreci[k].first*100.0 + 0.5)) + ".bmp";
//								mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//							}
//						}
//					}
//				}
//			}
//		}
//		/*if(thPreci.back().second.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}*/
//
//		for(int k = 0; k < thPreci.size(); k++)
//		{
//			if(useMedian)
//			{
//				for(int i = 0; i < thPreci[k].second.size(); i++)
//				{
//					if(thPreci[k].second[i].second.size() > 2)
//					{
//						sort(thPreci[k].second[i].second.begin(), thPreci[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//						sort(thReca[k].second[i].second.begin(), thReca[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//						sort(thFpr[k].second[i].second.begin(), thFpr[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//						sort(thAcc[k].second[i].second.begin(), thAcc[k].second[i].second.end(),
//								[](double first, double second){return first < second;});
//					}
//
//					if(!thPreci[k].second[i].second.empty())
//					{
//						if(thPreci[i].second.size() % 2) //Get median of GT inliers and its corresponding overall matching time
//						{
//							thPreciMed[k].second[i].second = thPreci[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//							thRecaMed[k].second[i].second = thReca[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//							thFprMed[k].second[i].second = thFpr[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//							thAccMed[k].second[i].second = thAcc[k].second[i].second[(thPreci[k].second[i].second.size() - 1)/2];
//						}
//						else
//						{
//							thPreciMed[k].second[i].second = (thPreci[k].second[i].second[thPreci[k].second[i].second.size()/2] + thPreci[k].second[i].second[thPreci[k].second[i].second.size()/2 - 1]/2);
//							thRecaMed[k].second[i].second = (thReca[k].second[i].second[thPreci[k].second[i].second.size()/2] + thReca[k].second[i].second[thPreci[k].second[i].second.size()/2-1])/2;
//							thFprMed[k].second[i].second = (thFpr[k].second[i].second[thPreci[k].second[i].second.size()/2] + thFpr[k].second[i].second[thPreci[k].second[i].second.size()/2-1])/2;
//							thAccMed[k].second[i].second = (thAcc[k].second[i].second[thPreci[k].second[i].second.size()/2] + thAcc[k].second[i].second[thPreci[k].second[i].second.size()/2-1])/2;
//						}
//					}
//				}
//			}
//			else
//			{
//				for(int i = 0; i < thPreci[k].second.size(); i++)
//				{
//					if(thPreci[k].second[i].second.empty())
//						continue;
//					thPreciMed[k].second[i].second = 0;
//					thRecaMed[k].second[i].second = 0;
//					thFprMed[k].second[i].second = 0;
//					thAccMed[k].second[i].second = 0;
//					for(int j = 0; j < thPreci[k].second[i].second.size(); j++)
//					{
//						thPreciMed[k].second[i].second += thPreci[k].second[i].second[j];
//						thRecaMed[k].second[i].second += thReca[k].second[i].second[j];
//						thFprMed[k].second[i].second += thFpr[k].second[i].second[j];
//						thAccMed[k].second[i].second += thAcc[k].second[i].second[j];
//					}
//					thPreciMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//					thRecaMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//					thFprMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//					thAccMed[k].second[i].second /= (double)thPreci[k].second[i].second.size();
//				}
//			}
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//
//		//Output of 2D figures with different inlier ratios
//		for(int j = 0; j < 17; j = j + 4)
//		{
//			{
//				outfilename = "tex_th_2D_H_preci_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thPreciMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thPreciMed[j].second.size(); i++)
//				{
//					evalsToFile << thPreciMed[j].second[i].first << " " << thPreciMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//
//			{
//				outfilename = "tex_th_2D_H_recall_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thRecaMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thRecaMed[j].second.size(); i++)
//				{
//					evalsToFile << thRecaMed[j].second[i].first << " " << thRecaMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_th_2D_H_fpr_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thFprMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thFprMed[j].second.size(); i++)
//				{
//					evalsToFile << thFprMed[j].second[i].first << " " << thFprMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_th_2D_H_acc_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(thAccMed[j].first*100.0 + 0.5)) + ".dat";
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Statistical flow filtering threshold in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "th mean" << endl;
//				for(size_t i = 0; i < thAccMed[j].second.size(); i++)
//				{
//					evalsToFile << thAccMed[j].second[i].first << " " << thAccMed[j].second[i].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//
//		//Output of 3D plots
//		{
//			outfilename = "tex_th_3D_H_preci_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thPreciMed.size(); k++)
//			{
//				for(size_t i = 0; i < thPreciMed[k].second.size(); i++)
//				{
//					evalsToFile << thPreciMed[k].first << " " << thPreciMed[k].second[i].first << " " << thPreciMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_th_3D_H_recall_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thRecaMed.size(); k++)
//			{
//				for(size_t i = 0; i < thRecaMed[k].second.size(); i++)
//				{
//					evalsToFile << thRecaMed[k].first << " " << thRecaMed[k].second[i].first << " " << thRecaMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_th_3D_H_fpr_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thFprMed.size(); k++)
//			{
//				for(size_t i = 0; i < thFprMed[k].second.size(); i++)
//				{
//					evalsToFile << thFprMed[k].first << " " << thFprMed[k].second[i].first << " " << thFprMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_th_3D_H_acc_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Statistical flow filtering threshold in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos) at different inlier ratios" << endl;
//			evalsToFile << "inl th mean" << endl;
//			for(int k = 0; k < thAccMed.size(); k++)
//			{
//				for(size_t i = 0; i < thAccMed[k].second.size(); i++)
//				{
//					evalsToFile << thAccMed[k].first << " " << thAccMed[k].second[i].first << " " << thAccMed[k].second[i].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//	return 0;
//}
//
//
//bool dirExists(const std::string& dirName_in)
//{
//  DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
//  if (ftyp == INVALID_FILE_ATTRIBUTES)
//    return false;  //something is wrong with your path!
//
//  if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
//    return true;   // this is a directory!
//
//  return false;    // this is not a directory!
//}
//
//void getSmoothComulDist(std::vector<double> precision, std::vector<std::pair<double,double>> & cumuldistPrecision, 
//						std::vector<std::pair<double,double>> & cumuldistPrecisionBins, double distrval, double bindist)
//{
//	int j = 0;
//	double maxcount = 0;
//	if(precision.empty())
//	{
//		for(int i = 0; i < (int)cumuldistPrecision.size(); i++)
//		{
//				cumuldistPrecision[i].second = 0;
//				cumuldistPrecisionBins[i].second = 0;
//		}
//		for(int i = 0; i < (int)cumuldistPrecision.size(); i++)
//		{
//			double hlp = cumuldistPrecision[i].first;
//			cumuldistPrecision[i].first = cumuldistPrecision[i].second;
//			cumuldistPrecision[i].second = hlp;
//		}
//		for(int i = 0; i < (int)cumuldistPrecisionBins.size(); i++)
//		{
//			double hlp = cumuldistPrecisionBins[i].first;
//			cumuldistPrecisionBins[i].first = cumuldistPrecisionBins[i].second;
//			cumuldistPrecisionBins[i].second = hlp;
//		}
//		return;
//	}
//	for(int i = 0; i < (int)cumuldistPrecision.size(); i++)
//	{
//		bool breakfor = false;
//		while(precision[j] <= (distrval + cumuldistPrecision[i].first))
//		{
//			if(cumuldistPrecision[i].second < 0)
//				cumuldistPrecision[i].second = 0;
//			cumuldistPrecision[i].second++;
//			if(j == (int)precision.size() - 1)
//			{
//				breakfor = true;
//				break;
//			}
//			j++;
//		}
//		if(maxcount < cumuldistPrecision[i].second)
//			maxcount = cumuldistPrecision[i].second;
//		if(breakfor)
//		{
//			break;
//		}
//	}
//	for(int i = 0; i < (int)cumuldistPrecision.size(); i++)
//	{
//		if(cumuldistPrecision[i].second > 0)
//			cumuldistPrecision[i].second /= maxcount;
//	}
//	cumuldistPrecisionBins = cumuldistPrecision;
//	for(int i = 0; i < (int)cumuldistPrecisionBins.size(); i++) //Get only valid measurements for storing to file
//	{
//		if(cumuldistPrecisionBins[i].second < 0)
//		{
//			cumuldistPrecisionBins.erase(cumuldistPrecisionBins.begin() + i);
//			i--;
//		}
//	}
//	double kernelwith = bindist * 2;
//	for(int i = 0; i < (int)cumuldistPrecision.size() - 1; i++) //Fill missing bins with average value
//	{
//		j = i + 1;
//		while(cumuldistPrecision[j].second < 0)
//		{
//			if(j < (int)cumuldistPrecision.size() - 1)
//				j++;
//			else
//				break;
//		}
//		if(j > i + 1)
//		{
//			if((i == 0) && (cumuldistPrecision[0].second < 0))
//			{
//				for(int k = 0; k < j; k++)
//				{
//					if(cumuldistPrecision[k].second < 0)
//						cumuldistPrecision[k].second = 0;
//				}
//			}
//			else if(j == (int)cumuldistPrecision.size() - 1)
//			{
//				for(int k = i + 1; k <= j; k++)
//				{
//					if(cumuldistPrecision[k].second < 0)
//						cumuldistPrecision[k].second = 0;
//				}
//				break;
//			}
//			else
//			{
//				double meanval = 0;//(cumuldistPrecision[i].second + cumuldistPrecision[j].second) / 2.0;//Change the 0 to commented expression if you want filling
//				for(int k = i + 1; k < j; k++)
//				{
//					cumuldistPrecision[k].second = meanval;
//				}
//				double hlp = ceil(((double)(j - i) * bindist) / 2.0) * bindist;
//				if(hlp > kernelwith)
//					kernelwith = hlp;
//			}
//			i = j - 1;
//		}
//		else if((j == (int)cumuldistPrecision.size() - 1) && (cumuldistPrecision[j].second < 0))
//		{
//			cumuldistPrecision[j].second = 0;
//		}
//		else if((i == 0) && (cumuldistPrecision[0].second < 0))
//		{
//			cumuldistPrecision[0].second = 0;
//		}
//	}
//	//Use Epanechnikov kernel for smoothing
//	vector<pair<double,double>> cumuldistPrecision_tmp;
//	cumuldistPrecision_tmp = cumuldistPrecision;//Delete this if you want smoothing
//	/*int numweights = (int)floor(kernelwith / bindist + 0.5) * 2 + 1;
//	std::vector<double> weights(numweights);
//	double overallweight = 0;
//	for(int i = 0; i < numweights; i++)
//	{
//		double hlp = ((double)i * bindist - kernelwith) / kernelwith;
//		weights[i] = 0.75 * (1.0 - hlp * hlp);
//		overallweight += weights[i];
//	}
//	for(int i = 0; i < numweights; i++)
//	{
//		weights[i] /= overallweight;
//	}
//	vector<double> kernelvalues(numweights,0);
//	double maxresval = 0;
//	for(int i = 0; i < (int)cumuldistPrecision.size(); i++)
//	{
//		double newval = 0;
//		j = i - (numweights - 1) / 2;
//		int k = 0;
//		while(j < 0)
//		{
//			kernelvalues[k] = cumuldistPrecision[i].second;
//			j++;
//			k++;
//		}
//		while((j < (int)cumuldistPrecision.size()) && (k < (int)kernelvalues.size()))
//		{
//			kernelvalues[k] = cumuldistPrecision[j].second;
//			j++;
//			k++;
//		}
//		while((j >= (int)cumuldistPrecision.size()) && (k < (int)kernelvalues.size()))
//		{
//			kernelvalues[k] = cumuldistPrecision[i].second;
//			j++;
//			k++;
//		}
//		for(int k1 = 0; k1 < numweights; k1++)
//		{
//			newval += kernelvalues[k1] * weights[k1];
//		}
//		cumuldistPrecision_tmp.push_back(make_pair(cumuldistPrecision[i].first, newval));
//		if(maxresval < newval)
//			maxresval = newval;
//	}
//	for(int i = 0; i < (int)cumuldistPrecision.size(); i++)
//	{
//		cumuldistPrecision_tmp[i].second /= maxresval;
//	}*/
//	cumuldistPrecision.clear();
//	cumuldistPrecision.push_back(make_pair(0,cumuldistPrecision_tmp[0].second));
//	cumuldistPrecision.insert(cumuldistPrecision.end(), cumuldistPrecision_tmp.begin(), cumuldistPrecision_tmp.end());
//	cumuldistPrecision.push_back(make_pair(1.0,cumuldistPrecision_tmp.back().second));
//	for(int i = 0; i < (int)cumuldistPrecision.size(); i++)
//	{
//		double hlp = cumuldistPrecision[i].first;
//		cumuldistPrecision[i].first = cumuldistPrecision[i].second;
//		cumuldistPrecision[i].second = hlp;
//	}
//	for(int i = 0; i < (int)cumuldistPrecisionBins.size(); i++)
//	{
//		double hlp = cumuldistPrecisionBins[i].first;
//		cumuldistPrecisionBins[i].first = cumuldistPrecisionBins[i].second;
//		cumuldistPrecisionBins[i].second = hlp;
//	}
//}
//
///* Calculates statistical parameters for the given values in the vector. The following parameters
// * are calculated: median, arithmetic mean value, standard deviation and standard deviation using the median.
// *
// * vector<double> vals		Input  -> Input vector from which the statistical parameters should be calculated
// * qualityParm1* stats		Output -> Structure holding the statistical parameters
// * bool rejQuartiles		Input  -> If true, the lower and upper quartiles are rejected before calculating
// *									  the parameters
// *
// * Return value:		 none
// */
//void getStatisticfromVec2(const std::vector<double> vals, qualityParm1 *stats, bool rejQuartiles)
//{
//	if(vals.empty())
//	{
//		stats->arithErr = 0;
//		stats->arithStd = 0;
//		stats->medErr = 0;
//		stats->medStd = 0;
//		stats->lowerQuart = 0;
//		stats->upperQuart = 0;
//		return;
//	}
//	int n = vals.size();
//	if(rejQuartiles && (n < 4))
//		rejQuartiles = false;
//	int qrt_si = (int)floor(0.25 * (double)n);
//	std::vector<double> vals_tmp(vals);
//
//	std::sort(vals_tmp.begin(),vals_tmp.end(),[](double const & first, double const & second){
//		return first < second;});
//	//INLINEQSORTSIMPLE(double,vals_tmp.data(),n,Pfe32u_lt);
//
//	if(n % 2)
//		stats->medErr = vals_tmp[(n-1)/2];
//	else
//		stats->medErr = (vals_tmp[n/2]+vals_tmp[n/2-1])/2;
//
//	stats->lowerQuart = vals_tmp[qrt_si];
//	if(n > 3)
//		stats->upperQuart = vals_tmp[n-qrt_si];
//	else
//		stats->upperQuart = vals_tmp[qrt_si];
//
//	stats->arithErr = 0.0;
//	double err2sum = 0.0;
//	double medstdsum = 0.0;
//	double hlp;
//	for(int i = rejQuartiles ? qrt_si:0; i < (rejQuartiles ? (n-qrt_si):n); i++)
//	{
//		stats->arithErr += vals_tmp[i];
//		err2sum += vals_tmp[i] * vals_tmp[i];
//		hlp = (vals_tmp[i] - stats->medErr);
//		medstdsum += hlp * hlp;
//	}
//	if(rejQuartiles)
//		n -= 2 * qrt_si;
//	stats->arithErr /= n;
//
//	hlp = err2sum-n*(stats->arithErr)*(stats->arithErr);
//	if(std::abs(hlp) < 1e-6)
//		stats->arithStd = 0.0;
//	else
//		stats->arithStd = std::sqrt(hlp/(n-1));
//
//	if(std::abs(medstdsum) < 1e-6)
//		stats->medStd = 0.0;
//	else
//		stats->medStd = std::sqrt(medstdsum/(n-1));
//}
//
//bool file_exists(const std::string& name) {
//    ifstream f(name.c_str());
//    if (f.good()) {
//        f.close();
//        return true;
//    } else {
//        f.close();
//        return false;
//    }   
//}
//
//void writeAddDataToFile(string pathFile, string matcherType, string description, qualityParm1 *param, std::vector<double> vals, bool boxplot)
//{
//	string lastline;
//	int nr = 1;
//	if(file_exists(pathFile))
//	{
//		std::stringstream streamfile;
//		std::string line, line2, lastline, line3;
//		std::ifstream ifile(pathFile);
//		std::getline(ifile,line);
//		streamfile << line << endl;
//		std::getline(ifile,line);
//		line += "; " + matcherType;
//		streamfile << line << endl;
//		while (std::getline (ifile,line2) )
//		{
//			lastline = line2;
//			streamfile << line2 << endl;
//		}
//		line3 = lastline.substr(0,lastline.find_first_of(" "));
//		nr = atoi(line3.c_str()) + 1;
//		ifile.close();
//		std::remove(pathFile.c_str());
//		if(boxplot)
//		{
//			streamfile << nr << " " << param->medErr << " " << param->upperQuart << " " << 
//				param->lowerQuart << " " << *std::max_element(vals.begin(),vals.end()) << " " << 
//				*std::min_element(vals.begin(),vals.end()) << endl;
//		}
//		else
//		{
//			streamfile << nr << " " << param->arithErr << " " << param->arithStd << endl;
//		}
//		std::ofstream ofile(pathFile);
//		ofile << streamfile.rdbuf();
//		ofile.close();
//	}
//	else
//	{
//		std::ofstream ofile(pathFile);
//		ofile << description << endl;
//		ofile << "# " << matcherType << endl;
//		if(boxplot)
//		{
//			ofile << "x med stdmax stdmin max min" << endl;
//			ofile << "1 " << param->medErr << " " << param->upperQuart << " " << 
//				param->lowerQuart << " " << *std::max_element(vals.begin(),vals.end()) << " " << 
//				*std::min_element(vals.begin(),vals.end()) << endl;
//		}
//		else
//		{
//			ofile << "x mean error" << endl;
//			ofile << "1 " << param->arithErr << " " << param->arithStd << endl;
//		}
//		ofile.close();
//	}
//}
//
///* Starts testing different multiplication factors for the search range estimated by the statistical flow calculation step.
// *
// * string imgsPath				Input  -> Path which includes both left and right images
// * string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
// * int flowDispH				Input  -> Indicates which type of ground truth data is used:
// *										  0: flow files from KITTI database
// *										  1: disparity files from KITTI database
// *										  2: homography files (Please note that a homography always relates
// *											 to the first image (e.g. 1->2, 1->3, ...))
// * string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images 
// *									      (after prefix only comes the image number)
// * string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
// *									      (after prefix only comes the image number). For testing with homographies,
// *										  this string can be empty.
// * string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
// *									      (after prefix only comes the image number)
// * string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
// *										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
// *										  detectors are possible.
// * string descriptorExtractor	Input  -> The used descriptor extractor. Possible inputs should only be FREAK
// *										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
// * string storeResultPath		Input  -> Path were the resulting measurements should be stored
// * double inlRatio				Input  -> The inlier ratio which should be generated using the ground truth data. If
// *										  set, this one specific inlier ratio is used. If smaller than or equal 0, a few inlier
// *										  ratios are tested. [DEFAULT = -1.0]
// * double validationTh			Input  -> Threshold value for verifying the statistical optical flow [DEFAULT = 0.3]
// * int showResult				Input  -> If >= 0, the result from the matching algorithm is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * string storeImgResPath		Input  -> Optional path for storing the resulting matches drawn into the images,
// *										  where the options of which results should be drawn are specified in
// *										  "showResult". If this path is set, the images are NOT displayed.
// * 
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Failed
// */
//int testGMbSOFsearchRange(std::string imgsPath, std::string flowDispHPath, int flowDispH, 
//						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//						 std::string featureDetector, std::string descriptorExtractor,
//						 std::string storeResultPath, double inlRatio, double validationTh, int showResult, 
//						 std::string storeImgResPath)
//{
//	int err;
//	cv::Mat src[2];
//	const double minInlRat = 0.1, InlRatChange = 0.1, maxInlRat = 1.0;
//	const double InlRat2DFig[4] = {0.2, 0.5, 0.8, 1.0}; //Inlier ratios for which seperate data files should be generated (x: stdMult, y: qualP)
//	const double minStdMult = 1.0, stdMultChange = 0.5, maxStdMult = 7.0;
//	double actInlRat, actStdMult, stdMultCnt, inlRatCnt;
//	inlRatCnt = (maxInlRat - minInlRat) / InlRatChange + 1.0;
//	if((fmod(inlRatCnt, 1.0) > DBL_EPSILON) || (inlRatCnt < 1.0) || (maxInlRat > 1.0) || (minInlRat <= 0.001))
//	{
//		cout << "Wrong range of inlier ratios! Exiting." << endl;
//		exit(0);
//	}
//	stdMultCnt = (maxStdMult - minStdMult) / stdMultChange + 1.0;
//	if((fmod(stdMultCnt, 1.0) > DBL_EPSILON) || (stdMultCnt < 1.0) || (maxStdMult > 10.0) || (minStdMult <= 0.1))
//	{
//		cout << "Wrong range of standard deviation multiplication factors! Exiting." << endl;
//		exit(0);
//	}
//	if((inlRatio > 1.0) || ((inlRatio > 0) && (inlRatio <= 0.001)))
//	{
//		cout << "Wrong inlier ratio specified! Exiting." << endl;
//		exit(0);
//	}
//	if(!storeImgResPath.empty() && (showResult == -1))
//	{
//		cout << "If you want to store the resulting images you must specify the showResult flag! Exiting." << endl;
//		exit(0);
//	}
//	vector<pair<double,vector<pair<double,vector<double>>>>> rangePreci, rangeReca, rangeFpr, rangeAcc; //Holds [stdMult_1, (inlRat_1, qualP_1...qualP_n)...(inlRat_m, qualP_1...qualP_n)]...[stdMult_k, (inlRat_1, qualP_1...qualP_n)...(inlRat_m, qualP_1...qualP_n)]
//	vector<pair<double,vector<pair<double,qualityParm1>>>> rangePreciStat, rangeRecaStat, rangeFprStat, rangeAccStat;
//	vector<pair<double,vector<pair<double,double>>>> rangeAvgMatchTime;
//	qualityParm1 stats_tmp;
//	std::auto_ptr<GMbSOF_matcher> mymatcher;
//	string outpath, outfilename;
//	double sumMatchTime, sumFeatures;
//
//	if(flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//
//		//For all different standard deviation multiplication factors do
//		actStdMult = minStdMult;
//		while(actStdMult <= maxStdMult)
//		{
//			rangePreci.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//			rangeReca.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//			rangeFpr.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//			rangeAcc.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//			rangeAvgMatchTime.push_back(std::make_pair(actStdMult,vector<pair<double,double>>()));
//
//			//For all different inlier ratios do
//			actInlRat = minInlRat;
//			inlRatCnt = 0;
//			while((actInlRat <= maxInlRat) && (inlRatCnt >= 0))
//			{
//				if(inlRatio > 0)
//				{
//					inlRatCnt = -1.0;
//					actInlRat = inlRatio;
//				}
//				rangePreci.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//				rangeReca.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//				rangeFpr.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//				rangeAcc.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//
//				sumFeatures = 0;
//				sumMatchTime = 0;
//				for(int i = 0; i < (int)filenamesl.size(); i++)
//				{
//					cv::Mat flowimg;
//					src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//					src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//					err = convertImageFlowFile(flowDispHPath, filenamesflow[i], &flowimg);
//					if(err)
//					{
//						cout << "Could not open flow file with index " << i << endl;
//						continue;
//					}
//					//mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], thPreci[j].first));
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], validationTh, actStdMult));
//
//					if(mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					err = mymatcher->performMatching(actInlRat, true);
//					if(err)
//					{
//						if( err == -2)
//						{
//							rangePreci.back().second.back().second.push_back(0);
//							rangeReca.back().second.back().second.push_back(0);
//							rangeFpr.back().second.back().second.push_back(1.0);
//							rangeAcc.back().second.back().second.push_back(0);
//
//							rangePreci.back().second.back().first += mymatcher->inlRatioO;
//							rangeReca.back().second.back().first += mymatcher->inlRatioO;
//							rangeFpr.back().second.back().first += mymatcher->inlRatioO;
//							rangeAcc.back().second.back().first += mymatcher->inlRatioO;
//						}
//						continue;
//					}
//
//					sumFeatures += mymatcher->positivesGT + mymatcher->negativesGTl;
//					sumMatchTime += mymatcher->tm;
//
//					rangePreci.back().second.back().second.push_back(mymatcher->qpm.ppv);
//					rangeReca.back().second.back().second.push_back(mymatcher->qpm.tpr);
//					rangeFpr.back().second.back().second.push_back(mymatcher->qpm.fpr);
//					rangeAcc.back().second.back().second.push_back(mymatcher->qpm.acc);
//
//					rangePreci.back().second.back().first += mymatcher->inlRatioO;
//					rangeReca.back().second.back().first += mymatcher->inlRatioO;
//					rangeFpr.back().second.back().first += mymatcher->inlRatioO;
//					rangeAcc.back().second.back().first += mymatcher->inlRatioO;
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_stdMult_flow_idx" + std::to_string((ULONGLONG)i) + "_mult" + 
//										std::to_string((ULONGLONG)(floor(actStdMult * 100.0 +0.5))) +
//										"_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + "_" + 
//										featureDetector + "_" + descriptorExtractor + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(actInlRat*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//				}
//				if(rangePreci.back().second.back().second.size() > 0)
//				{
//					rangePreci.back().second.back().first /= (double)rangePreci.back().second.back().second.size();
//					rangeReca.back().second.back().first /= (double)rangeReca.back().second.back().second.size();
//					rangeFpr.back().second.back().first /= (double)rangeFpr.back().second.back().second.size();
//					rangeAcc.back().second.back().first /= (double)rangeAcc.back().second.back().second.size();
//				}
//				else
//				{
//					rangePreci.back().second.back().first = actInlRat;
//					rangeReca.back().second.back().first = actInlRat;
//					rangeFpr.back().second.back().first = actInlRat;
//					rangeAcc.back().second.back().first = actInlRat;
//				}
//				rangeAvgMatchTime.back().second.push_back(std::make_pair(rangePreci.back().second.back().first, sumMatchTime / sumFeatures));
//				if(inlRatio <= 0)
//				{
//					actInlRat += InlRatChange;
//					inlRatCnt++;
//				}
//			}
//			actStdMult += stdMultChange;
//		}
//		if(rangePreci.back().second.back().second.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		for(int i = 0; i < rangePreci.size(); i++)
//		{
//			rangePreciStat.push_back(make_pair(rangePreci[i].first,vector<pair<double,qualityParm1>>()));
//			rangeRecaStat.push_back(make_pair(rangeReca[i].first,vector<pair<double,qualityParm1>>()));
//			rangeFprStat.push_back(make_pair(rangeFpr[i].first,vector<pair<double,qualityParm1>>()));
//			rangeAccStat.push_back(make_pair(rangeAcc[i].first,vector<pair<double,qualityParm1>>()));
//
//			for(int j = 0; j < rangePreci[i].second.size(); j++)
//			{
//				getStatisticfromVec2(rangePreci[i].second[j].second, &stats_tmp, false);
//				rangePreciStat.back().second.push_back(make_pair(rangePreci[i].second[j].first, stats_tmp));
//
//				getStatisticfromVec2(rangeReca[i].second[j].second, &stats_tmp, false);
//				rangeRecaStat.back().second.push_back(make_pair(rangeReca[i].second[j].first, stats_tmp));
//
//				getStatisticfromVec2(rangeFpr[i].second[j].second, &stats_tmp, false);
//				rangeFprStat.back().second.push_back(make_pair(rangeFpr[i].second[j].first, stats_tmp));
//
//				getStatisticfromVec2(rangeAcc[i].second[j].second, &stats_tmp, false);
//				rangeAccStat.back().second.push_back(make_pair(rangeAcc[i].second[j].first, stats_tmp));
//			}
//		}
//		
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		
//		//Output of 2D plots with specific inlier ratio for the std. multiplication factors and their statistical values of PPV, TPR, FPR, ACC
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_flow_preci_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_flow_preci_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangePreciStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangePreciStat[j].second.size())
//							break;
//						if(abs(rangePreciStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) with an inlier ratio of " << rangePreciStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangePreciStat[j].first << " " << rangePreciStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangePreciStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangePreci[j].second[inlRat2DIdx].second.begin(), rangePreci[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangePreci[j].second[inlRat2DIdx].second.begin(), rangePreci[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangePreciStat[j].second[inlRat2DIdx].second.arithStd << " " << rangePreciStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangePreciStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangePreciStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_flow_recall_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_flow_recall_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeRecaStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeRecaStat[j].second.size())
//							break;
//						if(abs(rangeRecaStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg) with an inlier ratio of " << rangeRecaStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangeRecaStat[j].first << " " << rangeRecaStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangeRecaStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangeReca[j].second[inlRat2DIdx].second.begin(), rangeReca[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangeReca[j].second[inlRat2DIdx].second.begin(), rangeReca[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangeRecaStat[j].second[inlRat2DIdx].second.arithStd << " " << rangeRecaStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangeRecaStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangeRecaStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_flow_fpr_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_flow_fpr_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeFprStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeFprStat[j].second.size())
//							break;
//						if(abs(rangeFprStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg) with an inlier ratio of " << rangeFprStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangeFprStat[j].first << " " << rangeFprStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangeFprStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangeFpr[j].second[inlRat2DIdx].second.begin(), rangeFpr[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangeFpr[j].second[inlRat2DIdx].second.begin(), rangeFpr[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangeFprStat[j].second[inlRat2DIdx].second.arithStd << " " << rangeFprStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangeFprStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangeFprStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_flow_acc_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_flow_acc_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeAccStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeAccStat[j].second.size())
//							break;
//						if(abs(rangeAccStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos) with an inlier ratio of " << rangeAccStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangeAccStat[j].first << " " << rangeAccStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangeAccStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangeAcc[j].second[inlRat2DIdx].second.begin(), rangeAcc[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangeAcc[j].second[inlRat2DIdx].second.begin(), rangeAcc[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangeAccStat[j].second[inlRat2DIdx].second.arithStd << " " << rangeAccStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangeAccStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangeAccStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		//Output of 2D plots with specific inlier ratio for the std. multiplication factors and their average matching times per keypoint
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_flow_time_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_flow_time_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeAvgMatchTime.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeAvgMatchTime[j].second.size())
//							break;
//						if(abs(rangeAvgMatchTime[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to the average matching time per keypoint (TP + TN) with an inlier ratio of " << rangeAvgMatchTime[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult time" << endl;
//					}
//
//					evalsToFile << rangeAvgMatchTime[j].first << " " << rangeAvgMatchTime[j].second[inlRat2DIdx].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		//Output of 3D plots with multiple inlier ratios for the std. multiplication factors and their statistical values of PPV, TPR, FPR, ACC as well as the average matching time per keypoint
//		if(inlRatio <= 0)
//		{
//			{
//				outfilename = "tex_stdMult_flow_preci_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) with varying inlier ratios" << endl;
//						evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangePreciStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangePreciStat[i].second.size(); j++)
//					{
//						evalsToFile << rangePreciStat[i].first << " " << rangePreciStat[i].second[j].first << " "
//									<< rangePreciStat[i].second[j].second.arithErr << " "<< rangePreciStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangePreci[i].second[j].second.begin(), rangePreci[i].second[j].second.end()) << " "
//									<< *std::max_element(rangePreci[i].second[j].second.begin(), rangePreci[i].second[j].second.end()) << " "
//									<< rangePreciStat[i].second[j].second.arithStd << " " << rangePreciStat[i].second[j].second.medStd << " "
//									<< rangePreciStat[i].second[j].second.lowerQuart << " " << rangePreciStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_stdMult_flow_recall_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg) with varying inlier ratios" << endl;
//				evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangeRecaStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeRecaStat[i].second.size(); j++)
//					{
//						evalsToFile << rangeRecaStat[i].first << " " << rangeRecaStat[i].second[j].first << " "
//									<< rangeRecaStat[i].second[j].second.arithErr << " "<< rangeRecaStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangeReca[i].second[j].second.begin(), rangeReca[i].second[j].second.end()) << " "
//									<< *std::max_element(rangeReca[i].second[j].second.begin(), rangeReca[i].second[j].second.end()) << " "
//									<< rangeRecaStat[i].second[j].second.arithStd << " " << rangeRecaStat[i].second[j].second.medStd << " "
//									<< rangeRecaStat[i].second[j].second.lowerQuart << " " << rangeRecaStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_stdMult_flow_fpr_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg) with varying inlier ratios" << endl;
//				evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangeFprStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeFprStat[i].second.size(); j++)
//					{
//						evalsToFile << rangeFprStat[i].first << " " << rangeFprStat[i].second[j].first << " "
//									<< rangeFprStat[i].second[j].second.arithErr << " "<< rangeFprStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangeFpr[i].second[j].second.begin(), rangeFpr[i].second[j].second.end()) << " "
//									<< *std::max_element(rangeFpr[i].second[j].second.begin(), rangeFpr[i].second[j].second.end()) << " "
//									<< rangeFprStat[i].second[j].second.arithStd << " " << rangeFprStat[i].second[j].second.medStd << " "
//									<< rangeFprStat[i].second[j].second.lowerQuart << " " << rangeFprStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_stdMult_flow_acc_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos) with varying inlier ratios" << endl;
//						evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangeAccStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeAccStat[i].second.size(); j++)
//					{
//						evalsToFile << rangeAccStat[i].first << " " << rangeAccStat[i].second[j].first << " "
//									<< rangeAccStat[i].second[j].second.arithErr << " "<< rangeAccStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangeAcc[i].second[j].second.begin(), rangeAcc[i].second[j].second.end()) << " "
//									<< *std::max_element(rangeAcc[i].second[j].second.begin(), rangeAcc[i].second[j].second.end()) << " "
//									<< rangeAccStat[i].second[j].second.arithStd << " " << rangeAccStat[i].second[j].second.medStd << " "
//									<< rangeAccStat[i].second[j].second.lowerQuart << " " << rangeAccStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			//Time output
//			{
//				outfilename = "tex_stdMult_flow_time_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to the average matching time per keypoint (TP + TN) with varying inlier ratios" << endl;
//				evalsToFile << "stdMult inlRat time" << endl;
//				for(int i = 0; i < (int)rangeAvgMatchTime.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeAvgMatchTime[i].second.size(); j++)
//					{
//						evalsToFile << rangeAvgMatchTime[i].first << " " << rangeAvgMatchTime[i].second[j].first << " "
//									<< rangeAvgMatchTime[i].second[j].second << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else if(flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		//For all different standard deviation multiplication factors do
//		actStdMult = minStdMult;
//		while(actStdMult <= maxStdMult)
//		{
//			rangePreci.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//			rangeReca.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//			rangeFpr.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//			rangeAcc.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//			rangeAvgMatchTime.push_back(std::make_pair(actStdMult,vector<pair<double,double>>()));
//
//			//For all different inlier ratios do
//			actInlRat = minInlRat;
//			inlRatCnt = 0;
//			while((actInlRat <= maxInlRat) && (inlRatCnt >= 0))
//			{
//				if(inlRatio > 0)
//				{
//					inlRatCnt = -1.0;
//					actInlRat = inlRatio;
//				}
//				rangePreci.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//				rangeReca.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//				rangeFpr.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//				rangeAcc.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//
//				sumFeatures = 0;
//				sumMatchTime = 0;
//				for(int i = 0; i < (int)filenamesl.size(); i++)
//				{
//					cv::Mat flowimg;
//					src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//					src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//					err = convertImageDisparityFile(flowDispHPath, filenamesflow[i], &flowimg);
//					if(err)
//					{
//						cout << "Could not open disparity file with index " << i << endl;
//						continue;
//					}
//
//					//mymatcher = std::auto_ptr<GMbSOF_matcher>(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], thPreci[j].first));
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i], validationTh, actStdMult));
//
//					if(mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					err = mymatcher->performMatching(actInlRat, true);
//					if(err)
//					{
//						if( err == -2)
//						{
//							rangePreci.back().second.back().second.push_back(0);
//							rangeReca.back().second.back().second.push_back(0);
//							rangeFpr.back().second.back().second.push_back(1.0);
//							rangeAcc.back().second.back().second.push_back(0);
//
//							rangePreci.back().second.back().first += mymatcher->inlRatioO;
//							rangeReca.back().second.back().first += mymatcher->inlRatioO;
//							rangeFpr.back().second.back().first += mymatcher->inlRatioO;
//							rangeAcc.back().second.back().first += mymatcher->inlRatioO;
//						}
//						continue;
//					}
//
//					sumFeatures += mymatcher->positivesGT + mymatcher->negativesGTl;
//					sumMatchTime += mymatcher->tm;
//
//					rangePreci.back().second.back().second.push_back(mymatcher->qpm.ppv);
//					rangeReca.back().second.back().second.push_back(mymatcher->qpm.tpr);
//					rangeFpr.back().second.back().second.push_back(mymatcher->qpm.fpr);
//					rangeAcc.back().second.back().second.push_back(mymatcher->qpm.acc);
//
//					rangePreci.back().second.back().first += mymatcher->inlRatioO;
//					rangeReca.back().second.back().first += mymatcher->inlRatioO;
//					rangeFpr.back().second.back().first += mymatcher->inlRatioO;
//					rangeAcc.back().second.back().first += mymatcher->inlRatioO;
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_stdMult_disp_idx" + std::to_string((ULONGLONG)i) + "_mult" + 
//										std::to_string((ULONGLONG)(floor(actStdMult * 100.0 +0.5))) +
//										"_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + "_" + 
//										featureDetector + "_" + descriptorExtractor + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(actInlRat*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//				}
//				if(rangePreci.back().second.back().second.size() > 0)
//				{
//					rangePreci.back().second.back().first /= (double)rangePreci.back().second.back().second.size();
//					rangeReca.back().second.back().first /= (double)rangeReca.back().second.back().second.size();
//					rangeFpr.back().second.back().first /= (double)rangeFpr.back().second.back().second.size();
//					rangeAcc.back().second.back().first /= (double)rangeAcc.back().second.back().second.size();
//				}
//				else
//				{
//					rangePreci.back().second.back().first = actInlRat;
//					rangeReca.back().second.back().first = actInlRat;
//					rangeFpr.back().second.back().first = actInlRat;
//					rangeAcc.back().second.back().first = actInlRat;
//				}
//				rangeAvgMatchTime.back().second.push_back(std::make_pair(rangePreci.back().second.back().first, sumMatchTime / sumFeatures));
//				if(inlRatio <= 0)
//				{
//					actInlRat += InlRatChange;
//					inlRatCnt++;
//				}
//			}
//			actStdMult += stdMultChange;
//		}
//		if(rangePreci.back().second.back().second.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		for(int i = 0; i < rangePreci.size(); i++)
//		{
//			rangePreciStat.push_back(make_pair(rangePreci[i].first,vector<pair<double,qualityParm1>>()));
//			rangeRecaStat.push_back(make_pair(rangeReca[i].first,vector<pair<double,qualityParm1>>()));
//			rangeFprStat.push_back(make_pair(rangeFpr[i].first,vector<pair<double,qualityParm1>>()));
//			rangeAccStat.push_back(make_pair(rangeAcc[i].first,vector<pair<double,qualityParm1>>()));
//
//			for(int j = 0; j < rangePreci[i].second.size(); j++)
//			{
//				getStatisticfromVec2(rangePreci[i].second[j].second, &stats_tmp, false);
//				rangePreciStat.back().second.push_back(make_pair(rangePreci[i].second[j].first, stats_tmp));
//
//				getStatisticfromVec2(rangeReca[i].second[j].second, &stats_tmp, false);
//				rangeRecaStat.back().second.push_back(make_pair(rangeReca[i].second[j].first, stats_tmp));
//
//				getStatisticfromVec2(rangeFpr[i].second[j].second, &stats_tmp, false);
//				rangeFprStat.back().second.push_back(make_pair(rangeFpr[i].second[j].first, stats_tmp));
//
//				getStatisticfromVec2(rangeAcc[i].second[j].second, &stats_tmp, false);
//				rangeAccStat.back().second.push_back(make_pair(rangeAcc[i].second[j].first, stats_tmp));
//			}
//		}
//		
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//
//		//Output of 2D plots with specific inlier ratio for the std. multiplication factors and their statistical values of PPV, TPR, FPR, ACC
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_disp_preci_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_disp_preci_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangePreciStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangePreciStat[j].second.size())
//							break;
//						if(abs(rangePreciStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) with an inlier ratio of " << rangePreciStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangePreciStat[j].first << " " << rangePreciStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangePreciStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangePreci[j].second[inlRat2DIdx].second.begin(), rangePreci[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangePreci[j].second[inlRat2DIdx].second.begin(), rangePreci[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangePreciStat[j].second[inlRat2DIdx].second.arithStd << " " << rangePreciStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangePreciStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangePreciStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_disp_recall_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_disp_recall_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeRecaStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeRecaStat[j].second.size())
//							break;
//						if(abs(rangeRecaStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg) with an inlier ratio of " << rangeRecaStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangeRecaStat[j].first << " " << rangeRecaStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangeRecaStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangeReca[j].second[inlRat2DIdx].second.begin(), rangeReca[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangeReca[j].second[inlRat2DIdx].second.begin(), rangeReca[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangeRecaStat[j].second[inlRat2DIdx].second.arithStd << " " << rangeRecaStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangeRecaStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangeRecaStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_disp_fpr_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_disp_fpr_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeFprStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeFprStat[j].second.size())
//							break;
//						if(abs(rangeFprStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg) with an inlier ratio of " << rangeFprStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangeFprStat[j].first << " " << rangeFprStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangeFprStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangeFpr[j].second[inlRat2DIdx].second.begin(), rangeFpr[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangeFpr[j].second[inlRat2DIdx].second.begin(), rangeFpr[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangeFprStat[j].second[inlRat2DIdx].second.arithStd << " " << rangeFprStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangeFprStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangeFprStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_disp_acc_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_disp_acc_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeAccStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeAccStat[j].second.size())
//							break;
//						if(abs(rangeAccStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos) with an inlier ratio of " << rangeAccStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangeAccStat[j].first << " " << rangeAccStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangeAccStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangeAcc[j].second[inlRat2DIdx].second.begin(), rangeAcc[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangeAcc[j].second[inlRat2DIdx].second.begin(), rangeAcc[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangeAccStat[j].second[inlRat2DIdx].second.arithStd << " " << rangeAccStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangeAccStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangeAccStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		//Output of 2D plots with specific inlier ratio for the std. multiplication factors and their average matching times per keypoint
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_disp_time_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_disp_time_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeAvgMatchTime.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeAvgMatchTime[j].second.size())
//							break;
//						if(abs(rangeAvgMatchTime[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to the average matching time per keypoint (TP + TN) with an inlier ratio of " << rangeAvgMatchTime[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult time" << endl;
//					}
//
//					evalsToFile << rangeAvgMatchTime[j].first << " " << rangeAvgMatchTime[j].second[inlRat2DIdx].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		//Output of 3D plots with multiple inlier ratios for the std. multiplication factors and their statistical values of PPV, TPR, FPR, ACC
//		if(inlRatio <= 0)
//		{
//			{
//				outfilename = "tex_stdMult_disp_preci_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) with varying inlier ratios" << endl;
//						evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangePreciStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangePreciStat[i].second.size(); j++)
//					{
//						evalsToFile << rangePreciStat[i].first << " " << rangePreciStat[i].second[j].first << " "
//									<< rangePreciStat[i].second[j].second.arithErr << " "<< rangePreciStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangePreci[i].second[j].second.begin(), rangePreci[i].second[j].second.end()) << " "
//									<< *std::max_element(rangePreci[i].second[j].second.begin(), rangePreci[i].second[j].second.end()) << " "
//									<< rangePreciStat[i].second[j].second.arithStd << " " << rangePreciStat[i].second[j].second.medStd << " "
//									<< rangePreciStat[i].second[j].second.lowerQuart << " " << rangePreciStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_stdMult_disp_recall_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg) with varying inlier ratios" << endl;
//				evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangeRecaStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeRecaStat[i].second.size(); j++)
//					{
//						evalsToFile << rangeRecaStat[i].first << " " << rangeRecaStat[i].second[j].first << " "
//									<< rangeRecaStat[i].second[j].second.arithErr << " "<< rangeRecaStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangeReca[i].second[j].second.begin(), rangeReca[i].second[j].second.end()) << " "
//									<< *std::max_element(rangeReca[i].second[j].second.begin(), rangeReca[i].second[j].second.end()) << " "
//									<< rangeRecaStat[i].second[j].second.arithStd << " " << rangeRecaStat[i].second[j].second.medStd << " "
//									<< rangeRecaStat[i].second[j].second.lowerQuart << " " << rangeRecaStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_stdMult_disp_fpr_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg) with varying inlier ratios" << endl;
//				evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangeFprStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeFprStat[i].second.size(); j++)
//					{
//						evalsToFile << rangeFprStat[i].first << " " << rangeFprStat[i].second[j].first << " "
//									<< rangeFprStat[i].second[j].second.arithErr << " "<< rangeFprStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangeFpr[i].second[j].second.begin(), rangeFpr[i].second[j].second.end()) << " "
//									<< *std::max_element(rangeFpr[i].second[j].second.begin(), rangeFpr[i].second[j].second.end()) << " "
//									<< rangeFprStat[i].second[j].second.arithStd << " " << rangeFprStat[i].second[j].second.medStd << " "
//									<< rangeFprStat[i].second[j].second.lowerQuart << " " << rangeFprStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_stdMult_disp_acc_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos) with varying inlier ratios" << endl;
//						evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangeAccStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeAccStat[i].second.size(); j++)
//					{
//						evalsToFile << rangeAccStat[i].first << " " << rangeAccStat[i].second[j].first << " "
//									<< rangeAccStat[i].second[j].second.arithErr << " "<< rangeAccStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangeAcc[i].second[j].second.begin(), rangeAcc[i].second[j].second.end()) << " "
//									<< *std::max_element(rangeAcc[i].second[j].second.begin(), rangeAcc[i].second[j].second.end()) << " "
//									<< rangeAccStat[i].second[j].second.arithStd << " " << rangeAccStat[i].second[j].second.medStd << " "
//									<< rangeAccStat[i].second[j].second.lowerQuart << " " << rangeAccStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			//Time output
//			{
//				outfilename = "tex_stdMult_disp_time_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to the average matching time per keypoint (TP + TN) with varying inlier ratios" << endl;
//				evalsToFile << "stdMult inlRat time" << endl;
//				for(int i = 0; i < (int)rangeAvgMatchTime.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeAvgMatchTime[i].second.size(); j++)
//					{
//						evalsToFile << rangeAvgMatchTime[i].first << " " << rangeAvgMatchTime[i].second[j].first << " "
//									<< rangeAvgMatchTime[i].second[j].second << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else if(flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		//cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if(err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if(err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if(err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		if(fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//
//			//For all different standard deviation multiplication factors do
//			actStdMult = minStdMult;
//			while(actStdMult <= maxStdMult)
//			{
//				rangePreci.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//				rangeReca.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//				rangeFpr.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//				rangeAcc.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//				rangeAvgMatchTime.push_back(std::make_pair(actStdMult,vector<pair<double,double>>()));
//
//				//For all different inlier ratios do
//				actInlRat = minInlRat;
//				inlRatCnt = 0;
//				while((actInlRat <= maxInlRat) && (inlRatCnt >= 0))
//				{
//					if(inlRatio > 0)
//					{
//						inlRatCnt = -1.0;
//						actInlRat = inlRatio;
//					}
//					rangePreci.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//					rangeReca.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//					rangeFpr.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//					rangeAcc.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//
//					sumFeatures = 0;
//					sumMatchTime = 0;
//					for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//					{
//						cv::Mat H = Hs[idx1];
//						src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//						mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], validationTh, actStdMult));
//
//						if(mymatcher->specialGMbSOFtest)
//						{
//							cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//							exit(0);
//						}
//
//						err = mymatcher->performMatching(actInlRat, true);
//						if(err)
//						{
//							if( err == -2)
//							{
//								rangePreci.back().second.back().second.push_back(0);
//								rangeReca.back().second.back().second.push_back(0);
//								rangeFpr.back().second.back().second.push_back(1.0);
//								rangeAcc.back().second.back().second.push_back(0);
//
//								rangePreci.back().second.back().first += mymatcher->inlRatioO;
//								rangeReca.back().second.back().first += mymatcher->inlRatioO;
//								rangeFpr.back().second.back().first += mymatcher->inlRatioO;
//								rangeAcc.back().second.back().first += mymatcher->inlRatioO;
//							}
//							continue;
//						}
//
//						sumFeatures += mymatcher->positivesGT + mymatcher->negativesGTl;
//						sumMatchTime += mymatcher->tm;
//
//						rangePreci.back().second.back().second.push_back(mymatcher->qpm.ppv);
//						rangeReca.back().second.back().second.push_back(mymatcher->qpm.tpr);
//						rangeFpr.back().second.back().second.push_back(mymatcher->qpm.fpr);
//						rangeAcc.back().second.back().second.push_back(mymatcher->qpm.acc);
//
//						rangePreci.back().second.back().first += mymatcher->inlRatioO;
//						rangeReca.back().second.back().first += mymatcher->inlRatioO;
//						rangeFpr.back().second.back().first += mymatcher->inlRatioO;
//						rangeAcc.back().second.back().first += mymatcher->inlRatioO;
//
//						if(showResult >= 0)
//						{
//							if(storeImgResPath.empty())
//							{
//								mymatcher->showMatches(showResult);
//							}
//							else
//							{
//								if(dirExists(storeImgResPath)) //Check if output directory existis
//								{
//									outpath = storeImgResPath;
//								}
//								else
//								{
//									outpath = imgsPath + "\\evalImgs";
//									if(!dirExists(outpath))
//										_mkdir(outpath.c_str());
//								}
//								outfilename = "img_stdMult_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_mult" + 
//											std::to_string((ULONGLONG)(floor(actStdMult * 100.0 +0.5))) +
//											"_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + "_" + 
//											featureDetector + "_" + descriptorExtractor + "inlRat" + 
//											std::to_string((ULONGLONG)std::floor(actInlRat*100.0 + 0.5)) + ".bmp";
//								mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//							}
//						}
//					}
//					//Generate new homographys to evaluate all other possible configurations of the images to each other
//					for(int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//					{
//						for(int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//						{
//							//H = (Hs[idx2].inv() * Hs[idx1]).inv();
//							cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//							src[0] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//							src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx2 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//							mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], validationTh, actStdMult));
//
//							if(mymatcher->specialGMbSOFtest)
//							{
//								cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//								exit(0);
//							}
//
//							err = mymatcher->performMatching(actInlRat, true);
//							if(err)
//							{
//								if( err == -2)
//								{
//									rangePreci.back().second.back().second.push_back(0);
//									rangeReca.back().second.back().second.push_back(0);
//									rangeFpr.back().second.back().second.push_back(1.0);
//									rangeAcc.back().second.back().second.push_back(0);
//
//									rangePreci.back().second.back().first += mymatcher->inlRatioO;
//									rangeReca.back().second.back().first += mymatcher->inlRatioO;
//									rangeFpr.back().second.back().first += mymatcher->inlRatioO;
//									rangeAcc.back().second.back().first += mymatcher->inlRatioO;
//								}
//								continue;
//							}
//
//							sumFeatures += mymatcher->positivesGT + mymatcher->negativesGTl;
//							sumMatchTime += mymatcher->tm;
//
//							rangePreci.back().second.back().second.push_back(mymatcher->qpm.ppv);
//							rangeReca.back().second.back().second.push_back(mymatcher->qpm.tpr);
//							rangeFpr.back().second.back().second.push_back(mymatcher->qpm.fpr);
//							rangeAcc.back().second.back().second.push_back(mymatcher->qpm.acc);
//
//							rangePreci.back().second.back().first += mymatcher->inlRatioO;
//							rangeReca.back().second.back().first += mymatcher->inlRatioO;
//							rangeFpr.back().second.back().first += mymatcher->inlRatioO;
//							rangeAcc.back().second.back().first += mymatcher->inlRatioO;
//
//							if(showResult >= 0)
//							{
//								if(storeImgResPath.empty())
//								{
//									mymatcher->showMatches(showResult);
//								}
//								else
//								{
//									if(dirExists(storeImgResPath)) //Check if output directory existis
//									{
//										outpath = storeImgResPath;
//									}
//									else
//									{
//										outpath = imgsPath + "\\evalImgs";
//										if(!dirExists(outpath))
//											_mkdir(outpath.c_str());
//									}
//									outfilename = "img_stdMult_H_idxs" + std::to_string((ULONGLONG)(idx1 + 1)) + "-" + std::to_string((ULONGLONG)(idx2 + 1)) + "_mult" + 
//												std::to_string((ULONGLONG)(floor(actStdMult * 100.0 +0.5))) +
//												"_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + "_" + 
//												featureDetector + "_" + descriptorExtractor + "inlRat" + 
//												std::to_string((ULONGLONG)std::floor(actInlRat*100.0 + 0.5)) + ".bmp";
//									mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//								}
//							}
//						}
//					}
//					if(rangePreci.back().second.back().second.size() > 0)
//					{
//						rangePreci.back().second.back().first /= (double)rangePreci.back().second.back().second.size();
//						rangeReca.back().second.back().first /= (double)rangeReca.back().second.back().second.size();
//						rangeFpr.back().second.back().first /= (double)rangeFpr.back().second.back().second.size();
//						rangeAcc.back().second.back().first /= (double)rangeAcc.back().second.back().second.size();
//					}
//					else
//					{
//						rangePreci.back().second.back().first = actInlRat;
//						rangeReca.back().second.back().first = actInlRat;
//						rangeFpr.back().second.back().first = actInlRat;
//						rangeAcc.back().second.back().first = actInlRat;
//					}
//					rangeAvgMatchTime.back().second.push_back(std::make_pair(rangePreci.back().second.back().first, sumMatchTime / sumFeatures));
//					if(inlRatio <= 0)
//					{
//						actInlRat += InlRatChange;
//						inlRatCnt++;
//					}
//				}
//				actStdMult += stdMultChange;
//			}
//		}
//		else
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//
//			//For all different standard deviation multiplication factors do
//			actStdMult = minStdMult;
//			while(actStdMult <= maxStdMult)
//			{
//				rangePreci.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//				rangeReca.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//				rangeFpr.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//				rangeAcc.push_back(std::make_pair(actStdMult,vector<pair<double,vector<double>>>()));
//				rangeAvgMatchTime.push_back(std::make_pair(actStdMult,vector<pair<double,double>>()));
//
//				//For all different inlier ratios do
//				actInlRat = minInlRat;
//				inlRatCnt = 0;
//				while((actInlRat <= maxInlRat) && (inlRatCnt >= 0))
//				{
//					if(inlRatio > 0)
//					{
//						inlRatCnt = -1.0;
//						actInlRat = inlRatio;
//					}
//					rangePreci.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//					rangeReca.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//					rangeFpr.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//					rangeAcc.back().second.push_back(std::make_pair<double,vector<double>>(0,vector<double>()));
//
//					sumFeatures = 0;
//					sumMatchTime = 0;
//					for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//					{
//						cv::Mat H = Hs[idx1];
//						src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//						mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], validationTh, actStdMult));
//
//						if(mymatcher->specialGMbSOFtest)
//						{
//							cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//							exit(0);
//						}
//
//						err = mymatcher->performMatching(actInlRat, true);
//						if(err)
//						{
//							if( err == -2)
//							{
//								rangePreci.back().second.back().second.push_back(0);
//								rangeReca.back().second.back().second.push_back(0);
//								rangeFpr.back().second.back().second.push_back(1.0);
//								rangeAcc.back().second.back().second.push_back(0);
//
//								rangePreci.back().second.back().first += mymatcher->inlRatioO;
//								rangeReca.back().second.back().first += mymatcher->inlRatioO;
//								rangeFpr.back().second.back().first += mymatcher->inlRatioO;
//								rangeAcc.back().second.back().first += mymatcher->inlRatioO;
//							}
//							continue;
//						}
//
//						sumFeatures += mymatcher->positivesGT + mymatcher->negativesGTl;
//						sumMatchTime += mymatcher->tm;
//
//						rangePreci.back().second.back().second.push_back(mymatcher->qpm.ppv);
//						rangeReca.back().second.back().second.push_back(mymatcher->qpm.tpr);
//						rangeFpr.back().second.back().second.push_back(mymatcher->qpm.fpr);
//						rangeAcc.back().second.back().second.push_back(mymatcher->qpm.acc);
//
//						rangePreci.back().second.back().first += mymatcher->inlRatioO;
//						rangeReca.back().second.back().first += mymatcher->inlRatioO;
//						rangeFpr.back().second.back().first += mymatcher->inlRatioO;
//						rangeAcc.back().second.back().first += mymatcher->inlRatioO;
//
//						if(showResult >= 0)
//						{
//							if(storeImgResPath.empty())
//							{
//								mymatcher->showMatches(showResult);
//							}
//							else
//							{
//								if(dirExists(storeImgResPath)) //Check if output directory existis
//								{
//									outpath = storeImgResPath;
//								}
//								else
//								{
//									outpath = imgsPath + "\\evalImgs";
//									if(!dirExists(outpath))
//										_mkdir(outpath.c_str());
//								}
//								outfilename = "img_stdMult_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_mult" + 
//											std::to_string((ULONGLONG)(floor(actStdMult * 100.0 +0.5))) +
//											"_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + "_" + 
//											featureDetector + "_" + descriptorExtractor + "inlRat" + 
//											std::to_string((ULONGLONG)std::floor(actInlRat*100.0 + 0.5)) + ".bmp";
//								mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//							}
//						}
//					}
//					if(rangePreci.back().second.back().second.size() > 0)
//					{
//						rangePreci.back().second.back().first /= (double)rangePreci.back().second.back().second.size();
//						rangeReca.back().second.back().first /= (double)rangeReca.back().second.back().second.size();
//						rangeFpr.back().second.back().first /= (double)rangeFpr.back().second.back().second.size();
//						rangeAcc.back().second.back().first /= (double)rangeAcc.back().second.back().second.size();
//					}
//					else
//					{
//						rangePreci.back().second.back().first = actInlRat;
//						rangeReca.back().second.back().first = actInlRat;
//						rangeFpr.back().second.back().first = actInlRat;
//						rangeAcc.back().second.back().first = actInlRat;
//					}
//					rangeAvgMatchTime.back().second.push_back(std::make_pair(rangePreci.back().second.back().first, sumMatchTime / sumFeatures));
//					if(inlRatio <= 0)
//					{
//						actInlRat += InlRatChange;
//						inlRatCnt++;
//					}
//				}
//				actStdMult += stdMultChange;
//			}
//		}
//		if(rangePreci.back().second.back().second.empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		for(int i = 0; i < rangePreci.size(); i++)
//		{
//			rangePreciStat.push_back(make_pair(rangePreci[i].first,vector<pair<double,qualityParm1>>()));
//			rangeRecaStat.push_back(make_pair(rangeReca[i].first,vector<pair<double,qualityParm1>>()));
//			rangeFprStat.push_back(make_pair(rangeFpr[i].first,vector<pair<double,qualityParm1>>()));
//			rangeAccStat.push_back(make_pair(rangeAcc[i].first,vector<pair<double,qualityParm1>>()));
//
//			for(int j = 0; j < rangePreci[i].second.size(); j++)
//			{
//				getStatisticfromVec2(rangePreci[i].second[j].second, &stats_tmp, false);
//				rangePreciStat.back().second.push_back(make_pair(rangePreci[i].second[j].first, stats_tmp));
//
//				getStatisticfromVec2(rangeReca[i].second[j].second, &stats_tmp, false);
//				rangeRecaStat.back().second.push_back(make_pair(rangeReca[i].second[j].first, stats_tmp));
//
//				getStatisticfromVec2(rangeFpr[i].second[j].second, &stats_tmp, false);
//				rangeFprStat.back().second.push_back(make_pair(rangeFpr[i].second[j].first, stats_tmp));
//
//				getStatisticfromVec2(rangeAcc[i].second[j].second, &stats_tmp, false);
//				rangeAccStat.back().second.push_back(make_pair(rangeAcc[i].second[j].first, stats_tmp));
//			}
//		}
//		
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//
//		//Output of 2D plots with specific inlier ratio for the std. multiplication factors and their statistical values of PPV, TPR, FPR, ACC
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_H_preci_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_H_preci_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangePreciStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangePreciStat[j].second.size())
//							break;
//						if(abs(rangePreciStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) with an inlier ratio of " << rangePreciStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangePreciStat[j].first << " " << rangePreciStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangePreciStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangePreci[j].second[inlRat2DIdx].second.begin(), rangePreci[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangePreci[j].second[inlRat2DIdx].second.begin(), rangePreci[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangePreciStat[j].second[inlRat2DIdx].second.arithStd << " " << rangePreciStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangePreciStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangePreciStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_H_recall_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_H_recall_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeRecaStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeRecaStat[j].second.size())
//							break;
//						if(abs(rangeRecaStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg) with an inlier ratio of " << rangeRecaStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangeRecaStat[j].first << " " << rangeRecaStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangeRecaStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangeReca[j].second[inlRat2DIdx].second.begin(), rangeReca[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangeReca[j].second[inlRat2DIdx].second.begin(), rangeReca[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangeRecaStat[j].second[inlRat2DIdx].second.arithStd << " " << rangeRecaStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangeRecaStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangeRecaStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_H_fpr_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_H_fpr_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeFprStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeFprStat[j].second.size())
//							break;
//						if(abs(rangeFprStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg) with an inlier ratio of " << rangeFprStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangeFprStat[j].first << " " << rangeFprStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangeFprStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangeFpr[j].second[inlRat2DIdx].second.begin(), rangeFpr[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangeFpr[j].second[inlRat2DIdx].second.begin(), rangeFpr[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangeFprStat[j].second[inlRat2DIdx].second.arithStd << " " << rangeFprStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangeFprStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangeFprStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_H_acc_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_H_acc_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeAccStat.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeAccStat[j].second.size())
//							break;
//						if(abs(rangeAccStat[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos) with an inlier ratio of " << rangeAccStat[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//					}
//
//					evalsToFile << rangeAccStat[j].first << " " << rangeAccStat[j].second[inlRat2DIdx].second.arithErr << " "
//								<< rangeAccStat[j].second[inlRat2DIdx].second.medErr << " "
//								<< *std::min_element(rangeAcc[j].second[inlRat2DIdx].second.begin(), rangeAcc[j].second[inlRat2DIdx].second.end()) << " "
//								<< *std::max_element(rangeAcc[j].second[inlRat2DIdx].second.begin(), rangeAcc[j].second[inlRat2DIdx].second.end()) << " "
//								<< rangeAccStat[j].second[inlRat2DIdx].second.arithStd << " " << rangeAccStat[j].second[inlRat2DIdx].second.medStd << " "
//								<< rangeAccStat[j].second[inlRat2DIdx].second.lowerQuart << " " << rangeAccStat[j].second[inlRat2DIdx].second.upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		//Output of 2D plots with specific inlier ratio for the std. multiplication factors and their average matching times per keypoint
//		{
//			for(int i = 0; i < (inlRatio <= 0 ? 4:1); i++)
//			{
//				int inlRat2DIdx;
//				if(inlRatio <= 0)
//				{
//					inlRat2DIdx = (int)floor((InlRat2DFig[i] - minInlRat) / InlRatChange + 0.5);
//					if(inlRat2DIdx < 0)
//						continue;
//					outfilename = "tex_stdMult_H_time_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(InlRat2DFig[i]*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//				else
//				{
//					inlRat2DIdx = 0;
//					outfilename = "tex_stdMult_H_time_2D_" + featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//								std::to_string((ULONGLONG)std::floor(inlRatio*100.0 + 0.5)) + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				}
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				for(int j = 0; j < rangeAvgMatchTime.size(); j++)
//				{
//					if(inlRatio <= 0)
//					{
//						if(inlRat2DIdx >= (int)rangeAvgMatchTime[j].second.size())
//							break;
//						if(abs(rangeAvgMatchTime[j].second[inlRat2DIdx].first - InlRat2DFig[i]) > 0.07)
//						{
//							if(j == 0)
//								break;
//							else
//								continue;
//						}
//					}
//					if(j == 0)
//					{
//						evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to the average matching time per keypoint (TP + TN) with an inlier ratio of " << rangeAvgMatchTime[0].second[inlRat2DIdx].first << endl;
//						evalsToFile << "stdMult time" << endl;
//					}
//
//					evalsToFile << rangeAvgMatchTime[j].first << " " << rangeAvgMatchTime[j].second[inlRat2DIdx].second << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//		//Output of 3D plots with multiple inlier ratios for the std. multiplication factors and their statistical values of PPV, TPR, FPR, ACC
//		if(inlRatio <= 0)
//		{
//			{
//				outfilename = "tex_stdMult_H_preci_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) with varying inlier ratios" << endl;
//						evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangePreciStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangePreciStat[i].second.size(); j++)
//					{
//						evalsToFile << rangePreciStat[i].first << " " << rangePreciStat[i].second[j].first << " "
//									<< rangePreciStat[i].second[j].second.arithErr << " "<< rangePreciStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangePreci[i].second[j].second.begin(), rangePreci[i].second[j].second.end()) << " "
//									<< *std::max_element(rangePreci[i].second[j].second.begin(), rangePreci[i].second[j].second.end()) << " "
//									<< rangePreciStat[i].second[j].second.arithStd << " " << rangePreciStat[i].second[j].second.medStd << " "
//									<< rangePreciStat[i].second[j].second.lowerQuart << " " << rangePreciStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_stdMult_H_recall_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg) with varying inlier ratios" << endl;
//				evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangeRecaStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeRecaStat[i].second.size(); j++)
//					{
//						evalsToFile << rangeRecaStat[i].first << " " << rangeRecaStat[i].second[j].first << " "
//									<< rangeRecaStat[i].second[j].second.arithErr << " "<< rangeRecaStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangeReca[i].second[j].second.begin(), rangeReca[i].second[j].second.end()) << " "
//									<< *std::max_element(rangeReca[i].second[j].second.begin(), rangeReca[i].second[j].second.end()) << " "
//									<< rangeRecaStat[i].second[j].second.arithStd << " " << rangeRecaStat[i].second[j].second.medStd << " "
//									<< rangeRecaStat[i].second[j].second.lowerQuart << " " << rangeRecaStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_stdMult_H_fpr_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg) with varying inlier ratios" << endl;
//				evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangeFprStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeFprStat[i].second.size(); j++)
//					{
//						evalsToFile << rangeFprStat[i].first << " " << rangeFprStat[i].second[j].first << " "
//									<< rangeFprStat[i].second[j].second.arithErr << " "<< rangeFprStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangeFpr[i].second[j].second.begin(), rangeFpr[i].second[j].second.end()) << " "
//									<< *std::max_element(rangeFpr[i].second[j].second.begin(), rangeFpr[i].second[j].second.end()) << " "
//									<< rangeFprStat[i].second[j].second.arithStd << " " << rangeFprStat[i].second[j].second.medStd << " "
//									<< rangeFprStat[i].second[j].second.lowerQuart << " " << rangeFprStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_stdMult_H_acc_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos) with varying inlier ratios" << endl;
//						evalsToFile << "stdMult inlRat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(int i = 0; i < (int)rangeAccStat.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeAccStat[i].second.size(); j++)
//					{
//						evalsToFile << rangeAccStat[i].first << " " << rangeAccStat[i].second[j].first << " "
//									<< rangeAccStat[i].second[j].second.arithErr << " "<< rangeAccStat[i].second[j].second.medErr << " "
//									<< *std::min_element(rangeAcc[i].second[j].second.begin(), rangeAcc[i].second[j].second.end()) << " "
//									<< *std::max_element(rangeAcc[i].second[j].second.begin(), rangeAcc[i].second[j].second.end()) << " "
//									<< rangeAccStat[i].second[j].second.arithStd << " " << rangeAccStat[i].second[j].second.medStd << " "
//									<< rangeAccStat[i].second[j].second.lowerQuart << " " << rangeAccStat[i].second[j].second.upperQuart << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//			//Time output
//			{
//				outfilename = "tex_stdMult_H_time_3D_" + featureDetector + "_" + descriptorExtractor + "_th" + std::to_string((ULONGLONG)(floor(validationTh * 100.0 +0.5))) + ".dat";
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Multiplication factor for standard deviation (to generate search range) in relation to the average matching time per keypoint (TP + TN) with varying inlier ratios" << endl;
//				evalsToFile << "stdMult inlRat time" << endl;
//				for(int i = 0; i < (int)rangeAvgMatchTime.size(); i++)
//				{
//					for(int j = 0; j < (int)rangeAvgMatchTime[i].second.size(); j++)
//					{
//						evalsToFile << rangeAvgMatchTime[i].first << " " << rangeAvgMatchTime[i].second[j].first << " "
//									<< rangeAvgMatchTime[i].second[j].second << endl;
//					}
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//	return 0;
//}
//
///* Captures the estimated inlier ratio after initial matching and the precision (PPV) of the initial matches and
// * the SOF-filtered initial matches.
// *
// * string imgsPath				Input  -> Path which includes both left and right images
// * string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
// * int flowDispH				Input  -> Indicates which type of ground truth data is used:
// *										  0: flow files from KITTI database
// *										  1: disparity files from KITTI database
// *										  2: homography files (Please note that a homography always relates
// *											 to the first image (e.g. 1->2, 1->3, ...))
// * string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images 
// *									      (after prefix only comes the image number)
// * string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
// *									      (after prefix only comes the image number). For testing with homographies,
// *										  this string can be empty.
// * string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
// *									      (after prefix only comes the image number)
// * string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
// *										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
// *										  detectors are possible.
// * string descriptorExtractor	Input  -> The used descriptor extractor. Possible inputs should only be FREAK
// *										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
// * string storeResultPath		Input  -> Path were the resulting measurements should be stored
// * int showResult				Input  -> If >= 0, the result from the matching algorithm is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * string storeImgResPath		Input  -> Optional path for storing the resulting matches drawn into the images,
// *										  where the options of which results should be drawn are specified in
// *										  "showResult". If this path is set, the images are NOT displayed.
// * 
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Failed
// */
//int testGMbSOFinitMatching(std::string imgsPath, std::string flowDispHPath, int flowDispH, 
//						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//						 std::string featureDetector, std::string descriptorExtractor,
//						 std::string storeResultPath, int showResult, std::string storeImgResPath)
//{
//	int err;
//	cv::Mat src[2];
//	const double inlratboreders[3] = {0.2, 0.1, 0.01};
//	const double inlratdecrease[3] = {0.05, 0.02, 0.01};
//	const double inlratfirstval = 1.0;
//	int nrinlrats = 1;
//
//	nrinlrats += (int)((inlratfirstval - inlratboreders[0]) / inlratdecrease[0]);
//	for(int i = 1; i < 3; i++)
//	{
//		nrinlrats += (int)((inlratboreders[i-1] - inlratboreders[i]) / inlratdecrease[i]);
//	}
//
//	vector<vector<double>> initPreci(nrinlrats), initPreciF(nrinlrats), estiInlRat(nrinlrats); //{[inl1, (q1, ..., qn)], ..., [inln, (q1, ..., qn)]}
//	vector<qualityParm1> initPreciStat(nrinlrats), initPreciFStat(nrinlrats), estiInlRatStat(nrinlrats);
//	vector<double> inlierRatios;
//	std::auto_ptr<GMbSOF_matcher> mymatcher;
//	string outpath, outfilename;
//
//	if(!storeImgResPath.empty() && (showResult == -1))
//	{
//		cout << "If you want to store the resulting images you must specify the showResult flag! Exiting." << endl;
//		exit(0);
//	}
//
//	//Generate inlier ratios
//	double startInlRatio = inlratfirstval;
//	inlierRatios.push_back(startInlRatio);
//	for(int i = 0; i < 3; i++)
//	{
//		while(startInlRatio > inlratboreders[i])
//		{
//			startInlRatio -= inlratdecrease[i];
//			inlierRatios.push_back(startInlRatio);
//		}
//	}
//	
//	if(flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int k = 0; k < nrinlrats; k++)
//		{
//			double inlRat_tmp = inlierRatios[k];
//			for(int i = 0; i < (int)filenamesl.size(); i++)
//			{
//				cv::Mat flowimg;
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//				src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//				err = convertImageFlowFile(flowDispHPath, filenamesflow[i], &flowimg);
//				if(err)
//				{
//					cout << "Could not open flow file with index " << i << endl;
//					continue;
//				}
//				mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i]));
//
//				if(!mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework must specially be compiled for this test! Exiting." << endl;
//					exit(0);
//				}
//
//				err = mymatcher->performMatching(inlierRatios[k]);
//				if(err)
//				{
//					/*if( err == -2)
//					{
//						initPreci[k].push_back(0);
//						initPreciF[k].push_back(0);
//						estiInlRat[k].push_back(0);
//					}*/
//					continue;
//				}
//				inlRat_tmp = mymatcher->inlRatioO;
//
//				initPreci[k].push_back(mymatcher->qpm.ppv);
//				initPreciF[k].push_back(mymatcher->qpr.ppv);
//#if(INITMATCHQUALEVAL_O)
//				estiInlRat[k].push_back(mymatcher->initEstiInlRatio);
//#endif
//
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_initM_flow_idxs" + std::to_string((ULONGLONG)i) + "_" + 
//									featureDetector + "_" + descriptorExtractor + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlierRatios[k]*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//			}
//			inlierRatios[k] = inlRat_tmp;
//		}
//		/*if(initPreci.back().empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}*/
//		
//		for(int k = 0; k < nrinlrats; k++)
//		{
//			getStatisticfromVec2(initPreci[k],&initPreciStat[k],false);
//			getStatisticfromVec2(initPreciF[k],&initPreciFStat[k],false);
//			getStatisticfromVec2(estiInlRat[k],&estiInlRatStat[k],false);
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		
//		//Output of inlier ratios in relation to initial matching statistics
//		{
//			outfilename = "tex_initM_flow_preci_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) statistics of the initial matches" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < initPreciStat.size(); i++)
//			{
//				if(initPreci[i].empty())
//					continue;
//				evalsToFile << inlierRatios[i] << " " << initPreciStat[i].arithErr << " " << initPreciStat[i].medErr << " "
//							<< *std::min_element(initPreci[i].begin(), initPreci[i].end()) << " "
//							<< *std::max_element(initPreci[i].begin(), initPreci[i].end()) << " "
//							<< initPreciStat[i].arithStd << " " << initPreciStat[i].medStd << " "
//							<< initPreciStat[i].lowerQuart << " " << initPreciStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_initMFilt_flow_preci_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) statistics of the filtered initial matches" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < initPreciStat.size(); i++)
//			{
//				if(initPreciF[i].empty())
//					continue;
//				evalsToFile << inlierRatios[i] << " " << initPreciFStat[i].arithErr << " " << initPreciFStat[i].medErr << " "
//							<< *std::min_element(initPreciF[i].begin(), initPreciF[i].end()) << " "
//							<< *std::max_element(initPreciF[i].begin(), initPreciF[i].end()) << " "
//							<< initPreciFStat[i].arithStd << " " << initPreciFStat[i].medStd << " "
//							<< initPreciFStat[i].lowerQuart << " " << initPreciFStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_initM_flow_estiInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio in relation to the statistics of the estimated inlier ratio after initial matching" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < initPreciStat.size(); i++)
//			{
//				if(estiInlRat[i].empty())
//					continue;
//				evalsToFile << inlierRatios[i] << " " << estiInlRatStat[i].arithErr << " " << estiInlRatStat[i].medErr << " "
//							<< *std::min_element(estiInlRat[i].begin(), estiInlRat[i].end()) << " "
//							<< *std::max_element(estiInlRat[i].begin(), estiInlRat[i].end()) << " "
//							<< estiInlRatStat[i].arithStd << " " << estiInlRatStat[i].medStd << " "
//							<< estiInlRatStat[i].lowerQuart << " " << estiInlRatStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//	}
//	else if(flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int k = 0; k < nrinlrats; k++)
//		{
//			double inlRat_tmp = inlierRatios[k];
//			for(int i = 0; i < (int)filenamesl.size(); i++)
//			{
//				cv::Mat flowimg;
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//				src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//				err = convertImageDisparityFile(flowDispHPath, filenamesflow[i], &flowimg);
//				if(err)
//				{
//					cout << "Could not open disparity file with index " << i << endl;
//					continue;
//				}
//				mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i]));
//
//				if(!mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework must specially be compiled for this test! Exiting." << endl;
//					exit(0);
//				}
//
//				err = mymatcher->performMatching(inlierRatios[k]);
//				if(err)
//				{
//					/*if( err == -2)
//					{
//						initPreci[k].push_back(0);
//						initPreciF[k].push_back(0);
//						estiInlRat[k].push_back(0);
//					}*/
//					continue;
//				}
//				inlRat_tmp = mymatcher->inlRatioO;
//
//				initPreci[k].push_back(mymatcher->qpm.ppv);
//				initPreciF[k].push_back(mymatcher->qpr.ppv);
//#if(INITMATCHQUALEVAL_O)
//				estiInlRat[k].push_back(mymatcher->initEstiInlRatio);
//#endif
//
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_initM_disp_idxs" + std::to_string((ULONGLONG)i) + "_" + 
//									featureDetector + "_" + descriptorExtractor + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlierRatios[k]*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//			}
//			inlierRatios[k] = inlRat_tmp;
//		}
//		/*if(initPreci.back().empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}*/
//		
//		for(int k = 0; k < nrinlrats; k++)
//		{
//			getStatisticfromVec2(initPreci[k],&initPreciStat[k],false);
//			getStatisticfromVec2(initPreciF[k],&initPreciFStat[k],false);
//			getStatisticfromVec2(estiInlRat[k],&estiInlRatStat[k],false);
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		
//		//Output of inlier ratios in relation to initial matching statistics
//		{
//			outfilename = "tex_initM_disp_preci_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) statistics of the initial matches" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < initPreciStat.size(); i++)
//			{
//				if(initPreci[i].empty())
//					continue;
//				evalsToFile << inlierRatios[i] << " " << initPreciStat[i].arithErr << " " << initPreciStat[i].medErr << " "
//							<< *std::min_element(initPreci[i].begin(), initPreci[i].end()) << " "
//							<< *std::max_element(initPreci[i].begin(), initPreci[i].end()) << " "
//							<< initPreciStat[i].arithStd << " " << initPreciStat[i].medStd << " "
//							<< initPreciStat[i].lowerQuart << " " << initPreciStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_initMFilt_disp_preci_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) statistics of the filtered initial matches" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < initPreciStat.size(); i++)
//			{
//				if(initPreciF[i].empty())
//					continue;
//				evalsToFile << inlierRatios[i] << " " << initPreciFStat[i].arithErr << " " << initPreciFStat[i].medErr << " "
//							<< *std::min_element(initPreciF[i].begin(), initPreciF[i].end()) << " "
//							<< *std::max_element(initPreciF[i].begin(), initPreciF[i].end()) << " "
//							<< initPreciFStat[i].arithStd << " " << initPreciFStat[i].medStd << " "
//							<< initPreciFStat[i].lowerQuart << " " << initPreciFStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_initM_disp_estiInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio in relation to the statistics of the estimated inlier ratio after initial matching" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < initPreciStat.size(); i++)
//			{
//				if(estiInlRat[i].empty())
//					continue;
//				evalsToFile << inlierRatios[i] << " " << estiInlRatStat[i].arithErr << " " << estiInlRatStat[i].medErr << " "
//							<< *std::min_element(estiInlRat[i].begin(), estiInlRat[i].end()) << " "
//							<< *std::max_element(estiInlRat[i].begin(), estiInlRat[i].end()) << " "
//							<< estiInlRatStat[i].arithStd << " " << estiInlRatStat[i].medStd << " "
//							<< estiInlRatStat[i].lowerQuart << " " << estiInlRatStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//	}
//	else if(flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		//cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if(err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if(err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if(err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		if(fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//		{
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				double inlRat_tmp = inlierRatios[k];
//				//Take the stored homographys and perform evaluation
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//				for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//				{
//					cv::Mat H = Hs[idx1];
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//					if(!mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework must specially be compiled for this test! Exiting." << endl;
//						exit(0);
//					}
//
//					err = mymatcher->performMatching(inlierRatios[k]);
//					if(err)
//					{
//						/*if( err == -2)
//						{
//							initPreci[k].push_back(0);
//							initPreciF[k].push_back(0);
//							estiInlRat[k].push_back(0);
//						}*/
//						continue;
//					}
//					inlRat_tmp = mymatcher->inlRatioO;
//
//					initPreci[k].push_back(mymatcher->qpm.ppv);
//					initPreciF[k].push_back(mymatcher->qpr.ppv);
//#if(INITMATCHQUALEVAL_O)
//					estiInlRat[k].push_back(mymatcher->initEstiInlRatio);
//#endif
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_initM_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + 
//										featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//										std::to_string((ULONGLONG)std::floor(inlierRatios[k]*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//				}
//				//Generate new homographys to evaluate all other possible configurations of the images to each other
//				for(int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//				{
//					for(int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//					{
//						//H = (Hs[idx2].inv() * Hs[idx1]).inv();
//						cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//						src[0] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//						src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx2 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//						mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//
//						if(!mymatcher->specialGMbSOFtest)
//						{
//							cout << "Test framework must specially be compiled for this test! Exiting." << endl;
//							exit(0);
//						}
//
//						err = mymatcher->performMatching(inlierRatios[k]);
//						if(err)
//						{
//							/*if( err == -2)
//							{
//								initPreci[k].push_back(0);
//								initPreciF[k].push_back(0);
//								estiInlRat[k].push_back(0);
//							}*/
//							continue;
//						}
//						inlRat_tmp = mymatcher->inlRatioO;
//
//						initPreci[k].push_back(mymatcher->qpm.ppv);
//						initPreciF[k].push_back(mymatcher->qpr.ppv);
//#if(INITMATCHQUALEVAL_O)
//						estiInlRat[k].push_back(mymatcher->initEstiInlRatio);
//#endif
//
//						if(showResult >= 0)
//						{
//							if(storeImgResPath.empty())
//							{
//								mymatcher->showMatches(showResult);
//							}
//							else
//							{
//								if(dirExists(storeImgResPath)) //Check if output directory existis
//								{
//									outpath = storeImgResPath;
//								}
//								else
//								{
//									outpath = imgsPath + "\\evalImgs";
//									if(!dirExists(outpath))
//										_mkdir(outpath.c_str());
//								}
//								outfilename = "img_initM_H_idxs" + std::to_string((ULONGLONG)(idx1 + 1)) + "-" + std::to_string((ULONGLONG)(idx2 + 1)) + "_" + 
//											featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//											std::to_string((ULONGLONG)std::floor(inlierRatios[k]*100.0 + 0.5)) + ".bmp";
//								mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//							}
//						}
//					}
//				}
//				inlierRatios[k] = inlRat_tmp;
//			}
//		}
//		else
//		{
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				double inlRat_tmp = inlierRatios[k];
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//				for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//				{
//					cv::Mat H = Hs[idx1];
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//				
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//					if(!mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework must specially be compiled for this test! Exiting." << endl;
//						exit(0);
//					}
//
//					err = mymatcher->performMatching(inlierRatios[k]);
//					if(err)
//					{
//						/*if( err == -2)
//						{
//							initPreci[k].push_back(0);
//							initPreciF[k].push_back(0);
//							estiInlRat[k].push_back(0);
//						}*/
//						continue;
//					}
//					inlRat_tmp = mymatcher->inlRatioO;
//
//					initPreci[k].push_back(mymatcher->qpm.ppv);
//					initPreciF[k].push_back(mymatcher->qpr.ppv);
//#if(INITMATCHQUALEVAL_O)
//					estiInlRat[k].push_back(mymatcher->initEstiInlRatio);
//#endif
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_initM_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + 
//										featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//										std::to_string((ULONGLONG)std::floor(inlierRatios[k]*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//				}
//				inlierRatios[k] = inlRat_tmp;
//			}
//		}
//		/*if(initPreci.back().empty())
//		{
//			cout << "Algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}*/
//		
//		for(int k = 0; k < nrinlrats; k++)
//		{
//			getStatisticfromVec2(initPreci[k],&initPreciStat[k],false);
//			getStatisticfromVec2(initPreciF[k],&initPreciFStat[k],false);
//			getStatisticfromVec2(estiInlRat[k],&estiInlRatStat[k],false);
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		
//		//Output of inlier ratios in relation to initial matching statistics
//		{
//			outfilename = "tex_initM_H_preci_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) statistics of the initial matches" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < initPreciStat.size(); i++)
//			{
//				if(initPreci[i].empty())
//					continue;
//				evalsToFile << inlierRatios[i] << " " << initPreciStat[i].arithErr << " " << initPreciStat[i].medErr << " "
//							<< *std::min_element(initPreci[i].begin(), initPreci[i].end()) << " "
//							<< *std::max_element(initPreci[i].begin(), initPreci[i].end()) << " "
//							<< initPreciStat[i].arithStd << " " << initPreciStat[i].medStd << " "
//							<< initPreciStat[i].lowerQuart << " " << initPreciStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_initMFilt_H_preci_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio in relation to Precision or positive predictive value ppv=truePos/(truePos+falsePos) statistics of the filtered initial matches" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < initPreciStat.size(); i++)
//			{
//				if(initPreciF[i].empty())
//					continue;
//				evalsToFile << inlierRatios[i] << " " << initPreciFStat[i].arithErr << " " << initPreciFStat[i].medErr << " "
//							<< *std::min_element(initPreciF[i].begin(), initPreciF[i].end()) << " "
//							<< *std::max_element(initPreciF[i].begin(), initPreciF[i].end()) << " "
//							<< initPreciFStat[i].arithStd << " " << initPreciFStat[i].medStd << " "
//							<< initPreciFStat[i].lowerQuart << " " << initPreciFStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_initM_H_estiInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio in relation to the statistics of the estimated inlier ratio after initial matching" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < initPreciStat.size(); i++)
//			{
//				if(estiInlRat[i].empty())
//					continue;
//				evalsToFile << inlierRatios[i] << " " << estiInlRatStat[i].arithErr << " " << estiInlRatStat[i].medErr << " "
//							<< *std::min_element(estiInlRat[i].begin(), estiInlRat[i].end()) << " "
//							<< *std::max_element(estiInlRat[i].begin(), estiInlRat[i].end()) << " "
//							<< estiInlRatStat[i].arithStd << " " << estiInlRatStat[i].medStd << " "
//							<< estiInlRatStat[i].lowerQuart << " " << estiInlRatStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//	return 0;
//}
//
//
///* Captures the cost and distance (from right keypoint to SOF-estimated position) ratios to the local neighbors of the matches in relation to
// * true and false positives and calculates a cumulative distribution for different inlier ratios.
// *
// * string imgsPath				Input  -> Path which includes both left and right images
// * string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
// * int flowDispH				Input  -> Indicates which type of ground truth data is used:
// *										  0: flow files from KITTI database
// *										  1: disparity files from KITTI database
// *										  2: homography files (Please note that a homography always relates
// *											 to the first image (e.g. 1->2, 1->3, ...))
// * string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images 
// *									      (after prefix only comes the image number)
// * string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
// *									      (after prefix only comes the image number). For testing with homographies,
// *										  this string can be empty.
// * string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
// *									      (after prefix only comes the image number)
// * string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
// *										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
// *										  detectors are possible.
// * string descriptorExtractor	Input  -> The used descriptor extractor. Possible inputs should only be FREAK
// *										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
// * string storeResultPath		Input  -> Path were the resulting measurements should be stored
// * int showResult				Input  -> If >= 0, the result from the matching algorithm is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * string storeImgResPath		Input  -> Optional path for storing the resulting matches drawn into the images,
// *										  where the options of which results should be drawn are specified in
// *										  "showResult". If this path is set, the images are NOT displayed.
// * 
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Failed
// */
//int testGMbSOF_CDratios(std::string imgsPath, std::string flowDispHPath, int flowDispH, 
//						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//						 std::string featureDetector, std::string descriptorExtractor,
//						 std::string storeResultPath, int showResult, std::string storeImgResPath)
//{
//	int err;
//	cv::Mat src[2];
//	const double inlratboreders[3] = {0.4, 0.2, 0.02};
//	const double inlratdecrease[3] = {0.1, 0.05, 0.02};
//	const double inlratfirstval = 1.0;
//	int nrinlrats = 1;
//	const float granularity = 0.2;
//
//#if COSTDISTRATIOEVAL == 0
//	return -1;
//#else
//
//	nrinlrats += (int)floor(((inlratfirstval - inlratboreders[0]) / inlratdecrease[0]) + 0.5);
//	for(int i = 1; i < 3; i++)
//	{
//		nrinlrats += (int)floor(((inlratboreders[i-1] - inlratboreders[i]) / inlratdecrease[i]) + 0.5);
//	}
//
//	vector<vector<float>> costRatios(nrinlrats), distRatios(nrinlrats); //{[inl1, (rat1, ..., ratn)], ..., [inln, (rat1, ..., ratn)]}
//	vector<vector<double>> costCumulDistrTP(nrinlrats), distCumulDistrTP(nrinlrats), costCumulDistrFP(nrinlrats), distCumulDistrFP(nrinlrats);
//	vector<double> costCumulDallInlTP, distCumulDallInlTP, costCumulDallInlFP, distCumulDallInlFP, costOutBinHeightsTP, distOutBinHeightsTP, costOutBinHeightsFP, distOutBinHeightsFP;
//	vector<double> addedCDcumulDistTP, addedCDcumulDistFP, add2threshCDcumulDistTP, add2threshCDcumulDistFP, add1threshCDcumulDistTP, add1threshCDcumulDistFP;
//	vector<vector<bool>> tpfps(nrinlrats);
//	vector<float> cumulDistXvalues, cumulDistXborders;
//	vector<double> inlierRatios, inlierRatios_orig;
//	std::auto_ptr<GMbSOF_matcher> mymatcher;
//	string outpath, outfilename;
//	int maxelementsCostsAll = 0;
//	int maxelementsDistsAll = 0;
//	int maxelementsCosts = 0;
//	int maxelementsDists = 0;
//
//	if(!storeImgResPath.empty() && (showResult == -1))
//	{
//		cout << "If you want to store the resulting images you must specify the showResult flag! Exiting." << endl;
//		exit(0);
//	}
//
//	//Generate inlier ratios
//	double startInlRatio = inlratfirstval;
//	inlierRatios.push_back(startInlRatio);
//	for(int i = 0; i < 3; i++)
//	{
//		while((startInlRatio - inlratboreders[i]) > 1e-4)
//		{
//			startInlRatio -= inlratdecrease[i];
//			inlierRatios.push_back(startInlRatio);
//		}
//	}
//	inlierRatios_orig = inlierRatios;
//	
//	if(flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int k = 0; k < nrinlrats; k++)
//		{
//			double inlRat_tmp = inlierRatios[k];
//			for(int i = 0; i < (int)filenamesl.size(); i++)
//			{
//				cv::Mat flowimg;
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//				src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//				err = convertImageFlowFile(flowDispHPath, filenamesflow[i], &flowimg);
//				if(err)
//				{
//					cout << "Could not open flow file with index " << i << endl;
//					continue;
//				}
//				mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i]));
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				err = mymatcher->performMatching(inlierRatios[k]);
//				if(err)
//				{
//					continue;
//				}
//				inlRat_tmp = mymatcher->inlRatioO;
//
//				if(costRatios[k].empty())
//				{
//					costRatios[k].reserve(mymatcher->costRatios.size());
//					distRatios[k].reserve(mymatcher->distRatios.size());
//					costRatios[k].insert(costRatios[k].end(), mymatcher->costRatios.begin(), mymatcher->costRatios.end());
//					distRatios[k].insert(distRatios[k].end(), mymatcher->distRatios.begin(), mymatcher->distRatios.end());
//					tpfps[k].reserve(mymatcher->tpfp.size());
//					tpfps[k].insert(tpfps[k].end(), mymatcher->tpfp.begin(), mymatcher->tpfp.end());
//				}
//				else
//				{
//					costRatios[k].insert(costRatios[k].end(), mymatcher->costRatios.begin(), mymatcher->costRatios.end());
//					distRatios[k].insert(distRatios[k].end(), mymatcher->distRatios.begin(), mymatcher->distRatios.end());
//					tpfps[k].insert(tpfps[k].end(), mymatcher->tpfp.begin(), mymatcher->tpfp.end());
//				}
//
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_CDratio_flow_idxs" + std::to_string((ULONGLONG)i) + "_" + 
//									featureDetector + "_" + descriptorExtractor + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlierRatios[k]*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//			}
//			inlierRatios[k] = inlRat_tmp;
//		}
//
//		calcMultCumulDistCD(costRatios, distRatios, tpfps, costCumulDistrTP, distCumulDistrTP, costCumulDistrFP, distCumulDistrFP, costCumulDallInlTP, 
//						 distCumulDallInlTP, costCumulDallInlFP, distCumulDallInlFP, costOutBinHeightsTP, distOutBinHeightsTP, costOutBinHeightsFP, distOutBinHeightsFP,
//						 cumulDistXvalues, cumulDistXborders, addedCDcumulDistTP, addedCDcumulDistFP, add2threshCDcumulDistTP, add2threshCDcumulDistFP, add1threshCDcumulDistTP, add1threshCDcumulDistFP,
//						 nrinlrats, granularity, maxelementsCostsAll, maxelementsDistsAll, maxelementsCosts, maxelementsDists);
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		
//		//Output of general cumulative distribution over all inlier ratios
//		{
//			outfilename = "tex_Cratio_TP_flow_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCostsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDallInlTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_TP_flow_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDistsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDallInlTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Cratio_FP_flow_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCostsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDallInlFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_FP_flow_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDistsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDallInlFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		//Output of 2D data for different specific inlier ratios
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Cratio_TP_flow_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCosts; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDistrTP[j][i] / costOutBinHeightsTP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Dratio_TP_flow_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDists; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDistrTP[j][i] / distOutBinHeightsTP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Cratio_FP_flow_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCosts; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDistrFP[j][i] / costOutBinHeightsFP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Dratio_FP_flow_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDists; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDistrFP[j][i] / distOutBinHeightsFP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		//Output of 3D cumulative distributions (3rd dimension is inlier ratio)
//		{
//			outfilename = "tex_Cratio_TP_flow_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsCosts; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << costCumulDistrTP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_TP_flow_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsDists; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << distCumulDistrTP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Cratio_FP_flow_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsCosts; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << costCumulDistrFP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_FP_flow_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsDists; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << distCumulDistrFP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		//Output of summed (cost & distance) ratios as well as filtered summed ratios
//		{
//			outfilename = "tex_Aratio_TP_flow_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < addedCDcumulDistTP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << addedCDcumulDistTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Aratio_FP_flow_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < addedCDcumulDistFP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << addedCDcumulDistFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A1ratio_TP_flow_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add1threshCDcumulDistTP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add1threshCDcumulDistTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A1ratio_FP_flow_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add1threshCDcumulDistFP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add1threshCDcumulDistFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A2ratio_TP_flow_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the double filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.5 and the distance ratio was greater than 2.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add2threshCDcumulDistTP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add2threshCDcumulDistTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A2ratio_FP_flow_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the double filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.5 and the distance ratio was greater than 2.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add2threshCDcumulDistFP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add2threshCDcumulDistFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//	}
//	else if(flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int k = 0; k < nrinlrats; k++)
//		{
//			double inlRat_tmp = inlierRatios[k];
//			for(int i = 0; i < (int)filenamesl.size(); i++)
//			{
//				cv::Mat flowimg;
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//				src[1] = cv::imread(imgsPath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//				err = convertImageDisparityFile(flowDispHPath, filenamesflow[i], &flowimg);
//				if(err)
//				{
//					cout << "Could not open disparity file with index " << i << endl;
//					continue;
//				}
//				mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[i]));
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				err = mymatcher->performMatching(inlierRatios[k]);
//				if(err)
//				{
//					continue;
//				}
//				inlRat_tmp = mymatcher->inlRatioO;
//
//				if(costRatios[k].empty())
//				{
//					costRatios[k].reserve(mymatcher->costRatios.size());
//					distRatios[k].reserve(mymatcher->distRatios.size());
//					costRatios[k].insert(costRatios[k].end(), mymatcher->costRatios.begin(), mymatcher->costRatios.end());
//					distRatios[k].insert(distRatios[k].end(), mymatcher->distRatios.begin(), mymatcher->distRatios.end());
//					tpfps[k].reserve(mymatcher->tpfp.size());
//					tpfps[k].insert(tpfps[k].end(), mymatcher->tpfp.begin(), mymatcher->tpfp.end());
//				}
//				else
//				{
//					costRatios[k].insert(costRatios[k].end(), mymatcher->costRatios.begin(), mymatcher->costRatios.end());
//					distRatios[k].insert(distRatios[k].end(), mymatcher->distRatios.begin(), mymatcher->distRatios.end());
//					tpfps[k].insert(tpfps[k].end(), mymatcher->tpfp.begin(), mymatcher->tpfp.end());
//				}
//
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_CDratio_disp_idxs" + std::to_string((ULONGLONG)i) + "_" + 
//									featureDetector + "_" + descriptorExtractor + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(inlierRatios[k]*100.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//			}
//			inlierRatios[k] = inlRat_tmp;
//		}
//
//		calcMultCumulDistCD(costRatios, distRatios, tpfps, costCumulDistrTP, distCumulDistrTP, costCumulDistrFP, distCumulDistrFP, costCumulDallInlTP, 
//						 distCumulDallInlTP, costCumulDallInlFP, distCumulDallInlFP, costOutBinHeightsTP, distOutBinHeightsTP, costOutBinHeightsFP, distOutBinHeightsFP,
//						 cumulDistXvalues, cumulDistXborders, addedCDcumulDistTP, addedCDcumulDistFP, add2threshCDcumulDistTP, add2threshCDcumulDistFP, add1threshCDcumulDistTP, add1threshCDcumulDistFP,
//						 nrinlrats, granularity, maxelementsCostsAll, maxelementsDistsAll, maxelementsCosts, maxelementsDists);
//		
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		
//		//Output of general cumulative distribution over all inlier ratios
//		{
//			outfilename = "tex_Cratio_TP_disp_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCostsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDallInlTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_TP_disp_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDistsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDallInlTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Cratio_FP_disp_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCostsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDallInlFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_FP_disp_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDistsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDallInlFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		//Output of 2D data for different specific inlier ratios
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Cratio_TP_disp_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCosts; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDistrTP[j][i] / costOutBinHeightsTP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Dratio_TP_disp_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDists; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDistrTP[j][i] / distOutBinHeightsTP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Cratio_FP_disp_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCosts; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDistrFP[j][i] / costOutBinHeightsFP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Dratio_FP_disp_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDists; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDistrFP[j][i] / distOutBinHeightsFP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		//Output of 3D cumulative distributions (3rd dimension is inlier ratio)
//		{
//			outfilename = "tex_Cratio_TP_disp_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsCosts; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << costCumulDistrTP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_TP_disp_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsDists; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << distCumulDistrTP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Cratio_FP_disp_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsCosts; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << costCumulDistrFP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_FP_disp_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsDists; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << distCumulDistrFP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		//Output of summed (cost & distance) ratios as well as filtered summed ratios
//		{
//			outfilename = "tex_Aratio_TP_disp_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < addedCDcumulDistTP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << addedCDcumulDistTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Aratio_FP_disp_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < addedCDcumulDistFP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << addedCDcumulDistFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A1ratio_TP_disp_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add1threshCDcumulDistTP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add1threshCDcumulDistTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A1ratio_FP_disp_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add1threshCDcumulDistFP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add1threshCDcumulDistFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A2ratio_TP_disp_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the double filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.5 and the distance ratio was greater than 2.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add2threshCDcumulDistTP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add2threshCDcumulDistTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A2ratio_FP_disp_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the double filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.5 and the distance ratio was greater than 2.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add2threshCDcumulDistFP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add2threshCDcumulDistFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//	}
//	else if(flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		//cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if(err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if(err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if(err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		if(fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//		{
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				double inlRat_tmp = inlierRatios[k];
//				//Take the stored homographys and perform evaluation
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//				for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//				{
//					cv::Mat H = Hs[idx1];
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//					if(mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					err = mymatcher->performMatching(inlierRatios[k]);
//					if(err)
//					{
//						continue;
//					}
//					inlRat_tmp = mymatcher->inlRatioO;
//
//					if(costRatios[k].empty())
//					{
//						costRatios[k].reserve(mymatcher->costRatios.size());
//						distRatios[k].reserve(mymatcher->distRatios.size());
//						costRatios[k].insert(costRatios[k].end(), mymatcher->costRatios.begin(), mymatcher->costRatios.end());
//						distRatios[k].insert(distRatios[k].end(), mymatcher->distRatios.begin(), mymatcher->distRatios.end());
//						tpfps[k].reserve(mymatcher->tpfp.size());
//						tpfps[k].insert(tpfps[k].end(), mymatcher->tpfp.begin(), mymatcher->tpfp.end());
//					}
//					else
//					{
//						costRatios[k].insert(costRatios[k].end(), mymatcher->costRatios.begin(), mymatcher->costRatios.end());
//						distRatios[k].insert(distRatios[k].end(), mymatcher->distRatios.begin(), mymatcher->distRatios.end());
//						tpfps[k].insert(tpfps[k].end(), mymatcher->tpfp.begin(), mymatcher->tpfp.end());
//					}
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_CDratio_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + 
//										featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//										std::to_string((ULONGLONG)std::floor(inlierRatios[k]*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//				}
//				//Generate new homographys to evaluate all other possible configurations of the images to each other
//				for(int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//				{
//					for(int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//					{
//						//H = (Hs[idx2].inv() * Hs[idx1]).inv();
//						cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//						src[0] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//						src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx2 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//						mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//
//						if(mymatcher->specialGMbSOFtest)
//						{
//							cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//							exit(0);
//						}
//
//						err = mymatcher->performMatching(inlierRatios[k]);
//						if(err)
//						{
//							continue;
//						}
//						inlRat_tmp = mymatcher->inlRatioO;
//
//						if(costRatios[k].empty())
//						{
//							costRatios[k].reserve(mymatcher->costRatios.size());
//							distRatios[k].reserve(mymatcher->distRatios.size());
//							costRatios[k].insert(costRatios[k].end(), mymatcher->costRatios.begin(), mymatcher->costRatios.end());
//							distRatios[k].insert(distRatios[k].end(), mymatcher->distRatios.begin(), mymatcher->distRatios.end());
//							tpfps[k].reserve(mymatcher->tpfp.size());
//							tpfps[k].insert(tpfps[k].end(), mymatcher->tpfp.begin(), mymatcher->tpfp.end());
//						}
//						else
//						{
//							costRatios[k].insert(costRatios[k].end(), mymatcher->costRatios.begin(), mymatcher->costRatios.end());
//							distRatios[k].insert(distRatios[k].end(), mymatcher->distRatios.begin(), mymatcher->distRatios.end());
//							tpfps[k].insert(tpfps[k].end(), mymatcher->tpfp.begin(), mymatcher->tpfp.end());
//						}
//
//						if(showResult >= 0)
//						{
//							if(storeImgResPath.empty())
//							{
//								mymatcher->showMatches(showResult);
//							}
//							else
//							{
//								if(dirExists(storeImgResPath)) //Check if output directory existis
//								{
//									outpath = storeImgResPath;
//								}
//								else
//								{
//									outpath = imgsPath + "\\evalImgs";
//									if(!dirExists(outpath))
//										_mkdir(outpath.c_str());
//								}
//								outfilename = "img_CDratio_H_idxs" + std::to_string((ULONGLONG)(idx1 + 1)) + "-" + std::to_string((ULONGLONG)(idx2 + 1)) + "_" + 
//											featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//											std::to_string((ULONGLONG)std::floor(inlierRatios[k]*100.0 + 0.5)) + ".bmp";
//								mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//							}
//						}
//					}
//				}
//				inlierRatios[k] = inlRat_tmp;
//			}
//		}
//		else
//		{
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				double inlRat_tmp = inlierRatios[k];
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//				for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//				{
//					cv::Mat H = Hs[idx1];
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//				
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//					if(mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					err = mymatcher->performMatching(inlierRatios[k]);
//					if(err)
//					{
//						continue;
//					}
//					inlRat_tmp = mymatcher->inlRatioO;
//
//					if(costRatios[k].empty())
//					{
//						costRatios[k].reserve(mymatcher->costRatios.size());
//						distRatios[k].reserve(mymatcher->distRatios.size());
//						costRatios[k].insert(costRatios[k].end(), mymatcher->costRatios.begin(), mymatcher->costRatios.end());
//						distRatios[k].insert(distRatios[k].end(), mymatcher->distRatios.begin(), mymatcher->distRatios.end());
//						tpfps[k].reserve(mymatcher->tpfp.size());
//						tpfps[k].insert(tpfps[k].end(), mymatcher->tpfp.begin(), mymatcher->tpfp.end());
//					}
//					else
//					{
//						costRatios[k].insert(costRatios[k].end(), mymatcher->costRatios.begin(), mymatcher->costRatios.end());
//						distRatios[k].insert(distRatios[k].end(), mymatcher->distRatios.begin(), mymatcher->distRatios.end());
//						tpfps[k].insert(tpfps[k].end(), mymatcher->tpfp.begin(), mymatcher->tpfp.end());
//					}
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_CDratio_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) + "_" + 
//										featureDetector + "_" + descriptorExtractor + "_inlRat" + 
//										std::to_string((ULONGLONG)std::floor(inlierRatios[k]*100.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//				}
//				inlierRatios[k] = inlRat_tmp;
//			}
//		}
//
//		calcMultCumulDistCD(costRatios, distRatios, tpfps, costCumulDistrTP, distCumulDistrTP, costCumulDistrFP, distCumulDistrFP, costCumulDallInlTP, 
//						 distCumulDallInlTP, costCumulDallInlFP, distCumulDallInlFP, costOutBinHeightsTP, distOutBinHeightsTP, costOutBinHeightsFP, distOutBinHeightsFP,
//						 cumulDistXvalues, cumulDistXborders, addedCDcumulDistTP, addedCDcumulDistFP, add2threshCDcumulDistTP, add2threshCDcumulDistFP, add1threshCDcumulDistTP, add1threshCDcumulDistFP,
//						 nrinlrats, granularity, maxelementsCostsAll, maxelementsDistsAll, maxelementsCosts, maxelementsDists);
//		
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		
//		//Output of general cumulative distribution over all inlier ratios
//		{
//			outfilename = "tex_Cratio_TP_H_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCostsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDallInlTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_TP_H_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDistsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDallInlTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Cratio_FP_H_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCostsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDallInlFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_FP_H_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDistsAll; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDallInlFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		//Output of 2D data for different specific inlier ratios
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Cratio_TP_H_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCosts; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDistrTP[j][i] / costOutBinHeightsTP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Dratio_TP_H_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDists; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDistrTP[j][i] / distOutBinHeightsTP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Cratio_FP_H_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsCosts; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << costCumulDistrFP[j][i] / costOutBinHeightsFP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		for(int j = nrinlrats - 1; j > 0; j = j - 2)
//		{
//			outfilename = "tex_Dratio_FP_H_2D_inlRat" + std::to_string((ULONGLONG)std::floor(inlierRatios_orig[j]*100.0 + 0.5)) 
//							+ featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "# An inlier ratio of " << inlierRatios[j] << " was used." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < maxelementsDists; i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << distCumulDistrFP[j][i] / distOutBinHeightsFP[j] << endl;
//			}
//			evalsToFile.close();
//		}
//		//Output of 3D cumulative distributions (3rd dimension is inlier ratio)
//		{
//			outfilename = "tex_Cratio_TP_H_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsCosts; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << costCumulDistrTP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_TP_H_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the true positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsDists; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << distCumulDistrTP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Cratio_FP_H_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the descriptor distance of each match and the median of the descriptor distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsCosts; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << costCumulDistrFP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Dratio_FP_H_3D_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulative distribution over the ratios between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each ones neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the ratios of the false positive matches." << endl;
//			evalsToFile << "inlrat ratio distr" << endl;
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				for(size_t i = 0; i < maxelementsDists; i++)
//				{
//					evalsToFile << inlierRatios[k] << " " << cumulDistXvalues[i] << " " << distCumulDistrFP[k][i] << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		//Output of summed (cost & distance) ratios as well as filtered summed ratios
//		{
//			outfilename = "tex_Aratio_TP_H_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < addedCDcumulDistTP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << addedCDcumulDistTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_Aratio_FP_H_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < addedCDcumulDistFP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << addedCDcumulDistFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A1ratio_TP_H_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add1threshCDcumulDistTP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add1threshCDcumulDistTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A1ratio_FP_H_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add1threshCDcumulDistFP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add1threshCDcumulDistFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A2ratio_TP_H_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the double filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.5 and the distance ratio was greater than 2.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the true positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add2threshCDcumulDistTP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add2threshCDcumulDistTP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_A2ratio_FP_H_2D_allInlRat_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Cumulativ distribution over the double filtered average cost and distance ratio of each match. Multiple inlier ratios were used." << endl;
//			evalsToFile << "# Filtering was performed by only accepting cost & their corresponding distance ratios for which the cost ratio was greater than 1.5 and the distance ratio was greater than 2.0." << endl;
//			evalsToFile << "# The cost ratio corresponds to the ratio between the descriptor distance of a match and the median of the descriptor distances of its neighbors." << endl;
//			evalsToFile << "# The distance ratio corresponds to the ratio between the spatial distance of a matching right keypoint to its SOF-estimated value and the median of the spatial distances of its neighbors." << endl;
//			evalsToFile << "# This file holds the cumulative distribution utilizing the average ratios ( (cost + distance)/2 ) of the false positive matches." << endl;
//			evalsToFile << "ratio distr" << endl;
//			for(size_t i = 0; i < add2threshCDcumulDistFP.size(); i++)
//			{
//				evalsToFile << cumulDistXvalues[i] << " " << add2threshCDcumulDistFP[i] << endl;
//			}
//			evalsToFile.close();
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//	return 0;
//#endif
//}
//
//
//void normalizeCumulDistBins(std::vector<std::vector<double>> & io3Dvec, 
//							std::vector<double> & io2Dvec,
//							int nrinlrats, std::vector<double> *outBinHeights, double *vec2DmaxBinVal, double *takevec2DmaxBinVal)
//{
//	//Normalize the distributions
//	unsigned int maxelements;
//	if(!io3Dvec.empty())
//		maxelements = io3Dvec[0].size();
//	else if(!io2Dvec.empty())
//		maxelements = io2Dvec.size();
//	else
//		return;
//
//	double heighestBin = 0;
//	if(!io3Dvec.empty())
//	{
//		for(int k = 0; k < nrinlrats; k++)
//		{
//			double tmp = *max_element(io3Dvec[k].begin(), io3Dvec[k].end());
//			if(tmp > heighestBin)
//				heighestBin = tmp;
//		}
//		for(int k = 0; k < nrinlrats; k++)
//		{
//			for(unsigned int i = 0; i < maxelements; i++)
//			{
//				io3Dvec[k][i] /= heighestBin;
//			}
//		}
//		if(outBinHeights != NULL)
//		{
//			for(int k = 0; k < nrinlrats; k++)
//			{
//				outBinHeights->push_back(*max_element(io3Dvec[k].begin(), io3Dvec[k].end()));
//			}
//		}
//	}
//	if(!nrinlrats && !io2Dvec.empty())
//	{
//		if(takevec2DmaxBinVal)
//			heighestBin = *takevec2DmaxBinVal;
//		else
//			heighestBin = *max_element(io2Dvec.begin(), io2Dvec.end());
//		for(int i = 0; i < maxelements; i++)
//		{
//			io2Dvec[i] /= heighestBin;
//		}
//		if(vec2DmaxBinVal)
//			*vec2DmaxBinVal = heighestBin;
//	}
//}
//
//
//void calcMultCumulDistCD(std::vector<std::vector<float>> costRatios,
//						 std::vector<std::vector<float>> distRatios,
//						 std::vector<std::vector<bool>> tpfps,
//						 std::vector<std::vector<double>> & costCumulDistrTP,
//						 std::vector<std::vector<double>> & distCumulDistrTP,
//						 std::vector<std::vector<double>> & costCumulDistrFP,
//						 std::vector<std::vector<double>> & distCumulDistrFP,
//						 std::vector<double> & costCumulDallInlTP,
//						 std::vector<double> & distCumulDallInlTP,
//						 std::vector<double> & costCumulDallInlFP,
//						 std::vector<double> & distCumulDallInlFP,
//						 std::vector<double> & costOutBinHeightsTP,
//						 std::vector<double> & distOutBinHeightsTP,
//						 std::vector<double> & costOutBinHeightsFP,
//						 std::vector<double> & distOutBinHeightsFP,
//						 std::vector<float> & cumulDistXvalues,
//						 std::vector<float> & cumulDistXborders,
//						 std::vector<double> & addedCDcumulDistTP, 
//						 std::vector<double> & addedCDcumulDistFP, 
//						 std::vector<double> & add2threshCDcumulDistTP, 
//						 std::vector<double> & add2threshCDcumulDistFP, 
//						 std::vector<double> & add1threshCDcumulDistTP, 
//						 std::vector<double> & add1threshCDcumulDistFP,
//						 int nrinlrats,
//						 float granularity,
//						 int & maxelementsCostsAll, 
//						 int & maxelementsDistsAll,
//						 int & maxelementsCosts,
//						 int & maxelementsDists
//						 )
//{
//
//	//Get largest values
//	float maxCost = 0, maxDist = 0;
//	vector<float> maxCosts, maxDists;
//	for(int k = 0; k < nrinlrats; k++)
//	{
//		maxCosts.push_back(*max_element(costRatios[k].begin(), costRatios[k].end()));
//		maxDists.push_back(*max_element(distRatios[k].begin(), distRatios[k].end()));
//		if(maxCosts.back() > maxCost)
//			maxCost = maxCosts.back();
//		if(maxDists.back() > maxDist)
//			maxDist = maxDists.back();
//	}
//
//	//get max number of bins with a given granularity
//	maxelementsCosts = (int)std::ceilf((maxCost + 0.0001) / granularity);//+ 0.0001 is necessary to account for values directly at the highest border value
//	maxelementsDists = (int)std::ceilf((maxDist + 0.0001) / granularity);
//
//	//Get the mid values for each bin and the upper and lower border values for each bin
//	cumulDistXvalues.push_back(granularity / 2.0f);
//	cumulDistXborders.push_back(0);
//	cumulDistXborders.push_back(granularity);
//	for(int i = 1; i < (maxelementsCosts > maxelementsDists ? maxelementsCosts:maxelementsDists); i++)
//	{
//		cumulDistXvalues.push_back(cumulDistXvalues.back() + granularity);
//		cumulDistXborders.push_back(cumulDistXborders.back() + granularity);
//	}
//
//	//generate a (not normalized) cumulative distribution for every inlier ratio seperately
//	for(int k = 0; k < nrinlrats; k++)
//	{
//		for(int i = 0; i < maxelementsCosts; i++)
//		{
//			costCumulDistrTP[k].push_back(0);
//			costCumulDistrFP[k].push_back(0);
//			for(unsigned int j = 0; j < costRatios[k].size(); j++)
//			{
//				if((costRatios[k][j] >= cumulDistXborders[i]) && (costRatios[k][j] < cumulDistXborders[i+1]))
//				{
//					if(tpfps[k][j])
//						costCumulDistrTP[k][i]++;
//					else
//						costCumulDistrFP[k][i]++;
//				}
//			}
//		}
//		for(int i = 0; i < maxelementsDists; i++)
//		{
//			distCumulDistrTP[k].push_back(0);
//			distCumulDistrFP[k].push_back(0);
//			for(unsigned int j = 0; j < distRatios[k].size(); j++)
//			{
//				if((distRatios[k][j] >= cumulDistXborders[i]) && (distRatios[k][j] < cumulDistXborders[i+1]))
//				{
//					if(tpfps[k][j])
//						distCumulDistrTP[k][i]++;
//					else
//						distCumulDistrFP[k][i]++;
//				}
//			}
//		}
//	}
//
//	//Generate a cumulative distribution over all inlier ratios together
//	for(int i = 0; i < maxelementsCosts; i++)
//	{
//		costCumulDallInlTP.push_back(costCumulDistrTP[0][i]);
//		costCumulDallInlFP.push_back(costCumulDistrFP[0][i]);
//		for(int k = 1; k < nrinlrats; k++)
//		{
//			costCumulDallInlTP.back() += costCumulDistrTP[k][i];
//			costCumulDallInlFP.back() += costCumulDistrFP[k][i];
//		}
//	}
//	for(int i = 0; i < maxelementsDists; i++)
//	{
//		distCumulDallInlTP.push_back(distCumulDistrTP[0][i]);
//		distCumulDallInlFP.push_back(distCumulDistrFP[0][i]);
//		for(int k = 1; k < nrinlrats; k++)
//		{
//			distCumulDallInlTP.back() += distCumulDistrTP[k][i];
//			distCumulDallInlFP.back() += distCumulDistrFP[k][i];
//		}
//	}
//
//	//Generate the cumulative distribution of the different combined (cost & distance) ratios
//	{
//		//Combine distance and cost ratios
//		vector<vector<float>> distcostadd(nrinlrats);
//		vector<vector<pair<bool,float>>> distcostadd2tresh(nrinlrats), distcostadd1tresh(nrinlrats);
//		for(int k = 0; k < nrinlrats; k++)
//		{
//			for(size_t i = 0; i < costRatios[k].size(); i++)
//			{
//				distcostadd[k].push_back((costRatios[k][i] + distRatios[k][i]) / 2.0f);
//				if((costRatios[k][i] > 1.5f) && (distRatios[k][i] > 2.0f))
//				{
//					distcostadd2tresh[k].push_back(make_pair<bool,float>(tpfps[k][i], (costRatios[k][i] + distRatios[k][i]) / 2.0f));
//				}
//				if(costRatios[k][i] > 1.0f)
//				{
//					distcostadd1tresh[k].push_back(make_pair<bool,float>(tpfps[k][i], (costRatios[k][i] + distRatios[k][i]) / 2.0f));
//				}
//			}
//		}
//		//generate a (not normalized) cumulative distribution for every inlier ratio seperately
//		std::vector<vector<double>> addedCDcumulDistTP_tmp(nrinlrats), addedCDcumulDistFP_tmp(nrinlrats), add2threshCDcumulDistTP_tmp(nrinlrats), add2threshCDcumulDistFP_tmp(nrinlrats);
//		std::vector<vector<double>> add1threshCDcumulDistTP_tmp(nrinlrats), add1threshCDcumulDistFP_tmp(nrinlrats);
//		for(int k = 0; k < nrinlrats; k++)
//		{
//			for(unsigned int i = 0; i < cumulDistXvalues.size(); i++)
//			{
//				addedCDcumulDistTP_tmp[k].push_back(0);
//				addedCDcumulDistFP_tmp[k].push_back(0);
//				for(unsigned int j = 0; j < distcostadd[k].size(); j++)
//				{
//					if((distcostadd[k][j] >= cumulDistXborders[i]) && (distcostadd[k][j] < cumulDistXborders[i+1]))
//					{
//						if(tpfps[k][j])
//							addedCDcumulDistTP_tmp[k][i]++;
//						else
//							addedCDcumulDistFP_tmp[k][i]++;
//					}
//				}
//				add2threshCDcumulDistTP_tmp[k].push_back(0);
//				add2threshCDcumulDistFP_tmp[k].push_back(0);
//				for(unsigned int j = 0; j < distcostadd2tresh[k].size(); j++)
//				{
//					if((distcostadd2tresh[k][j].second >= cumulDistXborders[i]) && (distcostadd2tresh[k][j].second < cumulDistXborders[i+1]))
//					{
//						if(distcostadd2tresh[k][j].first)
//							add2threshCDcumulDistTP_tmp[k][i]++;
//						else
//							add2threshCDcumulDistFP_tmp[k][i]++;
//					}
//				}
//				add1threshCDcumulDistTP_tmp[k].push_back(0);
//				add1threshCDcumulDistFP_tmp[k].push_back(0);
//				for(unsigned int j = 0; j < distcostadd1tresh[k].size(); j++)
//				{
//					if((distcostadd1tresh[k][j].second >= cumulDistXborders[i]) && (distcostadd1tresh[k][j].second < cumulDistXborders[i+1]))
//					{
//						if(distcostadd1tresh[k][j].first)
//							add1threshCDcumulDistTP_tmp[k][i]++;
//						else
//							add1threshCDcumulDistFP_tmp[k][i]++;
//					}
//				}
//			}
//		}
//		//Generate a cumulative distribution over all inlier ratios together
//		for(int i = 0; i < cumulDistXvalues.size(); i++)
//		{
//			addedCDcumulDistTP.push_back(addedCDcumulDistTP_tmp[0][i]);
//			addedCDcumulDistFP.push_back(addedCDcumulDistFP_tmp[0][i]);
//			add2threshCDcumulDistTP.push_back(add2threshCDcumulDistTP_tmp[0][i]);
//			add2threshCDcumulDistFP.push_back(add2threshCDcumulDistFP_tmp[0][i]);
//			add1threshCDcumulDistTP.push_back(add1threshCDcumulDistTP_tmp[0][i]);
//			add1threshCDcumulDistFP.push_back(add1threshCDcumulDistFP_tmp[0][i]);
//			for(int k = 1; k < nrinlrats; k++)
//			{
//				addedCDcumulDistTP.back() += addedCDcumulDistTP_tmp[k][i];
//				addedCDcumulDistFP.back() += addedCDcumulDistFP_tmp[k][i];
//				add2threshCDcumulDistTP.back() += add2threshCDcumulDistTP_tmp[k][i];
//				add2threshCDcumulDistFP.back() += add2threshCDcumulDistFP_tmp[k][i];
//				add1threshCDcumulDistTP.back() += add1threshCDcumulDistTP_tmp[k][i];
//				add1threshCDcumulDistFP.back() += add1threshCDcumulDistFP_tmp[k][i];
//			}
//		}
//	}
//
//	//Normalize the distributions
//	normalizeCumulDistBins(costCumulDistrTP, std::vector<double>(), nrinlrats, &costOutBinHeightsTP);
//	normalizeCumulDistBins(distCumulDistrTP, std::vector<double>(), nrinlrats, &distOutBinHeightsTP);
//	normalizeCumulDistBins(costCumulDistrFP, std::vector<double>(), nrinlrats, &costOutBinHeightsFP);
//	normalizeCumulDistBins(distCumulDistrFP, std::vector<double>(), nrinlrats, &distOutBinHeightsFP);
//	normalizeCumulDistBins(std::vector<std::vector<double>>(), costCumulDallInlTP);
//	normalizeCumulDistBins(std::vector<std::vector<double>>(), distCumulDallInlTP);
//	normalizeCumulDistBins(std::vector<std::vector<double>>(), costCumulDallInlFP);
//	normalizeCumulDistBins(std::vector<std::vector<double>>(), distCumulDallInlFP);
//
//	double binheight;
//	normalizeCumulDistBins(std::vector<std::vector<double>>(), addedCDcumulDistTP);
//	normalizeCumulDistBins(std::vector<std::vector<double>>(), addedCDcumulDistFP);
//	normalizeCumulDistBins(std::vector<std::vector<double>>(), add2threshCDcumulDistTP, 0, NULL, &binheight);
//	normalizeCumulDistBins(std::vector<std::vector<double>>(), add2threshCDcumulDistFP, 0, NULL, NULL, &binheight);
//	normalizeCumulDistBins(std::vector<std::vector<double>>(), add1threshCDcumulDistTP, 0, NULL, &binheight);
//	normalizeCumulDistBins(std::vector<std::vector<double>>(), add1threshCDcumulDistFP, 0, NULL, NULL, &binheight);
//
//	
//	//Cut away far outliers to get a smallest possible diagram
//	const double minBinHeight = 0.01;
//	vector<int> maxelementsCostsTP_tmp(nrinlrats, maxelementsCosts);
//	vector<int> maxelementsDistsTP_tmp(nrinlrats, maxelementsDists);
//	vector<int> maxelementsCostsFP_tmp(nrinlrats, maxelementsCosts);
//	vector<int> maxelementsDistsFP_tmp(nrinlrats, maxelementsDists);
//	/*int maxelementsCostsAllTP = maxelementsCosts;
//	int	maxelementsDistsAllTP = maxelementsDists;
//	int maxelementsCostsAllFP = maxelementsCosts;
//	int	maxelementsDistsAllFP = maxelementsDists;*/
//	int maxelementsCostsTP, maxelementsDistsTP, maxelementsCostsFP, maxelementsDistsFP;
//	for(int k = 0; k < nrinlrats; k++)
//	{
//		while(costCumulDistrTP[k][maxelementsCostsTP_tmp[k]-1] < minBinHeight)
//		{
//			maxelementsCostsTP_tmp[k]--;
//		}
//		if((float)maxelementsCostsTP_tmp[k] * granularity > 12.0f)
//			maxelementsCostsTP_tmp[k] = (int)std::floor(12.0f / granularity + 0.5f);
//		while(distCumulDistrTP[k][maxelementsDistsTP_tmp[k]-1] < minBinHeight)
//		{
//			maxelementsDistsTP_tmp[k]--;
//		}
//		if((float)maxelementsDistsTP_tmp[k] * granularity > 12.0f)
//			maxelementsDistsTP_tmp[k] = (int)std::floor(12.0f / granularity + 0.5f);
//		while(costCumulDistrFP[k][maxelementsCostsFP_tmp[k]-1] < minBinHeight)
//		{
//			maxelementsCostsFP_tmp[k]--;
//		}
//		if((float)maxelementsCostsFP_tmp[k] * granularity > 12.0f)
//			maxelementsCostsFP_tmp[k] = (int)std::floor(12.0f / granularity + 0.5f);
//		while(distCumulDistrFP[k][maxelementsDistsFP_tmp[k]-1] < minBinHeight)
//		{
//			maxelementsDistsFP_tmp[k]--;
//		}
//		if((float)maxelementsDistsFP_tmp[k] * granularity > 12.0f)
//			maxelementsDistsFP_tmp[k] = (int)std::floor(12.0f / granularity + 0.5f);
//	}
//	maxelementsCostsTP = *max_element(maxelementsCostsTP_tmp.begin(), maxelementsCostsTP_tmp.end());
//	maxelementsDistsTP = *max_element(maxelementsDistsTP_tmp.begin(), maxelementsDistsTP_tmp.end());
//	maxelementsCostsFP = *max_element(maxelementsCostsFP_tmp.begin(), maxelementsCostsFP_tmp.end());
//	maxelementsDistsFP = *max_element(maxelementsDistsFP_tmp.begin(), maxelementsDistsFP_tmp.end());
//	maxelementsCosts = std::max(maxelementsCostsTP, maxelementsCostsFP);
//	maxelementsDists = std::max(maxelementsDistsTP, maxelementsDistsFP);
//	for(int k = 0; k < nrinlrats; k++)
//	{
//		costCumulDistrTP[k].erase(costCumulDistrTP[k].begin() + maxelementsCosts, costCumulDistrTP[k].end());
//		distCumulDistrTP[k].erase(distCumulDistrTP[k].begin() + maxelementsDists, distCumulDistrTP[k].end());
//		costCumulDistrFP[k].erase(costCumulDistrFP[k].begin() + maxelementsCosts, costCumulDistrFP[k].end());
//		distCumulDistrFP[k].erase(distCumulDistrFP[k].begin() + maxelementsDists, distCumulDistrFP[k].end());
//	}
//
//	cutFarElemts(costCumulDallInlTP, costCumulDallInlFP, granularity, minBinHeight, &maxelementsCostsAll);
//	cutFarElemts(distCumulDallInlTP, distCumulDallInlFP, granularity, minBinHeight, &maxelementsDistsAll);
//
//	/*while(costCumulDallInlTP[maxelementsCostsAllTP-1] < minBinHeight)
//	{
//		maxelementsCostsAllTP--;
//	}
//	if((float)maxelementsCostsAllTP * granularity > 12.0f)
//			maxelementsCostsAllTP = (int)std::floor(12.0f / granularity + 0.5f);
//	while(distCumulDallInlTP[maxelementsDistsAllTP-1] < minBinHeight)
//	{
//		maxelementsDistsAllTP--;
//	}
//	while(costCumulDallInlFP[maxelementsCostsAllFP-1] < minBinHeight)
//	{
//		maxelementsCostsAllFP--;
//	}
//	while(distCumulDallInlFP[maxelementsDistsAllFP-1] < minBinHeight)
//	{
//		maxelementsDistsAllFP--;
//	}
//	maxelementsCostsAll = std::max(maxelementsCostsAllTP, maxelementsCostsAllFP);
//	maxelementsDistsAll = std::max(maxelementsDistsAllTP, maxelementsDistsAllFP);
//	costCumulDallInlTP.erase(costCumulDallInlTP.begin() + maxelementsCostsAll, costCumulDallInlTP.end());
//	distCumulDallInlTP.erase(distCumulDallInlTP.begin() + maxelementsDistsAll, distCumulDallInlTP.end());
//	costCumulDallInlFP.erase(costCumulDallInlFP.begin() + maxelementsCostsAll, costCumulDallInlFP.end());
//	distCumulDallInlFP.erase(distCumulDallInlFP.begin() + maxelementsDistsAll, distCumulDallInlFP.end());*/
//
//	cutFarElemts(addedCDcumulDistTP, addedCDcumulDistFP, granularity, minBinHeight);
//	cutFarElemts(add2threshCDcumulDistTP, add2threshCDcumulDistFP, granularity, minBinHeight);
//	cutFarElemts(add1threshCDcumulDistTP, add1threshCDcumulDistFP, granularity, minBinHeight);
//}
//
//void cutFarElemts(std::vector<double> & vec1, std::vector<double> & vec2, float granularity, double minBinHeight, int *maxElements)
//{
//	int maxelements1 = (int)vec1.size();
//	int maxelements2 = (int)vec2.size();
//	while((vec1[maxelements1-1] < minBinHeight) && (maxelements1 > 0))
//	{
//		maxelements1--;
//	}
//	if((float)maxelements1 * granularity > 12.0f)
//			maxelements1 = (int)std::floor(12.0f / granularity + 0.5f);
//	while((vec2[maxelements2-1] < minBinHeight) && (maxelements2 > 0))
//	{
//		maxelements2--;
//	}
//	if((float)maxelements2 * granularity > 12.0f)
//			maxelements2 = (int)std::floor(12.0f / granularity + 0.5f);
//	maxelements1 = std::max(maxelements1, maxelements2);
//	if(maxElements)
//		*maxElements = maxelements1;
//
//	vec1.erase(vec1.begin() + maxelements1, vec1.end());
//	vec2.erase(vec2.begin() + maxelements1, vec2.end());
//}
//
///* Starts the test for estimating the runtime for descriptor extraction. Here, no matching is performed.
//*
//* string imgsPath				Input  -> Path which includes both left and right images
//* string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
//* int flowDispH				Input  -> Indicates which type of ground truth data is used:
//*										  0: flow files from KITTI database
//*										  1: disparity files from KITTI database
//*										  2: homography files (Please note that a homography always relates
//*											 to the first image (e.g. 1->2, 1->3, ...))
//* string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images
//*									      (after prefix only comes the image number)
//* string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
//*									      (after prefix only comes the image number). For testing with homographies,
//*										  this string can be empty.
//* string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
//*									      (after prefix only comes the image number)
//* string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
//*										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
//*										  detectors are possible.
//* string descriptorExtractor	Input  -> The used descriptor extractor. Possible inputs should only be FREAK
//*										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
//* string storeResultPath		Input  -> Path were the resulting measurements should be stored
//*
//*
//* Return value:				 0:		  Everything ok
//*								-1:		  Failed
//*/
//int startDescriptorTimeMeasurement(std::string imgsPath, std::string flowDispHPath, int flowDispH,
//	std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//	std::string featureDetector, std::string descriptorExtractor, std::string storeResultPath)
//{
//	int err;
//	cv::Mat src[2];
//	std::auto_ptr<baseMatcher> mymatcher;
//	string outpath, outfilename;
//	const int nrTestImgs = 10;
//	vector<double> descriptortimes;
//	qualityParm1 timestats;
//	
//
//	if (flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if (err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if (err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//
//		for (int k = 0; k < ((int)filenamesl.size() < nrTestImgs ? (int)filenamesl.size() : nrTestImgs); k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[k], CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[k], CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageFlowFile(flowDispHPath, filenamesflow[k], &flowimg);
//			if (err)
//			{
//				cout << "Could not open flow file with index " << k << endl;
//				continue;
//			}
//			mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//			if (mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//			mymatcher->measureTd = true;
//			err = mymatcher->performMatching(1.0);
//			if (mymatcher->tmeanD > 1e-6 && mymatcher->tmeanD < 3000)
//				descriptortimes.push_back(std::round(mymatcher->tmeanD * 10000)/10);//*1000 to get us
//			else
//			{
//				cout << "Descriptor extraction time not valid!" << endl;
//				return -1;
//			}
//		}
//	}
//	else if (flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if (err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if (err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		for (int k = 0; k < ((int)filenamesl.size() < nrTestImgs ? (int)filenamesl.size() : nrTestImgs); k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[k], CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[k], CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageDisparityFile(flowDispHPath, filenamesflow[k], &flowimg);
//			if (err)
//			{
//				cout << "Could not open disparity file with index " << k << ". Exiting." << endl;
//				exit(0);
//			}
//
//			mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//			mymatcher->measureTd = true;
//			if (mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			err = mymatcher->performMatching(1.0);
//			if (mymatcher->tmeanD > 1e-6 && mymatcher->tmeanD < 3000)
//				descriptortimes.push_back(std::round(mymatcher->tmeanD * 10000) / 10);//*1000 to get us
//			else
//			{
//				cout << "Descriptor extraction time not valid!" << endl;
//				return -1;
//			}
//		}
//	}
//	else if (flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if (err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if (err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for (int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if (err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		//Take the stored homographys and perform evaluation
//		src[0] = cv::imread(imgsPath + "\\" + filenamesl[0], CV_LOAD_IMAGE_GRAYSCALE);
//		for (int idx1 = 0; idx1 < ((int)fnames.size() < nrTestImgs ? (int)fnames.size() : nrTestImgs); idx1++)
//		{
//			cv::Mat H = Hs[idx1];
//			src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1], CV_LOAD_IMAGE_GRAYSCALE);
//			mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//
//			if (mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//			mymatcher->measureTd = true;
//			err = mymatcher->performMatching(1.0);
//			if (mymatcher->tmeanD > 1e-6 && mymatcher->tmeanD < 3000)
//				descriptortimes.push_back(std::round(mymatcher->tmeanD * 10000) / 10);//*1000 to get us
//			else
//			{
//				cout << "Descriptor extraction time not valid!" << endl;
//				return -1;
//			}
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//	
//
//	getStatisticfromVec2(descriptortimes, &timestats, false);
//
//	if (dirExists(storeResultPath)) //Check if output directory existis
//	{
//		outpath = storeResultPath;
//	}
//	else
//	{
//		outpath = imgsPath + "\\evalResult";
//		if (!dirExists(outpath))
//			_mkdir(outpath.c_str());
//	}
//	outfilename = "tex_descriptor_runtime_" + featureDetector + "_" + descriptorExtractor + ".dat";
//
//	std::ofstream evalsToFile(outpath + "\\" + outfilename);
//	evalsToFile << "# Statistics over the mean runtime of a descriptor" << endl;
//	evalsToFile << "mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//	evalsToFile << std::setprecision(1) << std::fixed << timestats.arithErr << " "
//		<< timestats.medErr << " "
//		<< *std::min_element(descriptortimes.begin(), descriptortimes.end()) << " "
//		<< *std::max_element(descriptortimes.begin(), descriptortimes.end()) << " "
//		<< timestats.arithStd << " " << timestats.medStd << " "
//		<< timestats.lowerQuart << " " << timestats.upperQuart << endl;
//	evalsToFile.close();
//
//	return 0;
//	
//}
//
///* Starts the test for different inlier ratios on all image pairs of a szene for different matching algorithms and szenes.
// * The output is generated for precision, recall, fall-out and accuracy and is the mean value over all images.
// *
// * string imgsPath				Input  -> Path which includes both left and right images
// * string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
// * int flowDispH				Input  -> Indicates which type of ground truth data is used:
// *										  0: flow files from KITTI database
// *										  1: disparity files from KITTI database
// *										  2: homography files (Please note that a homography always relates
// *											 to the first image (e.g. 1->2, 1->3, ...))
// * string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images 
// *									      (after prefix only comes the image number)
// * string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
// *									      (after prefix only comes the image number). For testing with homographies,
// *										  this string can be empty.
// * string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
// *									      (after prefix only comes the image number)
// * string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
// *										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
// *										  detectors are possible.
// * string descriptorExtractor	Input  -> The used descriptor extractor. Possible inputs should only be FREAK
// *										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
// * string matcherType			Input  -> The matcher type under test. Possible inputs are:
// *											CASCHASH: Cascade Hashing matcher
// *											GEOMAWARE: Geometry-aware Feature matching algorithm
// *											GMBSOF: Guided matching based on statistical optical flow
// *											HIRCLUIDX: Hirarchical Clustering Index Matching from the FLANN library
// *											HIRKMEANS: hierarchical k-means tree matcher from the FLANN library
// *											VFCKNN: Vector field consensus (VFC) algorithm with k nearest neighbor 
// *													matches provided from the Hirarchical Clustering Index Matching 
// *													algorithm from the FLANN library
// *											LIBVISO: matcher from the libviso2 library
// *											LINEAR: linear Matching algorithm (Brute force) from the FLANN library
// *											LSHIDX: LSH Index Matching algorithm from the FLANN library
// *											RANDKDTREE: randomized KD-trees matcher from the FLANN library
// * bool useRatioTest			Input  -> Specifies if a ratio test should be performed on the results of a matching
// *										  algorithm. The ratio test is only possible for the following algorithms:
// *										  HIRCLUIDX, HIRKMEANS, LINEAR, LSHIDX, RANDKDTREE
// * string storeResultPath		Input  -> Path were the resulting measurements should be stored
// * bool refine					Input  -> If true [DEFAULT = false], the results from the matching algorithm are
// *										  refined using VFC.
// * int showResult				Input  -> If >= 0, the result from the matching algorithm is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * int showRefinedResult		Input  -> If >= 0, the result after refinement with VFC is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * string storeImgResPath		Input  -> Optional path for storing the resulting matches drawn into the images,
// *										  where the options of which results should be drawn are specified in
// *										  "showResult". If this path is set, the images are NOT displayed.
// * string storeRefResPath		Input  -> Optional path for storing the resulting matches after refinement 
// *										  drawn into the images, where the options of which results should be 
// *										  drawn are specified in "showRefinedResult". If this path is set, the images 
// *										  are NOT displayed.
// * 
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Failed
// */
//int startInlierRatioMeasurementWholeSzene(std::string imgsPath, std::string flowDispHPath, int flowDispH, 
//										std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//										std::string featureDetector, std::string descriptorExtractor, std::string matcherType,
//										bool useRatioTest, std::string storeResultPath, bool refine, 
//										int showResult, int showRefinedResult, std::string storeImgResPath, 
//										std::string storeRefResPath, std::string idxPars_NMSLIB, std::string queryPars_NMSLIB)
//{
//	int err;
//	cv::Mat src[2];
//	std::auto_ptr<baseMatcher> mymatcher;
//	string outpath, outfilename;
//	vector<double> inlierRatios;
//	vector<double> inlierRatios_real;
//	int nr_inlratios;
//
//	if(!storeImgResPath.empty() && (showResult == -1))
//	{
//		cout << "If you want to store the resulting images you must specify the showResult flag! Exiting." << endl;
//		exit(0);
//	}
//	if(!storeRefResPath.empty() && (showRefinedResult == -1))
//	{
//		cout << "If you want to store the resulting images with refined matches you must specify the showRefinedResult flag! Exiting." << endl;
//		exit(0);
//	}
//	if(!refine && (showRefinedResult >= 0))
//	{
//		cout << "Cant show refined results if refinement is disabled" << endl;
//	}
//
//	//Generate inlier ratios
//	double startInlRatio = 1.0;
//	inlierRatios.push_back(startInlRatio);
//	while(startInlRatio > 0.2)
//	{
//		startInlRatio -= 0.05;
//		inlierRatios.push_back(startInlRatio);
//	}
//	while(startInlRatio > 0.1)
//	{
//		startInlRatio -= 0.02;
//		inlierRatios.push_back(startInlRatio);
//	}
//	while(startInlRatio > 0.01)
//	{
//		startInlRatio -= 0.01;
//		inlierRatios.push_back(startInlRatio);
//	}
//	/*while(startInlRatio > 0.005)
//	{
//		startInlRatio -= 0.005;
//		inlierRatios.push_back(startInlRatio);
//	}*/
//	nr_inlratios = (int)inlierRatios.size();
//	inlierRatios_real = inlierRatios;
//
//	vector<vector<double>> inlRatPrecision(nr_inlratios), inlRatRecall(nr_inlratios), inlRatFallOut(nr_inlratios), inlRatAccuracy(nr_inlratios);
//	vector<vector<double>> inlRatPrecisionRef(nr_inlratios), inlRatRecallRef(nr_inlratios), inlRatFallOutRef(nr_inlratios), inlRatAccuracyRef(nr_inlratios);
//	vector<qualityParm1> inlRatPrecisionStat(nr_inlratios), inlRatRecallStat(nr_inlratios), inlRatFallOutStat(nr_inlratios), inlRatAccuracyStat(nr_inlratios);
//	vector<qualityParm1> inlRatPrecisionStatRef(nr_inlratios), inlRatRecallStatRef(nr_inlratios), inlRatFallOutStatRef(nr_inlratios), inlRatAccuracyStatRef(nr_inlratios);
//	vector<vector<double>> meanInlRatios(nr_inlratios);
//
//	if(flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//		
//		for(int k = 0; k < (int)filenamesl.size(); k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[k],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[k],CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageFlowFile(flowDispHPath, filenamesflow[k], &flowimg);
//			if(err)
//			{
//				cout << "Could not open flow file with index " << k << endl;
//				continue;
//			}
//			if(!matcherType.compare("CASCHASH"))
//			{
//				//mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//				mymatcher.reset(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//			}
//			else if(!matcherType.compare("GMBSOF"))
//			{
//				mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//			}
//			else if(!matcherType.compare("HIRCLUIDX"))
//			{
//				mymatcher.reset(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("HIRKMEANS"))
//			{
//				mymatcher.reset(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("VFCKNN"))
//			{
//				mymatcher.reset(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], 2, useRatioTest));
//			}
//			else if(!matcherType.compare("LINEAR"))
//			{
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("LSHIDX"))
//			{
//				mymatcher.reset(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("RANDKDTREE"))
//			{
//				mymatcher.reset(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if (!matcherType.compare("BRUTEFORCENMS"))
//			{
//				mymatcher.reset(new bruteforceNMS_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if (!matcherType.compare("SWGRAPH"))
//			{
//				mymatcher = std::auto_ptr<swgraph_matcher>(new swgraph_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("HNSW"))
//			{
//				mymatcher = std::auto_ptr<hnsw_matcher>(new hnsw_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("VPTREE"))
//			{
//				mymatcher = std::auto_ptr<vptree_matcher>(new vptree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("ANNOY"))
//			{
//				mymatcher = std::auto_ptr<annoy_matcher>(new annoy_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else
//			{
//				cout << "No valid matcher specified! Exiting." << endl;
//				exit(1);
//			}
//			if(mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			for(int i = 0; i < nr_inlratios; i++)
//			{
//				cout << "Testing image " << k << "of " << filenamesl.size() << " with inlier ratio " << inlierRatios[i] << endl;
//				err = mymatcher->performMatching(inlierRatios[i]);
//				if(err)
//				{
//					if(err == -2)
//					{
//						meanInlRatios[i].push_back(mymatcher->inlRatioL);
//						//inlierRatios_real[i] = mymatcher->inlRatioL;
//						inlRatPrecision[i].push_back(0);
//						inlRatRecall[i].push_back(0);
//						inlRatFallOut[i].push_back(1.0);
//						inlRatAccuracy[i].push_back(0);
//						continue;
//					}
//					else
//					{
//						continue;
//					}
//				}
//				meanInlRatios[i].push_back(mymatcher->inlRatioL);
//				//inlierRatios_real[i] = mymatcher->inlRatioL;
//				inlRatPrecision[i].push_back(mymatcher->qpm.ppv);
//				inlRatRecall[i].push_back(mymatcher->qpm.tpr);
//				inlRatFallOut[i].push_back(mymatcher->qpm.fpr);
//				inlRatAccuracy[i].push_back(mymatcher->qpm.acc);
//				
//				if(refine)
//				{
//					err = mymatcher->refineMatches();
//					if(err)
//					{
//						inlRatPrecisionRef[i].push_back(0);
//						inlRatRecallRef[i].push_back(0);
//						inlRatFallOutRef[i].push_back(1.0);
//						inlRatAccuracyRef[i].push_back(0);
//					}
//					else
//					{
//						inlRatPrecisionRef[i].push_back(mymatcher->qpr.ppv);
//						inlRatRecallRef[i].push_back(mymatcher->qpr.tpr);
//						inlRatFallOutRef[i].push_back(mymatcher->qpr.fpr);
//						inlRatAccuracyRef[i].push_back(mymatcher->qpr.acc);
//					}
//				}
//
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_inlRatio_flow_idx" + std::to_string((ULONGLONG)k) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//				if((showRefinedResult >= 0) && refine && !err)
//				{
//					if(storeRefResPath.empty())
//					{
//						mymatcher->showMatches(showRefinedResult);
//					}
//					else
//					{
//						if(dirExists(storeRefResPath)) //Check if output directory existis
//						{
//							outpath = storeRefResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "imgRef_inlRatio_flow_idx" + std::to_string((ULONGLONG)k) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//					}
//				}
//			}
//		}
//		if(inlRatPrecision[0].empty())
//		{
//			cout << "Ground truth generation or algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		for(int i = 0; i < nr_inlratios; i++)
//		{
//			if(!meanInlRatios[i].empty())
//			{
//				double mean_val = 0;
//				for(size_t j = 0; j < meanInlRatios[i].size(); j++)
//				{
//					mean_val += meanInlRatios[i][j];
//				}
//				inlierRatios_real[i] = mean_val / (double)meanInlRatios[i].size();
//			}
//
//			getStatisticfromVec2(inlRatPrecision[i], &inlRatPrecisionStat[i], false);
//			getStatisticfromVec2(inlRatRecall[i], &inlRatRecallStat[i], false);
//			getStatisticfromVec2(inlRatFallOut[i], &inlRatFallOutStat[i], false);
//			getStatisticfromVec2(inlRatAccuracy[i], &inlRatAccuracyStat[i], false);
//		}
//
//		if(refine)
//		{
//			if(inlRatPrecisionRef[0].empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			for(int i = 0; i < nr_inlratios; i++)
//			{
//				getStatisticfromVec2(inlRatPrecisionRef[i], &inlRatPrecisionStatRef[i], false);
//				getStatisticfromVec2(inlRatRecallRef[i], &inlRatRecallStatRef[i], false);
//				getStatisticfromVec2(inlRatFallOutRef[i], &inlRatFallOutStatRef[i], false);
//				getStatisticfromVec2(inlRatAccuracyRef[i], &inlRatAccuracyStatRef[i], false);
//			}
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_inlRatio_flow_inliersGT_precision_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatPrecision[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatPrecisionStat[i].arithErr << " "
//								<< inlRatPrecisionStat[i].medErr << " "
//								<< *std::min_element(inlRatPrecision[i].begin(), inlRatPrecision[i].end()) << " "
//								<< *std::max_element(inlRatPrecision[i].begin(), inlRatPrecision[i].end()) << " "
//								<< inlRatPrecisionStat[i].arithStd << " " << inlRatPrecisionStat[i].medStd << " "
//								<< inlRatPrecisionStat[i].lowerQuart << " " << inlRatPrecisionStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_flow_inliersGT_recall_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatRecall[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatRecallStat[i].arithErr << " "
//								<< inlRatRecallStat[i].medErr << " "
//								<< *std::min_element(inlRatRecall[i].begin(), inlRatRecall[i].end()) << " "
//								<< *std::max_element(inlRatRecall[i].begin(), inlRatRecall[i].end()) << " "
//								<< inlRatRecallStat[i].arithStd << " " << inlRatRecallStat[i].medStd << " "
//								<< inlRatRecallStat[i].lowerQuart << " " << inlRatRecallStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_flow_inliersGT_fpr_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatFallOut[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatFallOutStat[i].arithErr << " "
//								<< inlRatFallOutStat[i].medErr << " "
//								<< *std::min_element(inlRatFallOut[i].begin(), inlRatFallOut[i].end()) << " "
//								<< *std::max_element(inlRatFallOut[i].begin(), inlRatFallOut[i].end()) << " "
//								<< inlRatFallOutStat[i].arithStd << " " << inlRatFallOutStat[i].medStd << " "
//								<< inlRatFallOutStat[i].lowerQuart << " " << inlRatFallOutStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_flow_inliersGT_acc_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatAccuracy[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatAccuracyStat[i].arithErr << " "
//								<< inlRatAccuracyStat[i].medErr << " "
//								<< *std::min_element(inlRatAccuracy[i].begin(), inlRatAccuracy[i].end()) << " "
//								<< *std::max_element(inlRatAccuracy[i].begin(), inlRatAccuracy[i].end()) << " "
//								<< inlRatAccuracyStat[i].arithStd << " " << inlRatAccuracyStat[i].medStd << " "
//								<< inlRatAccuracyStat[i].lowerQuart << " " << inlRatAccuracyStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		if(refine)
//		{
//			{
//				outfilename = "tex_inlRatio_flow_inliersGT_precisionRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatPrecisionRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatPrecisionStatRef[i].arithErr << " "
//								<< inlRatPrecisionStatRef[i].medErr << " "
//								<< *std::min_element(inlRatPrecisionRef[i].begin(), inlRatPrecisionRef[i].end()) << " "
//								<< *std::max_element(inlRatPrecisionRef[i].begin(), inlRatPrecisionRef[i].end()) << " "
//								<< inlRatPrecisionStatRef[i].arithStd << " " << inlRatPrecisionStatRef[i].medStd << " "
//								<< inlRatPrecisionStatRef[i].lowerQuart << " " << inlRatPrecisionStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_flow_inliersGT_recallRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatRecallRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatRecallStatRef[i].arithErr << " "
//								<< inlRatRecallStatRef[i].medErr << " "
//								<< *std::min_element(inlRatRecallRef[i].begin(), inlRatRecallRef[i].end()) << " "
//								<< *std::max_element(inlRatRecallRef[i].begin(), inlRatRecallRef[i].end()) << " "
//								<< inlRatRecallStatRef[i].arithStd << " " << inlRatRecallStatRef[i].medStd << " "
//								<< inlRatRecallStatRef[i].lowerQuart << " " << inlRatRecallStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_flow_inliersGT_fprRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatFallOutRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatFallOutStatRef[i].arithErr << " "
//								<< inlRatFallOutStatRef[i].medErr << " "
//								<< *std::min_element(inlRatFallOutRef[i].begin(), inlRatFallOutRef[i].end()) << " "
//								<< *std::max_element(inlRatFallOutRef[i].begin(), inlRatFallOutRef[i].end()) << " "
//								<< inlRatFallOutStatRef[i].arithStd << " " << inlRatFallOutStatRef[i].medStd << " "
//								<< inlRatFallOutStatRef[i].lowerQuart << " " << inlRatFallOutStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_flow_inliersGT_accRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatAccuracyRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatAccuracyStatRef[i].arithErr << " "
//								<< inlRatAccuracyStatRef[i].medErr << " "
//								<< *std::min_element(inlRatAccuracyRef[i].begin(), inlRatAccuracyRef[i].end()) << " "
//								<< *std::max_element(inlRatAccuracyRef[i].begin(), inlRatAccuracyRef[i].end()) << " "
//								<< inlRatAccuracyStatRef[i].arithStd << " " << inlRatAccuracyStatRef[i].medStd << " "
//								<< inlRatAccuracyStatRef[i].lowerQuart << " " << inlRatAccuracyStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else if(flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int k = 0; k < (int)filenamesl.size(); k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[k],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[k],CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageDisparityFile(flowDispHPath, filenamesflow[k], &flowimg);
//			if(err)
//			{
//				cout << "Could not open disparity file with index " << k << ". Exiting." << endl;
//				exit(0);
//			}
//			if(!matcherType.compare("CASCHASH"))
//			{
//				//mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//				mymatcher.reset(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//			}
//			else if(!matcherType.compare("GMBSOF"))
//			{
//				mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//			}
//			else if(!matcherType.compare("HIRCLUIDX"))
//			{
//				mymatcher.reset(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("HIRKMEANS"))
//			{
//				mymatcher.reset(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("VFCKNN"))
//			{
//				mymatcher.reset(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], 2, useRatioTest));
//			}
//			else if(!matcherType.compare("LINEAR"))
//			{
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("LSHIDX"))
//			{
//				mymatcher.reset(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("RANDKDTREE"))
//			{
//				mymatcher.reset(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if (!matcherType.compare("BRUTEFORCENMS"))
//			{
//				mymatcher.reset(new bruteforceNMS_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if (!matcherType.compare("SWGRAPH"))
//			{
//				mymatcher = std::auto_ptr<swgraph_matcher>(new swgraph_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("HNSW"))
//			{
//				mymatcher = std::auto_ptr<hnsw_matcher>(new hnsw_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("VPTREE"))
//			{
//				mymatcher = std::auto_ptr<vptree_matcher>(new vptree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//			}
//			else if (!matcherType.compare("ANNOY"))
//			{
//				mymatcher = std::auto_ptr<annoy_matcher>(new annoy_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else
//			{
//				cout << "No valid matcher specified! Exiting." << endl;
//				exit(1);
//			}
//
//			if(mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			for(int i = 0; i < nr_inlratios; i++)
//			{
//				//if((k == 80) && (i == 3)) continue;//----------------------------------------> must be removed
//				cout << "Testing image " << k << "of " << filenamesl.size() << " with inlier ratio " << inlierRatios[i] << endl;
//				err = mymatcher->performMatching(inlierRatios[i]);
//				if(err)
//				{
//					if(err == -2)
//					{
//						meanInlRatios[i].push_back(mymatcher->inlRatioL);
//						//inlierRatios_real[i] = mymatcher->inlRatioL;
//						inlRatPrecision[i].push_back(0);
//						inlRatRecall[i].push_back(0);
//						inlRatFallOut[i].push_back(1.0);
//						inlRatAccuracy[i].push_back(0);
//						continue;
//					}
//					else
//					{
//						continue;
//					}
//				}
//				meanInlRatios[i].push_back(mymatcher->inlRatioL);
//				//inlierRatios_real[i] = mymatcher->inlRatioL;
//				inlRatPrecision[i].push_back(mymatcher->qpm.ppv);
//				inlRatRecall[i].push_back(mymatcher->qpm.tpr);
//				inlRatFallOut[i].push_back(mymatcher->qpm.fpr);
//				inlRatAccuracy[i].push_back(mymatcher->qpm.acc);
//				
//				if(refine)
//				{
//					err = mymatcher->refineMatches();
//					if(err)
//					{
//						inlRatPrecisionRef[i].push_back(0);
//						inlRatRecallRef[i].push_back(0);
//						inlRatFallOutRef[i].push_back(1.0);
//						inlRatAccuracyRef[i].push_back(0);
//					}
//					else
//					{
//						inlRatPrecisionRef[i].push_back(mymatcher->qpr.ppv);
//						inlRatRecallRef[i].push_back(mymatcher->qpr.tpr);
//						inlRatFallOutRef[i].push_back(mymatcher->qpr.fpr);
//						inlRatAccuracyRef[i].push_back(mymatcher->qpr.acc);
//					}
//				}
//
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_inlRatio_disp_idx" + std::to_string((ULONGLONG)k) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//				if((showRefinedResult >= 0) && refine && !err)
//				{
//					if(storeRefResPath.empty())
//					{
//						mymatcher->showMatches(showRefinedResult);
//					}
//					else
//					{
//						if(dirExists(storeRefResPath)) //Check if output directory existis
//						{
//							outpath = storeRefResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "imgRef_inlRatio_disp_idx" + std::to_string((ULONGLONG)k) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//					}
//				}
//			}
//		}
//
//		if(inlRatPrecision[0].empty())
//		{
//			cout << "Ground truth generation or algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		for(int i = 0; i < nr_inlratios; i++)
//		{
//			if(!meanInlRatios[i].empty())
//			{
//				double mean_val = 0;
//				for(size_t j = 0; j < meanInlRatios[i].size(); j++)
//				{
//					mean_val += meanInlRatios[i][j];
//				}
//				inlierRatios_real[i] = mean_val / (double)meanInlRatios[i].size();
//			}
//
//			getStatisticfromVec2(inlRatPrecision[i], &inlRatPrecisionStat[i], false);
//			getStatisticfromVec2(inlRatRecall[i], &inlRatRecallStat[i], false);
//			getStatisticfromVec2(inlRatFallOut[i], &inlRatFallOutStat[i], false);
//			getStatisticfromVec2(inlRatAccuracy[i], &inlRatAccuracyStat[i], false);
//		}
//
//		if(refine)
//		{
//			if(inlRatPrecisionRef[0].empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			for(int i = 0; i < nr_inlratios; i++)
//			{
//				getStatisticfromVec2(inlRatPrecisionRef[i], &inlRatPrecisionStatRef[i], false);
//				getStatisticfromVec2(inlRatRecallRef[i], &inlRatRecallStatRef[i], false);
//				getStatisticfromVec2(inlRatFallOutRef[i], &inlRatFallOutStatRef[i], false);
//				getStatisticfromVec2(inlRatAccuracyRef[i], &inlRatAccuracyStatRef[i], false);
//			}
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_inlRatio_disp_inliersGT_precision_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatPrecision[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatPrecisionStat[i].arithErr << " "
//								<< inlRatPrecisionStat[i].medErr << " "
//								<< *std::min_element(inlRatPrecision[i].begin(), inlRatPrecision[i].end()) << " "
//								<< *std::max_element(inlRatPrecision[i].begin(), inlRatPrecision[i].end()) << " "
//								<< inlRatPrecisionStat[i].arithStd << " " << inlRatPrecisionStat[i].medStd << " "
//								<< inlRatPrecisionStat[i].lowerQuart << " " << inlRatPrecisionStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_disp_inliersGT_recall_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatRecall[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatRecallStat[i].arithErr << " "
//								<< inlRatRecallStat[i].medErr << " "
//								<< *std::min_element(inlRatRecall[i].begin(), inlRatRecall[i].end()) << " "
//								<< *std::max_element(inlRatRecall[i].begin(), inlRatRecall[i].end()) << " "
//								<< inlRatRecallStat[i].arithStd << " " << inlRatRecallStat[i].medStd << " "
//								<< inlRatRecallStat[i].lowerQuart << " " << inlRatRecallStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_disp_inliersGT_fpr_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatFallOut[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatFallOutStat[i].arithErr << " "
//								<< inlRatFallOutStat[i].medErr << " "
//								<< *std::min_element(inlRatFallOut[i].begin(), inlRatFallOut[i].end()) << " "
//								<< *std::max_element(inlRatFallOut[i].begin(), inlRatFallOut[i].end()) << " "
//								<< inlRatFallOutStat[i].arithStd << " " << inlRatFallOutStat[i].medStd << " "
//								<< inlRatFallOutStat[i].lowerQuart << " " << inlRatFallOutStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_disp_inliersGT_acc_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatAccuracy[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatAccuracyStat[i].arithErr << " "
//								<< inlRatAccuracyStat[i].medErr << " "
//								<< *std::min_element(inlRatAccuracy[i].begin(), inlRatAccuracy[i].end()) << " "
//								<< *std::max_element(inlRatAccuracy[i].begin(), inlRatAccuracy[i].end()) << " "
//								<< inlRatAccuracyStat[i].arithStd << " " << inlRatAccuracyStat[i].medStd << " "
//								<< inlRatAccuracyStat[i].lowerQuart << " " << inlRatAccuracyStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		if(refine)
//		{
//			{
//				outfilename = "tex_inlRatio_disp_inliersGT_precisionRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatPrecisionRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatPrecisionStatRef[i].arithErr << " "
//								<< inlRatPrecisionStatRef[i].medErr << " "
//								<< *std::min_element(inlRatPrecisionRef[i].begin(), inlRatPrecisionRef[i].end()) << " "
//								<< *std::max_element(inlRatPrecisionRef[i].begin(), inlRatPrecisionRef[i].end()) << " "
//								<< inlRatPrecisionStatRef[i].arithStd << " " << inlRatPrecisionStatRef[i].medStd << " "
//								<< inlRatPrecisionStatRef[i].lowerQuart << " " << inlRatPrecisionStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_disp_inliersGT_recallRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatRecallRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatRecallStatRef[i].arithErr << " "
//								<< inlRatRecallStatRef[i].medErr << " "
//								<< *std::min_element(inlRatRecallRef[i].begin(), inlRatRecallRef[i].end()) << " "
//								<< *std::max_element(inlRatRecallRef[i].begin(), inlRatRecallRef[i].end()) << " "
//								<< inlRatRecallStatRef[i].arithStd << " " << inlRatRecallStatRef[i].medStd << " "
//								<< inlRatRecallStatRef[i].lowerQuart << " " << inlRatRecallStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_disp_inliersGT_fprRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatFallOutRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatFallOutStatRef[i].arithErr << " "
//								<< inlRatFallOutStatRef[i].medErr << " "
//								<< *std::min_element(inlRatFallOutRef[i].begin(), inlRatFallOutRef[i].end()) << " "
//								<< *std::max_element(inlRatFallOutRef[i].begin(), inlRatFallOutRef[i].end()) << " "
//								<< inlRatFallOutStatRef[i].arithStd << " " << inlRatFallOutStatRef[i].medStd << " "
//								<< inlRatFallOutStatRef[i].lowerQuart << " " << inlRatFallOutStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_disp_inliersGT_accRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatAccuracyRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatAccuracyStatRef[i].arithErr << " "
//								<< inlRatAccuracyStatRef[i].medErr << " "
//								<< *std::min_element(inlRatAccuracyRef[i].begin(), inlRatAccuracyRef[i].end()) << " "
//								<< *std::max_element(inlRatAccuracyRef[i].begin(), inlRatAccuracyRef[i].end()) << " "
//								<< inlRatAccuracyStatRef[i].arithStd << " " << inlRatAccuracyStatRef[i].medStd << " "
//								<< inlRatAccuracyStatRef[i].lowerQuart << " " << inlRatAccuracyStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else if(flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if(err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if(err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if(err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		if(fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//			for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//				if(!matcherType.compare("CASCHASH"))
//				{
//					mymatcher.reset(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("GMBSOF"))
//				{
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("HIRCLUIDX"))
//				{
//					mymatcher.reset(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("HIRKMEANS"))
//				{
//					mymatcher.reset(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("VFCKNN"))
//				{
//					mymatcher.reset(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 2, useRatioTest));
//				}
//				else if(!matcherType.compare("LINEAR"))
//				{
//					mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("LSHIDX"))
//				{
//					mymatcher.reset(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("RANDKDTREE"))
//				{
//					mymatcher.reset(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if (!matcherType.compare("BRUTEFORCENMS"))
//				{
//					mymatcher.reset(new bruteforceNMS_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if (!matcherType.compare("SWGRAPH"))
//				{
//					mymatcher = std::auto_ptr<swgraph_matcher>(new swgraph_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("HNSW"))
//				{
//					mymatcher = std::auto_ptr<hnsw_matcher>(new hnsw_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("VPTREE"))
//				{
//					mymatcher = std::auto_ptr<vptree_matcher>(new vptree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("ANNOY"))
//				{
//					mymatcher = std::auto_ptr<annoy_matcher>(new annoy_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else
//				{
//					cout << "No valid matcher specified! Exiting." << endl;
//					exit(1);
//				}
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				for(int i = 0; i < nr_inlratios; i++)
//				{
//					cout << "Testing image " << idx1 << " with inlier ratio " << inlierRatios[i] << endl;
//					err = mymatcher->performMatching(inlierRatios[i]);
//					if(err)
//					{
//						if(err == -2)
//						{
//							meanInlRatios[i].push_back(mymatcher->inlRatioL);
//							//inlierRatios_real[i] = mymatcher->inlRatioL;
//							inlRatPrecision[i].push_back(0);
//							inlRatRecall[i].push_back(0);
//							inlRatFallOut[i].push_back(1.0);
//							inlRatAccuracy[i].push_back(0);
//							continue;
//						}
//						else
//						{
//							continue;
//						}
//					}
//					meanInlRatios[i].push_back(mymatcher->inlRatioL);
//					//inlierRatios_real[i] = mymatcher->inlRatioL;
//					inlRatPrecision[i].push_back(mymatcher->qpm.ppv);
//					inlRatRecall[i].push_back(mymatcher->qpm.tpr);
//					inlRatFallOut[i].push_back(mymatcher->qpm.fpr);
//					inlRatAccuracy[i].push_back(mymatcher->qpm.acc);
//				
//					if(refine)
//					{
//						err = mymatcher->refineMatches();
//						if(err)
//						{
//							inlRatPrecisionRef[i].push_back(0);
//							inlRatRecallRef[i].push_back(0);
//							inlRatFallOutRef[i].push_back(1.0);
//							inlRatAccuracyRef[i].push_back(0);
//						}
//						else
//						{
//							inlRatPrecisionRef[i].push_back(mymatcher->qpr.ppv);
//							inlRatRecallRef[i].push_back(mymatcher->qpr.tpr);
//							inlRatFallOutRef[i].push_back(mymatcher->qpr.fpr);
//							inlRatAccuracyRef[i].push_back(mymatcher->qpr.acc);
//						}
//					}
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_inlRatio_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) +
//										"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//										matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//					if((showRefinedResult >= 0) && refine && !err)
//					{
//						if(storeRefResPath.empty())
//						{
//							mymatcher->showMatches(showRefinedResult);
//						}
//						else
//						{
//							if(dirExists(storeRefResPath)) //Check if output directory existis
//							{
//								outpath = storeRefResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "imgRef_inlRatio_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1))  + 
//										"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//										matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//						}
//					}
//				}
//			}
//			//Generate new homographys to evaluate all other possible configurations of the images to each other
//			int nr_evalsH = (int)fnames.size() - 1;
//			for(int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//			{
//				for(int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//				{
//					//H = (Hs[idx2].inv() * Hs[idx1]).inv();
//					cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//					src[0] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx2 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//					nr_evalsH++;
//					for(int i = 0; i < nr_inlratios; i++)
//					{
//						if(!matcherType.compare("CASCHASH"))
//						{
//							mymatcher.reset(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//						}
//						else if(!matcherType.compare("GMBSOF"))
//						{
//							mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//						}
//						else if(!matcherType.compare("HIRCLUIDX"))
//						{
//							mymatcher.reset(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else if(!matcherType.compare("HIRKMEANS"))
//						{
//							mymatcher.reset(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else if(!matcherType.compare("VFCKNN"))
//						{
//							mymatcher.reset(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], 2, useRatioTest));
//						}
//						else if(!matcherType.compare("LINEAR"))
//						{
//							mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else if(!matcherType.compare("LSHIDX"))
//						{
//							mymatcher.reset(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else if(!matcherType.compare("RANDKDTREE"))
//						{
//							mymatcher.reset(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else if (!matcherType.compare("BRUTEFORCENMS"))
//						{
//							mymatcher.reset(new bruteforceNMS_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else if (!matcherType.compare("SWGRAPH"))
//						{
//							mymatcher = std::auto_ptr<swgraph_matcher>(new swgraph_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//						}
//						else if (!matcherType.compare("HNSW"))
//						{
//							mymatcher = std::auto_ptr<hnsw_matcher>(new hnsw_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//						}
//						else if (!matcherType.compare("VPTREE"))
//						{
//							mymatcher = std::auto_ptr<vptree_matcher>(new vptree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//						}
//						else if (!matcherType.compare("ANNOY"))
//						{
//							mymatcher = std::auto_ptr<annoy_matcher>(new annoy_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else
//						{
//							cout << "No valid matcher specified! Exiting." << endl;
//							exit(1);
//						}
//
//						if(mymatcher->specialGMbSOFtest)
//						{
//							cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//							exit(0);
//						}
//
//						cout << "Testing image " << nr_evalsH << " with inlier ratio " << inlierRatios[i] << endl;
//						err = mymatcher->performMatching(inlierRatios[i]);
//						if(err)
//						{
//							if(err == -2)
//							{
//								meanInlRatios[i].push_back(mymatcher->inlRatioL);
//								//inlierRatios_real[i] = mymatcher->inlRatioL;
//								inlRatPrecision[i].push_back(0);
//								inlRatRecall[i].push_back(0);
//								inlRatFallOut[i].push_back(1.0);
//								inlRatAccuracy[i].push_back(0);
//								continue;
//							}
//							else
//							{
//								continue;
//							}
//						}
//						meanInlRatios[i].push_back(mymatcher->inlRatioL);
//						//inlierRatios_real[i] = mymatcher->inlRatioL;
//						inlRatPrecision[i].push_back(mymatcher->qpm.ppv);
//						inlRatRecall[i].push_back(mymatcher->qpm.tpr);
//						inlRatFallOut[i].push_back(mymatcher->qpm.fpr);
//						inlRatAccuracy[i].push_back(mymatcher->qpm.acc);
//				
//						if(refine)
//						{
//							err = mymatcher->refineMatches();
//							if(err)
//							{
//								inlRatPrecisionRef[i].push_back(0);
//								inlRatRecallRef[i].push_back(0);
//								inlRatFallOutRef[i].push_back(1.0);
//								inlRatAccuracyRef[i].push_back(0);
//							}
//							else
//							{
//								inlRatPrecisionRef[i].push_back(mymatcher->qpr.ppv);
//								inlRatRecallRef[i].push_back(mymatcher->qpr.tpr);
//								inlRatFallOutRef[i].push_back(mymatcher->qpr.fpr);
//								inlRatAccuracyRef[i].push_back(mymatcher->qpr.acc);
//							}
//						}
//
//						if(showResult >= 0)
//						{
//							if(storeImgResPath.empty())
//							{
//								mymatcher->showMatches(showResult);
//							}
//							else
//							{
//								if(dirExists(storeImgResPath)) //Check if output directory existis
//								{
//									outpath = storeImgResPath;
//								}
//								else
//								{
//									outpath = imgsPath + "\\evalImgs";
//									if(!dirExists(outpath))
//										_mkdir(outpath.c_str());
//								}
//								outfilename = "img_inlRatio_H_idxs" + std::to_string((ULONGLONG)(idx1 + 1)) + "-" + std::to_string((ULONGLONG)(idx2 + 1)) +
//											"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//											matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//											std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//								mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//							}
//						}
//						if((showRefinedResult >= 0) && refine && !err)
//						{
//							if(storeRefResPath.empty())
//							{
//								mymatcher->showMatches(showRefinedResult);
//							}
//							else
//							{
//								if(dirExists(storeRefResPath)) //Check if output directory existis
//								{
//									outpath = storeRefResPath;
//								}
//								else
//								{
//									outpath = imgsPath + "\\evalImgs";
//									if(!dirExists(outpath))
//										_mkdir(outpath.c_str());
//								}
//								outfilename = "imgRef_inlRatio_H_idxs" + std::to_string((ULONGLONG)(idx1 + 1)) + "-" + std::to_string((ULONGLONG)(idx2 + 1)) +
//											"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//											matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//											std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//								mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//							}
//						}
//					}
//				}
//			}
//		}
//		else
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//			for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//				if(!matcherType.compare("CASCHASH"))
//				{
//					mymatcher.reset(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("GMBSOF"))
//				{
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("HIRCLUIDX"))
//				{
//					mymatcher.reset(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("HIRKMEANS"))
//				{
//					mymatcher.reset(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("VFCKNN"))
//				{
//					mymatcher.reset(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 2, useRatioTest));
//				}
//				else if(!matcherType.compare("LINEAR"))
//				{
//					mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("LSHIDX"))
//				{
//					mymatcher.reset(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("RANDKDTREE"))
//				{
//					mymatcher.reset(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if (!matcherType.compare("BRUTEFORCENMS"))
//				{
//					mymatcher.reset(new bruteforceNMS_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if (!matcherType.compare("SWGRAPH"))
//				{
//					mymatcher = std::auto_ptr<swgraph_matcher>(new swgraph_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("HNSW"))
//				{
//					mymatcher = std::auto_ptr<hnsw_matcher>(new hnsw_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("VPTREE"))
//				{
//					mymatcher = std::auto_ptr<vptree_matcher>(new vptree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest, idxPars_NMSLIB, queryPars_NMSLIB));
//				}
//				else if (!matcherType.compare("ANNOY"))
//				{
//					mymatcher = std::auto_ptr<annoy_matcher>(new annoy_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else
//				{
//					cout << "No valid matcher specified! Exiting." << endl;
//					exit(1);
//				}
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				for(int i = 0; i < nr_inlratios; i++)
//				{
//					err = mymatcher->performMatching(inlierRatios[i]);
//					if(err)
//					{
//						if(err == -2)
//						{
//							meanInlRatios[i].push_back(mymatcher->inlRatioL);
//							//inlierRatios_real[i] = mymatcher->inlRatioL;
//							inlRatPrecision[i].push_back(0);
//							inlRatRecall[i].push_back(0);
//							inlRatFallOut[i].push_back(1.0);
//							inlRatAccuracy[i].push_back(0);
//							continue;
//						}
//						else
//						{
//							continue;
//						}
//					}
//					meanInlRatios[i].push_back(mymatcher->inlRatioL);
//					//inlierRatios_real[i] = mymatcher->inlRatioL;
//					inlRatPrecision[i].push_back(mymatcher->qpm.ppv);
//					inlRatRecall[i].push_back(mymatcher->qpm.tpr);
//					inlRatFallOut[i].push_back(mymatcher->qpm.fpr);
//					inlRatAccuracy[i].push_back(mymatcher->qpm.acc);
//				
//					if(refine)
//					{
//						err = mymatcher->refineMatches();
//						if(err)
//						{
//							inlRatPrecisionRef[i].push_back(0);
//							inlRatRecallRef[i].push_back(0);
//							inlRatFallOutRef[i].push_back(1.0);
//							inlRatAccuracyRef[i].push_back(0);
//						}
//						else
//						{
//							inlRatPrecisionRef[i].push_back(mymatcher->qpr.ppv);
//							inlRatRecallRef[i].push_back(mymatcher->qpr.tpr);
//							inlRatFallOutRef[i].push_back(mymatcher->qpr.fpr);
//							inlRatAccuracyRef[i].push_back(mymatcher->qpr.acc);
//						}
//					}
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_inlRatio_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) +
//										"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//										matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//					if((showRefinedResult >= 0) && refine && !err)
//					{
//						if(storeRefResPath.empty())
//						{
//							mymatcher->showMatches(showRefinedResult);
//						}
//						else
//						{
//							if(dirExists(storeRefResPath)) //Check if output directory existis
//							{
//								outpath = storeRefResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "imgRef_inlRatio_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1))  + 
//										"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//										matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showRefinedResult, true, outpath, outfilename, true);
//						}
//					}
//				}
//			}
//		}
//		if(inlRatPrecision[0].empty())
//		{
//			cout << "Ground truth generation or algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		for(int i = 0; i < nr_inlratios; i++)
//		{
//			if(!meanInlRatios[i].empty())
//			{
//				double mean_val = 0;
//				for(size_t j = 0; j < meanInlRatios[i].size(); j++)
//				{
//					mean_val += meanInlRatios[i][j];
//				}
//				inlierRatios_real[i] = mean_val / (double)meanInlRatios[i].size();
//			}
//
//			getStatisticfromVec2(inlRatPrecision[i], &inlRatPrecisionStat[i], false);
//			getStatisticfromVec2(inlRatRecall[i], &inlRatRecallStat[i], false);
//			getStatisticfromVec2(inlRatFallOut[i], &inlRatFallOutStat[i], false);
//			getStatisticfromVec2(inlRatAccuracy[i], &inlRatAccuracyStat[i], false);
//		}
//
//		if(refine)
//		{
//			if(inlRatPrecisionRef[0].empty())
//			{
//				cout << "Refinement algorithm failed on dataset! Exiting." << endl;
//				exit(1);
//			}
//			for(int i = 0; i < nr_inlratios; i++)
//			{
//				getStatisticfromVec2(inlRatPrecisionRef[i], &inlRatPrecisionStatRef[i], false);
//				getStatisticfromVec2(inlRatRecallRef[i], &inlRatRecallStatRef[i], false);
//				getStatisticfromVec2(inlRatFallOutRef[i], &inlRatFallOutStatRef[i], false);
//				getStatisticfromVec2(inlRatAccuracyRef[i], &inlRatAccuracyStatRef[i], false);
//			}
//		}
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_inlRatio_H_inliersGT_precision_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatPrecision[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatPrecisionStat[i].arithErr << " "
//								<< inlRatPrecisionStat[i].medErr << " "
//								<< *std::min_element(inlRatPrecision[i].begin(), inlRatPrecision[i].end()) << " "
//								<< *std::max_element(inlRatPrecision[i].begin(), inlRatPrecision[i].end()) << " "
//								<< inlRatPrecisionStat[i].arithStd << " " << inlRatPrecisionStat[i].medStd << " "
//								<< inlRatPrecisionStat[i].lowerQuart << " " << inlRatPrecisionStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_H_inliersGT_recall_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatRecall[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatRecallStat[i].arithErr << " "
//								<< inlRatRecallStat[i].medErr << " "
//								<< *std::min_element(inlRatRecall[i].begin(), inlRatRecall[i].end()) << " "
//								<< *std::max_element(inlRatRecall[i].begin(), inlRatRecall[i].end()) << " "
//								<< inlRatRecallStat[i].arithStd << " " << inlRatRecallStat[i].medStd << " "
//								<< inlRatRecallStat[i].lowerQuart << " " << inlRatRecallStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_H_inliersGT_fpr_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatFallOut[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatFallOutStat[i].arithErr << " "
//								<< inlRatFallOutStat[i].medErr << " "
//								<< *std::min_element(inlRatFallOut[i].begin(), inlRatFallOut[i].end()) << " "
//								<< *std::max_element(inlRatFallOut[i].begin(), inlRatFallOut[i].end()) << " "
//								<< inlRatFallOutStat[i].arithStd << " " << inlRatFallOutStat[i].medStd << " "
//								<< inlRatFallOutStat[i].lowerQuart << " " << inlRatFallOutStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_inlRatio_H_inliersGT_acc_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(inlRatAccuracy[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << inlRatAccuracyStat[i].arithErr << " "
//								<< inlRatAccuracyStat[i].medErr << " "
//								<< *std::min_element(inlRatAccuracy[i].begin(), inlRatAccuracy[i].end()) << " "
//								<< *std::max_element(inlRatAccuracy[i].begin(), inlRatAccuracy[i].end()) << " "
//								<< inlRatAccuracyStat[i].arithStd << " " << inlRatAccuracyStat[i].medStd << " "
//								<< inlRatAccuracyStat[i].lowerQuart << " " << inlRatAccuracyStat[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//		if(refine)
//		{
//			{
//				outfilename = "tex_inlRatio_H_inliersGT_precisionRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined precision or positive predictive value ppv=truePos/(truePos+falsePos)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatPrecisionRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatPrecisionStatRef[i].arithErr << " "
//								<< inlRatPrecisionStatRef[i].medErr << " "
//								<< *std::min_element(inlRatPrecisionRef[i].begin(), inlRatPrecisionRef[i].end()) << " "
//								<< *std::max_element(inlRatPrecisionRef[i].begin(), inlRatPrecisionRef[i].end()) << " "
//								<< inlRatPrecisionStatRef[i].arithStd << " " << inlRatPrecisionStatRef[i].medStd << " "
//								<< inlRatPrecisionStatRef[i].lowerQuart << " " << inlRatPrecisionStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_H_inliersGT_recallRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatRecallRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatRecallStatRef[i].arithErr << " "
//								<< inlRatRecallStatRef[i].medErr << " "
//								<< *std::min_element(inlRatRecallRef[i].begin(), inlRatRecallRef[i].end()) << " "
//								<< *std::max_element(inlRatRecallRef[i].begin(), inlRatRecallRef[i].end()) << " "
//								<< inlRatRecallStatRef[i].arithStd << " " << inlRatRecallStatRef[i].medStd << " "
//								<< inlRatRecallStatRef[i].lowerQuart << " " << inlRatRecallStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_H_inliersGT_fprRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatFallOutRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatFallOutStatRef[i].arithErr << " "
//								<< inlRatFallOutStatRef[i].medErr << " "
//								<< *std::min_element(inlRatFallOutRef[i].begin(), inlRatFallOutRef[i].end()) << " "
//								<< *std::max_element(inlRatFallOutRef[i].begin(), inlRatFallOutRef[i].end()) << " "
//								<< inlRatFallOutStatRef[i].arithStd << " " << inlRatFallOutStatRef[i].medStd << " "
//								<< inlRatFallOutStatRef[i].lowerQuart << " " << inlRatFallOutStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//			{
//				outfilename = "tex_inlRatio_H_inliersGT_accRef_" + featureDetector + "_" + descriptorExtractor + "_" + 
//								matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//				std::ofstream evalsToFile(outpath + "\\" + outfilename);
//				evalsToFile << "# Inlier ratio from ground truth in relation to refined Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)" << endl;
//				evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//				for(size_t i = 0; i < nr_inlratios; i++)
//				{
//					if(inlRatAccuracyRef[i].empty())
//						continue;
//					evalsToFile << inlierRatios_real[i] << " " << inlRatAccuracyStatRef[i].arithErr << " "
//								<< inlRatAccuracyStatRef[i].medErr << " "
//								<< *std::min_element(inlRatAccuracyRef[i].begin(), inlRatAccuracyRef[i].end()) << " "
//								<< *std::max_element(inlRatAccuracyRef[i].begin(), inlRatAccuracyRef[i].end()) << " "
//								<< inlRatAccuracyStatRef[i].arithStd << " " << inlRatAccuracyStatRef[i].medStd << " "
//								<< inlRatAccuracyStatRef[i].lowerQuart << " " << inlRatAccuracyStatRef[i].upperQuart << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//	return 0;
//}
//
///* Starts a runtime measurement for different inlier ratios on all image pairs of a szene for different matching algorithms and szenes.
// * The output is data for 3D visualisation (# of features, inlier ratio, time) and for 2D visualization (inlier ratio, avg. time per keypoint)
// *
// * string imgsPath				Input  -> Path which includes both left and right images
// * string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
// * int flowDispH				Input  -> Indicates which type of ground truth data is used:
// *										  0: flow files from KITTI database
// *										  1: disparity files from KITTI database
// *										  2: homography files (Please note that a homography always relates
// *											 to the first image (e.g. 1->2, 1->3, ...))
// * string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images 
// *									      (after prefix only comes the image number)
// * string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
// *									      (after prefix only comes the image number). For testing with homographies,
// *										  this string can be empty.
// * string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
// *									      (after prefix only comes the image number)
// * string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
// *										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
// *										  detectors are possible.
// * string descriptorExtractor	Input  -> The used descriptor extractor. Possible inputs should only be FREAK
// *										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
// * string matcherType			Input  -> The matcher type under test. Possible inputs are:
// *											CASCHASH: Cascade Hashing matcher
// *											GEOMAWARE: Geometry-aware Feature matching algorithm
// *											GMBSOF: Guided matching based on statistical optical flow
// *											HIRCLUIDX: Hirarchical Clustering Index Matching from the FLANN library
// *											HIRKMEANS: hierarchical k-means tree matcher from the FLANN library
// *											VFCKNN: Vector field consensus (VFC) algorithm with k nearest neighbor 
// *													matches provided from the Hirarchical Clustering Index Matching 
// *													algorithm from the FLANN library
// *											LIBVISO: matcher from the libviso2 library
// *											LINEAR: linear Matching algorithm (Brute force) from the FLANN library
// *											LSHIDX: LSH Index Matching algorithm from the FLANN library
// *											RANDKDTREE: randomized KD-trees matcher from the FLANN library
// * bool useRatioTest			Input  -> Specifies if a ratio test should be performed on the results of a matching
// *										  algorithm. The ratio test is only possible for the following algorithms:
// *										  HIRCLUIDX, HIRKMEANS, LINEAR, LSHIDX, RANDKDTREE
// * string storeResultPath		Input  -> Path were the resulting measurements should be stored
// * int showResult				Input  -> If >= 0, the result from the matching algorithm is displayed. The following
// *										  options are possible [DEFAULT = -1]:
// *											0:	Only true positives
// *											1:	True positives and false positives
// *											2:	True positives, false positives, and false negatives
// * string storeImgResPath		Input  -> Optional path for storing the resulting matches drawn into the images,
// *										  where the options of which results should be drawn are specified in
// *										  "showResult". If this path is set, the images are NOT displayed.
// * 
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Failed
// */
//int startTimeMeasurementDiffInlRats(std::string imgsPath, std::string flowDispHPath, int flowDispH, 
//										std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//										std::string featureDetector, std::string descriptorExtractor, std::string matcherType,
//										bool useRatioTest, std::string storeResultPath, bool useSameKeyPSiAllInl,
//										int showResult, std::string storeImgResPath)
//{
//	int err;
//	cv::Mat src[2];
//	std::auto_ptr<baseMatcher> mymatcher;
//	string outpath, outfilename;
//	vector<double> inlierRatios;
//	vector<double> inlierRatios_real, inlierRatios_real3D;
//	int nr_inlratios;
//
//	if(!storeImgResPath.empty() && (showResult == -1))
//	{
//		cout << "If you want to store the resulting images you must specify the showResult flag! Exiting." << endl;
//		exit(0);
//	}
//
//	//Generate inlier ratios
//	double startInlRatio = 1.0;
//	inlierRatios.push_back(startInlRatio);
//	while(startInlRatio > 0.2)
//	{
//		startInlRatio -= 0.05;
//		inlierRatios.push_back(startInlRatio);
//	}
//	while(startInlRatio > 0.1)
//	{
//		startInlRatio -= 0.02;
//		inlierRatios.push_back(startInlRatio);
//	}
//	while(startInlRatio > 0.01)
//	{
//		startInlRatio -= 0.01;
//		inlierRatios.push_back(startInlRatio);
//	}
//	/*while(startInlRatio > 0.005)
//	{
//		startInlRatio -= 0.005;
//		inlierRatios.push_back(startInlRatio);
//	}*/
//	nr_inlratios = (int)inlierRatios.size();
//	inlierRatios_real = inlierRatios;
//
//	vector<vector<pair<size_t, double>>> fmatchtime(nr_inlratios);
//	vector<vector<double>> timePerKeypVec(nr_inlratios);
//	vector<qualityParm1> timePerKeypStats(nr_inlratios);
//	vector<vector<double>> meanInlRatios(nr_inlratios);
//
//	if(flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//		
//		for(int k = 0; k < (int)filenamesl.size(); k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[k],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[k],CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageFlowFile(flowDispHPath, filenamesflow[k], &flowimg);
//			if(err)
//			{
//				cout << "Could not open flow file with index " << k << endl;
//				continue;
//			}
//			if(!matcherType.compare("CASCHASH"))
//			{
//				//mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//				mymatcher.reset(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//			}
//			else if(!matcherType.compare("GMBSOF"))
//			{
//				mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//			}
//			else if(!matcherType.compare("HIRCLUIDX"))
//			{
//				mymatcher.reset(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("HIRKMEANS"))
//			{
//				mymatcher.reset(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("VFCKNN"))
//			{
//				mymatcher.reset(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], 2, useRatioTest));
//			}
//			else if(!matcherType.compare("LINEAR"))
//			{
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("LSHIDX"))
//			{
//				mymatcher.reset(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("RANDKDTREE"))
//			{
//				mymatcher.reset(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else
//			{
//				cout << "No valid matcher specified! Exiting." << endl;
//				exit(1);
//			}
//			if(mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//			if(useSameKeyPSiAllInl)
//			{
//				mymatcher->useSameKeypSiVarInl = true;
//			}
//
//			for(int i = 0; i < nr_inlratios; i++)
//			{
//				err = mymatcher->performMatching(inlierRatios[i], true);
//				if(err)
//				{					
//					continue;
//				}
//				meanInlRatios[i].push_back(mymatcher->inlRatioL);
//				timePerKeypVec[i].push_back(mymatcher->tkm * 1000.0);
//				fmatchtime[i].push_back(make_pair((size_t)floor(mymatcher->positivesGT + mymatcher->negativesGTl + 0.5), mymatcher->tm));
//								
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_time_flow_idx" + std::to_string((ULONGLONG)k) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//			}
//		}
//		if(meanInlRatios[0].empty())
//		{
//			cout << "Ground truth generation or algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		for(int i = 0; i < nr_inlratios; i++)
//		{
//			if(!meanInlRatios[i].empty())
//			{
//				double mean_val = 0;
//				for(size_t j = 0; j < meanInlRatios[i].size(); j++)
//				{
//					mean_val += meanInlRatios[i][j];
//				}
//				inlierRatios_real[i] = mean_val / (double)meanInlRatios[i].size();
//			}
//
//			getStatisticfromVec2(timePerKeypVec[i], &timePerKeypStats[i], false);
//		}
//
//		getSameVectorSizes(fmatchtime, inlierRatios_real, inlierRatios_real3D);
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_time_flow_DiffInlRats_3D_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation matching time and number of features." << endl;
//			evalsToFile << "inlrat nrfeatures timems" << endl;
//			for(size_t i = 0; i < inlierRatios_real3D.size(); i++)
//			{
//				for(size_t j = 0; j < fmatchtime[i].size(); j++)
//				{
//					evalsToFile << inlierRatios_real3D[i] << " " << fmatchtime[i][j].first << " " << fmatchtime[i][j].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_time_flow_DiffInlRats_2D_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to an average matching time per keypoint in microseconds [us]" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(timePerKeypVec[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << timePerKeypStats[i].arithErr << " "
//								<< timePerKeypStats[i].medErr << " "
//								<< *std::min_element(timePerKeypVec[i].begin(), timePerKeypVec[i].end()) << " "
//								<< *std::max_element(timePerKeypVec[i].begin(), timePerKeypVec[i].end()) << " "
//								<< timePerKeypStats[i].arithStd << " " << timePerKeypStats[i].medStd << " "
//								<< timePerKeypStats[i].lowerQuart << " " << timePerKeypStats[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//	}
//	else if(flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		for(int k = 0; k < (int)filenamesl.size(); k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[k],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[k],CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageDisparityFile(flowDispHPath, filenamesflow[k], &flowimg);
//			if(err)
//			{
//				cout << "Could not open disparity file with index " << k << ". Exiting." << endl;
//				exit(0);
//			}
//			if(!matcherType.compare("CASCHASH"))
//			{
//				//mymatcher = std::auto_ptr<CascadeHashing_matcher>(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//				mymatcher.reset(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//			}
//			else if(!matcherType.compare("GMBSOF"))
//			{
//				mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k]));
//			}
//			else if(!matcherType.compare("HIRCLUIDX"))
//			{
//				mymatcher.reset(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("HIRKMEANS"))
//			{
//				mymatcher.reset(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("VFCKNN"))
//			{
//				mymatcher.reset(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], 2, useRatioTest));
//			}
//			else if(!matcherType.compare("LINEAR"))
//			{
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("LSHIDX"))
//			{
//				mymatcher.reset(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else if(!matcherType.compare("RANDKDTREE"))
//			{
//				mymatcher.reset(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, flowimg, true, imgsPath, filenamesl[k], useRatioTest));
//			}
//			else
//			{
//				cout << "No valid matcher specified! Exiting." << endl;
//				exit(1);
//			}
//
//			if(mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			if(useSameKeyPSiAllInl)
//			{
//				mymatcher->useSameKeypSiVarInl = true;
//			}
//
//			for(int i = 0; i < nr_inlratios; i++)
//			{
//				err = mymatcher->performMatching(inlierRatios[i], true);
//				if(err)
//				{					
//					continue;
//				}
//				meanInlRatios[i].push_back(mymatcher->inlRatioL);
//				timePerKeypVec[i].push_back(mymatcher->tkm * 1000.0);
//				fmatchtime[i].push_back(make_pair((size_t)floor(mymatcher->positivesGT + mymatcher->negativesGTl + 0.5), mymatcher->tm));
//
//				if(showResult >= 0)
//				{
//					if(storeImgResPath.empty())
//					{
//						mymatcher->showMatches(showResult);
//					}
//					else
//					{
//						if(dirExists(storeImgResPath)) //Check if output directory existis
//						{
//							outpath = storeImgResPath;
//						}
//						else
//						{
//							outpath = imgsPath + "\\evalImgs";
//							if(!dirExists(outpath))
//								_mkdir(outpath.c_str());
//						}
//						outfilename = "img_time_disp_idx" + std::to_string((ULONGLONG)k) + "_" + featureDetector + "_" + descriptorExtractor + "_" + 
//									matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//									std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//						mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//					}
//				}
//			}
//		}
//
//		if(meanInlRatios[0].empty())
//		{
//			cout << "Ground truth generation or algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		for(int i = 0; i < nr_inlratios; i++)
//		{
//			if(!meanInlRatios[i].empty())
//			{
//				double mean_val = 0;
//				for(size_t j = 0; j < meanInlRatios[i].size(); j++)
//				{
//					mean_val += meanInlRatios[i][j];
//				}
//				inlierRatios_real[i] = mean_val / (double)meanInlRatios[i].size();
//			}
//
//			getStatisticfromVec2(timePerKeypVec[i], &timePerKeypStats[i], false);
//		}
//
//		getSameVectorSizes(fmatchtime, inlierRatios_real, inlierRatios_real3D);
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_time_disp_DiffInlRats_3D_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation matching time and number of features." << endl;
//			evalsToFile << "inlrat nrfeatures timems" << endl;
//			for(size_t i = 0; i < inlierRatios_real3D.size(); i++)
//			{
//				for(size_t j = 0; j < fmatchtime[i].size(); j++)
//				{
//					evalsToFile << inlierRatios_real3D[i] << " " << fmatchtime[i][j].first << " " << fmatchtime[i][j].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_time_disp_DiffInlRats_2D_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to an average matching time per keypoint in microseconds [us]" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(timePerKeypVec[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << timePerKeypStats[i].arithErr << " "
//								<< timePerKeypStats[i].medErr << " "
//								<< *std::min_element(timePerKeypVec[i].begin(), timePerKeypVec[i].end()) << " "
//								<< *std::max_element(timePerKeypVec[i].begin(), timePerKeypVec[i].end()) << " "
//								<< timePerKeypStats[i].arithStd << " " << timePerKeypStats[i].medStd << " "
//								<< timePerKeypStats[i].lowerQuart << " " << timePerKeypStats[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//	}
//	else if(flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if(err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if(err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if(err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		if(fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//			for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//				if(!matcherType.compare("CASCHASH"))
//				{
//					mymatcher.reset(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("GMBSOF"))
//				{
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("HIRCLUIDX"))
//				{
//					mymatcher.reset(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("HIRKMEANS"))
//				{
//					mymatcher.reset(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("VFCKNN"))
//				{
//					mymatcher.reset(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 2, useRatioTest));
//				}
//				else if(!matcherType.compare("LINEAR"))
//				{
//					mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("LSHIDX"))
//				{
//					mymatcher.reset(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("RANDKDTREE"))
//				{
//					mymatcher.reset(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else
//				{
//					cout << "No valid matcher specified! Exiting." << endl;
//					exit(1);
//				}
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				if(useSameKeyPSiAllInl)
//				{
//					mymatcher->useSameKeypSiVarInl = true;
//				}
//
//				for(int i = 0; i < nr_inlratios; i++)
//				{
//					err = mymatcher->performMatching(inlierRatios[i], true);
//					if(err)
//					{					
//						continue;
//					}
//					meanInlRatios[i].push_back(mymatcher->inlRatioL);
//					timePerKeypVec[i].push_back(mymatcher->tkm * 1000.0);
//					fmatchtime[i].push_back(make_pair((size_t)floor(mymatcher->positivesGT + mymatcher->negativesGTl + 0.5), mymatcher->tm));
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_time_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) +
//										"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//										matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//				}
//			}
//			//Generate new homographys to evaluate all other possible configurations of the images to each other
//			for(int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//			{
//				for(int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//				{
//					//H = (Hs[idx2].inv() * Hs[idx1]).inv();
//					cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//					src[0] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx2 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//					for(int i = 0; i < nr_inlratios; i++)
//					{
//						if(!matcherType.compare("CASCHASH"))
//						{
//							mymatcher.reset(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//						}
//						else if(!matcherType.compare("GMBSOF"))
//						{
//							mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//						}
//						else if(!matcherType.compare("HIRCLUIDX"))
//						{
//							mymatcher.reset(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else if(!matcherType.compare("HIRKMEANS"))
//						{
//							mymatcher.reset(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else if(!matcherType.compare("VFCKNN"))
//						{
//							mymatcher.reset(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], 2, useRatioTest));
//						}
//						else if(!matcherType.compare("LINEAR"))
//						{
//							mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else if(!matcherType.compare("LSHIDX"))
//						{
//							mymatcher.reset(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else if(!matcherType.compare("RANDKDTREE"))
//						{
//							mymatcher.reset(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1], useRatioTest));
//						}
//						else
//						{
//							cout << "No valid matcher specified! Exiting." << endl;
//							exit(1);
//						}
//
//						if(mymatcher->specialGMbSOFtest)
//						{
//							cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//							exit(0);
//						}
//
//						if(useSameKeyPSiAllInl)
//						{
//							mymatcher->useSameKeypSiVarInl = true;
//						}
//
//						err = mymatcher->performMatching(inlierRatios[i], true);
//						if(err)
//						{					
//							continue;
//						}
//						meanInlRatios[i].push_back(mymatcher->inlRatioL);
//						timePerKeypVec[i].push_back(mymatcher->tkm * 1000.0);
//						fmatchtime[i].push_back(make_pair((size_t)floor(mymatcher->positivesGT + mymatcher->negativesGTl + 0.5), mymatcher->tm));
//
//						if(showResult >= 0)
//						{
//							if(storeImgResPath.empty())
//							{
//								mymatcher->showMatches(showResult);
//							}
//							else
//							{
//								if(dirExists(storeImgResPath)) //Check if output directory existis
//								{
//									outpath = storeImgResPath;
//								}
//								else
//								{
//									outpath = imgsPath + "\\evalImgs";
//									if(!dirExists(outpath))
//										_mkdir(outpath.c_str());
//								}
//								outfilename = "img_time_H_idxs" + std::to_string((ULONGLONG)(idx1 + 1)) + "-" + std::to_string((ULONGLONG)(idx2 + 1)) +
//											"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//											matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//											std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//								mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//							}
//						}
//					}
//				}
//			}
//		}
//		else
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//			for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//				if(!matcherType.compare("CASCHASH"))
//				{
//					mymatcher.reset(new CascadeHashing_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("GMBSOF"))
//				{
//					mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				}
//				else if(!matcherType.compare("HIRCLUIDX"))
//				{
//					mymatcher.reset(new HirClustIdx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("HIRKMEANS"))
//				{
//					mymatcher.reset(new HirarchKMeans_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("VFCKNN"))
//				{
//					mymatcher.reset(new VFCknn_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], 2, useRatioTest));
//				}
//				else if(!matcherType.compare("LINEAR"))
//				{
//					mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("LSHIDX"))
//				{
//					mymatcher.reset(new LSHidx_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else if(!matcherType.compare("RANDKDTREE"))
//				{
//					mymatcher.reset(new RandKDTree_matcher(src[0], src[1], featureDetector, descriptorExtractor, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1], useRatioTest));
//				}
//				else
//				{
//					cout << "No valid matcher specified! Exiting." << endl;
//					exit(1);
//				}
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				if(useSameKeyPSiAllInl)
//				{
//					mymatcher->useSameKeypSiVarInl = true;
//				}
//
//				for(int i = 0; i < nr_inlratios; i++)
//				{
//					err = mymatcher->performMatching(inlierRatios[i], true);
//					if(err)
//					{					
//						continue;
//					}
//					meanInlRatios[i].push_back(mymatcher->inlRatioL);
//					timePerKeypVec[i].push_back(mymatcher->tkm * 1000.0);
//					fmatchtime[i].push_back(make_pair((size_t)floor(mymatcher->positivesGT + mymatcher->negativesGTl + 0.5), mymatcher->tm));
//
//					if(showResult >= 0)
//					{
//						if(storeImgResPath.empty())
//						{
//							mymatcher->showMatches(showResult);
//						}
//						else
//						{
//							if(dirExists(storeImgResPath)) //Check if output directory existis
//							{
//								outpath = storeImgResPath;
//							}
//							else
//							{
//								outpath = imgsPath + "\\evalImgs";
//								if(!dirExists(outpath))
//									_mkdir(outpath.c_str());
//							}
//							outfilename = "img_time_H_idxs0-" + std::to_string((ULONGLONG)(idx1 + 1)) +
//										"_" + featureDetector + "_" + descriptorExtractor + "_" + 
//										matcherType + (useRatioTest ? "_withRatTest_":"_noRatTest_") + "inlRat" + 
//										std::to_string((ULONGLONG)std::floor(mymatcher->inlRatioL*1000.0 + 0.5)) + ".bmp";
//							mymatcher->showMatches(showResult, false, outpath, outfilename, true);
//						}
//					}
//				}
//			}
//		}
//		if(meanInlRatios[0].empty())
//		{
//			cout << "Ground truth generation or algorithm failed on dataset! Exiting." << endl;
//			exit(1);
//		}
//
//		for(int i = 0; i < nr_inlratios; i++)
//		{
//			if(!meanInlRatios[i].empty())
//			{
//				double mean_val = 0;
//				for(size_t j = 0; j < meanInlRatios[i].size(); j++)
//				{
//					mean_val += meanInlRatios[i][j];
//				}
//				inlierRatios_real[i] = mean_val / (double)meanInlRatios[i].size();
//			}
//
//			getStatisticfromVec2(timePerKeypVec[i], &timePerKeypStats[i], false);
//		}
//
//		getSameVectorSizes(fmatchtime, inlierRatios_real, inlierRatios_real3D);
//
//		if(dirExists(storeResultPath)) //Check if output directory existis
//		{
//			outpath = storeResultPath;
//		}
//		else
//		{
//			outpath = imgsPath + "\\evalResult";
//			if(!dirExists(outpath))
//				_mkdir(outpath.c_str());
//		}
//		{
//			outfilename = "tex_time_H_DiffInlRats_3D_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation matching time and number of features." << endl;
//			evalsToFile << "inlrat nrfeatures timems" << endl;
//			for(size_t i = 0; i < inlierRatios_real3D.size(); i++)
//			{
//				for(size_t j = 0; j < fmatchtime[i].size(); j++)
//				{
//					evalsToFile << inlierRatios_real3D[i] << " " << fmatchtime[i][j].first << " " << fmatchtime[i][j].second << endl;
//				}
//			}
//			evalsToFile.close();
//		}
//		{
//			outfilename = "tex_time_H_DiffInlRats_2D_" + featureDetector + "_" + descriptorExtractor + "_" + 
//							matcherType + (useRatioTest ? "_withRatTest.dat":"_noRatTest.dat");
//
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "# Inlier ratio from ground truth in relation to an average matching time per keypoint in microseconds [us]" << endl;
//			evalsToFile << "inlrat mean med min max stdMean stdMed lowQuart uppQuart" << endl;
//			for(size_t i = 0; i < nr_inlratios; i++)
//			{
//				if(timePerKeypVec[i].empty())
//					continue;
//				evalsToFile << inlierRatios_real[i] << " " << timePerKeypStats[i].arithErr << " "
//								<< timePerKeypStats[i].medErr << " "
//								<< *std::min_element(timePerKeypVec[i].begin(), timePerKeypVec[i].end()) << " "
//								<< *std::max_element(timePerKeypVec[i].begin(), timePerKeypVec[i].end()) << " "
//								<< timePerKeypStats[i].arithStd << " " << timePerKeypStats[i].medStd << " "
//								<< timePerKeypStats[i].lowerQuart << " " << timePerKeypStats[i].upperQuart << endl;
//			}
//			evalsToFile.close();
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//	return 0;
//}
//
//void getSameVectorSizes(std::vector<std::vector<std::pair<size_t, double>>> & vec, std::vector<double> inlratsin, std::vector<double> & inlratsout)
//{
//	/*vector<vector<pair<size_t, double>>> vec1, vec_tmp, vec_tmp1;
//	bool mirrord = false;*/
//	//size_t min_size = UINT_MAX;
//	size_t max_size = 0;
//	//size_t maxidx, minidx;
//	//const size_t maxaddvec = 4;
//
//	inlratsout = inlratsin;
//
//	for(size_t i = 0; i < vec.size(); i++)
//	{
//		if(vec[i].size() > max_size)
//		{
//			max_size = vec[i].size();
//			//maxidx = i;
//		}
//		/*if(vec[i].size() < min_size)
//		{
//			min_size = vec[i].size();
//			minidx = i;
//		}*/
//		if(vec[i].size() == 0)
//		{
//			//min_size = UINT_MAX;
//			max_size = 0;
//			vec.erase(vec.begin() + i);
//			inlratsout.erase(inlratsout.begin() + i);
//			i = 0;
//		}
//	}
//
//	for(size_t i = 0; i < vec.size(); i++)
//	{
//		if(vec[i].size() < max_size)
//		{
//			int addfullmulti = (int)floor((double)max_size / (double)vec[i].size() + DBL_MIN);
//			vector<pair<size_t,double>> hlp_vec;
//
//			if(addfullmulti)
//			{
//				hlp_vec = vec[i];
//				for(int j = 0; j < addfullmulti - 1; j++)
//				{
//					hlp_vec.insert(hlp_vec.end(), vec[i].begin(), vec[i].end());
//				}
//			}
//			int idx = 0;
//			while(hlp_vec.size() < max_size)
//			{
//				hlp_vec.push_back(vec[i][idx]);
//				idx++;
//			}
//			vec[i] = hlp_vec;
//		}
//	}
//
///*
//
//
//
//
//	if(minidx > maxidx)
//	{
//		for(int i = (int)vec1.size() - 1; i >= 0; i--)
//		{
//			vec_tmp.push_back(vec1[i]);
//			inlrats_tmp.push_back(inlratsin[i]);
//		}
//		mirrord = true;
//	}
//	else
//	{
//		vec_tmp = vec1;
//		inlrats_tmp = inlratsin;
//	}
//
//
//	if(max_size != min_size)
//	{
//		int actidx = 0;
//		int lastidx = 0;
//		int addIdxs[2] = {-1,-1};
//
//		while(lastidx < vec_tmp.size() - 1)
//		{
//			int addedSize = vec_tmp[actidx].size();
//			addIdxs[0] = actidx;
//			for(size_t i = actidx + 1; (i < actidx + maxaddvec) && (i < vec_tmp.size()); i++)
//			{
//				lastidx = i;
//				int diffS = (int)max_size - addedSize;
//				if(diffS < 0)
//				{
//					if(i == actidx + 1)
//					{
//						addIdxs[1] = -1;
//					}
//					else
//					{
//						addIdxs[1] = i - 1;
//					}
//					break;
//				}
//				diffS -= (int)vec_tmp[i].size();
//				if((diffS < 0) && ((float)std::abs(diffS) / (float)vec_tmp[i].size() > 0.5f))
//				{
//					if(i == actidx + 1)
//					{
//						addIdxs[1] = -1;
//					}
//					else
//					{
//						addIdxs[1] = i - 1;
//					}
//					break;
//				}
//				addedSize += vec_tmp[i].size();
//				addIdxs[1] = i;
//			}
//			if(addIdxs[1] < 0)
//			{
//				vec_tmp1.push_back(vec_tmp[addIdxs[0]]);
//				inlrats_tmp1.push_back(inlrats_tmp1[addIdxs[0]]);
//			}
//			else
//			{
//				vec_tmp1.push_back(vec_tmp[addIdxs[0]]);
//				for(int i = addIdxs[0] + 1; i <= addIdxs[1]; i++)
//				{
//					vec_tmp1.back().insert(vec_tmp1.back().end(), vec_tmp[i].begin(), vec_tmp[i].end());
//				}
//				double suminlrats = 0;
//				size_t sumnrinl = 0;
//				for(int i = addIdxs[0]; i <= addIdxs[1]; i++)
//				{
//					suminlrats += inlrats_tmp[i] * (double)vec_tmp[i].size();
//					sumnrinl += vec_tmp[i].size();
//				}
//				inlrats_tmp1.push_back(suminlrats / (double)sumnrinl);
//			}
//
//		}
//	}*/
//}
//
///* Starts the test for different inlier ratios on all image pairs of a szene for different matching algorithms and szenes.
// * The output is generated for precision, recall, fall-out and accuracy and is the mean value over all images.
// *
// * string imgsPath				Input  -> Path which includes both left and right images
// * string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
// * int flowDispH				Input  -> Indicates which type of ground truth data is used:
// *										  0: flow files from KITTI database
// *										  1: disparity files from KITTI database
// *										  2: homography files (Please note that a homography always relates
// *											 to the first image (e.g. 1->2, 1->3, ...))
// * string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images 
// *									      (after prefix only comes the image number)
// * string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
// *									      (after prefix only comes the image number). For testing with homographies,
// *										  this string can be empty.
// * string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
// *									      (after prefix only comes the image number)
// * string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
// *										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
// *										  detectors are possible.
// * string storeResultPath		Input  -> Path were the resulting measurements should be stored
// * string descriptorExtractorGT	Input  -> The used descriptor extractor for generating GT. Possible inputs should only be FREAK
// *										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
// * double threshhTh				Input  -> Threshold for thresholding the difference of matching image patches to decide if
// *										  a match should be annotated manually or automatically. A value of 64 has proofen to
// *										  be a good value for normal images, whereas 20 should be chosen for synthetic images.
// * 
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Failed
// */
//int testGTmatches(std::string imgsPath, std::string flowDispHPath, int flowDispH, 
//				  std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//				  std::string featureDetector, std::string storeResultPath, std::string descriptorExtractorGT, double threshhTh)
//{
//	int err;
//	cv::Mat src[2];
//	std::auto_ptr<baseMatcher> mymatcher;
//	string outpath, outfilename, globOutfileName;
//	int nrGTmatchesDataset = 0;
//	bool datafileexists = false;
//	int nrTotalFails = 0, nrSamplesTillNow = 0;
//	std::vector<double> distances;
//	std::ofstream globEvalsToFile;
//	int nrimgs, nrremainingimgs;
//	int truePosArr[5] = {0,0,0,0,0};
//	int falsePosArr[5] = {0,0,0,0,0};
//	int falseNegArr[5] = {0,0,0,0,0};
//	int notMatchable = 0;
//	std::vector<cv::Point2f> errvecs;
//	std::vector<std::pair<cv::Point2f,cv::Point2f>> perfectMatches;
//	std::vector<cv::Point2f> errvecsGT;
//	std::vector<double> distancesGT;
//	std::vector<int> validityValGT;
//	std::vector<double> distancesEstModel;
//	annotImgPars annotationData;
//	string fileNamesLR;
//	if(dirExists(storeResultPath)) //Check if output directory existis
//	{
//		outpath = storeResultPath;
//	}
//	else
//	{
//		outpath = imgsPath + "\\GTTestResult";
//		if(!dirExists(outpath))
//			_mkdir(outpath.c_str());
//	}
//
//	if(flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//		
//		outfilename = "GTmatchesTest_" + featureDetector + "_" + descriptorExtractorGT + "_flow.txt";
//
//		//Check if evalation was done before
//		checkPrevEval(outpath, outfilename, filenamesl, datafileexists, nrGTmatchesDataset);
//
//		if(!datafileexists)
//		{
//			vector<double> GTths;
//			for(int k = 0; k < (int)filenamesl.size(); k++)
//			{
//				cv::Mat flowimg;
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[k],CV_LOAD_IMAGE_GRAYSCALE);
//				src[1] = cv::imread(imgsPath + "\\" + filenamesr[k],CV_LOAD_IMAGE_GRAYSCALE);
//				err = convertImageFlowFile(flowDispHPath, filenamesflow[k], &flowimg);
//				if(err)
//				{
//					cout << "Could not open flow file with index " << k << endl;
//					continue;
//				}
//
//				//mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k]));
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k]));
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				mymatcher->specifiedInlRatio = 1.0;
//				mymatcher->GTfilterExtractor = descriptorExtractorGT;
//				mymatcher->detectFeatures();
//				mymatcher->checkForGT();
//				nrGTmatchesDataset += (int)mymatcher->positivesGT;
//				GTths.push_back(mymatcher->usedMatchTH);
//			}
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "GTinliers " << nrGTmatchesDataset << endl;
//			evalsToFile.close();
//
//			std::ofstream GTthsToFile(outpath + "\\GT-thresholds_" + outfilename);
//			GTthsToFile << "thresholds" << endl;
//			for(int i = 0; i < (int)GTths.size(); i++)
//			{
//				GTthsToFile << GTths[i] << endl;
//			}
//			GTthsToFile.close();
//		}
//		
//		std::ifstream evalsToFilei(outpath + "\\" + outfilename);
//		if(!evalsToFilei.good() && !evalsToFilei.is_open())
//		{
//			cout << "Error opening file " << outfilename << endl;
//			cout << "Exiting." << endl;
//			evalsToFilei.close();
//			exit(0);
//		}
//		int starttestimgnr = 0;
//		int missingSamples = 0;
//		vector<vector<double>> StaticSamplesizes; //Elements of 2nd vector: p, e, n
//		double maxmaxSampleSize, sampleRatio, minminSampleSize;
//
//		readParametersTested(starttestimgnr, nrTotalFails, nrSamplesTillNow, filenamesl, &evalsToFilei, outpath, outfilename, missingSamples,
//						  StaticSamplesizes, maxmaxSampleSize, sampleRatio, minminSampleSize, nrGTmatchesDataset, flowDispH, distances,
//						  truePosArr, falsePosArr, falseNegArr, notMatchable, errvecs, perfectMatches, errvecsGT, distancesGT, 
//						  validityValGT, distancesEstModel, annotationData);
//
//		evalsToFilei.close();
//#if (CORRECT_MATCHING_RESULT != 1) && (CORRECT_MATCHING_RESULT != 3)
//		globEvalsToFile.open(outpath + "\\" + outfilename, ios::app);
//		if(!globEvalsToFile.good() && !globEvalsToFile.is_open())
//		{
//			cout << "Error opening file " << outfilename << endl;
//			cout << "Exiting." << endl;
//			globEvalsToFile.close();
//			exit(0);
//		}
//#elif CORRECT_MATCHING_RESULT == 3
//		globEvalsToFile.open(outpath + "\\" + outfilename + "_tmp", ios::app);
//		if(!globEvalsToFile.good() && !globEvalsToFile.is_open())
//		{
//			cout << "Error opening file " << outfilename << endl;
//			cout << "Exiting." << endl;
//			globEvalsToFile.close();
//			exit(0);
//		}
//#endif
//
//		nrimgs = (int)filenamesl.size();
//		for(int k = starttestimgnr; k < nrimgs; k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[k],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[k],CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageFlowFile(flowDispHPath, filenamesflow[k], &flowimg);
//			if(err)
//			{
//				cout << "Could not open flow file with index " << k << endl;
//				continue;
//			}
//
//			//mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k]));
//			mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k]));
//			
//			nrremainingimgs = nrimgs - k - 1;
//			fileNamesLR = filenamesl[k] + "_" + filenamesr[k];
//			if(testImagePairs(mymatcher, descriptorExtractorGT, sampleRatio, missingSamples, &globEvalsToFile, nrSamplesTillNow, nrTotalFails,
//							  minminSampleSize, StaticSamplesizes, nrGTmatchesDataset, distances, nrremainingimgs,
//							  truePosArr, falsePosArr, falseNegArr, notMatchable, errvecs, perfectMatches,
//							  errvecsGT, distancesGT, validityValGT, distancesEstModel, annotationData, fileNamesLR, threshhTh, k, outpath, outfilename) != 0) return -1;
//
//		}
//#if CORRECT_MATCHING_RESULT == 1
//		return 0;
//#endif
//		globOutfileName = outfilename;
//	}
//	else if(flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if(err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		outfilename = "GTmatchesTest_" + featureDetector + "_" + descriptorExtractorGT + "_disp.txt";
//
//		//Check if evalation was done before
//		checkPrevEval(outpath, outfilename, filenamesl, datafileexists, nrGTmatchesDataset);
//
//		if(!datafileexists)
//		{
//			vector<double> GTths;
//			for(int k = 0; k < (int)filenamesl.size(); k++)
//			{
//				cv::Mat flowimg;
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[k],CV_LOAD_IMAGE_GRAYSCALE);
//				src[1] = cv::imread(imgsPath + "\\" + filenamesr[k],CV_LOAD_IMAGE_GRAYSCALE);
//				err = convertImageDisparityFile(flowDispHPath, filenamesflow[k], &flowimg);
//				if(err)
//				{
//					cout << "Could not open disparity file with index " << k << ". Exiting." << endl;
//					exit(0);
//				}
//
//				//mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k]));
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k]));
//
//				if(mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				mymatcher->specifiedInlRatio = 1.0;
//				mymatcher->GTfilterExtractor = descriptorExtractorGT;
//				mymatcher->detectFeatures();
//				mymatcher->checkForGT();
//				nrGTmatchesDataset += (int)mymatcher->positivesGT;
//				GTths.push_back(mymatcher->usedMatchTH);
//			}
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "GTinliers " << nrGTmatchesDataset << endl;
//			evalsToFile.close();
//
//			std::ofstream GTthsToFile(outpath + "\\GT-thresholds_" + outfilename);
//			GTthsToFile << "thresholds" << endl;
//			for(int i = 0; i < (int)GTths.size(); i++)
//			{
//				GTthsToFile << GTths[i] << endl;
//			}
//			GTthsToFile.close();
//		}
//
//		std::ifstream evalsToFilei(outpath + "\\" + outfilename);
//		if(!evalsToFilei.good() && !evalsToFilei.is_open())
//		{
//			cout << "Error opening file " << outfilename << endl;
//			cout << "Exiting." << endl;
//			evalsToFilei.close();
//			exit(0);
//		}
//		int starttestimgnr = 0;
//		int missingSamples = 0;
//		vector<vector<double>> StaticSamplesizes; //Elements of 2nd vector: p, e, n
//		double maxmaxSampleSize, sampleRatio, minminSampleSize;
//
//		readParametersTested(starttestimgnr, nrTotalFails, nrSamplesTillNow, filenamesl, &evalsToFilei, outpath, outfilename, missingSamples,
//						  StaticSamplesizes, maxmaxSampleSize, sampleRatio, minminSampleSize, nrGTmatchesDataset, flowDispH, distances,
//						  truePosArr, falsePosArr, falseNegArr, notMatchable, errvecs, perfectMatches, errvecsGT, distancesGT, 
//						  validityValGT, distancesEstModel, annotationData);
//
//		evalsToFilei.close();
//		globEvalsToFile.open(outpath + "\\" + outfilename, ios::app);
//		if(!globEvalsToFile.good() && !globEvalsToFile.is_open())
//		{
//			cout << "Error opening file " << outfilename << endl;
//			cout << "Exiting." << endl;
//			globEvalsToFile.close();
//			exit(0);
//		}
//
//		nrimgs = (int)filenamesl.size();
//		for(int k = starttestimgnr; k < nrimgs; k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[k],CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[k],CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageDisparityFile(flowDispHPath, filenamesflow[k], &flowimg);
//			if(err)
//			{
//				cout << "Could not open disparity file with index " << k << ". Exiting." << endl;
//				exit(0);
//			}
//
//			//mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k]));
//			mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k]));
//			
//			nrremainingimgs = nrimgs - k - 1;
//			fileNamesLR = filenamesl[k] + "_" + filenamesr[k];
//			if(testImagePairs(mymatcher, descriptorExtractorGT, sampleRatio, missingSamples, &globEvalsToFile, nrSamplesTillNow, nrTotalFails,
//							  minminSampleSize, StaticSamplesizes, nrGTmatchesDataset, distances, nrremainingimgs,
//							  truePosArr, falsePosArr, falseNegArr, notMatchable, errvecs, perfectMatches,
//							  errvecsGT, distancesGT, validityValGT, distancesEstModel, annotationData, fileNamesLR, threshhTh, k) != 0) return -1;
//		}
//		globOutfileName = outfilename;
//	}
//	else if(flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if(err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if(err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if(err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		outfilename = "GTmatchesTest_" + featureDetector + "_" + descriptorExtractorGT + "_homo.txt";
//
//		//Check if evalation was done before
//		checkPrevEval(outpath, outfilename, filenamesl, datafileexists, nrGTmatchesDataset);
//
//		if(!datafileexists)
//		{
//			vector<double> GTths;
//			if(fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//			{
//				//Take the stored homographys and perform evaluation
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//				for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//				{
//					cv::Mat H = Hs[idx1];
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//					//mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//					mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//					if(mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					mymatcher->specifiedInlRatio = 1.0;
//					mymatcher->GTfilterExtractor = descriptorExtractorGT;
//					mymatcher->detectFeatures();
//					mymatcher->checkForGT();
//					nrGTmatchesDataset += (int)mymatcher->positivesGT;
//					GTths.push_back(mymatcher->usedMatchTH);
//				}
//				//Generate new homographys to evaluate all other possible configurations of the images to each other
//				for(int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//				{
//					for(int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//					{
//						//H = (Hs[idx2].inv() * Hs[idx1]).inv();
//						cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//						src[0] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//						src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx2 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//						//mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//						mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//
//						if(mymatcher->specialGMbSOFtest)
//						{
//							cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//							exit(0);
//						}
//
//						mymatcher->specifiedInlRatio = 1.0;
//						mymatcher->GTfilterExtractor = descriptorExtractorGT;
//						mymatcher->detectFeatures();
//						mymatcher->checkForGT();
//						nrGTmatchesDataset += (int)mymatcher->positivesGT;
//						GTths.push_back(mymatcher->usedMatchTH);
//					}
//				}
//			}
//			else
//			{
//				//Take the stored homographys and perform evaluation
//				src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//				for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++) 
//				{
//					cv::Mat H = Hs[idx1];
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//					//mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//					mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//					if(mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					mymatcher->specifiedInlRatio = 1.0;
//					mymatcher->GTfilterExtractor = descriptorExtractorGT;
//					mymatcher->detectFeatures();
//					mymatcher->checkForGT();
//					nrGTmatchesDataset += (int)mymatcher->positivesGT;
//					GTths.push_back(mymatcher->usedMatchTH);
//				}
//			}
//			std::ofstream evalsToFile(outpath + "\\" + outfilename);
//			evalsToFile << "GTinliers " << nrGTmatchesDataset << endl;
//			evalsToFile.close();
//
//			std::ofstream GTthsToFile(outpath + "\\GT-thresholds_" + outfilename);
//			GTthsToFile << "thresholds" << endl;
//			for(int i = 0; i < (int)GTths.size(); i++)
//			{
//				GTthsToFile << GTths[i] << endl;
//			}
//			GTthsToFile.close();
//		}
//
//		std::ifstream evalsToFilei(outpath + "\\" + outfilename);
//		if(!evalsToFilei.good() && !evalsToFilei.is_open())
//		{
//			cout << "Error opening file " << outfilename << endl;
//			cout << "Exiting." << endl;
//			evalsToFilei.close();
//			exit(0);
//		}
//		int starttestimgnr = 0;
//		int missingSamples = 0;
//		vector<vector<double>> StaticSamplesizes; //Elements of 2nd vector: p, e, n
//		double maxmaxSampleSize, sampleRatio, minminSampleSize;
//
//		readParametersTested(starttestimgnr, nrTotalFails, nrSamplesTillNow, filenamesl, &evalsToFilei, outpath, outfilename, missingSamples,
//						  StaticSamplesizes, maxmaxSampleSize, sampleRatio, minminSampleSize, nrGTmatchesDataset, flowDispH, distances,
//						  truePosArr, falsePosArr, falseNegArr, notMatchable, errvecs, perfectMatches, errvecsGT, distancesGT, 
//						  validityValGT, distancesEstModel, annotationData);
//
//		evalsToFilei.close();
//		globEvalsToFile.open(outpath + "\\" + outfilename, ios::app);
//		if(!globEvalsToFile.good() && !globEvalsToFile.is_open())
//		{
//			cout << "Error opening file " << outfilename << endl;
//			cout << "Exiting." << endl;
//			globEvalsToFile.close();
//			exit(0);
//		}
//
//		if(fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//		{
//			int starttestimgnr1, starttestimgnr2 = 0;
//			if(starttestimgnr < (int)fnames.size())
//			{
//				starttestimgnr1 = starttestimgnr;
//			}
//			else
//			{
//				starttestimgnr1 = (int)fnames.size();
//				starttestimgnr2 = starttestimgnr - (int)fnames.size();
//			}
//
//			//Take the stored homographys and perform evaluation
//			nrimgs = binomialCoeff((int)filenamesl.size(), 2);
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//			for(int idx1 = starttestimgnr1; idx1 < (int)fnames.size(); idx1++) 
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//				//mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//				nrremainingimgs = nrimgs - idx1 - 1;
//				fileNamesLR = filenamesl[0] + "_" + filenamesl[idx1 + 1];
//				if(testImagePairs(mymatcher, descriptorExtractorGT, sampleRatio, missingSamples, &globEvalsToFile, nrSamplesTillNow, nrTotalFails,
//								  minminSampleSize, StaticSamplesizes, nrGTmatchesDataset, distances, nrremainingimgs,
//								  truePosArr, falsePosArr, falseNegArr, notMatchable, errvecs, perfectMatches,
//							  errvecsGT, distancesGT, validityValGT, distancesEstModel, annotationData, fileNamesLR, threshhTh, idx1) != 0) return -1;
//			}
//			//Generate new homographys to evaluate all other possible configurations of the images to each other
//			nrremainingimgs = nrimgs -= (int)fnames.size();
//			for(int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//			{
//				for(int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//				{
//					nrremainingimgs--;
//					if(starttestimgnr2 > 0)
//					{
//						starttestimgnr2--;
//						continue;
//					}
//					//H = (Hs[idx2].inv() * Hs[idx1]).inv();
//					cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//					src[0] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx2 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//					//mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//					mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//
//					fileNamesLR = filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1];
//					if(testImagePairs(mymatcher, descriptorExtractorGT, sampleRatio, missingSamples, &globEvalsToFile, nrSamplesTillNow, nrTotalFails,
//									  minminSampleSize, StaticSamplesizes, nrGTmatchesDataset, distances, nrremainingimgs,
//									  truePosArr, falsePosArr, falseNegArr, notMatchable, errvecs, perfectMatches,
//									  errvecsGT, distancesGT, validityValGT, distancesEstModel, annotationData, fileNamesLR, threshhTh, nrimgs + (int)fnames.size() - nrremainingimgs - 1) != 0) return -1;
//				}
//			}
//		}
//		else
//		{
//			//Take the stored homographys and perform evaluation
//			nrimgs = (int)fnames.size();
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//			for(int idx1 = starttestimgnr; idx1 < nrimgs; idx1++) 
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//
//				//mymatcher.reset(new GMbSOF_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//				nrremainingimgs = nrimgs - idx1 - 1;
//				fileNamesLR = filenamesl[0] + "_" + filenamesl[idx1 + 1];
//				if(testImagePairs(mymatcher, descriptorExtractorGT, sampleRatio, missingSamples, &globEvalsToFile, nrSamplesTillNow, nrTotalFails,
//								  minminSampleSize, StaticSamplesizes, nrGTmatchesDataset, distances, nrremainingimgs,
//								  truePosArr, falsePosArr, falseNegArr, notMatchable, errvecs, perfectMatches,
//								  errvecsGT, distancesGT, validityValGT, distancesEstModel, annotationData, fileNamesLR, threshhTh, idx1) != 0) return -1;
//			}
//		}
//		globOutfileName = outfilename;
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//#if (CORRECT_MATCHING_RESULT == 0) || (CORRECT_MATCHING_RESULT == 3)
//	{
//		std::ofstream evalsToFile(outpath + "\\GT-errors_" + globOutfileName);
//		evalsToFile << "# Distances from the keypoint positions of the ground truth matches to the true matching position" << endl;
//		evalsToFile << "dists" << endl;
//		for(int i = 0; i < (int)distances.size(); i++)
//		{
//			evalsToFile << distances[i] << endl;
//		}
//		evalsToFile.close();
//	}
//	{
//		std::ofstream evalsToFile(outpath + "\\GT-error-vectors_" + globOutfileName);
//		evalsToFile << "# Vectors from the keypoint positions of the ground truth matches to the true matching position" << endl;
//		evalsToFile << "Xdists Ydists validityLevel" << endl;
//		for(int i = 0; i < (int)errvecs.size(); i++)
//		{
//			evalsToFile << errvecs[i].x << " " << errvecs[i].y << " " << validityValGT[i] << endl;
//		}
//		evalsToFile.close();
//	}
//	{
//		std::ofstream evalsToFile(outpath + "\\OrigSpatialGT-error-vectors_" + globOutfileName);
//		evalsToFile << "# Vectors from the spatial ground truth (e.g. KITTI) to the true matching position" << endl;
//		evalsToFile << "Xdists Ydists validityLevel" << endl;
//		for(int i = 0; i < (int)errvecsGT.size(); i++)
//		{
//			evalsToFile << errvecsGT[i].x << " " << errvecsGT[i].y << " " << validityValGT[i] << endl;
//		}
//		evalsToFile.close();
//	}
//	{
//		std::ofstream evalsToFile(outpath + "\\OrigSpatialGT-errors_" + globOutfileName);
//		evalsToFile << "# Distances from the spatial ground truth (e.g. KITTI) to the true matching position" << endl;
//		evalsToFile << "dists validityLevel" << endl;
//		for(int i = 0; i < (int)distancesGT.size(); i++)
//		{
//			evalsToFile << distancesGT[i] << " " << validityValGT[i] << endl;
//		}
//		evalsToFile.close();
//	}
//	{
//		std::ofstream evalsToFile(outpath + "\\DistsEstimGeomModel_" + globOutfileName);
//		evalsToFile << "# Distances from the estimated geometric models (from true matching positions of every image pair) to the true matching position" << endl;
//		evalsToFile << "dists" << endl;
//		for(int i = 0; i < (int)distancesEstModel.size(); i++)
//		{
//			evalsToFile << distancesEstModel[i] << endl;
//		}
//		evalsToFile.close();
//	}
//
//	{
//		int answere;
//		answere = MessageBox(NULL, "Is this a rectified disparity data set with fixed extrinsics over all tested image pairs?\n If true, it is possible to calculate a global essential matrix and to calculate the error to the estimated geometry. Do you want to calculate the geometry and error values?", "Global essential matrix", MB_YESNO | MB_DEFBUTTON2);
//		if(answere == IDYES)
//		{
//			vector<double> distancesGlobEstModel;
//			vector<cv::Point2f> leftPs, rightPs;
//			for(int i = 0; i < (int)perfectMatches.size(); i++)
//			{
//				leftPs.push_back(perfectMatches[i].first);
//				rightPs.push_back(perfectMatches[i].second);
//			}
//			cv::Mat HE = cv::findFundamentalMat(leftPs, rightPs, CV_FM_LMEDS);
//			if(!HE.empty())
//			{
//				HE.convertTo(HE, CV_64FC1);
//				cv::Mat Et = HE.t();
//				for (int i = 0; i < (int)perfectMatches.size(); i++)
//				{
//					cv::Mat x1 = (cv::Mat_<double>(3, 1) << leftPs[i].x, leftPs[i].y, 1.0); 
//					cv::Mat x2 = (cv::Mat_<double>(3, 1) << rightPs[i].x, rightPs[i].y, 1.0); 
//					double x2tEx1 = x2.dot(HE * x1); 
//					cv::Mat Ex1 = HE * x1; 
//					cv::Mat Etx2 = Et * x2; 
//					double a = Ex1.at<double>(0) * Ex1.at<double>(0); 
//					double b = Ex1.at<double>(1) * Ex1.at<double>(1); 
//					double c = Etx2.at<double>(0) * Etx2.at<double>(0); 
//					double d = Etx2.at<double>(1) * Etx2.at<double>(1); 
//
//					distancesGlobEstModel.push_back(x2tEx1 * x2tEx1 / (a + b + c + d));
//				}
//				std::ofstream evalsToFile(outpath + "\\GlobGeometryErrors_" + globOutfileName);
//				evalsToFile << "# Distances from the estimated geometric model (from all true matching positions over every image pair) to the true matching position" << endl;
//				evalsToFile << "# Essential matrix:";
//				for(int i = 0; i < 3; i++)
//				{
//					for(int j = 0; j < 3; j++)
//					{
//						evalsToFile << " " << HE.at<double>(i,j);
//					}
//				}
//				evalsToFile << endl;
//				evalsToFile << "dists" << endl;
//				for(int i = 0; i < (int)distancesGlobEstModel.size(); i++)
//				{
//					evalsToFile << distancesGlobEstModel[i] << endl;
//				}
//				evalsToFile.close();
//			}
//		}
//	}
//#endif
//
//	double preci[4], recall[4], sasi, e, p = (double)nrTotalFails / (double)nrSamplesTillNow;
//	getMinSampleSize(nrGTmatchesDataset, p, e, sasi);
//	globEvalsToFile << "TotalWrongMatches " << nrTotalFails << " TotalSamples " << nrSamplesTillNow << " FailureRate " << p << " ErrorRange " << e << endl;
//
//	string usedDistTH[5] = {"lowOrEquTH ","higherTH ","higher4Pix ","higher8Pix ","notMatchable "};
//	globEvalsToFile << "TruePositives:" << endl;
//	for(int i = 0; i < 5; i++)
//	{
//		globEvalsToFile << usedDistTH[i] << truePosArr[i] << " ";
//	}
//	globEvalsToFile << endl;
//	globEvalsToFile << "FalsePositives:" << endl;
//	for(int i = 0; i < 5; i++)
//	{
//		globEvalsToFile << usedDistTH[i] << falsePosArr[i] << " ";
//	}
//	globEvalsToFile << endl;
//	globEvalsToFile << "FalseNegatives:" << endl;
//	for(int i = 0; i < 5; i++)
//	{
//		globEvalsToFile << usedDistTH[i] << falseNegArr[i] << " ";
//	}
//	globEvalsToFile << endl;
//	globEvalsToFile << "Precision:" << endl;
//	for(int i = 0; i < 5; i++)
//	{
//		//Precision or positive predictive value ppv=truePos/(truePos+falsePos)
//		preci[i] = (truePosArr[i] == 0) ? 0:(double)truePosArr[i] / (double)(truePosArr[i] + falsePosArr[i]);
//		globEvalsToFile << usedDistTH[i] << preci[i] << " ";
//	}
//	globEvalsToFile << endl;
//	globEvalsToFile << "Recall:" << endl;
//	for(int i = 0; i < 5; i++)
//	{
//		//Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)
//		recall[i] = (truePosArr[i] == 0) ? 0:(double)truePosArr[i] / (double)(truePosArr[i] + falseNegArr[i]);
//		globEvalsToFile << usedDistTH[i] << recall[i] << " ";
//	}
//	globEvalsToFile << endl;
//	globEvalsToFile << "NotMatchable " << notMatchable << endl;
//	globEvalsToFile.close();
//	cout << "Total wrong matches " << nrTotalFails << endl;
//	cout << "Total samples " << nrSamplesTillNow << endl;
//	cout << "Failure rate " << p << endl;
//	cout << "Error range " << e << endl;
//	writeStats(outpath, outfilename, distances, distancesGT, distancesEstModel, annotationData);
//	system("pause");
//	
//	return 0;
//}
//
////Check if evalation was done before
//void checkPrevEval(std::string outpath, std::string outfilename, std::vector<string> filenamesl, bool & datafileexists, int & nrGTmatchesDataset)
//{
//	ifstream infstream(outpath + "\\" + outfilename);
//	if(infstream.good())//Is true if file exists
//	{
//		string line, word;
//		int totalnr;
//		std::istringstream is;
//		std::getline(infstream, line);
//		if(line.empty())
//		{
//			infstream.close();
//			cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//			exit(0);
//		}
//		else
//		{
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 9, "GTinliers") != 0) 
//			{
//				infstream.close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			if(is >> totalnr)
//			{
//				if(totalnr > 100 * (int)filenamesl.size())
//				{
//					datafileexists = true;
//					nrGTmatchesDataset = totalnr;
//				}
//				else
//				{
//					infstream.close();
//					cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//					exit(0);
//				}
//			}
//		}
//		infstream.close();
//	}
//	else
//	{
//		infstream.close();
//	}
//}
//
//void readParametersTested(int & starttestimgnr, int & nrTotalFails, int & nrSamplesTillNow, std::vector<std::string> filenamesl, std::ifstream *evalsToFile, std::string path, 
//						  std::string outfilename, int & missingSamples, std::vector<std::vector<double>> & StaticSamplesizes, double & maxmaxSampleSize, double & sampleRatio, 
//						  double & minminSampleSize, int & nrGTmatchesDataset, int flowDispH, std::vector<double> & distances, int *truePosArr, int *falsePosArr, int *falseNegArr, 
//						  int & notMatchable, std::vector<cv::Point2f> & errvecs, std::vector<std::pair<cv::Point2f,cv::Point2f>> & perfectMatches, 
//						  std::vector<cv::Point2f> & errvecsGT, std::vector<double> & distancesGT, std::vector<int> & validityValGT, std::vector<double> & distancesEstModel,
//						  annotImgPars & annotationData)
//{
//#if CORRECT_MATCHING_RESULT == 1
//	return;
//#elif CORRECT_MATCHING_RESULT == 2
//	starttestimgnr = (int)filenamesl.size();
//#endif
//	std::pair<int, int> lmissingS;
//	const int nrOfRows = 19;
//	lmissingS.first = -1;
//	int nrentries = ((flowDispH == 2) && (filenamesl.size() < 30)) ? (nrOfRows * binomialCoeff((int)filenamesl.size(), 2) + 1):(nrOfRows * (int)filenamesl.size() + 1);
//	int k;
//	for(k = 0; k < nrentries; k++)
//	{
//		string line;
//		getline(*evalsToFile, line);
//		if(line.empty())//If the line is empty, the end of the file is reached
//		{
//			float lnr = ((float)k - 1.0f) / (float)nrOfRows;
//			if((lnr - floor(lnr + 0.1f)) > 0.1f)//Check if there are nrOfRows entries per image pair
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			starttestimgnr = (int)floor(lnr + 0.5f);
//			if((lmissingS.first >= 0) && ((k - lmissingS.first) == 2))
//				missingSamples = lmissingS.second;
//			break;
//		}
//		else if((k > 0) && ((k % nrOfRows) == 1)) //First entry (of nrOfRows) for every image pair that holds the ID of the image pair
//		{
//			std::istringstream is;
//			int singleFails = 0;
//			string word;
//			is.str(line);
//			is >> word;
//			annotationData.id.push_back(word);
//		}
//		else if((k > 0) && ((k % nrOfRows) == 2)) //Second entry (of nrOfRows) for every image pair that holds the total number of wrong GT matches
//		{
//			std::istringstream is;
//			int singleFails = 0;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 9, "NrWrongGT") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			is >> singleFails;
//			nrTotalFails += singleFails;
//			annotationData.nrTotalFails.push_back(singleFails);
//		}
//		else if((k > 0) && ((k % nrOfRows) == 3)) //Third entry (of nrOfRows) for every image pair that holds the total number of wrong GT matches
//		{
//			std::istringstream is;
//			int singleSampleSi = 0;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 10, "SampleSize") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			is >> singleSampleSi;
//			nrSamplesTillNow += singleSampleSi;
//			annotationData.selectedSamples.push_back(singleSampleSi);
//			word.clear();
//			is >> word;
//			if(!word.empty() && !word.compare("missedSamples"))
//			{
//				is >> lmissingS.second;
//				lmissingS.first = k;
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 4)) //4rd entry (of nrOfRows) that holds the coordinates of false GT matches
//		{
//			std::istringstream is;
//			float fcoord;
//			int j = 0;
//			cv::Point2f ml, mr;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 6, "Coords") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			annotationData.falseGT.push_back(std::vector<std::pair<cv::Point2f,cv::Point2f>>());
//			while(is >> fcoord)
//			{
//				switch(j)
//				{
//				case 0:
//					ml.x = fcoord;
//					break;
//				case 1:
//					ml.y = fcoord;
//					break;
//				case 2:
//					mr.x = fcoord;
//					break;
//				case 3:
//					mr.y = fcoord;
//					break;
//				}
//				j++;
//				if(j > 3)
//				{
//					j = 0;
//					annotationData.falseGT.back().push_back(std::make_pair(ml, mr));
//				}
//			}
//			if(j != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 5)) //5th entry (of nrOfRows) that holds the distance histogram from GT matches to annoted positions
//		{
//			std::istringstream is;
//			bool alternate = true;
//			double matchdistAndNr;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 17, "DistanceHistogram") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			annotationData.distanceHisto.push_back(std::vector<std::pair<double,int>>());
//			while(is >> matchdistAndNr)
//			{
//				if(alternate)
//					annotationData.distanceHisto.back().push_back(std::make_pair(matchdistAndNr, 0));
//				else
//					annotationData.distanceHisto.back().back().second = (int)floor(matchdistAndNr + 0.5);
//				alternate = !alternate;
//			}
//			if(!alternate)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 6)) //6th entry (of nrOfRows) for every image pair that holds the measured distances to the real matching positions
//		{
//			std::istringstream is;
//			double matchdist;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 9, "Distances") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			annotationData.distances.push_back(std::vector<double>());
//			while(is >> matchdist)
//			{
//				distances.push_back(matchdist);
//				annotationData.distances.back().push_back(matchdist);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 7)) //7th entry (of nrOfRows) for every image pair that holds the number of true positives of the sampled and additionally matched (linear matcher) GT matches 
//		{								   //for GT matches with measured distances lower or equal and above the GT match threshold in addition to distances higher 4 and 8 pixels
//			std::istringstream is;
//			int val, j = 0;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 36, "NrTruePos[leTH-hTH-h4-h8_pix-noMtch]") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			while(is >> val)
//			{
//				if(j > 4)
//				{
//					evalsToFile->close();
//					cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//					exit(0);
//				}
//				truePosArr[j] += val;
//				annotationData.truePosArr[j].push_back(val);
//				j++;
//			}
//			if(j != 5)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 8)) //8th entry (of nrOfRows) for every image pair that holds the number of false positives of the sampled and additionally matched (linear matcher) GT matches 
//		{								   //for GT matches with measured distances lower or equal and above the GT match threshold in addition to distances higher 4 and 8 pixels
//			std::istringstream is;
//			int val, j = 0;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 37, "NrFalsePos[leTH-hTH-h4-h8_pix-noMtch]") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			while(is >> val)
//			{
//				if(j > 4)
//				{
//					evalsToFile->close();
//					cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//					exit(0);
//				}
//				falsePosArr[j] += val;
//				annotationData.falsePosArr[j].push_back(val);
//				j++;
//			}
//			if(j != 5)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 9)) //9th entry (of nrOfRows) for every image pair that holds the number of false negatives of the sampled and additionally matched (linear matcher) GT matches 
//		{								   //for GT matches with measured distances lower or equal and above the GT match threshold in addition to distances higher 4 and 8 pixels
//			std::istringstream is;
//			int val, j = 0;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 37, "NrFalseNeg[leTH-hTH-h4-h8_pix-noMtch]") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			while(is >> val)
//			{
//				if(j > 4)
//				{
//					evalsToFile->close();
//					cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//					exit(0);
//				}
//				falseNegArr[j] += val;
//				annotationData.falseNegArr[j].push_back(val);
//				j++;
//			}
//			if(j != 5)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 10)) //10th entry (of nrOfRows) for every image pair that holds the number of not matchable GT matches
//		{
//			std::istringstream is;
//			int val = 0;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 12, "NotMatchable") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			is >> val;
//			notMatchable += val;
//			annotationData.notMatchable.push_back(val);
//		}
//		else if((k > 0) && ((k % nrOfRows) == 11)) //11th entry (of nrOfRows) for every image pair that holds the manually selected correspondences in the order: left x, left y, right x, right y
//		{
//			std::istringstream is;
//			float val = 0;
//			int j = 0;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 14, "PerfectMatches") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			annotationData.perfectMatches.push_back(std::vector<std::pair<cv::Point2f,cv::Point2f>>());
//			while(is >> val)
//			{
//				switch(j)
//				{
//				case 0:
//					perfectMatches.push_back(std::make_pair<cv::Point2f,cv::Point2f>(cv::Point2f(val,0), cv::Point2f(0,0)));
//					annotationData.perfectMatches.back().push_back(std::make_pair<cv::Point2f,cv::Point2f>(cv::Point2f(val,0), cv::Point2f(0,0)));
//					break;
//				case 1:
//					perfectMatches.back().first.y = val;
//					annotationData.perfectMatches.back().back().first.y = val;
//					break;
//				case 2:
//					perfectMatches.back().second.x = val;
//					annotationData.perfectMatches.back().back().second.x = val;
//					break;
//				case 3:
//					perfectMatches.back().second.y = val;
//					annotationData.perfectMatches.back().back().second.y = val;
//					break;
//				}
//				if(j >= 3)
//					j = 0;
//				else
//					j++;
//			}
//			if(j != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 12)) //12th entry (of nrOfRows) for every image pair that holds the error vectors for each match (Vector from keypoint position to selected real matching position) in the form: x, y
//		{
//			std::istringstream is;
//			float val = 0;
//			bool j = true;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 12, "ErrorVectors") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			annotationData.errvecs.push_back(std::vector<cv::Point2f>());
//			while(is >> val)
//			{
//				if(j)
//				{
//					errvecs.push_back(cv::Point2f(val,0));
//					annotationData.errvecs.back().push_back(cv::Point2f(val,0));
//				}
//				else
//				{
//					errvecs.back().y = val;
//					annotationData.errvecs.back().back().y = val;
//				}
//				j = !j;
//			}
//			if(!j)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 13)) //13th entry (of nrOfRows) for every image pair that holds the estimated model (homography or fundamental matrix) from the annotated correspondences
//		{
//			std::istringstream is;
//			double val;
//			int u = 0, v = 0;
//			cv::Mat HEr(3, 3, CV_64FC1);
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 6, "Matrix") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			while(is >> val)
//			{
//				if(u > 2)
//				{
//					evalsToFile->close();
//					cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//					exit(0);
//				}
//				HEr.at<double>(u, v) = val;
//				if(v >= 2)
//				{
//					v = 0;
//					u++;
//				}
//				else
//					v++;
//			}
//			if((u == 0) && (v == 0))
//			{
//				annotationData.HE.push_back(cv::Mat());
//			}
//			else if(u < 3)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			else
//			{
//				annotationData.HE.push_back(HEr);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 14)) //14th entry (of nrOfRows) for every image pair that holds the error vectors for to the spatial ground truth (e.g. KITTI) and each match (Vector from spatial GT position to selected real matching position) in the form: x, y
//		{
//			std::istringstream is;
//			float val = 0;
//			bool j = true;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 21, "ErrorVectorsSpatialGT") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			annotationData.errvecsGT.push_back(std::vector<cv::Point2f>());
//			while(is >> val)
//			{
//				if(j)
//				{
//					errvecsGT.push_back(cv::Point2f(val,0));
//					annotationData.errvecsGT.back().push_back(cv::Point2f(val,0));
//				}
//				else
//				{
//					errvecsGT.back().y = val;
//					annotationData.errvecsGT.back().back().y = val;
//				}
//				j = !j;
//			}
//			if(!j)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 15)) //15th entry (of nrOfRows) for every image pair that holds the measured distances from the spatial GT positions (e.g. KITTI) to the real matching positions
//		{
//			std::istringstream is;
//			double matchdist;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 18, "DistancesSpatialGT") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			annotationData.distancesGT.push_back(std::vector<double>());
//			while(is >> matchdist)
//			{
//				distancesGT.push_back(matchdist);
//				annotationData.distancesGT.back().push_back(matchdist);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 16)) //16th entry (of nrOfRows) for every image pair that holds the validity information about the spatial GT (especially KITTI, for which 0 means invalid, 1 valid and 2 means filled with Oliver Zendels median filter)
//		{
//			std::istringstream is;
//			int GTvalidity;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 22, "ValidityLevelSpatialGT") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			annotationData.validityValGT.push_back(std::vector<int>());
//			while(is >> GTvalidity)
//			{
//				validityValGT.push_back(GTvalidity);
//				annotationData.validityValGT.back().push_back(GTvalidity);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 17)) //17th entry (of nrOfRows) for every image pair that holds the measured distances from an estimated geometric model (based on perfectMatches) to the real matching positions
//		{
//			std::istringstream is;
//			double matchdist;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 19, "DistancesEstimModel") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			annotationData.distancesEstModel.push_back(std::vector<double>());
//			while(is >> matchdist)
//			{
//				distancesEstModel.push_back(matchdist);
//				annotationData.distancesEstModel.back().push_back(matchdist);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 18)) //18th entry (of nrOfRows) for every image pair that holds validity level of the original GT dataset for every false match of the GT matches (for filled data points, a 2 is used and for original GT a 1)
//		{
//			std::istringstream is;
//			int vallevel;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 20, "ValidityLevelFalseGT") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			annotationData.validityValFalseGT.push_back(std::vector<int>());
//			while(is >> vallevel)
//			{
//				annotationData.validityValFalseGT.back().push_back(vallevel);
//			}
//		}
//		else if((k > 0) && ((k % nrOfRows) == 0)) //19th entry (of nrOfRows) for every image pair that holds a vector with 'M's for manual and 'A's for automatic annotated matches
//		{
//			std::istringstream is;
//			char ama;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 22, "ManualOrAutomaticAnnot") != 0)
//			{
//				evalsToFile->close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			annotationData.autoManualAnnot.push_back(std::vector<char>());
//			while(is >> ama)
//			{
//				annotationData.autoManualAnnot.back().push_back(ama);
//			}
//		}
//	}
//#if CORRECT_MATCHING_RESULT == 0
//	if(k == nrentries)
//	{
//		countAutoManualAnnotations(path, outfilename, annotationData);
//		writeStatsV2(path, outfilename, annotationData);
//		if(writeErrIndividualImg(path, outfilename, annotationData) < 0)
//		{
//			cout << "Error lists for individual images not generated." << endl;
//		}
//		else
//		{
//			cout << "Error lists for individual images written to output." << endl;
//		}
//		evalsToFile->close();
//		if(writeStats(path, outfilename, distances, distancesGT, distancesEstModel, annotationData) < 0)
//			MessageBox(NULL, "This dataset was already tested! Exiting.", "Dataset already tested", MB_OK | MB_ICONWARNING);
//		else
//			cout << "Statistics written to output." << endl;
//		exit(0);
//	}
//#elif CORRECT_MATCHING_RESULT == 3
//	starttestimgnr = 0;
//	nrTotalFails = 0;
//	nrSamplesTillNow = 0;
//	missingSamples = 0;
//	distances.clear(); 
//	for(int i = 0; i < 5; i++)
//	{
//		truePosArr[i] = 0;
//		falsePosArr[i] = 0;
//		falseNegArr[i] = 0;
//	}
//	notMatchable = 0;
//	errvecs.clear();
//	perfectMatches.clear();
//	errvecsGT.clear();
//	distancesGT.clear();
//	validityValGT.clear();
//	distancesEstModel.clear();
//#endif
//
//	//vector<vector<double>> StaticSamplesizes; //Elements of 2nd vector: p, e, n
//	double p = 0.5;//, maxmaxSampleSize, sampleRatio, minminSampleSize;
//	while(p >= 0.0001)
//	{
//		vector<double> oneSample(3);
//		oneSample[0] = p;
//		getMinSampleSize(nrGTmatchesDataset, oneSample[0], oneSample[1], oneSample[2]);
//		StaticSamplesizes.push_back(oneSample);
//		if(StaticSamplesizes.size() == 1)
//		{
//			maxmaxSampleSize = oneSample[2];
//			minminSampleSize =oneSample[2];
//		}
//		else if(maxmaxSampleSize < oneSample[2])
//		{
//			maxmaxSampleSize = oneSample[2];
//		}
//		else if(minminSampleSize > oneSample[2])
//		{
//			minminSampleSize = oneSample[2];
//		}
//		if(p > 0.1 + DBL_EPSILON)
//			p -= 0.1;
//		else if(p > 0.05 + DBL_EPSILON)
//			p -= 0.05;
//		else if(p > 0.01 + DBL_EPSILON)
//			p -= 0.01;
//		else if(p > 0.001 + DBL_EPSILON)
//			p -= 0.001;
//		else if(p > 0.0001 + DBL_EPSILON)
//			p -= 0.0001;
//		else
//			break;
//	}
//	sampleRatio = maxmaxSampleSize/(double)nrGTmatchesDataset;
//	cout << "First estimate of number of neccessary samples: " << maxmaxSampleSize << endl;
//}
//
//int testImagePairs(std::auto_ptr<baseMatcher> mymatcher, std::string descriptorExtractorGT, double & sampleRatio, int & missingSamples, std::ofstream *evalsToFile, int & nrSamplesTillNow, int & nrTotalFails,
//				   double & minminSampleSize, std::vector<std::vector<double>> StaticSamplesizes, int nrGTmatchesDataset, std::vector<double> & distances, int remainingImgs,
//				   int *truePosArr, int *falsePosArr, int *falseNegArr, int & notMatchable, std::vector<cv::Point2f> & errvecs, std::vector<std::pair<cv::Point2f,cv::Point2f>> & perfectMatches,
//				   std::vector<cv::Point2f> & errvecsGT, std::vector<double> & distancesGT, std::vector<int> & validityValGT, std::vector<double> & distancesEstModel, annotImgPars & annotationData,
//				   std::string fileNamesLR, double threshhTh, int imgNr, std::string path, std::string outfilename)
//{
//	int samples, selectedSamples;
//	std::vector<double> distances2;
//	mymatcher->specifiedInlRatio = 1.0;
//	int truePosArr1[5] = {0,0,0,0,0};
//	int falsePosArr1[5] = {0,0,0,0,0};
//	int falseNegArr1[5] = {0,0,0,0,0};
//	int notMatchable1 = 0;
//	std::vector<cv::Point2f> errvecs1;
//	std::vector<std::pair<cv::Point2f,cv::Point2f>> perfectMatches1;
//	cv::Mat HE;
//	std::vector<cv::Point2f> errvecsGT1;
//	std::vector<double> distancesGT1;
//	std::vector<int> validityValGT1;
//	std::vector<double> distancesEstModel1;
//	std::vector<int> validityValFalseGT;
//	std::vector<char> autoManualAnno;//'M' for manual annotation, 'A' for automatic annotation
//
//	if(mymatcher->specialGMbSOFtest)
//	{
//		cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//		exit(0);
//	}
//
//	std::vector<std::pair<cv::Point2f,cv::Point2f>> falseGT;
//	mymatcher->detectFeatures();
//	mymatcher->GTfilterExtractor = descriptorExtractorGT;
//	mymatcher->checkForGT();
//	reEstimateSampleSize(nrSamplesTillNow, minminSampleSize, nrTotalFails, StaticSamplesizes, nrGTmatchesDataset, sampleRatio);
//	samples = (int)ceil(mymatcher->positivesGT * sampleRatio) + missingSamples;
//	samples = samples < 10 ? 10:samples;
//	std::vector<std::pair<double,int>> distanceHisto;
//#if (CORRECT_MATCHING_RESULT == 1) || (CORRECT_MATCHING_RESULT == 2)
//	correctMatchingResult(imgNr, path, outfilename, mymatcher);
//	return 0;
//#elif CORRECT_MATCHING_RESULT == 3
//	samples = annotationData.selectedSamples[imgNr];
//	if(mymatcher->testGTmatches(samples, falseGT, selectedSamples, distanceHisto, distances2, remainingImgs, 
//	   notMatchable1, truePosArr1, falsePosArr1, falseNegArr1, errvecs1, perfectMatches1, HE, validityValFalseGT,
//	   errvecsGT1, distancesGT1, validityValGT1, distancesEstModel1, annotationData, autoManualAnno, threshhTh, imgNr,
//	   &nrGTmatchesDataset, &nrSamplesTillNow, &nrTotalFails) != 0)
//	{
//		evalsToFile->close();
//		return -1;
//	}
//#else
//	if(mymatcher->testGTmatches(samples, falseGT, selectedSamples, distanceHisto, distances2, remainingImgs, 
//	   notMatchable1, truePosArr1, falsePosArr1, falseNegArr1, errvecs1, perfectMatches1, HE, validityValFalseGT,
//	   errvecsGT1, distancesGT1, validityValGT1, distancesEstModel1, annotationData, autoManualAnno, threshhTh, imgNr,
//	   &nrGTmatchesDataset, &nrSamplesTillNow, &nrTotalFails) != 0)
//	{
//		evalsToFile->close();
//		return -1;
//	}
//#endif
//	nrSamplesTillNow += selectedSamples;
//	nrTotalFails += (int)falseGT.size();
//	missingSamples = samples - selectedSamples;
//
//	*evalsToFile << fileNamesLR << endl;
//	*evalsToFile << "NrWrongGT " << falseGT.size() << endl;
//	*evalsToFile << "SampleSize " << selectedSamples;
//	if(missingSamples > 0) *evalsToFile << " missedSamples " << missingSamples;
//	*evalsToFile << endl;
//	*evalsToFile << "Coords";
//	for(int i = 0; i < (int)falseGT.size(); i++)
//	{
//		*evalsToFile << " " << falseGT[i].first.x << " " << falseGT[i].first.y << " " << falseGT[i].second.x << " " << falseGT[i].second.y;
//	}
//	*evalsToFile << endl;
//	*evalsToFile << "DistanceHistogram";
//	for(int i = 0; i < (int)distanceHisto.size(); i++)
//	{
//		*evalsToFile << " " << distanceHisto[i].first << " " << distanceHisto[i].second;
//	}
//	*evalsToFile << endl;
//	*evalsToFile << "Distances";
//	for(int i = 0; i < (int)distances2.size(); i++)
//	{
//		*evalsToFile << " " << distances2[i];
//	}
//	*evalsToFile << endl;
//	distances.insert(distances.end(), distances2.begin(), distances2.end());
//	*evalsToFile << "NrTruePos[leTH-hTH-h4-h8_pix-noMtch] " << truePosArr1[0] << " " << truePosArr1[1] << " " << truePosArr1[2] << " " << truePosArr1[3] << " " << truePosArr1[4] << endl;
//	*evalsToFile << "NrFalsePos[leTH-hTH-h4-h8_pix-noMtch] " << falsePosArr1[0] << " " << falsePosArr1[1] << " " << falsePosArr1[2] << " " << falsePosArr1[3] << " " << falsePosArr1[4] << endl;
//	*evalsToFile << "NrFalseNeg[leTH-hTH-h4-h8_pix-noMtch] " << falseNegArr1[0] << " " << falseNegArr1[1] << " " << falseNegArr1[2] << " " << falseNegArr1[3] << " " << falseNegArr1[4] << endl;
//	for(int i = 0; i < 5; i++)
//	{
//		truePosArr[i] += truePosArr1[i];
//		falsePosArr[i] += falsePosArr1[i];
//		falseNegArr[i] += falseNegArr1[i];
//	}
//	*evalsToFile << "NotMatchable " << notMatchable1 << endl;;
//	notMatchable += notMatchable1;
//	*evalsToFile << "PerfectMatches";
//	for(int i = 0; i < (int)perfectMatches1.size(); i++)
//	{
//		*evalsToFile << " " << perfectMatches1[i].first.x << " " << perfectMatches1[i].first.y << " " << perfectMatches1[i].second.x << " " << perfectMatches1[i].second.y;
//	}
//	*evalsToFile << endl;
//	perfectMatches.insert(perfectMatches.end(), perfectMatches1.begin(), perfectMatches1.end());
//	*evalsToFile << "ErrorVectors";
//	for(int i = 0; i < (int)errvecs1.size(); i++)
//	{
//		*evalsToFile << " " << errvecs1[i].x << " " << errvecs1[i].y;
//	}
//	*evalsToFile << endl;
//	errvecs.insert(errvecs.end(), errvecs1.begin(), errvecs1.end());
//	*evalsToFile << "Matrix";
//	if(!HE.empty())
//	{
//		for(int i = 0; i < 3; i++)
//		{
//			for(int j = 0; j < 3; j++)
//			{
//				if(HE.type() == CV_64FC1)
//					*evalsToFile << " " << HE.at<double>(i,j);
//				else if(HE.type() == CV_32FC1)
//					*evalsToFile << " " << HE.at<float>(i,j);
//				else
//					*evalsToFile << " 0";
//			}
//		}
//	}
//	*evalsToFile << endl;
//	*evalsToFile << "ErrorVectorsSpatialGT";
//	for(int i = 0; i < (int)errvecsGT1.size(); i++)
//	{
//		*evalsToFile << " " << errvecsGT1[i].x << " " << errvecsGT1[i].y;
//	}
//	*evalsToFile << endl;
//	errvecsGT.insert(errvecsGT.end(), errvecsGT1.begin(), errvecsGT1.end());
//	*evalsToFile << "DistancesSpatialGT";
//	for(int i = 0; i < (int)distancesGT1.size(); i++)
//	{
//		*evalsToFile << " " << distancesGT1[i];
//	}
//	*evalsToFile << endl;
//	distancesGT.insert(distancesGT.end(), distancesGT1.begin(), distancesGT1.end());
//	*evalsToFile << "ValidityLevelSpatialGT";
//	for(int i = 0; i < (int)validityValGT1.size(); i++)
//	{
//		*evalsToFile << " " << validityValGT1[i];
//	}
//	*evalsToFile << endl;
//	validityValGT.insert(validityValGT.end(), validityValGT1.begin(), validityValGT1.end());
//	*evalsToFile << "DistancesEstimModel";
//	for(int i = 0; i < (int)distancesEstModel1.size(); i++)
//	{
//		*evalsToFile << " " << distancesEstModel1[i];
//	}
//	*evalsToFile << endl;
//	distancesEstModel.insert(distancesEstModel.end(), distancesEstModel1.begin(), distancesEstModel1.end());
//	*evalsToFile << "ValidityLevelFalseGT";
//	for(int i = 0; i < (int)validityValFalseGT.size(); i++)
//	{
//		*evalsToFile << " " << validityValFalseGT[i];
//	}
//	*evalsToFile << endl;
//	*evalsToFile << "ManualOrAutomaticAnnot";
//	for(int i = 0; i < (int)autoManualAnno.size(); i++)
//	{
//		*evalsToFile << " " << autoManualAnno[i];
//	}
//	*evalsToFile << endl;
//
//	reEstimateSampleSize(nrSamplesTillNow, minminSampleSize, nrTotalFails, StaticSamplesizes, nrGTmatchesDataset, sampleRatio);
//	return 0;
//}
//
//// Returns value of Binomial Coefficient C(n, k)
//int binomialCoeff(int n, int k)
//{
//  // Base Cases
//  if (k==0 || k==n)
//    return 1;
//
//  // Recur
//  return  binomialCoeff(n-1, k-1) + binomialCoeff(n-1, k);
//}
//
//void reEstimateSampleSize(int nrSamplesTillNow, double & minminSampleSize, int nrTotalFails, std::vector<std::vector<double>> StaticSamplesizes, int nrGTmatchesDataset, double & sampleRatio)
//{
//	if(((nrSamplesTillNow >= (int)(0.5 * minminSampleSize)) && (nrTotalFails > 0)) || (nrSamplesTillNow >= (int)minminSampleSize))
//	{
//		if(nrTotalFails == 0)
//		{
//			double secminSaSi = DBL_MAX;
//			for(int i = 0; i < StaticSamplesizes.size(); i++)
//			{
//				if((StaticSamplesizes[i][2] > minminSampleSize) && (StaticSamplesizes[i][2] < secminSaSi))
//					secminSaSi = StaticSamplesizes[i][2];
//			}
//			minminSampleSize = secminSaSi;
//		}
//		else
//		{
//			double e, p = (double)nrTotalFails / (double)nrSamplesTillNow;
//			double sasi[3];
//			getMinSampleSize(nrGTmatchesDataset, p, e, sasi[1]);
//			if((double)nrSamplesTillNow / sasi[1] > 0.85)
//			{
//				sampleRatio = sasi[1] / (double)nrGTmatchesDataset;
//				minminSampleSize = sasi[1];
//				cout << "Updated estimation of number of neccessary samples: " << sasi[1] << endl;
//			}
//			else
//			{
//				double maxsasi;
//				getMinSampleSize(nrGTmatchesDataset, 2.0 * p, e, sasi[0]);
//				getMinSampleSize(nrGTmatchesDataset, 0.5 * p, e, sasi[2]);
//				maxsasi = max(sasi[0], max(sasi[1], sasi[2]));
//				sampleRatio = maxsasi / (double)nrGTmatchesDataset;
//				minminSampleSize = min(sasi[0], min(sasi[1], sasi[2]));
//				cout << "Updated estimation of number of neccessary samples: " << maxsasi << endl;
//			}
//		}
//	}
//}
//
//int writeStats(std::string path, std::string filename, std::vector<double> distances, std::vector<double> distancesGT, std::vector<double> distancesEstModel, annotImgPars annotationData)
//{
//	if(file_exists(path + "\\stats-DistsBoxPl_" + filename))
//		return -1;
//
//	std::vector<double> distancesMatchesAuto, distancesMatchesManual, distancesDatasetGTAuto, distancesDatasetGTManual, distancesEstModelAuto, distancesEstModelManual;
//
//	for(int i = 0; i < (int)annotationData.autoManualAnnot.size(); i++)
//	{
//		for (int j = 0; j < (int)annotationData.autoManualAnnot[i].size(); j++)
//		{
//			if(annotationData.autoManualAnnot[i][j] == 'A')
//			{
//				distancesMatchesAuto.push_back(annotationData.distances[i][j]);
//				distancesDatasetGTAuto.push_back(annotationData.distancesGT[i][j]);
//				distancesEstModelAuto.push_back(annotationData.distancesEstModel[i][j]);
//			}
//			else
//			{
//				distancesMatchesManual.push_back(annotationData.distances[i][j]);
//				distancesDatasetGTManual.push_back(annotationData.distancesGT[i][j]);
//				distancesEstModelManual.push_back(annotationData.distancesEstModel[i][j]);
//			}
//		}
//	}
//
//	qualityParm1 distMatchStat, distGTStat, distEstStat;
//	qualityParm1 distMatchStatAuto, distGTStatAuto, distEstStatAuto, distMatchStatManual, distGTStatManual, distEstStatManual;
//	getStatisticfromVec2(distances, &distMatchStat, false);
//	getStatisticfromVec2(distancesGT, &distGTStat, false);
//	getStatisticfromVec2(distancesEstModel, &distEstStat, false);
//	getStatisticfromVec2(distancesMatchesAuto, &distMatchStatAuto, false);
//	getStatisticfromVec2(distancesDatasetGTAuto, &distGTStatAuto, false);
//	getStatisticfromVec2(distancesEstModelAuto, &distEstStatAuto, false);
//	getStatisticfromVec2(distancesMatchesManual, &distMatchStatManual, false);
//	getStatisticfromVec2(distancesDatasetGTManual, &distGTStatManual, false);
//	getStatisticfromVec2(distancesEstModelManual, &distEstStatManual, false);
//	writeAddDataToFile(path + "\\stats-DistsBoxPl_" + filename, "MatchDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStat, distances, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPl_" + filename, "GTDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStat, distancesGT, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPl_" + filename, "GeomModelDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStat, distancesEstModel, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPl_" + filename, "MatchDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStatAuto, distancesMatchesAuto, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPl_" + filename, "MatchDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStatManual, distancesMatchesManual, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPl_" + filename, "GTDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStatAuto, distancesDatasetGTAuto, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPl_" + filename, "GTDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStatManual, distancesDatasetGTManual, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPl_" + filename, "GeomModelDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStatAuto, distancesEstModelAuto, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPl_" + filename, "GeomModelDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStatManual, distancesEstModelManual, true);
//
//	writeAddDataToFile(path + "\\stats-DistsMeanVal_" + filename, "MatchDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStat, distances, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanVal_" + filename, "GTDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStat, distancesGT, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanVal_" + filename, "GeomModelDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStat, distancesEstModel, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanVal_" + filename, "MatchDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStatAuto, distancesMatchesAuto, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanVal_" + filename, "MatchDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStatManual, distancesMatchesManual, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanVal_" + filename, "GTDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStatAuto, distancesDatasetGTAuto, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanVal_" + filename, "GTDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStatManual, distancesDatasetGTManual, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanVal_" + filename, "GeomModelDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStatAuto, distancesEstModelAuto, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanVal_" + filename, "GeomModelDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStatManual, distancesEstModelManual, false);
//
//	return 0;
//}
//
//int writeStatsV2(std::string path, std::string filename, annotImgPars annotationData)
//{
//	if (file_exists(path + "\\stats-DistsBoxPlV2_" + filename))
//		return -1;
//
//	std::vector<double> distancesMatchesAuto, distancesMatchesManual, distancesDatasetGTAuto, distancesDatasetGTManual, distancesEstModelAuto, distancesEstModelManual;
//	std::vector<double> distances, distancesGT, distancesEstModel;
//
//	for (int i = 0; i < (int)annotationData.autoManualAnnot.size(); i++)
//	{
//		for (int j = 0; j < (int)annotationData.autoManualAnnot[i].size(); j++)
//		{
//			if (annotationData.autoManualAnnot[i][j] == 'A')
//			{
//				distancesMatchesAuto.push_back(annotationData.distances[i][j]);
//				distancesDatasetGTAuto.push_back(annotationData.distancesGT[i][j]);
//				distancesEstModelAuto.push_back(annotationData.distancesEstModel[i][j]);
//			}
//			else
//			{
//				distancesMatchesManual.push_back(annotationData.distances[i][j]);
//				distancesDatasetGTManual.push_back(annotationData.distancesGT[i][j]);
//				distancesEstModelManual.push_back(annotationData.distancesEstModel[i][j]);
//			}
//		}
//	}
//
//	distances = distancesMatchesAuto;
//	distances.insert(distances.end(), distancesMatchesManual.begin(), distancesMatchesManual.end());
//
//	distancesGT = distancesDatasetGTAuto;
//	distancesGT.insert(distancesGT.end(), distancesDatasetGTManual.begin(), distancesDatasetGTManual.end());
//
//	distancesEstModel = distancesEstModelAuto;
//	distancesEstModel.insert(distancesEstModel.end(), distancesEstModelManual.begin(), distancesEstModelManual.end());
//
//	qualityParm1 distMatchStat, distGTStat, distEstStat;
//	qualityParm1 distMatchStatAuto, distGTStatAuto, distEstStatAuto, distMatchStatManual, distGTStatManual, distEstStatManual;
//	getStatisticfromVec2(distances, &distMatchStat, false);
//	getStatisticfromVec2(distancesGT, &distGTStat, false);
//	getStatisticfromVec2(distancesEstModel, &distEstStat, false);
//	getStatisticfromVec2(distancesMatchesAuto, &distMatchStatAuto, false);
//	getStatisticfromVec2(distancesDatasetGTAuto, &distGTStatAuto, false);
//	getStatisticfromVec2(distancesEstModelAuto, &distEstStatAuto, false);
//	getStatisticfromVec2(distancesMatchesManual, &distMatchStatManual, false);
//	getStatisticfromVec2(distancesDatasetGTManual, &distGTStatManual, false);
//	getStatisticfromVec2(distancesEstModelManual, &distEstStatManual, false);
//	writeAddDataToFile(path + "\\stats-DistsBoxPlV2_" + filename, "MatchDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStat, distances, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPlV2_" + filename, "GTDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStat, distancesGT, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPlV2_" + filename, "GeomModelDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStat, distancesEstModel, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPlV2_" + filename, "MatchDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStatAuto, distancesMatchesAuto, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPlV2_" + filename, "MatchDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStatManual, distancesMatchesManual, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPlV2_" + filename, "GTDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStatAuto, distancesDatasetGTAuto, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPlV2_" + filename, "GTDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStatManual, distancesDatasetGTManual, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPlV2_" + filename, "GeomModelDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStatAuto, distancesEstModelAuto, true);
//	writeAddDataToFile(path + "\\stats-DistsBoxPlV2_" + filename, "GeomModelDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStatManual, distancesEstModelManual, true);
//
//	writeAddDataToFile(path + "\\stats-DistsMeanValV2_" + filename, "MatchDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStat, distances, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanValV2_" + filename, "GTDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStat, distancesGT, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanValV2_" + filename, "GeomModelDists", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStat, distancesEstModel, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanValV2_" + filename, "MatchDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStatAuto, distancesMatchesAuto, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanValV2_" + filename, "MatchDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distMatchStatManual, distancesMatchesManual, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanValV2_" + filename, "GTDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStatAuto, distancesDatasetGTAuto, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanValV2_" + filename, "GTDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distGTStatManual, distancesDatasetGTManual, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanValV2_" + filename, "GeomModelDistsAuto", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStatAuto, distancesEstModelAuto, false);
//	writeAddDataToFile(path + "\\stats-DistsMeanValV2_" + filename, "GeomModelDistsManual", "# Statistics over the distances from annotated references to matching keypoint and  ground truth position in addition to the distance to the estimated geometry. Moreover, statistics are also provided for automatic and manual annotated matches.", &distEstStatManual, distancesEstModelManual, false);
//
//	return 0;
//}
//
//int countAutoManualAnnotations(std::string path, std::string filename, annotImgPars annotationData)
//{
//	size_t autoan = 0, manan = 0;
//
//	if (file_exists(path + "\\stats-NrManAuto_" + filename))
//		return -1;
//
//	for (int i = 0; i < (int)annotationData.autoManualAnnot.size(); i++)
//	{
//		for (int j = 0; j < (int)annotationData.autoManualAnnot[i].size(); j++)
//		{
//			if (annotationData.autoManualAnnot[i][j] == 'A')
//			{
//				autoan++;
//			}
//			else
//			{
//				manan++;
//			}
//		}
//	}
//
//	std::ofstream ofile(path + "\\stats-NrManAuto_" + filename);
//	ofile << "Number of automatic annotated samples: " << autoan << endl;
//	ofile << "Number of manual annotated samples:  " << manan << endl;
//	ofile << "Number of overall annotated samples:  " << manan + autoan << endl;
//	ofile.close();
//
//	return 0;
//}
//
//void correctMatchingResult(int entrynumber, std::string path, std::string outfilename, std::auto_ptr<baseMatcher> mymatcher)
//{
//	int truePosArr[5] = {0,0,0,0,0};
//	int falsePosArr[5] = {0,0,0,0,0};
//	int falseNegArr[5] = {0,0,0,0,0};
//	int notMatchable;
//	std::vector<std::pair<cv::Point2f,cv::Point2f>> perfectMatches, wrongMatches, wrongMatchesNoMatch;
//	vector<bool> notMatchableVec;
//	vector<std::pair<int,double>> wrongGTidxDist;
//	vector<int> GTidx;
//	string oldFileName, newFileName;
//	oldFileName = path + "\\" + outfilename;
//	newFileName = path + "\\tmp_" + outfilename;
//	std::ifstream evalsToFile(oldFileName);
//	std::ofstream replacedFile(newFileName);
//
//	const int nrOfRows = 17;
//	string rowcontent[17];
//	int rowNr = nrOfRows * entrynumber + 1;
//
//	evalsToFile.seekg(std::ios::beg);
//    for(int i = 0; i < rowNr; ++i)
//	{
//		string line;
//		getline(evalsToFile, line);
//		replacedFile << line << endl;
//        //evalsToFile.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
//    }
//
//	int k;
//	for(k = 1; k <= nrOfRows; k++)
//	{
//		string line;
//		getline(evalsToFile, line);
//		rowcontent[k-1] = line;
//		if(line.empty())//If the line is empty, the end of the file is reached
//		{
//			evalsToFile.close();
//			cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//			exit(0);
//		}
//		else if(((k % nrOfRows) == 3)) //Third entry (of nrOfRows) for every image pair that holds the coordinates of wrong GT matches in the order: left x, left y, right x, right y
//		{
//			std::istringstream is;
//			float val = 0;
//			int j = 0;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 6, "Coords") != 0)
//			{
//				evalsToFile.close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			while(is >> val)
//			{
//				switch(j)
//				{
//				case 0:
//					wrongMatches.push_back(std::make_pair<cv::Point2f,cv::Point2f>(cv::Point2f(val,0), cv::Point2f(0,0)));
//					break;
//				case 1:
//					wrongMatches.back().first.y = val;
//					break;
//				case 2:
//					wrongMatches.back().second.x = val;
//					break;
//				case 3:
//					wrongMatches.back().second.y = val;
//					break;
//				}
//				if(j >= 3)
//					j = 0;
//				else
//					j++;
//			}
//			if(j != 0)
//			{
//				evalsToFile.close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//		}
//		else if(((k % nrOfRows) == 9)) //9th entry (of nrOfRows) for every image pair that holds the number of not matchable GT matches
//		{
//			std::istringstream is;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 12, "NotMatchable") != 0)
//			{
//				evalsToFile.close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			is >> notMatchable;
//		}
//		else if(((k % nrOfRows) == 10)) //10th entry (of nrOfRows) for every image pair that holds the manually selected correspondences in the order: left x, left y, right x, right y
//		{
//			std::istringstream is;
//			float val = 0;
//			int j = 0;
//			string word;
//			is.str(line);
//			is >> word;
//			if(word.compare(0, 14, "PerfectMatches") != 0)
//			{
//				evalsToFile.close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//			while(is >> val)
//			{
//				switch(j)
//				{
//				case 0:
//					perfectMatches.push_back(std::make_pair<cv::Point2f,cv::Point2f>(cv::Point2f(val,0), cv::Point2f(0,0)));
//					break;
//				case 1:
//					perfectMatches.back().first.y = val;
//					break;
//				case 2:
//					perfectMatches.back().second.x = val;
//					break;
//				case 3:
//					perfectMatches.back().second.y = val;
//					break;
//				}
//				if(j >= 3)
//					j = 0;
//				else
//					j++;
//			}
//			if(j != 0)
//			{
//				evalsToFile.close();
//				cout << "Datafile for GT testing " << outfilename << " is corrupt! Exiting." << endl;
//				exit(0);
//			}
//		}
//		else
//		{
//			continue;
//		}
//	}
//
//	notMatchableVec = vector<bool>(wrongMatches.size(), true);
//
//	for(int i = 0; i < (int)perfectMatches.size(); i++)
//	{
//		for(int j = 0; j < (int)mymatcher->matchesGT.size(); j++)
//		{
//			if((abs(perfectMatches[i].first.x - mymatcher->keypL[mymatcher->matchesGT[j].queryIdx].pt.x) < 0.01) && (abs(perfectMatches[i].first.y - mymatcher->keypL[mymatcher->matchesGT[j].queryIdx].pt.y) < 0.01))
//			{
//				GTidx.push_back(j);
//			}
//		}
//		for(int j = 0; j < (int)wrongMatches.size(); j++)
//		{
//			if((abs(perfectMatches[i].first.x - wrongMatches[j].first.x) < 0.01) && (abs(perfectMatches[i].first.y - wrongMatches[j].first.y) < 0.01))
//			{
//				notMatchableVec[j] = false;
//				float dist[2];
//				dist[0] = perfectMatches[i].second.x - wrongMatches[j].second.x;
//				dist[1] = perfectMatches[i].second.y - wrongMatches[j].second.y;
//				wrongGTidxDist.push_back(make_pair(GTidx.back(), (double)std::sqrt(dist[0] * dist[0] + dist[1] * dist[1])));
//				break;
//			}
//		}
//	}
//	if(notMatchable > 0)
//	{
//		for(int j = 0; j < (int)notMatchableVec.size(); j++)
//		{
//			if(notMatchableVec[j])
//				wrongMatchesNoMatch.push_back(wrongMatches[j]);
//		}
//	}
//	for(int i = 0; i < (int)wrongMatchesNoMatch.size(); i++)
//	{
//		for(int j = 0; j < (int)mymatcher->matchesGT.size(); j++)
//		{
//			if((abs(wrongMatchesNoMatch[i].first.x - mymatcher->keypL[mymatcher->matchesGT[j].queryIdx].pt.x) < 0.01) && (abs(wrongMatchesNoMatch[i].first.y - mymatcher->keypL[mymatcher->matchesGT[j].queryIdx].pt.y) < 0.01))
//			{
//				GTidx.push_back(j);
//				wrongGTidxDist.push_back(make_pair(j, -1.0));
//			}
//		}
//	}
//	mymatcher->helpOldCodeBug(wrongGTidxDist, GTidx, truePosArr, falsePosArr, falseNegArr);
//
//	for(k = 1; k <= nrOfRows; k++)
//	{
//		if((k % nrOfRows) == 6)
//		{
//			replacedFile << "NrTruePos[leTH-hTH-h4-h8_pix-noMtch] " << truePosArr[0] << " " << truePosArr[1] << " " << truePosArr[2] << " " << truePosArr[3] << " " << truePosArr[4] << endl;
//		}
//		else if((k % nrOfRows) == 7)
//		{
//			replacedFile << "NrFalsePos[leTH-hTH-h4-h8_pix-noMtch] " << falsePosArr[0] << " " << falsePosArr[1] << " " << falsePosArr[2] << " " << falsePosArr[3] << " " << falsePosArr[4] << endl;
//		}
//		else if((k % nrOfRows) == 8)
//		{
//			replacedFile << "NrFalseNeg[leTH-hTH-h4-h8_pix-noMtch] " << falseNegArr[0] << " " << falseNegArr[1] << " " << falseNegArr[2] << " " << falseNegArr[3] << " " << falseNegArr[4] << endl;
//		}
//		else
//		{
//			replacedFile << rowcontent[k-1] << endl;
//		}
//	}
//	string str;
//	int act_pos = evalsToFile.tellg();
//	int length;
//	evalsToFile.seekg(0, std::ios::end);
//	length = (int)evalsToFile.tellg() - act_pos;
//	str.reserve(length);
//	evalsToFile.seekg(act_pos, std::ios::beg);
//	str.assign((std::istreambuf_iterator<char>(evalsToFile)), std::istreambuf_iterator<char>());
//	replacedFile << str;
//	evalsToFile.close();
//	replacedFile.close();
//
//	
//	if(remove(oldFileName.c_str()) != 0)
//		perror( "Error deleting file" );
//	else
//		puts( "File successfully deleted" );
//
//	if (rename(newFileName.c_str(), oldFileName.c_str()) == 0 )
//		puts ( "File successfully renamed" );
//	else
//		perror( "Error renaming file" );
//
//}
//
//int writeErrIndividualImg(std::string path, std::string filename, annotImgPars annotationData)
//{
//	std::string pathFile = path + "\\individualImgErrors_" + filename;
//	int answere, nrColums, maxRows = 0;
//	if(file_exists(pathFile))
//		return -1;
//
//	answere = MessageBox(NULL, "Do you want to generate error lists for every image seperatly within one file?", "Generate error lists for every image pair?", MB_YESNO | MB_DEFBUTTON2);
//	if(answere == IDNO)
//	{
//		return -2;
//	}
//
//	nrColums = (int)annotationData.distances.size();
//
//	for(int i = 0; i < nrColums; i++)
//	{
//		if((int)annotationData.distances[i].size() > maxRows)
//			maxRows = (int)annotationData.distances[i].size();
//	}
//
//
//	std::ofstream ofile(pathFile);
//	ofile << "# Every column contains the annotated error values for one image pair." << endl;
//	for(int i = 0; i < nrColums; i++)
//		ofile << "ErrImg" << i << " ";
//	ofile << endl;
//	for(int i = 0; i < maxRows; i++)
//	{
//		for(int j = 0; j < nrColums; j++)
//		{
//			if((int)annotationData.distances[j].size() <= i)
//			{
//				ofile << "  ";
//			}
//			else
//			{
//				ofile << annotationData.distances[j][i] << " ";
//			}
//		}
//		ofile << endl;
//	}
//	ofile.close();
//}
//
//
///* Generates initial GTMs if they are missing
//*
//* string imgsPath				Input  -> Path which includes both left and right images
//* string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
//* int flowDispH				Input  -> Indicates which type of ground truth data is used:
//*										  0: flow files from KITTI database
//*										  1: disparity files from KITTI database
//*										  2: homography files (Please note that a homography always relates
//*											 to the first image (e.g. 1->2, 1->3, ...))
//* string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images
//*									      (after prefix only comes the image number)
//* string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
//*									      (after prefix only comes the image number). For testing with homographies,
//*										  this string can be empty.
//* string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
//*									      (after prefix only comes the image number)
//* string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
//*										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
//*										  detectors are possible.
//* string descriptorExtractorGT	Input  -> The used descriptor extractor for generating GT. Possible inputs should only be FREAK
//*										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
//*
//*
//* Return value:				 0:		  Everything ok
//*								-1:		  Failed
//*/
//int generateMissingInitialGTMs(std::string imgsPath, std::string flowDispHPath, int flowDispH,
//	std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//	std::string featureDetector, std::string descriptorExtractorGT)
//{
//	int err;
//	cv::Mat src[2];
//	std::auto_ptr<baseMatcher> mymatcher;	
//
//	if (flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if (err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if (err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//		
//		for (int k = 0; k < (int)filenamesl.size(); k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[k], CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[k], CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageFlowFile(flowDispHPath, filenamesflow[k], &flowimg);
//			if (err)
//			{
//				cout << "Could not open flow file with index " << k << endl;
//				continue;
//			}
//
//			mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k]));
//
//			if (mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			mymatcher->generateOnlyInitialGTM = true;
//			mymatcher->specifiedInlRatio = 0;
//			mymatcher->GTfilterExtractor = descriptorExtractorGT;
//			mymatcher->detectFeatures();
//			mymatcher->checkForGT();
//		}
//			
//	}
//	else if (flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequence(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if (err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequence(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if (err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		for (int k = 0; k < (int)filenamesl.size(); k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[k], CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(imgsPath + "\\" + filenamesr[k], CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageDisparityFile(flowDispHPath, filenamesflow[k], &flowimg);
//			if (err)
//			{
//				cout << "Could not open disparity file with index " << k << ". Exiting." << endl;
//				exit(0);
//			}
//
//			mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k]));
//
//			if (mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			mymatcher->generateOnlyInitialGTM = true;
//			mymatcher->specifiedInlRatio = 0;
//			mymatcher->GTfilterExtractor = descriptorExtractorGT;
//			mymatcher->detectFeatures();
//			mymatcher->checkForGT();
//		}
//	}
//	else if (flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if (err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if (err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for (int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if (err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		if (fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0], CV_LOAD_IMAGE_GRAYSCALE);
//			for (int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1], CV_LOAD_IMAGE_GRAYSCALE);
//
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//				if (mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				mymatcher->generateOnlyInitialGTM = true;
//				mymatcher->specifiedInlRatio = 0;
//				mymatcher->GTfilterExtractor = descriptorExtractorGT;
//				mymatcher->detectFeatures();
//				mymatcher->checkForGT();
//			}
//			//Generate new homographys to evaluate all other possible configurations of the images to each other
//			for (int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//			{
//				for (int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//				{
//					//H = (Hs[idx2].inv() * Hs[idx1]).inv();
//					cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//					src[0] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1], CV_LOAD_IMAGE_GRAYSCALE);
//					src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx2 + 1], CV_LOAD_IMAGE_GRAYSCALE);
//
//					mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//
//					if (mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					mymatcher->generateOnlyInitialGTM = true;
//					mymatcher->specifiedInlRatio = 0;
//					mymatcher->GTfilterExtractor = descriptorExtractorGT;
//					mymatcher->detectFeatures();
//					mymatcher->checkForGT();
//				}
//			}
//		}
//		else
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "\\" + filenamesl[0], CV_LOAD_IMAGE_GRAYSCALE);
//			for (int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "\\" + filenamesl[idx1 + 1], CV_LOAD_IMAGE_GRAYSCALE);
//
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//				if (mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				mymatcher->generateOnlyInitialGTM = true;
//				mymatcher->specifiedInlRatio = 0;
//				mymatcher->GTfilterExtractor = descriptorExtractorGT;
//				mymatcher->detectFeatures();
//				mymatcher->checkForGT();
//			}
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//
//	return 0;
//}
//
///* Generates GTMs with different inlier ratios for a given dataset 
//*
//* string imgsPath				Input  -> Path which includes both left and right images
//* string flowDispHPath			Input  -> Path to the flow files, disparity files or homography files
//* int flowDispH				Input  -> Indicates which type of ground truth data is used:
//*										  0: flow files from KITTI database
//*										  1: disparity files from KITTI database
//*										  2: homography files (Please note that a homography always relates
//*											 to the first image (e.g. 1->2, 1->3, ...))
//* string filePrefImgL			Input  -> File prefix including a "_" at the end for the left or first images
//*									      (after prefix only comes the image number)
//* string filePrefImgR			Input  -> File prefix including a "_" at the end for the right or second images
//*									      (after prefix only comes the image number). For testing with homographies,
//*										  this string can be empty.
//* string filePrefFlowDispH		Input  -> File prefix for the flow, disparity, or homography files
//*									      (after prefix only comes the image number)
//* string featureDetector		Input  -> The used feature detector. Possible imputs should only be FAST or SIFT,
//*										  although other detectors from OpenCV 2.4.9 excluding MSER and blob
//*										  detectors are possible.
//* string descriptorExtractorGT	Input  -> The used descriptor extractor for generating GT. Possible inputs should only be FREAK
//*										  or SIFT, although other extractors from OpenCV 2.4.9 are possible.
//*
//*
//* Return value:				 0:		  Everything ok
//*								-1:		  Failed
//*/
//int generateGTMs(std::string imgsPath, std::string flowDispHPath, int flowDispH,
//	std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
//	std::string featureDetector, std::string descriptorExtractorGT)
//{
//	int err;
//	cv::Mat src[2];
//	std::auto_ptr<baseMatcher> mymatcher;
//	vector<double> inlierRatios;
//	int nr_inlratios;
//
//	//Generate inlier ratios
//	double startInlRatio = 1.0;
//	inlierRatios.push_back(startInlRatio);
//	while (startInlRatio > 0.2)
//	{
//		startInlRatio -= 0.05;
//		inlierRatios.push_back(startInlRatio);
//	}
//	while (startInlRatio > 0.1)
//	{
//		startInlRatio -= 0.02;
//		inlierRatios.push_back(startInlRatio);
//	}
//	while (startInlRatio > 0.01)
//	{
//		startInlRatio -= 0.01;
//		inlierRatios.push_back(startInlRatio);
//	}
//	/*while(startInlRatio > 0.005)
//	{
//	startInlRatio -= 0.005;
//	inlierRatios.push_back(startInlRatio);
//	}*/
//	nr_inlratios = (int)inlierRatios.size();
//
//	if (flowDispH == 0)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequenceNew(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if (err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find flow images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequenceNew(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if (err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find flow files! Exiting." << endl;
//			exit(0);
//		}
//
//		//Get length of path
//		size_t pathPos1 = filenamesflow.back().rfind("/") + 1;
//		if (pathPos1 == std::string::npos)
//			pathPos1 = 0;
//		size_t pathPos2 = filenamesl.back().rfind("/") + 1;
//		if (pathPos2 == std::string::npos)
//			pathPos2 = 0;
//
//		for (int k = 0; k < (int)filenamesl.size(); k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(filenamesl[k], CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(filenamesr[k], CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageFlowFile(flowDispHPath, filenamesflow[k].substr(pathPos1), &flowimg);
//			if (err)
//			{
//				cout << "Could not open flow file with index " << k << endl;
//				continue;
//			}
//
//			mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k].substr(pathPos2)));
//
//			if (mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			cout << "Generating GTMs for image " << filenamesl[k] << endl;
//			cout << std::fixed;
//			cout << "Computing inlier ratio: ";
//			for (int i = 0; i < nr_inlratios; i++)
//			{
//				cout << std::setprecision(2) << inlierRatios[i];
//				mymatcher->specifiedInlRatio = inlierRatios[i];
//				mymatcher->GTfilterExtractor = descriptorExtractorGT;
//				//mymatcher->detectFeatures();
//				mymatcher->checkForGT();
//				cout << "; ";
//			}
//			cout << endl;
//		}
//
//	}
//	else if (flowDispH == 1)
//	{
//		vector<string> filenamesl, filenamesr, filenamesflow;
//		err = loadStereoSequenceNew(imgsPath, filePrefImgL, filePrefImgR, filenamesl, filenamesr);
//		if (err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
//		{
//			cout << "Could not find disparity images! Exiting." << endl;
//			exit(0);
//		}
//		err = loadImageSequenceNew(flowDispHPath, filePrefFlowDispH, filenamesflow);
//		if (err || filenamesflow.empty() || (filenamesflow.size() != filenamesl.size()))
//		{
//			cout << "Could not find disparity files! Exiting." << endl;
//			exit(0);
//		}
//
//		//Get length of path
//		size_t pathPos1 = filenamesflow.back().rfind("/") + 1;
//		if (pathPos1 == std::string::npos)
//			pathPos1 = 0;
//		size_t pathPos2 = filenamesl.back().rfind("/") + 1;
//		if (pathPos2 == std::string::npos)
//			pathPos2 = 0;
//
//		for (int k = 0; k < (int)filenamesl.size(); k++)
//		{
//			cv::Mat flowimg;
//			src[0] = cv::imread(filenamesl[k], CV_LOAD_IMAGE_GRAYSCALE);
//			src[1] = cv::imread(filenamesr[k], CV_LOAD_IMAGE_GRAYSCALE);
//			err = convertImageDisparityFile(flowDispHPath, filenamesflow[k].substr(pathPos1), &flowimg);
//			if (err)
//			{
//				cout << "Could not open disparity file with index " << k << ". Exiting." << endl;
//				exit(0);
//			}
//
//			mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, flowimg, true, imgsPath, filenamesl[k].substr(pathPos2)));
//
//			if (mymatcher->specialGMbSOFtest)
//			{
//				cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//				exit(0);
//			}
//
//			cout << "Generating GTMs for image " << filenamesl[k] << endl;
//			cout << std::fixed;
//			cout << "Computing inlier ratio: ";
//			for (int i = 0; i < nr_inlratios; i++)
//			{
//				cout << std::setprecision(2) << inlierRatios[i];
//				mymatcher->specifiedInlRatio = inlierRatios[i];
//				mymatcher->GTfilterExtractor = descriptorExtractorGT;
//				//mymatcher->detectFeatures();
//				mymatcher->checkForGT();
//				cout << "; ";
//			}
//			cout << endl;
//		}
//	}
//	else if (flowDispH == 2)
//	{
//		vector<string> filenamesl, fnames;
//		cv::Mat H;
//		err = loadImageSequence(imgsPath, filePrefImgL, filenamesl);
//		if (err || filenamesl.empty())
//		{
//			cout << "Could not find homography images! Exiting." << endl;
//			exit(0);
//		}
//		err = readHomographyFiles(flowDispHPath, filePrefFlowDispH, fnames);
//		if (err || fnames.empty() || ((fnames.size() + 1) != filenamesl.size()))
//		{
//			cout << "Could not find homography files or number of provided homography files is wrong! Exiting." << endl;
//			exit(0);
//		}
//		std::vector<cv::Mat> Hs(fnames.size());
//		for (int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//		{
//			err = readHomographyFromFile(flowDispHPath, fnames[idx1], &(Hs[idx1]));
//			if (err)
//			{
//				cout << "Error opening homography file with index " << idx1 << ". Exiting." << endl;
//				exit(0);
//			}
//		}
//
//		if (fnames.size() < 30) //Perform evaluation on all possible configurations of the homography
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "/" + filenamesl[0], CV_LOAD_IMAGE_GRAYSCALE);
//			for (int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "/" + filenamesl[idx1 + 1], CV_LOAD_IMAGE_GRAYSCALE);
//
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//				if (mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				cout << "Generating GTMs for images " << filenamesl[0] << "and " << filenamesl[idx1 + 1] << endl;
//				cout << std::fixed;
//				cout << "Computing inlier ratio: ";
//				for (int i = 0; i < nr_inlratios; i++)
//				{
//					cout << std::setprecision(2) << inlierRatios[i];
//					mymatcher->specifiedInlRatio = inlierRatios[i];
//					mymatcher->GTfilterExtractor = descriptorExtractorGT;
//					//mymatcher->detectFeatures();
//					mymatcher->checkForGT();
//					cout << "; ";
//				}
//				cout << endl;
//			}
//			//Generate new homographys to evaluate all other possible configurations of the images to each other
//			for (int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//			{
//				for (int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//				{
//					//H = (Hs[idx2].inv() * Hs[idx1]).inv();
//					cv::Mat H = Hs[idx2] * Hs[idx1].inv();
//					src[0] = cv::imread(imgsPath + "/" + filenamesl[idx1 + 1], CV_LOAD_IMAGE_GRAYSCALE);
//					src[1] = cv::imread(imgsPath + "/" + filenamesl[idx2 + 1], CV_LOAD_IMAGE_GRAYSCALE);
//
//					mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[idx1 + 1] + "_" + filenamesl[idx2 + 1]));
//
//					if (mymatcher->specialGMbSOFtest)
//					{
//						cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//						exit(0);
//					}
//
//					cout << "Generating GTMs for images " << filenamesl[idx1 + 1] << "and " << filenamesl[idx2 + 1] << endl;
//					cout << std::fixed;
//					cout << "Computing inlier ratio: ";
//					for (int i = 0; i < nr_inlratios; i++)
//					{
//						cout << std::setprecision(2) << inlierRatios[i];
//						mymatcher->specifiedInlRatio = inlierRatios[i];
//						mymatcher->GTfilterExtractor = descriptorExtractorGT;
//						//mymatcher->detectFeatures();
//						mymatcher->checkForGT();
//						cout << "; ";
//					}
//					cout << endl;
//				}
//			}
//		}
//		else
//		{
//			//Take the stored homographys and perform evaluation
//			src[0] = cv::imread(imgsPath + "/" + filenamesl[0], CV_LOAD_IMAGE_GRAYSCALE);
//			for (int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//			{
//				cv::Mat H = Hs[idx1];
//				src[1] = cv::imread(imgsPath + "/" + filenamesl[idx1 + 1], CV_LOAD_IMAGE_GRAYSCALE);
//
//				mymatcher.reset(new Linear_matcher(src[0], src[1], featureDetector, descriptorExtractorGT, H, false, imgsPath, filenamesl[0] + "_" + filenamesl[idx1 + 1]));
//
//				if (mymatcher->specialGMbSOFtest)
//				{
//					cout << "Test framework was specially compiled for testing GMbSOF! Exiting." << endl;
//					exit(0);
//				}
//
//				cout << "Generating GTMs for images " << filenamesl[0] << "and " << filenamesl[idx1 + 1] << endl;
//				cout << std::fixed;
//				cout << "Computing inlier ratio: ";
//				for (int i = 0; i < nr_inlratios; i++)
//				{
//					cout << std::setprecision(2) << inlierRatios[i];
//					mymatcher->specifiedInlRatio = inlierRatios[i];
//					mymatcher->GTfilterExtractor = descriptorExtractorGT;
//					mymatcher->detectFeatures();
//					mymatcher->checkForGT();
//					cout << "; ";
//				}
//				cout << endl;
//			}
//		}
//	}
//	else
//	{
//		cout << "The paramter you specified for the scenetype is out of range! Use 0 for flow, 1 for disparity, and 2 for homography! Exiting." << endl;
//		exit(0);
//	}
//
//	return 0;
//}