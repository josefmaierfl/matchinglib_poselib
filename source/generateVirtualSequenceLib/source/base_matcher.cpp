///**********************************************************************************************************
// FILE: base_matcher.cpp
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
// DISCRIPTION: This file provides functionalities for testing different matching algorithms. The class
// baseMatcher provides all functionalities necessary for before and after matching, like feature and 
// descriptor extraction, quality measurement on the final matches as well as refinement of the found
// matches. The matching algorithms themself must be implemented as a child class of this base class.
//**********************************************************************************************************/
//
//#include "base_matcher.h"
//#include "opencv2/xfeatures2d/nonfree.hpp"
//#include "opencv2/xfeatures2d.hpp"
//#include <opencv2/features2d/features2d.hpp>
////#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/video/tracking.hpp"
//
//#include <Eigen/Core>
////#include <opencv2/core/eigen.hpp>
////#include <Eigen/Dense>
////#include <Eigen/StdVector>
//
//#include "nanoflann.hpp"
//
////#include "vfc.h"
//
//#include <intrin.h>
//#include <bitset>
//
//#include <direct.h>
//#include <fstream>
//#include <Windows.h>
//
//#include <time.h>
//#include <opencv2/imgproc/imgproc.hpp>
//
//#include "inscribeRectangle.h"
//
//#include "matchinglib/matchinglib.h"
//#include "matchinglib/vfcMatches.h"
//
//using namespace cv;
//using namespace std;
//
///* --------------------------- Defines --------------------------- */
//
//#define LEFT_TO_RIGHT 0 //For testing the GT matches: Changes the order of images from left-right to right-left if 0
//
//typedef Eigen::Matrix<float,Eigen::Dynamic,2, Eigen::RowMajor> EMatFloat2;
//typedef nanoflann::KDTreeEigenMatrixAdaptor< EMatFloat2, 2,nanoflann::metric_L2_Simple>  KDTree_D2float;
//
//enum SpecialKeyCode{
//        NONE, SPACE, BACKSPACE, ESCAPE, CARRIAGE_RETURN, ARROW_UP, ARROW_RIGHT, ARROW_DOWN, ARROW_LEFT, PAGE_UP, PAGE_DOWN, POS1, END_KEY ,INSERT, DELETE_KEY
//    };
//
///* --------------------- Function prototypes --------------------- */
//
//float getDescriptorDistance(cv::Mat descriptor1, cv::Mat descriptor2);
//bool readDoubleVal(ifstream & gtFromFile, std::string keyWord, double *value);
//void getSubPixPatchDiff(cv::Mat patch, cv::Mat image, cv::Point2f &diff);
//void iterativeTemplMatch(cv::InputArray patch, cv::InputArray img, int maxPatchSize, cv::Point2f & minDiff, int maxiters = INT_MAX);
//void on_mouse_click(int event, int x, int y, int flags, void* param);
//SpecialKeyCode getSpecialKeyCode(int & val);
//float BilinearInterpolation(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y);
//void calcErrorToSpatialGT(cv::Point2f perfectMatchesFirst, cv::Point2f perfectMatchesSecond, 
//						  std::vector<cv::Mat> channelsFlow, bool flowGtIsUsed, std::vector<cv::Point2f> & errvecsGT, 
//						  std::vector<int> & validityValGT, cv::Mat homoGT, cv::Point2f lkp, cv::Point2f rkp);
//void intersectPolys(std::vector<cv::Point2d> pntsPoly1, std::vector<cv::Point2d> pntsPoly2, std::vector<cv::Point2f> &pntsRes);
//void findLocalMin(Mat patchL, Mat patchR, float quarterpatch, float eigthpatch, cv::Point2f &winPos, float patchsizeOrig);
//
////from CV, but slightly changed:
//void drawMatchesCV( const Mat& img1, const vector<KeyPoint>& keypoints1,
//                  const Mat& img2, const vector<KeyPoint>& keypoints2,
//                  const vector<DMatch>& matches1to2, Mat& outImg,
//                  const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
//                  const vector<char>& matchesMask=vector<char>(), int flags=DrawMatchesFlags::DEFAULT );
//static void _prepareImgAndDrawKeypointsCV( const Mat& img1, const vector<KeyPoint>& keypoints1,
//                                         const Mat& img2, const vector<KeyPoint>& keypoints2,
//                                         Mat& outImg, Mat& outImg1, Mat& outImg2,
//                                         const Scalar& singlePointColor, int flags );
//static inline void _drawMatchCV( Mat& outImg, Mat& outImg1, Mat& outImg2 ,
//                          const KeyPoint& kp1, const KeyPoint& kp2, const Scalar& matchColor, int flags );
//static inline void _drawKeypointCV( Mat& img, const KeyPoint& p, const Scalar& color, int flags );
//
///* --------------------- Functions --------------------- */
//
//
///* Constructor of class baseMatcher. 
// *
// * Mat leftImg					Input  -> Left greyscale image
// * Mat rightImg					Input  -> Right greyscale image
// * string _featuretype			Input  -> Feature type like FAST, SIFT, ... that are defined in the OpenCV (FeatureDetector::create).
// *										  Only feature types MSER, Dense, and SimpleBlob are not allowed.
// * string _descriptortype		Input  -> Descriptor type like SIFT, FREAK, ... that are defined in the OpenCV (DescriptorExtractor::create).
// * Mat flowOrHomoGT				Input  -> Ground truth flow of the image like defined in function convertImageFlowFile (io_data.cpp) or a
// *										  ground truth 3x3 homography. If the optical flow is used, _flowGtIsUsed must be true, otherwise it
// *										  must be false.
// * bool _flowGtIsUsed			Input  -> Specifies if a ground truth optical flow or a homography is used. For the former it must be true 
// *										  and for the latter false.
// * string _imgsPath				Input  -> Path to the images which is necessary for loading and storing the ground truth matches
// * string _fileNameImgL			Input  -> Filename of the first (or left) image which is used to generate the filename for the GT matches.
// *										  Here, any string can be passed, but be sure to be unique, as the GT is stored to a filename with
// *										  the name provided. If dots are included in the string, these and the following 3 characters are removed.
// *
// * Return value:				none
// */
//baseMatcher::baseMatcher(cv::Mat leftImg, cv::Mat rightImg, std::string _featuretype, std::string _descriptortype, cv::Mat flowOrHomoGT, bool _flowGtIsUsed, std::string _imgsPath, std::string _fileNameImgL)
//{
//	CV_Assert(!leftImg.empty() && !rightImg.empty() && (leftImg.type() == CV_8UC1) && (rightImg.type() == CV_8UC1));
//	/*CV_Assert((_featuretype == "FAST") || (_featuretype == "SIFT") || (_featuretype == "SURF") || (_featuretype == "ORB") ||
//			  (_featuretype == "STAR") || (_featuretype == "BRISK") || (_featuretype == "GFTT") || (_featuretype == "HARRIS"));
//	CV_Assert((_descriptortype == "SIFT") || (_descriptortype == "SURF") || (_descriptortype == "BRIEF") || (_descriptortype == "BRISK") ||
//			  (_descriptortype == "ORB") || (_descriptortype == "FREAK"));*/
//	CV_Assert(((_flowGtIsUsed == false) && (flowOrHomoGT.rows == 3) && (flowOrHomoGT.cols == 3)) || ((_flowGtIsUsed == true) && 
//			  (flowOrHomoGT.rows > 3) && flowOrHomoGT.cols > 3));
//	
//	this->imgs[0] = leftImg;
//	this->imgs[1] = rightImg;
//
//	this->featuretype = _featuretype;
//	this->descriptortype = _descriptortype;
//	/*if((featuretype.compare("SIFT") && !featuretype.compare("SURF")) || (!featuretype.compare("SIFT") && featuretype.compare("SURF"))
//		|| (descriptortype.compare("SIFT") && !descriptortype.compare("SURF")) || (!descriptortype.compare("SIFT") && descriptortype.compare("SURF")))
//		initModule_nonfree();*/
//
//	this->flowGtIsUsed = _flowGtIsUsed;
//	if(_flowGtIsUsed)
//	{
//		this->flowGT = flowOrHomoGT;
//	}
//	else
//	{
//		this->homoGT = flowOrHomoGT;
//	}
//
//	cyklesTM = TIMEMEASITERS;
//
//	memset(&qpm, 0, sizeof(matchQualParams));
//	memset(&qpr, 0, sizeof(matchQualParams));
//	tm = 0;
//
//	imgsPath = _imgsPath;
//	fileNameImgL = _fileNameImgL;
//
//	useSameKeypSiVarInl = false;
//
//#if INITMATCHQUALEVAL_O
//	specialGMbSOFtest = true;
//#else
//	specialGMbSOFtest = false;
//#endif
//
//	//testGT = false;
//	GTfilterExtractor = "FREAK";
//	gtsubfolder = "matchesGT";
//	noGTgenBefore = true;
//	measureTd = false;
//	generateOnlyInitialGTM = false;
//}
//
///* Feature extraction in both images without filtering with ground truth.
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Error creating feature detector
// *								-2:		  Too less features detected
// */
//int baseMatcher::detectFeatures()
//{
//	//Ptr<FeatureDetector> detector = FeatureDetector::create( featuretype );
//	//if(detector.empty())
//	//{
//	//	cout << "Cannot create feature detector!" << endl;
//	//	return -1; //Error creating feature detector
//	//}
//	double tf_tmp;
//
//	tf = DBL_MAX;
//	if(!measureT)
//		cyklesTM = 1;
//
//	for(int i = 0; i < 1/*cyklesTM*/; i++)
//	{
//		//Clear variables
//		keypL.clear();
//		keypR.clear();
//
//		tf_tmp = (double)getTickCount(); //Start time measurement
//		//detector->detect( imgs[0], keypL );
//		
//		if (matchinglib::getKeypoints(imgs[0], keypL, featuretype, false, INT_MAX) != 0)
//			return -1;
//
//		if(keypL.size() < 15)
//		{
//			return -2; //Too less features detected
//		}
//
//#define showfeatures 0
//#if showfeatures
//		cv::Mat img1c;
//		drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//		imshow("Keypoints 1", img1c );
//#endif
//
//
//		//detector->detect( imgs[1], keypR );
//		if (matchinglib::getKeypoints(imgs[1], keypR, featuretype, false, INT_MAX) != 0)
//			return -1;
//		if(keypR.size() < 15)
//		{
//			return -2; //Too less features detected
//		}
//
//#if showfeatures
//		cv::Mat img2c;
//		drawKeypoints( imgs[1], keypR, img2c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//		imshow("Keypoints 2", img2c );
//		waitKey(0);
//#endif
//
//		tf_tmp = 1000 * ((double)getTickCount() - tf_tmp) / getTickFrequency(); //End time measurement
//		if(tf_tmp < tf)
//			tf = tf_tmp;
//	}
//	cout << "Feature extraction time (ms): " << tf << endl;
//
//	return 0;
//}
//
///* Generates ground truth matches from the keypoints and calculates a threshold for checking
// * the correctness of matches by searching for the nearest and second nearest neighbors. The
// * threshold is then set between the largest distance of the nearest neighbors and the shortest
// * distance to the second nearest neighbors. Therefore, some keypoints might be removed, if
// * they are too close to each other so that a clear identification of the correct keypoint
// * would be impossible. Moreover, keypoints in the left image are removed that point to the 
// * same keypoint in the right image.
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Too less features detected
// */
//int baseMatcher::filterInitFeaturesGT()
//{
//#define showfeatures 0
//	//Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(GTfilterExtractor);
//	Mat descriptors1, descriptors22nd;
//	std::vector<cv::KeyPoint> keypR_tmp, keypR_tmp1;//Right found keypoints
//	std::vector<cv::KeyPoint> keypL_tmp, keypL_tmp1;//Left found keypoints
//	vector<size_t> border_dellist; //Indices of keypoints in the right image that are too near at the border
//	vector<size_t> border_dellistL; //Indices of keypoints in the left image that are too near at the border
//	float h = (float)flowGT.rows;
//	float w = (float)flowGT.cols;
//	vector<Mat> channelsFlow(3);
//	vector<std::pair<size_t, float>> nearest_dist, second_nearest_dist;
//
//	//Check if for this image pair basic GT is available in a file
//	if (!noGTgenBefore)
//	{
//		for (size_t j = 0; j < this->matchesGT.size(); j++)
//		{
//			nearest_dist.push_back(std::make_pair((size_t)matchesGT[j].trainIdx, matchesGT[j].distance));
//		}
//
//		this->matchesGT.clear();
//	}
//
//	if(noGTgenBefore)
//	{
//	//Split 3 channel matrix for access
//	if(flowGtIsUsed)
//	{
//		cv::split(flowGT, channelsFlow);
//	}
//
//	//Get descriptors from left keypoints
//	//extractor->compute(imgs[0],keypL,descriptors1);
//	matchinglib::getDescriptors(imgs[0], keypL, GTfilterExtractor, descriptors1, featuretype);
//
//	//Get descriptors from right keypoints
//	//extractor->compute(imgs[1],keypR,descriptors22nd);
//	matchinglib::getDescriptors(imgs[1], keypR, GTfilterExtractor, descriptors22nd, featuretype);
//
//	//Prepare the coordinates of the keypoints for the KD-tree (must be after descriptor extractor because keypoints near the border are removed)
//	EMatFloat2 eigkeypts2(keypR.size(),2);
//	for(unsigned int i = 0;i<keypR.size();i++)
//	{
//		eigkeypts2(i,0) = keypR[i].pt.x;
//		eigkeypts2(i,1) = keypR[i].pt.y;
//	}
//	
//	//Generate the KD-tree index for the keypoint coordinates
//	float searchradius = (float)INITMATCHDISTANCETH_GT * (float)INITMATCHDISTANCETH_GT;
//	const int maxLeafNum     = 20;
//	const int maxDepthSearch = 32;
//	vector<std::pair<size_t,float>> radius_matches;
//	Eigen::Vector2f x2e;
//
//	KDTree_D2float keypts2idx(2,eigkeypts2,maxLeafNum);
//	keypts2idx.index->buildIndex();
//
//	//Search for the ground truth matches in the right (second) image by searching for the nearest and second nearest neighbor by a radius search
//	if(flowGtIsUsed)
//	{
//		int xd, yd;
//		//Search for the ground truth matches using optical flow data
//		for(int i = 0;i<keypL.size();i++)
//		{
//			cv::Point2i hlp;
//			hlp.x = (int)floor(keypL[i].pt.x + 0.5f); //Round to nearest integer
//			hlp.y = (int)floor(keypL[i].pt.y + 0.5f); //Round to nearest integer
//			if(channelsFlow[2].at<float>(hlp.y, hlp.x) == 1.0)
//			{
//				x2e(0) = keypL[i].pt.x + channelsFlow[0].at<float>(hlp.y, hlp.x);
//				x2e(1) = keypL[i].pt.y + channelsFlow[1].at<float>(hlp.y, hlp.x);
//			}
//			else if(channelsFlow[2].at<float>(hlp.y, hlp.x) > 1.0) //Check if the filled flow (with median interpolation) is near a border with invalid flow -> if yes, reject the keypoint
//			{
//				int dx;
//				for(dx = -5; dx < 6; dx++)
//				{
//					int dy;
//					for(dy = -5; dy < 6; dy++)
//					{
//						xd = hlp.x + dx;
//						yd = hlp.y + dy;
//						if((xd > 0) && (xd < (int)w) && (yd > 0) && (yd < (int)h))
//						{
//							if(channelsFlow[2].at<float>(yd, xd) == 0)
//								break;
//						}
//						else
//						{
//							continue;
//						}
//					}
//					if(dy < 6)
//						break;
//				}
//				if(dx < 6)
//				{
//					keypL.erase(keypL.begin()+i);
//					i--;
//					continue;
//				}
//				else
//				{
//					x2e(0) = keypL[i].pt.x + channelsFlow[0].at<float>(hlp.y, hlp.x);
//					x2e(1) = keypL[i].pt.y + channelsFlow[1].at<float>(hlp.y, hlp.x);
//				}
//			}
//			else
//			{
//				keypL.erase(keypL.begin()+i);
//				i--;
//				continue;
//			}
//
//			keypts2idx.index->radiusSearch(&x2e(0),searchradius,radius_matches,nanoflann::SearchParams(maxDepthSearch));
//			if(radius_matches.empty())
//			{
//				this->leftInlier.push_back(false);
//				radius_matches.clear();
//				continue;
//			}
//			this->leftInlier.push_back(true);
//			nearest_dist.push_back(radius_matches[0]);
//			if(radius_matches.size() > 1)
//			{
//				second_nearest_dist.insert(second_nearest_dist.end(), radius_matches.begin()+1, radius_matches.end());
//			}
//			radius_matches.clear();
//		}
//	}
//	else
//	{
//		//Search for ground truth matches using a homography
//		for(unsigned int i = 0;i<keypL.size();i++)
//		{
//			float hlp = (float)homoGT.at<double>(2,0) * keypL[i].pt.x + (float)homoGT.at<double>(2,1) * keypL[i].pt.y + (float)homoGT.at<double>(2,2);;
//			x2e(0) = (float)homoGT.at<double>(0,0) * keypL[i].pt.x + (float)homoGT.at<double>(0,1) * keypL[i].pt.y + (float)homoGT.at<double>(0,2);
//			x2e(1) = (float)homoGT.at<double>(1,0) * keypL[i].pt.x + (float)homoGT.at<double>(1,1) * keypL[i].pt.y + (float)homoGT.at<double>(1,2);
//			x2e(0) /= hlp;
//			x2e(1) /= hlp;
//			keypts2idx.index->radiusSearch(&x2e(0),searchradius,radius_matches,nanoflann::SearchParams(maxDepthSearch));
//			if(radius_matches.empty())
//			{
//				this->leftInlier.push_back(false);
//				radius_matches.clear();
//				continue;
//			}
//			this->leftInlier.push_back(true);
//			nearest_dist.push_back(radius_matches[0]);
//			if(radius_matches.size() > 1)
//			{
//				second_nearest_dist.insert(second_nearest_dist.end(), radius_matches.begin()+1, radius_matches.end());
//			}
//			radius_matches.clear();
//		}
//	}
//
//	if(nearest_dist.empty())
//		return -1; //No corresponding keypoints found
//
//#if showfeatures
//	cv::Mat img1c;
//	drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//	imshow("Keypoints 1 after invalid GT filtering", img1c );
//	img1c.release();
//	drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//	imshow("Keypoints 2", img1c );
//	cv::waitKey(0);
//#endif
//		
//	//Sort distances to get the largest distances first
//	sort(nearest_dist.begin(), nearest_dist.end(),
//		[](pair<size_t,float> first, pair<size_t,float> second){return first.second > second.second;});
//
//	if(!second_nearest_dist.empty())
//	{
//		//Check for outliers in the distances to the nearest matches
//		float median, hlp, medstdsum = 0, medStd, distantth;
//		if(nearest_dist.size() % 2)
//			median = nearest_dist[(nearest_dist.size()-1)/2].second;
//		else
//			median = (nearest_dist[nearest_dist.size() / 2].second + nearest_dist[nearest_dist.size() / 2 - 1].second) / 2;
//
//		int startexcludeval = (int)floor((float)nearest_dist.size() * 0.2); //to exclude the first 20% of large distances
//		for(int i = startexcludeval; i < nearest_dist.size(); i++)
//		{
//			hlp = (nearest_dist[i].second - median);
//			medstdsum += hlp * hlp;
//		}
//		if(std::abs(medstdsum) < 1e-6)
//			medStd = 0.0;
//		else
//			medStd = std::sqrtf(medstdsum/((float)nearest_dist.size() - (float)startexcludeval - 1.0f));
//
//		distantth = median + 3.5f * medStd;
//		if(distantth >= INITMATCHDISTANCETH_GT * INITMATCHDISTANCETH_GT)
//		{
//			distantth = (float)INITMATCHDISTANCETH_GT / 2.0;
//			distantth *= distantth;
//		}
//
//		//Reject outliers in the distances to the nearest matches
//		while(nearest_dist[0].second >= distantth)
//		{
//			nearest_dist.erase(nearest_dist.begin());
//		}
//
//		//Sort second nearest distances to get smallest distances first
//		sort(second_nearest_dist.begin(), second_nearest_dist.end(),
//			[](pair<size_t,float> first, pair<size_t,float> second){return first.second < second.second;});
//		//Mark too near keypoints for deleting
//		size_t k = 0;
//		while(second_nearest_dist[k].second <= nearest_dist[0].second)
//		{
//			k++;
//		}
//
//		//Set the threshold
//		usedMatchTH = ((double)std::sqrt(nearest_dist[0].second) + (double)std::sqrt(second_nearest_dist[k].second)) / 2.0;
//		if(usedMatchTH < 2.0)
//			usedMatchTH = 2.0;
//	}
//	else
//	{
//		usedMatchTH = (double)std::sqrt(nearest_dist[0].second) + 0.5;
//		usedMatchTH = usedMatchTH > (double)INITMATCHDISTANCETH_GT ? (double)INITMATCHDISTANCETH_GT:usedMatchTH;
//	}
//	searchradius = (float)usedMatchTH * (float)usedMatchTH;//floor(usedMatchTH * usedMatchTH + 0.5f);
//	//cout << "Threshold: " << usedMatchTH << endl;
//
//	//Search for the ground truth matches in the right (second) image by searching for the nearest and second nearest neighbor by a radius search
//	//Recalculate descriptors to exclude descriptors from deleted left keypoints
//	descriptors1.release();
//	//extractor->compute(imgs[0],keypL,descriptors1);
//	matchinglib::getDescriptors(imgs[0], keypL, GTfilterExtractor, descriptors1, featuretype);
//	this->leftInlier.clear();
//	nearest_dist.clear();
//	vector<vector<std::pair<size_t,float>>> second_nearest_dist_vec;
//	if(flowGtIsUsed)
//	{
//		//Search for the ground truth matches using optical flow data
//		for(int i = 0;i<keypL.size();i++)
//		{
//			Mat descriptors2;
//			float descr_dist1, descr_dist2; //Descriptor distance of found right keypoints
//			size_t minDescrDist = 0;
//			vector<size_t> border_marklist;//Number of points that are too near to the image border
//			cv::Point2i hlp;
//			hlp.x = (int)floor(keypL[i].pt.x + 0.5); //Round to nearest integer
//			hlp.y = (int)floor(keypL[i].pt.y + 0.5); //Round to nearest integer
//			if(channelsFlow[2].at<float>(hlp.y, hlp.x) >= 1.0)
//			{
//				x2e(0) = keypL[i].pt.x + channelsFlow[0].at<float>(hlp.y, hlp.x);
//				x2e(1) = keypL[i].pt.y + channelsFlow[1].at<float>(hlp.y, hlp.x);
//			}
//			else
//			{
//				this->keypL.erase(keypL.begin() + i);
//				if(i == 0)
//				{
//					descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
//				}
//				else
//				{
//					Mat descr_tmp = descriptors1.rowRange(0, i);
//					descr_tmp.push_back(descriptors1.rowRange(i+1, descriptors1.rows));
//					descriptors1.release();
//					descr_tmp.copyTo(descriptors1);
//				}
//				i--;
//				continue;
//			}
//
//			keypts2idx.index->radiusSearch(&x2e(0),searchradius,radius_matches,nanoflann::SearchParams(maxDepthSearch));
//			if(radius_matches.empty())
//			{
//				this->leftInlier.push_back(false);
//				second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//				radius_matches.clear();
//				continue;
//			}
//
//			//Get descriptors from found right keypoints
//			for(size_t j = 0; j < radius_matches.size(); j++)
//			{
//				keypR_tmp.push_back(keypR[radius_matches[j].first]);
//			}
//			keypR_tmp1 = keypR_tmp;
//			//extractor->compute(imgs[1],keypR_tmp,descriptors2);
//			matchinglib::getDescriptors(imgs[1], keypR_tmp, GTfilterExtractor, descriptors2, featuretype);
//			if(radius_matches.size() > keypR_tmp.size())
//			{
//				if(keypR_tmp.empty())
//				{
//					for(size_t j = 0; j < radius_matches.size(); j++)
//					{
//						border_dellist.push_back(radius_matches[j].first);
//					}
//					this->leftInlier.push_back(false);
//					second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//					radius_matches.clear();
//					continue;
//				}
//				else
//				{
//					size_t k = 0;
//					for(size_t j = 0; j < radius_matches.size(); j++)
//					{
//						if((keypR_tmp1[j].pt.x == keypR_tmp[k].pt.x) && (keypR_tmp1[j].pt.y == keypR_tmp[k].pt.y))
//						{
//							k++;
//							if(k == keypR_tmp.size())
//								k--;
//						}
//						else
//						{
//							border_dellist.push_back(radius_matches[j].first);
//							border_marklist.push_back(j);
//						}
//					}
//				}
//			}
//			keypR_tmp.clear();
//
//			//Get index of smallest descriptor distance
//			descr_dist1 = getDescriptorDistance(descriptors1.row(i), descriptors2.row(0));
//			for(size_t j = 1; j < (size_t)descriptors2.rows; j++)
//			{
//				descr_dist2 = getDescriptorDistance(descriptors1.row((int)i), descriptors2.row((int)j));
//				if(descr_dist1 > descr_dist2)
//				{
//					if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
//					{
//						minDescrDist = j;
//						for(size_t k = 0; k < border_marklist.size(); k++)
//						{
//							if(minDescrDist >= border_marklist[k])
//								minDescrDist++;
//							else
//								break;
//						}
//					}
//					else
//					{
//						minDescrDist = j;
//					}
//					descr_dist1 = descr_dist2;
//				}
//			}
//			if((descr_dist1 > 160) && (descriptors1.type() == CV_8U))
//			{
//				this->keypL.erase(keypL.begin() + i);
//				if(i == 0)
//				{
//					descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
//				}
//				else
//				{
//					Mat descr_tmp = descriptors1.rowRange(0, i);
//					descr_tmp.push_back(descriptors1.rowRange(i+1, descriptors1.rows));
//					descriptors1.release();
//					descr_tmp.copyTo(descriptors1);
//				}
//				i--;
//				continue;
//			}
//
//			this->leftInlier.push_back(true);
//			nearest_dist.push_back(radius_matches[minDescrDist]);
//			if(radius_matches.size() > 1)
//			{
//				if(minDescrDist == 0)
//				{
//					second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//					second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin()+1, radius_matches.end());
//				}
//				else
//				{
//					second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//					second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin(), radius_matches.begin() + minDescrDist);
//					second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin() + minDescrDist + 1, radius_matches.end());
//				}
//			}
//			else
//			{
//				second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//			}
//			radius_matches.clear();
//		}
//	}
//	else
//	{
//		//Search for ground truth matches using a homography
//		for(unsigned int i = 0;i<keypL.size();i++)
//		{
//			Mat descriptors2;
//			float descr_dist1, descr_dist2; //Descriptor distance of found right keypoints
//			size_t minDescrDist = 0; 
//			vector<size_t> border_marklist;//Number of points that are too near to the image border
//			float hlp = (float)homoGT.at<double>(2,0) * keypL[i].pt.x + (float)homoGT.at<double>(2,1) * keypL[i].pt.y + (float)homoGT.at<double>(2,2);;
//			x2e(0) = (float)homoGT.at<double>(0,0) * keypL[i].pt.x + (float)homoGT.at<double>(0,1) * keypL[i].pt.y + (float)homoGT.at<double>(0,2);
//			x2e(1) = (float)homoGT.at<double>(1,0) * keypL[i].pt.x + (float)homoGT.at<double>(1,1) * keypL[i].pt.y + (float)homoGT.at<double>(1,2);
//			x2e(0) /= hlp;
//			x2e(1) /= hlp;
//			keypts2idx.index->radiusSearch(&x2e(0),searchradius,radius_matches,nanoflann::SearchParams(maxDepthSearch));
//			if(radius_matches.empty())
//			{
//				this->leftInlier.push_back(false);
//				second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//				radius_matches.clear();
//				continue;
//			}
//			//Get descriptors from found right keypoints
//			for(size_t j = 0; j < radius_matches.size(); j++)
//			{
//				keypR_tmp.push_back(keypR[radius_matches[j].first]);
//			}
//			keypR_tmp1 = keypR_tmp;
//			//extractor->compute(imgs[1],keypR_tmp,descriptors2);
//			matchinglib::getDescriptors(imgs[1], keypR_tmp, GTfilterExtractor, descriptors2, featuretype);
//			if(radius_matches.size() > keypR_tmp.size())
//			{
//				if(keypR_tmp.empty())
//				{
//					for(size_t j = 0; j < radius_matches.size(); j++)
//					{
//						border_dellist.push_back(radius_matches[j].first);
//					}
//					this->leftInlier.push_back(false);
//					second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//					radius_matches.clear();
//					continue;
//				}
//				else
//				{
//					size_t k = 0;
//					for(size_t j = 0; j < radius_matches.size(); j++)
//					{
//						if((keypR_tmp1[j].pt.x == keypR_tmp[k].pt.x) && (keypR_tmp1[j].pt.y == keypR_tmp[k].pt.y))
//						{
//							k++;
//							if(k == keypR_tmp.size())
//								k--;
//						}
//						else
//						{
//							border_dellist.push_back(radius_matches[j].first);
//							border_marklist.push_back(j);
//						}
//					}
//				}
//			}
//			keypR_tmp.clear();
//
//			//Get index of smallest descriptor distance
//			descr_dist1 = getDescriptorDistance(descriptors1.row(i), descriptors2.row(0));
//			for(size_t j = 1; j < (size_t)descriptors2.rows; j++)
//			{
//				descr_dist2 = getDescriptorDistance(descriptors1.row((int)i), descriptors2.row((int)j));
//				if(descr_dist1 > descr_dist2)
//				{
//					if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
//					{
//						minDescrDist = j;
//						for(size_t k = 0; k < border_marklist.size(); k++)
//						{
//							if(minDescrDist >= border_marklist[k])
//								minDescrDist++;
//							else
//								break;
//						}
//					}
//					else
//					{
//						minDescrDist = j;
//					}
//					descr_dist1 = descr_dist2;
//				}
//			}
//			if((descr_dist1 > 160) && (descriptors1.type() == CV_8U))
//			{
//				this->keypL.erase(keypL.begin() + i);
//				if(i == 0)
//				{
//					descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
//				}
//				else
//				{
//					Mat descr_tmp = descriptors1.rowRange(0, i);
//					descr_tmp.push_back(descriptors1.rowRange(i+1, descriptors1.rows));
//					descriptors1.release();
//					descr_tmp.copyTo(descriptors1);
//				}
//				i--;
//				continue;
//			}
//
//			this->leftInlier.push_back(true);
//			nearest_dist.push_back(radius_matches[minDescrDist]);
//			if(radius_matches.size() > 1)
//			{
//				if(minDescrDist == 0)
//				{
//					second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//					second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin()+1, radius_matches.end());
//				}
//				else
//				{
//					second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//					second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin(), radius_matches.begin() + minDescrDist);
//					second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin() + minDescrDist + 1, radius_matches.end());
//				}
//			}
//			else
//			{
//				second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//			}
//			radius_matches.clear();
//		}
//	}
//
//#if showfeatures
//	//cv::Mat img1c;
//	img1c.release();
//	drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//	imshow("Keypoints 1 after min. Similarity filtering", img1c );
//	img1c.release();
//	drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//	imshow("Keypoints 2", img1c );
//	cv::waitKey(0);
//#endif
//
//	//Generate flow from right to left image using neighbor interpolation
//	vector<Mat> channels21;
//	if(flowGtIsUsed)
//	{
//		int x, y;
//		float nfx, nfy;
//		channels21.push_back(Mat(flowGT.rows, flowGT.cols, CV_32FC1, -1.0));
//		channels21.push_back(Mat(flowGT.rows, flowGT.cols, CV_32FC1, -1.0));
//		channels21.push_back(Mat(flowGT.rows, flowGT.cols, CV_32FC1, -1.0));
//		for(int v = 0; v < flowGT.rows; v++)
//		{
//			for(int u = 0; u < flowGT.cols; u++)
//			{
//				if(channelsFlow[2].at<float>(v,u))
//				{
//					nfx = channelsFlow[0].at<float>(v, u);
//					nfy = channelsFlow[1].at<float>(v, u);
//					x = (int)floor((float)u + nfx + 0.5);
//					y = (int)floor((float)v + nfy + 0.5);
//					if((x > 0) && (x < w) && (y > 0) && (y < h))
//					{
//						channels21[0].at<float>((int)y, (int)x) = -1.0f * nfx;
//						channels21[1].at<float>((int)y, (int)x) = -1.0f * nfy;
//						channels21[2].at<float>((int)y, (int)x) = channelsFlow[2].at<float>(v, u);
//					}
//				}
//			}
//		}
//		//Interpolate missing values using the median from the neighborhood
//		for(int v = 0; v < channels21[0].rows; v++)
//		{
//			for(int u = 0; u < channels21[0].cols; u++)
//			{
//				if(channels21[2].at<float>(v,u) == -1.0)
//				{
//					vector<float> mx, my;
//					float mv = 0;
//					for(int dx = -1; dx < 2; dx++)
//					{
//						for(int dy = -1; dy < 2; dy++)
//						{
//							x = u + dx;
//							y = v + dy;
//							if((x > 0) && (x < w) && (y > 0) && (y < h))
//							{
//								if(channels21[2].at<float>(y, x) > 0)
//								{
//									mx.push_back(channels21[0].at<float>(y, x));
//									my.push_back(channels21[1].at<float>(y, x));
//									mv += channels21[2].at<float>(y, x);
//								}
//							}
//						}
//					}
//					if(mx.size() > 2)
//					{
//						sort(mx.begin(), mx.end());
//						sort(my.begin(), my.end());
//						mv = floor(mv / (float)mx.size() + 0.5f);
//						if(mx.size() % 2)
//						{
//							channels21[0].at<float>(v,u) = mx[(mx.size()-1)/2];
//							channels21[1].at<float>(v,u) = my[(my.size()-1)/2];
//						}
//						else
//						{
//							channels21[0].at<float>(v,u) = (float)(mx[mx.size() / 2] + mx[mx.size() / 2 - 1]) / 2.0f;
//							channels21[1].at<float>(v,u) = (float)(my[my.size() / 2] + my[my.size() / 2 - 1]) / 2.0f;
//						}
//						channels21[2].at<float>(v,u) = mv;
//					}
//					else
//					{
//						channels21[0].at<float>(v,u) = 0;
//						channels21[1].at<float>(v,u) = 0;
//						channels21[2].at<float>(v,u) = 0;
//					}
//				}
//			}
//		}
//	}
//
//	int roundcounter = 0;
//	bool additionalRound = true;
//	bool initround = false;
////trySecondRound:
//	do
//	{
//		if(((additionalRound == true) || (roundcounter < 2)) && initround)
//		{
//			roundcounter++;
//			additionalRound = false;
//			//cout << "Starting next round" << endl;
//			//goto trySecondRound;
//		}
//		initround = true;
//		//Remove matches that match the same keypoints in the right image
//		if(roundcounter > 0)
//		{
//			size_t k = 0, k1;
//			int i = 0;
//			vector<pair<size_t,size_t>> dellist;
//			while(i < (int)this->leftInlier.size())
//			{
//				if(this->leftInlier[i])
//				{
//					k1 = k;
//					for(size_t j = (size_t)i+1; j < this->leftInlier.size(); j++)
//					{
//						if(this->leftInlier[j])
//						{
//							k1++;
//							if(nearest_dist[k].first == nearest_dist[k1].first)
//							{
//								dellist.push_back(make_pair(j, k1));
//							}
//						}
//					}
//					if(!dellist.empty())
//					{
//						additionalRound = true;
//						if(dellist.size() > 1)
//						{
//							sort(dellist.begin(), dellist.end(),
//								[](pair<size_t,size_t> first, pair<size_t,size_t> second){return first.first > second.first;});
//						}
//						for(k1 = 0; k1 < dellist.size(); k1++)
//						{
//							keypL.erase(keypL.begin() + dellist[k1].first);
//							this->leftInlier.erase(this->leftInlier.begin() + dellist[k1].first);
//							nearest_dist.erase(nearest_dist.begin() + dellist[k1].second);
//							second_nearest_dist_vec.erase(second_nearest_dist_vec.begin() + dellist[k1].first);
//						}
//						keypL.erase(keypL.begin() + i);
//						this->leftInlier.erase(this->leftInlier.begin() + i);
//						nearest_dist.erase(nearest_dist.begin() + k);
//						second_nearest_dist_vec.erase(second_nearest_dist_vec.begin() + i);
//						i--;
//					}
//					else
//					{
//						k++;
//					}
//					dellist.clear();
//				}
//				i++;
//			}
//		}
//
//		//Check for better matches of the second nearest matches
//		eigkeypts2.resize(keypL.size(),2);
//		for(unsigned int i = 0;i<keypL.size();i++)
//		{
//			eigkeypts2(i,0) = keypL[i].pt.x;
//			eigkeypts2(i,1) = keypL[i].pt.y;
//		}
//		KDTree_D2float keypts2idx2(2,eigkeypts2,maxLeafNum);
//		keypts2idx2.index->buildIndex();
//		float searchradius1;
//		if(flowGtIsUsed)
//		{
//			searchradius1 = std::sqrt((float)searchradius);
//			searchradius1 += 0.5; //Compensate for max. error during interpolation of channels21
//			searchradius1 *= searchradius1;
//		}
//		else
//		{
//			searchradius1 = searchradius;
//		}
//		vector<size_t> delInvalR, delNoCorrL, delCorrR;
//		typedef struct match2QPar{
//			size_t indexL;
//			size_t indexR;
//			size_t indexMarkInlR;
//			float similarity;
//			float similMarkInl;
//			bool isQestInl;
//		}match2QPar;
//		if(flowGtIsUsed)
//		{
//			for(size_t i = 0; i < second_nearest_dist_vec.size(); i++)
//			{
//				if(second_nearest_dist_vec[i].empty())
//					continue;
//
//				vector<match2QPar> m2qps;
//				for(size_t j = 0; j < second_nearest_dist_vec[i].size(); j++)
//				{
//					Mat descriptorsL;
//					size_t borderidx = 0;
//					size_t idxm = second_nearest_dist_vec[i][j].first;
//					vector<size_t> border_marklist;//Number of points that are too near to the image border
//					cv::Point2i hlp;
//					hlp.x = (int)floor(keypR[idxm].pt.x + 0.5f); //Round to nearest integer
//					hlp.y = (int)floor(keypR[idxm].pt.y + 0.5f); //Round to nearest integer
//					if(channels21[2].at<float>(hlp.y, hlp.x) >= 1.0)
//					{
//						x2e(0) = keypR[idxm].pt.x + channels21[0].at<float>(hlp.y, hlp.x);
//						x2e(1) = keypR[idxm].pt.y + channels21[1].at<float>(hlp.y, hlp.x);
//					}
//					//else if(channels21[2].at<float>(hlp.y, hlp.x) > 1.0) //Check if the filled flow (with median interpolation) is near a border with invalid flow -> if yes, reject the keypoint
//					//{
//					//	int dx;
//					//	for(dx = -5; dx < 6; dx++)
//					//	{
//					//		int dy;
//					//		for(dy = -5; dy < 6; dy++)
//					//		{
//					//			xd = hlp.x + dx;
//					//			yd = hlp.y + dy;
//					//			if((xd > 0) && (xd < (int)w) && (yd > 0) && (yd < (int)h))
//					//			{
//					//				if(channels21[2].at<float>(yd, xd) == 0)
//					//					break;
//					//			}
//					//			else
//					//			{
//					//				continue;
//					//			}
//					//		}
//					//		if(dy < 6)
//					//			break;
//					//	}
//					//	if(dx < 6)
//					//	{
//					//		delInvalR.push_back(i);
//					//		continue;
//					//	}
//					//	else
//					//	{
//					//		x2e(0) = keypR[idxm].pt.x + channels21[0].at<float>(hlp.y, hlp.x);
//					//		x2e(1) = keypR[idxm].pt.y + channels21[1].at<float>(hlp.y, hlp.x);
//					//	}
//					//}
//					else
//					{
//						delInvalR.push_back(idxm);
//						continue;
//					}
//
//					keypts2idx2.index->radiusSearch(&x2e(0),searchradius1,radius_matches,nanoflann::SearchParams(maxDepthSearch));
//					if(radius_matches.empty())
//					{
//						radius_matches.clear();
//						continue;
//					}
//
//					//Get descriptors from found left keypoints
//					for(size_t k = 0; k < radius_matches.size(); k++)
//					{
//						keypL_tmp.push_back(keypL[radius_matches[k].first]);
//					}
//					keypL_tmp1 = keypL_tmp;
//					//extractor->compute(imgs[0],keypL_tmp,descriptorsL);
//					matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL, featuretype);
//					if(radius_matches.size() > keypL_tmp.size())
//					{
//						if(keypL_tmp.empty())
//						{
//							for(size_t k = 0; k < radius_matches.size(); k++)
//							{
//								border_dellistL.push_back(radius_matches[k].first);
//							}
//							radius_matches.clear();
//							continue;
//						}
//						else
//						{
//							size_t k = 0;
//							for(size_t j1 = 0; j1 < radius_matches.size(); j1++)
//							{
//								if((keypL_tmp1[j1].pt.x == keypL_tmp[k].pt.x) && (keypL_tmp1[j1].pt.y == keypL_tmp[k].pt.y))
//								{
//									k++;
//									if(k == keypL_tmp.size())
//										k--;
//								}
//								else
//								{
//									border_dellistL.push_back(radius_matches[j1].first);
//									border_marklist.push_back(j1);
//								}
//							}
//						}
//					}
//					keypL_tmp.clear();
//
//					//Get index, descriptor distance, distance to GT, left index and right index for every found keypoint
//					for(size_t j1 = 0; j1 < (size_t)descriptorsL.rows; j1++)
//					{
//						match2QPar hlpq;
//						hlpq.similarity = getDescriptorDistance(descriptors22nd.row((int)idxm), descriptorsL.row((int)j1)); 
//						if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
//						{
//							borderidx = j1;
//							for(size_t k = 0; k < border_marklist.size(); k++)
//							{
//								if(borderidx >= border_marklist[k])
//									borderidx++;
//								else
//									break;
//							}
//						}
//						else
//						{
//							borderidx = j1;
//						}
//						hlpq.indexL = radius_matches[borderidx].first;
//						hlpq.indexR = idxm;
//						hlpq.similMarkInl = FLT_MAX;
//						hlpq.isQestInl = false;
//						m2qps.push_back(hlpq);
//					}
//				}
//				if(m2qps.empty())
//				{
//					continue;
//				}
//				{
//					match2QPar hlpq;
//					Mat descriptorsL;
//					size_t k3 = 0;
//					for(size_t k1 = 0; k1 <= i; k1++)
//					{
//						if(this->leftInlier[k1])
//							k3++;
//					}
//					hlpq.indexL = i;
//					hlpq.indexR = nearest_dist[k3 - 1].first;
//					keypL_tmp.push_back(keypL[i]);
//					//extractor->compute(imgs[0],keypL_tmp,descriptorsL);
//					matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL, featuretype);
//					keypL_tmp.clear();
//					hlpq.similarity = getDescriptorDistance(descriptors22nd.row((int)hlpq.indexR), descriptorsL);
//					hlpq.similMarkInl = hlpq.similarity;
//					hlpq.indexMarkInlR = hlpq.indexR;
//					hlpq.isQestInl = true;
//					m2qps.push_back(hlpq);
//				}
//				for(int k = (int)m2qps.size() - 2; k >= 0; k--) //use m2qps.size() - 2 to exclude the nearest match added two lines above
//				{
//					size_t idxm = m2qps[k].indexL;
//					if(!this->leftInlier[idxm])//If an outlier was found as inlier
//					{
//						delCorrR.push_back(m2qps[k].indexR);
//						m2qps.erase(m2qps.begin() + k);
//						continue;
//					}
//					//Generate the indexes of the already found nearest matches
//					if(idxm == i)
//					{
//						m2qps[k].indexMarkInlR = m2qps.back().indexMarkInlR;
//						m2qps[k].similMarkInl = m2qps.back().similMarkInl;
//					}
//					else
//					{
//						size_t k2 = 0;
//						for(size_t k1 = 0; k1 <= idxm; k1++)
//						{
//							if(this->leftInlier[k1])
//								k2++;
//						}				
//						m2qps[k].indexMarkInlR = nearest_dist[k2-1].first;			
//						//Calculate the similarity to the already found nearest matches
//						keypL_tmp.push_back(keypL[idxm]);
//						//extractor->compute(imgs[0],keypL_tmp,descriptorsL);
//						matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL, featuretype);
//						keypL_tmp.clear();
//						m2qps[k].similMarkInl = getDescriptorDistance(descriptors22nd.row((int)m2qps[k].indexMarkInlR), descriptorsL);
//					}
//				}
//				{
//					int cntidx = 0;
//					int corrMatchIdx = INT_MAX;
//					bool delLR = false;
//					for(int k = 0; k < (int)m2qps.size() - 1; k++) //The ckeck of the last element can be neglected sine it is the nearest match (and there is only one index of this keypoint)
//					{
//						if(m2qps[k].indexMarkInlR == m2qps[k].indexR)
//						{
//							corrMatchIdx = k;
//							cntidx++;
//							if((m2qps[k].indexR != m2qps[k+1].indexR) && (cntidx > 0))
//							{
//								cntidx--;
//								if(delLR)
//								{
//									goto posdelLRf;
//								}
//								for(int k1 = k - cntidx; k1 < k; k1++)
//								{
//									if(m2qps[k1].similarity < 1.5 * m2qps[corrMatchIdx].similMarkInl)
//									{
//										delLR = true;
//										break;
//									}
//								}
//								goto posdelLRf;
//							}
//							continue;
//						}
//						if(m2qps[k].similarity < 1.5 * m2qps[k].similMarkInl)
//						{
//							delLR = true;
//						}
//						if(m2qps[k].indexR == m2qps[k+1].indexR)
//						{
//							cntidx++;						
//						}
//						else
//						{
//							if((corrMatchIdx < INT_MAX) && !delLR)
//							{
//								for(int k1 = k - cntidx; k1 <= k; k1++)
//								{
//									if(m2qps[k1].indexMarkInlR != m2qps[k1].indexR)
//									{
//										if(m2qps[k1].similarity < 1.5 * m2qps[corrMatchIdx].similMarkInl)
//										{
//											delLR = true;
//											break;
//										}
//									}
//								}
//							}
//
//							if(corrMatchIdx == INT_MAX)
//							{
//								if(m2qps[k].indexL != i)//if the forward-backward search does not result in the same correspondence, delete the right keypoint
//								{
//									delCorrR.push_back(m2qps[k].indexR);
//									m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
//									k -= cntidx;
//									delLR = false;
//								}
//								else if(delLR)
//								{
//									delCorrR.push_back(m2qps[k].indexR);
//									delNoCorrL.push_back(i);
//									m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
//									m2qps.pop_back();
//									k -= cntidx;
//									delLR = false;
//								}
//							}
//							posdelLRf:
//							if(delLR)
//							{
//								delCorrR.push_back(m2qps[k].indexR);
//								delNoCorrL.push_back(m2qps[corrMatchIdx].indexL);
//								m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
//								k -= cntidx;
//							}
//							corrMatchIdx = INT_MAX;
//							delLR = false;
//							cntidx = 0;
//						}
//					}
//				}
//				for(int k = (int)m2qps.size() - 1; k >= 0; k--)
//				{
//					size_t idxm = m2qps[k].indexL;
//					if((i == idxm) && !m2qps[k].isQestInl) //If the left match corresponds to the already found nearest match
//					{
//						m2qps.erase(m2qps.begin() + k);
//						continue;
//					}
//					//Remove entries of m2qps for which their found match is not a already found nearest match
//					if(m2qps[k].indexMarkInlR != m2qps[k].indexR)
//					{
//						m2qps.erase(m2qps.begin() + k);
//					}
//				}
//				if(m2qps.size() > 1)
//				{
//					float simith, smallestSimi;
//					//Get the smallest similarity
//					smallestSimi = m2qps[0].similarity;
//					for(size_t k = 1; k < m2qps.size(); k++)
//					{
//						if(m2qps[k].similarity < smallestSimi)
//							smallestSimi = m2qps[k].similarity;
//					}
//					simith = 1.25f * smallestSimi; //Generate a threshold to delete matches with a too large similarity
//					for(int k = (int)m2qps.size() - 1; k >= 0; k--)
//					{
//						if(m2qps[k].similarity > simith)
//						{
//							delNoCorrL.push_back(m2qps[k].indexL);
//						}
//					}
//				}
//			}
//		}
//		else
//		{
//			Mat H1 = this->homoGT.inv();
//			for(size_t i = 0; i < second_nearest_dist_vec.size(); i++)
//			{
//				if(second_nearest_dist_vec[i].empty())
//					continue;
//
//				vector<match2QPar> m2qps;
//				for(size_t j = 0; j < second_nearest_dist_vec[i].size(); j++)
//				{
//					Mat descriptorsL;
//					size_t borderidx = 0;
//					size_t idxm = second_nearest_dist_vec[i][j].first;
//					vector<size_t> border_marklist;//Number of points that are too near to the image border
//					float hlp = (float)H1.at<double>(2,0) * keypR[idxm].pt.x + (float)H1.at<double>(2,1) * keypR[idxm].pt.y + (float)H1.at<double>(2,2);;
//					x2e(0) = (float)H1.at<double>(0,0) * keypR[idxm].pt.x + (float)H1.at<double>(0,1) * keypR[idxm].pt.y + (float)H1.at<double>(0,2);
//					x2e(1) = (float)H1.at<double>(1,0) * keypR[idxm].pt.x + (float)H1.at<double>(1,1) * keypR[idxm].pt.y + (float)H1.at<double>(1,2);
//					x2e(0) /= hlp;
//					x2e(1) /= hlp;
//				
//					keypts2idx2.index->radiusSearch(&x2e(0),searchradius1,radius_matches,nanoflann::SearchParams(maxDepthSearch));
//					if(radius_matches.empty())
//					{
//						radius_matches.clear();
//						continue;
//					}
//
//					//Get descriptors from found left keypoints
//					for(size_t k = 0; k < radius_matches.size(); k++)
//					{
//						keypL_tmp.push_back(keypL[radius_matches[k].first]);
//					}
//					keypL_tmp1 = keypL_tmp;
//					//extractor->compute(imgs[0],keypL_tmp,descriptorsL);
//					matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL, featuretype);
//					if(radius_matches.size() > keypL_tmp.size())
//					{
//						if(keypL_tmp.empty())
//						{
//							for(size_t k = 0; k < radius_matches.size(); k++)
//							{
//								border_dellistL.push_back(radius_matches[k].first);
//							}
//							radius_matches.clear();
//							continue;
//						}
//						else
//						{
//							size_t k = 0;
//							for(size_t j1 = 0; j1 < radius_matches.size(); j1++)
//							{
//								if((keypL_tmp1[j1].pt.x == keypL_tmp[k].pt.x) && (keypL_tmp1[j1].pt.y == keypL_tmp[k].pt.y))
//								{
//									k++;
//									if(k == keypL_tmp.size())
//										k--;
//								}
//								else
//								{
//									border_dellistL.push_back(radius_matches[j1].first);
//									border_marklist.push_back(j1);
//								}
//							}
//						}
//					}
//					keypL_tmp.clear();
//
//					//Get index, descriptor distance, distance to GT, left index and right index for every found keypoint
//					for(size_t j1 = 0; j1 < (size_t)descriptorsL.rows; j1++)
//					{
//						match2QPar hlpq;
//						hlpq.similarity = getDescriptorDistance(descriptors22nd.row((int)idxm), descriptorsL.row((int)j1)); 
//						if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
//						{
//							borderidx = j1;
//							for(size_t k = 0; k < border_marklist.size(); k++)
//							{
//								if(borderidx >= border_marklist[k])
//									borderidx++;
//								else
//									break;
//							}
//						}
//						else
//						{
//							borderidx = j1;
//						}
//						hlpq.indexL = radius_matches[borderidx].first;
//						hlpq.indexR = idxm;
//						hlpq.similMarkInl = FLT_MAX;
//						hlpq.isQestInl = false;
//						m2qps.push_back(hlpq);
//					}
//				}
//				if(m2qps.empty())
//				{
//					continue;
//				}
//				{
//					match2QPar hlpq;
//					Mat descriptorsL;
//					size_t k3 = 0;
//					for(size_t k1 = 0; k1 <= i; k1++)
//					{
//						if(this->leftInlier[k1])
//							k3++;
//					}
//					hlpq.indexL = i;
//					hlpq.indexR = nearest_dist[k3 - 1].first;
//					keypL_tmp.push_back(keypL[i]);
//					//extractor->compute(imgs[0],keypL_tmp,descriptorsL);
//					matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL, featuretype);
//					keypL_tmp.clear();
//					hlpq.similarity = getDescriptorDistance(descriptors22nd.row((int)hlpq.indexR), descriptorsL);
//					hlpq.similMarkInl = hlpq.similarity;
//					hlpq.indexMarkInlR = hlpq.indexR;
//					hlpq.isQestInl = true;
//					m2qps.push_back(hlpq);
//				}
//				for(int k = (int)m2qps.size() - 2; k >= 0; k--) //use m2qps.size() - 2 to exclude the nearest match added two lines above
//				{
//					size_t idxm = m2qps[k].indexL;
//					if(!this->leftInlier[idxm])//If an outlier was found as inlier
//					{
//						delCorrR.push_back(m2qps[k].indexR);
//						m2qps.erase(m2qps.begin() + k);
//						continue;
//					}
//					//Generate the indexes of the already found nearest matches
//					if(idxm == i)
//					{
//						m2qps[k].indexMarkInlR = m2qps.back().indexMarkInlR;
//						m2qps[k].similMarkInl = m2qps.back().similMarkInl;
//					}
//					else
//					{
//						size_t k2 = 0;
//						for(size_t k1 = 0; k1 <= idxm; k1++)
//						{
//							if(this->leftInlier[k1])
//								k2++;
//						}				
//						m2qps[k].indexMarkInlR = nearest_dist[k2-1].first;			
//						//Calculate the similarity to the already found nearest matches
//						keypL_tmp.push_back(keypL[idxm]);
//						//extractor->compute(imgs[0],keypL_tmp,descriptorsL);
//						matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL, featuretype);
//						keypL_tmp.clear();
//						m2qps[k].similMarkInl = getDescriptorDistance(descriptors22nd.row((int)m2qps[k].indexMarkInlR), descriptorsL);
//					}
//				}
//				{
//					int cntidx = 0;
//					int corrMatchIdx = INT_MAX;
//					bool delLR = false;
//					for(int k = 0; k < (int)m2qps.size() - 1; k++) //The ckeck of the last element can be neglected sine it is the nearest match (and there is only one index of this keypoint)
//					{
//						if(m2qps[k].indexMarkInlR == m2qps[k].indexR)
//						{
//							corrMatchIdx = k;
//							cntidx++;
//							if((m2qps[k].indexR != m2qps[k+1].indexR) && (cntidx > 0))
//							{
//								cntidx--;
//								if(delLR)
//								{
//									goto posdelLRh;
//								}
//								for(int k1 = k - cntidx; k1 < k; k1++)
//								{
//									if(m2qps[k1].similarity < 1.5 * m2qps[corrMatchIdx].similMarkInl)
//									{
//										delLR = true;
//										break;
//									}
//								}
//								goto posdelLRh;
//							}
//							continue;
//						}
//						if(m2qps[k].similarity < 1.5 * m2qps[k].similMarkInl)
//						{
//							delLR = true;
//						}
//						if(m2qps[k].indexR == m2qps[k+1].indexR)
//						{
//							cntidx++;						
//						}
//						else
//						{
//							if((corrMatchIdx < INT_MAX) && !delLR)
//							{
//								for(int k1 = k - cntidx; k1 <= k; k1++)
//								{
//									if(m2qps[k1].indexMarkInlR != m2qps[k1].indexR)
//									{
//										if(m2qps[k1].similarity < 1.5 * m2qps[corrMatchIdx].similMarkInl)
//										{
//											delLR = true;
//											break;
//										}
//									}
//								}
//							}
//
//							if(corrMatchIdx == INT_MAX)
//							{
//								if(m2qps[k].indexL != i)
//								{
//									delCorrR.push_back(m2qps[k].indexR);
//									m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
//									k -= cntidx;
//									delLR = false;
//								}
//								else if(delLR)
//								{
//									delCorrR.push_back(m2qps[k].indexR);
//									delNoCorrL.push_back(i);
//									m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
//									m2qps.pop_back();
//									k -= cntidx;
//									delLR = false;
//								}
//							}
//							posdelLRh:
//							if(delLR)
//							{
//								delCorrR.push_back(m2qps[k].indexR);
//								delNoCorrL.push_back(m2qps[corrMatchIdx].indexL);
//								m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
//								k -= cntidx;
//							}
//							corrMatchIdx = INT_MAX;
//							delLR = false;
//							cntidx = 0;
//						}
//					}
//				}
//				for(int k = (int)m2qps.size() - 1; k >= 0; k--)
//				{
//					size_t idxm = m2qps[k].indexL;
//					if((i == idxm) && !m2qps[k].isQestInl) //If the left match corresponds to the already found nearest match
//					{
//						m2qps.erase(m2qps.begin() + k);
//						continue;
//					}
//					//Remove entries of m2qps for which their found match is not a already found nearest match
//					if(m2qps[k].indexMarkInlR != m2qps[k].indexR)
//					{
//						m2qps.erase(m2qps.begin() + k);
//					}
//				}
//				if(m2qps.size() > 1)
//				{
//					float simith, smallestSimi;
//					//Get the smallest similarity
//					smallestSimi = m2qps[0].similarity;
//					for(size_t k = 1; k < m2qps.size(); k++)
//					{
//						if(m2qps[k].similarity < smallestSimi)
//							smallestSimi = m2qps[k].similarity;
//					}
//					simith = 1.25f * smallestSimi;//Generate a threshold to delete matches with a too large similarity
//					for(int k = (int)m2qps.size() - 1; k >= 0; k--)
//					{
//						if(m2qps[k].similarity > simith)
//						{
//							delNoCorrL.push_back(m2qps[k].indexL);
//						}
//					}
//				}
//			}
//		}
//
//		{
//			//Add keypoints for deletion if they are too near at the border
//			vector<size_t> dellist = border_dellist;
//			vector<size_t> dellistL = border_dellistL;
//
//			//Add invalid (due to flow) right keypoints
//			if(!delInvalR.empty())
//				dellist.insert(dellist.end(), delInvalR.begin(), delInvalR.end());
//
//			//Add invalid left keypoints for deletion
//			if(!delNoCorrL.empty())
//				dellistL.insert(dellistL.end(), delNoCorrL.begin(), delNoCorrL.end());
//
//			//Add right keypoints for deletion
//			if(!delCorrR.empty())
//				dellist.insert(dellist.end(), delCorrR.begin(), delCorrR.end());
//
//			if((dellistL.size() > 50) || (dellist.size() > 50))
//				additionalRound = true;
//
//			delInvalR.clear();
//			delNoCorrL.clear();
//			delCorrR.clear();
//			border_dellist.clear();
//			border_dellistL.clear();
//	
//			if(!dellistL.empty())
//			{
//				//Exclude multiple entries in dellistL
//				sort(dellistL.begin(), dellistL.end(),
//						[](size_t first, size_t second){return first > second;});
//				for(int i = 0; i < (int)dellistL.size() - 1; i++)
//				{
//					while(dellistL[i] == dellistL[i+1])
//					{
//						dellistL.erase(dellistL.begin() + i + 1);
//						if(i == (int)dellistL.size() - 1)
//							break;
//					}
//				}
//
//				//Delete left keypoints
//				int k = (int)nearest_dist.size() - 1, i = 0;
//				for(int j = (int)this->leftInlier.size() - 1; j >= 0; j--)
//				{
//					if(dellistL[i] == (int)j)
//					{
//						keypL.erase(keypL.begin() + j);
//						if(leftInlier[j])
//						{
//							nearest_dist.erase(nearest_dist.begin() + k);
//							k--;
//						}
//						leftInlier.erase(leftInlier.begin() + j);
//						i++;
//						if(i == (int)dellistL.size())
//							break;
//					}
//					else if(leftInlier[j])
//					{
//						k--;
//					}
//				}
//			}
//
//			if(!dellist.empty())
//			{
//				//Exclude multiple entries in dellist
//				sort(dellist.begin(), dellist.end(),
//						[](size_t first, size_t second){return first > second;});
//				for(int i = 0; i < (int)dellist.size() - 1; i++)
//				{
//					while(dellist[i] == dellist[i+1])
//					{
//						dellist.erase(dellist.begin() + i + 1);
//						if(i == (int)dellist.size() - 1)
//							break;
//					}
//				}
//
//				//Remove right keypoints listed in dellist
//				sort(dellist.begin(), dellist.end(),
//						[](size_t first, size_t second){return first > second;});
//				for(size_t i = 0; i < dellist.size(); i++)
//				{
//					keypR.erase(keypR.begin() + dellist[i]);
//				}
//			}
//			delNoCorrL.clear();
//			delCorrR.clear();
//		}
//
//#if showfeatures
//		//cv::Mat img1c;
//		img1c.release();
//		drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//		imshow("Keypoints 1 after crosscheck", img1c );
//		img1c.release();
//		drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//		imshow("Keypoints 2", img1c );
//		cv::waitKey(0);
//#endif
//
//		//Search for matching keypoints in the right image as the found indexes arent valid anymore
//		//Recalculate descriptors to exclude descriptors from deleted left keypoints
//		descriptors1.release();
//		//extractor->compute(imgs[0],keypL,descriptors1);
//		matchinglib::getDescriptors(imgs[0], keypL, GTfilterExtractor, descriptors1, featuretype);
//		eigkeypts2.resize(keypR.size(),2);
//		for(unsigned int i = 0;i<keypR.size();i++)
//		{
//			eigkeypts2(i,0) = keypR[i].pt.x;
//			eigkeypts2(i,1) = keypR[i].pt.y;
//		}
//		KDTree_D2float keypts2idx1(2,eigkeypts2,maxLeafNum);
//		keypts2idx1.index->buildIndex();
//
//		//Search for the ground truth matches in the right (second) image by searching for the nearest and second nearest neighbor by a radius search
//		this->leftInlier.clear();
//		nearest_dist.clear();
//		second_nearest_dist_vec.clear();
//		if(flowGtIsUsed)
//		{
//			//int xd, yd;
//			//Search for the ground truth matches using optical flow data
//			for(int i = 0;i<keypL.size();i++)
//			{
//				Mat descriptors2;
//				float descr_dist1, descr_dist2; //Descriptor distance of found right keypoints
//				size_t minDescrDist = 0;
//				vector<size_t> border_marklist;//Number of points that are too near to the image border
//				cv::Point2i hlp;
//				hlp.x = (int)floor(keypL[i].pt.x + 0.5f); //Round to nearest integer
//				hlp.y = (int)floor(keypL[i].pt.y + 0.5f); //Round to nearest integer
//				if(channelsFlow[2].at<float>(hlp.y, hlp.x) >= 1.0)
//				{
//					x2e(0) = keypL[i].pt.x + channelsFlow[0].at<float>(hlp.y, hlp.x);
//					x2e(1) = keypL[i].pt.y + channelsFlow[1].at<float>(hlp.y, hlp.x);
//				}
//				else
//				{
//					keypL.erase(keypL.begin() + i);
//					if(i == 0)
//					{
//						descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
//					}
//					else
//					{
//						Mat descr_tmp = descriptors1.rowRange(0, i);
//						descr_tmp.push_back(descriptors1.rowRange(i+1, descriptors1.rows));
//						descriptors1.release();
//						descr_tmp.copyTo(descriptors1);
//					}
//					i--;
//					continue;
//				}
//
//				keypts2idx1.index->radiusSearch(&x2e(0),searchradius,radius_matches,nanoflann::SearchParams(maxDepthSearch));
//				if(radius_matches.empty())
//				{
//					this->leftInlier.push_back(false);
//					second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//					radius_matches.clear();
//					continue;
//				}
//
//				//Get descriptors from found right keypoints
//				for(size_t j = 0; j < radius_matches.size(); j++)
//				{
//					keypR_tmp.push_back(keypR[radius_matches[j].first]);
//				}
//				//extractor->compute(imgs[1],keypR_tmp,descriptors2);
//				matchinglib::getDescriptors(imgs[1], keypR_tmp, GTfilterExtractor, descriptors2, featuretype);
//				if(radius_matches.size() > keypR_tmp.size())
//				{
//					if(keypR_tmp.empty())
//					{
//						for(size_t j = 0; j < radius_matches.size(); j++)
//						{
//							border_dellist.push_back(radius_matches[j].first);
//						}
//						this->leftInlier.push_back(false);
//						second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//						radius_matches.clear();
//						continue;
//					}
//					else
//					{
//						size_t k = 0;
//						vector<cv::KeyPoint> keypR_tmp1;
//						for(size_t j = 0; j < radius_matches.size(); j++)
//						{
//							keypR_tmp1.push_back(keypR[radius_matches[j].first]);
//						}
//						for(size_t j = 0; j < radius_matches.size(); j++)
//						{
//							if((keypR_tmp1[j].pt.x == keypR_tmp[k].pt.x) && (keypR_tmp1[j].pt.y == keypR_tmp[k].pt.y))
//							{
//								k++;
//								if(k == keypR_tmp.size())
//									k--;
//							}
//							else
//							{
//								border_dellist.push_back(radius_matches[j].first);
//								border_marklist.push_back(j);
//							}
//						}
//					}
//				}
//				keypR_tmp.clear();
//
//				//Get index of smallest descriptor distance
//				descr_dist1 = getDescriptorDistance(descriptors1.row(i), descriptors2.row(0));
//				for(size_t j = 1; j < (size_t)descriptors2.rows; j++)
//				{
//					descr_dist2 = getDescriptorDistance(descriptors1.row(i), descriptors2.row((int)j));
//					if(descr_dist1 > descr_dist2)
//					{
//						if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
//						{
//							minDescrDist = j;
//							for(size_t k = 0; k < border_marklist.size(); k++)
//							{
//								if(minDescrDist >= border_marklist[k])
//									minDescrDist++;
//								else
//									break;
//							}
//						}
//						else
//						{
//							minDescrDist = j;
//						}
//						descr_dist1 = descr_dist2;
//					}
//				}
//				if((descr_dist1 > 160) && (descriptors1.type() == CV_8U))
//				{
//					additionalRound = true;
//					keypL.erase(keypL.begin() + i);
//					if(i == 0)
//					{
//						descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
//					}
//					else
//					{
//						Mat descr_tmp = descriptors1.rowRange(0, i);
//						descr_tmp.push_back(descriptors1.rowRange(i+1, descriptors1.rows));
//						descriptors1.release();
//						descr_tmp.copyTo(descriptors1);
//					}
//					i--;
//					continue;
//				}
//
//				this->leftInlier.push_back(true);
//				nearest_dist.push_back(radius_matches[minDescrDist]);
//				if(radius_matches.size() > 1)
//				{
//					if(minDescrDist == 0)
//					{
//						second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//						second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin()+1, radius_matches.end());
//					}
//					else
//					{
//						second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//						second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin(), radius_matches.begin() + minDescrDist);
//						second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin() + minDescrDist + 1, radius_matches.end());
//					}
//				}
//				else
//				{
//					second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//				}
//				radius_matches.clear();
//			}
//		}
//		else
//		{
//			//Search for ground truth matches using a homography
//			for(unsigned int i = 0;i<keypL.size();i++)
//			{
//				Mat descriptors2;
//				float descr_dist1, descr_dist2; //Descriptor distance of found right keypoints
//				size_t minDescrDist = 0; 
//				vector<size_t> border_marklist;//Number of points that are too near to the image border
//				float hlp = (float)homoGT.at<double>(2,0) * keypL[i].pt.x + (float)homoGT.at<double>(2,1) * keypL[i].pt.y + (float)homoGT.at<double>(2,2);;
//				x2e(0) = (float)homoGT.at<double>(0,0) * keypL[i].pt.x + (float)homoGT.at<double>(0,1) * keypL[i].pt.y + (float)homoGT.at<double>(0,2);
//				x2e(1) = (float)homoGT.at<double>(1,0) * keypL[i].pt.x + (float)homoGT.at<double>(1,1) * keypL[i].pt.y + (float)homoGT.at<double>(1,2);
//				x2e(0) /= hlp;
//				x2e(1) /= hlp;
//				keypts2idx1.index->radiusSearch(&x2e(0),searchradius,radius_matches,nanoflann::SearchParams(maxDepthSearch));
//				if(radius_matches.empty())
//				{
//					this->leftInlier.push_back(false);
//					second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//					radius_matches.clear();
//					continue;
//				}
//				
//				//Get descriptors from found right keypoints
//				for(size_t j = 0; j < radius_matches.size(); j++)
//				{
//					keypR_tmp.push_back(keypR[radius_matches[j].first]);
//				}
//				//extractor->compute(imgs[1],keypR_tmp,descriptors2);
//				matchinglib::getDescriptors(imgs[1], keypR_tmp, GTfilterExtractor, descriptors2, featuretype);
//				if(radius_matches.size() > keypR_tmp.size())
//				{
//					if(keypR_tmp.empty())
//					{
//						for(size_t j = 0; j < radius_matches.size(); j++)
//						{
//							border_dellist.push_back(radius_matches[j].first);
//						}
//						this->leftInlier.push_back(false);
//						second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//						radius_matches.clear();
//						continue;
//					}
//					else
//					{
//						size_t k = 0;
//						vector<cv::KeyPoint> keypR_tmp1;
//						for(size_t j = 0; j < radius_matches.size(); j++)
//						{
//							keypR_tmp1.push_back(keypR[radius_matches[j].first]);
//						}
//						for(size_t j = 0; j < radius_matches.size(); j++)
//						{
//							if((keypR_tmp1[j].pt.x == keypR_tmp[k].pt.x) && (keypR_tmp1[j].pt.y == keypR_tmp[k].pt.y))
//							{
//								k++;
//								if(k == keypR_tmp.size())
//									k--;
//							}
//							else
//							{
//								border_dellist.push_back(radius_matches[j].first);
//								border_marklist.push_back(j);
//							}
//						}
//					}
//				}
//				keypR_tmp.clear();
//
//				//Get index of smallest descriptor distance
//				descr_dist1 = getDescriptorDistance(descriptors1.row(i), descriptors2.row(0));
//				for(size_t j = 1; j < (size_t)descriptors2.rows; j++)
//				{
//					descr_dist2 = getDescriptorDistance(descriptors1.row((int)i), descriptors2.row((int)j));
//					if(descr_dist1 > descr_dist2)
//					{
//						if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
//						{
//							minDescrDist = j;
//							for(size_t k = 0; k < border_marklist.size(); k++)
//							{
//								if(minDescrDist >= border_marklist[k])
//									minDescrDist++;
//								else
//									break;
//							}
//						}
//						else
//						{
//							minDescrDist = j;
//						}
//						descr_dist1 = descr_dist2;
//					}
//				}
//				if((descr_dist1 > 160) && (descriptors1.type() == CV_8U))
//				{
//					additionalRound = true;
//					keypL.erase(keypL.begin() + i);
//					if(i == 0)
//					{
//						descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
//					}
//					else
//					{
//						Mat descr_tmp = descriptors1.rowRange(0, i);
//						descr_tmp.push_back(descriptors1.rowRange(i+1, descriptors1.rows));
//						descriptors1.release();
//						descr_tmp.copyTo(descriptors1);
//					}
//					i--;
//					continue;
//				}
//
//				this->leftInlier.push_back(true);
//				nearest_dist.push_back(radius_matches[minDescrDist]);
//				if(radius_matches.size() > 1)
//				{
//					if(minDescrDist == 0)
//					{
//						second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//						second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin()+1, radius_matches.end());
//					}
//					else
//					{
//						second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//						second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin(), radius_matches.begin() + minDescrDist);
//						second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin() + minDescrDist + 1, radius_matches.end());
//					}
//				}
//				else
//				{
//					second_nearest_dist_vec.push_back(vector<pair<size_t,float>>());
//				}
//				radius_matches.clear();
//			}
//		}
//
//#if showfeatures
//		//cv::Mat img1c;
//		img1c.release();
//		drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//		imshow("Keypoints 1 after final radius search", img1c );
//		img1c.release();
//		drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//		imshow("Keypoints 2", img1c );
//		cv::waitKey(0);
//#endif
//
//	}while(((additionalRound == true) || (roundcounter < 2)));
//	
//
//	//Remove matches that match the same keypoints in the right image
//	if(roundcounter > 0)
//	{
//		size_t k = 0, k1;
//		int i = 0;
//		vector<pair<size_t,size_t>> dellist;
//		while(i < (int)this->leftInlier.size())
//		{
//			if(this->leftInlier[i])
//			{
//				k1 = k;
//				for(size_t j = (size_t)i+1; j < this->leftInlier.size(); j++)
//				{
//					if(this->leftInlier[j])
//					{
//						k1++;
//						if(nearest_dist[k].first == nearest_dist[k1].first)
//						{
//							dellist.push_back(make_pair(j, k1));
//						}
//					}
//				}
//				if(!dellist.empty())
//				{
//					if(dellist.size() > 1)
//					{
//						sort(dellist.begin(), dellist.end(),
//							[](pair<size_t,size_t> first, pair<size_t,size_t> second){return first.first > second.first;});
//					}
//					for(k1 = 0; k1 < dellist.size(); k1++)
//					{
//						keypL.erase(keypL.begin() + dellist[k1].first);
//						this->leftInlier.erase(this->leftInlier.begin() + dellist[k1].first);
//						nearest_dist.erase(nearest_dist.begin() + dellist[k1].second);
//						second_nearest_dist_vec.erase(second_nearest_dist_vec.begin() + dellist[k1].first);
//					}
//					keypL.erase(keypL.begin() + i);
//					this->leftInlier.erase(this->leftInlier.begin() + i);
//					nearest_dist.erase(nearest_dist.begin() + k);
//					second_nearest_dist_vec.erase(second_nearest_dist_vec.begin() + i);
//					i--;
//				}
//				else
//				{
//					k++;
//				}
//				dellist.clear();
//			}
//			i++;
//		}
//	}
//
//	//Store original ground truth matches without a specific inlier ratio
//	{
//		int err = 0;
//		DMatch singleMatch;
//		size_t k = 0;
//		for (size_t j = 0; j < this->leftInlier.size(); j++)
//		{
//			if (this->leftInlier[j])
//			{
//				singleMatch.queryIdx = (int)j;
//				singleMatch.trainIdx = (int)nearest_dist[k].first;
//				singleMatch.distance = nearest_dist[k].second; //Squared distance of the keypoint in the right image to the calculated position from ground truth
//				this->matchesGT.push_back(singleMatch);
//				k++;
//			}
//		}
//
//		this->positivesGT = (double)k;
//		this->negativesGTl = (double)(this->leftInlier.size() - k);
//		this->negativesGTr = (double)(keypR.size() - k);
//		this->inlRatioL = positivesGT / (double)this->leftInlier.size();
//		this->inlRatioR = positivesGT / (double)keypR.size();
//		this->inlRatioO = 2 * positivesGT / (double)(keypR.size() + this->leftInlier.size());
//
//		//Write to disk
//		if (!fileNameImgL_initial.empty() && !imgsPath_initial.empty())
//			err = writeGTMatchesDisk(filenameGT_initial, (k < 15) ? true : false);
//		if (err)
//			cout << "Not possible to write GT to disk!" << endl;
//
//		this->matchesGT.clear();
//
//		if (generateOnlyInitialGTM)
//			return 0;
//	}
//	}
//
//	//Generate a user specific outlier ratio
//	if(specifiedInlRatio > 0)
//	{
//		//Get outliers in both images	
//		size_t leftInlSize = nearest_dist.size();
//		size_t leftOutlSize = this->leftInlier.size() - leftInlSize;
//		size_t rightOutlSize = keypR.size() - leftInlSize;
//		int outlDiff = (int)rightOutlSize - (int)leftOutlSize;
//		int wantedOutl;
//		vector<size_t> outlR, outlL;
//		vector<cv::KeyPoint> keypLOutl_tmp;
//
//		if(rightOutlSize > 0)
//		{
//			vector<pair<size_t,float>> nearest_dist_tmp = nearest_dist;
//			sort(nearest_dist_tmp.begin(), nearest_dist_tmp.end(),
//				[](pair<size_t,float> first, pair<size_t,float> second){return first.first < second.first;});
//		
//			size_t j = 0;
//			for(size_t i = 0; i < keypR.size(); i++)
//			{
//				if(j < leftInlSize)
//				{
//					if(nearest_dist_tmp[j].first == i)
//					{
//						j++;
//					}
//					else
//					{
//						outlR.push_back(i);
//					}
//				}
//				else
//				{
//					outlR.push_back(i);
//				}
//			}
//		}
//		if(leftOutlSize > 0)
//		{
//			for(size_t i = 0; i < leftInlier.size(); i++)
//			{
//				if(!leftInlier[i])
//				{
//					outlL.push_back(i);
//				}
//			}
//		}
//
//		//I required, removes all outliers to get the same number of features for all inlier ratios
//		if(useSameKeypSiVarInl)
//		{
//			//Delete all outliers from the right image
//			for(int i = 0; i < rightOutlSize; i++)
//			{
//				size_t rKeyPIdx = outlR[i];
//				for(size_t k1 = 0; k1 < nearest_dist.size(); k1++)
//				{
//					if(nearest_dist[k1].first > rKeyPIdx)
//					{
//						nearest_dist[k1].first--;
//					}
//				}
//				keypR.erase(keypR.begin() + rKeyPIdx);
//				for(size_t k1 = i + 1; k1 < rightOutlSize; k1++)
//				{
//					if(outlR[k1] > rKeyPIdx)
//					{
//						outlR[k1]--;
//					}
//				}
//			}
//			outlR.clear();
//			rightOutlSize = 0;
//
//			//Delete all outliers from the left image
//			for(int i = 0; i < leftOutlSize; i++)
//			{
//				size_t lKeyPIdx = outlL[i];
//				keypLOutl_tmp.push_back(keypL[lKeyPIdx]);
//				keypL.erase(keypL.begin() + lKeyPIdx);
//				leftInlier.erase(leftInlier.begin() + lKeyPIdx);
//				for(size_t k1 = i + 1; k1 < leftOutlSize; k1++)
//				{
//					if(outlL[k1] > lKeyPIdx)
//					{
//						outlL[k1]--;
//					}
//				}
//			}
//			outlL.clear();
//			leftOutlSize = 0;
//
//			outlDiff = 0;
//		}
//
//		if(outlDiff > 0)
//		{
//			//Delete all outliers from the right image so that it is equal to the number of outliers in the left image
//			for(int i = 0; i < outlDiff; i++)
//			{
//				int delOutlIdx = rand() % rightOutlSize;
//				size_t rKeyPIdx = outlR[delOutlIdx];
//				for(size_t k1 = 0; k1 < nearest_dist.size(); k1++)
//				{
//					if(nearest_dist[k1].first > rKeyPIdx)
//					{
//						nearest_dist[k1].first--;
//					}
//				}
//				keypR.erase(keypR.begin() + rKeyPIdx);
//				for(size_t k1 = 0; k1 < rightOutlSize; k1++)
//				{
//					if(outlR[k1] > rKeyPIdx)
//					{
//						outlR[k1]--;
//					}
//				}
//				outlR.erase(outlR.begin() + delOutlIdx);
//				rightOutlSize--;
//			}
//		}
//		else if(outlDiff < 0)
//		{
//			outlDiff *= -1;
//			//Delete as many outliers from the left image so that it is equal to the number of outliers in the right image
//			for(int i = 0; i < outlDiff; i++)
//			{
//				int delOutlIdx = rand() % leftOutlSize;
//				keypL.erase(keypL.begin() + outlL[delOutlIdx]);
//				leftInlier.erase(leftInlier.begin() + outlL[delOutlIdx]);
//				for(size_t k1 = 0; k1 < leftOutlSize; k1++)
//				{
//					if(outlL[k1] > outlL[delOutlIdx])
//					{
//						outlL[k1]--;
//					}
//				}
//				outlL.erase(outlL.begin() + delOutlIdx);
//				leftOutlSize--;
//			}
//		}
//		//Calculate the number of outliers that should be present in both images
//		wantedOutl = (int)floor((float)leftInlSize / (float)specifiedInlRatio + 0.5f) - (int)leftInlSize;
//
//#if showfeatures
//	//cv::Mat img1c;
//	img1c.release();
//	drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//	imshow("Keypoints 1 after equalization of outliers", img1c );
//	img1c.release();
//	drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//	imshow("Keypoints 2 after equalization of outliers", img1c );
//	cv::waitKey(0);
//#endif
//
//		if(wantedOutl < (int)rightOutlSize)
//		{
//			outlDiff = (int)rightOutlSize - wantedOutl;
//			//Delete as many outliers until the wanted outlier size is reached
//			for(int i = 0; i < outlDiff; i++)
//			{
//				//Delete right outliers
//				int delOutlIdx = rand() % rightOutlSize;
//				size_t rKeyPIdx = outlR[delOutlIdx];
//				for(size_t k1 = 0; k1 < nearest_dist.size(); k1++)
//				{
//					if(nearest_dist[k1].first > rKeyPIdx)
//					{
//						nearest_dist[k1].first--;
//					}
//				}
//				keypR.erase(keypR.begin() + rKeyPIdx);
//				for(size_t k1 = 0; k1 < rightOutlSize; k1++)
//				{
//					if(outlR[k1] > rKeyPIdx)
//					{
//						outlR[k1]--;
//					}
//				}
//				outlR.erase(outlR.begin() + delOutlIdx);
//				rightOutlSize--;
//
//				//Delete left outliers
//				delOutlIdx = rand() % leftOutlSize;
//				keypL.erase(keypL.begin() + outlL[delOutlIdx]);
//				leftInlier.erase(leftInlier.begin() + outlL[delOutlIdx]);
//				for(size_t k1 = 0; k1 < leftOutlSize; k1++)
//				{
//					if(outlL[k1] > outlL[delOutlIdx])
//					{
//						outlL[k1]--;
//					}
//				}
//				outlL.erase(outlL.begin() + delOutlIdx);
//				leftOutlSize--;
//			}
//		}
//		else if(wantedOutl > (int)rightOutlSize)
//		{
//			int inlDiff = (int)floor(((float)leftInlSize - (float)specifiedInlRatio * (float)(leftInlSize + leftOutlSize)) / 2.0f + 0.5f);
//			if(!useSameKeypSiVarInl)
//			{
//				if((2 * inlDiff - (int)leftInlSize) > -15)
//				{
//					cout << "The specified inlier ratio is too low for this data!" << endl;
//					return -1; //Too less features remaining
//				}
//			}
//			else
//			{
//				inlDiff = (int)floor(((float)leftInlSize - (float)specifiedInlRatio * (float)(leftInlSize + leftOutlSize)) + 0.5f);
//				if((inlDiff - (int)leftInlSize) > -15)
//				{
//					cout << "The specified inlier ratio is too low for this data!" << endl;
//					return -1; //Too less features remaining
//				}
//			}
//			
//			//Delete left inliers to rise the number of right ouliers and to lower the number of inliers
//			for(size_t i = 0; i < inlDiff; i++)
//			{
//				int delInlIdx = rand() % leftInlSize;
//				int k = 0;
//				for(size_t j = 0; j < leftInlier.size(); j++)
//				{
//					if(leftInlier[j])
//					{
//						if(k == delInlIdx)
//						{
//							nearest_dist.erase(nearest_dist.begin() + k);
//							keypL.erase(keypL.begin() + j);
//							leftInlier.erase(leftInlier.begin() + j);
//							rightOutlSize++;
//							leftInlSize--;
//							break;
//						}
//						k++;
//					}
//				}
//			}
//			if(useSameKeypSiVarInl)
//			{
//				//Add left outliers to get the same size of keypoints in the left and right image
//				for(size_t i = 0; (i < inlDiff) && (i < keypLOutl_tmp.size()); i++)
//				{
//					keypL.push_back(keypLOutl_tmp[i]);
//					leftInlier.push_back(false);
//					leftOutlSize++;
//				}
//			}
//			if(!useSameKeypSiVarInl)
//			{
//				//Delete right inliers to rise the number of left ouliers and to lower the number of inliers
//				for(size_t i = 0; i < inlDiff; i++)
//				{
//					int delInlIdx = rand() % leftInlSize;
//					int k = 0;
//					for(size_t j = 0; j < leftInlier.size(); j++)
//					{
//						if(leftInlier[j])
//						{
//							if(k == delInlIdx)
//							{
//								size_t rKeyPIdx = nearest_dist[k].first;
//								for(size_t k1 = 0; k1 < nearest_dist.size(); k1++)
//								{
//									if(nearest_dist[k1].first > rKeyPIdx)
//									{
//										nearest_dist[k1].first--;
//									}
//								}
//								keypR.erase(keypR.begin() + rKeyPIdx);
//								nearest_dist.erase(nearest_dist.begin() + k);
//								leftInlier[j] = false;
//								leftOutlSize++;
//								leftInlSize--;
//								break;
//							}
//							k++;
//						}
//					}
//				}
//			}
//		}
//	}
//
//#if showfeatures
//	//cv::Mat img1c;
//	img1c.release();
//	drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//	imshow("Keypoints 1 after inlier ratio filtering", img1c );
//	img1c.release();
//	drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//	imshow("Keypoints 2 after inlier ratio filtering", img1c );
//	cv::waitKey(0);
//#endif
//
//	//Store ground truth matches
//	DMatch singleMatch;
//	size_t k = 0;
//	for(size_t j = 0; j < this->leftInlier.size(); j++)
//	{
//		if(this->leftInlier[j])
//		{
//			singleMatch.queryIdx = (int)j;
//			singleMatch.trainIdx = (int)nearest_dist[k].first;
//			singleMatch.distance = nearest_dist[k].second; //Squared distance of the keypoint in the right image to the calculated position from ground truth
//			this->matchesGT.push_back(singleMatch);
//			k++;
//
//			//{
//			//	int idx = j;//matchesGT[k].queryIdx;
//			//	int idx1 = nearest_dist[k-1].first;//matchesGT[k].trainIdx;
//			//	float x;
//			//	float y;
//			//	if(flowGtIsUsed)
//			//	{
//			//		cv::Point2i hlp;
//			//		hlp.x = (int)floor(keypL[idx].pt.x + 0.5f); //Round to nearest integer
//			//		hlp.y = (int)floor(keypL[idx].pt.y + 0.5f); //Round to nearest integer
//			//		x = keypL[idx].pt.x + channelsFlow[0].at<float>(hlp.y, hlp.x);
//			//		y = keypL[idx].pt.y + channelsFlow[1].at<float>(hlp.y, hlp.x);
//			//	}
//			//	else
//			//	{
//			//		float hlp = (float)homoGT.at<double>(2,0) * keypL[idx].pt.x + (float)homoGT.at<double>(2,1) * keypL[idx].pt.y + (float)homoGT.at<double>(2,2);;
//			//		x = (float)homoGT.at<double>(0,0) * keypL[idx].pt.x + (float)homoGT.at<double>(0,1) * keypL[idx].pt.y + (float)homoGT.at<double>(0,2);
//			//		y = (float)homoGT.at<double>(1,0) * keypL[idx].pt.x + (float)homoGT.at<double>(1,1) * keypL[idx].pt.y + (float)homoGT.at<double>(1,2);
//			//		x /= hlp;
//			//		y /= hlp;
//			//	}
//			//	x = x - keypR[idx1].pt.x;
//			//	y = y - keypR[idx1].pt.y;
//			//	x = std::sqrt(x*x + y*y);
//			//	if(x > (float)usedMatchTH)
//			//	{
//			//		cout << "wrong ground truth, dist: " << x << "idx: " << idx << endl;
//			//	}
//			//}
//		}
//	}
//	if(k < 15)
//		return -1; //Too less features remaining
//	this->positivesGT = (double)k;
//	this->negativesGTl = (double)(this->leftInlier.size() - k);
//	this->negativesGTr = (double)(keypR.size() - k);
//	this->inlRatioL = positivesGT / (double)this->leftInlier.size();
//	this->inlRatioR = positivesGT / (double)keypR.size();
//	this->inlRatioO = 2 * positivesGT / (double)(keypR.size() + this->leftInlier.size());
//
//	//Show ground truth matches
//	/*{
//		Mat img_match;
//		std::vector<cv::KeyPoint> keypL_reduced;//Left keypoints
//		std::vector<cv::KeyPoint> keypR_reduced;//Right keypoints
//		std::vector<cv::DMatch> matches_reduced;
//		std::vector<cv::KeyPoint> keypL_reduced1;//Left keypoints
//		std::vector<cv::KeyPoint> keypR_reduced1;//Right keypoints
//		std::vector<cv::DMatch> matches_reduced1;
//		int j = 0;
//		size_t keepNMatches = 100;
//		size_t keepXthMatch = 1;
//		if(matchesGT.size() > keepNMatches)
//			keepXthMatch = matchesGT.size() / keepNMatches;
//		for (unsigned int i = 0; i < matchesGT.size(); i++)
//		{
//			int idx = matchesGT[i].queryIdx;
//			if(leftInlier[idx])
//			{
//				keypL_reduced.push_back(keypL[idx]);
//				matches_reduced.push_back(matchesGT[i]);
//				matches_reduced.back().queryIdx = j;
//				keypR_reduced.push_back(keypR[matches_reduced.back().trainIdx]);
//				matches_reduced.back().trainIdx = j;
//				j++;
//			}
//		}
//		j = 0;
//		for (unsigned int i = 0; i < matches_reduced.size(); i++)
//		{
//			if((i % (int)keepXthMatch) == 0)
//			{
//				keypL_reduced1.push_back(keypL_reduced[i]);
//				matches_reduced1.push_back(matches_reduced[i]);
//				matches_reduced1.back().queryIdx = j;
//				keypR_reduced1.push_back(keypR_reduced[i]);
//				matches_reduced1.back().trainIdx = j;
//				j++;
//			}
//		}
//		drawMatches(imgs[0], keypL_reduced1, imgs[1], keypR_reduced1, matches_reduced1, img_match);
//		imshow("Ground truth matches", img_match);
//		waitKey(0);
//	}*/
//
//	return 0;
//}
//
///* Extracts keypoints & their descriptors from both images, generates ground truth matches and
// * filters the keypoints.
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Error creating feature detector
// *								-2:		  Too less features detected
// *								-3:		  Too less features remaining after filtering
// *								-4:		  Cannot create descriptor extractor
// *								-5:		  Further precess should be aborted as only the time for descriptor calculations is measured
// */
//int baseMatcher::getValidFeaturesDescriptors()
//{
//	int err;
//	int cyklesTMd = 20;
//	//Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(descriptortype);
//	//if(extractor.empty())
//	//{
//	//	cout << "Cannot create descriptor extractor!" << endl;
//	//	return -4; //Cannot create descriptor extractor
//	//}
//
//	err = detectFeatures();
//	if(err)
//	{
//		tf = -1.0;
//		return err;
//	}
//
//	err = checkForGT();
//	//err = filterInitFeaturesGT();
//	if(err)
//		return err;//-2;
//
//	double td_tmp;
//
//	td = DBL_MAX;
//	if(!measureTd)
//		cyklesTMd = 1;
//
//	for(int i = 0; i < cyklesTMd; i++)
//	{
//		//Clear variables
//		descriptorsL.release();
//		descriptorsR.release();
//
//		//Get descriptors
//		td_tmp = (double)getTickCount(); //Start time measurement
//		/*extractor->compute(imgs[0],keypL,descriptorsL);
//		extractor->compute(imgs[1],keypR,descriptorsR);*/
//		if(matchinglib::getDescriptors(imgs[0], keypL, descriptortype, descriptorsL, featuretype) != 0)
//			return -4;
//		if (matchinglib::getDescriptors(imgs[1], keypR, descriptortype, descriptorsR, featuretype) != 0)
//			return -4;
//		td_tmp = 1000 * ((double)getTickCount() - td_tmp) / getTickFrequency(); //End time measurement
//
//		if(td_tmp < td)
//			td = td_tmp;
//	}
//	tmeanD = td / (double)(keypL.size() + keypR.size());
//	cout << "Descriptor extraction time (ms): " << td << endl;
//	if (measureTd)
//		return -5;
//
//	return 0;
//}
//
///* Checks the validity of the found matches and calculates the quality parameters
// *
// * bool refinedMatches			Input  -> If true, the evaluation is done on the refined matches.
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Sum of matches is wrong
// */
//int baseMatcher::evalMatches(bool refinedMatches)
//{
//	vector<Mat> channelsFlow(3);
//	std::vector<bool> falseNegMatches_tmp(leftInlier.size(), false);
//	std::vector<bool> falsePosMatches_tmp(leftInlier.size(), false);
//	std::vector<bool> truePosMatches_tmp(leftInlier.size(), false);
//	std::vector<cv::DMatch> matches_tmp;
//	matchQualParams qp_tmp;
//	memset(&qp_tmp,0,sizeof(matchQualParams));
//
//	size_t j = 0, k = 0;
//	if(!refinedMatches)
//	{
//		//Sort matches to ensure increasing query indexes
//#if COSTDISTRATIOEVAL
//		std::vector<std::pair<int,cv::DMatch>> matches_idx;
//		std::vector<float> costRatios_tmp;
//		std::vector<float> distRatios_tmp;
//		for(size_t i = 0; i < matches.size(); i++)
//		{
//			matches_idx.push_back(std::make_pair<int,cv::DMatch>(i, matches[i]));
//		}
//		sort(matches_idx.begin(), matches_idx.end(),
//				[](std::pair<int,cv::DMatch> first, std::pair<int,cv::DMatch> second){return first.second.queryIdx < second.second.queryIdx;});
//		matches.clear();
//		for(size_t i = 0; i < matches_idx.size(); i++)
//		{
//			matches.push_back(matches_idx[i].second);
//			costRatios_tmp.push_back(costRatios[matches_idx[i].first]);
//			distRatios_tmp.push_back(distRatios[matches_idx[i].first]);
//		}
//		costRatios = costRatios_tmp;
//		distRatios = distRatios_tmp;
//#else
//		sort(matches.begin(), matches.end(),
//				[](DMatch first, DMatch second){return first.queryIdx < second.queryIdx;});
//#endif
//		matches_tmp = matches;
//	}
//	else
//	{
//#if INITMATCHQUALEVAL_O
//		sort(matchesRefined.begin(), matchesRefined.end(),
//				[](DMatch first, DMatch second){return first.queryIdx < second.queryIdx;});
//#endif
//		matches_tmp = matchesRefined;
//	}
//
//	//Split 3 channel matrix for access
//	if(flowGtIsUsed)
//	{
//		cv::split(flowGT, channelsFlow);
//	}
//
//	//Check for true postives, false positives, true negatives and false negatives
//	for(size_t i = 0; i < leftInlier.size(); i++)
//	{
//		if(leftInlier[i])
//		{
//			/*{
//				int idx = matchesGT[k].queryIdx;
//				int idx1 = matchesGT[k].trainIdx;
//				float hlp = (float)homoGT.at<double>(2,0) * keypL[idx].pt.x + (float)homoGT.at<double>(2,1) * keypL[idx].pt.y + (float)homoGT.at<double>(2,2);;
//				float x = (float)homoGT.at<double>(0,0) * keypL[idx].pt.x + (float)homoGT.at<double>(0,1) * keypL[idx].pt.y + (float)homoGT.at<double>(0,2);
//				float y = (float)homoGT.at<double>(1,0) * keypL[idx].pt.x + (float)homoGT.at<double>(1,1) * keypL[idx].pt.y + (float)homoGT.at<double>(1,2);
//				x /= hlp;
//				y /= hlp;
//				x = x - keypR[idx1].pt.x;
//				y = y - keypR[idx1].pt.y;
//				x = std::sqrt(x*x + y*y);
//				if(x >= (float)usedMatchTH)
//				{
//					cout << "wrong ground truth" << endl;
//				}
//			}*/
//			if(j < matches_tmp.size())
//			{
//				if(matches_tmp[j].queryIdx == (int)i)
//				{				
//					int idx = matches_tmp[j].queryIdx;
//					int idx1 = matches_tmp[j].trainIdx;
//					float x;
//					float y;
//					if(flowGtIsUsed)
//					{
//						cv::Point2i hlp;
//						hlp.x = (int)floor(keypL[idx].pt.x + 0.5f); //Round to nearest integer
//						hlp.y = (int)floor(keypL[idx].pt.y + 0.5f); //Round to nearest integer
//						x = keypL[idx].pt.x + channelsFlow[0].at<float>(hlp.y, hlp.x);
//						y = keypL[idx].pt.y + channelsFlow[1].at<float>(hlp.y, hlp.x);
//					}
//					else
//					{
//						float hlp = (float)homoGT.at<double>(2,0) * keypL[idx].pt.x + (float)homoGT.at<double>(2,1) * keypL[idx].pt.y + (float)homoGT.at<double>(2,2);;
//						x = (float)homoGT.at<double>(0,0) * keypL[idx].pt.x + (float)homoGT.at<double>(0,1) * keypL[idx].pt.y + (float)homoGT.at<double>(0,2);
//						y = (float)homoGT.at<double>(1,0) * keypL[idx].pt.x + (float)homoGT.at<double>(1,1) * keypL[idx].pt.y + (float)homoGT.at<double>(1,2);
//						x /= hlp;
//						y /= hlp;
//					}
//					x = x - keypR[idx1].pt.x;
//					y = y - keypR[idx1].pt.y;
//					x = std::sqrt(x*x + y*y);
//					if(x < (float)usedMatchTH)
//					{
//						qp_tmp.truePos++;
//						truePosMatches_tmp[i] = true;
//#if COSTDISTRATIOEVAL
//						tpfp.push_back(true);
//#endif
//						/*if(matchesGT[k].trainIdx != idx1)
//						{
//							float x_d, y_d;
//							x_d = keypR[idx1].pt.x - keypR[matchesGT[k].trainIdx].pt.x;
//							y_d = keypR[idx1].pt.y - keypR[matchesGT[k].trainIdx].pt.y;
//							x_d = std::sqrt(x_d*x_d + y_d*y_d);
//							cout << "Is no true positive! Distance: " << x_d << endl;
//						}*/
//					}
//					else
//					{
//						falsePosMatches_tmp[i] = true;
//						qp_tmp.falsePos++;
//#if COSTDISTRATIOEVAL
//						tpfp.push_back(false);
//#endif
//					}
//					j++;
//				}
//				else
//				{
//					falseNegMatches_tmp[i] = true;
//					qp_tmp.falseNeg++;
//				}
//			}
//			else
//			{
//				falseNegMatches_tmp[i] = true;
//				qp_tmp.falseNeg++;
//			}
//			k++;
//		}
//		else
//		{
//			if(j < matches_tmp.size())
//			{
//				if(matches_tmp[j].queryIdx == (int)i)
//				{
//					falsePosMatches_tmp[i] = true;
//					qp_tmp.falsePos++;
//					j++;
//#if COSTDISTRATIOEVAL
//					tpfp.push_back(false);
//#endif
//				}
//				else
//				{
//					qp_tmp.trueNeg++;
//				}
//			}
//			else
//			{
//				qp_tmp.trueNeg++;
//			}
//		}
//	}
//	if((qp_tmp.falseNeg + qp_tmp.falsePos + qp_tmp.trueNeg + qp_tmp.truePos) != leftInlier.size())
//	{
//		cout << "Evaluation of quality parameters after matching failed!" << endl;
//		return -1; //Sum of matches is wrong
//	}
//
//	//Precision or positive predictive value ppv=truePos/(truePos+falsePos)
//	qp_tmp.ppv = (qp_tmp.truePos == 0) ? 0:(double)qp_tmp.truePos / (double)(qp_tmp.truePos + qp_tmp.falsePos);
//	//Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)
//	qp_tmp.tpr = (qp_tmp.truePos == 0) ? 0:(double)qp_tmp.truePos / (double)(qp_tmp.truePos + qp_tmp.falseNeg);
//	//Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)
//	qp_tmp.fpr = (qp_tmp.falsePos == 0) ? 0:(double)qp_tmp.falsePos / (double)(qp_tmp.falsePos + qp_tmp.trueNeg);
//	//Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)
//	qp_tmp.acc = ((qp_tmp.truePos + qp_tmp.trueNeg) == 0) ? 0:(double)(qp_tmp.truePos + qp_tmp.trueNeg) / (double)leftInlier.size();
//
//	if(refinedMatches)
//	{
//		falseNegMatchesRef = falseNegMatches_tmp;
//		falsePosMatchesRef = falsePosMatches_tmp;
//		truePosMatchesRef = truePosMatches_tmp;
//		qpr = qp_tmp;
//	}
//	else
//	{
//		falseNegMatches = falseNegMatches_tmp;
//		falsePosMatches = falsePosMatches_tmp;
//		truePosMatches = truePosMatches_tmp;
//		qpm = qp_tmp;
//	}
//
//	return 0;
//}
//
///* Performs the whole process of feature and descriptor extraction, ground truth generation,
// * matching and evaluation of the matching result.
// *
// * double UsrInlRatio			Input  -> From the user specified inlier ratio. If 0, the original
// *										  inlier ratio after ground truth extraction is used [Default = 0]
// * bool _measureT				Input  -> If true [Default = false], time measurement is performed
// *										  independently for feature extraction, descriptor extraction and
// *										  matching
// * unsigned int repEvals		Input  -> Number of times the matching and quality paramter evaluations should
// *										  be performed to get the mean quality paramters for the algorithms
// *										  for the same image pair (quality paramters can change due to random
// *										  initialization of some algorithms)
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Feature detection failed
// *								-2:		  Matching failed
// *								-3:		  Evaluation of matching result failed
// *								-4:		  No matching time available
// */
//int baseMatcher::performMatching(double UsrInlRatio, bool _measureT, unsigned int repEvals)
//{
//	int err;
//	int nr_success = 0;
//	unsigned int repEvals_tmp, repEvals_tmp1;
//	double tm_tmp = DBL_MAX;
//	matchQualParams qpm_avg;
//	memset(&qpm_avg,0,sizeof(matchQualParams));
//#if INITMATCHQUALEVAL_O
//	matchQualParams qpm_avg_init;
//	memset(&qpm_avg_init,0,sizeof(matchQualParams));
//	double initEstiInlRatio_avg = 0;
//#endif
//
//	if(!repEvals)
//		repEvals = 1;
//	else if(repEvals > 1000)
//		repEvals = 1000;
//	repEvals_tmp = repEvals_tmp1 = repEvals;
//
//	specifiedInlRatio = UsrInlRatio;
//	measureT = _measureT;
//
//	//Clear all variables
//	cyklesTM = TIMEMEASITERS;
//	falseNegMatches.clear();
//	falsePosMatches.clear();
//	truePosMatches.clear();
//	falseNegMatchesRef.clear();
//	falsePosMatchesRef.clear();
//	truePosMatchesRef.clear();
//	leftInlier.clear();
//	matchesGT.clear();
//	keypL.clear();
//	keypR.clear();
//	descriptorsL.release();
//	descriptorsR.release();
//	matches.clear();
//	matchesRefined.clear();
//#if COSTDISTRATIOEVAL
//	tpfp.clear();
//	costRatios.clear();
//	distRatios.clear();
//#endif
//
//	err = getValidFeaturesDescriptors();
//	if(err)
//	{
//		cout << "Feature and descriptor extraction failed with code " << err << endl;
//		return -1; //Feature detection failed
//	}
//
//	for(size_t i = 0; i < repEvals_tmp1; i++)
//	{
//		clearMatchingResult();
//		cyklesTM = (TIMEMEASITERS / (int)repEvals) > 0 ? (TIMEMEASITERS / (int)repEvals):1;
//		err = matchValidKeypoints();
//		if(err)
//		{
//			if(i > 2)
//			{
//				if(nr_success == 0)
//				{
//					cout << "Feature matching failed with code " << err << endl;
//					return -2; //Matching failed
//				}
//				else if((float)(i+1)/(float)nr_success > 2.0f)
//				{
//					cout << "Feature matching failed with code " << err << endl;
//					return -2; //Matching failed
//				}
//				else //To get a valid result
//					repEvals_tmp1++;
//			}
//			else if(repEvals <= 2)
//			{
//				cout << "Feature matching failed with code " << err << endl;
//				return -2; //Matching failed
//			}
//			else
//			{
//				repEvals_tmp--;
//			}
//			continue;
//		}
//		else
//		{
//			nr_success++;
//		}
//
//		err = evalMatches(false);
//		if(err)
//		{
//			cout << "Evaluation of matching performance failed!" << err << endl;
//			return -3; //Evaluation of matching result failed
//		}
//		qpm_avg.falseNeg += qpm.falseNeg;
//		qpm_avg.falsePos += qpm.falsePos;
//		qpm_avg.trueNeg += qpm.trueNeg;
//		qpm_avg.truePos += qpm.truePos;
//#if INITMATCHQUALEVAL_O
//		err = evalMatches(true);
//		if(err)
//		{
//			cout << "Evaluation of matching performance failed!" << err << endl;
//			return -3; //Evaluation of matching result failed
//		}
//		qpm_avg_init.falseNeg += qpr.falseNeg;
//		qpm_avg_init.falsePos += qpr.falsePos;
//		qpm_avg_init.trueNeg += qpr.trueNeg;
//		qpm_avg_init.truePos += qpr.truePos;
//		initEstiInlRatio_avg += initEstiInlRatio;
//#endif
//		if(tm < tm_tmp)
//			tm_tmp = tm;
//	}
//	qpm_avg.falseNeg /= repEvals_tmp;
//	qpm_avg.falsePos /= repEvals_tmp;
//	qpm_avg.trueNeg /= repEvals_tmp;
//	qpm_avg.truePos /= repEvals_tmp;
//	//Precision or positive predictive value ppv=truePos/(truePos+falsePos)
//	qpm_avg.ppv = (qpm_avg.truePos == 0) ? 0:(double)qpm_avg.truePos / (double)(qpm_avg.truePos + qpm_avg.falsePos);
//	//Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)
//	qpm_avg.tpr = (qpm_avg.truePos == 0) ? 0:(double)qpm_avg.truePos / (double)(qpm_avg.truePos + qpm_avg.falseNeg);
//	//Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)
//	qpm_avg.fpr = (qpm_avg.falsePos == 0) ? 0:(double)qpm_avg.falsePos / (double)(qpm_avg.falsePos + qpm_avg.trueNeg);
//	//Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)
//	qpm_avg.acc = ((qpm_avg.truePos + qpm_avg.trueNeg) == 0) ? 0:(double)(qpm_avg.truePos + qpm_avg.trueNeg) / (double)leftInlier.size();
//	qpm = qpm_avg;
//#if INITMATCHQUALEVAL_O
//	qpm_avg_init.falseNeg /= repEvals_tmp;
//	qpm_avg_init.falsePos /= repEvals_tmp;
//	qpm_avg_init.trueNeg /= repEvals_tmp;
//	qpm_avg_init.truePos /= repEvals_tmp;
//	//Precision or positive predictive value ppv=truePos/(truePos+falsePos)
//	qpm_avg_init.ppv = (qpm_avg_init.truePos == 0) ? 0:(double)qpm_avg_init.truePos / (double)(qpm_avg_init.truePos + qpm_avg_init.falsePos);
//	//Recall or sensivity or true positive rate tpr=truePos/(truePos+falseNeg)
//	qpm_avg_init.tpr = (qpm_avg_init.truePos == 0) ? 0:(double)qpm_avg_init.truePos / (double)(qpm_avg_init.truePos + qpm_avg_init.falseNeg);
//	//Fall-out or false positive rate fpr=falsePos/(falsePos+trueNeg)
//	qpm_avg_init.fpr = (qpm_avg_init.falsePos == 0) ? 0:(double)qpm_avg_init.falsePos / (double)(qpm_avg_init.falsePos + qpm_avg_init.trueNeg);
//	//Accuracy acc=(truePos+trueNeg)/(trueNeg+truePos+falseNeg+falsePos)
//	qpm_avg_init.acc = ((qpm_avg_init.truePos + qpm_avg_init.trueNeg) == 0) ? 0:(double)(qpm_avg_init.truePos + qpm_avg_init.trueNeg) / (double)leftInlier.size();
//	qpr = qpm_avg_init;
//	initEstiInlRatio = initEstiInlRatio_avg / (double)repEvals_tmp;
//#endif
//	tm = tm_tmp;
//
//	if(tm == 0)
//	{
//		cout << "No time measurement for the matching available!" << endl;
//		return -4; //No matching time available
//	}
//	cout << "Matching time (ms): " << tm << endl;
//
//	//Overall runtime
//	to = tf+td+tm;
//
//	//Average runtime per keypoint
//	tkm = tm / (positivesGT + negativesGTl);
//	
//	return 0;
//}
//
//
//void baseMatcher::clearMatchingResult()
//{
//	falseNegMatches.clear();
//	falsePosMatches.clear();
//	truePosMatches.clear();
//	matches.clear();
//#if COSTDISTRATIOEVAL
//	tpfp.clear();
//	costRatios.clear();
//	distRatios.clear();
//#endif
//}
//
///* Performs refinement of the matches based on the VFC algorithm.
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Too less matches for refinement
// */
//int baseMatcher::refineMatches()
//{
//	double tr_tmp;
//
//	tr = DBL_MAX;
//	if(!measureT)
//		cyklesTM = 1;
//
//	//Remove mismatches by vector field consensus (VFC)
//	// preprocess data format
//	/*vector<Point2f> X;
//	vector<Point2f> Y;
//	X.clear();
//	Y.clear();
//	for (unsigned int i = 0; i < matches.size(); i++) {
//		int idx1 = matches[i].queryIdx;
//		int idx2 = matches[i].trainIdx;
//		X.push_back(keypL[idx1].pt);
//		Y.push_back(keypR[idx2].pt);
//	}*/
//	for(int i = 0; i < cyklesTM; i++)
//	{
//		//Clear variables
//		matchesRefined.clear();
//
//		// main process
//		tr_tmp = (double)getTickCount();
//		//VFC myvfc;
//		//if(!myvfc.setData(X, Y))
//		//{
//		//	cout << "Too less matches for refinement with VFC!" << endl;
//		//	return -1; //Too less matches for refinement
//		//}
//		//myvfc.optimize();
//		//vector<int> matchIdx = myvfc.obtainCorrectMatch();
//		tr_tmp = 1000 * ((double)getTickCount() - tr_tmp) / getTickFrequency();
//
//		//for (unsigned int i = 0; i < matchIdx.size(); i++) {
//		//	int idx = matchIdx[i];
//		//	matchesRefined.push_back(matches[idx]);
//		//}
//
//		filterWithVFC(keypL, keypR, matches, matchesRefined);
//
//		if(tr_tmp < tr)
//			tr = tr_tmp;
//	}
//	cout << "Refinement time with VFC (ms): " << tr << endl;
//
//	//Sort matches to ensure increasing query indexes
//	sort(matchesRefined.begin(), matchesRefined.end(),
//			[](DMatch first, DMatch second){return first.queryIdx < second.queryIdx;});
//	
//	//Evaluate the refined matches to get the quality parameters
//	evalMatches(true);
//
//	return 0;
//}
//
///* Shows the matches and optionally stores the matched keypoints as image to the disk. It can be specified if only true positives (blue),
// * all matches (true positives in blue, false positives in red) or all matches with false negatives (true positives in blue, false 
// * positives in red, false negatives in green) should be shown.
// *
// * int drawflags				Input  -> The following matches are drawn in different colors for different flags:
// *										  0:	Only true positives
// *										  1:	True positives and false positives [DEFAULT]
// *										  2:	True positives, false positives, and false negatives
// * bool refinedMatches			Input  -> If false [DEFAULT], the matches before refinement are drawn. Otherwise the refined matches
// *										  are drawn
// * string path					Input  -> If not empty, this string specifies the path where the image with drawn matches should be stored.
// *										  If empty [DEFAULT], no image is stored to disk.
// * string file					Input  -> If not empty, this string specifies the file name including the file extension.
// *										  If empty [DEFAULT], no image is stored to disk.
// * bool storeOnly				Input  -> If true, the output image is only stored to disk but not displayed [DEFAULT = false]
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  No refined matches available
// */
//int baseMatcher::showMatches(int drawflags, bool refinedMatches, std::string path, std::string file, bool storeOnly)
//{
//	vector<cv::DMatch> matches_tmp, matches_tp, matches_fp, matches_fn;
//	vector<cv::DMatch> matches_tp_reduced, matches_fp_reduced, matches_fn_reduced;
//	vector<KeyPoint> kps1_tp, kps2_tp, kps1_fp, kps2_fp, kps1_fn, kps2_fn;
//	vector<KeyPoint> kps1_tp_reduced, kps2_tp_reduced, kps1_fp_reduced, kps2_fp_reduced, kps1_fn_reduced, kps2_fn_reduced;
//	std::vector<bool> falseNegMatches_tmp;
//	std::vector<bool> falsePosMatches_tmp;
//	vector<char> matchesMask;
//	Mat img_correctMatches;
//	size_t keepNMatches = 20; //The overall number of matches that should be displayed
//	size_t keepNMatches_tp, keepNMatches_fp, keepNMatches_fn;
//	float keepXthMatch;
//	float oldremainder, newremainder;
//	int j = 0;
//	
//	if(refinedMatches && matchesRefined.empty())
//		return -1; //No refined matches available
//
//	if(refinedMatches)
//	{
//		matches_tmp = matchesRefined;
//		falseNegMatches_tmp = falseNegMatchesRef;
//		falsePosMatches_tmp = falsePosMatchesRef;
//	}
//	else
//	{
//		matches_tmp = matches;
//		falseNegMatches_tmp = falseNegMatches;
//		falsePosMatches_tmp = falsePosMatches;
//	}
//
//	//Reduce number of displayed matches
//	if(refinedMatches)
//	{
//		size_t hlp;
//		if(drawflags)
//		{
//			if(drawflags == 2)
//			{
//				hlp = qpr.falseNeg + qpr.falsePos + qpr.truePos;
//				keepNMatches_tp = (size_t)floor((float)qpr.truePos / (float)hlp * (float)keepNMatches +0.5f);
//				keepNMatches_fp = (size_t)floor((float)qpr.falsePos / (float)hlp * (float)keepNMatches +0.5f);
//				keepNMatches_fn = (size_t)floor((float)qpr.falseNeg / (float)hlp * (float)keepNMatches +0.5f);
//			}
//			else
//			{
//				hlp = qpr.falsePos + qpr.truePos;
//				keepNMatches_tp = (size_t)floor((float)qpr.truePos / (float)hlp * (float)keepNMatches +0.5f);
//				keepNMatches_fp = (size_t)floor((float)qpr.falsePos / (float)hlp * (float)keepNMatches +0.5f);
//				keepNMatches_fn = 0;
//			}
//		}
//		else
//		{
//			keepNMatches_tp = keepNMatches;
//			keepNMatches_fp = 0;
//			keepNMatches_fn = 0;
//		}
//	}
//	else
//	{
//		size_t hlp;
//		if(drawflags)
//		{
//			if(drawflags == 2)
//			{
//				hlp = qpm.falseNeg + qpm.falsePos + qpm.truePos;
//				keepNMatches_tp = (size_t)floor((float)qpm.truePos / (float)hlp * (float)keepNMatches +0.5f);
//				keepNMatches_fp = (size_t)floor((float)qpm.falsePos / (float)hlp * (float)keepNMatches +0.5f);
//				keepNMatches_fn = (size_t)floor((float)qpm.falseNeg / (float)hlp * (float)keepNMatches +0.5f);
//			}
//			else
//			{
//				hlp = qpm.falsePos + qpm.truePos;
//				keepNMatches_tp = (size_t)floor((float)qpm.truePos / (float)hlp * (float)keepNMatches +0.5f);
//				keepNMatches_fp = (size_t)floor((float)qpm.falsePos / (float)hlp * (float)keepNMatches +0.5f);
//				keepNMatches_fn = 0;
//			}
//		}
//		else
//		{
//			keepNMatches_tp = keepNMatches;
//			keepNMatches_fp = 0;
//			keepNMatches_fn = 0;
//		}
//	}
//
//	if(drawflags)
//	{
//		if(drawflags == 2)
//		{
//			//Get only false negatives
//			j = 0;
//			for (unsigned int i = 0; i < matchesGT.size(); i++)
//			{
//				int idx = matchesGT[i].queryIdx;
//				if(falseNegMatches_tmp[idx])
//				{
//					kps1_fn.push_back(keypL[idx]);
//					matches_fn.push_back(matchesGT[i]);
//					matches_fn.back().queryIdx = j;
//					kps2_fn.push_back(keypR[matches_fn.back().trainIdx]);
//					matches_fn.back().trainIdx = j;
//					j++;
//				}
//			}
//			//Reduce number of displayed matches
//			keepXthMatch = 1.0f;
//			if(matches_fn.size() > keepNMatches_fn)
//				keepXthMatch = (float)matches_fn.size() / (float)keepNMatches_fn;
//			j = 0;
//			oldremainder = 0;
//			for (unsigned int i = 0; i < matches_fn.size(); i++)
//			{
//				newremainder = fmod((float)i, keepXthMatch);
//				//if((i % (int)keepXthMatch) == 0)
//				if(oldremainder >= newremainder)
//				{
//					kps1_fn_reduced.push_back(kps1_fn[i]);
//					matches_fn_reduced.push_back(matches_fn[i]);
//					matches_fn_reduced.back().queryIdx = j;
//					kps2_fn_reduced.push_back(kps2_fn[i]);
//					matches_fn_reduced.back().trainIdx = j;
//					j++;
//				}
//				oldremainder = newremainder;
//			}
//
//			//matchesMask.clear();
//			//matchesMask.assign(matches_fn_reduced.size(),1);
//			drawMatchesCV(imgs[0], kps1_fn_reduced, imgs[1], kps2_fn_reduced, matches_fn_reduced, img_correctMatches, CV_RGB(0,255,0));//, CV_RGB(0,255,0), matchesMask, DrawMatchesFlags::DRAW_OVER_OUTIMG);
//		}
//		//Get only false positives
//		j = 0;
//		for (unsigned int i = 0; i < matches_tmp.size(); i++)
//		{
//			int idx = matches_tmp[i].queryIdx;
//			if(falsePosMatches_tmp[idx])
//			{
//				kps1_fp.push_back(keypL[idx]);
//				matches_fp.push_back(matches_tmp[i]);
//				matches_fp.back().queryIdx = j;
//				kps2_fp.push_back(keypR[matches_fp.back().trainIdx]);
//				matches_fp.back().trainIdx = j;
//				j++;
//			}
//		}
//		//Reduce number of displayed matches
//		keepXthMatch = 1.0f;
//		if(matches_fp.size() > keepNMatches_fp)
//			keepXthMatch = (float)matches_fp.size() / (float)keepNMatches_fp;
//		j = 0;
//		oldremainder = 0;
//		for (unsigned int i = 0; i < matches_fp.size(); i++)
//		{
//			newremainder = fmod((float)i, keepXthMatch);
//			//if((i % (int)keepXthMatch) == 0)
//			if(oldremainder >= newremainder)
//			{
//				kps1_fp_reduced.push_back(kps1_fp[i]);
//				matches_fp_reduced.push_back(matches_fp[i]);
//				matches_fp_reduced.back().queryIdx = j;
//				kps2_fp_reduced.push_back(kps2_fp[i]);
//				matches_fp_reduced.back().trainIdx = j;
//				j++;
//			}
//			oldremainder = newremainder;
//		}
//		matchesMask.clear();
//		matchesMask.assign(matches_fp_reduced.size(),1);
//		drawMatchesCV(imgs[0], kps1_fp_reduced, imgs[1], kps2_fp_reduced, matches_fp_reduced, img_correctMatches, CV_RGB(255,0,0), CV_RGB(255,0,0), matchesMask, DrawMatchesFlags::DRAW_OVER_OUTIMG);
//	}
//
//	//Get only true positives without true negatives
//	j = 0;
//	for (unsigned int i = 0; i < matches_tmp.size(); i++)
//	{
//		int idx = matches_tmp[i].queryIdx;
//		if(truePosMatches[idx])
//		{
//			kps1_tp.push_back(keypL[idx]);
//			matches_tp.push_back(matches_tmp[i]);
//			matches_tp.back().queryIdx = j;
//			kps2_tp.push_back(keypR[matches_tp.back().trainIdx]);
//			matches_tp.back().trainIdx = j;
//			j++;
//		}
//	}
//	//Reduce number of displayed matches
//	keepXthMatch = 1.0f;
//	if(matches_tp.size() > keepNMatches_tp)
//		keepXthMatch = (float)matches_tp.size() / (float)keepNMatches_tp;
//	j = 0;
//	oldremainder = 0;
//	for (unsigned int i = 0; i < matches_tp.size(); i++)
//	{
//		newremainder = fmod((float)i, keepXthMatch);
//		//if((i % (int)keepXthMatch) == 0)
//		if(oldremainder >= newremainder)
//		{
//			kps1_tp_reduced.push_back(kps1_tp[i]);
//			matches_tp_reduced.push_back(matches_tp[i]);
//			matches_tp_reduced.back().queryIdx = j;
//			kps2_tp_reduced.push_back(kps2_tp[i]);
//			matches_tp_reduced.back().trainIdx = j;
//			j++;
//		}
//		oldremainder = newremainder;
//	}
//	//Draw true positive matches
//	if(drawflags)
//	{
//		matchesMask.clear();
//		matchesMask.assign(matches_tp_reduced.size(),1);
//		drawMatchesCV(imgs[0], kps1_tp_reduced, imgs[1], kps2_tp_reduced, matches_tp_reduced, img_correctMatches, CV_RGB(0,0,255), CV_RGB(0,0,255), matchesMask, DrawMatchesFlags::DRAW_OVER_OUTIMG);
//	}
//	else
//	{
//		drawMatchesCV(imgs[0], kps1_tp_reduced, imgs[1], kps2_tp_reduced, matches_tp_reduced, img_correctMatches, CV_RGB(0,0,255));
//	}
//	
//	if(!storeOnly)
//	{
//		//Show result
//		imshow("Detected Correct Matches", img_correctMatches);
//		waitKey(0);
//	}
//
//	if(!path.empty() && !file.empty())
//	{
//		//Store image with matches to disk
//		imwrite(path + "\\" + file, img_correctMatches);
//	}
//
//	return 0;
//}
//
//float getDescriptorDistance(cv::Mat descriptor1, cv::Mat descriptor2)
//{
//	CV_Assert((descriptor1.rows == 1) && ((descriptor1.type() == CV_32F) || (descriptor1.type() == CV_8U)));
//
//	static bool BinaryOrVector = true;
//	static bool useBinPopCnt = false;
//	static bool fixProperties = true;
//	static unsigned int byte8width = 8;
//	static unsigned char descrCols = 64;
//
//	if(fixProperties)
//	{
//		if(descriptor1.type() == CV_32F)
//		{
//			BinaryOrVector = false;
//		}
//
//		int cpuInfo[4];
//		__cpuid(cpuInfo,0x00000001);
//		std::bitset<32> f_1_ECX_ = cpuInfo[2];
//		static bool popcnt_available = f_1_ECX_[23];
//		if(popcnt_available)
//			useBinPopCnt = true;
//
//		if(descriptor1.cols != 64)
//		{
//			byte8width = descriptor1.cols / 8;
//			descrCols = descriptor1.cols;
//		}
//
//		fixProperties = false;
//	}
//
//	if(BinaryOrVector)
//	{
//		unsigned int hamsum = 0;
//		if(useBinPopCnt)
//		{
//			unsigned __int64 hamsum1 = 0;
//			const unsigned __int64 *inputarr1 = reinterpret_cast<const unsigned __int64*>(descriptor1.data);
//			const unsigned __int64 *inputarr2 = reinterpret_cast<const unsigned __int64*>(descriptor2.data);
//			for(unsigned char i = 0; i < byte8width; i++)
//			{
//				hamsum1 += __popcnt64(*(inputarr1 + i) ^ *(inputarr2 + i));
//			}
//			return (float)hamsum1;
//		}
//		else
//		{
//			static const unsigned char BitsSetTable256[256] = 
//			{
//			#   define B2(n) n,     n+1,     n+1,     n+2
//			#   define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
//			#   define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
//				B6(0), B6(1), B6(1), B6(2)
//			};
//
//			for(int i = 0; i<descriptor1.cols;i++)
//			{
//				hamsum += (unsigned int)BitsSetTable256[(descriptor1.at<uchar>(i)^descriptor2.at<uchar>(i))];
//			}
//		}
//		return (float)hamsum;
//	}
//	else
//	{
//		float vecsum = 0;
//
//		for(unsigned char i = 0; i < descrCols; i++)
//		{
//			float hlp = descriptor1.at<float>(i) - descriptor2.at<float>(i);
//			vecsum += hlp * hlp;
//		}
//		return vecsum;
//	}
//
//	//return 0;
//}
//
///* Checks, if ground truth information for the given image pair, inlier ratio and feature detector type is already available on the harddisk.
// * If yes, it is loaded. Otherwise, the ground truth is calculated and stored to the hard disk.
// *
// * Return value:				 0:	Success
// *								-1:	Error creating feature detector
// *								-2:	Too less features detected
// *								-3:	Too less features detected (GT generation)
// *								-4: No error: Initial GTM was already calculated (only used if generateOnlyInitialGTM==true)
// */
//int baseMatcher::checkForGT()
//{
//	std::string filenameGT;
//	std::string fileNameImgL_tmp, imgsPath_tmp;
//	
//	prepareFileNameGT(filenameGT, imgsPath_tmp, fileNameImgL_tmp);
//	prepareFileNameGT(filenameGT_initial, imgsPath_initial, fileNameImgL_initial, true);
//
//	if(!readGTMatchesDisk(filenameGT) || generateOnlyInitialGTM)
//	{
//		int err, err_tmp;
//
//		if (readGTMatchesDisk(filenameGT_initial))
//		{
//			noGTgenBefore = false;
//			if (generateOnlyInitialGTM)
//				return -4;
//		}
//		else
//		{
//			if (keypL.empty() || keypR.empty())
//			{
//				err = detectFeatures();
//				if (err)
//				{
//					tf = -1.0;
//					return err;
//				}
//			}
//		}
//		err_tmp = filterInitFeaturesGT();
//		
//		if(!fileNameImgL_tmp.empty() && !imgsPath_tmp.empty())
//			err = writeGTMatchesDisk(filenameGT, err_tmp ? true:false);
//		if(err)
//			cout << "Not possible to write GT to disk!" << endl;
//		if(err_tmp)
//			return err_tmp - 2;
//	}
//	else if((usedMatchTH == 0) && !(int)floor(100000.0 * (inlRatioL - 0.00034) + 0.5) && 
//								  !(int)floor(100000.0 * (inlRatioR - 0.00021) + 0.5) && (positivesGT == 0) && (negativesGTl == 0) && (negativesGTr == 0))
//	{
//		inlRatioL = 0;
//		inlRatioR = 0;
//		return -3;
//	}
//
//	/*if(testGT)
//	{
//		std::vector<std::pair<cv::Point2f,cv::Point2f>> falseGT;
//		int nrMatches;
//		testGTmatches(1000,falseGT,nrMatches);
//	}*/
//
//	return 0;
//}
//
////Prepares the filename for reading or writing GT files including the path
//void baseMatcher::prepareFileNameGT(std::string& filenameGT, std::string& imgsPath_tmp, std::string& fileNameImgL_tmp, bool noInlRatFilter)
//{
//	fileNameImgL_tmp = fileNameImgL;
//	imgsPath_tmp = imgsPath;
//
//	if (useSameKeypSiVarInl)
//		gtsubfolder = "matchesGTequaKeyInl";
//
//	if (!fileNameImgL_tmp.empty() && !imgsPath_tmp.empty())
//	{
//		//Check the filename and delete dots and the 3 characters after the dots
//		size_t strpos;
//		strpos = fileNameImgL_tmp.find(".");
//		while (strpos != std::string::npos)
//		{
//			if ((strpos + 4) < fileNameImgL_tmp.size()) //4 = 3 characters + 1 to convert from index to size
//			{
//				fileNameImgL_tmp.erase(fileNameImgL_tmp.begin() + strpos, fileNameImgL_tmp.begin() + strpos + 3);
//			}
//			else
//			{
//				fileNameImgL_tmp.erase(fileNameImgL_tmp.begin() + strpos, fileNameImgL_tmp.end());
//			}
//			strpos = fileNameImgL_tmp.find(".");
//		}
//		if (noInlRatFilter)
//			fileNameImgL_tmp += "_inlRatInitial";
//		else
//			fileNameImgL_tmp += "_inlRat" + std::to_string((ULONGLONG)std::floor(this->specifiedInlRatio * 1000.0 + 0.5));
//		fileNameImgL_tmp += featuretype;
//		if (GTfilterExtractor.compare("FREAK")) fileNameImgL_tmp += "_GTd" + GTfilterExtractor;
//		fileNameImgL_tmp += ".gtm"; //File ending: gtm = ground truth matches
//
//									//Check the path
//		DWORD ftyp = GetFileAttributesA(imgsPath_tmp.c_str());
//		if ((ftyp != INVALID_FILE_ATTRIBUTES) && (ftyp & FILE_ATTRIBUTE_DIRECTORY))
//		{
//			imgsPath_tmp += "\\" + gtsubfolder;
//			ftyp = GetFileAttributesA(imgsPath_tmp.c_str());
//			if (ftyp == INVALID_FILE_ATTRIBUTES)
//			{
//				_mkdir(imgsPath_tmp.c_str());
//			}
//			filenameGT = imgsPath_tmp + "\\" + fileNameImgL_tmp;
//		}
//		else
//		{
//			imgsPath_tmp = "";
//		}
//	}
//}
//
///* Checks, if ground truth information for the given image pair, inlier ratio and feature detector type is already available on the harddisk.
// *
// * string filenameGT			Input  -> The path and filename of the ground truth file
// *
// * Return value:				true:	Reading GT file successful
// *								false:	Reading GT file failed
// */
//bool baseMatcher::readGTMatchesDisk(std::string filenameGT)
//{
//	if(filenameGT.empty())
//		return false;
//
//	ifstream gtFromFile(filenameGT.c_str());
//	if(!gtFromFile.good())
//	{
//		gtFromFile.close();
//		return false;
//	}
//
//	if(!readDoubleVal(gtFromFile, "irl ", &inlRatioL)) return false;
//	if(!readDoubleVal(gtFromFile, "irr ", &inlRatioR)) return false;
//	if(!readDoubleVal(gtFromFile, "iro ", &inlRatioO)) return false;
//	if(!readDoubleVal(gtFromFile, "posGT ", &positivesGT)) return false;
//	if(!readDoubleVal(gtFromFile, "negGTl ", &negativesGTl)) return false;
//	if(!readDoubleVal(gtFromFile, "negGTr ", &negativesGTr)) return false;
//	if(!readDoubleVal(gtFromFile, "th ", &usedMatchTH)) return false;
//
//	if(usedMatchTH == 0)
//	{
//		if(!(int)floor(100000.0 * (inlRatioL - 0.00034) + 0.5) && 
//		   !(int)floor(100000.0 * (inlRatioR - 0.00021) + 0.5) && (positivesGT == 0) && (negativesGTl == 0) && (negativesGTr == 0))
//			return true;
//		else
//			return false;
//	}
//	if(std::abs(positivesGT / (positivesGT + negativesGTl) - inlRatioL) > 0.005) return false;
//	if(std::abs(positivesGT / (positivesGT + negativesGTr) - inlRatioR) > 0.005) return false;
//	if(2.0 * positivesGT / (double)(negativesGTr + 2 * positivesGT + negativesGTl) - inlRatioO > 0.005) return false;
//	
//	bool isInlier;
//	int newIntVal;
//	float newFloatVal;
//	cv::DMatch singleMatch;
//	cv::KeyPoint singleKeyPoint;
//
//	//Get inlier vector
//	{
//		string line, word;
//		std::istringstream is;
//		std::getline(gtFromFile, line);
//		if(line.empty())
//		{
//			gtFromFile.close();
//			return false;
//		}
//		is.str(line);
//		is >> word;
//		if(word.compare(0,7,"inliers") != 0) 
//		{
//			gtFromFile.close();
//			return false;
//		}
//		leftInlier.clear();
//		while(is >> std::boolalpha >> isInlier)
//		{
//			leftInlier.push_back(isInlier);
//		}
//		if(leftInlier.empty())
//		{
//			gtFromFile.close();
//			return false;
//		}
//	}
//
//	//get matches
//	{
//		string line, word;
//		std::istringstream is;
//		std::getline(gtFromFile, line);
//		if(line.empty())
//		{
//			clearGTvars();
//			gtFromFile.close();
//			return false;
//		}
//		is.str(line);
//		is >> word;
//		if(word.compare(0,7,"matches") != 0)
//		{
//			clearGTvars();
//			gtFromFile.close();
//			return false;
//		}
//		matchesGT.clear();
//		while(is >> newIntVal)
//		{
//			singleMatch.queryIdx = newIntVal;
//			if(is >> newIntVal)
//				singleMatch.trainIdx = newIntVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			if(is >> newFloatVal)
//				singleMatch.distance = newFloatVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			matchesGT.push_back(singleMatch);
//		}
//		if(matchesGT.empty())
//		{
//			clearGTvars();
//			gtFromFile.close();
//			return false;
//		}
//	}
//
//	//get left keypoints
//	{
//		string line, word;
//		std::istringstream is;
//		std::getline(gtFromFile, line);
//		if(line.empty())
//		{
//			clearGTvars();
//			gtFromFile.close();
//			return false;
//		}
//		is.str(line);
//		is >> word;
//		if(word.compare(0,5,"keypl") != 0) 
//		{
//			clearGTvars();
//			gtFromFile.close();
//			return false;
//		}
//		keypL.clear();
//		while(is >> newFloatVal)
//		{
//			singleKeyPoint.pt.x = newFloatVal;
//			if(is >> newFloatVal)
//				singleKeyPoint.pt.y = newFloatVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			if(is >> newFloatVal)
//				singleKeyPoint.response = newFloatVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			if(is >> newFloatVal)
//				singleKeyPoint.angle = newFloatVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			if(is >> newFloatVal)
//				singleKeyPoint.size = newFloatVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			if(is >> newIntVal)
//				singleKeyPoint.octave = newIntVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			if(is >> newIntVal)
//				singleKeyPoint.class_id = newIntVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			keypL.push_back(singleKeyPoint);
//		}
//		if(keypL.empty())
//		{
//			clearGTvars();
//			gtFromFile.close();
//			return false;
//		}
//	}
//
//	//get right keypoints
//	{
//		string line, word;
//		std::istringstream is;
//		std::getline(gtFromFile, line);
//		if(line.empty())
//		{
//			clearGTvars();
//			gtFromFile.close();
//			return false;
//		}
//		is.str(line);
//		is >> word;
//		if(word.compare(0,5,"keypr") != 0) 
//		{
//			clearGTvars();
//			gtFromFile.close();
//			return false;
//		}
//		keypR.clear();
//		while(is >> newFloatVal)
//		{
//			singleKeyPoint.pt.x = newFloatVal;
//			if(is >> newFloatVal)
//				singleKeyPoint.pt.y = newFloatVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			if(is >> newFloatVal)
//				singleKeyPoint.response = newFloatVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			if(is >> newFloatVal)
//				singleKeyPoint.angle = newFloatVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			if(is >> newFloatVal)
//				singleKeyPoint.size = newFloatVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			if(is >> newIntVal)
//				singleKeyPoint.octave = newIntVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			if(is >> newIntVal)
//				singleKeyPoint.class_id = newIntVal;
//			else
//			{
//				clearGTvars();
//				gtFromFile.close();
//				return false;
//			}
//			keypR.push_back(singleKeyPoint);
//		}
//		if(keypR.empty())
//		{
//			clearGTvars();
//			gtFromFile.close();
//			return false;
//		}
//	}
//
//	if(keypL.size() != leftInlier.size()) return false;
//	if((double)keypR.size() - (positivesGT + negativesGTr) != 0) return false;
//	if((double)keypL.size() - (positivesGT + negativesGTl) != 0) return false;
//	if((double)matchesGT.size() - positivesGT != 0) return false;
//
//	return true;
//}
//
///* Clears vectors containing ground truth information
// *
// * Return value:				none
// */
//void baseMatcher::clearGTvars()
//{
//	leftInlier.clear();
//	matchesGT.clear();
//	keypL.clear();
//	keypR.clear();
//}
//
///* Reads a line from the stream, checks if the first word corresponds to the given keyword and if true, reads the value after the keyword.
// *
// * ifstream gtFromFile			Input  -> Input stream
// * string keyWord				Input  -> Keyword that is compared to the first word of the stream
// * double *value				Output -> Value from the stream after the keyword
// *
// * Return value:				true:	Success
// *								false:	Failed
// */
//bool readDoubleVal(ifstream & gtFromFile, std::string keyWord, double *value)
//{
//	//char stringline[100];
//	string singleLine;
//	size_t strpos;
//
//	//gtFromFile.getline(singleLine);
//	std::getline(gtFromFile, singleLine);
//	//singleLine = stringline;
//	if(singleLine.empty()) return false;
//	strpos = singleLine.find(keyWord);
//	if(strpos == std::string::npos)
//	{
//		gtFromFile.close();
//		return false;
//	}
//	singleLine = singleLine.substr(strpos + keyWord.size());
//	*value = strtod(singleLine.c_str(), NULL);
//	//if(*value == 0) return false;
//
//	return true;
//}
//
///* Writes ground truth information to the hard disk.
// *
// * string filenameGT			Input  -> The path and filename of the ground truth file
// * bool writeEmptyFile			Input  -> If true [Default = false], the gound truth generation
// *										  was not successful due to too less positives or a
// *										  too small inlier ratio. For this reason the following
// *										  values are written to the file: usedMatchTH = 0, 
// *										  inlRatioL = 0.00034, inlRatioR = 0.00021, positivesGT = 0,
// *										  negativesGTl = 0, and negativesGTr = 0. Thus, if these
// *										  values are read from the file, no new ground truth generation
// *										  has to be performed which would fail again.
// *
// * Return value:				 0:	Success
// *								-1: Bad filename or directory
// */
//int baseMatcher::writeGTMatchesDisk(std::string filenameGT, bool writeEmptyFile)
//{
//	std::ofstream gtToFile(filenameGT);
//
//	if(!gtToFile.good())
//	{
//		gtToFile.close();
//		return -1; //Bad filename or directory
//	}
//	
//	if(writeEmptyFile)
//	{
//		usedMatchTH = 0;
//		inlRatioL = 0.00034;
//		inlRatioR = 0.00021;
//		positivesGT = 0;
//		negativesGTl = 0;
//		negativesGTr = 0;
//		inlRatioO = 0;
//	}
//
//	gtToFile << "irl " << inlRatioL << endl;
//	gtToFile << "irr " << inlRatioR << endl;
//	gtToFile << "iro " << inlRatioO << endl;
//	gtToFile << "posGT " << positivesGT << endl;
//	gtToFile << "negGTl " << negativesGTl << endl;
//	gtToFile << "negGTr " << negativesGTr << endl;
//	gtToFile << "th " << usedMatchTH << endl;
//
//	if(writeEmptyFile)
//	{
//		inlRatioL = 0;
//		inlRatioR = 0;
//		gtToFile.close();
//		return 0;
//	}
//	
//	gtToFile << "inliers ";
//	for(size_t i = 0; i < leftInlier.size(); i++)
//	{
//		gtToFile << std::boolalpha << leftInlier[i] << " ";
//	}
//	gtToFile << endl;
//
//	gtToFile << "matches ";
//	for(size_t i = 0; i < matchesGT.size(); i++)
//	{
//		gtToFile << matchesGT[i].queryIdx << " " << matchesGT[i].trainIdx << " " << matchesGT[i].distance << " ";
//	}
//	gtToFile << endl;
//
//	gtToFile << "keypl ";
//	for(size_t i = 0; i < keypL.size(); i++)
//	{
//		gtToFile << keypL[i].pt.x << " " << keypL[i].pt.y << " " << keypL[i].response << " " << keypL[i].angle << " " << keypL[i].size << " " << keypL[i].octave << " " << keypL[i].class_id << " ";
//	}
//	gtToFile << endl;
//
//	gtToFile << "keypr ";
//	for(size_t i = 0; i < keypR.size(); i++)
//	{
//		gtToFile << keypR[i].pt.x << " " << keypR[i].pt.y << " " << keypR[i].response << " " << keypR[i].angle << " " << keypR[i].size << " " << keypR[i].octave << " " << keypR[i].class_id << " ";
//	}
//	gtToFile << endl;
//
//	gtToFile.close();
//
//	return 0;
//}
//
//int baseMatcher::testGTmatches(int & samples, std::vector<std::pair<cv::Point2f,cv::Point2f>> & falseGT, int & usedSamples, 
//							   std::vector<std::pair<double,int>> & distanceHisto, std::vector<double> & distances, int remainingImgs, int & notMatchable,
//							   int *truePosArr, int *falsePosArr, int *falseNegArr, //Arrays truePosArr, falsePosArr, and falseNegArr must be of size 4 and initialized
//							   std::vector<cv::Point2f> & errvecs, std::vector<std::pair<cv::Point2f,cv::Point2f>> & perfectMatches, cv::Mat & HE, std::vector<int> & validityValFalseGT,
//							   std::vector<cv::Point2f> & errvecsGT, std::vector<double> & distancesGT, std::vector<int> & validityValGT, std::vector<double> & distancesEstModel,
//							   annotImgPars & annotationData, std::vector<char> & autoManualAnno, double threshhTh, int imgNr, int *fullN, int *fullSamples, int *fullFails)
//{
//	CV_Assert(!imgs[0].empty() && !imgs[1].empty());
//	if(matchesGT.empty())
//	{
//		usedSamples = 0;
//		falseGT.clear();
//		cout << "No GT matches available!" << endl;
//		return 0;
//	}
//
//	int maxSampleSize;
//	int maxHorImgSize = 800;
//	int maxVerImgSize;
//	const int patchsizeOrig = 48;//must be devidable by 16
//	int patchsizeShow = 3 * patchsizeOrig;
//	const int selMult = 4;
//	int patchSelectShow = selMult * patchsizeOrig;
//	int textheight = 100;//80;//30;
//	int bottomtextheight = 20;;
//	int GTsi = (int)matchesGT.size();
//	cv::Size imgSize;
//	Mat composed, img_tmp[2], img_exch[2];
//	Mat homoGT_exch;
//	vector<int> used_matches;
//	vector<int> skipped;
//	vector<std::pair<int,double>> wrongGTidxDist;
//	vector<Mat> channelsFlow(3);
//	static bool autoHestiSift = true;
//	maxSampleSize = samples > GTsi ? GTsi:samples;
//
//	CV_Assert(!(patchsizeOrig % 16));
//
//	float halfpatch = floor((float)patchsizeOrig / 2.0f);
//	float quarterpatch = floor((float)patchsizeOrig / 4.0f);
//	float eigthpatch = floor((float)patchsizeOrig / 8.0f);
//
//	double oldP = 0, newP = 0;
//
//	//Split 3 channel matrix for access
//	if(flowGtIsUsed)
//	{
//		cv::split(flowGT, channelsFlow);
//	}
//
//	if(fullN && fullSamples && fullFails)
//	{
//		oldP = newP = (*fullSamples == 0) ? 0:((double)(*fullFails) / (double)(*fullSamples));
//	}
//
//	usedSamples = maxSampleSize;
//	falseGT.clear();
//
//	//Generate empty histogram of distances to true matching positon
//	const double binwidth = 0.25, maxDist = 20.0; //in pixels
//	distanceHisto.push_back(std::make_pair<double,int>(0,0));
//	while(distanceHisto.back().first < maxDist)
//		distanceHisto.push_back(std::make_pair<double,int>(distanceHisto.back().first + binwidth,0));
//
//#if LEFT_TO_RIGHT
//	img_exch[0] = imgs[0];
//	img_exch[1] = imgs[1];
//	homoGT_exch = homoGT;
//#else
//	img_exch[0] = imgs[1];
//	img_exch[1] = imgs[0];
//	if(!flowGtIsUsed)
//		homoGT_exch = homoGT.inv();
//#endif
//
//	imgSize.width = img_exch[0].cols > img_exch[1].cols ? img_exch[0].cols:img_exch[1].cols;
//	imgSize.height = img_exch[0].rows > img_exch[1].rows ? img_exch[0].rows:img_exch[1].rows;
//
//	//If the images do not have the same size, enlarge them with a black border
//	if((img_exch[0].cols < imgSize.width) || (img_exch[0].rows < imgSize.height))
//	{
//		img_tmp[0] = Mat::zeros(imgSize, CV_8UC1);
//		img_exch[0].copyTo(img_tmp[0](cv::Rect(0, 0, img_exch[0].cols ,img_exch[0].rows)));
//	}
//	else
//	{
//		img_exch[0].copyTo(img_tmp[0]);
//	}
//	if((img_exch[1].cols < imgSize.width) || (img_exch[1].rows < imgSize.height))
//	{
//		img_tmp[1] = Mat::zeros(imgSize, CV_8UC1);
//		img_exch[1].copyTo(img_tmp[1](cv::Rect(0, 0, img_exch[1].cols ,img_exch[1].rows)));
//	}
//	else
//	{
//		img_exch[1].copyTo(img_tmp[1]);
//	}
//
//	if (imgSize.width > maxHorImgSize)
//    {
//		maxVerImgSize = (int)((float)maxHorImgSize * (float)imgSize.height / (float)imgSize.width);
//        composed = cv::Mat(cv::Size(maxHorImgSize * 2, maxVerImgSize + patchsizeShow + textheight + patchSelectShow + bottomtextheight), CV_8UC3);
//    }
//    else
//    {
//        composed = cv::Mat(cv::Size(imgSize.width * 2, imgSize.height + patchsizeShow + textheight + patchSelectShow + bottomtextheight), CV_8UC3);
//		maxHorImgSize = imgSize.width;
//		maxVerImgSize = imgSize.height;
//    }
//
//	//SIFT features & descriptors used for generating local homographies
//	Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();// FeatureDetector::create( "SIFT" );
//	if(detector.empty())
//	{
//		cout << "Cannot create feature detector!" << endl;
//		return -2; //Cannot create descriptor extractor
//	}
//	Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create(); //DescriptorExtractor::create("SIFT");
//	if(extractor.empty())
//	{
//		cout << "Cannot create descriptor extractor!" << endl;
//		return -3; //Cannot create descriptor extractor
//	}
//
//	srand(time(NULL));
//	for(int i = 0; i < maxSampleSize; i++)
//	{
//		int idx = rand() % GTsi;
//#if CORRECT_MATCHING_RESULT != 3
//		{
//			int j = 0;
//			while(j < (int)used_matches.size())
//			{
//				idx = rand() % GTsi;
//				for(j = 0; j < (int)used_matches.size(); j++)
//				{
//					if(idx == used_matches[j])
//						break;
//				}
//			}
//		}
//#else
//		int missM = 0;
//		bool isNotMatchable = false;
//		{
//			cv::Point2f actKeyL;
//			static int remainingNM = annotationData.notMatchable[imgNr];
//			if((annotationData.notMatchable[imgNr] > 0) && (i == 0))
//			{
//				remainingNM = annotationData.notMatchable[imgNr];
//			}
//			if(remainingNM > 0)
//			{
//				vector<cv::Point2f> falseGTleftMs, notMatchCoords;
//				for(int k = 0; k < annotationData.falseGT[imgNr].size(); k++)
//				{
//					falseGTleftMs.push_back(annotationData.falseGT[imgNr][k].first);
//				}
//				vector<bool> isNotMatchableVec(annotationData.falseGT[imgNr].size(), true);
//				for(int k = 0; k < falseGTleftMs.size(); k++)
//				{
//					for(int k1 = 0; k1 < annotationData.perfectMatches[imgNr].size(); k1++)
//					{
//						cv::Point2f tmpPnt = annotationData.perfectMatches[imgNr][k1].first;
//						if((abs(falseGTleftMs[k].x - tmpPnt.x) < 10 * FLT_MIN) && (abs(falseGTleftMs[k].y - tmpPnt.y) < 10 * FLT_MIN))
//						{
//							isNotMatchableVec[k] = false;
//							break;
//						}
//					}
//				}
//				for(int k = 0; k < falseGTleftMs.size(); k++)
//				{
//					if(isNotMatchableVec[k])
//						notMatchCoords.push_back(falseGTleftMs[k]);
//				}
//				actKeyL = notMatchCoords[(int)notMatchCoords.size() - remainingNM];//this works only if LEFT_TO_RIGHT 0
//				isNotMatchable = true;
//				remainingNM--;
//			}
//			else
//			{	
//				if(annotationData.notMatchable[imgNr] > 0)
//					missM = annotationData.notMatchable[imgNr];
//				actKeyL = annotationData.perfectMatches[imgNr][i - missM].first;//this works only if LEFT_TO_RIGHT 0
//			}
//			for(int j = 0; j < (int)keypL.size(); j++)
//			{
//				if((abs(keypL[j].pt.x - actKeyL.x) < 10 * FLT_MIN) && (abs(keypL[j].pt.y - actKeyL.y) < 10 * FLT_MIN))
//				{
//					idx = j;
//					break;
//				}
//			}
//			for(int j = 0; j < (int)matchesGT.size(); j++)
//			{
//				if(idx == matchesGT[j].queryIdx)
//				{
//					idx = j;
//					break;
//				}
//			}
//		}
//#endif
//		used_matches.push_back(idx);
//
//#if LEFT_TO_RIGHT
//		cv::Point2f lkp = keypL[matchesGT[idx].queryIdx].pt;
//		cv::Point2f rkp = keypR[matchesGT[idx].trainIdx].pt;
//#else
//		cv::Point2f lkp = keypR[matchesGT[idx].trainIdx].pt;
//		cv::Point2f rkp = keypL[matchesGT[idx].queryIdx].pt;
//#endif
//
//		//Generate homography from SIFT
//		Mat Hl, Haff;
//		Mat origPos;
//		Mat tm;
//		if(flowGtIsUsed)
//		{
//			if(autoHestiSift)
//			{
//				cv::Mat mask;
//				vector<cv::KeyPoint> lkpSift, rkpSift;
//				double angle_rot = 0, scale = 1.;
//				Mat D = Mat::ones(2,1, CV_64F), Rrot, Rdeform = Mat::eye(2,2,CV_64F);
//				mask = Mat::zeros(img_tmp[0].rows + patchsizeOrig, img_tmp[0].cols + patchsizeOrig, CV_8UC1);
//				mask(Rect(lkp.x, lkp.y, patchsizeOrig, patchsizeOrig)) = Mat::ones(patchsizeOrig, patchsizeOrig, CV_8UC1); //generate a mask to detect keypoints only within the image roi for the selected match
//				mask = mask(Rect(halfpatch, halfpatch, img_tmp[0].cols, img_tmp[0].rows));
//				detector->detect(img_tmp[0], lkpSift, mask);//extract SIFT keypoints in the left patch
//				if(!lkpSift.empty())
//				{
//					mask = Mat::zeros(img_tmp[1].rows + patchsizeOrig, img_tmp[1].cols + patchsizeOrig, CV_8UC1);
//					mask(Rect(rkp.x, rkp.y, patchsizeOrig, patchsizeOrig)) = Mat::ones(patchsizeOrig, patchsizeOrig, CV_8UC1);
//					mask = mask(Rect(halfpatch, halfpatch, img_tmp[1].cols, img_tmp[1].rows));
//					detector->detect(img_tmp[1], rkpSift, mask);//extract SIFT keypoints in the right patch
//					if(!rkpSift.empty())
//					{
//						Mat ldescSift, rdescSift;
//						vector<vector<cv::DMatch>> matchesBf;
//						vector<cv::DMatch> matchesBfTrue;
//						extractor->compute(img_tmp[0], lkpSift, ldescSift);//compute descriptors for the SIFT keypoints
//						extractor->compute(img_tmp[1], rkpSift, rdescSift);//compute descriptors for the SIFT keypoints
//						if(!lkpSift.empty() && !rkpSift.empty())
//						{
//							cv::BFMatcher matcher(NORM_L2, false);
//							matcher.knnMatch(ldescSift, rdescSift, matchesBf, 2);//match the SIFT features with a BF matcher and 2 nearest neighbors
//							cv::Point2f deltaP = lkp - rkp;
//							for(int k = 0; k < (int)matchesBf.size(); k++)
//							{
//								if(matchesBf[k].size() < 2)
//									continue;
//								if(matchesBf[k][0].distance < (0.75 * matchesBf[k][1].distance))//ratio test
//								{
//									float dc;
//									cv::Point2f deltaPe = lkpSift[matchesBf[k][0].queryIdx].pt - rkpSift[matchesBf[k][0].trainIdx].pt;
//									deltaPe = deltaP - deltaPe;
//									dc = deltaPe.x * deltaPe.x + deltaPe.y * deltaPe.y;
//									if(dc < halfpatch * halfpatch / 4)//the flow of the new matching features sould not differ more than a quarter of the patchsize to the flow of the match under test
//									{
//										matchesBfTrue.push_back(matchesBf[k][0]);
//									}
//								}
//							}
//							if(!matchesBfTrue.empty())
//							{
//								if(matchesBfTrue.size() > 2)//for a minimum of 3 matches, an affine homography can be calculated
//								{
//									int k1;
//									vector<bool> lineardep(matchesBfTrue.size(), true);
//									vector<cv::DMatch> matchesBfTrue_tmp;
//									for(k1 = 0; k1 < matchesBfTrue.size(); k1++ )
//									{
//										// check that the i-th selected point does not belong
//										// to a line connecting some previously selected points
//										for(int j = 0; j < k1; j++ )
//										{
//											if(!lineardep[j]) continue;
//											double dx1 = lkpSift[matchesBfTrue[j].queryIdx].pt.x - lkpSift[matchesBfTrue[k1].queryIdx].pt.x;
//											double dy1 = lkpSift[matchesBfTrue[j].queryIdx].pt.y - lkpSift[matchesBfTrue[k1].queryIdx].pt.y;
//											for(int k = 0; k < j; k++ )
//											{
//												if(!lineardep[k]) continue;
//												double dx2 = lkpSift[matchesBfTrue[k].queryIdx].pt.x - lkpSift[matchesBfTrue[k1].queryIdx].pt.x;
//												double dy2 = lkpSift[matchesBfTrue[k].queryIdx].pt.y - lkpSift[matchesBfTrue[k1].queryIdx].pt.y;
//												if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
//													lineardep[k] = false;
//											}
//										}
//									}
//
//									for(k1 = 0; k1 < matchesBfTrue.size(); k1++ )
//									{
//										if(lineardep[k1])
//											matchesBfTrue_tmp.push_back(matchesBfTrue[k1]);
//									}
//									matchesBfTrue = matchesBfTrue_tmp;
//								}
//						
//								if(matchesBfTrue.size() > 2)//for a minimum of 3 matches, an affine homography can be calculated
//								{
//									vector<cv::Point2f> lps, rps;
//									int msize = (int)matchesBfTrue.size();
//									for(int k = 0; k < msize; k++)
//									{
//										lps.push_back(lkpSift[matchesBfTrue[k].queryIdx].pt);
//										rps.push_back(rkpSift[matchesBfTrue[k].trainIdx].pt);
//									}
//
//									if(matchesBfTrue.size() > 7) //Estimate a projective homography with RANSAC and remove outliers
//									{
//										Mat Hprmask;
//										Mat Hpr = findHomography(lps, rps, CV_RANSAC, 1.5, Hprmask);
//										if(!Hpr.empty())
//										{
//											vector<cv::DMatch> matchesBfTrue_tmp;
//											vector<cv::Point2f> lps_tmp, rps_tmp;
//											for(int k = 0; k < Hprmask.rows; k++)
//											{
//												if(Hprmask.at<unsigned char>(k) > 0)
//												{
//													matchesBfTrue_tmp.push_back(matchesBfTrue[k]);
//													lps_tmp.push_back(lps[k]);
//													rps_tmp.push_back(rps[k]);
//												}
//											}
//											if(matchesBfTrue_tmp.size() > 3)
//											{
//												matchesBfTrue = matchesBfTrue_tmp;
//												lps = lps_tmp;
//												rps = rps_tmp;
//												msize = (int)matchesBfTrue.size();
//											}
//										}
//									}
//
//									if(matchesBfTrue.size() > 2)
//									{
//										Hl = cv::estimateRigidTransform(lps, rps, true);//estimate an affine homography
//										if(!Hl.empty())
//										{
//											scale = cv::norm(Hl.colRange(0,2));//get the scale from the homography
//											double scales_med, angles_med;
//											vector<double> scales, angles_rot;
//											vector<cv::KeyPoint> lks, rks;
//											for(int k = 0; k < msize; k++)
//											{
//												lks.push_back(lkpSift[matchesBfTrue[k].queryIdx]);
//												rks.push_back(rkpSift[matchesBfTrue[k].trainIdx]);
//											}
//											for(int k = 0; k < msize; k++)//get the scales and angles from the matching SIFT keypoints
//											{
//												double sil = lks[k].size;
//												double sir = rks[k].size;
//												scales.push_back(sil / sir);
//
//												double angl = lks[k].angle;
//												double angr = rks[k].angle;
//												angles_rot.push_back(angl - angr);
//											}
//											std::sort(scales.begin(), scales.end());
//											std::sort(angles_rot.begin(), angles_rot.end());
//											if(msize % 2 == 0)//get the median of the scales and angles from the matching SIFT keypoints
//											{
//												scales_med = (scales[msize / 2 - 1] + scales[msize / 2]) / 2.0;
//												angles_med = (angles_rot[msize / 2 - 1] + angles_rot[msize / 2]) / 2.0;
//											}
//											else
//											{
//												scales_med = scales[msize / 2];
//												angles_med = angles_rot[msize / 2];
//											}
//
//											cv::SVD Hsvd;
//											Mat U, Vt;
//											Hsvd.compute(Hl.colRange(0,2), D, U, Vt, cv::SVD::FULL_UV);//perform a SVD decomposition to extract the angles from the homography
//											Rrot = U * Vt;//Calculate the rotation matrix
//											Rdeform = Vt;//This is the rotation matrix for the deformation (shear) in conjuction with the diagonal matrix D
//											angle_rot = -1.0 * std::atan2(Rrot.at<double>(1,0), Rrot.at<double>(0,0)) * 180.0 / 3.14159265; //As sin, cos are defined counterclockwise, but cv::keypoint::angle is clockwise, multiply it with -1
//											double scalechange = scales_med / scale;
//											double rotchange = abs(angles_med - angle_rot);
//											if((scalechange > 1.1) || (1. / scalechange > 1.1))//Check the correctness of the scale and angle using the values from the homography and the SIFT keypoints
//											{
//												if(rotchange < 11.25)
//												{
//													if((scale < 2.) && (scalechange < 2.))
//														scale = abs(1. - scale) < abs(1. - scales_med) ? scale : scales_med;
//													scale = (scale > 1.2) || (scale < 0.8) ? 1.0:scale;
//
//													if(angles_med * angle_rot > 0)//angles should have the same direction
//													{
//														if(abs(angle_rot) > abs(angles_med))
//															angle_rot = (angle_rot + angles_med) / 2.0;
//													}
//													else
//													{
//														angle_rot = 0;
//														Rdeform = Mat::eye(2, 2, Rdeform.type());
//														D.setTo(1.0);
//													}
//												}
//												else
//												{
//													scale = 1.0;
//													angle_rot = 0;
//													Rdeform = Mat::eye(2, 2, Rdeform.type());
//													D.setTo(1.0);
//												}
//											}
//											else if(rotchange > 11.25)
//											{
//												angle_rot = 0;
//												Rdeform = Mat::eye(2, 2, Rdeform.type());
//												D.setTo(1.0);
//											}
//										}
//									}
//								}
//							}
//						}
//					}
//				}
//				angle_rot = angle_rot * 3.14159265 / 180.0;
//				Rrot = (Mat_<double>(2,2) << std::cos(angle_rot), (-1. * std::sin(angle_rot)), std::sin(angle_rot), std::cos(angle_rot)); //generate the new rotation matrix
//				Haff = scale * Rrot * Rdeform.t() * Mat::diag(D) * Rdeform; //Calculate the new affine homography (without translation)
//				origPos = (Mat_<double>(2,1) << (double)halfpatch, (double)halfpatch);
//				tm = Haff * origPos;//align the translkation vector of the homography in a way that the middle of the patch has the same coordinate
//				tm = origPos - tm;
//				cv::hconcat(Haff, tm, Haff);//Construct the final affine homography
//			}
//			else
//			{
//				Haff = Mat::eye(2, 3, CV_64FC1);
//				tm = Mat::zeros(2, 1, CV_64FC1);
//			}
//		}
//		else
//		{
//			Haff = Mat::eye(3,3,homoGT_exch.type());
//			Haff.at<double>(0,2) = (double)lkp.x - (double)halfpatch;
//			Haff.at<double>(1,2) = (double)lkp.y - (double)halfpatch;
//			Haff = homoGT_exch * Haff;
//			Mat origPos1 = (Mat_<double>(3,1) << (double)halfpatch, (double)halfpatch, 1.0);
//			tm = Haff * origPos1;//align the translation vector of the homography in a way that the middle of the patch has the same coordinate
//			tm /= tm.at<double>(2);
//			tm = origPos1 - tm;
//			tm = tm.rowRange(0,2);
//			Mat helpH1 = Mat::eye(3,3,homoGT_exch.type());
//			helpH1.at<double>(0,2) = tm.at<double>(0);
//			helpH1.at<double>(1,2) = tm.at<double>(1);
//			Haff = helpH1 * Haff;
//		}
//
//		// create images to display
//		composed.setTo(0);
//		string str;
//		vector<cv::Mat> show_color(2), patches(2), show_color_tmp(2);
//		Mat blended, diffImg;
//		cv::cvtColor(img_tmp[0], show_color[0], CV_GRAY2RGB);
//		cv::cvtColor(img_tmp[1], show_color[1], CV_GRAY2RGB);
//		show_color_tmp[0] = show_color[0].clone();
//		show_color_tmp[1] = show_color[1].clone();
//
//		//mark the keypoint positions
//		cv::line(show_color[0], lkp - Point2f(5.0f, 0), lkp + Point2f(5.0f, 0), cv::Scalar(0, 0, 255));
//		cv::line(show_color[0], lkp - Point2f(0, 5.0f), lkp + Point2f(0, 5.0f), cv::Scalar(0, 0, 255));
//		cv::line(show_color[1], rkp - Point2f(5.0f, 0), rkp + Point2f(5.0f, 0), cv::Scalar(0, 0, 255));
//		cv::line(show_color[1], rkp - Point2f(0, 5.0f), rkp + Point2f(0, 5.0f), cv::Scalar(0, 0, 255));
//
//		//Extract the patches around the matching keypoints
//		vector<cv::Mat> show_color_border(2);
//		copyMakeBorder(show_color_tmp[0], show_color_border[0], halfpatch, halfpatch, halfpatch, halfpatch, BORDER_CONSTANT, Scalar(0, 0, 0));
//		copyMakeBorder(show_color_tmp[1], show_color_border[1], halfpatch, halfpatch, halfpatch, halfpatch, BORDER_CONSTANT, Scalar(0, 0, 0));
//
//		patches[0] = show_color_border[0](Rect(lkp, Size(patchsizeOrig, patchsizeOrig))).clone();
//		patches[1] = show_color_border[1](Rect(rkp, Size(patchsizeOrig, patchsizeOrig))).clone();
//
//		//Generate a blended patch
//		double alpha = 0.5;
//		double beta = 1.0 - alpha;
//		vector<Mat> patches_color(2);
//		patches[0].copyTo(patches_color[0]);
//		patches[1].copyTo(patches_color[1]);
//		patches_color[0].reshape(1, patches_color[0].rows * patches_color[0].cols).col(1).setTo(Scalar(0));
//		patches_color[0].reshape(1, patches_color[0].rows * patches_color[0].cols).col(0).setTo(Scalar(0));
//		patches_color[1].reshape(1, patches_color[1].rows * patches_color[1].cols).col(0).setTo(Scalar(0));
//		patches_color[1].reshape(1, patches_color[1].rows * patches_color[1].cols).col(2).setTo(Scalar(0));
//		addWeighted(patches_color[0], alpha, patches_color[1], beta, 0.0, blended);
//
//		//Generate image diff
//		cv::Point2f pdiff[2];
//		cv::Mat img_wdiff[3], patches_wdiff[2], patch_wdhist2[2], patch_equal1, largerpatch, largerpatchhist, shiftedpatch, patch_equal1_tmp;
//		copyMakeBorder(img_tmp[0], img_wdiff[0], halfpatch, halfpatch, halfpatch, halfpatch, BORDER_CONSTANT, Scalar(0, 0, 0));//Extract new images without keypoint markers
//		copyMakeBorder(img_tmp[1], img_wdiff[1], halfpatch, halfpatch, halfpatch, halfpatch, BORDER_CONSTANT, Scalar(0, 0, 0));
//		patches_wdiff[0] = img_wdiff[0](Rect(lkp, Size(patchsizeOrig, patchsizeOrig))).clone();
//		patches_wdiff[1] = img_wdiff[1](Rect(rkp, Size(patchsizeOrig, patchsizeOrig))).clone();
//		equalizeHist(patches_wdiff[0], patch_wdhist2[0]);
//		equalizeHist(patches_wdiff[1], patch_wdhist2[1]);
//		copyMakeBorder(img_tmp[1], img_wdiff[2], patchsizeOrig, patchsizeOrig, patchsizeOrig, patchsizeOrig, BORDER_CONSTANT, Scalar(0, 0, 0));//Get a larger border in the right image to extract a larger patch
//		largerpatch = img_wdiff[2](Rect(rkp, Size(2 * patchsizeOrig, 2 * patchsizeOrig))).clone();//Extract a larger patch to enable template matching
//		equalizeHist(largerpatch, largerpatchhist);
//
//		iterativeTemplMatch(patch_wdhist2[0], largerpatchhist, patchsizeOrig, pdiff[0]);
//
//		Mat compDiff = Mat::eye(2,2,CV_64F);
//		Mat tdiff = (Mat_<double>(2,1) << (double)pdiff[0].x, (double)pdiff[0].y);
//		cv::hconcat(compDiff, tdiff, compDiff);
//		warpAffine(patch_wdhist2[0], shiftedpatch, compDiff, patch_wdhist2[0].size(), INTER_LANCZOS4); //Shift the left patch according to the value from template matching
//		{
//			cv::Point2f newpdiff = cv::Point2f(0,0);
//			findLocalMin(shiftedpatch, patch_wdhist2[1], quarterpatch, eigthpatch, newpdiff, patchsizeOrig);
//			compDiff = Mat::eye(2,2,CV_64F);
//			Mat tdiff2 = (Mat_<double>(2,1) << (double)newpdiff.x, (double)newpdiff.y);
//			cv::hconcat(compDiff, tdiff2, compDiff);
//			warpAffine(shiftedpatch, shiftedpatch, compDiff, shiftedpatch.size(), INTER_LANCZOS4); //Shift the left patch according to the value from template matching
//			pdiff[0] += newpdiff;
//		}
//		absdiff(patch_wdhist2[0], patch_wdhist2[1], patch_equal1);
//		cv::cvtColor(patch_equal1, diffImg, CV_GRAY2RGB);
//		absdiff(shiftedpatch, patch_wdhist2[1], patch_equal1);//Diff with compensated shift but without warp
//		patch_equal1_tmp = patch_equal1.clone();
//
//		//Generate warped image diff
//		double errlevels[2] = {0, 0};
//		Mat border[4];
//		double minmaxXY[4];
//		cv::Mat patch_wdhist1, diffImg2, warped_patch0;
//		if(flowGtIsUsed)
//		{
//			warpAffine(patch_wdhist2[0],warped_patch0,Haff,patch_wdhist2[0].size(), INTER_LANCZOS4); //Warp the left patch
//		}
//		else
//		{
//			warpPerspective(patch_wdhist2[0], warped_patch0, Haff, patch_wdhist2[0].size(), INTER_LANCZOS4); //Warp the left patch
//		}
//
//		iterativeTemplMatch(warped_patch0, largerpatchhist, patchsizeOrig, pdiff[1]);
//
//		compDiff = Mat::eye(2,2,CV_64F);
//		tdiff = (Mat_<double>(2,1) << (double)pdiff[1].x, (double)pdiff[1].y);
//		cv::hconcat(compDiff, tdiff, compDiff);
//		warpAffine(warped_patch0, warped_patch0, compDiff, patch_wdhist2[0].size(), INTER_LANCZOS4); //Shift the left patch
//		{
//			cv::Point2f newpdiff = cv::Point2f(0,0);
//			findLocalMin(warped_patch0, patch_wdhist2[1], quarterpatch, eigthpatch, newpdiff, patchsizeOrig);
//			compDiff = Mat::eye(2,2,CV_64F);
//			Mat tdiff2 = (Mat_<double>(2,1) << (double)newpdiff.x, (double)newpdiff.y);
//			cv::hconcat(compDiff, tdiff2, compDiff);
//			warpAffine(warped_patch0, warped_patch0, compDiff, warped_patch0.size(), INTER_LANCZOS4); //Shift the left patch according to the value from template matching
//			pdiff[1] += newpdiff;
//			tdiff = (Mat_<double>(2,1) << (double)pdiff[1].x, (double)pdiff[1].y);
//		}
//		absdiff(warped_patch0, patch_wdhist2[1], patch_wdhist1);//Diff with compensated shift and warped
//		for(int k = 0; k < 2; k++) //get the new (smaller) patch size after warping
//		{
//			for(int m = 0; m < 2; m++)
//			{
//				Mat tmp;
//				int midx = k * 2 + m;
//				if(flowGtIsUsed)
//				{
//					origPos = (Mat_<double>(2,1) << (double)k * (double)patchsizeOrig, (double)m * (double)patchsizeOrig);
//					border[midx] = Haff.colRange(0,2) * origPos;
//					border[midx] += tm;
//				}
//				else
//				{
//					origPos = (Mat_<double>(3,1) << (double)k * (double)patchsizeOrig, (double)m * (double)patchsizeOrig, 1.0);
//					border[midx] = Haff * origPos;
//					border[midx] /= border[midx].at<double>(2);
//					border[midx] = border[midx].rowRange(0,2);
//				}
//				border[midx].at<double>(0) = border[midx].at<double>(0) > patchsizeOrig ? (double)patchsizeOrig:border[midx].at<double>(0);
//				border[midx].at<double>(0) = border[midx].at<double>(0) < 0 ? 0:border[midx].at<double>(0);
//				border[midx].at<double>(1) = border[midx].at<double>(1) > patchsizeOrig ? (double)patchsizeOrig:border[midx].at<double>(1);
//				border[midx].at<double>(1) = border[midx].at<double>(1) < 0 ? 0:border[midx].at<double>(1);
//				//tmp = (Mat_<double>(2,1) << (k ? (double)min(pdiff[0].x, pdiff[1].x):(double)max(pdiff[0].x, pdiff[1].x)), (m ? (double)min(pdiff[0].y, pdiff[1].y):(double)max(pdiff[0].y, pdiff[1].y)));//from shifting
//				//border[midx] += tmp;
//			}
//		}
//		{
//			vector<std::pair<int,double>> border_tmp;
//			for(int k = 0; k < 4; k++)
//				border_tmp.push_back(make_pair(k, border[k].at<double>(0)));
//			sort(border_tmp.begin(), border_tmp.end(),
//				[](pair<int,double> first, pair<int,double> second){return first.second < second.second;});
//			border[border_tmp[0].first].at<double>(0) += (double)max(pdiff[0].x, pdiff[1].x);
//			border[border_tmp[1].first].at<double>(0) += (double)max(pdiff[0].x, pdiff[1].x);
//			border[border_tmp[2].first].at<double>(0) += (double)min(pdiff[0].x, pdiff[1].x);
//			border[border_tmp[3].first].at<double>(0) += (double)min(pdiff[0].x, pdiff[1].x);
//			border_tmp.clear();
//			for(int k = 0; k < 4; k++)
//				border_tmp.push_back(make_pair(k, border[k].at<double>(1)));
//			sort(border_tmp.begin(), border_tmp.end(),
//				[](pair<int,double> first, pair<int,double> second){return first.second < second.second;});
//			border[border_tmp[0].first].at<double>(1) += (double)max(pdiff[0].y, pdiff[1].y);
//			border[border_tmp[1].first].at<double>(1) += (double)max(pdiff[0].y, pdiff[1].y);
//			border[border_tmp[2].first].at<double>(1) += (double)min(pdiff[0].y, pdiff[1].y);
//			border[border_tmp[3].first].at<double>(1) += (double)min(pdiff[0].y, pdiff[1].y);
//		}
//
//		std::vector<cv::Point2d> pntsPoly1, pntsPoly2, pntsRes;
//		cv::Point2d retLT, retRB;
//		pntsPoly1.push_back(cv::Point2d(0,0));
//		pntsPoly1.push_back(cv::Point2d(patchsizeOrig,0));
//		pntsPoly1.push_back(cv::Point2d(patchsizeOrig,patchsizeOrig));
//		pntsPoly1.push_back(cv::Point2d(0,patchsizeOrig));
//		pntsPoly2.push_back(cv::Point2d(border[0].at<double>(0),border[0].at<double>(1)));
//		pntsPoly2.push_back(cv::Point2d(border[2].at<double>(0),border[2].at<double>(1)));
//		pntsPoly2.push_back(cv::Point2d(border[3].at<double>(0),border[3].at<double>(1)));
//		pntsPoly2.push_back(cv::Point2d(border[1].at<double>(0),border[1].at<double>(1)));
//		intersectPolys(pntsPoly1, pntsPoly2, pntsRes);
//		if(maxInscribedRect(pntsRes, retLT, retRB))
//		{
//			minmaxXY[0] = retLT.x;//minX
//			minmaxXY[1] = retRB.x;//maxX
//			minmaxXY[2] = retLT.y;//minY
//			minmaxXY[3] = retRB.y;//maxY
//		}
//		else
//		{
//			minmaxXY[0] = max(border[0].at<double>(0), border[1].at<double>(0));//minX
//			minmaxXY[1] = min(border[2].at<double>(0), border[3].at<double>(0));//maxX
//			minmaxXY[2] = max(border[0].at<double>(1), border[2].at<double>(1));//minY
//			minmaxXY[3] = min(border[1].at<double>(1), border[3].at<double>(1));//maxY
//		}
//
//		for(int k = 0; k < 4; k++)
//		{
//			minmaxXY[k] = std::floor(minmaxXY[k] + 0.5);
//			minmaxXY[k] = minmaxXY[k] < 0 ? 0:minmaxXY[k];
//			if(k % 2 == 0)
//				minmaxXY[k] = minmaxXY[k] > patchsizeOrig ? 0:minmaxXY[k];
//			else
//				minmaxXY[k] = minmaxXY[k] > patchsizeOrig ? patchsizeOrig:minmaxXY[k];
//		}
//		if((minmaxXY[0] >= minmaxXY[1]) || (minmaxXY[2] >= minmaxXY[3]))
//			errlevels[0] = -1.0;
//		else
//		{
//			patch_wdhist1 = patch_wdhist1(Rect(minmaxXY[0], minmaxXY[2], minmaxXY[1] - minmaxXY[0], minmaxXY[3] - minmaxXY[2])).clone(); //with warp
//			patch_equal1 = patch_equal1(Rect(minmaxXY[0], minmaxXY[2], minmaxXY[1] - minmaxXY[0], minmaxXY[3] - minmaxXY[2])).clone(); //without warp
//			errlevels[0] = cv::sum(patch_equal1)[0];
//			errlevels[1] = cv::sum(patch_wdhist1)[0];
//		}
//		if(errlevels[0] < errlevels[1])//Check if the diff without warp is better
//		{
//			minmaxXY[0] = (double)pdiff[0].x;//minX
//			minmaxXY[1] = (double)patchsizeOrig + (double)pdiff[0].x;//maxX
//			minmaxXY[2] = (double)pdiff[0].y;//minY
//			minmaxXY[3] = (double)patchsizeOrig + (double)pdiff[0].y;//maxY
//			for(int k = 0; k < 4; k++)
//			{
//				minmaxXY[k] = std::floor(minmaxXY[k] + 0.5);
//				minmaxXY[k] = minmaxXY[k] < 0 ? 0:minmaxXY[k];
//				if(k % 2 == 0)
//					minmaxXY[k] = minmaxXY[k] > patchsizeOrig ? 0:minmaxXY[k];
//				else
//					minmaxXY[k] = minmaxXY[k] > patchsizeOrig ? patchsizeOrig:minmaxXY[k];
//			}
//			patch_wdhist1 = patch_equal1_tmp(Rect(minmaxXY[0], minmaxXY[2], minmaxXY[1] - minmaxXY[0], minmaxXY[3] - minmaxXY[2]));
//			warped_patch0 = shiftedpatch;
//			pdiff[1] = -pdiff[0];
//		}
//		else
//		{
//			if(flowGtIsUsed)
//			{
//				tdiff = Haff.colRange(0,2).inv() * (-tdiff);//Calculate the displacement in pixels of the original image (not warped)
//			}
//			else
//			{
//				pdiff[1] = -pdiff[1];
//				tdiff = (Mat_<double>(3,1) << (double)pdiff[1].x, (double)pdiff[1].y, 1.0);
//
//				Mat Haffi = Mat::eye(3,3,homoGT_exch.type());
//				Haffi.at<double>(0,2) = (double)rkp.x;
//				Haffi.at<double>(1,2) = (double)rkp.y;
//				Haffi = homoGT_exch.inv() * Haffi;
//				Mat helpH1 = Mat::eye(3,3,homoGT_exch.type());
//				helpH1.at<double>(0,2) = -Haffi.at<double>(0,2) / Haffi.at<double>(2,2);
//				helpH1.at<double>(1,2) = -Haffi.at<double>(1,2) / Haffi.at<double>(2,2);
//				Haffi = helpH1 * Haffi;
//				tdiff = Haffi * tdiff;//Calculate the displacement in pixels of the original image (not warped)
//				tdiff /= tdiff.at<double>(2);
//			}
//			pdiff[1].x = tdiff.at<double>(0);
//			pdiff[1].y = tdiff.at<double>(1);
//		}
//		cv::cvtColor(patch_wdhist1, diffImg2, CV_GRAY2RGB);
//
//		//Get thresholded and eroded image
//		Mat resultimg, resultimg_color, erdielement;
//		threshold( patch_wdhist1, resultimg, threshhTh, 255.0, 3 );//3: Threshold to Zero; th-values: 64 for normal images, 20 for synthetic
//		erdielement = getStructuringElement( cv::MORPH_RECT, Size( 2, 2 ));
//		erode(resultimg, resultimg, erdielement );
//		dilate(resultimg, resultimg, erdielement );
//		int nrdiff[2];
//		nrdiff[0] = cv::countNonZero(resultimg);
//		Mat resultimg2 = resultimg(Rect(resultimg.cols / 4, resultimg.rows / 4, resultimg.cols / 2, resultimg.rows / 2));
//		nrdiff[1] = cv::countNonZero(resultimg2);
//		double badFraction[2];
//		badFraction[0] = (double)nrdiff[0] / (double)(resultimg.rows * resultimg.cols);
//		badFraction[1] = (double)nrdiff[1] / (double)(resultimg2.rows * resultimg2.cols);
//		cv::cvtColor(resultimg, resultimg_color, CV_GRAY2RGB);
//
//		cv::Point2f singleErrVec = pdiff[1];
//		double diffdist = (double)std::sqrt(pdiff[1].x * pdiff[1].x + pdiff[1].y * pdiff[1].y);
//#if CORRECT_MATCHING_RESULT == 3
//		if(isNotMatchable || (annotationData.distances[imgNr][i - missM] >= usedMatchTH))
//			badFraction[1] = 10.0;
//#endif
//		if(!(((badFraction[0] < badFraction[1]) && (badFraction[1] > 0.05)) || (badFraction[0] > 0.18) || (badFraction[1] > 0.1) || (diffdist >= usedMatchTH)) || ((diffdist < 0.7) && (badFraction[1] < 0.04)))
//		{
//			stringstream ss;
////#ifdef _DEBUG
////			ss << "OK! Diff: " << diffdist;
////			str = ss.str();
////			putText(composed, str.c_str(), cv::Point2d(15, 70), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));
////#endif
//			distances.push_back(diffdist);
//			autoManualAnno.push_back('A');
//			int k = 1;
//			while((k < distanceHisto.size()) && (distanceHisto[k].first < diffdist))
//				k++;
//			distanceHisto[k - 1].second++;
//			errvecs.push_back(singleErrVec);
//#if LEFT_TO_RIGHT
//			perfectMatches.push_back(std::make_pair(lkp + singleErrVec, rkp));
//#else
//			perfectMatches.push_back(std::make_pair(rkp, lkp + singleErrVec));
//#endif
//			calcErrorToSpatialGT(perfectMatches.back().first, perfectMatches.back().second, channelsFlow, flowGtIsUsed, errvecsGT, validityValGT, homoGT, lkp, rkp);
//			distancesGT.push_back(std::sqrt(errvecsGT.back().x * errvecsGT.back().x + errvecsGT.back().y * errvecsGT.back().y));
//			if(diffdist >= usedMatchTH)
//			{
//#if LEFT_TO_RIGHT
//				falseGT.push_back(std::make_pair(lkp, rkp));
//#else
//				falseGT.push_back(std::make_pair(rkp, lkp));
//#endif
//				wrongGTidxDist.push_back(std::make_pair(idx, diffdist));
//				if(flowGtIsUsed)
//				{
//#if LEFT_TO_RIGHT
//					validityValFalseGT.push_back(channelsFlow[2].at<float>((int)floor(lkp.y + 0.5f), (int)floor(lkp.x + 0.5f)));
//#else
//					validityValFalseGT.push_back(channelsFlow[2].at<float>((int)floor(rkp.y + 0.5f), (int)floor(rkp.x + 0.5f)));
//#endif
//				}
//				else
//				{
//					validityValFalseGT.push_back(1);
//				}
//			}
////#ifndef _DEBUG
//			continue;
////#endif
//		}
//		else
//		{
//			autoManualAnno.push_back('M');
//#if CORRECT_MATCHING_RESULT == 3
//			/*bool isNotMatchable = false;
//			if(annotationData.notMatchable[imgNr] > 0)
//			{
//				int k;
//				for(k = 0; k < annotationData.falseGT[imgNr].size(); k++)
//				{
//					cv::Point2f tmpPnt = annotationData.falseGT[imgNr][k].first;
//					if((abs(rkp.x - tmpPnt.x) < 10 * FLT_MIN) && (abs(rkp.y - tmpPnt.y) < 10 * FLT_MIN))
//						break;
//				}
//				if(k < annotationData.falseGT[imgNr].size())
//				{
//					for(k = 0; k < annotationData.perfectMatches[imgNr].size(); k++)
//					{
//						cv::Point2f tmpPnt = annotationData.perfectMatches[imgNr][k].first;
//						if((abs(rkp.x - tmpPnt.x) < 10 * FLT_MIN) && (abs(rkp.y - tmpPnt.y) < 10 * FLT_MIN))
//							break;
//					}
//					if(k >= annotationData.perfectMatches[imgNr].size())
//						isNotMatchable = true;
//				}
//			}*/
//			if(!isNotMatchable)
//			{
//				diffdist = annotationData.distances[imgNr][i - missM];
//				distances.push_back(diffdist);
//				singleErrVec = annotationData.errvecs[imgNr][i - missM];
//				errvecs.push_back(singleErrVec);
//				perfectMatches.push_back(annotationData.perfectMatches[imgNr][i - missM]);
//				errvecsGT.push_back(annotationData.errvecsGT[imgNr][i - missM]);
//				validityValGT.push_back(annotationData.validityValGT[imgNr][i - missM]);
//				distancesGT.push_back(annotationData.distancesGT[imgNr][i - missM]);
//			}
//			else
//				missM++;
//
//			if((diffdist >= usedMatchTH) || isNotMatchable)
//			{
//				falseGT.push_back(std::make_pair(rkp, lkp));
//				wrongGTidxDist.push_back(std::make_pair(idx, diffdist));
//				if(flowGtIsUsed)
//				{
//					validityValFalseGT.push_back(channelsFlow[2].at<float>((int)floor(rkp.y + 0.5f), (int)floor(rkp.x + 0.5f)));
//				}
//				else
//				{
//					validityValFalseGT.push_back(1);
//				}
//			}
//			continue;
//#endif
//			stringstream ss;
//#ifndef _DEBUG
//			ss << "Diff: " << diffdist;
//#else
//			ss << "Fail! Diff: " << diffdist;
//#endif
//			str = ss.str();
//			putText(composed, str.c_str(), cv::Point2d(15, 90), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));
//		}
//
//		//Mark "new" matching position in left patch
//		Mat warped_patch0_c;
//		cv::cvtColor(warped_patch0, warped_patch0_c, CV_GRAY2RGB);
//		cv::Point2f newmidpos = cv::Point2f(std::floorf((float)warped_patch0.cols / 2.f), std::floorf((float)warped_patch0.rows / 2.f));
//		/*cv::line(warped_patch0_c, newmidpos - Point2f(5.0f, 0), newmidpos + Point2f(5.0f, 0), cv::Scalar(0, 0, 255));
//		cv::line(warped_patch0_c, newmidpos - Point2f(0, 5.0f), newmidpos + Point2f(0, 5.0f), cv::Scalar(0, 0, 255));*/
//
//		//Show right patch with equalized histogram
//		Mat patch_wdhist2_color, patch_wdhist2_color2;
//		cv::cvtColor(patch_wdhist2[1], patch_wdhist2_color, CV_GRAY2RGB);
//		patch_wdhist2_color2 = patch_wdhist2_color.clone();
//		newmidpos = cv::Point2f(std::floorf((float)patch_wdhist2[1].cols / 2.f), std::floorf((float)patch_wdhist2[1].rows / 2.f));
//		/*cv::line(patch_wdhist2_color, newmidpos - Point2f(5.0f, 0), newmidpos + Point2f(5.0f, 0), cv::Scalar(0, 0, 255));
//		cv::line(patch_wdhist2_color, newmidpos - Point2f(0, 5.0f), newmidpos + Point2f(0, 5.0f), cv::Scalar(0, 0, 255));*/
//
//		//Show left patch with equalized histogram
//		Mat leftHist_color;
//		cv::cvtColor(patch_wdhist2[0], leftHist_color, CV_GRAY2RGB);
//
//		//Generate warped, shifted & blended patches
//		Mat patches_warp[2], wsp_color[2], blended_w;
//		cv::cvtColor(warped_patch0, wsp_color[0], CV_GRAY2RGB);
//		cv::cvtColor(patch_wdhist2[1], wsp_color[1], CV_GRAY2RGB);
//		wsp_color[0].copyTo(patches_warp[0]);
//		wsp_color[1].copyTo(patches_warp[1]);
//		patches_warp[0].reshape(1, patches_warp[0].rows * patches_warp[0].cols).col(1).setTo(Scalar(0));
//		patches_warp[0].reshape(1, patches_warp[0].rows * patches_warp[0].cols).col(0).setTo(Scalar(0));
//		patches_warp[1].reshape(1, patches_warp[1].rows * patches_warp[1].cols).col(0).setTo(Scalar(0));
//		patches_warp[1].reshape(1, patches_warp[1].rows * patches_warp[1].cols).col(2).setTo(Scalar(0));
//		addWeighted(patches_warp[0], alpha, patches_warp[1], beta, 0.0, blended_w);
//		blended_w = blended_w(Rect(minmaxXY[0], minmaxXY[2], minmaxXY[1] - minmaxXY[0], minmaxXY[3] - minmaxXY[2])).clone();
//
//		
//		//Generate both images including the marked match
//		Mat bothimgs(imgSize.height, 2 * imgSize.width, CV_8UC3);
//#if LEFT_TO_RIGHT
//		str = "I1";
//#else
//		str = "I2";
//#endif
//		putText(show_color[0], str.c_str(), cv::Point2d(20, 20), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.6, cv::Scalar(0, 0, 255));
//		show_color[0].copyTo(bothimgs(Rect(Point(0, 0), imgSize)));
//#if LEFT_TO_RIGHT
//		str = "I2";
//#else
//		str = "I1";
//#endif
//		putText(show_color[1], str.c_str(), cv::Point2d(20, 20), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.6, cv::Scalar(0, 0, 255));
//		show_color[1].copyTo(bothimgs(Rect(Point(imgSize.width, 0), imgSize)));
//		cv::line(bothimgs, lkp, rkp + Point2f(imgSize.width, 0), cv::Scalar(0, 0, 255), 2);
//
//		//Scale images and form a single image out of them
//		cv::resize(bothimgs, composed(Rect(0, textheight, maxHorImgSize * 2, maxVerImgSize)), cv::Size(maxHorImgSize * 2, maxVerImgSize), 0, 0, INTER_LANCZOS4);
//		cv::resize(blended, composed(Rect(0, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow));
//		newmidpos = cv::Point2f(std::floorf((float)patchsizeShow / 2.f), std::floorf((float)patchsizeShow / 2.f));
//		cv::line(composed(Rect(0, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
//		cv::line(composed(Rect(0, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
//		str = "blended";
//		putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//		cv::resize(diffImg, composed(Rect(patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
//		str = "Diff";
//		putText(composed, str.c_str(), cv::Point2d(patchsizeShow + 5, maxVerImgSize + textheight + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//		cv::resize(patches[0], composed(Rect(2 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
//		cv::line(composed(Rect(2 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
//		cv::line(composed(Rect(2 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
//#if LEFT_TO_RIGHT
//		str = "left";
//#else
//		str = "right";
//#endif
//		putText(composed, str.c_str(), cv::Point2d(2 * patchsizeShow + 5, maxVerImgSize + textheight + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//		cv::resize(patches[1], composed(Rect(3 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
//		cv::line(composed(Rect(3 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
//		cv::line(composed(Rect(3 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
//#if LEFT_TO_RIGHT
//		str = "right";
//#else
//		str = "left";
//#endif
//		putText(composed, str.c_str(), cv::Point2d(3 * patchsizeShow + 5, maxVerImgSize + textheight + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//		cv::resize(warped_patch0_c, composed(Rect(4 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
//		cv::line(composed(Rect(4 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
//		cv::line(composed(Rect(4 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
//#if LEFT_TO_RIGHT
//		str = "NEW left position";
//#else
//		str = "NEW right position";
//#endif
//		putText(composed, str.c_str(), cv::Point2d(4 * patchsizeShow + 5, maxVerImgSize + textheight + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//		cv::resize(patch_wdhist2_color, composed(Rect(5 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
//		cv::line(composed(Rect(5 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
//		cv::line(composed(Rect(5 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
//#if LEFT_TO_RIGHT
//		str = "right equal hist";
//#else
//		str = "left equal hist";
//#endif
//		putText(composed, str.c_str(), cv::Point2d(5 * patchsizeShow + 5, maxVerImgSize + textheight + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//		cv::resize(blended_w, composed(Rect(6 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
//		str = "NEW (warped) blended";
//		putText(composed, str.c_str(), cv::Point2d(6 * patchsizeShow + 5, maxVerImgSize + textheight + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//		cv::resize(diffImg2, composed(Rect(7 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
//		str = "NEW (warped) diff";
//		putText(composed, str.c_str(), cv::Point2d(7 * patchsizeShow + 5, maxVerImgSize + textheight + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//		/*cv::resize(resultimg_color, composed(Rect(8 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
//		str = "Result";
//		putText(composed, str.c_str(), cv::Point2d(9 * patchsizeShow + 5, maxVerImgSize + textheight + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));*/
//		cv::resize(leftHist_color, composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//		newmidpos = cv::Point2f(std::floorf((float)patchSelectShow / 2.f), std::floorf((float)patchSelectShow / 2.f));
//		cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 140, 0));
//		cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 140, 0));
//#if LEFT_TO_RIGHT
//		str = "left equal hist - select pt";
//#else
//		str = "right equal hist - select pt";
//#endif
//		putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//		cv::resize(patch_wdhist2_color2, composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//		cv::line(composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
//		cv::line(composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
//#if LEFT_TO_RIGHT
//		str = "right equal hist";
//#else
//		str = "left equal hist";
//#endif
//		putText(composed, str.c_str(), cv::Point2d(patchSelectShow + 5, maxVerImgSize + textheight + patchsizeShow + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//		
//		str = "Press 'space' to accept the estimated or selected (preferred, if available) location, 'e' for the estimated, and 'i' for the initial location. Hit 'n' if not matchable, 's' to skip the match, and";
//		putText(composed, str.c_str(), cv::Point2d(15, 15), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
//#if LEFT_TO_RIGHT
//		str = "ESC to stop testing. To specify a new matching position, click at the desired position inside the area of 'left equal hist - select pt'. To refine the location, use the arrow keys.";
//#else
//		str = "ESC to stop testing. To specify a new matching position, click at the desired position inside the area of 'right equal hist - select pt'. To refine the location, use the arrow keys.";
//#endif
//		putText(composed, str.c_str(), cv::Point2d(15, 35), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
//#if LEFT_TO_RIGHT
//		str = "To select a new patch for manual refinement at a different location, hold 'Strg' while clicking at the desired center position inside the left image 'I1'. Press 'h' for more options.";
//#else
//		str = "To select a new patch for manual refinement at a different location, hold 'Strg' while clicking at the desired center position inside the left image 'I2'. Press 'h' for more options.";
//#endif
//		putText(composed, str.c_str(), cv::Point2d(15, 55), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
//		{
//			stringstream ss;
//			ss << "Remaining matches: " << maxSampleSize - i - 1 << "  Remaining Images: " << remainingImgs;
//			str = ss.str();
//			putText(composed, str.c_str(), cv::Point2d(15, 73), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));
//		}
//		
//		cvNamedWindow("GT match");
//		cv::Point2f winPos = cv::Point2f(-FLT_MAX, -FLT_MAX), winPos2 = cv::Point2f(-FLT_MAX, -FLT_MAX), winPosOld = cv::Point2f(-FLT_MAX, -FLT_MAX);
//		cvSetMouseCallback("GT match", on_mouse_click, (void*)(&winPos) );
//		cv::imshow("GT match",composed);
//		minmaxXY[0] = 0;//minX
//		minmaxXY[1] = patchSelectShow;//maxX
//		minmaxXY[2] = maxVerImgSize + textheight + patchsizeShow;//minY
//		minmaxXY[3] = minmaxXY[2] + patchSelectShow;//maxY
//		double shownLeftImgBorder[4];
//		double diffdist2 = 0, diffdist2last = 0;
//		cv::Point2f singleErrVec2;
//		unsigned int refiters = 0;
//		int c;
//		bool noIterRef = false;
//		static bool noIterRefProg = false;
//		unsigned int manualMode = 3, diffManualMode = 3;
//		bool manualHomoDone = false;
//		int autocalc2 = 0;
//		cv::Mat Hmanual, Hmanual2 = Mat::eye(2,3,CV_64FC1);
//		double manualScale = 1.0, manualRot = 0;
//		cv::Point2f oldpos = cv::Point2f(0,0);;
//		SpecialKeyCode skey = NONE;
//		pdiff[1] = cv::Point2f(0,0);
//		shownLeftImgBorder[0] = 0;//minX
//		shownLeftImgBorder[1] = maxHorImgSize;//maxX
//		shownLeftImgBorder[2] = textheight;//minY
//		shownLeftImgBorder[3] = textheight + maxVerImgSize;//maxY
//		do
//		{
//			c = cv::waitKey(30);
//
//			if(c != -1)
//				skey = getSpecialKeyCode(c);
//			else
//				skey = NONE;
//
//			if(!manualHomoDone)
//			{
//				manualHomoDone = true;
//				if(manualMode != 3)//Use homography for left shown image
//				{
//						equalizeHist(patches_wdiff[0], patch_wdhist2[0]);
//						switch(manualMode)
//						{
//						case 0://Use the homography from GT or the one that was calculated using matches within the left and right patches and properties of SIFT keypoints
//							if(flowGtIsUsed)
//								warpAffine(patch_wdhist2[0], patch_wdhist2[0], Haff, patch_wdhist2[0].size(), INTER_LANCZOS4);
//							else
//								warpPerspective(patch_wdhist2[0], patch_wdhist2[0], Haff, patch_wdhist2[0].size(), INTER_LANCZOS4);
//							break;
//						case 1://Use the homography that was calculated using matches within the left and right patches only
//							if(Hl.empty())
//							{
//								MessageBox(NULL, "The feature-based homography is not available for this match!", "Homography not available", MB_OK | MB_ICONINFORMATION);
//								manualMode = 3;
//								manualHomoDone = false;
//							}
//							else
//							{
//								Hl = Hl.colRange(0,2);
//								Mat origPos1 = (Mat_<double>(2,1) << (double)halfpatch, (double)halfpatch);
//								Mat tm1;
//								tm1 = Hl * origPos1;//align the translkation vector of the homography in a way that the middle of the patch has the same coordinate
//								tm1 = origPos1 - tm1;
//								cv::hconcat(Hl, tm1, Hl);//Construct the final affine homography
//								warpAffine(patch_wdhist2[0], patch_wdhist2[0], Hl, patch_wdhist2[0].size(), INTER_LANCZOS4);
//							}
//							break;
//						case 2://Select 4 points to calculate a new homography
//							{
//								MessageBox(NULL, "Select 4 matching points in each patch to generate a homography (in the order 1st left, 1st right, 2nd left, 2nd right, ...). They must not be on a line! To abort while selecting, press 'ESC'.", "Manual homography estimation", MB_OK | MB_ICONINFORMATION);
//								cv::cvtColor(patch_wdhist2[0], leftHist_color, CV_GRAY2RGB);
//								cv::resize(leftHist_color, composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//								cv::resize(patch_wdhist2_color2, composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//								Mat selectPatches[2];
//								vector<cv::Point2f> lps, rps;
//								cv::resize(leftHist_color, selectPatches[0], cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//								cv::resize(patch_wdhist2_color2, selectPatches[1], cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//								string numstr[4] = {"first","second","third","fourth"};
//								cv::Scalar colours[4] = {cv::Scalar(0, 0, 255),cv::Scalar(0, 255, 0),cv::Scalar(255, 0, 0),cv::Scalar(209, 35, 233)};
//								double minmaxXYrp[2];
//								bool noSkipmode = true;
//								minmaxXYrp[0] = patchSelectShow;//minX
//								minmaxXYrp[1] = 2* patchSelectShow;//maxX
//								for(int k = 0; k < 4; k++)
//								{
//									str = "Select " + numstr[k] + " point";
//									putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//									winPos = cv::Point2f(-FLT_MAX, -FLT_MAX);
//									cv::imshow("GT match",composed);
//									bool lineardep = false;
//									do
//									{
//										lineardep = false;
//										while((winPos.x < minmaxXY[0]) || (winPos.x >= minmaxXY[1]) || (winPos.y < minmaxXY[2]) || (winPos.y >= minmaxXY[3]))
//										{
//											c = cv::waitKey(30);
//											if(c != -1)
//												skey = getSpecialKeyCode(c);
//											else
//												skey = NONE;
//											if(skey == ESCAPE)
//											{
//												skey = NONE;
//												c = -1;
//												k = 4;
//												noSkipmode = false;
//												break;
//											}
//										}
//										if(noSkipmode)
//										{
//											lps.push_back(cv::Point2f(winPos.x, winPos.y - (maxVerImgSize + textheight + patchsizeShow)));
//											// check that the i-th selected point does not belong
//											// to a line connecting some previously selected points
//											for(int j = 0; j < k; j++ )
//											{
//												double dx1 = lps[j].x - lps.back().x;
//												double dy1 = lps[j].y - lps.back().y;
//												for(int k1 = 0; k1 < j; k1++ )
//												{
//													double dx2 = lps[k1].x - lps.back().x;
//													double dy2 = lps[k1].y - lps.back().y;
//													if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
//														lineardep = true;
//												}
//											}
//											if(lineardep)
//											{
//												lps.pop_back();
//												MessageBox(NULL, "Selection is linear dependent - select a different one!", "Linear dependency", MB_OK | MB_ICONINFORMATION);
//											}
//										}
//									}
//									while(lineardep);
//									if(noSkipmode)
//									{
//										cv::line(selectPatches[0], lps.back() - Point2f(10.0f, 0), lps.back() + Point2f(10.0f, 0), colours[k]);
//										cv::line(selectPatches[0], lps.back() - Point2f(0, 10.0f), lps.back() + Point2f(0, 10.0f), colours[k]);
//										selectPatches[0].copyTo(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)));
//										//cv::resize(selectPatches[0], composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//										lps.back().x /= (float)selMult;
//										lps.back().y /= (float)selMult;
//
//										putText(composed, str.c_str(), cv::Point2d(patchSelectShow + 5, maxVerImgSize + textheight + patchsizeShow + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//										cv::imshow("GT match",composed);
//									}
//									while(((winPos.x < minmaxXYrp[0]) || (winPos.x >= minmaxXYrp[1]) || (winPos.y < minmaxXY[2]) || (winPos.y >= minmaxXY[3])) && noSkipmode)
//									{
//										c = cv::waitKey(30);
//										if(c != -1)
//											skey = getSpecialKeyCode(c);
//										else
//											skey = NONE;
//										if(skey == ESCAPE)
//										{
//											skey = NONE;
//											c = -1;
//											k = 4;
//											noSkipmode = false;
//											break;
//										}
//									}
//									if(noSkipmode)
//									{
//										rps.push_back(cv::Point2f(winPos.x - patchSelectShow, winPos.y - (maxVerImgSize + textheight + patchsizeShow)));
//										cv::line(selectPatches[1], rps.back() - Point2f(10.0f, 0), rps.back() + Point2f(10.0f, 0), colours[k]);
//										cv::line(selectPatches[1], rps.back() - Point2f(0, 10.0f), rps.back() + Point2f(0, 10.0f), colours[k]);
//										selectPatches[1].copyTo(composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)));
//										//cv::resize(selectPatches[1], composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//										rps.back().x /= (float)selMult;
//										rps.back().y /= (float)selMult;
//									}
//								}
//								if(noSkipmode)
//								{
//									cv::imshow("GT match",composed);
//									c = cv::waitKey(1300);
//									Hmanual = cv::estimateRigidTransform(lps, rps, true);//estimate an affine homography
//									if(Hmanual.empty())
//									{
//										MessageBox(NULL, "It was not possible to estimate a homography from these correspondences!", "Homography not available", MB_OK | MB_ICONINFORMATION);
//										manualMode = 3;
//										manualHomoDone = false;
//									}
//									else
//									{
//										Mat Hmanual1 = Hmanual.colRange(0,2);
//										Mat origPos1 = (Mat_<double>(2,1) << (double)halfpatch, (double)halfpatch);
//										Mat tm1;
//										tm1 = Hmanual1 * origPos1;//align the translation vector of the homography in a way that the middle of the patch has the same coordinate
//										tm1 = origPos1 - tm1;
//										cv::hconcat(Hmanual1, tm1, Hmanual1);//Construct the final affine homography
//										warpAffine(patch_wdhist2[0], patch_wdhist2[0], Hmanual1, patch_wdhist2[0].size(), INTER_LANCZOS4);
//										winPos.x = (float)selMult * ((float)Hmanual.at<double>(0,2) - (float)tm1.at<double>(0) + halfpatch);
//										winPos.y = (float)selMult * ((float)Hmanual.at<double>(1,2) - (float)tm1.at<double>(1) + halfpatch) + (float)(maxVerImgSize + textheight + patchsizeShow);
//										noIterRef = false;
//									}
//								}
//								cv::resize(patch_wdhist2_color2, composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//								cv::line(composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
//								cv::line(composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
//#if LEFT_TO_RIGHT
//								str = "right equal hist";
//#else
//								str = "left equal hist";
//#endif
//								putText(composed, str.c_str(), cv::Point2d(patchSelectShow + 5, maxVerImgSize + textheight + patchsizeShow + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//							break;
//							}
//							case 4://Use the homography that was calculated using the user input for scale and rotation
//							{
//								Hmanual2 = Hmanual2.colRange(0,2);
//								Mat origPos1 = (Mat_<double>(2,1) << (double)halfpatch, (double)halfpatch);
//								Mat tm1;
//								tm1 = Hmanual2 * origPos1;//align the translkation vector of the homography in a way that the middle of the patch has the same coordinate
//								tm1 = origPos1 - tm1;
//								cv::hconcat(Hmanual2, tm1, Hmanual2);//Construct the final affine homography
//								warpAffine(patch_wdhist2[0], patch_wdhist2[0], Hmanual2, patch_wdhist2[0].size(), INTER_LANCZOS4);
//							break;
//							}
//						}
//						cv::cvtColor(patch_wdhist2[0], leftHist_color, CV_GRAY2RGB);
//				}
//				else
//				{
//					equalizeHist(patches_wdiff[0], patch_wdhist2[0]);
//					cv::cvtColor(patch_wdhist2[0], leftHist_color, CV_GRAY2RGB);
//				}
//				cv::resize(leftHist_color, composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//				cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 140, 0));
//				cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 140, 0));
//#if LEFT_TO_RIGHT
//				str = "left equal hist - select pt";
//#else
//				str = "right equal hist - select pt";
//#endif
//				putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//				cv::imshow("GT match",composed);
//			}
//
//			if(autocalc2 > 0)//find a global minimum (best template match within patch) on the patches for manual selection
//			{
//				iterativeTemplMatch(patch_wdhist2[0], largerpatchhist, patchsizeOrig, winPos, autocalc2);
//				winPos = -winPos;
//				winPos.y += halfpatch;
//				winPos.x += halfpatch;
//				winPos.y *= (float)selMult;
//				winPos.x *= (float)selMult;
//				winPos.y += (float)(maxVerImgSize + textheight + patchsizeShow);
//				noIterRef = true;//deactivate local refinement
//				autocalc2 = 0;
//			}
//
//			if((winPos.x >= minmaxXY[0]) && (winPos.x < minmaxXY[1]) && (winPos.y >= minmaxXY[2]) && (winPos.y < minmaxXY[3]))//if a position inside the "manual selection" area was chosen
//			{
//				cv::Point2f newmidpos2, singleErrVec2Old;
//				int autseait = 0;
//				cv::Point2f addWinPos = cv::Point2f(0,0);
//				int direction = 0;//0=left, 1=right, 2=up, 3=down
//				int oldDir[4] = {0,0,0,0};
//				int horVerCnt[2] = {0,0};
//				int atMinimum = 0;
//				double diffdist2Old;
//
//				if(skey == ARROW_UP) //up key is pressed
//					winPos.y -= 0.25f;
//				else if(skey == ARROW_DOWN) //down key is pressed
//					winPos.y += 0.25f;
//				else if(skey == ARROW_LEFT) //left key is pressed
//					winPos.x -= 0.25f;
//				else if(skey == ARROW_RIGHT) //right key is pressed
//					winPos.x += 0.25f;
//
//				winPos2 = winPos;
//				do
//				{
//					cv::resize(leftHist_color, composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//					if((abs(pdiff[1].x) <= 10 * FLT_MIN) && (abs(pdiff[1].y) <= 10 * FLT_MIN))
//					{
//						cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 140, 0));
//						cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 140, 0));
//					}
//
//					newmidpos2 = cv::Point2f(winPos.x, winPos.y - (maxVerImgSize + textheight + patchsizeShow));
//					cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos2 - Point2f(10.0f, 0), newmidpos2 + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
//					cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos2 - Point2f(0, 10.0f), newmidpos2 + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
//#if LEFT_TO_RIGHT
//					str = "left equal hist - select pt";
//#else
//					str = "right equal hist - select pt";
//#endif
//					putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//
//					//shift left img
//					compDiff = Mat::eye(2,2,CV_64F);
//					pdiff[0] = cv::Point2f((float)(newmidpos.x - newmidpos2.x) / (float)selMult, (float)(newmidpos.y - newmidpos2.y) / (float)selMult);
//					//newmidpos2 = pdiff[1] - pdiff[0];
//					if(refiters == 0)
//					{
//						diffManualMode = manualMode;
//					}
//
//					bool recalc;
//					do
//					{
//						recalc = false;
//						if(diffManualMode == 3)
//						{
//							newmidpos2 = pdiff[1] - pdiff[0];
//							singleErrVec2 = newmidpos2;
//							diffdist2 = (double)std::sqrt(newmidpos2.x * newmidpos2.x + newmidpos2.y * newmidpos2.y);
//						}
//						else if(diffManualMode == 2)
//						{
//							newmidpos2 = -pdiff[0];
//							Mat tdiff1 = (Mat_<double>(2,1) << (double)newmidpos2.x, (double)newmidpos2.y);
//							tdiff1 = Hmanual.colRange(0,2).inv() * tdiff1;//Calculate the displacement in pixels of the original image (not warped)
//							tdiff1 += (Mat_<double>(2,1) << (double)pdiff[1].x, (double)pdiff[1].y);
//							diffdist2 = (double)std::sqrt(tdiff1.at<double>(0) * tdiff1.at<double>(0) + tdiff1.at<double>(1) * tdiff1.at<double>(1));
//							singleErrVec2 = cv::Point2f((float)tdiff1.at<double>(0), (float)tdiff1.at<double>(1));
//						}
//						else if(diffManualMode == 1)
//						{
//							newmidpos2 = -pdiff[0];
//							Mat tdiff1 = (Mat_<double>(2,1) << (double)newmidpos2.x, (double)newmidpos2.y);
//							tdiff1 = Hl.colRange(0,2).inv() * tdiff1;//Calculate the displacement in pixels of the original image (not warped)
//							tdiff1 += (Mat_<double>(2,1) << (double)pdiff[1].x, (double)pdiff[1].y);
//							diffdist2 = (double)std::sqrt(tdiff1.at<double>(0) * tdiff1.at<double>(0) + tdiff1.at<double>(1) * tdiff1.at<double>(1));
//							singleErrVec2 = cv::Point2f((float)tdiff1.at<double>(0), (float)tdiff1.at<double>(1));
//						}
//						else if(diffManualMode == 0)
//						{
//							Mat tdiff1;
//							newmidpos2 = -pdiff[0];
//							if(flowGtIsUsed)
//							{
//								tdiff1 = (Mat_<double>(2,1) << (double)newmidpos2.x, (double)newmidpos2.y);
//								tdiff1 = Haff.colRange(0,2).inv() * tdiff1;//Calculate the displacement in pixels of the original image (not warped)
//								tdiff1 += (Mat_<double>(2,1) << (double)pdiff[1].x, (double)pdiff[1].y);
//							}
//							else
//							{
//								tdiff1 = (Mat_<double>(3,1) << (double)newmidpos2.x, (double)newmidpos2.y, 1.0);
//
//								Mat Haffi = Mat::eye(3,3,homoGT_exch.type());
//								Haffi.at<double>(0,2) = (double)rkp.x;
//								Haffi.at<double>(1,2) = (double)rkp.y;
//								Haffi = homoGT_exch.inv() * Haffi;
//								Mat helpH1 = Mat::eye(3,3,homoGT_exch.type());
//								helpH1.at<double>(0,2) = -Haffi.at<double>(0,2) / Haffi.at<double>(2,2);
//								helpH1.at<double>(1,2) = -Haffi.at<double>(1,2) / Haffi.at<double>(2,2);
//								Haffi = helpH1 * Haffi;
//								tdiff1 = Haffi * tdiff1;//Calculate the displacement in pixels of the original image (not warped)
//								tdiff1 /= tdiff1.at<double>(2);
//								tdiff1 += (Mat_<double>(3,1) << (double)pdiff[1].x, (double)pdiff[1].y, 0);
//							}
//							diffdist2 = (double)std::sqrt(tdiff1.at<double>(0) * tdiff1.at<double>(0) + tdiff1.at<double>(1) * tdiff1.at<double>(1));
//							singleErrVec2 = cv::Point2f((float)tdiff1.at<double>(0), (float)tdiff1.at<double>(1));
//						}
//						else if(diffManualMode == 4)
//						{
//							newmidpos2 = -pdiff[0];
//							Mat tdiff1 = (Mat_<double>(2,1) << (double)newmidpos2.x, (double)newmidpos2.y);
//							tdiff1 = Hmanual2.colRange(0,2).inv() * tdiff1;//Calculate the displacement in pixels of the original image (not warped)
//							tdiff1 += (Mat_<double>(2,1) << (double)pdiff[1].x, (double)pdiff[1].y);
//							diffdist2 = (double)std::sqrt(tdiff1.at<double>(0) * tdiff1.at<double>(0) + tdiff1.at<double>(1) * tdiff1.at<double>(1));
//							singleErrVec2 = cv::Point2f((float)tdiff1.at<double>(0), (float)tdiff1.at<double>(1));
//						}
//						if((abs(diffdist2last - diffdist2) > DBL_EPSILON) || (refiters == 0))
//						{
//							diffdist2last = diffdist2;
//							if(diffManualMode != manualMode)
//								recalc = true;
//							diffManualMode = manualMode;
//						}
//						if(refiters < UINT_MAX)
//							refiters++;
//						else
//							refiters = 1;
//					}
//					while(recalc);
//
//					tdiff = (Mat_<double>(2,1) << (double)pdiff[0].x, (double)pdiff[0].y);
//					cv::hconcat(compDiff, tdiff, compDiff);
//					warpAffine(patch_wdhist2[0], shiftedpatch, compDiff, patch_wdhist2[0].size(), INTER_LANCZOS4); //Shift the left patch according to the value from template matching
//
//					//show blended image
//					Mat newshift_color[2];
//					cv::cvtColor(shiftedpatch, newshift_color[0], CV_GRAY2RGB);
//					patch_wdhist2_color2.copyTo(newshift_color[1]);
//					newshift_color[0].reshape(1, newshift_color[0].rows * newshift_color[0].cols).col(1).setTo(Scalar(0));
//					newshift_color[0].reshape(1, newshift_color[0].rows * newshift_color[0].cols).col(0).setTo(Scalar(0));
//					newshift_color[1].reshape(1, newshift_color[1].rows * newshift_color[1].cols).col(0).setTo(Scalar(0));
//					newshift_color[1].reshape(1, newshift_color[1].rows * newshift_color[1].cols).col(2).setTo(Scalar(0));
//					addWeighted(newshift_color[0], alpha, newshift_color[1], beta, 0.0, blended_w);
//					cv::resize(blended_w, composed(Rect(2 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//					cv::line(composed(Rect(2 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
//					cv::line(composed(Rect(2 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
//					str = "shifted blended";
//					putText(composed, str.c_str(), cv::Point2d(2 * patchSelectShow + 5, maxVerImgSize + textheight + patchsizeShow + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//
//					//show Diff
//					Mat userDiff_color;
//					absdiff(shiftedpatch, patch_wdhist2[1], patch_equal1);
//					cv::cvtColor(patch_equal1, userDiff_color, CV_GRAY2RGB);
//					cv::resize(userDiff_color, composed(Rect(3 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//					cv::line(composed(Rect(3 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
//					cv::line(composed(Rect(3 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
//					str = "shifted Diff";
//					putText(composed, str.c_str(), cv::Point2d(3 * patchSelectShow + 5, maxVerImgSize + textheight + patchsizeShow + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//
//					//get sum of difference near match
//					if(autseait == 0)
//					{
//						errlevels[0] = cv::sum(patch_equal1(Rect(quarterpatch + eigthpatch, quarterpatch + eigthpatch, quarterpatch, quarterpatch)))[0];
//					}
//					else
//					{
//						errlevels[1] = cv::sum(patch_equal1(Rect(quarterpatch + eigthpatch, quarterpatch + eigthpatch, quarterpatch, quarterpatch)))[0];
//					}
//
//					stringstream ss;
//					ss << "New Diff: " << diffdist2 << " Sum of differences: " << errlevels[0];
//					str = ss.str();
//					composed(Rect(0, maxVerImgSize + textheight + patchsizeShow + patchSelectShow, composed.cols, bottomtextheight)).setTo(0);
//					putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + patchSelectShow + 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));
//
//					if(noIterRef || noIterRefProg)
//					{
//						cv::imshow("GT match",composed);
//					}
//					else
//					{
//						if(autseait == 0)
//						{
//							cv::imshow("GT match",composed);
//							winPosOld = winPos;
//							diffdist2Old = diffdist2;
//							singleErrVec2Old = singleErrVec2;
//						}
//						else //local refinement of the manual selected match position
//						{
//							if(errlevels[1] >= errlevels[0])
//							{
//								switch(direction)
//								{
//								case 0:
//									addWinPos.x += 0.25f;
//									if((oldDir[0] > 0) || (horVerCnt[0] >= 2))
//									{
//										direction = 2;
//										if(oldDir[0] > 0)
//											atMinimum = 0;
//									}
//									else
//									{
//										direction = 1;
//										atMinimum++;
//										horVerCnt[0]++;
//									}
//									oldDir[0] = 0;
//									horVerCnt[1] = 0;
//									break;
//								case 1:
//									addWinPos.x -= 0.25f;
//									if((oldDir[1] > 0) || (horVerCnt[0] >= 2))
//									{
//										direction = 2;
//										if(oldDir[1] > 0)
//											atMinimum = 0;
//									}
//									else
//									{
//										direction = 0;
//										atMinimum++;
//										horVerCnt[0]++;
//									}
//									oldDir[1] = 0;
//									horVerCnt[1] = 0;
//									break;
//								case 2:
//									addWinPos.y += 0.25f;
//									if((oldDir[2] > 0) || (horVerCnt[1] >= 2))
//									{
//										direction = 0;
//										if(oldDir[2] > 0)
//											atMinimum = 0;
//									}
//									else
//									{
//										direction = 3;
//										atMinimum++;
//										horVerCnt[1]++;
//									}
//									oldDir[2] = 0;
//									horVerCnt[0] = 0;
//									break;
//								case 3:
//									addWinPos.y -= 0.25f;
//									if((oldDir[3] > 0) || (horVerCnt[1] >= 2))
//									{
//										direction = 0;
//										if(oldDir[3] > 0)
//											atMinimum = 0;
//									}
//									else
//									{
//										direction = 2;
//										atMinimum++;
//										horVerCnt[1]++;
//									}
//									oldDir[3] = 0;
//									horVerCnt[0] = 0;
//									break;
//								}
//							}
//							else
//							{
//								oldDir[direction]++;
//								errlevels[0] = errlevels[1];
//								if((abs(winPos.x - winPosOld.x) > 0.3) || (abs(winPos.y - winPosOld.y) > 0.3))
//									noIterRef = true;
//								winPosOld = winPos;
//								diffdist2Old = diffdist2;
//								singleErrVec2Old = singleErrVec2;
//								cv::imshow("GT match",composed);
//								c = cv::waitKey(100);
//								if(c != -1)
//									skey = getSpecialKeyCode(c);
//								else
//									skey = NONE;
//								if(skey == ESCAPE)
//								{
//									winPos = winPos2;
//									noIterRef = true;
//									skey = NONE;
//									c = -1;
//								}
//
//							}
//						}
//						switch(direction)
//						{
//						case 0:
//							addWinPos.x -= 0.25f;
//							break;
//						case 1:
//							addWinPos.x += 0.25f;
//							break;
//						case 2:
//							addWinPos.y -= 0.25f;
//							break;
//						case 3:
//							addWinPos.y += 0.25f;
//							break;
//						}
//						if(!noIterRef && ((abs(winPos.x - winPosOld.x) <= 0.3) && (abs(winPos.y - winPosOld.y) <= 0.3)))
//						{
//							if(atMinimum < 5)
//								winPos = winPos2 + addWinPos;
//							else
//							{
//								diffdist2 = diffdist2Old;
//								singleErrVec2 = singleErrVec2Old;
//							}
//						}
//						autseait++;
//					}
//				}
//				while((winPos.x >= minmaxXY[0]) && (winPos.x < minmaxXY[1]) && (winPos.y >= minmaxXY[2]) && (winPos.y < minmaxXY[3])
//					  && (addWinPos.x < 5.f) && (addWinPos.y < 5.f) && !noIterRef && (atMinimum < 5) && !noIterRefProg);
//				if((winPos.x < minmaxXY[0]) || (winPos.x >= minmaxXY[1]) || (winPos.y < minmaxXY[2]) || (winPos.y >= minmaxXY[3])
//					  || (addWinPos.x >= 5.f) || (addWinPos.y >= 5.f) || noIterRef)
//				{
//					noIterRef = true;
//				}
//				if(atMinimum > 4)
//				{
//					noIterRef = true;
//					winPos2 = winPosOld;//winPos;
//				}
//			}
//			else if((-winPos.x >= shownLeftImgBorder[0]) && (-winPos.x < shownLeftImgBorder[1]) && (-winPos.y >= shownLeftImgBorder[2]) && (-winPos.y < shownLeftImgBorder[3]))
//			{
//				pdiff[1] = -winPos;
//				pdiff[1].y -= textheight;
//				pdiff[1].x *= (float)imgSize.width / (float)maxHorImgSize;
//				pdiff[1].y *= (float)imgSize.height / (float)maxVerImgSize;
//
//				patches_wdiff[0] = img_wdiff[0](Rect(pdiff[1], Size(patchsizeOrig, patchsizeOrig))).clone();
//				equalizeHist(patches_wdiff[0], patch_wdhist2[0]);
//				cv::cvtColor(patch_wdhist2[0], leftHist_color, CV_GRAY2RGB);
//				cv::resize(leftHist_color, composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
//#if LEFT_TO_RIGHT
//				str = "left equal hist - select pt";
//#else
//				str = "right equal hist - select pt";
//#endif
//				putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
//
//				composed(Rect(2 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)).setTo(0);
//				composed(Rect(3 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)).setTo(0);
//				composed(Rect(0, maxVerImgSize + textheight + patchsizeShow + patchSelectShow, composed.cols, bottomtextheight)).setTo(0);
//				cv::imshow("GT match",composed);
//
//				pdiff[1] -= lkp;
//				winPos = cv::Point2f(-FLT_MAX, -FLT_MAX);
//				winPos2 = winPos;
//				manualMode = 3;
//			}
//			else
//			{
//				winPos = cv::Point2f(-FLT_MAX, -FLT_MAX);
//			}
//
//			if(skey == POS1)
//			{
//				noIterRef = false;
//			}
//
//			if(skey == ESCAPE)
//			{
//				int answere;
//				answere = MessageBox(NULL, "Are you sure? Reviewed matches of current image pair will be lost!", "Exit?", MB_YESNO | MB_DEFBUTTON1);
//				if(answere == IDYES)
//				{
//					cv::destroyWindow("GT match");
//					return -1;
//				}
//				else
//				{
//					skey = NONE;
//					c = -1;
//				}
//			}
//
//			if(c == 'm')//Deactivate automatic minimum search in manual mode forever
//			{
//				string mbtext;
//				std::ostringstream strstr;
//				noIterRefProg = !noIterRefProg;
//				strstr << "The automatic minimum search after manual match point selection is now " << (noIterRefProg ? "deactivated.":"activated.");
//				mbtext = strstr.str();
//				MessageBox(NULL, mbtext.c_str(), "Activation of automatic minimum search", MB_OK | MB_ICONINFORMATION);
//				skey = NONE;
//				c = -1;
//			}
//
//			if(c == 'k')//Deactivate automatic homography estimation based on SIFT keypoints forever for speedup
//			{
//				string mbtext;
//				std::ostringstream strstr;
//				autoHestiSift = !autoHestiSift;
//				strstr << "The automatic homography estimation based on SIFT keypoints is now " << (autoHestiSift ? "activated.":"deactivated.");
//				mbtext = strstr.str();
//				MessageBox(NULL, mbtext.c_str(), "Activation of automatic homography estimation", MB_OK | MB_ICONINFORMATION);
//				skey = NONE;
//				c = -1;
//			}
//
//			if(c == '0')
//			{
//				manualMode = 0;
//				if(manualHomoDone)
//					manualHomoDone = false;
//				skey = NONE;
//				c = -1;
//			}
//			else if(c == '1')
//			{
//				manualMode = 1;
//				if(manualHomoDone)
//					manualHomoDone = false;
//				skey = NONE;
//				c = -1;
//			}
//			else if(c == '2')
//			{
//				manualMode = 2;
//				if(manualHomoDone)
//					manualHomoDone = false;
//				skey = NONE;
//				c = -1;
//			}
//			else if(c == '3')
//			{
//				manualMode = 3;
//				if(manualHomoDone)
//					manualHomoDone = false;
//				skey = NONE;
//				c = -1;
//			}
//			else if(c == '+')
//			{
//				manualMode = 4;
//				if(manualHomoDone)
//					manualHomoDone = false;
//				skey = NONE;
//				c = -1;
//				manualScale *= 1.05;
//				Hmanual2 *= 1.05;
//			}
//			else if(c == '-')
//			{
//				manualMode = 4;
//				if(manualHomoDone)
//					manualHomoDone = false;
//				skey = NONE;
//				c = -1;
//				manualScale /= 1.05;
//				Hmanual2 /= 1.05;
//			}
//			else if(c == 'r')
//			{
//				manualMode = 4;
//				if(manualHomoDone)
//					manualHomoDone = false;
//				skey = NONE;
//				c = -1;
//				manualRot += 3.14159265 / 64;
//				Mat Rrot = (Mat_<double>(2,2) << std::cos(manualRot), (-1. * std::sin(manualRot)), std::sin(manualRot), std::cos(manualRot)); //generate the new rotation matrix
//				Hmanual2.colRange(0,2) = manualScale * Rrot;
//			}
//			else if(c == 'l')
//			{
//				manualMode = 4;
//				if(manualHomoDone)
//					manualHomoDone = false;
//				skey = NONE;
//				c = -1;
//				manualRot -= 3.14159265 / 64;
//				Mat Rrot = (Mat_<double>(2,2) << std::cos(manualRot), (-1. * std::sin(manualRot)), std::sin(manualRot), std::cos(manualRot)); //generate the new rotation matrix
//				Hmanual2.colRange(0,2) = manualScale * Rrot;
//			}
//
//
//			if(c == 'a')
//			{
//				int waitvalue = 1500;
//				do
//				{
//					c = cv::waitKey(waitvalue);
//					waitvalue = 5000;
//					if(c != -1)
//						skey = getSpecialKeyCode(c);
//					else
//					{
//						skey = NONE;
//						MessageBox(NULL, "Specify the minimal patch size factor (the optimal patch size is automatically chosen but is restricted to be larger than the given value) for finding the optimal matching position:\n 1 - Original patch size\n 2 - Half the size\n 3 - Size dived by 4\n 4 - Size dived by 8\n 5 - Size dived by 16\n Please press one of the above mentioned buttons after clicking 'OK' within the next 5 seconds or abort by pressing 'ESC'. Otherwise this message is shown again.", "Specify patch size", MB_OK | MB_ICONINFORMATION);
//					}
//					if(skey == ESCAPE)
//					{
//						skey = NONE;
//						c = -1;
//						break;
//					}
//				}
//				while((c != '1') && (c != '2') && (c != '3') && (c != '4') && (c != '5'));
//
//				if((c == '1') || (c == '2') || (c == '3') || (c == '4') || (c == '5'))
//				{
//					autocalc2 = atoi((char*)(&c));
//				}
//				skey = NONE;
//				c = -1;
//			}
//
//			if(c == 'h')
//			{
//#if LEFT_TO_RIGHT
//				MessageBox(NULL, "If a new matching position is selected inside the area of 'left equal hist - select pt', an local refinement is automatically started. If you want to cancel the local minimum search after manual selection of the matching position, press 'ESC' to go back to the selected position or select a new position by clicking inside the area of 'left equal hist - select pt'. After aborting the local refinement or after it has finished, it is deactivated for further manual selections. If you want the local refinement to start again from the current position, press 'Pos1'. To deactivate or activate local refinement for the whole program runtime, press 'm'.\n\n The patch 'left equal hist - select pt' can be processed by a homography to better align to the right patch. Thus, the following options are possible:\n '0' - Use the ground truth homography (if available) or the one generated by matches and properties of SIFT keypoints within the left and right patches.\n '1' - Use the homography generated by the found matches only\n '2' - Manually select 4 matching positions in the left and right patch to estimate a homography (local refinement (see above) is started afterwards)\n '3' - Use the original patch\n\n Global refinement using 'left equal hist - select pt' can be started by pressing 'a' followed by '1' to '5'. The number specifies the minimal patch size for optimization (For more details press 'a' and wait for a short amount of time).\n\n The automatic homography estimation based on SIFT keypoints can be activated/deactivated for all remaining image pairs using 'k'.\n\n To scale the original left patch and display the result within 'left equal hist - select pt', press '+' or '-'. To rotate the original left patch, use the keys 'r' and 'l'.", "Help", MB_OK | MB_ICONINFORMATION);
//#else
//				MessageBox(NULL, "If a new matching position is selected inside the area of 'right equal hist - select pt', an local refinement is automatically started. If you want to cancel the local minimum search after manual selection of the matching position, press 'ESC' to go back to the selected position or select a new position by clicking inside the area of 'right equal hist - select pt'. After aborting the local refinement or after it has finished, it is deactivated for further manual selections. If you want the local refinement to start again from the current position, press 'Pos1'. To deactivate or activate local refinement for the whole program runtime, press 'm'.\n\n The patch 'right equal hist - select pt' can be processed by a homography to better align to the right patch. Thus, the following options are possible:\n '0' - Use the ground truth homography (if available) or the one generated by matches and properties of SIFT keypoints within the left and right patches.\n '1' - Use the homography generated by the found matches only\n '2' - Manually select 4 matching positions in the left and right patch to estimate a homography (local refinement (see above) is started afterwards)\n '3' - Use the original patch\n\n Global refinement using 'right equal hist - select pt' can be started by pressing 'a' followed by '1' to '5'. The number specifies the minimal patch size for optimization (For more details press 'a' and wait for a short amount of time).\n\n The automatic homography estimation based on SIFT keypoints can be activated/deactivated for all remaining image pairs using 'k'.\n\n To scale the original left patch and display the result within 'right equal hist - select pt', press '+' or '-'. To rotate the original left patch, use the keys 'r' and 'l'.", "Help", MB_OK | MB_ICONINFORMATION);
//#endif
//				c = -1;
//				skey = NONE;
//			}
//		}
//		while(((c == -1) || (skey == ARROW_UP) || (skey == ARROW_DOWN) || (skey == ARROW_LEFT) || (skey == ARROW_RIGHT) || (skey == POS1) || (c == 'm') || (c == 'h')) ||
//			  ((c != 's') && (c != 'i') && (c != 'e') && (c != 'n') && (skey != SPACE)));
//
//		if(c == 's')
//		{
//			autoManualAnno.pop_back();
//			skipped.push_back(i + (int)skipped.size());
//			i--;
//		}
//		else if(c == 'i')
//		{
//			distanceHisto[0].second++;
//			distances.push_back(0);
//			errvecs.push_back(cv::Point2f(0,0));
//#if LEFT_TO_RIGHT
//			perfectMatches.push_back(std::make_pair(lkp, rkp));
//#else
//			perfectMatches.push_back(std::make_pair(rkp, lkp));
//#endif
//			calcErrorToSpatialGT(perfectMatches.back().first, perfectMatches.back().second, channelsFlow, flowGtIsUsed, errvecsGT, validityValGT, homoGT, lkp, rkp);
//			distancesGT.push_back(std::sqrt(errvecsGT.back().x * errvecsGT.back().x + errvecsGT.back().y * errvecsGT.back().y));
//		}
//		else if(c == 'e')
//		{
//			int k = 1;
//			distances.push_back(diffdist);
//			while((k < distanceHisto.size()) && (distanceHisto[k].first < diffdist))
//				k++;
//			distanceHisto[k - 1].second++;
//			errvecs.push_back(singleErrVec);
//#if LEFT_TO_RIGHT
//			perfectMatches.push_back(std::make_pair(lkp + singleErrVec, rkp));
//#else
//			perfectMatches.push_back(std::make_pair(rkp, lkp + singleErrVec));
//#endif
//			calcErrorToSpatialGT(perfectMatches.back().first, perfectMatches.back().second, channelsFlow, flowGtIsUsed, errvecsGT, validityValGT, homoGT, lkp, rkp);
//			distancesGT.push_back(std::sqrt(errvecsGT.back().x * errvecsGT.back().x + errvecsGT.back().y * errvecsGT.back().y));
//			if(diffdist >= usedMatchTH)
//			{
//#if LEFT_TO_RIGHT
//				falseGT.push_back(std::make_pair(lkp, rkp));
//#else
//				falseGT.push_back(std::make_pair(rkp, lkp));
//#endif
//				wrongGTidxDist.push_back(std::make_pair(idx, diffdist));
//				if(flowGtIsUsed)
//				{
//#if LEFT_TO_RIGHT
//					validityValFalseGT.push_back(channelsFlow[2].at<float>((int)floor(lkp.y + 0.5f), (int)floor(lkp.x + 0.5f)));
//#else
//					validityValFalseGT.push_back(channelsFlow[2].at<float>((int)floor(rkp.y + 0.5f), (int)floor(rkp.x + 0.5f)));
//#endif
//				}
//				else
//				{
//					validityValFalseGT.push_back(1);
//				}
//			}
//		}
//		else if(c == 'n')
//		{
//#if LEFT_TO_RIGHT
//			falseGT.push_back(std::make_pair(lkp, rkp));
//#else
//			falseGT.push_back(std::make_pair(rkp, lkp));
//#endif
//			notMatchable++;
//			autoManualAnno.pop_back();
//			wrongGTidxDist.push_back(std::make_pair(idx, -1.0));
//			if(flowGtIsUsed)
//			{
//#if LEFT_TO_RIGHT
//				validityValFalseGT.push_back(channelsFlow[2].at<float>((int)floor(lkp.y + 0.5f), (int)floor(lkp.x + 0.5f)));
//#else
//				validityValFalseGT.push_back(channelsFlow[2].at<float>((int)floor(rkp.y + 0.5f), (int)floor(rkp.x + 0.5f)));
//#endif
//			}
//			else
//			{
//				validityValFalseGT.push_back(1);
//			}
//		}
//		else if((winPos2.x >= 0) && (winPos2.y >= 0))
//		{
//			int k = 1;
//			distances.push_back(diffdist2);
//			while((k < distanceHisto.size()) && (distanceHisto[k].first < diffdist2))
//				k++;
//			distanceHisto[k - 1].second++;
//			errvecs.push_back(singleErrVec2);
//#if LEFT_TO_RIGHT
//			perfectMatches.push_back(std::make_pair(lkp + singleErrVec2, rkp));
//#else
//			perfectMatches.push_back(std::make_pair(rkp, lkp + singleErrVec2));
//#endif
//			calcErrorToSpatialGT(perfectMatches.back().first, perfectMatches.back().second, channelsFlow, flowGtIsUsed, errvecsGT, validityValGT, homoGT, lkp, rkp);
//			distancesGT.push_back(std::sqrt(errvecsGT.back().x * errvecsGT.back().x + errvecsGT.back().y * errvecsGT.back().y));
//			if(diffdist2 >= usedMatchTH)
//			{
//#if LEFT_TO_RIGHT
//				falseGT.push_back(std::make_pair(lkp, rkp));
//#else
//				falseGT.push_back(std::make_pair(rkp, lkp));
//#endif
//				wrongGTidxDist.push_back(std::make_pair(idx, diffdist2));
//				if(flowGtIsUsed)
//				{
//#if LEFT_TO_RIGHT
//					validityValFalseGT.push_back(channelsFlow[2].at<float>((int)floor(lkp.y + 0.5f), (int)floor(lkp.x + 0.5f)));
//#else
//					validityValFalseGT.push_back(channelsFlow[2].at<float>((int)floor(rkp.y + 0.5f), (int)floor(rkp.x + 0.5f)));
//#endif
//				}
//				else
//				{
//					validityValFalseGT.push_back(1);
//				}
//			}
//		}
//		else
//		{
//			int k = 1;
//			distances.push_back(diffdist);
//			while((k < distanceHisto.size()) && (distanceHisto[k].first < diffdist))
//				k++;
//			distanceHisto[k - 1].second++;
//			errvecs.push_back(singleErrVec);
//#if LEFT_TO_RIGHT
//			perfectMatches.push_back(std::make_pair(lkp + singleErrVec, rkp));
//#else
//			perfectMatches.push_back(std::make_pair(rkp, lkp + singleErrVec));
//#endif
//			calcErrorToSpatialGT(perfectMatches.back().first, perfectMatches.back().second, channelsFlow, flowGtIsUsed, errvecsGT, validityValGT, homoGT, lkp, rkp);
//			distancesGT.push_back(std::sqrt(errvecsGT.back().x * errvecsGT.back().x + errvecsGT.back().y * errvecsGT.back().y));
//			if(diffdist >= usedMatchTH)
//			{
//#if LEFT_TO_RIGHT
//				falseGT.push_back(std::make_pair(lkp, rkp));
//#else
//				falseGT.push_back(std::make_pair(rkp, lkp));
//#endif
//				wrongGTidxDist.push_back(std::make_pair(idx, diffdist));
//				if(flowGtIsUsed)
//				{
//#if LEFT_TO_RIGHT
//					validityValFalseGT.push_back(channelsFlow[2].at<float>((int)floor(lkp.y + 0.5f), (int)floor(lkp.x + 0.5f)));
//#else
//					validityValFalseGT.push_back(channelsFlow[2].at<float>((int)floor(rkp.y + 0.5f), (int)floor(rkp.x + 0.5f)));
//#endif
//				}
//				else
//				{
//					validityValFalseGT.push_back(1);
//				}
//			}
//		}
//		cv::destroyWindow("GT match");
//
//		//Reestimate number of samples
//		if(!falseGT.empty() && fullN && fullSamples && fullFails)
//		{
//			double newFullSamples = (double)(*fullSamples + i + 1);
//			newP = (double)(*fullFails + (int)falseGT.size()) / newFullSamples;
//			if((abs(oldP) <= 10 * DBL_MIN) || (abs(newP - oldP) / oldP > 0.1))
//			{
//				double e;
//				double minSampleSize, sampleRatio;
//				int newSamples;
//				getMinSampleSize(*fullN, newP, e, minSampleSize);
//
//				if(newFullSamples / minSampleSize > 0.85)
//				{
//					sampleRatio = minSampleSize / (double)(*fullN);
//				}
//				else
//				{
//					double sasi[2];
//					getMinSampleSize(*fullN, 2.0 * newP, e, sasi[0]);
//					getMinSampleSize(*fullN, 0.5 * newP, e, sasi[1]);
//					minSampleSize = max(sasi[0], max(minSampleSize, sasi[1]));
//					sampleRatio = minSampleSize / (double)(*fullN);
//				}
//
//				newSamples = (int)ceil((double)GTsi * sampleRatio);
//				newSamples = newSamples < 10 ? 10:newSamples;
//				newSamples = newSamples > GTsi ? GTsi:newSamples;
//#if CORRECT_MATCHING_RESULT != 3
//				if(abs(maxSampleSize - newSamples) > 10)
//				{
//					if(newSamples < (i + 1))
//					{
//						samples = usedSamples = maxSampleSize = i + 1;
//					}
//					else
//					{
//						samples = usedSamples = maxSampleSize = newSamples;
//					}
//				}
//#endif
//			}
//		}
//	}
//
//#if CORRECT_MATCHING_RESULT != 3
//	//Check the matchability of "wrong" ground truth matches
//	clearMatchingResult();
//	descriptorsL.release();
//	descriptorsR.release();
//	extractor->compute(imgs[0],keypL,descriptorsL);
//	extractor->compute(imgs[1],keypR,descriptorsR);
//	if(matchValidKeypoints())
//	{
//		cout << "Matching failed! Connot check matchability of 'wrong' ground truth." << endl;
//	}
//	else
//	{
//		evalMatches(false);
//		for(int i = (int)skipped.size() - 1; i >= 0; i--)
//		{
//			used_matches.erase(used_matches.begin() + skipped[i]);
//		}
//		sort(used_matches.begin(), used_matches.end(),
//				[](int first, int second){return first < second;});
//		sort(wrongGTidxDist.begin(), wrongGTidxDist.end(),
//				[](std::pair<int,double> first, std::pair<int,double> second){return first.first < second.first;});
//		int j = 0;
//		for(int i = 0;i < used_matches.size(); i++)
//		{
//			if((j < (int)wrongGTidxDist.size()) && (wrongGTidxDist[j].first == used_matches[i]))
//			{
//				if(wrongGTidxDist[j].second > 8.0)
//				{
//					if(truePosMatches[used_matches[i]])
//						truePosArr[3]++;
//					else if(falsePosMatches[used_matches[i]])
//						falsePosArr[3]++;
//					else if(falseNegMatches[used_matches[i]])
//						falseNegArr[3]++;
//					else
//					{
//						cout << "There is a bug in the eval framework for calculating the quality measures!" << endl;
//					}
//				}
//				else if(wrongGTidxDist[j].second > 4.0)
//				{
//					if(truePosMatches[used_matches[i]])
//						truePosArr[2]++;
//					else if(falsePosMatches[used_matches[i]])
//						falsePosArr[2]++;
//					else if(falseNegMatches[used_matches[i]])
//						falseNegArr[2]++;
//					else
//					{
//						cout << "There is a bug in the eval framework for calculating the quality measures!" << endl;
//					}
//				}
//				else if(wrongGTidxDist[j].second < 0)
//				{
//					if(truePosMatches[used_matches[i]])
//						truePosArr[4]++;
//					else if(falsePosMatches[used_matches[i]])
//						falsePosArr[4]++;
//					else if(falseNegMatches[used_matches[i]])
//						falseNegArr[4]++;
//					else
//					{
//						cout << "There is a bug in the eval framework for calculating the quality measures!" << endl;
//					}
//				}
//				else
//				{
//					if(truePosMatches[used_matches[i]])
//						truePosArr[1]++;
//					else if(falsePosMatches[used_matches[i]])
//						falsePosArr[1]++;
//					else if(falseNegMatches[used_matches[i]])
//						falseNegArr[1]++;
//					else
//					{
//						cout << "There is a bug in the eval framework for calculating the quality measures!" << endl;
//					}
//				}
//				j++;
//			}
//			else
//			{
//				if(truePosMatches[used_matches[i]])
//					truePosArr[0]++;
//				else if(falsePosMatches[used_matches[i]])
//					falsePosArr[0]++;
//				else if(falseNegMatches[used_matches[i]])
//					falseNegArr[0]++;
//				else
//				{
//					cout << "There is a bug in the eval framework for calculating the quality measures!" << endl;
//				}
//			}
//		}
//	}
//#else
//	truePosArr[0] = annotationData.truePosArr[0][imgNr];
//	truePosArr[1] = annotationData.truePosArr[1][imgNr];
//	truePosArr[2] = annotationData.truePosArr[2][imgNr];
//	truePosArr[3] = annotationData.truePosArr[3][imgNr];
//	truePosArr[4] = annotationData.truePosArr[4][imgNr];
//	falsePosArr[0] = annotationData.falsePosArr[0][imgNr];
//	falsePosArr[1] = annotationData.falsePosArr[1][imgNr];
//	falsePosArr[2] = annotationData.falsePosArr[2][imgNr];
//	falsePosArr[3] = annotationData.falsePosArr[3][imgNr];
//	falsePosArr[4] = annotationData.falsePosArr[4][imgNr];
//	falseNegArr[0] = annotationData.falseNegArr[0][imgNr];
//	falseNegArr[1] = annotationData.falseNegArr[1][imgNr];
//	falseNegArr[2] = annotationData.falseNegArr[2][imgNr];
//	falseNegArr[3] = annotationData.falseNegArr[3][imgNr];
//	falseNegArr[4] = annotationData.falseNegArr[4][imgNr];
//	distanceHisto = annotationData.distanceHisto[imgNr];
//	notMatchable = annotationData.notMatchable[imgNr];
//#endif
//
//	//Generate homography or essential matrix from selected correspondences
//	vector<Point2f> leftPs, rightPs;
//	for(int i = 0; i < (int)perfectMatches.size(); i++)
//	{
//		leftPs.push_back(perfectMatches[i].first);
//		rightPs.push_back(perfectMatches[i].second);
//	}
//	if(flowGtIsUsed)
//	{		
//		HE = cv::findFundamentalMat(leftPs, rightPs, CV_FM_RANSAC, 0.75);
//		if(!HE.empty())
//		{
//			HE.convertTo(HE, CV_64FC1);
//			Mat Et = HE.t();
//			for (int i = 0; i < (int)perfectMatches.size(); i++)
//			{
//				Mat x1 = (Mat_<double>(3, 1) << leftPs[i].x, leftPs[i].y, 1.0); 
//				Mat x2 = (Mat_<double>(3, 1) << rightPs[i].x, rightPs[i].y, 1.0); 
//				double x2tEx1 = x2.dot(HE * x1); 
//				Mat Ex1 = HE * x1; 
//				Mat Etx2 = Et * x2; 
//				double a = Ex1.at<double>(0) * Ex1.at<double>(0); 
//				double b = Ex1.at<double>(1) * Ex1.at<double>(1); 
//				double c = Etx2.at<double>(0) * Etx2.at<double>(0); 
//				double d = Etx2.at<double>(1) * Etx2.at<double>(1); 
//
//				distancesEstModel.push_back(x2tEx1 * x2tEx1 / (a + b + c + d));
//			}
//		}
//	}
//	else
//	{
//		HE = cv::findHomography(leftPs, rightPs, CV_LMEDS);
//		if(!HE.empty())
//		{
//			HE.convertTo(HE, CV_64FC1);
//			for (int i = 0; i < (int)perfectMatches.size(); i++)
//			{
//				Mat x1 = (Mat_<double>(3, 1) << leftPs[i].x, leftPs[i].y, 1.0); 
//				Mat x2 = (Mat_<double>(3, 1) << rightPs[i].x, rightPs[i].y, 1.0);
//				Mat Hx1 = HE * x1;
//				Hx1 /= Hx1.at<double>(2);
//				Hx1 = x2 - Hx1;
//				distancesEstModel.push_back(std::sqrt(Hx1.at<double>(0) * Hx1.at<double>(0) + Hx1.at<double>(1) * Hx1.at<double>(1)));
//			}
//		}
//	}
//
//	return 0;
//}
//
///* This function calculates the subpixel-difference in the position of two patches.
//   *
//   * Mat patch	                Input  -> First patch that will be dynamic in its position
//   * Mat image                  Input  -> Second patch (or image) that will be static in its position
//   * Point2f diff				Output -> The position diffenrence between the patches
//   *									  (Position (upper left corner of patch1) in patch2 for which patch1 fits perfectly)
//   *
//   * Return value:              none
//   */
//void getSubPixPatchDiff(cv::Mat patch, cv::Mat image, cv::Point2f &diff)
//{
//	cv::Mat results;
//	float featuresize_2, nx, ny, valPxy, valPxp, valPxn, valPyp, valPyn;
//	cv::Point minLoc;
//
//    	
//     
//	cv::matchTemplate(image, patch, results, CV_TM_SQDIFF);
//	cv::minMaxLoc(results,(double *)0,(double *)0,&minLoc);
//
//	diff = cv::Point2f((float)minLoc.x, (float)minLoc.y);
//	if((minLoc.x >= results.cols - 1) || (minLoc.x <= 0) || (minLoc.y >= results.rows - 1) || (minLoc.y <= 0))
//		return;
//
//	valPxy = results.at<float>(minLoc.y,minLoc.x);
//	valPxp = results.at<float>(minLoc.y,minLoc.x+1);
//	valPxn = results.at<float>(minLoc.y,minLoc.x-1);
//	valPyp = results.at<float>(minLoc.y+1,minLoc.x);
//	valPyn = results.at<float>(minLoc.y-1,minLoc.x);
//
//	nx = 2*(2*valPxy-valPxn-valPxp);
//	ny = 2*(2*valPxy-valPyn-valPyp);
//
//	if((nx != 0) && (ny != 0))
//	{
//		nx = (valPxp-valPxn)/nx;
//		ny = (valPyp-valPyn)/ny;
//		diff += cv::Point2f(nx, ny);
//	}
//}
//
//void iterativeTemplMatch(cv::InputArray patch, cv::InputArray img, int maxPatchSize, cv::Point2f & minDiff, int maxiters)
//{
//	Mat _patch, _img;
//	_patch = patch.getMat();
//	_img = img.getMat();
//	CV_Assert((_patch.rows >= maxPatchSize) && (_patch.cols == _patch.rows));
//	CV_Assert((_img.rows == _img.cols) && (_img.rows > maxPatchSize));
//	int paImDi = _img.rows - maxPatchSize;
//	CV_Assert(!(paImDi % 2));
//	vector<int> sizeDivs;
//
//	paImDi /= 2;
//	sizeDivs.push_back(maxPatchSize);
//	while(!(sizeDivs.back() % 2))
//	{
//		sizeDivs.push_back(sizeDivs.back() / 2);
//	}
//	int actualborder = paImDi;
//	int actPatchborder = 0;
//	float minDist = FLT_MAX;
//	for(int i = 0; i < (int)sizeDivs.size() - 1; i++)
//	{
//		float dist_tmp;
//		cv::Point2f minDiff_tmp;
//		getSubPixPatchDiff(_patch(Rect(actPatchborder, actPatchborder, sizeDivs[i], sizeDivs[i])), _img, minDiff_tmp);//Template matching with subpixel accuracy
//		minDiff_tmp -= Point2f((float)actualborder, (float)actualborder);//Compensate for the large patch size
//		dist_tmp = minDiff_tmp.x * minDiff_tmp.x + minDiff_tmp.y * minDiff_tmp.y;
//		if(dist_tmp < minDist)
//		{
//			minDist = dist_tmp;
//			minDiff = minDiff_tmp;
//		}
//		if((sizeDivs.size() > 1) && (i < (int)sizeDivs.size() - 2))
//		{
//			actPatchborder += sizeDivs[i+2];
//			actualborder += sizeDivs[i+2];
//		}
//		maxiters--;
//		if(maxiters <= 0)
//			break;
//	}
//}
//
//void on_mouse_click(int event, int x, int y, int flags, void* param)
//{
//	if(event == cv::EVENT_LBUTTONDOWN)
//	{
//		if(flags == (cv::EVENT_FLAG_CTRLKEY | cv::EVENT_FLAG_LBUTTON))
//		{
//			*((cv::Point2f*)param) = cv::Point2f((float)(-x), (float)(-y));
//		}
//		else
//		{
//			*((cv::Point2f*)param) = cv::Point2f((float)x, (float)y);
//		}
//	}
//}
//
//SpecialKeyCode getSpecialKeyCode(int & val)
//{
//    int sk = NONE;
//#ifdef _LINUX
//    //see X11/keysymdef.h and X11/XF86keysym.h for more mappings
//    switch(val & 0x0000ffff){
//    case (0xffb0):{ val = '0';      break;}
//    case (0xffb1):{ val = '1';      break;}
//    case (0xffb2):{ val = '2';      break;}
//    case (0xffb3):{ val = '3';      break;}
//    case (0xffb4):{ val = '4';      break;}
//    case (0xffb5):{ val = '5';      break;}
//    case (0xffb6):{ val = '6';      break;}
//    case (0xffb7):{ val = '7';      break;}
//    case (0xffb8):{ val = '8';      break;}
//    case (0xffb9):{ val = '9';      break;}
//    case (0xffab):{ val = '+';      break;}
//    case (0xffad):{ val = '-';      break;}
//    case (0xffaa):{ val = '*';      break;}
//    case (0xffaf):{ val = '/';      break;}
//    case (0xff09): case (0x0009): { val = '\t'; break;} //TAB
//    case (0xffae): case (0xffac): { val = '.';      break;}             
//    case (0x0020): case (0xff80): { sk = SPACE;     break;}
//    case (0xff08): { sk = BACKSPACE; break;}
//    case (0xff8d) : { sk = CARRIAGE_RETURN; break;}
//    case (0x001b):{ sk = ESCAPE;    break;}
//    case (0x5B41): case (0xff52): case (0xff97): { sk = ARROW_UP;    break;}
//    case (0x5B43): case (0xff53): case (0xff98): { sk = ARROW_RIGHT; break;}
//    case (0x5B42): case (0xff54): case (0xff99): { sk = ARROW_DOWN;  break;}
//    case (0x5B44): case (0xff51): case (0xff96): { sk = ARROW_LEFT;  break;}
//    case (0x5B35): case (0xff55): case (0xff9a): { sk = PAGE_UP  ;   break;}
//    case (0x5B36): case (0xff56): case (0xff9b): { sk = PAGE_DOWN;   break;}
//    case (0x4F48): case (0xff50): case (0xff95): case (0xff58): { sk = POS1         ;break;}
//    case (0x4F46): case (0xff57): case (0xff9c): { sk = END_KEY      ;break;}
//    case (0xff63): case (0xff9e): { sk = INSERT       ;break;} //TODO: missing for kbhit case
//    case (0xffff): case (0xff9f): { sk = DELETE_KEY   ;break;} //TODO: missing for kbhit case
//    
//#else
//    switch(val & 0x00ffffff){
//    case (0x090000): { val = '\t'; break;} //TAB
//    case (0x000020 ):{ sk = SPACE;break;}
//    case (0x000008 ):{ sk = BACKSPACE;break;}
//    case (0x00001b ):{ sk = ESCAPE;break;}
//    case (0xe048): case (0x260000 ):{ sk = ARROW_UP;break;}
//    case (0xe04D): case (0x270000 ):{ sk = ARROW_RIGHT;break;}
//    case (0xe050): case (0x280000 ):{ sk = ARROW_DOWN;break;}
//    case (0xe04B): case (0x250000 ):{ sk = ARROW_LEFT; break;}
//    case (0xe049): case (0x210000 ):{ sk = PAGE_UP  ;break;}
//    case (0xe051): case (0x220000 ):{ sk = PAGE_DOWN;break;}
//    case (0xe047): case (0x240000 ):{ sk = POS1     ;break;}
//    case (0xe04F): case (0x230000 ):{ sk = END_KEY  ;break;}
//    case (0xe052): case (0x2d0000 ):{ sk = INSERT   ;break;}
//    case (0xe053): case (0x2e0000 ):{ sk = DELETE_KEY  ;break;}
//#endif
//    default:
//        sk=NONE;
//    }
//
//    if (sk == NONE)
//      val = (val&0xFF);
//    else
//      val = 0x00;
//
//    return (SpecialKeyCode)sk;
//}
//
////from CV:
//void drawMatchesCV( const Mat& img1, const vector<KeyPoint>& keypoints1,
//                  const Mat& img2, const vector<KeyPoint>& keypoints2,
//                  const vector<DMatch>& matches1to2, Mat& outImg,
//                  const Scalar& matchColor, const Scalar& singlePointColor,
//                  const vector<char>& matchesMask, int flags )
//{
//    if( !matchesMask.empty() && matchesMask.size() != matches1to2.size() )
//        CV_Error( CV_StsBadSize, "matchesMask must have the same size as matches1to2" );
//
//    Mat outImg1, outImg2;
//    _prepareImgAndDrawKeypointsCV( img1, keypoints1, img2, keypoints2,
//                                 outImg, outImg1, outImg2, singlePointColor, flags );
//
//    // draw matches
//    for( size_t m = 0; m < matches1to2.size(); m++ )
//    {
//        if( matchesMask.empty() || matchesMask[m] )
//        {
//            int i1 = matches1to2[m].queryIdx;
//            int i2 = matches1to2[m].trainIdx;
//            CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints1.size()));
//            CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints2.size()));
//
//            const KeyPoint &kp1 = keypoints1[i1], &kp2 = keypoints2[i2];
//            _drawMatchCV( outImg, outImg1, outImg2, kp1, kp2, matchColor, flags );
//        }
//    }
//}
//
////from CV:
//static void _prepareImgAndDrawKeypointsCV( const Mat& img1, const vector<KeyPoint>& keypoints1,
//                                         const Mat& img2, const vector<KeyPoint>& keypoints2,
//                                         Mat& outImg, Mat& outImg1, Mat& outImg2,
//                                         const Scalar& singlePointColor, int flags )
//{
//    Size size( img1.cols + img2.cols, MAX(img1.rows, img2.rows) );
//    if( flags & DrawMatchesFlags::DRAW_OVER_OUTIMG )
//    {
//        if( size.width > outImg.cols || size.height > outImg.rows )
//            CV_Error( CV_StsBadSize, "outImg has size less than need to draw img1 and img2 together" );
//        outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
//        outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
//    }
//    else
//    {
//        outImg.create( size, CV_MAKETYPE(img1.depth(), 3) );
//        outImg = Scalar::all(0);
//        outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
//        outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
//
//        if( img1.type() == CV_8U )
//            cvtColor( img1, outImg1, CV_GRAY2BGR );
//        else
//            img1.copyTo( outImg1 );
//
//        if( img2.type() == CV_8U )
//            cvtColor( img2, outImg2, CV_GRAY2BGR );
//        else
//            img2.copyTo( outImg2 );
//    }
//
//    // draw keypoints
//    if( !(flags & DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS) )
//    {
//        Mat _outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
//        drawKeypoints( _outImg1, keypoints1, _outImg1, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG );
//
//        Mat _outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
//        drawKeypoints( _outImg2, keypoints2, _outImg2, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG );
//    }
//}
////from CV with changed line width:
//static inline void _drawMatchCV( Mat& outImg, Mat& outImg1, Mat& outImg2 ,
//                          const KeyPoint& kp1, const KeyPoint& kp2, const Scalar& matchColor, int flags )
//{
//	const int draw_shift_bits = 4;
//	const int draw_multiplier = 1 << draw_shift_bits;
//    RNG& rng = theRNG();
//    bool isRandMatchColor = matchColor == Scalar::all(-1);
//    Scalar color = isRandMatchColor ? Scalar( rng(256), rng(256), rng(256) ) : matchColor;
//
//    _drawKeypointCV( outImg1, kp1, color, flags );
//    _drawKeypointCV( outImg2, kp2, color, flags );
//
//    Point2f pt1 = kp1.pt,
//            pt2 = kp2.pt,
//            dpt2 = Point2f( min(pt2.x + (float)outImg1.cols, (float)(outImg.cols-1)), pt2.y );
//
//    line( outImg,
//          Point(cvRound(pt1.x*draw_multiplier), cvRound(pt1.y*draw_multiplier)),
//          Point(cvRound(dpt2.x*draw_multiplier), cvRound(dpt2.y*draw_multiplier)),
//          color, 2, CV_AA, draw_shift_bits );
//}
////from CV:
//static inline void _drawKeypointCV( Mat& img, const KeyPoint& p, const Scalar& color, int flags )
//{
//    CV_Assert( !img.empty() );
//	const int draw_shift_bits = 4;
//	const int draw_multiplier = 1 << draw_shift_bits;
//    Point center( cvRound(p.pt.x * draw_multiplier), cvRound(p.pt.y * draw_multiplier) );
//
//    if( flags & DrawMatchesFlags::DRAW_RICH_KEYPOINTS )
//    {
//        int radius = cvRound(p.size/2 * draw_multiplier); // KeyPoint::size is a diameter
//
//        // draw the circles around keypoints with the keypoints size
//        circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
//
//        // draw orientation of the keypoint, if it is applicable
//        if( p.angle != -1 )
//        {
//            float srcAngleRad = p.angle*(float)CV_PI/180.f;
//            Point orient( cvRound(cos(srcAngleRad)*radius ),
//                          cvRound(sin(srcAngleRad)*radius )
//                        );
//            line( img, center, center+orient, color, 1, CV_AA, draw_shift_bits );
//        }
//#if 0
//        else
//        {
//            // draw center with R=1
//            int radius = 1 * draw_multiplier;
//            circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
//        }
//#endif
//    }
//    else
//    {
//        // draw center with R=3
//        int radius = 3 * draw_multiplier;
//        circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
//    }
//}
//
///* Estimates the minimum sample size for a given population
// *
// * int N						Input  -> Size of the whole population (dataset)
// * double p						Input  -> Expected frequency of occurance
// * double e						Output -> error range
// * double minSampleSize			Output -> Minimum sample size that should be used
// * 
// *
// * Return value:				none
// */
//void getMinSampleSize(int N, double p, double & e, double & minSampleSize)
//{
//	double q;
//	//double e; //error range
//	const double z = 1.96; //a z-value of 1.96 corresponds to a confidence level of 95%
//	p = p >= 1.0 ? 1.0:p;
//	q = 1.0 - p;
//	if(p < 0.02)
//		e = p / 2.0;
//	else
//		e = 0.01;
//	minSampleSize = z * z * p * q / (e * e);
//	minSampleSize = floor(minSampleSize / (1.0 + minSampleSize / (double)N) + 0.5);
//	minSampleSize = minSampleSize > 15000 ? 15000:minSampleSize;
//}
//
///* Bilinear interpolation function for interpolating the value of a coordinate between 4 pixel locations.
// * The function was copied from https://helloacm.com/cc-function-to-compute-the-bilinear-interpolation/
// *
// * float q11					Input  -> Value (e.g. intensity) at first coordinate (e.g. bottom left)
// * float q12					Input  -> Value (e.g. intensity) at second coordinate (e.g. top left)
// * float q22					Input  -> Value (e.g. intensity) at third coordinate (e.g. top right)
// * float q21					Input  -> Value (e.g. intensity) at fourth coordinate (e.g. bottom right)
// * float x1						Input  -> x-coordinate of q11 and q12
// * float x2						Input  -> x-coordinate of q21 and q22
// * float y1						Input  -> y-coordinate of q11 and q21
// * float y1						Input  -> y-coordinate of q12 and q22
// * float x						Input  -> x-coordinate of the position for which the interpolation is needed
// * float y						Input  -> y-coordinate of the position for which the interpolation is needed
// * 
// *
// * Return value:				The interpolated value
// */
//float BilinearInterpolation(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y)
//{
//    float x2x1, y2y1, x2x, y2y, yy1, xx1;
//    x2x1 = x2 - x1;
//    y2y1 = y2 - y1;
//    x2x = x2 - x;
//    y2y = y2 - y;
//    yy1 = y - y1;
//    xx1 = x - x1;
//    return 1.0 / (x2x1 * y2y1) * (
//        q11 * x2x * y2y +
//        q21 * xx1 * y2y +
//        q12 * x2x * yy1 +
//        q22 * xx1 * yy1
//    );
//}
//
//
//void calcErrorToSpatialGT(cv::Point2f perfectMatchesFirst, cv::Point2f perfectMatchesSecond, 
//						  std::vector<cv::Mat> channelsFlow, bool flowGtIsUsed, std::vector<cv::Point2f> & errvecsGT, 
//						  std::vector<int> & validityValGT, cv::Mat homoGT, cv::Point2f lkp, cv::Point2f rkp)
//{
//#if LEFT_TO_RIGHT
//	if(flowGtIsUsed)
//	{
//		Point2f newFlow;
//		Point2f surroundingPts[4];
//		bool validSurrounding[4];
//		surroundingPts[0] = Point2f(ceil(perfectMatchesFirst.x - 1.f), ceil(perfectMatchesFirst.y - 1.f));
//		surroundingPts[1] = Point2f(ceil(perfectMatchesFirst.x - 1.f), floor(perfectMatchesFirst.y + 1.f));
//		surroundingPts[2] = Point2f(floor(perfectMatchesFirst.x + 1.f), floor(perfectMatchesFirst.y + 1.f));
//		surroundingPts[3] = Point2f(floor(perfectMatchesFirst.x + 1.f), ceil(perfectMatchesFirst.y - 1.f));
//		validSurrounding[0] = channelsFlow[2].at<float>(surroundingPts[0].y, surroundingPts[0].x) > 0;
//		validSurrounding[1] = channelsFlow[2].at<float>(surroundingPts[1].y, surroundingPts[1].x) > 0;
//		validSurrounding[2] = channelsFlow[2].at<float>(surroundingPts[2].y, surroundingPts[2].x) > 0;
//		validSurrounding[3] = channelsFlow[2].at<float>(surroundingPts[3].y, surroundingPts[3].x) > 0;
//		if(validSurrounding[0] && validSurrounding[1] && validSurrounding[2] && validSurrounding[3])
//		{
//			newFlow.x = BilinearInterpolation(channelsFlow[0].at<float>(surroundingPts[0].y, surroundingPts[0].x), channelsFlow[0].at<float>(surroundingPts[1].y, surroundingPts[1].x),
//												channelsFlow[0].at<float>(surroundingPts[3].y, surroundingPts[3].x), channelsFlow[0].at<float>(surroundingPts[2].y, surroundingPts[2].x),
//												surroundingPts[0].x, surroundingPts[2].x, surroundingPts[0].y, surroundingPts[2].y,
//												perfectMatchesFirst.x, perfectMatchesFirst.y);
//			newFlow.y = BilinearInterpolation(channelsFlow[1].at<float>(surroundingPts[0].y, surroundingPts[0].x), channelsFlow[1].at<float>(surroundingPts[1].y, surroundingPts[1].x),
//												channelsFlow[1].at<float>(surroundingPts[3].y, surroundingPts[3].x), channelsFlow[1].at<float>(surroundingPts[2].y, surroundingPts[2].x),
//												surroundingPts[0].x, surroundingPts[2].x, surroundingPts[0].y, surroundingPts[2].y,
//												perfectMatchesFirst.x, perfectMatchesFirst.y);
//			newFlow += perfectMatchesFirst;
//		}
//		else
//		{
//			Point2i intP = Point2i((int)floor(perfectMatchesFirst.x + 0.5f), (int)floor(perfectMatchesFirst.y + 0.5f));
//			if(channelsFlow[2].at<float>(intP.y, intP.x) > 0)
//			{
//				newFlow = Point2f(channelsFlow[0].at<float>(intP.y, intP.x), channelsFlow[1].at<float>(intP.y, intP.x));
//				newFlow += perfectMatchesFirst;
//			}
//			else
//			{
//				int i1;
//				for(i1 = 0; i1 < 4; i1++)
//				{
//					if(channelsFlow[2].at<float>(surroundingPts[i1].y, surroundingPts[i1].x) > 0)
//						break;
//				}
//				if(i1 < 4)
//				{
//					newFlow = Point2f(channelsFlow[0].at<float>(surroundingPts[i1].y, surroundingPts[i1].x), channelsFlow[1].at<float>(surroundingPts[i1].y, surroundingPts[i1].x));
//					newFlow += perfectMatchesFirst;
//				}
//				else
//					newFlow = Point2f(FLT_MAX, FLT_MAX);
//			}
//		}
//		errvecsGT.push_back(perfectMatchesSecond - newFlow);
//		validityValGT.push_back(channelsFlow[2].at<float>(lkp.y, lkp.x));
//	}
//	else
//	{
//		Mat lpt_tmp;
//		lpt_tmp = (Mat_<double>(3,1) << (double)perfectMatchesSecond.x, (double)perfectMatchesSecond.y, 1.0);
//		lpt_tmp = homoGT.inv() * lpt_tmp;
//		lpt_tmp /= lpt_tmp.at<double>(2);
//		errvecsGT.push_back(perfectMatchesFirst - Point2f(lpt_tmp.at<double>(0),lpt_tmp.at<double>(1)));
//		validityValGT.push_back(1);
//	}
//#else
//	if(flowGtIsUsed)
//	{
//		Point2i intP = Point2i((int)floor(perfectMatchesFirst.x + 0.5f), (int)floor(perfectMatchesFirst.y + 0.5f));
//		Point2f newFlow;
//		newFlow = Point2f(channelsFlow[0].at<float>(intP.y, intP.x), channelsFlow[1].at<float>(intP.y, intP.x));
//		newFlow += perfectMatchesFirst;
//		errvecsGT.push_back(perfectMatchesSecond - newFlow);
//		validityValGT.push_back(channelsFlow[2].at<float>(rkp.y, rkp.x));
//	}
//	else
//	{
//		Mat lpt_tmp;
//		lpt_tmp = (Mat_<double>(3,1) << (double)rkp.x, (double)rkp.y, 1.0);
//		lpt_tmp = homoGT * lpt_tmp;
//		lpt_tmp /= lpt_tmp.at<double>(2);
//		errvecsGT.push_back(perfectMatchesSecond - Point2f(lpt_tmp.at<double>(0),lpt_tmp.at<double>(1)));
//		validityValGT.push_back(1);
//	}
//#endif
//}
//
//int baseMatcher::helpOldCodeBug(std::vector<std::pair<int,double>> wrongGTidxDist, std::vector<int> used_matches, 
//								int *truePosArr, int *falsePosArr, int *falseNegArr) //Arrays truePosArr, falsePosArr, and falseNegArr must be of size 4 and initialized
//{
//	//Check the matchability of "wrong" ground truth matches
//	clearMatchingResult();
//	descriptorsL.release();
//	descriptorsR.release();
//	Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create(); //DescriptorExtractor::create("SIFT");
//	if(extractor.empty())
//	{
//		cout << "Cannot create descriptor extractor!" << endl;
//		return -3; //Cannot create descriptor extractor
//	}
//	extractor->compute(imgs[0],keypL,descriptorsL);
//	extractor->compute(imgs[1],keypR,descriptorsR);
//	if(matchValidKeypoints())
//	{
//		cout << "Matching failed! Connot check matchability of 'wrong' ground truth." << endl;
//	}
//	else
//	{
//		evalMatches(false);
//		sort(used_matches.begin(), used_matches.end(),
//				[](int first, int second){return first < second;});
//		sort(wrongGTidxDist.begin(), wrongGTidxDist.end(),
//				[](std::pair<int,double> first, std::pair<int,double> second){return first.first < second.first;});
//		int j = 0;
//		for(int i = 0;i < used_matches.size(); i++)
//		{
//			if((j < (int)wrongGTidxDist.size()) && (wrongGTidxDist[j].first == used_matches[i]))
//			{
//				if(wrongGTidxDist[j].second > 8.0)
//				{
//					if(truePosMatches[used_matches[i]])
//						truePosArr[3]++;
//					else if(falsePosMatches[used_matches[i]])
//						falsePosArr[3]++;
//					else if(falseNegMatches[used_matches[i]])
//						falseNegArr[3]++;
//					else
//					{
//						cout << "There is a bug in the eval framework for calculating the quality measures!" << endl;
//					}
//				}
//				else if(wrongGTidxDist[j].second > 4.0)
//				{
//					if(truePosMatches[used_matches[i]])
//						truePosArr[2]++;
//					else if(falsePosMatches[used_matches[i]])
//						falsePosArr[2]++;
//					else if(falseNegMatches[used_matches[i]])
//						falseNegArr[2]++;
//					else
//					{
//						cout << "There is a bug in the eval framework for calculating the quality measures!" << endl;
//					}
//				}
//				else if(wrongGTidxDist[j].second < 0)
//				{
//					if(truePosMatches[used_matches[i]])
//						truePosArr[4]++;
//					else if(falsePosMatches[used_matches[i]])
//						falsePosArr[4]++;
//					else if(falseNegMatches[used_matches[i]])
//						falseNegArr[4]++;
//					else
//					{
//						cout << "There is a bug in the eval framework for calculating the quality measures!" << endl;
//					}
//				}
//				else
//				{
//					if(truePosMatches[used_matches[i]])
//						truePosArr[1]++;
//					else if(falsePosMatches[used_matches[i]])
//						falsePosArr[1]++;
//					else if(falseNegMatches[used_matches[i]])
//						falseNegArr[1]++;
//					else
//					{
//						cout << "There is a bug in the eval framework for calculating the quality measures!" << endl;
//					}
//				}
//				j++;
//			}
//			else
//			{
//				if(truePosMatches[used_matches[i]])
//					truePosArr[0]++;
//				else if(falsePosMatches[used_matches[i]])
//					falsePosArr[0]++;
//				else if(falseNegMatches[used_matches[i]])
//					falseNegArr[0]++;
//				else
//				{
//					cout << "There is a bug in the eval framework for calculating the quality measures!" << endl;
//				}
//			}
//		}
//	}
//	return 0;
//}
//
//void findLocalMin(Mat patchL, Mat patchR, float quarterpatch, float eigthpatch, cv::Point2f &winPos, float patchsizeOrig)
//{
//	double errlevels[3] = {0, 0, 0};
//	int autseait = 0;
//	int direction = 0;//0=left, 1=right, 2=up, 3=down
//	cv::Point2f addWinPos = cv::Point2f(0,0);
//	cv::Point2f winPosOld, winPos2;
//	int oldDir[4] = {0,0,0,0};
//	int horVerCnt[2] = {0,0};
//	int atMinimum = 0;
//	Mat patch_equal1, shiftedpatch;
//	float halfpatchsize = patchsizeOrig / 2.f;
//
//	winPos2 = winPos;
//	do
//	{
//		Mat compDiff, tdiff;
//		compDiff = Mat::eye(2,2,CV_64F);
//		tdiff = (Mat_<double>(2,1) << (double)addWinPos.x, (double)addWinPos.y);
//		cv::hconcat(compDiff, tdiff, compDiff);
//		warpAffine(patchL, shiftedpatch, compDiff, patchL.size(), INTER_LANCZOS4); //Shift the left patch according to the value from template matching
//		absdiff(shiftedpatch, patchR, patch_equal1);
//		//get sum of difference near match
//		if(autseait == 0)
//		{
//			errlevels[0] = cv::sum(patch_equal1(Rect(quarterpatch + eigthpatch, quarterpatch + eigthpatch, quarterpatch, quarterpatch)))[0];
//			errlevels[2] = errlevels[0];
//		}
//		else
//		{
//			errlevels[1] = cv::sum(patch_equal1(Rect(quarterpatch + eigthpatch, quarterpatch + eigthpatch, quarterpatch, quarterpatch)))[0];
//		}
//
//		if(autseait == 0)
//		{
//			winPosOld = winPos;
//		}
//		else //local refinement of the manual selected match position
//		{
//			if(errlevels[1] >= errlevels[0])
//			{
//				switch(direction)
//				{
//				case 0:
//					addWinPos.x += 0.0625f;
//					if((oldDir[0] > 0) || (horVerCnt[0] >= 2))
//					{
//						direction = 2;
//						if(oldDir[0] > 0)
//							atMinimum = 0;
//					}
//					else
//					{
//						direction = 1;
//						atMinimum++;
//						horVerCnt[0]++;
//					}
//					oldDir[0] = 0;
//					horVerCnt[1] = 0;
//					break;
//				case 1:
//					addWinPos.x -= 0.0625f;
//					if((oldDir[1] > 0) || (horVerCnt[0] >= 2))
//					{
//						direction = 2;
//						if(oldDir[1] > 0)
//							atMinimum = 0;
//					}
//					else
//					{
//						direction = 0;
//						atMinimum++;
//						horVerCnt[0]++;
//					}
//					oldDir[1] = 0;
//					horVerCnt[1] = 0;
//					break;
//				case 2:
//					addWinPos.y += 0.0625f;
//					if((oldDir[2] > 0) || (horVerCnt[1] >= 2))
//					{
//						direction = 0;
//						if(oldDir[2] > 0)
//							atMinimum = 0;
//					}
//					else
//					{
//						direction = 3;
//						atMinimum++;
//						horVerCnt[1]++;
//					}
//					oldDir[2] = 0;
//					horVerCnt[0] = 0;
//					break;
//				case 3:
//					addWinPos.y -= 0.0625f;
//					if((oldDir[3] > 0) || (horVerCnt[1] >= 2))
//					{
//						direction = 0;
//						if(oldDir[3] > 0)
//							atMinimum = 0;
//					}
//					else
//					{
//						direction = 2;
//						atMinimum++;
//						horVerCnt[1]++;
//					}
//					oldDir[3] = 0;
//					horVerCnt[0] = 0;
//					break;
//				}
//			}
//			else
//			{
//				oldDir[direction]++;
//				errlevels[0] = errlevels[1];
//				winPosOld = winPos;
//			}
//		}
//		switch(direction)
//		{
//		case 0:
//			addWinPos.x -= 0.0625f;
//			break;
//		case 1:
//			addWinPos.x += 0.0625f;
//			break;
//		case 2:
//			addWinPos.y -= 0.0625f;
//			break;
//		case 3:
//			addWinPos.y += 0.0625f;
//			break;
//		}
//		if((abs(winPos.x - winPosOld.x) <= 0.075f) && (abs(winPos.y - winPosOld.y) <= 0.075f) && (atMinimum < 5))
//		{
//			winPos = winPos2 + addWinPos;
//		}
//		autseait++;
//	}
//	while((winPos.x > -halfpatchsize) && (winPos.x < halfpatchsize) && (winPos.y > -halfpatchsize) && (winPos.y < halfpatchsize)
//		  && (addWinPos.x < 2.f) && (addWinPos.y < 2.f) && (atMinimum < 5));
//
//	winPos = winPosOld;
//
//	if((errlevels[2] < errlevels[0]) || (addWinPos.x >= 2.f) || (addWinPos.y >= 2.f) || (winPos.x <= -halfpatchsize) || (winPos.x >= halfpatchsize) || (winPos.y <= -halfpatchsize) || (winPos.y >= halfpatchsize))
//		winPos = winPos2;
//}
//
////To visually check the ground truth matches
////void baseMatcher::setTestGTMode()
////{
////	testGT = true;
////}