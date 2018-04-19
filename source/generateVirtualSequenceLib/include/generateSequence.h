/**********************************************************************************************************
FILE: generateSequence.h

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: March 2018

LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for generating stereo sequences with correspondences given
a view restrictions like depth ranges, moving objects, ...
**********************************************************************************************************/

#pragma once

#include "glob_includes.h"
#include "opencv2/highgui/highgui.hpp"
#include <random>

#include "generateVirtualSequenceLib\generateVirtualSequenceLib_api.h"

/* --------------------------- Defines --------------------------- */

struct depthPortion
{
	depthPortion()
	{
		near = 1.0 / 3.0;
		mid = 1.0 / 3.0;
		far = 1.0 / 3.0;
	}

	depthPortion(double near_, double mid_, double far_): near(near_), mid(mid_), far(far_)	
	{
		sumTo1();
	}

	void sumTo1()
	{
		checkCorrectValidity();
		double dsum = near + mid + far;
		near /= dsum;
		mid /= dsum;
		far /= dsum;
	}

	void checkCorrectValidity()
	{
		if (nearZero(near + mid + far))
		{
			near = 1.0 / 3.0;
			mid = 1.0 / 3.0;
			far = 1.0 / 3.0;
		}
	}

	double near;
	double mid;
	double far;
};

enum depthClass
{
	NEAR,
	MID,
	FAR
};

struct StereoSequParameters
{
	StereoSequParameters(std::vector<cv::Mat> camTrack_,
		size_t nFramesPerCamConf_ = 5,
		std::pair<double, double> inlRatRange_ = std::make_pair(0.1, 1.0),
		double inlRatChanges_ = 0,
		std::pair<size_t, size_t> truePosRange_ = std::make_pair(100, 2000),
		double truePosChanges_ = 0,
		bool keypPosErrType_ = false,
		std::pair<double, double> keypErrDistr_ = std::make_pair(0, 0.5),
		std::pair<double, double> imgIntNoise_ = std::make_pair(0, 5.0),
		double minKeypDist_ = 3.0,
		depthPortion corrsPerDepth_ = depthPortion(),
		//bool randDepth_ = true,
		std::vector<cv::Mat> corrsPerRegion_ = std::vector<cv::Mat>(),
		size_t corrsPerRegRepRate_ = 1,
		std::vector<std::vector<depthPortion>> depthsPerRegion_ = std::vector<std::vector<depthPortion>>(),
		std::vector<std::vector<std::pair<size_t, size_t>>> nrDepthAreasPReg_ = std::vector<std::vector<std::pair<size_t, size_t>>>(),
		double lostCorrPor_ = 0,
		double relCamVelocity_ = 0.5,
		cv::InputArray R_ = cv::noArray(),
		size_t nrMovObjs_ = 0,
		cv::InputArray startPosMovObjs_ =  cv::noArray(),
		std::pair<double, double> relAreaRangeMovObjs_ = std::make_pair(0.01, 0.1),
		std::vector<depthClass> movObjDepth_ = std::vector<depthClass>(),
		cv::InputArray movObjDir_ = cv::noArray(),
		std::pair<double, double> relMovObjVelRange_ = std::make_pair(0.5, 1.5),
		std::pair<double, double> relMinMaxDMovObj_ = std::make_pair(0.1, 2.0),
		double CorrMovObjPort_ = 0.25,
		size_t minNrMovObjs_ = 0
		): 
	nFramesPerCamConf(nFramesPerCamConf_), 
		inlRatRange(inlRatRange_),
		inlRatChanges(inlRatChanges_),
		truePosRange(truePosRange_),
		truePosChanges(truePosChanges_),
		keypPosErrType(keypPosErrType_),
		keypErrDistr(keypErrDistr_),
		imgIntNoise(imgIntNoise_),
		minKeypDist(minKeypDist_),
		corrsPerDepth(corrsPerDepth_),
		//randDepth(randDepth_),
		corrsPerRegion(corrsPerRegion_),
		corrsPerRegRepRate(corrsPerRegRepRate_),
		depthsPerRegion(depthsPerRegion_),
		nrDepthAreasPReg(nrDepthAreasPReg_),
		lostCorrPor(lostCorrPor_),
		camTrack(camTrack_),
		relCamVelocity(relCamVelocity_),
		R(R_),
		nrMovObjs(nrMovObjs_),
		startPosMovObjs(startPosMovObjs_),
		relAreaRangeMovObjs(relAreaRangeMovObjs_),
		movObjDepth(movObjDepth_),
		movObjDir(movObjDir_),
		relMovObjVelRange(relMovObjVelRange_),
		relMinMaxDMovObj(relMinMaxDMovObj_),
		CorrMovObjPort(CorrMovObjPort_),
		minNrMovObjs(minNrMovObjs_)
	{
		CV_Assert(nFramesPerCamConf > 0);
		CV_Assert((inlRatRange.first < 1.0) && (inlRatRange.first >= 0) && (inlRatRange.second <= 1.0) && (inlRatRange.second > 0));
		CV_Assert((inlRatChanges <= 100.0) && (inlRatChanges >= 0));
		CV_Assert((truePosRange.first > 0) && (truePosRange.second > 0) && (truePosRange.second >= truePosRange.first));
		CV_Assert((truePosChanges <= 100.0) && (truePosChanges >= 0));
		CV_Assert(keypPosErrType || (!keypPosErrType && (keypErrDistr.first > -5.0) && (keypErrDistr.first < 5.0) && (keypErrDistr.second > -5.0) && (keypErrDistr.second < 5.0)));
		CV_Assert((imgIntNoise.first > -25.0) && (imgIntNoise.first < 25.0) && (imgIntNoise.second > -25.0) && (imgIntNoise.second < 25.0));
		CV_Assert((minKeypDist > 0) && (minKeypDist < 100.0));
		//CV_Assert((corrsPerDepth.near >= 0) && (corrsPerDepth.near <= 1.0) && (corrsPerDepth.mid >= 0) && (corrsPerDepth.mid <= 1.0) && (corrsPerDepth.far >= 0) && (corrsPerDepth.far <= 1.0));
		CV_Assert((corrsPerDepth.near >= 0) && (corrsPerDepth.mid >= 0) && (corrsPerDepth.far >= 0));
		CV_Assert(corrsPerRegion.empty() || ((corrsPerRegion[0].rows == 3) && (corrsPerRegion[0].cols == 3) && (corrsPerRegion[0].type() == CV_64FC1)));
		CV_Assert(depthsPerRegion.empty() || ((depthsPerRegion.size() == 3) && (depthsPerRegion[0].size() == 3) && (depthsPerRegion[1].size() == 3) && (depthsPerRegion[2].size() == 3)));
		CV_Assert(nrDepthAreasPReg.empty() || ((nrDepthAreasPReg.size() == 3) && (nrDepthAreasPReg[0].size() == 3) && (nrDepthAreasPReg[1].size() == 3) && (nrDepthAreasPReg[2].size() == 3)));
		CV_Assert((lostCorrPor >= 0) && (lostCorrPor <= 1.0));

		CV_Assert(!camTrack.empty() && (camTrack[0].rows == 3) && (camTrack[0].cols == 1) && (camTrack[0].type() == CV_64FC1));
		CV_Assert((relCamVelocity > 0) && (relCamVelocity <= 10.0));
		CV_Assert(R.empty() || ((R.rows == 3) && (R.cols == 1) && (R.type() == CV_64FC1)));
		CV_Assert(nrMovObjs < 20);
		CV_Assert(startPosMovObjs.empty() || ((startPosMovObjs.rows == 3) && (startPosMovObjs.cols == 3) && (startPosMovObjs.type() == CV_8UC1)));
		CV_Assert((relAreaRangeMovObjs.first <= 1.0) && (relAreaRangeMovObjs.first >= 0) && (relAreaRangeMovObjs.second <= 1.0) && (relAreaRangeMovObjs.second > 0));
		CV_Assert(movObjDir.empty() || ((movObjDir.rows == 3) && (movObjDir.cols == 1) && (movObjDir.type() == CV_64FC1)));
		CV_Assert((relMovObjVelRange.first < 100.0) && (relAreaRangeMovObjs.first >= 0) && (relAreaRangeMovObjs.second <= 100.0) && (relAreaRangeMovObjs.second > 0));
		CV_Assert((relMinMaxDMovObj.first < 10.0) && (relMinMaxDMovObj.first >= 0) && (relMinMaxDMovObj.second <= 10.0) && (relMinMaxDMovObj.second > 0));
		CV_Assert((CorrMovObjPort > 0) && (CorrMovObjPort <= 1.0));
		CV_Assert(minNrMovObjs <= nrMovObjs);
	}

	//Parameters for generating correspondences
	size_t nFramesPerCamConf;//# of Frames per camera configuration
	std::pair<double, double> inlRatRange;//Inlier ratio range
	double inlRatChanges;//Inlier ratio change rate from pair to pair. If 0, the inlier ratio within the given range is always the same for every image pair. If 100, the inlier ratio is chosen completely random within the given range.For values between 0 and 100, the inlier ratio selected is not allowed to change more than this factor from the last inlier ratio.
	std::pair<size_t, size_t> truePosRange;//# true positives range
	double truePosChanges;//True positives change rate from pair to pair. If 0, the true positives within the given range are always the same for every image pair. If 100, the true positives are chosen completely random within the given range.For values between 0 and 100, the true positives selected are not allowed to change more than this factor from the true positives.
	bool keypPosErrType;//Keypoint detector error (true) or error normal distribution (false)
	std::pair<double, double> keypErrDistr;//Keypoint error distribution (mean, std)
	std::pair<double, double> imgIntNoise;//Noise (mean, std) on the image intensity for descriptor calculation
	double minKeypDist;//min. distance between keypoints
	depthPortion corrsPerDepth;//portion of correspondences at depths
	//bool randDepth;//Random depth definition of image regions? True: completely random; false: random by definition
	std::vector<cv::Mat> corrsPerRegion;//List of portions of image correspondences at regions (Matrix must be 3x3). Maybe doesnt hold: Also depends on 3D-points from prior frames.
	size_t corrsPerRegRepRate;//Repeat rate of portion of correspondences at regions. If more than one matrix of portions of correspondences at regions is provided, this number specifies the number of frames for which such a matrix is valid. After all matrices are used, the first one is used again. If 0 and no matrix of portions of correspondences at regions is provided, as many random matrizes as frames are randomly generated.
	std::vector<std::vector<depthPortion>> depthsPerRegion;//Portion of depths per region (must be 3x3). For each of the 3x3=9 image regions, the portion of near, mid, and far depths can be specified. If the overall depth definition is not met, this tensor is adapted.Maybe doesnt hold: Also depends on 3D - points from prior frames.
	std::vector<std::vector<std::pair<size_t, size_t>>> nrDepthAreasPReg;//Min and Max number of connected depth areas per region (must be 3x3). The minimum number (first) must be larger 0. The maximum number is bounded by the minimum area which is 16 pixels. Maybe doesnt hold: Also depends on 3D - points from prior frames.
	double lostCorrPor;//Portion of lost correspondences from frame to frame. It corresponds to the portion of 3D-points that would be visible in the next frame.

	//Paramters for camera and object movements
	std::vector<cv::Mat> camTrack;//Movement direction or track of the cameras (Mat must be 3x1). If 1 vector: Direction in the form [tx, ty, tz]. If more vectors: absolute position edges on a track.  The scaling of the track is calculated using the velocity information(The last frame is located at the last edge); tz is the main viewing direction of the first camera which can be changed using the rotation vector for the camera centre.The camera rotation during movement is based on the relative movement direction(like a fixed stereo rig mounted on a car).
	double relCamVelocity;//Relative velocity of the camera movement (value between 0 and 10; must be larger 0). The velocity is relative to the baseline length between the stereo cameras
	cv::InputArray R;//Rotation matrix of the first camera centre. This rotation can change the camera orientation for which without rotation the z - component of the relative movement vector coincides with the principal axis of the camera. Rotation matrix must be generated using the form R_y * R_z * R_x.
	size_t nrMovObjs;//Number of moving objects in the scene
	cv::InputArray startPosMovObjs;//Possible starting positions of moving objects in the image (must be 3x3 boolean (CV_8UC1))
	std::pair<double, double> relAreaRangeMovObjs;//Relative area range of moving objects. Area range relative to the image area at the beginning.
	std::vector<depthClass> movObjDepth;//Depth of moving objects. Moving objects are always visible and not covered by other static objects. If the number of paramters is equal "nrMovObjs", the corresponding depth is used for every object. If the number of parameters is smaller and between 2 and 3, the depths for the moving objects are selected uniformly distributed from the given depths. For a number of paramters larger 3 and unequal to "nrMovObjs", a portion for every depth that should be used can be defined (e.g. 3 x far, 2 x near, 1 x mid -> 3 / 6 x far, 2 / 6 x near, 1 / 6 x mid).
	cv::InputArray movObjDir;//Movement direction of moving objects relative to camera movementm (must be 3x1). The movement direction is linear and does not change if the movement direction of the camera changes.The moving object is removed, if it is no longer visible in both stereo cameras.
	std::pair<double, double> relMovObjVelRange;//Relative velocity range of moving objects based on relative camera velocity. Values between 0 and 100; Must be larger 0;
	std::pair<double, double> relMinMaxDMovObj;//Relative min and max depth based on z_max for moving objects to disappear (Values between 0 and 10; Must be larger 0). This value is relative to th max. usable depth depending on the camera configuration. If at least one 3D point of a moving object reaches one of these thresholds, the object is removed.
	double CorrMovObjPort;//Portion of correspondences on moving object (compared to static objects). It is limited by the size of the objects visible in the images and the minimal distance between correspondences.
	size_t minNrMovObjs;//Minimum number of moving objects over the whole track. If the number of moving obects drops below this number during camera movement, as many new moving objects are inserted until "nrMovObjs" is reached. If 0, no new moving objects are inserted if every preceding object is out of sight.
};

struct Poses
{
	Poses(cv::Mat R_, cv::Mat t_) : R(R_), t(t_) {}

	cv::Mat R;
	cv::Mat t;
};


/* --------------------------- Classes --------------------------- */

class genStereoSequ
{
	genStereoSequ(cv::Size imgSize_, cv::Mat K1_, cv::Mat K2_, std::vector<cv::Mat> R_, std::vector<cv::Mat> t_, StereoSequParameters pars_);

private:
	void constructCamPath();
	cv::Mat getTrackRot(cv::Mat tdiff);
	bool getDepthRanges();
	void adaptDepthsPerRegion();
	void updDepthReg(bool isNear, std::vector<std::vector<depthPortion>> &depthPerRegion, cv::Mat &cpr);
	void genInlierRatios();
	void genNrCorrsImg();
	bool initFracCorrImgReg();
	void initNrCorrespondences();
	void checkDepthAreas();
	void calcPixAreaPerDepth();
	void checkDepthSeeds();
	void backProject3D();
	void adaptIndicesNoDel(std::vector<size_t> &idxVec, std::vector<size_t> &delListSortedAsc);
	void adaptIndicesCVPtNoDel(std::vector<cv::Point3_<int32_t>> &seedVec, std::vector<size_t> &delListSortedAsc);
	void genRegMasks();
	void genDepthMaps();
	bool addAdditionalDepth(unsigned char pixVal,
		cv::Mat &imgD,
		cv::Mat &imgSD,
		cv::Mat &mask,
		cv::Mat &regMask,
		cv::Point_<int32_t> &startpos,
		cv::Point_<int32_t> &endpos,
		int32_t &addArea,
		int32_t &maxAreaReg,
		cv::Size &siM1,
		cv::Point_<int32_t> initSeed,
		cv::Rect &vROI,
		size_t &nrAdds,
		unsigned char &usedDilate);
	std::vector<int32_t> getPossibleDirections(cv::Point_<int32_t> &startpos, cv::Mat &mask, cv::Mat &regMask, cv::Mat &imgD, cv::Size &siM1);
	void getRandDepthFuncPars(std::vector<std::vector<double>> &pars, size_t n_pars);
	void getDepthVals(cv::Mat &dout, cv::Mat &din, double dmin, double dmax, std::vector<cv::Point3_<int32_t>> &initSeedInArea);
	inline double getDepthFuncVal(std::vector<double> &pars, double x, double y);
	void getDepthMaps(cv::Mat &dout, cv::Mat &din, double dmin, double dmax, std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> &initSeeds, int dNr);
	void getKeypoints();
	bool checkLKPInlier(cv::Point_<int32_t> pt, cv::Point2d &pt2, cv::Point3d &pCam);
	void getNrSizePosMovObj();
	void generateMovObjLabels(cv::Mat &mask, std::vector<cv::Point_<int32_t>> &seeds, std::vector<int32_t> &areas);

private:
	std::default_random_engine rand_gen;

	const int32_t minDArea = 36;//6*6: minimum area for a depth region in the image
	const double maxFarDistMultiplier = 20.0;//the maximum possible depth used is 

	cv::Size imgSize;
	cv::Mat K1, K1i;//Camera matrix 1 and its inverse
	cv::Mat K2, K2i;//Camera matrix 2 and its inverse
	std::vector<cv::Mat> R;
	std::vector<cv::Mat> t;
	StereoSequParameters pars;
	size_t nrStereoConfs;//Number of different stereo camera configurations

	size_t totalNrFrames;//Total number of frames
	double absCamVelocity;//in baselines from frame to frame
	std::vector<Poses> absCamCoordinates;//Absolute coordinates of the camera centres (left or bottom cam of stereo rig) for every frame

	std::vector<double> inlRat;//Inlier ratio for every frame
	std::vector<size_t> nrTruePos;//Absolute number of true positive correspondences per frame
	std::vector<size_t> nrCorrs;//Absolute number of correspondences (TP+TN) per frame
	std::vector<size_t> nrTrueNeg;//Absolute number of true negative correspondences per frame
	bool fixedNrCorrs = false;//If the inlier ratio and the absolute number of true positive correspondences are constant over all frames, the # of correspondences are as well const. and fixedNrCorrs = true
	std::vector<cv::Mat> nrTruePosRegs;//Absolute number of true positive correspondences per image region and frame; Type CV_32SC1
	std::vector<cv::Mat> nrCorrsRegs;//Absolute number of correspondences (TP+TN) per image region and frame; Type CV_32SC1
	std::vector<cv::Mat> nrTrueNegRegs;//Absolute number of true negative correspondences per image region and frame; Type CV_32SC1

	std::vector<std::vector<cv::Mat>> regmasks;//Mask for every of the 3x3 regions with the same size as the image and aspecific percentage of validity outside every region (overlap areas). Used for generating depth maps. See genRegMasks() for details on the overlap area.
	std::vector<std::vector<cv::Rect>> regmasksROIs;//ROIs used to generate regmasks.

	cv::Mat depthMap;//Mat of the same size as the image holding a depth value for every pixel (double)
	cv::Mat depthAreaMap;//Type CV_8UC1 and same size as image; Near depth areas are marked with 1, mid with 2 and far with 3
	std::vector<double> depthNear;//Lower border of near depths for every camera configuration
	std::vector<double> depthMid;//Upper border of near and lower border of mid depths for every camera configuration
	std::vector<double> depthFar;//Upper border of far depths for every camera configuration
	std::vector<std::vector<std::vector<depthPortion>>> depthsPerRegion;//same size as pars.corrsPerRegion: m x 3 x 3
	std::vector<cv::Mat> nrDepthAreasPRegNear;//Number of depth areas ord seeds holding near depth values per region; Type CV_32SC1, same size as depthsPerRegion
	std::vector<cv::Mat> nrDepthAreasPRegMid;//Number of depth areas ord seeds holding mid depth values per region; Type CV_32SC1, same size as depthsPerRegion
	std::vector<cv::Mat> nrDepthAreasPRegFar;//Number of depth areas ord seeds holding far depth values per region; Type CV_32SC1, same size as depthsPerRegion
	std::vector<cv::Mat> areaPRegNear;//Area in pixels per region that should hold near depth values; Type CV_32SC1, same size as depthsPerRegion
	std::vector<cv::Mat> areaPRegMid;//Area in pixels per region that should hold mid depth values; Type CV_32SC1, same size as depthsPerRegion
	std::vector<cv::Mat> areaPRegFar;//Area in pixels per region that should hold far depth values; Type CV_32SC1, same size as depthsPerRegion
	std::vector<std::vector<cv::Rect>> regROIs;//ROIs of every of the 9x9 image regions

	std::vector<cv::Point3d> actImgPointCloudFromLast;//3D coordiantes that were generated with a different frame. Coordinates are in the camera coordinate system.
	std::vector<cv::Point3d> actImgPointCloud;//All newly generated 3D coordiantes excluding actImgPointCloudFromLast. Coordinates are in the camera coordinate system.
	cv::Mat actCorrsImg1TPFromLast, actCorrsImg2TPFromLast;//TP correspondences in the stereo rig from actImgPointCloudFromLast in homogeneous image coordinates. Size: 3xn; Last row should be 1.0; Both Mat must have the same size.
	std::vector<size_t> actCorrsImg12TPFromLast_Idx;//Index to the corresponding 3D point within actImgPointCloudFromLast of correspondences in actCorrsImg1TPFromLast and actCorrsImg2TPFromLast
	cv::Mat actCorrsImg1TP, actCorrsImg2TP;//TP orrespondences in the stereo rig from actImgPointCloud in homogeneous image coordinates. Size: 3xn; Last row should be 1.0; Both Mat must have the same size.
	cv::Mat actCorrsImg1TNFromLast;//TN keypoint in the first stereo rig image from actImgPointCloudFromLast in homogeneous image coordinates. Size: 3xn; Last row should be 1.0
	std::vector<size_t> actCorrsImg1TNFromLast_Idx;//Index to the corresponding 3D point within actImgPointCloudFromLast of actCorrsImg1TNFromLast
	cv::Mat actCorrsImg2TNFromLast;//TN keypoint in the second stereo rig image from actImgPointCloudFromLast in homogeneous image coordinates. Size: 3xn; Last row should be 1.0
	std::vector<size_t> actCorrsImg2TNFromLast_Idx;//Index to the corresponding 3D point within actImgPointCloudFromLast of actCorrsImg2TNFromLast
	cv::Mat actCorrsImg1TN;//Newly created TN keypoint in the first stereo rig image (no 3D points were created for them)
	cv::Mat actCorrsImg2TN;//Newly created TN keypoint in the second stereo rig image (no 3D points were created for them)
	std::vector<double> distTNtoReal;//Distance values of the TN keypoint locations in the 2nd image to the location that would be a perfect correspondence to the TN in image 1. If the value is >= 50, the "perfect location" would be outside the image
	cv::Mat corrsIMG;//Every keypoint location is marked within this Mat with a square (ones) of the size of the minimal keypoint distance. The size is img-size plus 2 * ceil(min. keypoint dist)
	cv::Mat csurr;//Mat of ones with the size 2 * ceil(min. keypoint dist) + 1

	cv::Mat actR;//actual rotation matrix of the stereo rig: x2 = actR * x1 + actT
	cv::Mat actT;//actual translation vector of the stereo rig: x2 = actR * x1 + actT
	size_t actFrameCnt = 0;
	size_t actCorrsPRIdx = 0;//actual index (corresponding to the actual frame) for pars.corrsPerRegion, depthsPerRegion, nrDepthAreasPRegNear, ...
	double actDepthNear;//Lower border of near depths for the actual camera configuration
	double actDepthMid;//Upper border of near and lower border of mid depths for the actual camera configuration
	double actDepthFar;//Upper border of mid and lower border of far depths for the actual camera configuration

	std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> seedsNear;//Holds the actual near seeds for every region; Size 3x3xn; Point3 holds the seed coordinate (x,y) and a possible index (=z) to actCorrsImg12TPFromLast_Idx if the seed was generated from an existing 3D point (otherwise z=-1).
	std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> seedsMid;//Holds the actual mid seeds for every region; Size 3x3xn; Point3 holds the seed coordinate (x,y) and a possible index (=z) to actCorrsImg12TPFromLast_Idx if the seed was generated from an existing 3D point (otherwise z=-1).
	std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> seedsFar;//Holds the actual far seeds for every region; Size 3x3xn; Point3 holds the seed coordinate (x,y) and a possible index (=z) to actCorrsImg12TPFromLast_Idx if the seed was generated from an existing 3D point (otherwise z=-1).
	std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> seedsNearFromLast;//Holds the actual near seeds of backprojected 3D points for every region; Size 3x3xn;
	std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> seedsMidFromLast;//Holds the actual mid seeds of backprojected 3D points for every region; Size 3x3xn;
	std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> seedsFarFromLast;//Holds the actual far seeds of backprojected 3D points for every region; Size 3x3xn;

	cv::Mat startPosMovObjs; //Possible starting positions of moving objects in the image (must be 3x3 boolean (CV_8UC1))
	int minOArea, maxOArea;//Minimum and maximum area of single moving objects in the image
	int minODist;//Minimum distance between seeding positions (for areas) of moving objects
	std::vector<std::vector<std::vector<int32_t>>> movObjAreas;//Area for every moving object within the first frame per image region (must be 3x3xn)
	std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> movObjSeeds;//Seeding positions for every moving object within the first frame per image region (must be 3x3xn and the same size as movObjAreas)
	std::vector<std::vector<cv::Point>> convhullPtsObj;//Every vector element (size corresponds to number of moving objects) holds the convex hull of backprojected (into image) 3D-points from a moving object
	std::vector<std::vector<cv::Point3d>> movObj3DPtsCam;//Every vector element (size corresponds to number of moving objects) holds the 3D-points from a moving object in camera coordinates
	std::vector<std::vector<cv::Point3d>> movObj3DPtsWorld;//Every vector element (size corresponds to number of moving objects) holds the 3D-points from a moving object in world coordinates
	std::vector<cv::Mat> movObjLabels;//Every vector element (size corresponds to number of newly to add moving objects) holds a mask with the size of the image marking the area of the moving object
};


/* --------------------- Function prototypes --------------------- */

template<typename T, typename A, typename T1, typename A1>
void deleteVecEntriesbyIdx(std::vector<T, A> const& editVec, std::vector<T1, A1> const& delVec);

template<typename T, typename A>
void deleteMatEntriesByIdx(cv::Mat &editMat, std::vector<T, A> const& delVec, bool rowOrder);

/* -------------------------- Functions -------------------------- */