/**********************************************************************************************************
FILE: generateSequence.h

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: March 2018

LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

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
		bool randDepth_ = true,
		std::vector<cv::Mat> corrsPerRegion_ = std::vector<cv::Mat>(),
		size_t corrsPerRegRepRate_ = 1,
		std::vector<std::vector<depthPortion>> depthsPerRegion_ = std::vector<std::vector<depthPortion>>(),
		std::vector<std::vector<std::pair<double, double>>> nrDepthAreasPReg_ = std::vector<std::vector<std::pair<double, double>>>(),
		double lostCorrPor_ = 0,
		double relCamVelocity_ = 0.5,
		cv::InputArray R_ = cv::noArray(),
		size_t nrMovObjs_ = 0,
		cv::InputArray startPosMovObjs_ =  cv::noArray(),
		std::pair<double, double> relAreaRangeMovObjs_ = std::make_pair(0.01, 0.1),
		depthClass movObjDepth_ = depthClass::MID,
		cv::InputArray movObjDir_ = cv::noArray(),
		std::pair<double, double> relMovObjVelRange_ = std::make_pair(0.5, 1.5),
		std::pair<double, double> relMinMaxDMovObj_ = std::make_pair(0.1, 2.0),
		double CorrMovObjPort_ = 0.25
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
		randDepth(randDepth_),
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
		CorrMovObjPort(CorrMovObjPort_)
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
	bool randDepth;//Random depth definition of image regions? True: completely random; false: random by definition
	std::vector<cv::Mat> corrsPerRegion;//List of portions of image correspondences at regions (Matrix must be 3x3). Maybe doesnt hold: Also depends on 3D-points from prior frames.
	size_t corrsPerRegRepRate;//Repeat rate of portion of correspondences at regions. If more than one matrix of portions of correspondences at regions is provided, this number specifies the number of frames for which such a matrix is valid. After all matrices are used, the first one is used again. If 0 and no matrix of portions of correspondences at regions is provided, as many random matrizes as frames are randomly generated.
	std::vector<std::vector<depthPortion>> depthsPerRegion;//Portion of depths per region (must be 3x3). For each of the 3x3=9 image regions, the portion of near, mid, and far depths can be specified. If the overall depth definition is not met, this tensor is adapted.Maybe doesnt hold: Also depends on 3D - points from prior frames.
	std::vector<std::vector<std::pair<double, double>>> nrDepthAreasPReg;//Min and Max number of connected depth areas per region (must be 3x3). The minimum number (first) must be larger 0. The maximum number is bounded by the minimum area which is 16 pixels. Maybe doesnt hold: Also depends on 3D - points from prior frames.
	double lostCorrPor;//Portion of lost correspondences from frame to frame. It corresponds to the portion of 3D-points that would be visible in the next frame.

	//Paramters for camera and object movements
	std::vector<cv::Mat> camTrack;//Movement direction or track of the cameras (Mat must be 3x1). If 1 vector: Direction in the form [tx, ty, tz]. If more vectors: absolute position edges on a track.  The scaling of the track is calculated using the velocity information(The last frame is located at the last edge); tz is the main viewing direction of the first camera which can be changed using the rotation vector for the camera centre.The camera rotation during movement is based on the relative movement direction(like a fixed stereo rig mounted on a car).
	double relCamVelocity;//Relative velocity of the camera movement (value between 0 and 10; must be larger 0). The velocity is relative to the baseline length between the stereo cameras
	cv::InputArray R;//Rotation matrix of the first camera centre. This rotation can change the camera orientation for which without rotation the z - component of the relative movement vector coincides with the principal axis of the camera. Rotation matrix must be generated using the form R_y * R_z * R_x.
	size_t nrMovObjs;//Number of moving objects in the scene
	cv::InputArray startPosMovObjs;//Possible starting positions of moving objects in the image (must be 3x3 boolean (CV_8UC1))
	std::pair<double, double> relAreaRangeMovObjs;//Relative area range of moving objects. Area range relative to the image area at the beginning.
	depthClass movObjDepth;//Depth of moving objects. Moving objects are always visible and not covered by other static objects.
	cv::InputArray movObjDir;//Movement direction of moving objects relative to camera movementm (must be 3x1). The movement direction is linear and does not change if the movement direction of the camera changes.The moving object is removed, if it is no longer visible in both stereo cameras.
	std::pair<double, double> relMovObjVelRange;//Relative velocity range of moving objects based on relative camera velocity. Values between 0 and 100; Must be larger 0;
	std::pair<double, double> relMinMaxDMovObj;//Relative min and max depth based on z_max for moving objects to disappear (Values between 0 and 10; Must be larger 0). This value is relative to th max. usable depth depending on the camera configuration. If at least one 3D point of a moving object reaches one of these thresholds, the object is removed.
	double CorrMovObjPort;//Portion of correspondences on moving object (compared to static objects). It is limited by the size of the objects visible in the images and the minimal distance between correspondences.
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
	void combineDepthMaps();
	void updDepthReg(bool isNear, std::vector<std::vector<depthPortion>> &depthPerRegion, cv::Mat &cpr);
	void genInlierRatios();
	void genNrCorrsImg();
	void initFracCorrImgReg();

private:
	std::default_random_engine rand_gen;

	cv::Size imgSize;
	cv::Mat K1;
	cv::Mat K2;
	std::vector<cv::Mat> R;
	std::vector<cv::Mat> t;
	StereoSequParameters pars;
	size_t nrStereoConfs;

	size_t totalNrFrames;
	double absCamVelocity;//in baselines from frame to frame
	std::vector<Poses> absCamCoordinates;

	std::vector<double> inlRat;
	std::vector<size_t> nrTruePos;
	std::vector<size_t> nrCorrs;
	std::vector<size_t> nrTrueNeg;
	//std::vector<

	cv::Mat depthMapNear;
	cv::Mat depthMapMid;
	cv::Mat depthMapFar;
	std::vector<double> depthNear;//Lower border of near depths for every camera configuration
	std::vector<double> depthMid;//Upper border of near and lower border of mid depths for every camera configuration
	std::vector<double> depthFar;//Upper border of far depths for every camera configuration
	std::vector<std::vector<std::vector<depthPortion>>> depthsPerRegion;

	std::vector<cv::Point3d> actImgPointCloud;
	size_t actFrameCnt = 0;
};


/* --------------------- Function prototypes --------------------- */


/* -------------------------- Functions -------------------------- */