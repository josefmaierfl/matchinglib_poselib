/**********************************************************************************************************
 FILE: pose_estim.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: May 2016

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for the estimation and optimization of poses between
			  two camera views (images).
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "poselib/glob_includes.h"

#include "poselib/poselib_api.h"

namespace poselib
{

/* --------------------------- Defines --------------------------- */

#define PIX_TH_START 0.5 //Start value of the pixel treshold for all algorithms
#define MIN_PIX_TH ((0.25 < PIX_TH_START) ? 0.25:PIX_TH_START) //Minimal inlier/ooutlier threshold in pixels (image coordinate system)
#define MAX_PIX_TH 2.0 //Maximal inlier/ooutlier threshold in pixels (image coordinate system)
#define PIX_MIN_GOOD_TH 0.8 //If the pixel start threshold chosen is too small to give a result this is checked by this "normal" pixel threshold

	enum POSELIB_API UsacChkDegenType { DEGEN_NO_CHECK, DEGEN_QDEGSAC, DEGEN_USAC_INTERNAL };

	//Possibilities to estimate the essential matrix:
	enum POSELIB_API PoseEstimator { POSE_NISTER, POSE_EIG_KNEIP, POSE_STEWENIUS };

	//Possibilities to refine the model in the inner USAC:
	enum POSELIB_API RefineAlg {
		REF_WEIGHTS,
		REF_8PT_PSEUDOHUBER,
		REF_EIG_KNEIP,
		REF_EIG_KNEIP_WEIGHTS,
		REF_STEWENIUS,
		REF_STEWENIUS_WEIGHTS,
		REF_NISTER,
		REF_NISTER_WEIGHTS
	};

	//Possibilities for refinement and weighting after the first estimation with RANSAC, ARRSAC, or USAC (first 8bit choose refinement alg and second 8bit weighting function)
	enum POSELIB_API RefinePostAlg {
		PR_NO_REFINEMENT = 0x0,//no refinement
		PR_8PT = 0x1,//refinement using 8pt algorithm
		PR_NISTER = 0x2,//refinement using Nister
		PR_STEWENIUS = 0x3,//refinement using Stewenius
		PR_KNEIP = 0x4,//refinement using Kneips Eigensolver
		PR_TORR_WEIGHTS = 0x10,//Refinement using weighting with Torr weights. This option must be combined with one of the solvers (8pt, Nister, ...)
		PR_PSEUDOHUBER_WEIGHTS = 0x20,//Refinement using weighting with Pseudo-Huber weights. This option must be combined with one of the solvers (8pt, Nister, ...)
		PR_NO_WEIGHTS = 0x30//Do not use weights for refinement
	};

	//Possible initializations for SPRT
	enum POSELIB_API SprtInit { SPRT_DEFAULT_INIT = 0x0, SPRT_DELTA_AUTOM_INIT = 0x1, SPRT_EPSILON_AUTOM_INIT = 0x2};

	// problem specific/data-related parameters: essential matrix
	struct POSELIB_API ConfigUSAC
	{
		ConfigUSAC() : focalLength(800),
			th_pixels(0.8),
			degeneracyCheck(DEGEN_USAC_INTERNAL),
			estimator(POSE_STEWENIUS),
			refinealg(REF_STEWENIUS_WEIGHTS),
			prevalidateSample(false),
			noAutomaticProsacParamters(false),
			automaticSprtInit(SPRT_DELTA_AUTOM_INIT | SPRT_EPSILON_AUTOM_INIT),
			//sprt_delta_initial(0.05),
			//sprt_epsilon_initial(0.15),
			//sortedMatchIdx(NULL),
			matches(nullptr),
			keypoints1(nullptr),
			keypoints2(nullptr),
			nrMatchesVfcFiltered(0),
			imgSize(800,600),
			degenDecisionTh(0.85)
		{}

		double focalLength;
		double th_pixels;
		UsacChkDegenType degeneracyCheck;
		PoseEstimator estimator;
		RefineAlg refinealg;
		bool prevalidateSample;
		bool noAutomaticProsacParamters;
		int automaticSprtInit;//Check enum SprtInit and its combinations
		//double sprt_delta_initial;
		//double sprt_epsilon_initial;
		//std::vector<unsigned int> *sortedMatchIdx;
		std::vector<cv::DMatch> *matches;
		std::vector<cv::KeyPoint> *keypoints1;
		std::vector<cv::KeyPoint> *keypoints2;
		unsigned int nrMatchesVfcFiltered;
		cv::Size imgSize;
		double degenDecisionTh;//Used for the option UsacChkDegenType::DEGEN_USAC_INTERNAL: Threshold on the fraction of degenerate inliers compared to the E-inlier fraction
	};


/* --------------------------- Classes --------------------------- */

class POSELIB_API AutoThEpi
{
private:

	//Variables:
	double corr_filt_cam_th; //Threshold for for marking correspondences as in- or outliers in the camera coordinate system
	double corr_filt_pix_th; //Threshold for for marking correspondences as in- or outliers in the image coordinate system (in pixels)
	double corr_filt_min_pix_th; //Minimum pixel threshold for the automatic threshold estimation if MIN_PIX_TH was chosen too small (this value is automatically computed)
	bool th_stable; //If the threshold was found to be stable during the last few evaluations, this value should be set to true.
	double pixToCamFact; //Multiplication-factor to convert the threshold from the pixel coordinate system to the camera coordinate system

public:

	explicit AutoThEpi(double pixToCamFact_, bool thStable = false)
		: corr_filt_cam_th(-1.0),
          corr_filt_pix_th(PIX_TH_START),
		  corr_filt_min_pix_th(MIN_PIX_TH),
		  th_stable(thStable),
          pixToCamFact(pixToCamFact_)
	{
		corr_filt_cam_th = corr_filt_pix_th * pixToCamFact;
	}

	//get the estimated threshold in the camera coordinate system
	double getThCam()
	{
		return corr_filt_cam_th;
	}
	//get the estimated threshold in the image coordinate system (in pixels)
	double getThPix()
	{
		return corr_filt_pix_th;
	}

	//If the threshold was found to be stable during the last few evaluations, this value should be set to true.
	void setThStable(bool isStable)
	{
		th_stable = isStable;
	}

	//Estimation of the Essential matrix with an automatic threshold estimation
	int estimateEVarTH(cv::InputArray p1, cv::InputArray p2, cv::OutputArray E, cv::OutputArray mask, double *th, int *nrgoodPts);

	//Estimates a new threshold based on the correspondences and the given essential matrix
	double estimateThresh(cv::InputArray p1, cv::InputArray p2, cv::InputArray E, bool useImgCoordSystem = false, bool storeGlobally = false);

	//Sets a new threshold for marking correspondences as in- or outliers
	double setCorrTH(double thresh, bool useImgCoordSystem = false, bool storeGlobally = true);

};


/* --------------------- Function prototypes --------------------- */

//Recovers the rotation and translation from an essential matrix and triangulates the given correspondences to form 3D coordinates.
int POSELIB_API getPoseTriangPts(cv::InputArray E,
					 cv::InputArray p1,
					 cv::InputArray p2,
					 cv::OutputArray R,
					 cv::OutputArray t,
					 cv::OutputArray Q,
					 cv::InputOutputArray mask = cv::noArray(),
					 const double dist = 50.0,
					 bool translatE = false);
//Triangulates 3D-points from correspondences with provided R and t
int POSELIB_API triangPts3D(cv::InputArray R, cv::InputArray t, cv::InputArray _points1, cv::InputArray _points2, cv::OutputArray Q3D, cv::InputOutputArray mask = cv::noArray(), const double dist = 50.0);
//Estimation of the Essential matrix based on the 5-pt Nister algorithm integrated in an ARRSAC, RANSAC or LMEDS framework.
bool POSELIB_API estimateEssentialMat(cv::OutputArray E,
        cv::InputArray p1,
        cv::InputArray p2,
        const std::string &method = "ARRSAC",
        double threshold = PIX_MIN_GOOD_TH,
        bool refine = true,
        cv::OutputArray mask = cv::noArray());
//Estimation of the Essential matrix and/or pose using the USAC framework
int POSELIB_API estimateEssentialOrPoseUSAC(const cv::Mat & p1,
	const cv::Mat & p2,
	cv::OutputArray E,
	double th,
	ConfigUSAC & cfg,
	bool & isDegenerate,
	cv::OutputArray inliers = cv::noArray(),
	cv::OutputArray R_degenerate = cv::noArray(),
	cv::OutputArray inliers_degenerate_R = cv::noArray(),
	cv::OutputArray R = cv::noArray(),
	cv::OutputArray t = cv::noArray(),
	bool verbose = false);
//Refines the essential matrix E by using the 8-point-algorithm and SVD with a pseudo-huber cost function
void POSELIB_API robustEssentialRefine(cv::InputArray points1, cv::InputArray points2, cv::InputArray E_init, cv::Mat & E_refined,
						  double th = 0.005, unsigned int iters = 0, bool makeClosestE = true, double *sumSqrErr_init = nullptr,
						  double *sumSqrErr = nullptr, cv::OutputArray errors = cv::noArray(),
						  cv::InputOutputArray mask = cv::noArray(), int model = 0, bool tryOrientedEpipolar = false, bool normalizeCorrs = false);
//Bundle adjustment (BA) on motion (=extrinsics) and structure with or without camera metrices.
bool POSELIB_API refineStereoBA(cv::InputArray p1,
					cv::InputArray p2,
					cv::InputOutputArray R,
					cv::InputOutputArray t,
					cv::InputOutputArray Q,
					cv::InputOutputArray K1,
					cv::InputOutputArray K2,
					bool pointsInImgCoords = false,
					cv::InputArray mask = cv::noArray(),
					const double angleThresh = 1.25,
					const double t_norm_tresh = 0.05);

}
