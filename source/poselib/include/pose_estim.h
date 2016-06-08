/**********************************************************************************************************
 FILE: pose_estim.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: May 2016

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for the estimation and optimization of poses between
			  two camera views (images).
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "glob_includes.h"

/* --------------------------- Defines --------------------------- */

#define PIX_TH_START 0.5 //Start value of the pixel treshold for all algorithms
#define MIN_PIX_TH ((0.25 < PIX_TH_START) ? 0.25:PIX_TH_START) //Minimal inlier/ooutlier threshold in pixels (image coordinate system)
#define MAX_PIX_TH 2.0 //Maximal inlier/ooutlier threshold in pixels (image coordinate system)
#define PIX_MIN_GOOD_TH 0.8 //If the pixel start threshold chosen is too small to give a result this is checked by this "normal" pixel threshold


/* --------------------------- Classes --------------------------- */

class AutoThEpi
{
private:

	//Variables:
	double corr_filt_cam_th; //Threshold for for marking correspondences as in- or outliers in the camera coordinate system
	double corr_filt_pix_th; //Threshold for for marking correspondences as in- or outliers in the image coordinate system (in pixels)
	double corr_filt_min_pix_th; //Minimum pixel threshold for the automatic threshold estimation if MIN_PIX_TH was chosen too small (this value is automatically computed)
	bool th_stable; //If the threshold was found to be stable during the last few evaluations, this value should be set to true.
	double pixToCamFact; //Multiplication-factor to convert the threshold from the pixel coordinate system to the camera coordinate system

public:

	AutoThEpi(double pixToCamFact_, bool thStable = false)
		: corr_filt_pix_th(PIX_TH_START), 
		  corr_filt_cam_th(-1.0), 
		  corr_filt_min_pix_th(MIN_PIX_TH),
		  pixToCamFact(pixToCamFact_),
		  th_stable(thStable)
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
int getPoseTriangPts(cv::InputArray E, 
					 cv::InputArray p1, 
					 cv::InputArray p2, 
					 cv::OutputArray R, 
					 cv::OutputArray t, 
					 cv::OutputArray Q, 
					 cv::InputOutputArray mask = cv::noArray(), 
					 const double dist = 50.0, 
					 bool translatE = false);
//Estimation of the Essential matrix based on the 5-pt Nister algorithm integrated in an ARRSAC, RANSAC or LMEDS framework.
bool estimateEssentialMat(cv::OutputArray E, cv::InputArray p1, cv::InputArray p2, std::string method = "ARRSAC", double threshold = PIX_MIN_GOOD_TH, bool refine = true, cv::OutputArray mask = cv::noArray());
//Refines the essential matrix E by using the 8-point-algorithm and SVD with a pseudo-huber cost function
void robustEssentialRefine(cv::InputArray points1, cv::InputArray points2, cv::InputArray E_init, cv::Mat & E_refined,
						  double th = 0.005, unsigned int iters = 0, bool makeClosestE = true, double *sumSqrErr_init = NULL,
						  double *sumSqrErr = NULL, cv::OutputArray errors = cv::noArray(), 
						  cv::InputOutputArray mask = cv::noArray(), int model = 0);
//Bundle adjustment (BA) on motion (=extrinsics) and structure with or without camera metrices.
bool refineStereoBA(cv::InputArray p1, 
					cv::InputArray p2, 
					cv::InputOutputArray R, 
					cv::InputOutputArray t, 
					cv::InputOutputArray Q, 
					cv::InputOutputArray K1,
					cv::InputOutputArray K2,
					bool pointsInImgCoords = false,
					cv::InputArray mask = cv::noArray(),
					const double angleThresh = 0.3,
					const double t_norm_tresh = 0.05);
