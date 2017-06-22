/**********************************************************************************************************
FILE: usac_estimations.cpp

PLATFORM: Windows 7, MS Visual Studio 2014, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: May 2017

LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides interfaces to the USAC framework for robust parameter estimations.
**********************************************************************************************************/

#include <time.h>
#include <vector>

#include "usac/usac_estimations.h"
#include "usac/config/ConfigParams.h"
#include "usac/config/ConfigParamsFundmatrix.h"
#include "usac/estimators/FundmatrixEstimator.h"
#include "usac/estimators/EssentialMatEstimator.h"
#include "usac/estimators/RotationMatEstimator.h"
#include "opencv2/core/eigen.hpp"



/* --------------------------- Defines --------------------------- */
using namespace std;


/* --------------------- Function prototypes --------------------- */



/* --------------------- Functions --------------------- */

/* Robustly estimates a fundamental matrix using the USAC framework and checks for degenerate configurations. If a degenerate
* configuration is found (dominant plane), its model (homography) including its inliers is available.
*
* InputArray p1						Input  -> Observed point coordinates of the left image in the image coordinate system
*												  (n rows, 2 cols)
* InputArray p2						Input  -> Observed point coordinates of the right image in the image coordinate system
*												  (n rows, 2 cols)
* OutputArray F						Output -> Fundamental matrix
* OutputArray inliers				Output -> Inlier mask for the fundamental matrix
* OutputArray H						Output -> If a degenerate configuration is found, the corresponding homography can be accessed here
* OutputArray inliers_degenerate	Output -> Inlier mask of the degenerate (homography) model
* double fraction_degen_inliers		Output -> Fraction of inliers of the degenerate model compared to the inliers of the fundamental matrix
* vector<unsigned int> sortedMatchIdx Input  -> Indices of the correspondences p1 <-> p2 sorted according to their descriptor similarity 
*												with highest similarity (= smallest descriptor distance) first to enable PROSAC sampling
* double th							Input  -> Threshold for marking inliers [DEFAULT = 0.8]
*
* Return value:						true :	Success
*										false:	Failed
*/
int estimateFundMatrixUsac(cv::InputArray p1, 
						   cv::InputArray p2, 
						   cv::OutputArray F, 
						   double & sprt_delta_result,
						   double & sprt_epsilon_result,
						   double th,
						   double prosac_beta,
						   double sprt_delta,
						   double sprt_epsilon,
						   cv::OutputArray inliers, 
						   cv::OutputArray H, 
						   cv::OutputArray inliers_degenerate,
						   double *fraction_degen_inliers,
						   std::vector<unsigned int> sortedMatchIdx)
{
	CV_Assert((p1.cols() == 2) && (p2.cols() == 2) && (p1.rows() == p2.rows()) && (p1.type() == CV_64FC1) && (p2.type() == CV_64FC1));
	CV_Assert(sortedMatchIdx.empty() || (p1.rows() == (int)sortedMatchIdx.size()));
	cv::Mat p1_ = p1.getMat();
	cv::Mat p2_ = p2.getMat();
	unsigned int numPts = p1_.rows;


	// seed random number generator
	srand((unsigned int)time(NULL));

	//convert data into format for USAC fundamental matrix estimation
	std::vector<double> pointData(6 * numPts);
	for (unsigned int i = 0; i < numPts; ++i)
	{
		pointData[6 * i] = p1_.at<double>(i, 0);
		pointData[6 * i + 1] = p1_.at<double>(i, 1);
		pointData[6 * i + 2] = 1.0;

		pointData[6 * i + 3] = p2_.at<double>(i, 0);
		pointData[6 * i + 4] = p2_.at<double>(i, 1);
		pointData[6 * i + 5] = 1.0;
	}

	USACConfig::Common c_com;
	USACConfig::Losac c_lo;
	USACConfig::Prosac c_pro;
	USACConfig::Sprt c_sprt;
	USACConfig::Fund c_fund;

	//Common RANSAC parameters
	c_com.confThreshold = 0.99; // ransac_conf: 0.0 - 1.0 (must be double), specifies the confidence parameter
	c_com.minSampleSize = 7; //min_sample_size: int, number of points used to generate models
	c_com.inlierThreshold = th; //inlier_threshold: double, threshold for inlier classification
	c_com.maxHypotheses = 850000;//--------------> maybe should be changed
	c_com.maxSolutionsPerSample = 3; //max_solutions_per_sample: int, number of possible solutions using the minimal sample
	c_com.numDataPoints = numPts; //number of correspondences from matching
	c_com.prevalidateSample = true; //specifies whether samples are to be prevalidated prior to model generation
	c_com.prevalidateModel = true; //specifies whether models are to be prevalidated prior to verification against data points
	c_com.testDegeneracy = true; //specifies whether degeneracy testing is to be performed
	if(sortedMatchIdx.empty())
		c_com.randomSamplingMethod = USACConfig::SAMP_UNIFORM; //normal random sampling
	else
		c_com.randomSamplingMethod = USACConfig::SAMP_PROSAC; //PROSAC sampling
	c_com.verifMethod = USACConfig::VERIF_SPRT;//specifies the type of model verification to be performed
	c_com.localOptMethod = USACConfig::LO_LOSAC;//specifies what type of local optimization is to be performed

	//Parameters of inner RANSAC
	c_lo.innerRansacRepetitions = 20;//was in USAC example: 5;
	c_lo.innerSampleSize = 14;//was in USAC example: 15;
	c_lo.thresholdMultiplier = 2.0;//--------------> maybe should be changed
	c_lo.numStepsIterative = 4;//--------------> maybe should be changed

	//PROSAC parameters
	if (!sortedMatchIdx.empty())
	{
		c_pro.beta = prosac_beta;// 0.09;//probability of incorrect model being support by a random data point, can be adapted to use values from sprt, --------------> maybe should be changed
		c_pro.maxSamples = 100000; //number of samples after which PROSAC behaves like RANSAC
		c_pro.minStopLen = 20;
		c_pro.nonRandConf = 0.99; //find non-minimal subset with probability of randomness smaller than (1-non_rand_conf)
		c_pro.sortedPointIndices = &sortedMatchIdx[0];
	}

	//SPRT parameters
	c_sprt.delta = sprt_delta;// 0.05;//initial average of inlier ratios in bad models, it is reestimated as USAC proceeds, --------------> maybe should be changed
	c_sprt.epsilon = sprt_epsilon;// 0.15;//initial inlier ratio for largest size of support, it is reestimated as USAC proceeds; is set to 0.1 in ARRSAC --------------> maybe should be changed
	c_sprt.mS = 2.38;//average number of models verified for each sample (fundamental mat with 7-pt alg has 1 to 3 solutions)
					 //is used to estimate the decision treshold of SPRT -> we should try to estimate this value during runtime
	c_sprt.tM = 200.0;//Computation time of a model hypotheses (e.g. fundamental mat) expressed in time units for veryfying tm data points
					 //is used to estimate the decision treshold of SPRT -> we should try to estimate this value during runtime

	//Problem specific parameters (for estimating a fundamental matrix)
	c_fund.decompositionAlg = USACConfig::DECOMP_QR;//matrix_decomposition: QR, LU
	c_fund.hDegenThreshold = 6.5;//inlier threshold for the h-degeneracy test, --------------> maybe should be changed
	c_fund.maxUpgradeSamples = 8000;//maximum number of 2-point samples to draw in the model upgrade loop, --------------> maybe should be changed

	
	ConfigParamsFund cfg(c_com, c_pro, c_sprt, c_lo, c_fund);

	// initialize the fundamental matrix estimation problem
	FundMatrixEstimator* fund = new FundMatrixEstimator;
	fund->initParamsUSAC(cfg);
	fund->initDataUSAC(cfg);
	fund->initProblem(cfg, &pointData[0]);

	//Find the fundamental matrix
	if (!fund->solve())
	{
		return(EXIT_FAILURE);
	}

	sprt_delta_result = fund->usac_results_.sprt_delta_;
	sprt_epsilon_result = fund->usac_results_.sprt_epsilon_;

	//Get model parameters
	if (F.needed())
	{
		if (!fund->final_model_params_.empty())
		{
			cv::Mat F_(3, 3, CV_64FC1);
			if (F.empty())
				F.create(3, 3, CV_64F);
			for (unsigned int i = 0; i < 3; ++i)
			{
				for (unsigned int j = 0; j < 3; ++j)
				{
					F_.at<double>(i, j) = fund->final_model_params_[3 * i + j];
				}
			}
			F_.copyTo(F.getMat());
		}
	}
	if (H.needed())
	{
		if (!fund->degen_final_model_params_.empty())
		{
			cv::Mat H_(3, 3, CV_64FC1);
			if (H.empty())
				H.create(3, 3, CV_64F);
			for (unsigned int i = 0; i < 3; ++i)
			{
				for (unsigned int j = 0; j < 3; ++j)
				{
					H_.at<double>(i, j) = fund->degen_final_model_params_[3 * i + j];
				}
			}
			H_.copyTo(H.getMat());
		}
		if(fraction_degen_inliers)
			*fraction_degen_inliers = fund->usac_results_.degen_inlier_count_ / fund->usac_results_.best_inlier_count_;
	}
	else
	{
		if (!H.empty())
			H.clear();
		if (fraction_degen_inliers)
			*fraction_degen_inliers = 0;
	}
	if (inliers.needed())
	{
		if (!fund->usac_results_.inlier_flags_.empty())
		{
			cv::Mat inliers_(1, numPts, CV_8U);
			if (inliers.cols() != numPts)
				inliers.clear();
			if (inliers.empty())
				inliers.create(1, numPts, CV_8U);
			for (unsigned int i = 0; i < numPts; ++i)
			{
				if(fund->usac_results_.inlier_flags_[i] == 0)
					inliers_.at<bool>(i) = false;
				else
					inliers_.at<bool>(i) = true;
			}
			inliers_.copyTo(inliers.getMat());
		}
	}
	if (inliers_degenerate.needed())
	{
		if (!fund->usac_results_.degen_inlier_flags_.empty())
		{
			cv::Mat inliers_degenerate_(1, numPts, CV_8U);
			if (inliers_degenerate.cols() != numPts)
				inliers_degenerate.clear();
			if (inliers_degenerate.empty())
				inliers_degenerate.create(1, numPts, CV_8U);
			for (unsigned int i = 0; i < numPts; ++i)
			{
				if (fund->usac_results_.degen_inlier_flags_[i] == 0)
					inliers_degenerate_.at<bool>(i) = false;
				else
					inliers_degenerate_.at<bool>(i) = true;
			}
			inliers_degenerate_.copyTo(inliers_degenerate.getMat());
		}
	}

	// clean up
	pointData.clear();
	fund->cleanupProblem();
	delete fund;

	return(EXIT_SUCCESS);
}

int estimateEssentialMatUsac(cv::InputArray p1,
	cv::InputArray p2,
	cv::OutputArray E,
	double & sprt_delta_result,
	double & sprt_epsilon_result,
	double th,
	double focalLength,
	double th_pixels,
	double prosac_beta,
	double sprt_delta,
	double sprt_epsilon,
	bool checkDegeneracy,
	USACConfig::EssentialMatEstimatorUsed used_estimator,
	USACConfig::RefineAlgorithm	refineMethod,
	cv::OutputArray inliers,
	unsigned int *nr_inliers,
	cv::OutputArray H,
	cv::OutputArray inliers_degenerate_H,
	cv::OutputArray R,
	cv::OutputArray inliers_degenerate_R,
	cv::OutputArray t,
	cv::OutputArray inliers_degenerate_t,
	cv::OutputArray inliers_degenerate_noMotion,
	double *fraction_degen_inliers_H,
	double *fraction_degen_inliers_R,
	double *fraction_degen_inliers_t,
	double *fraction_degen_inliers_noMot,
	std::vector<unsigned int> sortedMatchIdx,
	cv::OutputArray R_E, 
	cv::OutputArray t_E)
{
	CV_Assert((p1.cols() == 2) && (p2.cols() == 2) && (p1.rows() == p2.rows()) && (p1.type() == CV_64FC1) && (p2.type() == CV_64FC1));
	CV_Assert(sortedMatchIdx.empty() || (p1.rows() == (int)sortedMatchIdx.size()));
	cv::Mat p1_ = p1.getMat();
	cv::Mat p2_ = p2.getMat();
	unsigned int numPts = p1_.rows;
	unsigned int nr_inliers_ = 0;

	if (numPts > INT_MAX)
	{
		cout << "Too many correspondences!" << endl;
		return EXIT_FAILURE;
	}

	// seed random number generator
	srand((unsigned int)time(NULL));

	//convert data into format for USAC essential matrix estimation
	std::vector<double> pointData(6 * numPts);
	for (unsigned int i = 0; i < numPts; ++i)
	{
		pointData[6 * i] = p1_.at<double>(i, 0);
		pointData[6 * i + 1] = p1_.at<double>(i, 1);
		pointData[6 * i + 2] = 1.0;

		pointData[6 * i + 3] = p2_.at<double>(i, 0);
		pointData[6 * i + 4] = p2_.at<double>(i, 1);
		pointData[6 * i + 5] = 1.0;
	}

	USACConfig::Common c_com;
	USACConfig::Losac c_lo;
	USACConfig::Prosac c_pro;
	USACConfig::Sprt c_sprt;
	//USACConfig::Fund c_fund;
	USACConfig::EssMat c_essential;
	

	//Common RANSAC parameters
	c_com.confThreshold = 0.99; // ransac_conf: 0.0 - 1.0 (must be double), specifies the confidence parameter
	c_com.minSampleSize = 5; //min_sample_size: int, number of points used to generate models
	c_com.inlierThreshold = th; //inlier_threshold: double, threshold for inlier classification
	c_com.maxHypotheses = 850000;//--------------> maybe should be changed
	c_com.maxSolutionsPerSample = 10; //max_solutions_per_sample: int, number of possible solutions using the minimal sample
	c_com.numDataPoints = numPts; //number of correspondences from matching
	c_com.prevalidateSample = true; //specifies whether samples are to be prevalidated prior to model generation
	c_com.prevalidateModel = true; //specifies whether models are to be prevalidated prior to verification against data points
	c_com.testDegeneracy = checkDegeneracy; //specifies whether degeneracy testing is to be performed
	if (sortedMatchIdx.empty())
		c_com.randomSamplingMethod = USACConfig::SAMP_UNIFORM; //normal random sampling
	else
		c_com.randomSamplingMethod = USACConfig::SAMP_PROSAC; //PROSAC sampling
	c_com.verifMethod = USACConfig::VERIF_SPRT;//specifies the type of model verification to be performed
	c_com.localOptMethod = USACConfig::LO_LOSAC;//specifies what type of local optimization is to be performed

	//Parameters of inner RANSAC
	c_lo.innerRansacRepetitions = 5;// 20;//was in USAC example: 5;
	c_lo.innerSampleSize = 14;//was in USAC example: 15;
	c_lo.thresholdMultiplier = 2.0;//--------------> maybe should be changed
	c_lo.numStepsIterative = 4;//--------------> maybe should be changed

	//PROSAC parameters
	if (!sortedMatchIdx.empty())
	{
		c_pro.beta = prosac_beta;// 0.09;//probability of incorrect model being support by a random data point, can be adapted to use values from sprt, --------------> maybe should be changed
		c_pro.maxSamples = 100000; //number of samples after which PROSAC behaves like RANSAC
		c_pro.minStopLen = 20;
		c_pro.nonRandConf = 0.99; //find non-minimal subset with probability of randomness smaller than (1-non_rand_conf)
		c_pro.sortedPointIndices = &sortedMatchIdx[0];
	}

	//SPRT parameters
	c_sprt.delta = sprt_delta;// 0.05;//initial average of inlier ratios in bad models, it is reestimated as USAC proceeds, --------------> maybe should be changed
	c_sprt.epsilon = sprt_epsilon;// 0.15;//initial inlier ratio for largest size of support, it is reestimated as USAC proceeds; is set to 0.1 in ARRSAC --------------> maybe should be changed
	c_sprt.mS = 2.38;//average number of models verified for each sample (fundamental mat with 7-pt alg has 1 to 3 solutions)
					 //is used to estimate the decision treshold of SPRT -> we should try to estimate this value during runtime
	c_sprt.tM = 200.0;//Computation time of a model hypotheses (e.g. fundamental mat) expressed in time units for veryfying tm data points
					  //is used to estimate the decision treshold of SPRT -> we should try to estimate this value during runtime

	//Problem specific parameters (for estimating a essential matrix)
	c_essential.focalLength = focalLength; //The focal length of the camera (for 2 different cameras, use the mean focal length if they are not completely different)
	c_essential.hDegenThreshold = 6.5 * th;//inlier threshold for the h-degeneracy test, --------------> maybe should be changed
	c_essential.maxUpgradeSamples = 8000;//maximum number of 2-point samples to draw in the model upgrade loop, --------------> maybe should be changed
	c_essential.refineMethod = refineMethod;//The used method for refinement in inner RANSAC
	c_essential.rotDegenThesholdPix = th_pixels;//Threshold in pixels
	c_essential.used_estimator = used_estimator;//specifies a specific minimal solver from OpenGV
	c_essential.ransacLikeUpgradeDegenerate = true;//Specifies if the upgrade of a degenerate model H to a higher order model E should be performed with a minimal solver (true) or the original method implemented in USAC (false)
	c_essential.enableHDegen = false; //Should remain disbled (false). Enable the H degeneracy check and upgrade to full model (H degeneracy is not a problem for the 5pt algorithm)
	c_essential.enableUpgradeDegenPose = true; //Enable the upgrade from degenerate configurations R or no Motion to R-> R+t or no Motion -> t

	ConfigParamsEssential cfg(c_com, c_pro, c_sprt, c_lo, c_essential);

	// initialize the fundamental matrix estimation problem
	EssentialMatEstimator* fund = new EssentialMatEstimator;
	fund->initParamsUSAC(cfg);
	fund->initDataUSAC(cfg);
	fund->initProblem(cfg, &pointData[0]);

	//Find the essential matrix
	if (!fund->solve())
	{
		return(EXIT_FAILURE);
	}

	sprt_delta_result = fund->usac_results_.sprt_delta_;
	sprt_epsilon_result = fund->usac_results_.sprt_epsilon_;

	//Get model parameters
	if (E.needed())
	{
		if (!fund->final_model_params_.empty())
		{
			cv::Mat E_(3, 3, CV_64FC1);
			if (E.empty())
				E.create(3, 3, CV_64F);
			for (unsigned int i = 0; i < 3; ++i)
			{
				for (unsigned int j = 0; j < 3; ++j)
				{
					E_.at<double>(i, j) = fund->final_model_params_[3 * i + j];
				}
			}
			E_.copyTo(E.getMat());
		}
	}
	if (R_E.needed())
	{
		if (used_estimator == USACConfig::ESTIM_EIG_KNEIP || refineMethod == USACConfig::REFINE_EIG_KNEIP || refineMethod == USACConfig::REFINE_EIG_KNEIP_WEIGHTS)
		{
			cv::Mat R_;
			cv::eigen2cv(fund->R_eigen, R_);
			if (R_E.empty())
				R_E.create(3, 3, CV_64F);
			R_.copyTo(R_E.getMat());
		}
		else if (!R_E.empty())
			R_E.clear();
	}
	if (t_E.needed())
	{
		if (used_estimator == USACConfig::ESTIM_EIG_KNEIP || refineMethod == USACConfig::REFINE_EIG_KNEIP || refineMethod == USACConfig::REFINE_EIG_KNEIP_WEIGHTS)
		{
			cv::Mat t_;
			cv::eigen2cv(fund->t_eigen, t_);
			if (t_.cols > t_.rows)
				t_ = t_.t();
			if (t_E.empty())
				t_E.create(3, 1, CV_64F);
			t_.copyTo(t_E.getMat());
		}
		else if (!t_E.empty())
			t_E.clear();
	}
	if (H.needed())
	{
		if (!fund->degen_final_model_params_.empty() && fund->usac_results_.degen_inlier_count_ > 4)
		{
			cv::Mat H_(3, 3, CV_64FC1);
			if (H.empty())
				H.create(3, 3, CV_64F);
			for (unsigned int i = 0; i < 3; ++i)
			{
				for (unsigned int j = 0; j < 3; ++j)
				{
					H_.at<double>(i, j) = fund->degen_final_model_params_[3 * i + j];
				}
			}
			H_.copyTo(H.getMat());
		}
		else
		{
			if (!H.empty())
				H.clear();
		}
	}
	else
	{
		if (!H.empty())
			H.clear();
	}
	if (fraction_degen_inliers_H)
	{
		if (fund->usac_results_.degen_inlier_count_ > 4)
			*fraction_degen_inliers_H = fund->usac_results_.best_inlier_count_ > 0 ? (fund->usac_results_.degen_inlier_count_ / fund->usac_results_.best_inlier_count_) : 0;
		else
			*fraction_degen_inliers_H = 0;
	}
	if (R.needed())
	{
		if (!fund->degen_final_model_params_rot.empty() && fund->usac_results_.degen_inlier_count_rot > 2)
		{
			cv::Mat R_(3, 3, CV_64FC1);
			if (R.empty())
				R.create(3, 3, CV_64F);
			for (unsigned int i = 0; i < 3; ++i)
			{
				for (unsigned int j = 0; j < 3; ++j)
				{
					R_.at<double>(i, j) = fund->degen_final_model_params_rot[3 * i + j];
				}
			}
			R_.copyTo(R.getMat());
		}
		else
		{
			if (!R.empty())
				R.clear();
		}
	}
	else
	{
		if (!R.empty())
			R.clear();
	}
	if (fraction_degen_inliers_R)
	{
		if (fund->usac_results_.degen_inlier_count_rot > 2)
			*fraction_degen_inliers_R = fund->usac_results_.best_inlier_count_ > 0 ? (fund->usac_results_.degen_inlier_count_rot / fund->usac_results_.best_inlier_count_) : 0;
		else
			*fraction_degen_inliers_R = 0;
	}
	if (t.needed())
	{
		if (!fund->degen_final_model_params_trans.empty() && fund->usac_results_.degen_inlier_count_trans > 2)
		{
			cv::Mat t_(3, 1, CV_64FC1);
			if (t.empty())
				t.create(3, 1, CV_64F);
			for (unsigned int i = 0; i < 3; ++i)
			{
				t_.at<double>(i) = fund->degen_final_model_params_trans[i];
			}
			t_.copyTo(t.getMat());
		}
		else 
		{
			if (!t.empty())
				t.clear();
		}
	}
	else
	{
		if (!t.empty())
			t.clear();
	}
	if (fraction_degen_inliers_t)
	{
		if (fund->usac_results_.degen_inlier_count_trans > 2)
			*fraction_degen_inliers_t = fund->usac_results_.best_inlier_count_ > 0 ? (fund->usac_results_.degen_inlier_count_trans / fund->usac_results_.best_inlier_count_) : 0;
		else
			*fraction_degen_inliers_t = 0;
	}
	if (fraction_degen_inliers_noMot)
	{
		if (fund->usac_results_.degen_inlier_count_noMot > 1)
			*fraction_degen_inliers_noMot = fund->usac_results_.best_inlier_count_ > 0 ? (fund->usac_results_.degen_inlier_count_noMot / fund->usac_results_.best_inlier_count_) : 0;
		else
			*fraction_degen_inliers_noMot = 0;
	}
	if (inliers.needed())
	{
		if (!fund->usac_results_.inlier_flags_.empty())
		{
			cv::Mat inliers_(1, numPts, CV_8U);
			if (inliers.cols() != numPts)
				inliers.clear();
			if (inliers.empty())
				inliers.create(1, numPts, CV_8U);
			for (unsigned int i = 0; i < numPts; ++i)
			{
				if (fund->usac_results_.inlier_flags_[i] == 0)
					inliers_.at<bool>(i) = false;
				else
				{
					inliers_.at<bool>(i) = true;
					nr_inliers_++;
				}
			}
			inliers_.copyTo(inliers.getMat());
			if (nr_inliers)
				*nr_inliers = nr_inliers_;
		}
	}
	else if (nr_inliers)
	{
		for (unsigned int i = 0; i < numPts; ++i)
		{
			if (fund->usac_results_.inlier_flags_[i])
				nr_inliers_++;
		}
		*nr_inliers = nr_inliers_;
	}
	if (inliers_degenerate_H.needed())
	{
		if (!fund->usac_results_.degen_inlier_flags_.empty())
		{
			cv::Mat inliers_degenerate_(1, numPts, CV_8U);
			if (inliers_degenerate_H.cols() != numPts)
				inliers_degenerate_H.clear();
			if (inliers_degenerate_H.empty())
				inliers_degenerate_H.create(1, numPts, CV_8U);
			for (unsigned int i = 0; i < numPts; ++i)
			{
				if (fund->usac_results_.degen_inlier_flags_[i] == 0)
					inliers_degenerate_.at<bool>(i) = false;
				else
					inliers_degenerate_.at<bool>(i) = true;
			}
			inliers_degenerate_.copyTo(inliers_degenerate_H.getMat());
		}
	}
	if (inliers_degenerate_R.needed())
	{
		if (!fund->usac_results_.degen_inlier_flags_rot.empty())
		{
			cv::Mat inliers_degenerate_(1, numPts, CV_8U);
			if (inliers_degenerate_R.cols() != numPts)
				inliers_degenerate_R.clear();
			if (inliers_degenerate_R.empty())
				inliers_degenerate_R.create(1, numPts, CV_8U);
			for (unsigned int i = 0; i < numPts; ++i)
			{
				if (fund->usac_results_.degen_inlier_flags_rot[i] == 0)
					inliers_degenerate_.at<bool>(i) = false;
				else
					inliers_degenerate_.at<bool>(i) = true;
			}
			inliers_degenerate_.copyTo(inliers_degenerate_R.getMat());
		}
	}
	if (inliers_degenerate_noMotion.needed())
	{
		if (!fund->usac_results_.degen_inlier_flags_noMot.empty())
		{
			cv::Mat inliers_degenerate_(1, numPts, CV_8U);
			if (inliers_degenerate_noMotion.cols() != numPts)
				inliers_degenerate_noMotion.clear();
			if (inliers_degenerate_noMotion.empty())
				inliers_degenerate_noMotion.create(1, numPts, CV_8U);
			for (unsigned int i = 0; i < numPts; ++i)
			{
				if (fund->usac_results_.degen_inlier_flags_noMot[i] == 0)
					inliers_degenerate_.at<bool>(i) = false;
				else
					inliers_degenerate_.at<bool>(i) = true;
			}
			inliers_degenerate_.copyTo(inliers_degenerate_noMotion.getMat());
		}
	}

	// clean up
	pointData.clear();
	fund->cleanupProblem();
	delete fund;

	return(EXIT_SUCCESS);
}

int estimateRotationMatUsac(cv::InputArray p1,
	cv::InputArray p2,
	cv::OutputArray R,
	double & sprt_delta_result,
	double & sprt_epsilon_result,
	double focalLength,
	double th_pixels,
	double prosac_beta,
	double sprt_delta,
	double sprt_epsilon,
	cv::OutputArray inliers,
	unsigned int *nr_inliers,
	std::vector<unsigned int> sortedMatchIdx)
{
	CV_Assert((p1.cols() == 2) && (p2.cols() == 2) && (p1.rows() == p2.rows()) && (p1.type() == CV_64FC1) && (p2.type() == CV_64FC1));
	CV_Assert(sortedMatchIdx.empty() || (p1.rows() == (int)sortedMatchIdx.size()));
	cv::Mat p1_ = p1.getMat();
	cv::Mat p2_ = p2.getMat();
	unsigned int numPts = p1_.rows;
	unsigned int nr_inliers_ = 0;

	if (numPts > INT_MAX)
	{
		cout << "Too many correspondences!" << endl;
		return EXIT_FAILURE;
	}

	// seed random number generator
	srand((unsigned int)time(NULL));

	//convert data into format for USAC essential matrix estimation
	std::vector<double> pointData(6 * numPts);
	for (unsigned int i = 0; i < numPts; ++i)
	{
		pointData[6 * i] = p1_.at<double>(i, 0);
		pointData[6 * i + 1] = p1_.at<double>(i, 1);
		pointData[6 * i + 2] = 1.0;

		pointData[6 * i + 3] = p2_.at<double>(i, 0);
		pointData[6 * i + 4] = p2_.at<double>(i, 1);
		pointData[6 * i + 5] = 1.0;
	}

	USACConfig::Common c_com;
	USACConfig::Losac c_lo;
	USACConfig::Prosac c_pro;
	USACConfig::Sprt c_sprt;
	
	//Common RANSAC parameters
	c_com.confThreshold = 0.99; // ransac_conf: 0.0 - 1.0 (must be double), specifies the confidence parameter
	c_com.minSampleSize = 2; //min_sample_size: int, number of points used to generate models
	c_com.inlierThreshold = std::sqrt(1.0 - std::cos(std::atan(th_pixels / focalLength))); //inlier_threshold: double, threshold for inlier classification
	c_com.maxHypotheses = 850000;//--------------> maybe should be changed
	c_com.maxSolutionsPerSample = 1; //max_solutions_per_sample: int, number of possible solutions using the minimal sample
	c_com.numDataPoints = numPts; //number of correspondences from matching
	c_com.prevalidateSample = false; //specifies whether samples are to be prevalidated prior to model generation
	c_com.prevalidateModel = false; //specifies whether models are to be prevalidated prior to verification against data points
	c_com.testDegeneracy = false; //specifies whether degeneracy testing is to be performed
	if (sortedMatchIdx.empty())
		c_com.randomSamplingMethod = USACConfig::SAMP_UNIFORM; //normal random sampling
	else
		c_com.randomSamplingMethod = USACConfig::SAMP_PROSAC; //PROSAC sampling
	c_com.verifMethod = USACConfig::VERIF_SPRT;//specifies the type of model verification to be performed
	c_com.localOptMethod = USACConfig::LO_LOSAC;//specifies what type of local optimization is to be performed

												//Parameters of inner RANSAC
	c_lo.innerRansacRepetitions = 5;// 20;//was in USAC example: 5;
	c_lo.innerSampleSize = 10;
	c_lo.thresholdMultiplier = 2.0;//--------------> maybe should be changed
	c_lo.numStepsIterative = 4;//--------------> maybe should be changed

							   //PROSAC parameters
	if (!sortedMatchIdx.empty())
	{
		c_pro.beta = prosac_beta;// 0.09;//probability of incorrect model being support by a random data point, can be adapted to use values from sprt, --------------> maybe should be changed
		c_pro.maxSamples = 100000; //number of samples after which PROSAC behaves like RANSAC
		c_pro.minStopLen = 20;
		c_pro.nonRandConf = 0.99; //find non-minimal subset with probability of randomness smaller than (1-non_rand_conf)
		c_pro.sortedPointIndices = &sortedMatchIdx[0];
	}

	//SPRT parameters
	c_sprt.delta = sprt_delta;// 0.05;//initial average of inlier ratios in bad models, it is reestimated as USAC proceeds, --------------> maybe should be changed
	c_sprt.epsilon = sprt_epsilon;// 0.15;//initial inlier ratio for largest size of support, it is reestimated as USAC proceeds; is set to 0.1 in ARRSAC --------------> maybe should be changed
	c_sprt.mS = 1.0;//average number of models verified for each sample (fundamental mat with 7-pt alg has 1 to 3 solutions)
					 //is used to estimate the decision treshold of SPRT -> we should try to estimate this value during runtime
	c_sprt.tM = 36.27;//Computation time of a model hypotheses (e.g. fundamental mat) expressed in time units for veryfying tm data points
					  //is used to estimate the decision treshold of SPRT -> we should try to estimate this value during runtime

	ConfigParamsRotationMat cfg(c_com, c_pro, c_sprt, c_lo);

	// initialize the fundamental matrix estimation problem
	RotationMatEstimator* fund = new RotationMatEstimator;
	fund->initParamsUSAC(cfg);
	fund->initDataUSAC(cfg);
	fund->initProblem(cfg, &pointData[0]);

	//Find the essential matrix
	if (!fund->solve())
	{
		return(EXIT_FAILURE);
	}

	sprt_delta_result = fund->usac_results_.sprt_delta_;
	sprt_epsilon_result = fund->usac_results_.sprt_epsilon_;

	//Get model parameters
	if (R.needed())
	{
		cv::Mat R_(3, 3, CV_64FC1);
		if (R.empty())
			R.create(3, 3, CV_64F);
		cv::eigen2cv(fund->final_model_params_, R_);
		R_.copyTo(R.getMat());
	}
	
	if (inliers.needed())
	{
		if (!fund->usac_results_.inlier_flags_.empty())
		{
			cv::Mat inliers_(1, numPts, CV_8U);
			if (inliers.cols() != numPts)
				inliers.clear();
			if (inliers.empty())
				inliers.create(1, numPts, CV_8U);
			for (unsigned int i = 0; i < numPts; ++i)
			{
				if (fund->usac_results_.inlier_flags_[i] == 0)
					inliers_.at<bool>(i) = false;
				else
				{
					inliers_.at<bool>(i) = true;
					nr_inliers_++;
				}
			}
			inliers_.copyTo(inliers.getMat());
			if (nr_inliers)
				*nr_inliers = nr_inliers_;
		}
	}
	else if (nr_inliers)
	{
		for (unsigned int i = 0; i < numPts; ++i)
		{
			if (fund->usac_results_.inlier_flags_[i])
				nr_inliers_++;
		}
		*nr_inliers = nr_inliers_;
	}
	
	// clean up
	pointData.clear();
	fund->cleanupProblem();
	delete fund;

	return(EXIT_SUCCESS);
}


int upgradeEssentialMatDegenUsac(cv::InputArray p1,
	cv::InputArray p2,
	cv::InputArray inliers_degen,
	cv::OutputArray E,
	double & sprt_delta_result,
	double & sprt_epsilon_result,
	double th,
	double focalLength,
	double th_pixels,
	double sprt_delta,
	double sprt_epsilon,
	USACConfig::EssentialMatEstimatorUsed used_estimator,
	USACConfig::RefineAlgorithm	refineMethod,
	cv::OutputArray inliers,
	unsigned int *nr_inliers,
	cv::OutputArray R,
	cv::OutputArray t)
{
	CV_Assert((p1.cols() == 2) && (p2.cols() == 2) && (p1.rows() == p2.rows()) && (p1.type() == CV_64FC1) && (p2.type() == CV_64FC1));
	cv::Mat p1_ = p1.getMat();
	cv::Mat p2_ = p2.getMat();
	cv::Mat inl_degen = inliers_degen.getMat();
	unsigned int numPts = p1_.rows;
	unsigned int degenInlCnt = 0;
	unsigned int nr_inliers_ = 0;

	if (numPts > INT_MAX)
	{
		cout << "Too many correspondences!" << endl;
		return EXIT_FAILURE;
	}

	// seed random number generator
	srand((unsigned int)time(NULL));

	//convert data into format for USAC essential matrix estimation and split into degenerate inliers and outliers
	std::vector<double> pointData;// (6 * numPts);
	pointData.reserve(6 * numPts);
	for (unsigned int i = 0; i < numPts; ++i)
	{
		if(inl_degen.at<bool>(i))
		{ 
			pointData.push_back(p1_.at<double>(i, 0));
			pointData.push_back(p1_.at<double>(i, 1));
			pointData.push_back(1.0);

			pointData.push_back(p2_.at<double>(i, 0));
			pointData.push_back(p2_.at<double>(i, 1));
			pointData.push_back(1.0);

			degenInlCnt++;
		}
	}
	for (unsigned int i = 0; i < numPts; ++i)
	{
		if (!inl_degen.at<bool>(i))
		{
			pointData.push_back(p1_.at<double>(i, 0));
			pointData.push_back(p1_.at<double>(i, 1));
			pointData.push_back(1.0);

			pointData.push_back(p2_.at<double>(i, 0));
			pointData.push_back(p2_.at<double>(i, 1));
			pointData.push_back(1.0);
		}
	}

	USACConfig::Common c_com;
	USACConfig::Losac c_lo;
	USACConfig::Prosac c_pro;
	USACConfig::Sprt c_sprt;
	USACConfig::EssMat c_essential;


	//Common RANSAC parameters
	c_com.confThreshold = 0.99; // ransac_conf: 0.0 - 1.0 (must be double), specifies the confidence parameter
	c_com.minSampleSize = 5; //min_sample_size: int, number of points used to generate models
	c_com.inlierThreshold = th; //inlier_threshold: double, threshold for inlier classification
	c_com.maxHypotheses = 850000;//--------------> maybe should be changed
	c_com.maxSolutionsPerSample = 10; //max_solutions_per_sample: int, number of possible solutions using the minimal sample
	c_com.numDataPoints = numPts; //number of correspondences from matching
	c_com.numDataPointsDegenerate = degenInlCnt; //Number of inliers to the degenerate model (must agree with the sorted inliers to the degenerate model at the beginning of pointData)
	c_com.prevalidateSample = true; //specifies whether samples are to be prevalidated prior to model generation
	c_com.prevalidateModel = true; //specifies whether models are to be prevalidated prior to verification against data points
	c_com.testDegeneracy = false; //specifies whether degeneracy testing is to be performed
	c_com.minSampleSizeDegenerate = 2; //For upgrading from a quasi-degenerate case (here rotation) to the full model (here essential matrix), 
										//the minimal sample size for the degenerate model has to be provided
	c_com.randomSamplingMethod = USACConfig::SAMP_MODEL_COMPLETE; //Sample from degenerate model inliers and outliers
	c_com.verifMethod = USACConfig::VERIF_SPRT;//specifies the type of model verification to be performed
	c_com.localOptMethod = USACConfig::LO_LOSAC;//specifies what type of local optimization is to be performed

												//Parameters of inner RANSAC
	c_lo.innerRansacRepetitions = 5;// 20;//was in USAC example: 5;
	c_lo.innerSampleSize = 14;//was in USAC example: 15;
	c_lo.thresholdMultiplier = 2.0;//--------------> maybe should be changed
	c_lo.numStepsIterative = 4;//--------------> maybe should be changed

	//SPRT parameters
	c_sprt.delta = sprt_delta;// 0.05;//initial average of inlier ratios in bad models, it is reestimated as USAC proceeds, --------------> maybe should be changed
	c_sprt.epsilon = sprt_epsilon;// 0.15;//initial inlier ratio for largest size of support, it is reestimated as USAC proceeds; is set to 0.1 in ARRSAC --------------> maybe should be changed
	c_sprt.mS = 2.38;//average number of models verified for each sample (fundamental mat with 7-pt alg has 1 to 3 solutions)
					 //is used to estimate the decision treshold of SPRT -> we should try to estimate this value during runtime
	c_sprt.tM = 200.0;//Computation time of a model hypotheses (e.g. fundamental mat) expressed in time units for veryfying tm data points
					  //is used to estimate the decision treshold of SPRT -> we should try to estimate this value during runtime

					  //Problem specific parameters (for estimating a essential matrix)
	c_essential.focalLength = focalLength; //The focal length of the camera (for 2 different cameras, use the mean focal length if they are not completely different)
	c_essential.hDegenThreshold = 6.5 * th;//inlier threshold for the h-degeneracy test, --------------> maybe should be changed
	c_essential.maxUpgradeSamples = 8000;//maximum number of 2-point samples to draw in the model upgrade loop, --------------> maybe should be changed
	c_essential.refineMethod = refineMethod;//The used method for refinement in inner RANSAC
	c_essential.rotDegenThesholdPix = th_pixels;//Threshold in pixels
	c_essential.used_estimator = used_estimator;//specifies a specific minimal solver from OpenGV
	//c_essential.ransacLikeUpgradeDegenerate = true;//Specifies if the upgrade of a degenerate model H to a higher order model E should be performed with a minimal solver (true) or the original method implemented in USAC (false)
	c_essential.enableHDegen = false; //Should remain disbled (false). Enable the H degeneracy check and upgrade to full model (H degeneracy is not a problem for the 5pt algorithm)
	c_essential.enableUpgradeDegenPose = false; //Enable the upgrade from degenerate configurations R or no Motion to R-> R+t or no Motion -> t

	ConfigParamsEssential cfg(c_com, c_pro, c_sprt, c_lo, c_essential);

	// initialize the fundamental matrix estimation problem
	EssentialMatEstimator* fund = new EssentialMatEstimator;
	fund->initParamsUSAC(cfg);
	fund->initDataUSAC(cfg);
	fund->initProblem(cfg, &pointData[0]);

	//Find the essential matrix
	if (!fund->solve())
	{
		return(EXIT_FAILURE);
	}

	sprt_delta_result = fund->usac_results_.sprt_delta_;
	sprt_epsilon_result = fund->usac_results_.sprt_epsilon_;

	//Get model parameters
	if (E.needed())
	{
		if (!fund->final_model_params_.empty())
		{
			cv::Mat E_(3, 3, CV_64FC1);
			if (E.empty())
				E.create(3, 3, CV_64F);
			for (unsigned int i = 0; i < 3; ++i)
			{
				for (unsigned int j = 0; j < 3; ++j)
				{
					E_.at<double>(i, j) = fund->final_model_params_[3 * i + j];
				}
			}
			E_.copyTo(E.getMat());
		}
	}
	if (R.needed())
	{
		if (used_estimator == USACConfig::ESTIM_EIG_KNEIP || refineMethod == USACConfig::REFINE_EIG_KNEIP || refineMethod == USACConfig::REFINE_EIG_KNEIP_WEIGHTS)
		{
			cv::Mat R_;
			cv::eigen2cv(fund->R_eigen, R_);
			if (R.empty())
				R.create(3, 3, CV_64F);
			R_.copyTo(R.getMat());
		}
		else if (!R.empty())
			R.clear();
	}
	if (t.needed())
	{
		if (used_estimator == USACConfig::ESTIM_EIG_KNEIP || refineMethod == USACConfig::REFINE_EIG_KNEIP || refineMethod == USACConfig::REFINE_EIG_KNEIP_WEIGHTS)
		{
			cv::Mat t_;
			cv::eigen2cv(fund->t_eigen, t_);
			if (t_.cols > t_.rows)
				t_ = t_.t();
			if (t.empty())
				t.create(3, 1, CV_64F);
			t_.copyTo(t.getMat());
		}
		else if (!t.empty())
			t.clear();
	}
	if (inliers.needed())
	{
		if (!fund->usac_results_.inlier_flags_.empty())
		{
			cv::Mat inliers_(1, numPts, CV_8U);
			if (inliers.cols() != numPts)
				inliers.clear();
			if (inliers.empty())
				inliers.create(1, numPts, CV_8U);
			for (unsigned int i = 0; i < numPts; ++i)
			{
				if (fund->usac_results_.inlier_flags_[i] == 0)
					inliers_.at<bool>(i) = false;
				else
				{
					inliers_.at<bool>(i) = true;
					nr_inliers_++;
				}
			}
			inliers_.copyTo(inliers.getMat());
			if (nr_inliers)
				*nr_inliers = nr_inliers_;
		}
	}
	else if (nr_inliers)
	{
		for (unsigned int i = 0; i < numPts; ++i)
		{
			if (fund->usac_results_.inlier_flags_[i])
				nr_inliers_++;
		}
		*nr_inliers = nr_inliers_;
	}
	
	// clean up
	pointData.clear();
	fund->cleanupProblem();
	delete fund;

	return(EXIT_SUCCESS);
}


int estimateEssentialQDEGSAC(cv::InputArray p1,
	cv::InputArray p2,
	cv::OutputArray E,
	double & sprt_delta_result,
	double & sprt_epsilon_result,
	double th,
	double focalLength,
	double th_pixels, 
	double prosac_beta,
	double sprt_delta,
	double sprt_epsilon,
	double t_red,
	USACConfig::EssentialMatEstimatorUsed used_estimator,
	USACConfig::RefineAlgorithm	refineMethod,
	cv::OutputArray inliers, 
	unsigned int *nr_inliers,
	cv::OutputArray R, 
	cv::OutputArray inliers_degenerate_R, 
	double *fraction_degen_inliers_R,
	std::vector<unsigned int> sortedMatchIdx,
	cv::OutputArray R_E, 
	cv::OutputArray t_E)
{
	CV_Assert((p1.cols() == 2) && (p2.cols() == 2) && (p1.rows() == p2.rows()) && (p1.type() == CV_64FC1) && (p2.type() == CV_64FC1));
	cv::Mat p1_ = p1.getMat();
	cv::Mat p2_ = p2.getMat();
	cv::Mat E_, E_upgrade, inliers_init, inliers_degen, inliers_degen_all_out, inliers_upgrade, R_kneip, t_kneip, R_kneip_upgr, t_kneip_upgr, R_, p1_inl, p2_inl;
	cv::Mat E_res, R_kneip_res, t_kneip_res, inliers_res;
	unsigned int nr_inliers_init = 0, nr_inliers_deg = 0, nr_inliers_upgrade = 0, nr_inliers_res = 0;
	double inl_deg_ratio = 0;
	unsigned int numPts = p1_.rows;
	std::vector<unsigned int> sortedMatchIdxDegen;
	bool isDegenerate = false;
	double sprt_delta_result_tmp = 0, sprt_epsilon_result_tmp = 0, sprt_delta_result_tmp_deg = 0, sprt_epsilon_result_tmp_deg = 0;

	if (estimateEssentialMatUsac(p1,
		p2,
		E_,
		sprt_delta_result_tmp,
		sprt_epsilon_result_tmp,
		th,
		focalLength,
		th_pixels,
		prosac_beta,
		sprt_delta,
		sprt_epsilon,
		false,
		used_estimator,
		refineMethod,
		inliers_init,
		&nr_inliers_init,
		cv::noArray(),
		cv::noArray(),
		R_kneip,
		cv::noArray(),
		t_kneip,
		cv::noArray(),
		cv::noArray(),
		NULL,
		NULL,
		NULL,
		NULL,
		sortedMatchIdx) == EXIT_FAILURE)
	{
		return EXIT_FAILURE;
	}

	p1_inl.create(nr_inliers_init, 2, CV_64FC1);
	p2_inl.create(nr_inliers_init, 2, CV_64FC1);
	sortedMatchIdxDegen.resize(nr_inliers_init);
	for (unsigned int i = 0, count = 0; i < numPts; i++)
	{
		if (inliers_init.at<bool>(i))
		{
			p1_inl.row(count) = p1_.row(i);
			p2_inl.row(count) = p2_.row(i);
			sortedMatchIdxDegen[count] = sortedMatchIdx[i];
			count++;
		}
	}

	if(estimateRotationMatUsac(p1_inl,
		p2_inl,
		R_,
		sprt_delta_result_tmp_deg,
		sprt_epsilon_result_tmp_deg,
		focalLength,
		th_pixels,
		prosac_beta,
		sprt_delta,
		sprt_epsilon,
		inliers_degen,
		&nr_inliers_deg,
		sortedMatchIdxDegen) == EXIT_FAILURE)
	{
		return EXIT_FAILURE;
	}

	inl_deg_ratio = (double)nr_inliers_deg / (double)nr_inliers_init;

	if (inl_deg_ratio > t_red)
	{
		inliers_degen_all_out.create(1, numPts, CV_8U);
		for (unsigned int i = 0, count = 0; i < numPts; i++)
		{
			if (inliers_init.at<bool>(i))
			{
				if (inliers_degen.at<bool>(count))
					inliers_degen_all_out.at<bool>(i) = true;
				else
					inliers_degen_all_out.at<bool>(i) = false;
				count++;
			}
			else
			{
				inliers_degen_all_out.at<bool>(i) = false;
			}
		}

		if(upgradeEssentialMatDegenUsac(p1,
			p2,
			inliers_degen_all_out,
			E_upgrade,
			sprt_delta_result_tmp,
			sprt_epsilon_result_tmp,
			th,
			focalLength,
			th_pixels,
			sprt_delta,
			sprt_epsilon,
			used_estimator,
			refineMethod,
			inliers_upgrade,
			&nr_inliers_upgrade,
			R_kneip_upgr,
			t_kneip_upgr) == EXIT_FAILURE)
		{
			return EXIT_FAILURE;
		}

		if (nr_inliers_upgrade > nr_inliers_deg)
		{
			E_res = E_upgrade;
			inliers_res = inliers_upgrade;
			R_kneip_res = R_kneip_upgr;
			t_kneip_res = t_kneip_upgr;
			nr_inliers_res = nr_inliers_upgrade;
			isDegenerate = false;
			sprt_delta_result = sprt_delta_result_tmp;
			sprt_epsilon_result = sprt_epsilon_result_tmp;
		}
		else
		{
			isDegenerate = true;
			E_res.release();
			inliers_res.release();
			R_kneip_res.release();
			t_kneip_res.release();
			nr_inliers_res = 0;
			sprt_delta_result = sprt_delta_result_tmp_deg;
			sprt_epsilon_result = sprt_epsilon_result_tmp_deg;
			if (R.needed())
			{
				if (R.empty())
					R.create(3, 3, CV_64F);
				R_.copyTo(R.getMat());
			}
			if (inliers_degenerate_R.needed())
			{
				if (inliers_degenerate_R.empty())
					inliers_degenerate_R.create(1, numPts, CV_8U);
				inliers_degen_all_out.copyTo(inliers_degenerate_R.getMat());
			}
			if (fraction_degen_inliers_R)
				*fraction_degen_inliers_R = (double)nr_inliers_deg / (double)numPts;
		}
	}
	else
	{
		E_res = E_;
		inliers_res = inliers_init;
		R_kneip_res = R_kneip;
		t_kneip_res = t_kneip;
		nr_inliers_res = nr_inliers_init;
	}

	if (E.needed())
	{
		if (!E_res.empty())
		{
			if (E.empty())
				E.create(3, 3, CV_64F);
			E_res.copyTo(E.getMat());
		}
		else
			E.clear();
	}
	if (inliers.needed())
	{
		if (!inliers_res.empty())
		{
			if (inliers.empty())
				inliers.create(3, 3, CV_64F);
			inliers_res.copyTo(inliers.getMat());
		}
		else
			inliers.clear();
	}
	if (R_E.needed())
	{
		if (!R_kneip_res.empty())
		{
			if (R_E.empty())
				R_E.create(3, 3, CV_64F);
			R_kneip_res.copyTo(R_E.getMat());
		}
		else
			R_E.clear();
	}
	if (t_E.needed())
	{
		if (!t_kneip_res.empty())
		{
			if (t_E.empty())
				t_E.create(3, 1, CV_64F);
			t_kneip_res.copyTo(t_E.getMat());
		}
		else
			t_E.clear();
	}
	if (nr_inliers)
		*nr_inliers = nr_inliers_res;
	if (!isDegenerate)
	{
		if (R.needed())
		{
			if (!R.empty())
				R.clear();
		}
		if (inliers_degenerate_R.needed())
		{
			if (!inliers_degenerate_R.empty())
				inliers_degenerate_R.release();
		}
		if (fraction_degen_inliers_R)
			*fraction_degen_inliers_R = 0;
	}


	return(EXIT_SUCCESS);
}

