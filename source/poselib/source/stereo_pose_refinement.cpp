/**********************************************************************************************************
FILE: stereo_pose_refinement.cpp

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: October 2017

LOCATION: TechGate Vienna, Donau-City-Stra?e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functions for pose refinement with multiple stereo pairs
**********************************************************************************************************/

#include "poselib/stereo_pose_refinement.h"
#include "poselib/pose_homography.h"
#include "poselib/pose_linear_refinement.h"
#include <iomanip>
#include <unordered_set>

//#include "poselib/pose_helper.h"
//#include <memory>
//#include <numeric>      // std::accumulate
//#include "opencv2/core/eigen.hpp"


using namespace std;
using namespace cv;

namespace poselib
{

	/* --------------------------- Defines --------------------------- */


	/* --------------------- Function prototypes --------------------- */

	double getWeightingValuesInv(double value, double max_value, double min_value = 0);
	double getWeightingValues(double value, double max_value, double min_value = 0);

	/* --------------------- Functions --------------------- */

	void StereoRefine::setNewParameters(ConfigPoseEstimation cfg_pose_)
	{
		bool checkTh = false;
		if (!poselib::nearZero(cfg_pose.th_pix_user - cfg_pose_.th_pix_user) && (cfg_pose_.th_pix_user < cfg_pose.th_pix_user) && !points1Cam.empty())
		{
			//check if the new threshold leads to a noticable reduction of the already found correspondences
			checkTh = true;
		}
		cfg_pose = cfg_pose_;
		th = cfg_pose_.th_pix_user * pixToCamFact; //Inlier threshold
		th2 = th * th;
		checkInputParamters();
		t_mea = 0;
		t_oa = 0;
		if (checkTh)
		{
			std::vector<double> errors;
			cv::Mat mask;
			size_t nr_inliers_tmp = getInliers(E_new, points1Cam, points2Cam, mask, errors);
			double inlRat_new = (double)nr_inliers_tmp / (double)points1Cam.rows;
			if (inlRat_new < inlier_ratio_history.back() * (1 - cfg_pose.relInlRatThNew))
			{
				//Too many correspondences from the last image pairs are marked now as outliers ->reinitializing System
				cout << std::setprecision(3) << "Due to changing the threshold, too many of the correspondences from the last image pairs are marked now as outliers. Reinitializing system!" << endl;
				clearHistoryAndPool();
			}
		}
	}

	void StereoRefine::checkInputParamters()
	{
		if ((cfg_pose.refineMethod_CorrPool & 0xF) == poselib::RefinePostAlg::PR_NO_REFINEMENT)
		{
			cout << "No refinement algorithm for estimating the pose with all correspondences is set. Taking default values!" << endl;
			cfg_pose.refineMethod_CorrPool = poselib::RefinePostAlg::PR_STEWENIUS | poselib::RefinePostAlg::PR_PSEUDOHUBER_WEIGHTS;
		}
		if (cfg_pose.minStartAggInlRat < 0.08)
		{
			cout << std::setprecision(4) << "The minimum inlier ratio treshold minStartAggInlRat = " << cfg_pose.minStartAggInlRat <<
				" to start aggregating point correspondence is chosen too small. "
				"This would lead to acceptance of possibly wrong initial poses and their correspondences for further refinement and thus to wrong poses in the future!"
				" Setting it to 0.1 which might be still too small!" << endl;
			cfg_pose.minStartAggInlRat = 0.1;
		}
		else if (cfg_pose.minStartAggInlRat > 0.75)
		{
			cout << std::setprecision(3) << "The minimum inlier ratio treshold minStartAggInlRat = " << cfg_pose.minStartAggInlRat <<
				" to start aggregating point correspondence is chosen too high. "
				"This would lead to rejection of correct initial poses and their correspondences for further refinement! Thus the behaviour of the algorithm might be "
				"similar to a pose estimation in the mono camera case."
				" Setting it to 0.75 which might be still too high!" << endl;
			cfg_pose.minStartAggInlRat = 0.75;
		}
		if (cfg_pose.relInlRatThLast > 0.75)
		{
			cout << std::setprecision(3) << "Setting the relative threshold relInlRatThLast to " <<
				cfg_pose.relInlRatThLast << " makes no sence. The value is too high! Thus a change in the pose might not be detected! Setting it to 0.6 which might be still to high!" << endl;
			cfg_pose.relInlRatThLast = 0.6;
		}
		else if (cfg_pose.relInlRatThLast < 0.1)
		{
			cout << std::setprecision(3) << "Setting the relative threshold relInlRatThLast to " <<
				cfg_pose.relInlRatThLast << " makes no sence. The value is too small! This might cause additional computational overhead!" << endl;
			if (cfg_pose.relInlRatThLast < 0.01)
			{
				cout << "Relative threshold relInlRatThLast is far too small! Setting it to 0.1!" << endl;
				cfg_pose.relInlRatThLast = 0.1;
			}
		}
		if (cfg_pose.relInlRatThNew < 0.04)
		{
			cout << std::setprecision(4) << "Setting the relative threshold relInlRatThNew to " <<
				cfg_pose.relInlRatThNew << " makes no sence. The value is too small! The algorithm might not realize that the pose has NOT changed. This can result in a loss of accuracy!"
				" Setting it to 0.04 which might be still too small!" << endl;
			cfg_pose.relInlRatThNew = 0.04;
		}
		else if (cfg_pose.relInlRatThNew > 0.5)
		{
			cout << std::setprecision(3) << "Setting the relative threshold relInlRatThNew to " <<
				cfg_pose.relInlRatThNew << " makes no sence. The value is too high! If the pose has changed, this might be not realized and correspondeces of different poses are used together"
				" leading to a wrong pose estimation! Setting it to 0.35 which might be still too high!" << endl;
			cfg_pose.relInlRatThNew = 0.35;
		}
		if (cfg_pose.minInlierRatSkip > 0.6)
		{
			cout << std::setprecision(3) << "Your chosen inlier threshold minInlierRatSkip = " << cfg_pose.minInlierRatSkip <<
				" for a newly robustly estimated pose to be temporally accepted seems to be very high. "
				"If the inlier ratio of the new pose is below this threshold, the new pose might not be accepted "
				"(except the chosen realtive threshold relMinInlierRatSkip helps to prevent this situation)! "
				"If the pose has changed, the new estimated pose might not be temporally accepted and the old wrong pose will be used instead!"
				" You should change this threshold!" << endl;
			if (cfg_pose.minInlierRatSkip > 0.95)
			{
				cout << "Changing minInlierRatSkip to 0.95" << endl;
				cfg_pose.minInlierRatSkip = 0.95;
			}
		}
		else if (cfg_pose.minInlierRatSkip < 0.1)
		{
			cout << std::setprecision(4) << "Your chosen inlier threshold minInlierRatSkip = " << cfg_pose.minInlierRatSkip <<
				" for a newly robustly estimated pose to be temporally accepted seems to be very small. "
				"Thus, if the pose has not changed but the quality of the actual image pair is very bad resulting in a bad temporal new estimate of the pose, "
				"this new bad pose will be available at the output. You should change this threshold!" << endl;
			if (cfg_pose.minInlierRatSkip < 0.01)
			{
				cout << "Changing minInlierRatSkip to 0.1" << endl;
				cfg_pose.minInlierRatSkip = 0.1;
			}
		}
		if (cfg_pose.relMinInlierRatSkip < 0.2)
		{
			cout << std::setprecision(4) << "Your chosen relative inlier threshold relMinInlierRatSkip = " << cfg_pose.relMinInlierRatSkip <<
				" for a newly robustly estimated pose to be temporally accepted seems to be very small. "
				"Thus, if the pose has not changed but the quality of the actual image pair is very bad resulting in a bad temporal new estimate of the pose, "
				"this new bad pose will be available at the output. You should change this threshold!" << endl;
			if (cfg_pose.relMinInlierRatSkip < 0.01)
			{
				cout << "Changing relMinInlierRatSkip to 0.1" << endl;
				cfg_pose.relMinInlierRatSkip = 0.1;
			}
		}
		else if (cfg_pose.relMinInlierRatSkip > 1.0)
		{
			cout << "Relative threshold relMinInlierRatSkip out of range! Changing relMinInlierRatSkip to 1.0" << endl;
			cfg_pose.relMinInlierRatSkip = 1.0;
		}
		if (cfg_pose.maxSkipPairs < 2)
		{
			cout << "The chosen threshold on the number of image pairs maxSkipPairs = " << cfg_pose.maxSkipPairs <<
				" after which the whole system is reinitialized is very small!"
				"Thus, if the structure within this number of image pairs is too bad to estimate a good pose, "
				"the system will be reinitialized rejecting all previous good correspondences." << endl;
			if (cfg_pose.maxSkipPairs == 0)
			{
				cout << "Changing maxSkipPairs to 1 which is still very small!" << endl;
				cfg_pose.maxSkipPairs = 1;
			}
		}
		else if (cfg_pose.maxSkipPairs > 30)
		{
			cout << "The chosen threshold on the number of image pairs maxSkipPairs = " << cfg_pose.maxSkipPairs <<
				" after which the whole system is reinitialized is very high!"
				" The system might need very long to reinitialize if the pose changes!" << endl;
			if (cfg_pose.maxSkipPairs > 200)
			{
				cout << "Setting maxSkipPairs to 200 which is still very high!" << endl;
				cfg_pose.maxSkipPairs = 200;
			}
		}
		if (cfg_pose.minInlierRatioReInit <= cfg_pose.minInlierRatSkip)
		{
			cout << std::setprecision(4) << "Your threshold minInlierRatioReInit = " << cfg_pose.minInlierRatioReInit <<
				" on the inlier ratio of the newest robust pose estimation after a few iterations (input image pairs) is too small!"
				" It should be larger than the threshold minInlierRatSkip = " << cfg_pose.minInlierRatSkip <<
				". Setting it to minInlierRatSkip + 0.05" << endl;
			cfg_pose.minInlierRatioReInit = cfg_pose.minInlierRatSkip + 0.05;
		}
		if (cfg_pose.minInlierRatioReInit > 0.8)
		{
			cout << std::setprecision(3) << "Your threshold minInlierRatioReInit = " << cfg_pose.minInlierRatioReInit <<
				" on the inlier ratio of the newest robust pose estimation after a few iterations (input image pairs) is too high!"
				" The system might not reinitialize very fast if the pose has changed!" 
				" Changing minInlierRatioReInit to 0.8 which is still very high!" << endl;
			cfg_pose.minInlierRatioReInit = 0.8;
		}
		else if (cfg_pose.minInlierRatioReInit < 0.3)
		{
			cout << std::setprecision(4) << "Your threshold minInlierRatioReInit = " << cfg_pose.minInlierRatioReInit <<
				" on the inlier ratio of the newest robust pose estimation after a few iterations (input image pairs) is too small!"
				" The system will reinitialize very often with possible false poses!" << endl;
			if (cfg_pose.minInlierRatioReInit < 0.15)
			{
				cout << "Changing minInlierRatioReInit to 0.15 which is still very small!" << endl;
				cfg_pose.minInlierRatioReInit = 0.15;
			}
		}
		if (cfg_pose.minPtsDistance < 1.5)
		{
			cout << std::setprecision(2) << "The search distance for correspondences of " << cfg_pose.minPtsDistance << " is too small. Setting it to the minimal value of 1.5" << endl;
			cfg_pose.minPtsDistance = 1.5;
		}
	}

	int StereoRefine::addNewCorrespondences(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2, poselib::ConfigUSAC cfg)
	{
		cfg_usac = cfg;
		t_mea = 0;
		t_oa = 0;
		points1new.clear();
		points2new.clear();
		nr_corrs_new = matches.size();
		//Extract coordinates from keypoints
		for (size_t i = 0; i < nr_corrs_new; i++)
		{
			points1new.push_back(kp1[matches[i].queryIdx].pt);
			points2new.push_back(kp2[matches[i].trainIdx].pt);
		}

		if (cfg_pose.verbose > 5)////////////////////////////Adapt verbose values
		{
			t_mea = (double)getTickCount(); //Start time measurement
		}

		//Transfer into camera coordinates
		poselib::ImgToCamCoordTrans(points1new, *cfg_pose.K0);
		poselib::ImgToCamCoordTrans(points2new, *cfg_pose.K1);

		//Undistort
		if (!poselib::Remove_LensDist(points1new, points2new, *cfg_pose.dist0_8, *cfg_pose.dist1_8))
		{
			std::cout << "Removing lens distortion failed!" << endl;
			exit(1);
		}
		//Convert into cv::Mat format
		points1newMat.release();
		points2newMat.release();
		cv::Mat(points1new).reshape(1).convertTo(points1newMat, CV_64FC1);
		cv::Mat(points2new).reshape(1).convertTo(points2newMat, CV_64FC1);
		points1newMat.copyTo(points1newMat_tmp);
		points2newMat.copyTo(points2newMat_tmp);

		if (cfg_pose.verbose > 5)
		{
			t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
			std::cout << "Time for coordinate conversion & undistortion (2 imgs): " << t_mea << "ms" << endl;
			t_oa = t_mea;
		}

		double inlier_ratio_new1 = 0;
		if (nrEstimation == 0)
		{
			//First estimation
			if (robustPoseEstimation())
			{
				return -1;
			}
			inlier_ratio_new1 = (double)nr_inliers_new / (double)nr_corrs_new;
			if (inlier_ratio_new1 < cfg_pose.minStartAggInlRat)
			{
				cout << "Inlier ratio too small! Skipping aggregation of correspondences! The output pose of this and the next iteration will be like in the mono camera case!" << endl;
				return 0;
			}
			if (addCorrespondencesToPool(matches, kp1, kp2))
				return -2;
			pose_history.push_back(poseHist(E_new.clone(), R_new.clone(), t_new.clone()));
			inlier_ratio_history.push_back(inlier_ratio_new1);
			nrEstimation++;
		}
		else
		{
			cv::Mat mask;
			std::vector<double> errorNew;
			bool addToPool = false;
			size_t nr_inliers_tmp = 0;
			double inlier_ratio_new;
			nr_inliers_tmp = getInliers(E_new, points1newMat, points2newMat, mask, errorNew);
			inlier_ratio_new = (double)nr_inliers_tmp / (double)nr_corrs_new;
			//check if the new inlier ratio is approximately equal to the old one as it is not likely that the image content changes completely from image pair to image pair
			if (inlier_ratio_new < ((1.0 - cfg_pose.relInlRatThLast) * inlier_ratio_history.back()))
			{
				//Perform new robust estimation to check if the pose has changed
				if (robustPoseEstimation())
				{
					return -1;
				}
				inlier_ratio_new1 = (double)nr_inliers_new / (double)nr_corrs_new;
				if (inlier_ratio_new < inlier_ratio_new1 * (1 - cfg_pose.relInlRatThNew))
				{
					//Either the pose has changed or the new image pair is really bad
					if (inlier_ratio_new1 >= cfg_pose.minInlierRatioReInit)
					{
						//It seems that the pose hase changed as the inlier ratio is quite good after robust estimation
						clearHistoryAndPool();
						if (addCorrespondencesToPool(matches, kp1, kp2))
							return -2;
						pose_history.push_back(poseHist(E_new.clone(), R_new.clone(), t_new.clone()));
						inlier_ratio_history.push_back(inlier_ratio_new1);
						nrEstimation++;
						cout << "The pose has changed! System is reinitialized!" << endl;
						return 0;
					}
					else
					{
						if ((inlier_ratio_new1 < cfg_pose.minInlierRatSkip) && (inlier_ratio_new1 < (cfg_pose.relMinInlierRatSkip * inlier_ratio_history.back())))
						{
							//It seems that the new image pair is really bad -> skip the whole estimation procedure and restore the old pose
							E_new = pose_history.back().E.clone();
							R_new = pose_history.back().R.clone();
							t_new = pose_history.back().t.clone();
							cout << "It seems that the new image pair is really bad. Restoring last valid pose! Be aware that the 3D points might not be valid!" << endl;
						}
						else
						{
							//We are not sure if the pose has changed or if the image pair is too bad -> taking new pose values but keeping history and 
							//do not add this correspondences to the pool or anything to the history
							cout << "Either the pose has changed or the image pair has bad quality! Taking new pose which might be wrong!" << endl;
						}
						skipCount++;
					}
				}
				else
				{
					//The new estimated pose seems to be similar to the old one, so add the correspondences to the pool and perform refinement on all availble correspondences
					addToPool = true;
					errorNew.clear();
					computeReprojError2(points1newMat, points2newMat, E_new, errorNew);
				}
			}
			else
			{
				//Add the new correspondences to the pool and perform refinement on all availble correspondences
				addToPool = true;
				mask_E_new.release();
				mask.copyTo(mask_E_new);
				nr_inliers_new = nr_inliers_tmp;
				inlier_ratio_new1 = inlier_ratio_new;
			}

			if (addToPool)
			{
				//Perform filtering of correspondences and refinement on all available correspondences
				if (filterNewCorrespondences(matches, kp1, kp2, errorNew))
				{
					//Failed to remove some old correspondences as an invalid iterator was detected -> reinitialize system
					clearHistoryAndPool();
					if (addCorrespondencesToPool(matches, kp1, kp2))
						return -2;
					pose_history.push_back(poseHist(E_new.clone(), R_new.clone(), t_new.clone()));
					inlier_ratio_history.push_back(inlier_ratio_new1);
					nrEstimation++;
				}
				if (addCorrespondencesToPool(matches, kp1, kp2))
					return -2;

				//Perform refinement using all available correspondences
				cv::Mat E_old = E_new.clone();
				cv::Mat R_old = R_new.clone();
				cv::Mat t_old = t_new.clone();
				static size_t failed_refinements = 0;
				if (refinePoseFromPool())
				{
					cout << "Taking old pose!" << endl;
					E_old.copyTo(E_new);
					R_old.copyTo(R_new);
					t_old.copyTo(t_new);
					skipCount++;
					if (failed_refinements > 2)
					{
						failed_refinements = 0;
						cout << "Reinitializing system!" << endl;
						clearHistoryAndPool();
					}
					failed_refinements++;
					return -1;
				}
				failed_refinements = 0;
				if ((double)nr_inliers_new < 0.75 * (double)correspondencePool.size())
				{
					cout << "Too less inliers (<75%) after refinement! Reinitializing system and taking old pose!" << endl;
					E_old.copyTo(E_new);
					R_old.copyTo(R_new);
					t_old.copyTo(t_new);
					clearHistoryAndPool();
					return -1;
				}

				//Calculate inliers of the new image pair with the new E from all image pairs
				nr_inliers_tmp = getInliers(E_new, points1newMat_tmp, points2newMat_tmp, mask, errorNew);
				inlier_ratio_new = (double)nr_inliers_tmp / (double)points1newMat_tmp.rows;
				if (inlier_ratio_new < inlier_ratio_new1 * (1 - cfg_pose.relInlRatThNew))
				{
					cout << "Inlier ratio of new image pair calculated with refined E over all image pairs is too small compared to its initial inlier ratio! Reinitializing system and taking old pose!" << endl;
					E_old.copyTo(E_new);
					R_old.copyTo(R_new);
					t_old.copyTo(t_new);
					clearHistoryAndPool();
					return -1;
				}
				inlier_ratio_history.push_back(inlier_ratio_new);
				pose_history.push_back(poseHist(E_new.clone(), R_new.clone(), t_new.clone()));

				//Get error statistic for the newest image pair
				vector<double> errorNew_tmp(nr_inliers_tmp);
				for (size_t i = 0, count = 0; i < nr_inliers_tmp; i++)
				{
					if (mask.at<bool>(i))
						errorNew_tmp[count++] = errorNew[i];
				}
				statVals err_stats;
				getStatsfromVec(errorNew_tmp, &err_stats);
				errorStatistic_history.push_back(err_stats);

				//Delete elements marked as outliers
				if (nr_inliers_new < correspondencePool.size())
				{
					cv::Mat mask_Q_tmp(1, nr_inliers_new, CV_8U, true);
					cv::Mat Q_tmp(nr_inliers_new, 3, Q.type());
					std::unordered_set<size_t> delIdxNew;
					vector<size_t> delIdxPool;
					for (size_t i = 0, count = 0; i < correspondencePool.size(); i++)
					{
						if (!mask_E_new.at<bool>(i))
						{
							delIdxNew.insert(i);
						}
						else
						{
							Q.row(i).copyTo(Q_tmp.row(count));
							if (!mask_Q_new.at<bool>(i))
							{
								mask_Q_tmp.at<bool>(count) = false;
							}
							count++;
						}
					}
					mask_Q_tmp.copyTo(mask_Q_new);
					Q_tmp.copyTo(Q);
					for (auto &it : correspondencePool)
					{
						if (delIdxNew.find(it.ptIdx) != delIdxNew.end())
						{
							delIdxPool.push_back(it.poolIdx);
						}
					}
					if (poolCorrespondenceDelete(delIdxPool))
					{
						//Failed to remove some old correspondences as an invalid iterator was detected -> reinitialize system
						clearHistoryAndPool();
						if (addCorrespondencesToPool(matches, kp1, kp2))
							return -2;
						pose_history.push_back(poseHist(E_new.clone(), R_new.clone(), t_new.clone()));
						inlier_ratio_history.push_back(inlier_ratio_new1);
						nrEstimation++;
						return -2;
					}

					//Calculate error values with new E, and add 3D points
					std::vector<double> error;
					computeReprojError2(points1Cam, points2Cam, E_new, error);
					size_t count = 0;
					for (auto &it : correspondencePool)
					{
						if (!Q.empty())
						{
							it.Q = cv::Point3d(Q.row(count));
							it.Q_tooFar = !mask_Q_new.at<bool>(count);
						}
						it.SampsonErrors.push_back(error[count++]);
						it.meanSampsonError = 0;
						for (auto &e : it.SampsonErrors)
						{
							it.meanSampsonError += e;
						}
						it.meanSampsonError /= (double)it.SampsonErrors.size();
					}

					nrEstimation++;
					skipCount = 0;
				}
			}

			if (skipCount > cfg_pose.maxSkipPairs)
			{
				//Reinitialize whole system
				clearHistoryAndPool();
				if (addCorrespondencesToPool(matches, kp1, kp2))
					return -2;
				pose_history.push_back(poseHist(E_new.clone(), R_new.clone(), t_new.clone()));
				inlier_ratio_history.push_back(inlier_ratio_new1);
				nrEstimation++;
			}
		}

		return 0;
	}

	void StereoRefine::clearHistoryAndPool()
	{
		points1Cam.release();
		points2Cam.release();
		correspondencePool.clear();
		correspondencePoolIdx.clear();
		if (kdTreeLeft)
		{
			kdTreeLeft->killTree();
			kdTreeLeft.release();
		}
		corrIdx = 0;
		nrEstimation = 0;
		skipCount = 0;
		/*mask_E_old.release();
		mask_Q_old.release();*/
		//deletionIdxs.clear();
		pose_history.clear();
		inlier_ratio_history.clear();
		errorStatistic_history.clear();
	}

	int StereoRefine::addCorrespondencesToPool(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2)
	{
		CoordinateProps tmp;
		std::vector<double> errors;
		size_t nrEntries = 0;
		bool isInitMat = true;
		if (!points1Cam.empty())
		{
			for (auto &it : correspondencePool)
			{
				it.age++;
			}
			isInitMat = false;
			nrEntries = (size_t)points1Cam.rows;
			size_t reservesize = nrEntries + nr_inliers_new;
			points1Cam.reserve(reservesize);
			points2Cam.reserve(reservesize);
			correspondencePoolIdx.reserve(reservesize);
		}
		//Initialize KD tree
		if (!kdTreeLeft)
		{
			kdTreeLeft.reset(new keyPointTreeInterface(&correspondencePool, &correspondencePoolIdx));
			if (kdTreeLeft->buildInitialTree())
			{
				clearHistoryAndPool();
				return -1;
			}
		}
		
		for (size_t i = 0, count = nrEntries; i < nr_corrs_new; i++)
		{
			if (mask_E_new.at<bool>(i))
			{
				points1Cam.push_back(points1newMat.row(i));
				points2Cam.push_back(points2newMat.row(i));
				tmp.age = 1;
				tmp.descrDist = matches[i].distance;
				if (descrDist_max < tmp.descrDist) descrDist_max = tmp.descrDist;
				tmp.keyPResponses[0] = kp1[matches[i].queryIdx].response;
				if (keyPRespons_max < tmp.keyPResponses[0]) keyPRespons_max = tmp.keyPResponses[0];
				tmp.keyPResponses[1] = kp2[matches[i].trainIdx].response;
				if (keyPRespons_max < tmp.keyPResponses[1]) keyPRespons_max = tmp.keyPResponses[1];
				if (isInitMat)
				{
					tmp.meanSampsonError = getSampsonL2Error(E_new, points1newMat.row(i), points2newMat.row(i));
					tmp.SampsonErrors.push_back(tmp.meanSampsonError);
					errors.push_back(tmp.meanSampsonError);
					if (!Q.empty())
					{
						tmp.Q = cv::Point3d(Q.row(i));
						tmp.Q_tooFar = !mask_Q_new.at<bool>(i);
					}
				}
				tmp.nrFound = 1;
				tmp.pt1 = kp1[matches[i].queryIdx].pt;
				tmp.pt2 = kp2[matches[i].trainIdx].pt;
				tmp.ptIdx = count;
				tmp.poolIdx = corrIdx;
				correspondencePool.push_back(tmp);
				tmp.SampsonErrors.clear();
				correspondencePoolIdx.insert({ corrIdx, --correspondencePool.end() });
				corrIdx++;
				count++;
			}
		}
		if (isInitMat)
		{
			statVals err_stats;
			getStatsfromVec(errors, &err_stats);
			errorStatistic_history.push_back(err_stats);
		}
		
		if(kdTreeLeft && ((double)correspondencePool.size() / (double)corrIdx < 0.5))//Check if not too many items were detelted from the KD tree index
		{
			corrIdx = 0;
			correspondencePoolIdx.clear();
			std::list<CoordinateProps>::iterator it = correspondencePool.begin();
			for (size_t i = 0; i < correspondencePool.size(); i++)
			{
				it->poolIdx = corrIdx;
				correspondencePoolIdx.insert({ corrIdx++, it++ });
			}
			if (kdTreeLeft->resetTree(&correspondencePool, &correspondencePoolIdx))
			{
				clearHistoryAndPool();
				return -1;
			}
		}

		//add coordinates to the KD tree
		if (kdTreeLeft->addElements(corrIdx - nr_inliers_new, nr_inliers_new))
		{
			clearHistoryAndPool();
			return -1;
		}

		return 0;
	}

	int StereoRefine::robustPoseEstimation()
	{
		//Get essential matrix
		cv::Mat E, mask;
		cv::Mat R, t;
		cv::Mat R_kneip = cv::Mat::eye(3, 3, CV_64FC1), t_kneip = cv::Mat::zeros(3, 1, CV_64FC1);
		Q.release();
		//double pixToCamFact = 4.0 / (std::sqrt(2.0) * (K0.at<double>(0, 0) + K0.at<double>(1, 1) + K1.at<double>(0, 0) + K1.at<double>(1, 1)));
		//double th = cfg_pose.th_pix_user * pixToCamFact; //Inlier threshold

		if (cfg_pose.verbose > 3)
		{
			t_mea = (double)getTickCount(); //Start time measurement
		}

		if (cfg_pose.autoTH)
		{
			int inlierPoints;
			poselib::AutoThEpi Eautoth(pixToCamFact);
			if (Eautoth.estimateEVarTH(points1newMat, points2newMat, E, mask, &th, &inlierPoints) != 0)
			{
				std::cout << "Estimation of essential matrix using automatic threshold estimation and ARRSAC failed!" << endl;
				return -1;
			}

			std::cout << "Estimated threshold: " << th / pixToCamFact << " pixels" << endl;
		}
		else if (cfg_pose.Halign)
		{
			int inliers;
			if (poselib::estimatePoseHomographies(points1newMat, points2newMat, R, t, E, th, inliers, mask, false, cfg_pose.Halign > 1 ? true : false) != 0)
			{
				std::cout << "Estimation of essential matrix using homography alignment failed!" << endl;
				return -1;
			}
		}
		else
		{
			if (!cfg_pose.RobMethod.compare("USAC"))
			{
				bool isDegenerate = false;
				Mat R_degenerate, inliers_degenerate_R;
				bool usacerror = false;
				if (cfg_usac.refinealg == poselib::RefineAlg::REF_EIG_KNEIP || cfg_usac.refinealg == poselib::RefineAlg::REF_EIG_KNEIP_WEIGHTS)
				{
					if (estimateEssentialOrPoseUSAC(points1newMat,
						points2newMat,
						E,
						th,
						cfg_usac,
						isDegenerate,
						mask,
						R_degenerate,
						inliers_degenerate_R,
						R_kneip,
						t_kneip) != 0)
					{
						usacerror = true;
					}
				}
				else
				{
					if (estimateEssentialOrPoseUSAC(points1newMat,
						points2newMat,
						E,
						th,
						cfg_usac,
						isDegenerate,
						mask,
						R_degenerate,
						inliers_degenerate_R) != 0)
					{
						usacerror = true;
					}
				}
				if (usacerror)
				{
					std::cout << "Estimation of essential matrix using USAC!" << endl;
					return -1;
				}
				if (isDegenerate)
				{
					std::cout << "Camera configuration is degenerate and, thus, rotation only. Skipping further calculations! Rotation angles: " << endl;
					double roll, pitch, yaw;
					poselib::getAnglesRotMat(R_degenerate, roll, pitch, yaw);
					std::cout << "roll: " << roll << char(248) << ", pitch: " << pitch << char(248) << ", yaw: " << yaw << char(248) << endl;
					return -2;
				}
			}
			else
			{
				if (!poselib::estimateEssentialMat(E, points1newMat, points2newMat, cfg_pose.RobMethod, th, cfg_pose.refineRTold, mask))
				{
					std::cout << "Estimation of essential matrix using " << cfg_pose.RobMethod << " failed!" << endl;
					return -1;
				}
			}
		}
		mask.copyTo(mask_E_new);
		nr_inliers_new = cv::countNonZero(mask);
		std::cout << "Number of inliers after robust estimation of E: " << nr_inliers_new << endl;

		//Get R & t
		bool availableRT = false;
		if (cfg_pose.Halign)
		{
			R_kneip = R;
			t_kneip = t;
		}
		if (cfg_pose.Halign ||
			(!cfg_pose.RobMethod.compare("USAC") && (cfg_usac.refinealg == poselib::RefineAlg::REF_EIG_KNEIP ||
				cfg_usac.refinealg == poselib::RefineAlg::REF_EIG_KNEIP_WEIGHTS)))
		{
			double sumt = 0;
			for (int i = 0; i < 3; i++)
			{
				sumt += t_kneip.at<double>(i);
			}
			if (!poselib::nearZero(sumt) && poselib::isMatRoationMat(R_kneip))
			{
				availableRT = true;
			}
		}
		
		if (cfg_pose.Halign && ((cfg_pose.refineMethod & 0xF) == poselib::RefinePostAlg::PR_NO_REFINEMENT))
		{
			poselib::triangPts3D(R, t, points1newMat, points2newMat, Q, mask);
		}
		else
		{
			if (cfg_pose.refineRTold)
			{
				poselib::robustEssentialRefine(points1newMat, points2newMat, E, E, th / 10.0, 0, true, NULL, NULL, cv::noArray(), mask, 0);
				availableRT = false;
			}
			else if (((cfg_pose.refineMethod & 0xF) != poselib::RefinePostAlg::PR_NO_REFINEMENT) && !cfg_pose.kneipInsteadBA)
			{
				cv::Mat R_tmp, t_tmp;
				if (availableRT)
				{
					R_kneip.copyTo(R_tmp);
					t_kneip.copyTo(t_tmp);

					if (poselib::refineEssentialLinear(points1newMat, points2newMat, E, mask, cfg_pose.refineMethod, nr_inliers_new, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
					{
						if (!R_tmp.empty() && !t_tmp.empty())
						{
							R_tmp.copyTo(R_kneip);
							t_tmp.copyTo(t_kneip);
						}
					}
					else
						std::cout << "Refinement failed!" << std::endl;
				}
				else if ((cfg_pose.refineMethod & 0xF) == poselib::RefinePostAlg::PR_KNEIP)
				{

					if (poselib::refineEssentialLinear(points1newMat, points2newMat, E, mask, cfg_pose.refineMethod, nr_inliers_new, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
					{
						if (!R_tmp.empty() && !t_tmp.empty())
						{
							R_tmp.copyTo(R_kneip);
							t_tmp.copyTo(t_kneip);
							availableRT = true;
						}
						else
							std::cout << "Refinement failed!" << std::endl;
					}
				}
				else
				{
					if (!poselib::refineEssentialLinear(points1newMat, points2newMat, E, mask, cfg_pose.refineMethod, nr_inliers_new, cv::noArray(), cv::noArray(), th, 4, 2.0, 0.1, 0.15))
						std::cout << "Refinement failed!" << std::endl;
				}
			}
			mask.copyTo(mask_E_new);

			if (!availableRT)
				poselib::getPoseTriangPts(E, points1newMat, points2newMat, R, t, Q, mask);
			else
			{
				R = R_kneip;
				t = t_kneip;
				if ((cfg_pose.BART > 0) && !cfg_pose.kneipInsteadBA)
					poselib::triangPts3D(R, t, points1newMat, points2newMat, Q, mask);
			}
		}

		if (cfg_pose.verbose > 3)
		{
			t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
			std::cout << "Time for pose estimation (includes possible linear refinement): " << t_mea << "ms" << endl;
			t_oa += t_mea;
		}
		if (cfg_pose.verbose > 4)
		{
			t_mea = (double)getTickCount(); //Start time measurement
		}

		//Bundle adjustment
		bool useBA = true;
		if (cfg_pose.kneipInsteadBA)
		{
			cv::Mat R_tmp, t_tmp;
			R.copyTo(R_tmp);
			t.copyTo(t_tmp);
			if (poselib::refineEssentialLinear(points1newMat, points2newMat, E, mask, cfg_pose.refineMethod, nr_inliers_new, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
			{
				if (!R_tmp.empty() && !t_tmp.empty())
				{
					R_tmp.copyTo(R);
					t_tmp.copyTo(t);
					useBA = false;
				}
			}
			else
			{
				std::cout << "Refinement using Kneips Eigen solver instead of bundle adjustment (BA) failed!" << std::endl;
				if (cfg_pose.BART > 0)
				{
					std::cout << "Trying bundle adjustment instead!" << std::endl;
					poselib::triangPts3D(R, t, points1newMat, points2newMat, Q, mask);
				}
			}
		}

		if (useBA)
		{
			if (cfg_pose.BART == 1)
			{
				poselib::refineStereoBA(points1newMat, points2newMat, R, t, Q, *cfg_pose.K0, *cfg_pose.K1, false, mask);
			}
			else if (cfg_pose.BART == 2)
			{
				cv::Mat points1newMat_tmp = points1newMat.clone();
				cv::Mat points2newMat_tmp = points2newMat.clone();
				poselib::CamToImgCoordTrans(points1newMat_tmp, *cfg_pose.K0);
				poselib::CamToImgCoordTrans(points2newMat_tmp, *cfg_pose.K1);
				poselib::refineStereoBA(points1newMat_tmp, points2newMat_tmp, R, t, Q, *cfg_pose.K0, *cfg_pose.K1, true, mask);
			}
			E = poselib::getEfromRT(R, t);
		}

		E.copyTo(E_new);
		R.copyTo(R_new);
		t.copyTo(t_new);
		mask.copyTo(mask_Q_new);

		if (cfg_pose.verbose > 4)
		{
			t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
			std::cout << "Time for bundle adjustment: " << t_mea << "ms" << endl;
			t_oa += t_mea;
		}
		if (cfg_pose.verbose > 5)
		{
			std::cout << "Overall pose estimation time: " << t_oa << "ms" << endl;

			std::cout << "Number of inliers after pose estimation and triangulation: " << cv::countNonZero(mask) << endl;
		}

		return 0;
	}

	int StereoRefine::refinePoseFromPool()
	{
		cv::Mat E = E_new.clone();
		cv::Mat mask = cv::Mat(1, points1Cam.rows, CV_8U, cv::Scalar((unsigned char)true));
		nr_inliers_new = (size_t)points1Cam.rows;
		Q.release();
		
		if (cfg_pose.verbose > 3)
		{
			t_mea = (double)getTickCount(); //Start time measurement
		}

		//Get R & t
		bool availableRT = false;
		if (!R_new.empty() && !t_new.empty())
		{
			double sumt = 0;
			for (int i = 0; i < 3; i++)
			{
				sumt += t_new.at<double>(i);
			}
			if (!poselib::nearZero(sumt) && poselib::isMatRoationMat(R_new))
			{
				availableRT = true;
			}
		}

		if (cfg_pose.refineRTold_CorrPool)
		{
			poselib::robustEssentialRefine(points1Cam, points2Cam, E, E, th / 10.0, 0, true, NULL, NULL, cv::noArray(), mask, 0);
			availableRT = false;
		}
		else if (((cfg_pose.refineMethod_CorrPool & 0xF) != poselib::RefinePostAlg::PR_NO_REFINEMENT) && !cfg_pose.kneipInsteadBA_CorrPool)
		{
			cv::Mat R_tmp, t_tmp;
			if (availableRT)
			{
				R_new.copyTo(R_tmp);
				t_new.copyTo(t_tmp);

				if (poselib::refineEssentialLinear(points1Cam, points2Cam, E, mask, cfg_pose.refineMethod_CorrPool, nr_inliers_new, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
				{
					if (!R_tmp.empty() && !t_tmp.empty())
					{
						R_tmp.copyTo(R_new);
						t_tmp.copyTo(t_new);
					}
				}
				else
				{
					std::cout << "Refinement failed!" << std::endl;
					return -1;
				}
			}
			else if ((cfg_pose.refineMethod_CorrPool & 0xF) == poselib::RefinePostAlg::PR_KNEIP)
			{

				if (poselib::refineEssentialLinear(points1Cam, points2Cam, E, mask, cfg_pose.refineMethod_CorrPool, nr_inliers_new, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
				{
					if (!R_tmp.empty() && !t_tmp.empty())
					{
						R_tmp.copyTo(R_new);
						t_tmp.copyTo(t_new);
						availableRT = true;
					}
					else
					{
						std::cout << "Refinement failed!" << std::endl;
						return -1;
					}
				}
			}
			else
			{
				if (!poselib::refineEssentialLinear(points1Cam, points2Cam, E, mask, cfg_pose.refineMethod_CorrPool, nr_inliers_new, cv::noArray(), cv::noArray(), th, 4, 2.0, 0.1, 0.15))
				{
					std::cout << "Refinement failed!" << std::endl;
					return -1;
				}
			}
		}
		mask.copyTo(mask_E_new);

		if (!availableRT)
			poselib::getPoseTriangPts(E, points1Cam, points2Cam, R_new, t_new, Q, mask);
		else
		{
			if ((cfg_pose.BART_CorrPool > 0) && !cfg_pose.kneipInsteadBA_CorrPool)
				poselib::triangPts3D(R_new, t_new, points1Cam, points2Cam, Q, mask);
		}

		if (cfg_pose.verbose > 3)
		{
			t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
			std::cout << "Time for pose estimation (includes possible linear refinement): " << t_mea << "ms" << endl;
			t_oa += t_mea;
		}
		if (cfg_pose.verbose > 4)
		{
			t_mea = (double)getTickCount(); //Start time measurement
		}

		//Bundle adjustment
		bool useBA = true;
		if (cfg_pose.kneipInsteadBA_CorrPool)
		{
			cv::Mat R_tmp, t_tmp;
			R_new.copyTo(R_tmp);
			t_new.copyTo(t_tmp);
			if (poselib::refineEssentialLinear(points1Cam, points2Cam, E, mask, cfg_pose.refineMethod_CorrPool, nr_inliers_new, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
			{
				if (!R_tmp.empty() && !t_tmp.empty())
				{
					R_tmp.copyTo(R_new);
					t_tmp.copyTo(t_new);
					useBA = false;
				}
			}
			else
			{
				std::cout << "Refinement using Kneips Eigen solver instead of bundle adjustment (BA) failed!" << std::endl;
				if (cfg_pose.BART_CorrPool > 0)
				{
					std::cout << "Trying bundle adjustment instead!" << std::endl;
					poselib::triangPts3D(R_new, t_new, points1Cam, points2Cam, Q, mask);
				}
			}
		}

		if (useBA)
		{
			if (cfg_pose.BART_CorrPool == 1)
			{
				poselib::refineStereoBA(points1Cam, points2Cam, R_new, t_new, Q, *cfg_pose.K0, *cfg_pose.K1, false, mask);
			}
			else if (cfg_pose.BART_CorrPool == 2)
			{
				cv::Mat points1Cam_tmp = points1Cam.clone();
				cv::Mat points2Cam_tmp = points2Cam.clone();
				poselib::CamToImgCoordTrans(points1Cam_tmp, *cfg_pose.K0);
				poselib::CamToImgCoordTrans(points2Cam_tmp, *cfg_pose.K1);
				poselib::refineStereoBA(points1Cam_tmp, points2Cam_tmp, R_new, t_new, Q, *cfg_pose.K0, *cfg_pose.K1, true, mask);
			}
			E = poselib::getEfromRT(R_new, t_new);
		}

		E.copyTo(E_new);
		mask.copyTo(mask_Q_new);

		if (cfg_pose.verbose > 4)
		{
			t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
			std::cout << "Time for bundle adjustment: " << t_mea << "ms" << endl;
			t_oa += t_mea;
		}
		if (cfg_pose.verbose > 5)
		{
			std::cout << "Overall pose estimation time: " << t_oa << "ms" << endl;

			std::cout << "Number of inliers after pose estimation and triangulation: " << cv::countNonZero(mask) << endl;
		}
		return 0;
	}

	size_t StereoRefine::getInliers(cv::Mat E, cv::Mat & p1, cv::Mat & p2, cv::Mat & mask, std::vector<double> & error)
	{
		error.clear();
		computeReprojError2(p1, p2, E, error);
		return getInlierMask(error, th2, mask);
	}

	int StereoRefine::filterNewCorrespondences(std::vector<cv::DMatch> & matches, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2, std::vector<double> error)
	{
		std::vector<CoordinatePropsNew> corrProbsNew;
		cv::Mat points1newMatnew = cv::Mat(nr_inliers_new, 2, points1newMat.type());
		cv::Mat points2newMatnew = cv::Mat(nr_inliers_new, 2, points2newMat.type());
		std::vector<cv::Point2f> points1newnew(nr_inliers_new);
		std::vector<cv::Point2f> points2newnew(nr_inliers_new);
		std::vector<cv::DMatch> matchesnew(nr_inliers_new);
		for (size_t i = 0, count = 0; i < nr_corrs_new; i++)
		{
			if (mask_E_new.at<bool>(i))
			{
				const int queryidx = matches[i].queryIdx;
				const int trainidx = matches[i].trainIdx;
				corrProbsNew.push_back(CoordinatePropsNew(kp1[queryidx].pt,
					kp2[trainidx].pt,
					matches[i].distance,
					kp1[queryidx].response,
					kp2[trainidx].response,
					error[i]));
				points1newMat.row(i).copyTo(points1newMatnew.row(count));
				points2newMat.row(i).copyTo(points2newMatnew.row(count));
				points1newnew[count] = points1new[i];
				points2newnew[count] = points2new[i];
				matchesnew[count] = matches[i];
				count++;
			}
		}
		points1newMatnew.copyTo(points1newMat);
		points2newMatnew.copyTo(points2newMat);
		points1new = points1newnew;
		points2new = points2newnew;
		matches = matchesnew;

		std::vector<size_t> delete_list_new;
		std::vector<size_t> delete_list_old;
		for (size_t i = 0; i < nr_inliers_new; i++)
		{
			std::vector<std::pair<size_t, float>> result;
			size_t nr_found = kdTreeLeft->radiusSearch(corrProbsNew[i].pt1, cfg_pose.minPtsDistance, result);
			if (nr_found)
			{
				bool deletionMarked = false;
				size_t j = 0;
				for (; j < nr_found; j++)
				{
					CoordinateProps corr_tmp = *correspondencePoolIdx[result[j].first];
					//Check, if the new point is equal to the nearest (< sqrt(2) pix difference)
					//If this is the case, only replace it (if it is better) and keep all in the surrounding
					if (result[j].second < 2.0)
					{
						cv::Point2f diff = corr_tmp.pt2 - corrProbsNew[i].pt2;
						double diff_dist = (double)diff.x * (double)diff.x + (double)diff.y * (double)diff.y;
						if (diff_dist < 2.0)
						{
							if (diff_dist < 0.01 && result[j].second < 0.01f)
							{
								delete_list_new.push_back(i);
								deletionMarked = true;
								correspondencePoolIdx[result[j].first]->nrFound++;
								break;
							}
							else
							{
								if (compareCorrespondences(corrProbsNew[i], corr_tmp)) //new correspondence is better
								{
									delete_list_old.push_back(result[j].first);
								}
								else //old correspondence is better
								{
									delete_list_new.push_back(i);
									deletionMarked = true;
									correspondencePoolIdx[result[j].first]->nrFound++;
									break;
								}
							}
						}
						// else -> keep both as the second coordinate is different
					}
					else
					{
						break;
					}
				}
				if (!deletionMarked && (j == 0))
				{
					//add the new correspondence only if it is the best within the old neighbors
					for (; j < nr_found; j++)
					{
						if (!compareCorrespondences(corrProbsNew[i], *correspondencePoolIdx[result[j].first])) //old correspondence is better
						{
							delete_list_new.push_back(i);
							break;
						}
					}
					if (j >= nr_found)
					{
						for (j = 0; j < nr_found; j++)
						{
							delete_list_old.push_back(result[j].first);
						}
					}
				}
			}
		}

		//Remove new data elements
		if (!delete_list_new.empty())
		{
			size_t n_del = delete_list_new.size();
			if (n_del == nr_inliers_new)
			{
				points1newMat.release();
				points2newMat.release();
				points1new.clear();
				points2new.clear();
				matches.clear();
				mask_E_new.release();
				mask_Q_new.release();
				nr_inliers_new = 0;
				nr_corrs_new = 0;
				Q.release();
			}
			else
			{
				size_t n_new = nr_inliers_new - n_del;
				points1newMatnew = cv::Mat(n_new, 2, points1newMat.type());
				points2newMatnew = cv::Mat(n_new, 2, points2newMat.type());
				points1newnew.clear();
				points2newnew.clear();
				matchesnew.clear();
				points1newnew.reserve(n_new);
				points2newnew.reserve(n_new);
				matchesnew.reserve(n_new);
				size_t old_idx = 0;
				int startRowNew = 0;
				for (size_t i = 0; i < n_del; i++)
				{
					if (old_idx == delete_list_new[i])
					{
						old_idx = delete_list_new[i] + 1;
						continue;
					}
					const int nr_new_cpy_elements = (int)delete_list_new[i] - (int)old_idx;
					const int endRowNew = startRowNew + nr_new_cpy_elements;
					points1newMat.rowRange((int)old_idx, (int)delete_list_new[i]).copyTo(points1newMatnew.rowRange(startRowNew, endRowNew));
					points2newMat.rowRange((int)old_idx, (int)delete_list_new[i]).copyTo(points2newMatnew.rowRange(startRowNew, endRowNew));
					points1newnew.insert(points1newnew.end(), points1new.begin() + old_idx, points1new.begin() + delete_list_new[i]);
					points2newnew.insert(points2newnew.end(), points2new.begin() + old_idx, points2new.begin() + delete_list_new[i]);
					matchesnew.insert(matchesnew.end(), matches.begin() + old_idx, matches.begin() + delete_list_new[i]);
					startRowNew = endRowNew;
					old_idx = delete_list_new[i] + 1;
				}
				points1newMat.rowRange((int)old_idx, points1newMat.rows).copyTo(points1newMatnew.rowRange(startRowNew, n_new));
				points1newnew.insert(points1newnew.end(), points1new.begin() + old_idx, points1new.end());
				points2newnew.insert(points2newnew.end(), points2new.begin() + old_idx, points2new.end());
				matchesnew.insert(matchesnew.end(), matches.begin() + old_idx, matches.end());
				points1newMatnew.copyTo(points1newMat);
				points2newMatnew.copyTo(points2newMat);
				points1new = points1newnew;
				points2new = points2newnew;
				matches = matchesnew;
				mask_E_new = cv::Mat(1, n_new, CV_8UC1, cv::Scalar((unsigned char)true));
				mask_Q_new.release();
				nr_inliers_new = n_new;
				nr_corrs_new = n_new;
				Q.release();
			}
		}

		//Remove old correspondences
		if (!delete_list_old.empty())
		{
			sort(delete_list_old.begin(), delete_list_old.end());
			for (size_t i = delete_list_old.size() - 1; i >= 1; i--)
			{
				if (delete_list_old[i] == delete_list_old[i - 1])
				{
					delete_list_old.erase(delete_list_old.begin() + i);
				}
			}
			if (poolCorrespondenceDelete(delete_list_old))
				return -1;
		}
		return 0;
	}

	int StereoRefine::poolCorrespondenceDelete(std::vector<size_t> delete_list)
	{
		//kdTreeLeft, points1Cam, points2Cam, correspondencePool, correspondencePoolIdx
		size_t nrToDel = delete_list.size();
		size_t poolSize = correspondencePool.size();
		if (nrToDel == poolSize)
		{
			kdTreeLeft->killTree();
			kdTreeLeft.release();
			points1Cam.release();
			points2Cam.release();
			correspondencePool.clear();
			correspondencePoolIdx.clear();
			corrIdx = 0;
			/*mask_E_old.release();
			mask_Q_old.release();*/
		}
		else
		{
			vector<size_t> ptsCamDelIdxs(nrToDel);
			for (size_t i = 0; i < nrToDel; i++)
			{
				ptsCamDelIdxs[i] = correspondencePoolIdx[delete_list[i]]->ptIdx;
			}
			std::sort(ptsCamDelIdxs.begin(), ptsCamDelIdxs.end());

			std::unordered_map<size_t, long long int> delDiffInfo;
			for (size_t i = 0, count = 0; i < poolSize; i++)
			{
				if ((i < ptsCamDelIdxs[count]) || (i > ptsCamDelIdxs[count]))
				{
					delDiffInfo.insert({ i, -1 * (long long int)count });
				}
				else if ((i == ptsCamDelIdxs[count]) && (count < (nrToDel - 1)))
				{
					count++;
				}
			}

			size_t n_new = poolSize - nrToDel;
			cv::Mat points1CamNew(n_new, 2, points1Cam.type());
			cv::Mat points2CamNew(n_new, 2, points2Cam.type());
			size_t old_idx = 0;
			int startRowNew = 0;
			for (size_t i = 0; i < nrToDel; i++)
			{
				if (old_idx == ptsCamDelIdxs[i])
				{
					old_idx = ptsCamDelIdxs[i] + 1;
					continue;
				}
				const int nr_new_cpy_elements = (int)ptsCamDelIdxs[i] - (int)old_idx;
				const int endRowNew = startRowNew + nr_new_cpy_elements;
				points1Cam.rowRange((int)old_idx, (int)ptsCamDelIdxs[i]).copyTo(points1CamNew.rowRange(startRowNew, endRowNew));
				points2Cam.rowRange((int)old_idx, (int)ptsCamDelIdxs[i]).copyTo(points2CamNew.rowRange(startRowNew, endRowNew));
				
				startRowNew = endRowNew;
				old_idx = ptsCamDelIdxs[i] + 1;
			}
			points1Cam.rowRange((int)old_idx, points1Cam.rows).copyTo(points1CamNew.rowRange(startRowNew, n_new));
			points2Cam.rowRange((int)old_idx, points1Cam.rows).copyTo(points2CamNew.rowRange(startRowNew, n_new));
			points1CamNew.copyTo(points1Cam);
			points2CamNew.copyTo(points2Cam);

			for (size_t i = 0; i < nrToDel; i++)
			{
				//Delete index elements from tree
				kdTreeLeft->removeElements(delete_list[i]);

				//Delete correspondences from pool
				std::unordered_map<size_t, std::list<CoordinateProps>::iterator>::iterator it = correspondencePoolIdx.find(delete_list[i]);
				try
				{
					if (it->second != correspondencePool.end())
						correspondencePool.erase(it->second);
					else
						throw "Invalid pool iterator";
					it->second = correspondencePool.end();
				}
				catch (string e)
				{
					cout << "Exception: " << e << endl;
					cout << "Clearing the whole correspondence pool!" << endl;
					return -1;
				}
			}
			for (std::list<CoordinateProps>::iterator it = correspondencePool.begin(); it != correspondencePool.end(); ++it)
			{
				it->ptIdx = it->ptIdx + delDiffInfo[it->ptIdx];
			}
		}
		return 0;
	}

	bool StereoRefine::compareCorrespondences(CoordinatePropsNew &newCorr, CoordinateProps &oldCorr)
	{
		double weight_error[2], weight_descrDist[2], weight_response[2];
		const double weighting_terms[3] = {0.3, 0.5, 0.2};//Weights for weight_error, weight_descrDist, weight_response
		double overall_weight[2];
		int idx;
		double rel_diff;
		const double weight_th = 0.2;//If one of the resulting weights is more than e.g. 20% better
		size_t max_age = 15;//If the quality of the old correspondence is slightly better but it is older (number of iterations) than this thershold, the new new one is preferred

		weight_error[0] = getWeightingValuesInv(newCorr.sampsonError, th2);
		weight_error[1] = getWeightingValuesInv(oldCorr.SampsonErrors.back(), th2);
		weight_descrDist[0] = getWeightingValuesInv((double)newCorr.descrDist, (double)descrDist_max);
		weight_descrDist[1] = getWeightingValuesInv((double)oldCorr.descrDist, (double)descrDist_max);
		weight_response[0] = (getWeightingValues((double)newCorr.keyPResponses[0], (double)keyPRespons_max) + getWeightingValues((double)newCorr.keyPResponses[1], (double)keyPRespons_max)) / 2.0;
		weight_response[1] = (getWeightingValues((double)oldCorr.keyPResponses[0], (double)keyPRespons_max) + getWeightingValues((double)oldCorr.keyPResponses[1], (double)keyPRespons_max)) / 2.0;
		overall_weight[0] = weighting_terms[0] * weight_error[0] + weighting_terms[1] * weight_descrDist[0] + weighting_terms[2] * weight_response[0];
		overall_weight[1] = weighting_terms[0] * weight_error[1] + weighting_terms[1] * weight_descrDist[1] + weighting_terms[2] * weight_response[1];
		idx = overall_weight[0] > overall_weight[1] ? 0 : 1;
		if (idx)
		{
			rel_diff = (overall_weight[1] - overall_weight[0]) / overall_weight[1];
			if ((rel_diff < 0.05) || (rel_diff > weight_th))
			{
				return false; //take the old correspondence
			}
		}
		else
		{
			rel_diff = (overall_weight[0] - overall_weight[1]) / overall_weight[0];
			if (rel_diff < 0.05)
			{
				return false; //take the old correspondence
			}
			else if ((rel_diff > weight_th))
			{
				return true; //take the new correspondence
			}
		}
		if (oldCorr.age > max_age)
		{
			return true; //take the new correspondence
		}
		else if ((oldCorr.SampsonErrors.size() > 1) && (oldCorr.SampsonErrors.back() > oldCorr.SampsonErrors[oldCorr.SampsonErrors.size() - 2]))//Check if the error is increasing
		{
			return true; //take the new correspondence
		}

		return false; //take the old correspondence
	}

	inline double getWeightingValuesInv(double value, double max_value, double min_value)
	{
		return 1.0 - value / (max_value - min_value);
	}

	inline double getWeightingValues(double value, double max_value, double min_value)
	{
		return value / (max_value - min_value);
	}

}