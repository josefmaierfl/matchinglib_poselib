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

	double getWeightingValues(double value, double max_value, double min_value = 0);

	/* --------------------- Functions --------------------- */

	void StereoRefine::setNewParameters(ConfigPoseEstimation cfg_pose_)
	{
		if (!poselib::nearZero(cfg_pose.th_pix_user - cfg_pose_.th_pix_user))
		{
			//use robust method to estimate E first to check if the new correspondences correspond to the old E
		}
		cfg_pose = cfg_pose_;
		th = cfg_pose_.th_pix_user * pixToCamFact; //Inlier threshold
		th2 = th * th;
		checkInlRatThresholds();
		t_mea = 0;
		t_oa = 0;
	}

	void StereoRefine::checkInlRatThresholds()
	{
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
			if (!robustPoseEstimation())
			{
				return -1;
			}
			inlier_ratio_new1 = (double)nr_inliers_new / (double)nr_corrs_new;
			if (inlier_ratio_new1 < cfg_pose.minStartAggInlRat)
			{
				cout << "Inlier ratio too small! Skipping aggregation of correspondences! The output pose of this and the next iteration will be like in the mono camera case!" << endl;
				return 0;
			}
			addCorrespondencesToPool(matches, kp1, kp2);
			pose_history.push_back(poseHist(E_new.clone(), R_new.clone(), t_new.clone()));
			inlier_ratio_history.push_back(inlier_ratio_new1);
			nrEstimation++;
		}
		else
		{
			cv::Mat mask;
			std::vector<double> errorNew;
			bool addToPool = false;
			unsigned int nr_inliers_tmp = 0;
			double inlier_ratio_new;
			nr_inliers_tmp = getInliers(E_new, mask, errorNew);
			inlier_ratio_new = (double)nr_inliers_tmp / (double)nr_corrs_new;
			//check if the new inlier ratio is approximately equal to the old one as it is not likely that the image content changes completely from image pair to image pair
			if (inlier_ratio_new < ((1.0 - cfg_pose.relInlRatThLast) * inlier_ratio_history.back()))
			{
				//Perform new robust estimation to check if the pose has changed
				if (!robustPoseEstimation())
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
						addCorrespondencesToPool(matches, kp1, kp2);
						pose_history.push_back(poseHist(E_new.clone(), R_new.clone(), t_new.clone()));
						inlier_ratio_history.push_back(inlier_ratio_new1);
						nrEstimation++;
						cout << "The pose has changed! System is reinitialized!" << endl;
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
			}

			if (addToPool)
			{
				//Perform filtering of correspondences and refinement on all available correspondences
				skipCount = 0;
				filterNewCorrespondences(matches, kp1, kp2, errorNew);
			}

			if (skipCount > cfg_pose.maxSkipPairs)
			{
				//Reinitialize whole system
				clearHistoryAndPool();
				addCorrespondencesToPool(matches, kp1, kp2);
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
		nrEstimation = 0;
		skipCount = 0;
		mask_E_old.release();
		mask_Q_old.release();
		deletionIdxs.clear();
		pose_history.clear();
		inlier_ratio_history.clear();
		errorStatistic_history.clear();
	}

	void StereoRefine::addCorrespondencesToPool(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2)
	{
		CoordinateProps tmp;
		std::vector<double> errors;
		unsigned int nrEntries = 0;
		if (!points1Cam.empty())
		{
			nrEntries = (unsigned int)points1Cam.rows;
			unsigned int reservesize = nrEntries + nr_inliers_new;
			points1Cam.reserve(reservesize);
			points2Cam.reserve(reservesize);
			correspondencePoolIdx.reserve(reservesize);
		}
		
		for (unsigned int i = 0, count = nrEntries; i < nr_corrs_new; i++)
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
				tmp.meanSampsonError = getSampsonL2Error(E_new, points1newMat.row(i), points2newMat.row(i));
				tmp.SampsonErrors.push_back(tmp.meanSampsonError);
				errors.push_back(tmp.meanSampsonError);
				tmp.nrFound = 1;
				tmp.pt1 = kp1[matches[i].queryIdx].pt;
				tmp.pt2 = kp2[matches[i].trainIdx].pt;
				tmp.ptIdx = count;
				if (!Q.empty())
				{
					tmp.Q = cv::Point3d(Q.row(i));
					tmp.Q_tooFar = !mask_Q_new.at<bool>(i);
				}
				tmp.poolIdx = corrIdx;
				correspondencePool.push_back(tmp);
				correspondencePoolIdx.insert({ corrIdx, --correspondencePool.end() });
				corrIdx++;
				count++;
			}
		}
		statVals err_stats;
		getStatsfromVec(errors, &err_stats);
		errorStatistic_history.push_back(err_stats);

		//Initialize KD tree
		if (!kdTreeLeft)
		{
			kdTreeLeft.reset(new keyPointTree(&correspondencePool, &correspondencePoolIdx));
			kdTreeLeft->buildInitialTree();
		}
		else if((double)correspondencePool.size() / (double)corrIdx < 0.5)//Check if not too many items were detelted from the KD tree index
		{
			corrIdx = 0;
			correspondencePoolIdx.clear();
			std::list<CoordinateProps>::iterator it = correspondencePool.begin();
			for (size_t i = 0; i < correspondencePool.size(); i++)
			{
				it->poolIdx = corrIdx;
				correspondencePoolIdx.insert({ corrIdx++, it++ });
			}
			kdTreeLeft->resetTree(&correspondencePool, &correspondencePoolIdx);
		}

		//add coordinates to the KD tree
		kdTreeLeft->addElements(corrIdx - nr_inliers_new, nr_inliers_new);

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
				poselib::CamToImgCoordTrans(points1newMat, *cfg_pose.K0);
				poselib::CamToImgCoordTrans(points2newMat, *cfg_pose.K1);
				poselib::refineStereoBA(points1newMat, points2newMat, R, t, Q, *cfg_pose.K0, *cfg_pose.K1, true, mask);
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

	unsigned int StereoRefine::getInliers(cv::Mat E, cv::Mat & mask, std::vector<double> & error)
	{
		error.clear();
		computeReprojError2(points1newMat, points2newMat, E, error);
		return getInlierMask(error, th2, mask);
	}

	void StereoRefine::filterNewCorrespondences(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2, std::vector<double> error)
	{
		std::vector<CoordinatePropsNew> corrProbsNew;

		for (unsigned int i = 0; i < nr_corrs_new; i++)
		{
			if (mask_E_new.at<bool>(i))
			{
				corrProbsNew.push_back(CoordinatePropsNew(kp1[matches[i].queryIdx].pt,
					kp2[matches[i].trainIdx].pt,
					matches[i].distance,
					kp1[matches[i].queryIdx].response,
					kp2[matches[i].trainIdx].response,
					error[i]));
			}
		}

		std::vector<size_t> delete_list_new;
		std::vector<size_t> delete_list_old;
		for (unsigned int i = 0; i < nr_inliers_new; i++)
		{
			std::vector<std::pair<size_t, float>> result;
			size_t nr_found = kdTreeLeft->radiusSearch(corrProbsNew[i].pt1, cfg_pose.minPtsDistance, result);
			if (nr_found)
			{
				CoordinateProps corr_tmp = *correspondencePoolIdx[result[0].first];
				//Check, if the new point is equal to the nearest (< 1 pix difference)
				if (result[0].second < 1.0)
				{
					cv::Point2f diff = corr_tmp.pt2 - corrProbsNew[i].pt2;
					double diff_dist = diff.x * diff.x + diff.y * diff.y;
					if (diff_dist < 1.0)
					{
						if (diff_dist < 0.01 && result[0].second < 0.01)
						{
							delete_list_new.push_back(i);
						}
						else
						{
							if (compareCorrespondences(corrProbsNew[i], corr_tmp)) //new correspondence is better
							{

							}
							else //old correspondence is better
							{

							}
						}
					}
					else
					{

					}
				}
				else
				{

				}
			}
		}

	}

	bool StereoRefine::compareCorrespondences(CoordinatePropsNew &newCorr, CoordinateProps &oldCorr)
	{
		double weight_error[2], weight_descrDist[2], weight_response[2];
		const double weighting_terms[3] = {0.3, 0.5, 0.2};//Weights for weight_error, weight_descrDist, weight_response
		double overall_weight[2];
		int idx;
		double rel_diff;
		const double weight_th = 0.2;//If one of the resulting weights is more than e.g. 20% better
		unsigned int max_age = 15;//If the quality of the old correspondence is slightly better but it is older (number of iterations) than this thershold, new new one is preferred

		weight_error[0] = getWeightingValues(newCorr.sampsonError, th2);
		weight_error[1] = getWeightingValues(oldCorr.SampsonErrors.back(), th2);
		weight_descrDist[0] = getWeightingValues((double)newCorr.descrDist, (double)descrDist_max);
		weight_descrDist[1] = getWeightingValues((double)oldCorr.descrDist, (double)descrDist_max);
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
			else if (oldCorr.age > max_age)
			{
				return true; //take the new correspondence
			}
			else
			{
				//Check if the error is increasing
			}
		}
		else
		{
			rel_diff = (overall_weight[0] - overall_weight[1]) / overall_weight[0];
		}
	}

	inline double getWeightingValues(double value, double max_value, double min_value)
	{
		return 1.0 - value / (max_value - min_value);
	}

}