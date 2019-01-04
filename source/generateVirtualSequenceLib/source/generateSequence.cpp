/**********************************************************************************************************
FILE: generateSequence.cpp

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: March 2018

LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for generating stereo sequences with correspondences given
a view restrictions like depth ranges, moving objects, ...
**********************************************************************************************************/

#include "generateSequence.h"
#include "opencv2/imgproc/imgproc.hpp"
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
#include <array>

#include "pcl/filters/frustum_culling.h"
#include "pcl/common/transforms.h"
#include "pcl/filters/voxel_grid_occlusion_estimation.h"

#include <boost/thread/thread.hpp>
//#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>


using namespace std;
using namespace cv;

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

/* -------------------------- Functions -------------------------- */

genStereoSequ::genStereoSequ(cv::Size imgSize_, cv::Mat K1_, cv::Mat K2_, std::vector<cv::Mat> R_, std::vector<cv::Mat> t_, StereoSequParameters & pars_) :
	imgSize(imgSize_), K1(K1_), K2(K2_), R(R_), t(t_), pars(pars_)
{
	CV_Assert((K1.rows == 3) && (K2.rows == 3) && (K1.cols == 3) && (K2.cols == 3) && (K1.type() == CV_64FC1) && (K2.type() == CV_64FC1));
	CV_Assert((imgSize.area() > 0) && (R.size() == t.size()) && (R.size() > 0));

	randSeed(rand_gen);

	//Generate a mask with the minimal distance between keypoints and a mask for marking used areas in the first stereo image
	genMasks();

	//Calculate inverse of camera matrices
	K1i = K1.inv();
	K2i = K2.inv();

	//Number of stereo configurations
	nrStereoConfs = R.size();

	//Construct the camera path
	constructCamPath();

	//Calculate the thresholds for the depths near, mid, and far for every camera configuration
	if(!getDepthRanges())
    {
	    throw SequenceException("Depth ranges are negative!\n");
    }

	//Used inlier ratios
	genInlierRatios();

	//Number of correspondences per image and Correspondences per image regions
	initNrCorrespondences();

	//Depths per image region
	adaptDepthsPerRegion();

	//Check if the given ranges of connected depth areas per image region are correct and initialize them for every definition of depths per image region
	checkDepthAreas();

	//Calculate the area in pixels for every depth and region
	calcPixAreaPerDepth();

	//Initialize region ROIs and masks
	genRegMasks();

	//Reset variables for the moving objects
	combMovObjLabelsAll = Mat::zeros(imgSize, CV_8UC1);
	movObjMask2All = Mat::zeros(imgSize, CV_8UC1);
	movObjMaskFromLast = Mat::zeros(imgSize, CV_8UC1);
	movObjMaskFromLast2 = Mat::zeros(imgSize, CV_8UC1);
	movObjHasArea = std::vector<std::vector<bool>>(3, std::vector<bool>(3, false));
	actCorrsOnMovObj = 0;
	actCorrsOnMovObjFromLast = 0;

	//Calculate the initial number, size, and positions of moving objects in the image
	getNrSizePosMovObj();

	//Get the relative movement direction (compared to the camera movement) for every moving object
	checkMovObjDirection();
}

void genStereoSequ::genMasks()
{
	//Generate a mask with the minimal distance between keypoints
	int sqrSi = 2 * max((int)ceil(pars.minKeypDist), 1) + 1;
	csurr = Mat::ones(sqrSi, sqrSi, CV_8UC1);

	//Generate a mask for marking used areas in the first stereo image
	corrsIMG = Mat::zeros(imgSize.width + sqrSi - 1, imgSize.height + sqrSi - 1, CV_8UC1);
}

//Get number of correspondences per image and Correspondences per image regions
//Check if there are too many correspondences per region as every correspondence needs a minimum distance to its neighbor. If yes, the minimum distance and/or number of correspondences are adapted.
void genStereoSequ::initNrCorrespondences()
{
	//Number of correspondences per image
	genNrCorrsImg();

	//Correspondences per image regions
	bool res = initFracCorrImgReg();
	while (!res)
	{
		genNrCorrsImg();
		res = initFracCorrImgReg();
	}
}

//Initialize fraction of correspondences per image region and calculate the absolute number of TP/TN correspondences per image region
bool genStereoSequ::initFracCorrImgReg()
{
	if ((pars.corrsPerRegRepRate == 0) && pars.corrsPerRegion.empty())
	{
		for (size_t i = 0; i < totalNrFrames; i++)
		{
			Mat newCorrsPerRegion(3, 3, CV_64FC1);
			cv::randu(newCorrsPerRegion, Scalar(0), Scalar(1.0));
			newCorrsPerRegion /= sum(newCorrsPerRegion)[0];
			pars.corrsPerRegion.push_back(newCorrsPerRegion.clone());
		}
		pars.corrsPerRegRepRate = 1;
	}
	else if (pars.corrsPerRegRepRate == 0)
	{
		pars.corrsPerRegRepRate = totalNrFrames / pars.corrsPerRegion.size();
	}
	else if (pars.corrsPerRegion.empty())
	{
		//Randomly initialize the fractions
		size_t nrMats = totalNrFrames / pars.corrsPerRegRepRate;
		for (size_t i = 0; i < nrMats; i++)
		{
			Mat newCorrsPerRegion(3, 3, CV_64FC1);
			cv::randu(newCorrsPerRegion, Scalar(0), Scalar(1.0));
			newCorrsPerRegion /= sum(newCorrsPerRegion)[0];
			pars.corrsPerRegion.push_back(newCorrsPerRegion.clone());
		}
	}

	for (size_t k = 0; k < pars.corrsPerRegion.size(); k++)
	{
		double regSum = sum(pars.corrsPerRegion[k])[0];
		if (!nearZero(regSum) && !nearZero(regSum - 1.0))
			pars.corrsPerRegion[k] /= regSum;
		else if (nearZero(regSum))
		{
			pars.corrsPerRegion[k] = Mat::ones(3, 3, CV_64FC1) / 9.0;
		}
	}

	//Generate absolute number of correspondences per image region and frame
	nrTruePosRegs.reserve(totalNrFrames);
	nrCorrsRegs.reserve(totalNrFrames);
	nrTrueNegRegs.reserve(totalNrFrames);
	size_t cnt = 0;
	for (size_t i = 0; i < totalNrFrames; i++)
	{		
		//Get number of correspondences per region
		Mat newCorrsPerRegion;
		newCorrsPerRegion = pars.corrsPerRegion[cnt] * nrCorrs[i];
		newCorrsPerRegion.convertTo(newCorrsPerRegion, CV_32SC1, 1.0, 0.5);//Corresponds to round
		int32_t chkSize = sum(newCorrsPerRegion)[0] - (int32_t)nrCorrs[i];
		if (chkSize > 0)
		{
			do
			{
				int pos = std::rand() % 9;
				if (newCorrsPerRegion.at<int32_t>(pos) > 0)
				{
					newCorrsPerRegion.at<int32_t>(pos)--;
					chkSize--;
				} /*else
                {
				    cout << "Zero corrs in region " << pos << "of frame " << i << endl;
                }*/
			} while (chkSize > 0);
		}
		else if (chkSize < 0)
		{
			do
			{
				int pos = std::rand() % 9;
				if (!nearZero(pars.corrsPerRegion[cnt].at<double>(pos)))
				{
					newCorrsPerRegion.at<int32_t>(pos)++;
					chkSize++;
				}
			} while (chkSize < 0);
		}

		//Check if there are too many correspondences per region as every correspondence needs a minimum distance to its neighbor
		double minCorr, maxCorr;
		cv::minMaxLoc(newCorrsPerRegion, &minCorr, &maxCorr);
		double regA = (double)imgSize.area() / 9.0;
		double areaCorrs = maxCorr * pars.minKeypDist * pars.minKeypDist * 1.33;//Multiply by 1.33 to take gaps into account that are a result of randomness
		
		if (areaCorrs > regA)
		{
			cout << "There are too many keypoints per region when demanding a minimum keypoint distance of " << pars.minKeypDist << ". Changing it!" << endl;
			double mKPdist = floor(10.0 * sqrt(regA / (1.33 * maxCorr))) / 10.0;
			if (mKPdist <= 1.414214)
			{
				cout << "Changed the minimum keypoint distance to 1.0. There are still too many keypoints. Changing the number of keypoints!" << endl;
				pars.minKeypDist = 1.0;
				genMasks();
				//Get max # of correspondences
				double maxFC = (double)*std::max_element(nrCorrs.begin(), nrCorrs.end());
				//Get the largest portion of correspondences within a single region
				vector<double> cMaxV(pars.corrsPerRegion.size());
				for (size_t k = 0; k < pars.corrsPerRegion.size(); k++)
				{
					cv::minMaxLoc(pars.corrsPerRegion[k], &minCorr, &maxCorr);
					cMaxV[k] = maxCorr;
				}
				maxCorr = *std::max_element(cMaxV.begin(), cMaxV.end());
				maxCorr *= maxFC;
				//# KPs reduction factor
				double reduF = regA / (2.0 * maxCorr);
				//Get worst inlier ratio
				double minILR = *std::min_element(inlRat.begin(), inlRat.end());
				//Calc max true positives
				size_t maxTPNew = (size_t)floor(maxCorr * reduF * minILR);
				cout << "Changing max. true positives to " << maxTPNew << endl;;
				if ((pars.truePosRange.second - pars.truePosRange.first) == 0)
				{
					pars.truePosRange.first = pars.truePosRange.second = maxTPNew;
				}
				else
				{
					if (pars.truePosRange.first >= maxTPNew)
					{
						pars.truePosRange.first = maxTPNew / 2;
						pars.truePosRange.second = maxTPNew;
					}
					else
					{
						pars.truePosRange.second = maxTPNew;
					}
				}
				nrTruePosRegs.clear();
				nrCorrsRegs.clear();
				nrTrueNegRegs.clear();
				return false;
			}
			else
			{
				cout << "Changed the minimum keypoint distance to " << mKPdist << endl;
				pars.minKeypDist = mKPdist;
				genMasks();
			}
		}

		nrCorrsRegs.push_back(newCorrsPerRegion.clone());

		//Get number of true negatives per region
		Mat negsReg(3, 3, CV_64FC1);
		cv::randu(negsReg, Scalar(0), Scalar(1.0));
		negsReg /= sum(negsReg)[0];
		Mat newCorrsPerRegiond;
		newCorrsPerRegion.convertTo(newCorrsPerRegiond, CV_64FC1);
		negsReg = negsReg.mul(newCorrsPerRegiond);//Max number of true negatives per region
		negsReg *= (double)nrTrueNeg[i] / sum(negsReg)[0];
		negsReg.convertTo(negsReg, CV_32SC1, 1.0, 0.5);//Corresponds to round
		for (size_t j = 0; j < 9; j++)
		{
			while (negsReg.at<int32_t>(j) > newCorrsPerRegion.at<int32_t>(j))
				negsReg.at<int32_t>(j)--;
		}
		chkSize = sum(negsReg)[0] - (int32_t)nrTrueNeg[i];
		if (chkSize > 0)
		{
			do
			{
				int pos = std::rand() % 9;
				if (negsReg.at<int32_t>(pos) > 0)
				{
					negsReg.at<int32_t>(pos)--;
					chkSize--;
				} /*else
                {
                    cout << "Zero neg corrs in region " << pos << "of frame " << i << endl;
                }*/
			} while (chkSize > 0);
		}
		else if (chkSize < 0)
		{
			do
			{
				int pos = std::rand() % 9;
				if (negsReg.at<int32_t>(pos) < newCorrsPerRegion.at<int32_t>(pos))
				{
					negsReg.at<int32_t>(pos)++;
					chkSize++;
				}
			} while (chkSize < 0);
		}
		nrTrueNegRegs.push_back(negsReg.clone());

		//Get number of true positives per region
		newCorrsPerRegion = newCorrsPerRegion - negsReg;
		nrTruePosRegs.push_back(newCorrsPerRegion.clone());

		//Check if the fraction of corrspondences per region must be changend
		if ((((i + 1) % (pars.corrsPerRegRepRate)) == 0))
		{
			cnt++;
			if (cnt >= pars.corrsPerRegion.size())
			{
				cnt = 0;
			}
		}
	}

	return true;
}

//Generate number of correspondences
void genStereoSequ::genNrCorrsImg()
{
	nrCorrs.resize(totalNrFrames);
	nrTrueNeg.resize(totalNrFrames);
	if ((pars.truePosRange.second - pars.truePosRange.first) == 0)
	{
	    if(pars.truePosRange.first == 0)
        {
	        throw SequenceException("Number of true positives specified 0 for all frames - nothing can be generated!\n");
        }
		nrTruePos.resize(totalNrFrames, pars.truePosRange.first);
		for (size_t i = 0; i < totalNrFrames; i++)
		{
			nrCorrs[i] = (size_t)round((double)pars.truePosRange.first / inlRat[i]);
			nrTrueNeg[i] = nrCorrs[i] - pars.truePosRange.first;
		}
		if (nearZero(pars.inlRatRange.first - pars.inlRatRange.second))
		{
			fixedNrCorrs = true;
		}
	}
	else
	{
		size_t initTruePos = std::max((size_t)round(getRandDoubleValRng((double)pars.truePosRange.first, (double)pars.truePosRange.second)), (size_t)1);
		if (nearZero(pars.truePosChanges))
		{
			nrTruePos.resize(totalNrFrames, initTruePos);
			for (size_t i = 0; i < totalNrFrames; i++)
			{
				nrCorrs[i] = (size_t)round((double)initTruePos / inlRat[i]);
				nrTrueNeg[i] = nrCorrs[i] - initTruePos;
			}
		}
		else if (nearZero(pars.truePosChanges - 100.0))
		{
			nrTruePos.resize(totalNrFrames);
			std::uniform_int_distribution<size_t> distribution(pars.truePosRange.first, pars.truePosRange.second);
			for (size_t i = 0; i < totalNrFrames; i++)
			{
				nrTruePos[i] = distribution(rand_gen);
				nrCorrs[i] = (size_t)round((double)nrTruePos[i] / inlRat[i]);
				nrTrueNeg[i] = nrCorrs[i] - nrTruePos[i];
			}
		}
		else
		{
			nrTruePos.resize(totalNrFrames);
			nrTruePos[0] = initTruePos;
			nrCorrs[0] = (size_t)round((double)nrTruePos[0] / inlRat[0]);
			nrTrueNeg[0] = nrCorrs[0] - nrTruePos[0];
			for (size_t i = 1; i < totalNrFrames; i++)
			{
				size_t rangeVal = (size_t)round(pars.truePosChanges * (double)nrTruePos[i - 1]);
				size_t maxTruePos = nrTruePos[i - 1] + rangeVal;
				maxTruePos = maxTruePos > pars.truePosRange.second ? pars.truePosRange.second : maxTruePos;
                int64_t minTruePos_ = (int64_t)nrTruePos[i - 1] - (int64_t)rangeVal;
                size_t minTruePos = minTruePos_ < 0 ? 0:((size_t)minTruePos_);
				minTruePos = minTruePos < pars.truePosRange.first ? pars.truePosRange.first : minTruePos;
				std::uniform_int_distribution<size_t> distribution(minTruePos, maxTruePos);
				nrTruePos[i] = distribution(rand_gen);
				nrCorrs[i] = (size_t)round((double)nrTruePos[i] / inlRat[i]);
				nrTrueNeg[i] = nrCorrs[i] - nrTruePos[i];
			}
		}
	}
}

//Generate the inlier ratio for every frame
void genStereoSequ::genInlierRatios()
{
	if (nearZero(pars.inlRatRange.first - pars.inlRatRange.second))
	{
		inlRat.resize(totalNrFrames, max(pars.inlRatRange.first, 0.01));
	}
	else
	{
		double initInlRat = getRandDoubleValRng(pars.inlRatRange.first, pars.inlRatRange.second, rand_gen);
		initInlRat = max(initInlRat, 0.01);
		if (nearZero(pars.inlRatChanges))
		{
			inlRat.resize(totalNrFrames, initInlRat);
		}
		else if (nearZero(pars.inlRatChanges - 100.0))
		{
			inlRat.resize(totalNrFrames);
			std::uniform_real_distribution<double> distribution(pars.inlRatRange.first, pars.inlRatRange.second);
			for (size_t i = 0; i < totalNrFrames; i++)
			{
				inlRat[i] = max(distribution(rand_gen), 0.01);
			}
		}
		else
		{
			inlRat.resize(totalNrFrames);
			inlRat[0] = initInlRat;
			for (size_t i = 1; i < totalNrFrames; i++)
			{
				double maxInlrat = inlRat[i - 1] + pars.inlRatChanges * inlRat[i - 1];
				maxInlrat = maxInlrat > pars.inlRatRange.second ? pars.inlRatRange.second : maxInlrat;
				double minInlrat = inlRat[i - 1] - pars.inlRatChanges * inlRat[i - 1];
				minInlrat = minInlrat < pars.inlRatRange.first ? pars.inlRatRange.first : minInlrat;
				inlRat[i] = max(getRandDoubleValRng(minInlrat, maxInlrat), 0.01);
			}
		}
	}
}

/* Constructs an absolute camera path including the position and rotation of the stereo rig (left/lower camera centre)
*/
void genStereoSequ::constructCamPath()
{
	//Calculate the absolute velocity of the cameras
	absCamVelocity = 0;
	for (size_t i = 0; i < t.size(); i++)
	{
		absCamVelocity += norm(t[i]);
	}
	absCamVelocity /= (double)t.size();
	absCamVelocity *= pars.relCamVelocity;//in baselines from frame to frame

	//Calculate total number of frames
	totalNrFrames = pars.nFramesPerCamConf * t.size();

	//Number of track elements
	size_t nrTracks = pars.camTrack.size();

	absCamCoordinates = vector<Poses>(totalNrFrames);
	Mat R0;
	if (pars.R.empty())
		R0 = Mat::eye(3, 3, CV_64FC1);
	else
		R0 = pars.R;
	Mat t1 = Mat::zeros(3, 1, CV_64FC1);
	if (nrTracks == 1)
	{
		pars.camTrack[0] /= norm(pars.camTrack[0]);
		Mat R1 = R0 * getTrackRot(pars.camTrack[0]);
		Mat t_piece = absCamVelocity * pars.camTrack[0];
		absCamCoordinates[0] = Poses(R1.clone(), t1.clone());
		for (size_t i = 1; i < totalNrFrames; i++)
		{
			t1 += t_piece;
			absCamCoordinates[i] = Poses(R1.clone(), t1.clone());
		}
	}
	else
	{
		//Get differential vectors of the path and the overall path length
		vector<Mat> diffTrack = vector<Mat>(nrTracks-1);
		vector<double> tdiffNorms = vector<double>(nrTracks-1);
		double trackNormSum = 0;//norm(pars.camTrack[0]);
//		diffTrack[0] = pars.camTrack[0].clone();// / trackNormSum;
//		tdiffNorms[0] = trackNormSum;
		for (size_t i = 0; i < nrTracks - 1; i++)
		{
			Mat tdiff = pars.camTrack[i + 1] - pars.camTrack[i];
			double tdiffnorm = norm(tdiff);
			trackNormSum += tdiffnorm;
			diffTrack[i] = tdiff.clone();// / tdiffnorm;
			tdiffNorms[i] = tdiffnorm;
		}

		//Calculate a new scaling for the path based on the original path length, total number of frames and camera velocity
		double trackScale = (double)(totalNrFrames - 1) * absCamVelocity / trackNormSum;
		//Rescale track diffs
		for (size_t i = 0; i < nrTracks - 1; i++)
		{
			diffTrack[i] *= trackScale;
			tdiffNorms[i] *= trackScale;
		}

		//Get camera positions
		Mat R_track = getTrackRot(diffTrack[0]);
		Mat R_track_old = R_track.clone();
		Mat R1 = R0 * R_track;
		pars.camTrack[0].copyTo(t1);
		t1 *= trackScale;
		absCamCoordinates[0] = Poses(R1.clone(), t1.clone());
		double actDiffLength = 0;
		size_t actTrackNr = 0, lastTrackNr = 0;
		for (size_t i = 1; i < totalNrFrames; i++)
		{
			bool firstAdd = true;
			Mat multTracks = Mat::zeros(3, 1, CV_64FC1);
			double usedLength = 0;
			while ((actDiffLength < (absCamVelocity - DBL_EPSILON)) && (actTrackNr < (nrTracks - 1)))
			{
				if (firstAdd)
				{
					multTracks += actDiffLength * diffTrack[lastTrackNr] / tdiffNorms[lastTrackNr];
					usedLength = actDiffLength;
					firstAdd = false;
				}
				else
				{
					multTracks += diffTrack[lastTrackNr];
					usedLength += tdiffNorms[lastTrackNr];
				}

				lastTrackNr = actTrackNr;

				actDiffLength += tdiffNorms[actTrackNr++];
			}
			multTracks += (absCamVelocity - usedLength) * diffTrack[lastTrackNr] / tdiffNorms[lastTrackNr];

			R_track = getTrackRot(diffTrack[lastTrackNr], R_track_old);
			R_track_old = R_track.clone();
			R1 = R0 * R_track;
			t1 += multTracks;
			absCamCoordinates[i] = Poses(R1.clone(), t1.clone());
			actDiffLength -= absCamVelocity;
		}
	}

	visualizeCamPath();
}

/*Calculates a rotation for every differential vector of a track segment to ensure that the camera looks always in the direction of the track segment.
* If the track segment equals the x-axis, the camera faces into positive x-direction (if the initial rotaion equals the identity).
* The y axis always points down as the up vector is defined as [0,-1,0]. Thus, there is no roll in the camera rotation.
*/
cv::Mat genStereoSequ::getTrackRot(cv::Mat tdiff, cv::InputArray R_old)
{
	CV_Assert((tdiff.rows == 3) && (tdiff.cols == 1) && (tdiff.type() == CV_64FC1));

	Mat R_C2W = Mat::eye(3, 3, CV_64FC1);

	if (nearZero(cv::norm(tdiff)))
		return R_C2W;

	tdiff /= norm(tdiff);

	Mat Rold;
	if (!R_old.empty())
		Rold = R_old.getMat();

	
	//Define up-vector as global -y axis
	Mat world_up = (Mat_<double>(3, 1) << 0, -1, 0);
	world_up /= norm(world_up);

	if (nearZero(cv::sum(tdiff - world_up)[0]))
	{
		R_C2W = (Mat_<double>(3, 1) << 1.0, 0, 0,
			0, 0, -1.0,
			0, 1.0, 0);
		if (!Rold.empty())
		{
			Mat Rr;
			if (roundR(Rold, Rr, R_C2W))
			{
				R_C2W = Rr;
			}
		}
	}
	else if (nearZero(cv::sum(tdiff + world_up)[0]))
	{
		R_C2W = (Mat_<double>(3, 1) << 1.0, 0, 0,
			0, 0, 1.0,
			0, -1.0, 0);
		if (!Rold.empty())
		{
			Mat Rr;
			if (roundR(Rold, Rr, R_C2W))
			{
				R_C2W = Rr;
			}
		}
	}
	else
	{
		//Get local axis that is perpendicular to up-vector and look-at vector (new x axis) -> to prevent roll
		Mat xa = tdiff.cross(world_up);
		xa /= norm(xa);
		//Get local axis that is perpendicular to new x-axis and look-at vector (new y axis)
		Mat ya = tdiff.cross(xa);
		ya /= norm(ya);

		//Build the rotation matrix (camera to world: x_world = R_C2W * x_local + t) by stacking the normalized axis vectors as columns of R=[x,y,z]
		xa.copyTo(R_C2W.col(0));
		ya.copyTo(R_C2W.col(1));
		tdiff.copyTo(R_C2W.col(2));
	}

	return R_C2W;//return rotation from camera to world
}

/*Rounds a rotation matrix to its nearest integer values and checks if it is still a rotation matrix and does not change more than 22.5deg from the original rotation matrix.
As an option, the error of the rounded rotation matrix can be compared to an angular difference of a second given rotation matrix R_fixed to R_old. 
The rotation matrix with the smaller angular difference is selected.
This function is used to select a proper rotation matrix if the "look at" and "up vector" are nearly equal. I trys to find the nearest rotation matrix aligened to the
"look at" vector taking into account the rotation matrix calculated from the old/last "look at" vector
*/
bool roundR(cv::Mat R_old, cv::Mat & R_round, cv::InputArray R_fixed)
{
	R_round = roundMat(R_old);
	if (!isMatRotationMat(R_round))
	{
		return false;
	}

	double rd = abs(RAD2DEG(rotDiff(R_old, R_round)));
	double rfd = 360.0;

	if (!R_fixed.empty())
	{
		Mat Rf = R_fixed.getMat();
		rfd = abs(RAD2DEG(rotDiff(Rf, R_old)));

		if (rfd < rd)
		{
			Rf.copyTo(R_round);
			return true;
		}
	}

	if (rd < 22.5)
	{
		return true;
	}
	
	return false;
}

void genStereoSequ::visualizeCamPath()
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(5.0);

	Eigen::Affine3f m;
	m.setIdentity();
	for (auto i : absCamCoordinates)
	{
		Eigen::Vector3d te;
		Eigen::Matrix3d Re;
		cv::cv2eigen(i.R, Re);
		cv::cv2eigen(i.t, te);
		m.matrix().block<3, 3>(0, 0) = Re.cast<float>();
		m.matrix().block<3, 1>(0, 3) = te.cast<float>();
		viewer->addCoordinateSystem(1.0, m);		
	}

	viewer->initCameraParameters();

	//--------------------
	// -----Main loop-----
	//--------------------
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

//Calculate the thresholds for the depths near, mid, and far for every camera configuration
bool genStereoSequ::getDepthRanges()
{
	depthFar = vector<double>(nrStereoConfs);
	depthMid = vector<double>(nrStereoConfs);
	depthNear = vector<double>(nrStereoConfs);
	for (size_t i = 0; i < nrStereoConfs; i++)
	{
		Mat x1, x2;
		if (abs(t[i].at<double>(0)) > abs(t[i].at<double>(1)))
		{
			if (t[i].at<double>(0) < t[i].at<double>(1))
			{
				x1 = (Mat_<double>(3, 1) << (double)imgSize.width, (double)imgSize.height / 2.0, 1.0);
				x2 = (Mat_<double>(3, 1) << 0, (double)imgSize.height / 2.0, 1.0);
			}
			else
			{
				x2 = (Mat_<double>(3, 1) << (double)imgSize.width, (double)imgSize.height / 2.0, 1.0);
				x1 = (Mat_<double>(3, 1) << 0, (double)imgSize.height / 2.0, 1.0);
			}
		}
		else
		{
			if (t[i].at<double>(1) < t[i].at<double>(0))
			{
				x1 = (Mat_<double>(3, 1) << (double)imgSize.width / 2.0, (double)imgSize.height, 1.0);
				x2 = (Mat_<double>(3, 1) << (double)imgSize.width / 2.0, 0, 1.0);
			}
			else
			{
				x2 = (Mat_<double>(3, 1) << (double)imgSize.width / 2.0, (double)imgSize.height, 1.0);
				x1 = (Mat_<double>(3, 1) << (double)imgSize.width / 2.0, 0, 1.0);
			}
		}

		double bl = norm(t[i]);
		depthFar[i] = sqrt(K1.at<double>(0, 0) * bl * bl / 0.15);//0.15 corresponds to the approx. typ. correspondence accuracy in pixels

		//Calculate min distance for 3D points visible in both images
		Mat b1 = getLineCam1(K1, x1);
		Mat a2, b2;
		getLineCam2(R[i], t[i], K2, x2, a2, b2);
		depthNear[i] = getLineIntersect(b1, a2, b2);
		depthNear[i] = depthNear[i] > 0 ? depthNear[i] : 0;
		depthMid[i] = (depthFar[i] - depthNear[i]) / 2.0;
		if (depthMid[i] < 0)
		{
			return false;
		}
	}

	return true;
}

/* As the user can specify portions of different depths (near, mid, far) globally for the whole image and also for regions within the image,
these fractions typically do not match. As a result, the depth range fractions per region must be adapted to match the overall fractions of the
whole image. Moreover, the fraction of correspondences per region have an impact on the effective depth portions that must be considered when
adapting the fractions in the image regions.
*/
void genStereoSequ::adaptDepthsPerRegion()
{
	if (pars.depthsPerRegion.empty())
	{
		pars.depthsPerRegion = std::vector<std::vector<depthPortion>>(3, std::vector<depthPortion>(3));
		std::uniform_real_distribution<double> distribution(0, 1.0);
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				pars.depthsPerRegion[i][j] = depthPortion(distribution(rand_gen), distribution(rand_gen), distribution(rand_gen));
			}
		}
	}
	else
	{
		//Check if the sum of fractions is 1.0
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				pars.depthsPerRegion[i][j].sumTo1();
			}
		}
	}
	pars.corrsPerDepth.sumTo1();
	
	depthsPerRegion = std::vector<std::vector<std::vector<depthPortion>>>(pars.corrsPerRegion.size(), pars.depthsPerRegion);

	//Correct the portion of depths per region so that they meet the global depth range requirement per image
	for (size_t k = 0; k < pars.corrsPerRegion.size(); k++)
	{
		//Adapt the fractions of near depths of every region to match the global requirement of the near depth fraction
		updDepthReg(true, depthsPerRegion[k], pars.corrsPerRegion[k]);

		//Update the mid and far depth fractions of each region according to the new near depth fractions
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				double splitrem = 1.0 - depthsPerRegion[k][i][j].near;
				if (!nearZero(splitrem))
				{
					if (!nearZero(depthsPerRegion[k][i][j].mid) && !nearZero(depthsPerRegion[k][i][j].far))
					{
						double fmsum = depthsPerRegion[k][i][j].mid + depthsPerRegion[k][i][j].far;
						depthsPerRegion[k][i][j].mid = splitrem * depthsPerRegion[k][i][j].mid / fmsum;
						depthsPerRegion[k][i][j].far = splitrem * depthsPerRegion[k][i][j].far / fmsum;
					}
					else if (nearZero(depthsPerRegion[k][i][j].mid) && nearZero(depthsPerRegion[k][i][j].far))
					{
						depthsPerRegion[k][i][j].mid = splitrem / 2.0;
						depthsPerRegion[k][i][j].far = splitrem / 2.0;
					}
					else if (nearZero(depthsPerRegion[k][i][j].mid))
					{
						depthsPerRegion[k][i][j].far = splitrem;
					}
					else
					{
						depthsPerRegion[k][i][j].mid = splitrem;
					}
				}
				else
				{
					depthsPerRegion[k][i][j].mid = 0;
					depthsPerRegion[k][i][j].far = 0;
				}
			}
		}

		//Adapt the fractions of far depths of every region to match the global requirement of the far depth fraction
		updDepthReg(false, depthsPerRegion[k], pars.corrsPerRegion[k]);

		//Update the mid depth fractions of each region according to the new near & far depth fractions
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				depthsPerRegion[k][i][j].mid = 1.0 - (depthsPerRegion[k][i][j].near + depthsPerRegion[k][i][j].far);
			}
		}

#if 1
		//Now, the sum of mid depth regions should correspond to the global requirement
		double portSum = 0;
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				portSum += depthsPerRegion[k][i][j].mid * pars.corrsPerRegion[k].at<double>(i, j);
			}
		}
		double c1 = pars.corrsPerDepth.mid - portSum;
		if (!nearZero(c1 / 10.0))
		{
			cout << "Adaption of depth fractions in regions failed!" << endl;
		}
#endif
	}
}

//Only adapt the fraction of near or far depths per region to the global requirement
void genStereoSequ::updDepthReg(bool isNear, std::vector<std::vector<depthPortion>> &depthPerRegion, cv::Mat &cpr)
{
	//If isNear=false, it is assumed that the fractions of near depths are already fixed
	std::vector<std::vector<double>> oneDepthPerRegion(3, std::vector<double>(3));
	std::vector<std::vector<double>> oneDepthPerRegionMaxVal(3, std::vector<double>(3, 1.0));
	if (isNear)
	{
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				oneDepthPerRegion[i][j] = depthPerRegion[i][j].near;
			}
		}
	}
	else
	{
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				oneDepthPerRegion[i][j] = depthPerRegion[i][j].far;
				oneDepthPerRegionMaxVal[i][j] = 1.0 - depthPerRegion[i][j].near;
			}
		}
	}

	double portSum = 0, c1 = 1.0, dsum = 0, dsum1 = 0;
	size_t cnt = 0;
	//Mat cpr = pars.corrsPerRegion[k];
	while (!nearZero(c1))
	{
		cnt++;
		portSum = 0;
		dsum = 0;
		dsum1 = 0;
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				portSum += oneDepthPerRegion[i][j] * cpr.at<double>(i, j);
				dsum += oneDepthPerRegion[i][j];
				dsum1 += 1.0 - oneDepthPerRegion[i][j];
			}
		}
		if (isNear)
			c1 = pars.corrsPerDepth.near - portSum;
		else
			c1 = pars.corrsPerDepth.far - portSum;

		bool breakit = false;
		if (!nearZero(c1))
		{
			double c12 = 0, c1sum = 0;
			for (size_t i = 0; i < 3; i++)
			{
				for (size_t j = 0; j < 3; j++)
				{
					double newval;
					if (cnt < 3)
					{
						newval = oneDepthPerRegion[i][j] + c1 * cpr.at<double>(i, j) * oneDepthPerRegion[i][j] / dsum;
					}
					else
					{
						c12 = c1 * cpr.at<double>(i, j) * (0.75 * oneDepthPerRegion[i][j] / dsum + 0.25 * (1.0 - oneDepthPerRegion[i][j]) / dsum1);
						double c1diff = c1 - (c1sum + c12);
						if ((c1 > 0) && (c1diff < 0) ||
							(c1 < 0) && (c1diff > 0))
						{
							c12 = c1 - c1sum;
						}
						newval = oneDepthPerRegion[i][j] + c12;
					}
					if (newval > oneDepthPerRegionMaxVal[i][j])
					{
						c1sum += oneDepthPerRegionMaxVal[i][j] - oneDepthPerRegion[i][j];
						oneDepthPerRegion[i][j] = oneDepthPerRegionMaxVal[i][j];

					}
					else if (newval < 0)
					{
						c1sum -= oneDepthPerRegion[i][j];
						oneDepthPerRegion[i][j] = 0;
					}
					else
					{
						c1sum += newval - oneDepthPerRegion[i][j];
						oneDepthPerRegion[i][j] = newval;
					}
					if (nearZero(c1sum - c1))
					{
						breakit = true;
						break;
					}
				}
				if (breakit) break;
			}
			if (breakit) break;
		}
	}

	if (isNear)
	{
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				depthPerRegion[i][j].near = oneDepthPerRegion[i][j];
			}
		}
	}
	else
	{
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				depthPerRegion[i][j].far = oneDepthPerRegion[i][j];
			}
		}
	}
}

//Check if the given ranges of connected depth areas per image region are correct and initialize them for every definition of depths per image region
void genStereoSequ::checkDepthAreas()
{
	//
	//Below: 9 is the nr of regions, minDArea is the min area and 2*sqrt(minDArea) is the gap between areas;
	//size_t maxElems = imgSize.area() / (9 * ((size_t)minDArea + 2 * (size_t)sqrt(minDArea)));
	//Below: 9 is the nr of regions; 4 * (minDArea + sqrt(minDArea)) + 1 corresponds to the area using the side length 2*sqrt(minDArea)+1
	size_t maxElems = (size_t)std::max(imgSize.area() / (9 * (int)(4 * (minDArea + sqrt(minDArea)) + 1)), 1);
	if (pars.nrDepthAreasPReg.empty())
	{
		pars.nrDepthAreasPReg = std::vector<std::vector<std::pair<size_t, size_t>>>(3, std::vector<std::pair<size_t, size_t>>(3));
		std::uniform_int_distribution<size_t> distribution(1, maxElems + 1);
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				size_t tmp = distribution(rand_gen);
				tmp = tmp < 2 ? 2 : tmp;
				size_t tmp1 = distribution(rand_gen) % tmp;
				tmp1 = tmp1 == 0 ? 1 : tmp1;
				pars.nrDepthAreasPReg[i][j] = make_pair(tmp1, tmp);
			}
		}
	}
	else
	{
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				if (pars.nrDepthAreasPReg[i][j].first == 0)
				{
					pars.nrDepthAreasPReg[i][j].first = 1;
				}
				else if (pars.nrDepthAreasPReg[i][j].first > (maxElems - 1))
				{
					pars.nrDepthAreasPReg[i][j].first = maxElems - 1;
				}

				if (pars.nrDepthAreasPReg[i][j].second == 0)
				{
					pars.nrDepthAreasPReg[i][j].second = 1;
				}
				else if (pars.nrDepthAreasPReg[i][j].second > maxElems)
				{
					pars.nrDepthAreasPReg[i][j].second = maxElems;
				}

				if (pars.nrDepthAreasPReg[i][j].second < pars.nrDepthAreasPReg[i][j].first)
				{
					size_t tmp = pars.nrDepthAreasPReg[i][j].first;
					pars.nrDepthAreasPReg[i][j].first = pars.nrDepthAreasPReg[i][j].second;
					pars.nrDepthAreasPReg[i][j].second = tmp;
				}
			}
		}
	}

	//Initialize the numbers for every region and depth definition
	nrDepthAreasPRegNear = std::vector<cv::Mat>(depthsPerRegion.size(), Mat::ones(3, 3, CV_32SC1));
	nrDepthAreasPRegMid = std::vector<cv::Mat>(depthsPerRegion.size(), Mat::ones(3, 3, CV_32SC1));
	nrDepthAreasPRegFar = std::vector<cv::Mat>(depthsPerRegion.size(), Mat::ones(3, 3, CV_32SC1));

	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			if (pars.nrDepthAreasPReg[y][x].second < 4)
			{
				for (size_t i = 0; i < depthsPerRegion.size(); i++)
				{
					if (!nearZero(depthsPerRegion[i][y][x].near) &&
						!nearZero(depthsPerRegion[i][y][x].mid) &&
						!nearZero(depthsPerRegion[i][y][x].far))
					{
						continue;//1 remains in every element
					}
					else
					{
						int cnt = (int)pars.nrDepthAreasPReg[y][x].second;
						int tmp = -10;
						nrDepthAreasPRegNear[i].at<int32_t>(y, x) = 0;
						nrDepthAreasPRegMid[i].at<int32_t>(y, x) = 0;
						nrDepthAreasPRegFar[i].at<int32_t>(y, x) = 0;
						bool lockdistr[3] = { true, true, true };
						while (cnt > 0)
						{
							if (!nearZero(depthsPerRegion[i][y][x].near) && lockdistr[0])
							{
								cnt--;
								nrDepthAreasPRegNear[i].at<int32_t>(y, x)++;
							}
							if (!nearZero(depthsPerRegion[i][y][x].mid) && lockdistr[1])
							{
								cnt--;
								nrDepthAreasPRegMid[i].at<int32_t>(y, x)++;
							}
							if (!nearZero(depthsPerRegion[i][y][x].far) && lockdistr[2])
							{
								cnt--;
								nrDepthAreasPRegFar[i].at<int32_t>(y, x)++;
							}
							if ((cnt > 0) && (tmp == -10))
							{
								if ((pars.nrDepthAreasPReg[y][x].second - pars.nrDepthAreasPReg[y][x].first) != 0)
								{
									tmp = cnt - (int)pars.nrDepthAreasPReg[y][x].second;
									tmp += pars.nrDepthAreasPReg[y][x].first + (std::rand() % (pars.nrDepthAreasPReg[y][x].second - pars.nrDepthAreasPReg[y][x].first + 1));
									cnt = tmp;
								}
								if (cnt > 0)
								{
									if (!(!nearZero(depthsPerRegion[i][y][x].near) && ((depthsPerRegion[i][y][x].near > depthsPerRegion[i][y][x].mid) ||
										(depthsPerRegion[i][y][x].near > depthsPerRegion[i][y][x].far))))
									{
										lockdistr[0] = false;
									}
									if (!(!nearZero(depthsPerRegion[i][y][x].mid) && ((depthsPerRegion[i][y][x].mid > depthsPerRegion[i][y][x].near) ||
										(depthsPerRegion[i][y][x].mid > depthsPerRegion[i][y][x].far))))
									{
										lockdistr[1] = false;
									}
									if (!(!nearZero(depthsPerRegion[i][y][x].far) && ((depthsPerRegion[i][y][x].far > depthsPerRegion[i][y][x].near) ||
										(depthsPerRegion[i][y][x].far > depthsPerRegion[i][y][x].mid))))
									{
										lockdistr[2] = false;
									}
								}
							}
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < depthsPerRegion.size(); i++)
				{
					int nra = pars.nrDepthAreasPReg[y][x].first + (std::rand() % (pars.nrDepthAreasPReg[y][x].second - pars.nrDepthAreasPReg[y][x].first + 1));
					int32_t maxAPReg[3];
					double maxAPRegd[3];
					maxAPRegd[0] = depthsPerRegion[i][y][x].near * (double)nra;
					maxAPRegd[1] = depthsPerRegion[i][y][x].mid * (double)nra;
					maxAPRegd[2] = depthsPerRegion[i][y][x].far * (double)nra;
					maxAPReg[0] = (int32_t)round(maxAPRegd[0]);
					maxAPReg[1] = (int32_t)round(maxAPRegd[1]);
					maxAPReg[2] = (int32_t)round(maxAPRegd[2]);
					int32_t diffap = (int32_t)nra - (maxAPReg[0] + maxAPReg[1] + maxAPReg[2]);
					if (diffap != 0)
					{
						maxAPRegd[0] -= (double)maxAPReg[0];
						maxAPRegd[1] -= (double)maxAPReg[1];
						maxAPRegd[2] -= (double)maxAPReg[2];
						if (diffap < 0)
						{
							int cnt = 0;
							std::ptrdiff_t pdiff = min_element(maxAPRegd, maxAPRegd + 3) - maxAPRegd;
							while ((diffap < 0) && (cnt < 3))
							{
								if (maxAPReg[pdiff] > 1)
								{
									maxAPReg[pdiff]--;
									diffap++;
								}
								if (diffap < 0)
								{
									if ((maxAPReg[(pdiff + 1) % 3] > 1) && (maxAPRegd[(pdiff + 1) % 3] <= maxAPRegd[(pdiff + 2) % 3]))
									{
										maxAPReg[(pdiff + 1) % 3]--;
										diffap++;
									}
									else if ((maxAPReg[(pdiff + 2) % 3] > 1) && (maxAPRegd[(pdiff + 2) % 3] < maxAPRegd[(pdiff + 1) % 3]))
									{
										maxAPReg[(pdiff + 2) % 3]--;
										diffap++;
									}
								}
								cnt++;
							}
						}
						else
						{
							std::ptrdiff_t pdiff = max_element(maxAPRegd, maxAPRegd + 3) - maxAPRegd;
							while (diffap > 0)
							{
								maxAPReg[pdiff]++;
								diffap--;
								if (diffap > 0)
								{
									if (maxAPRegd[(pdiff + 1) % 3] >= maxAPRegd[(pdiff + 2) % 3])
									{
										maxAPReg[(pdiff + 1) % 3]++;
										diffap--;
									}
									else
									{
										maxAPReg[(pdiff + 2) % 3]++;
										diffap--;
									}
								}
							}
						}
					}
					
					nrDepthAreasPRegNear[i].at<int32_t>(y, x) = maxAPReg[0];
					nrDepthAreasPRegMid[i].at<int32_t>(y, x) = maxAPReg[1];
					nrDepthAreasPRegFar[i].at<int32_t>(y, x) = maxAPReg[2];
					if (!nearZero(depthsPerRegion[i][y][x].near) && (maxAPReg[0] == 0))
					{
						nrDepthAreasPRegNear[i].at<int32_t>(y, x)++;
					}
					if (!nearZero(depthsPerRegion[i][y][x].mid) && (maxAPReg[1] == 0))
					{
						nrDepthAreasPRegMid[i].at<int32_t>(y, x)++;
					}
					if (!nearZero(depthsPerRegion[i][y][x].far) && (maxAPReg[2] == 0))
					{
						nrDepthAreasPRegFar[i].at<int32_t>(y, x)++;
					}
				}
			}
		}
	}
}

//Calculate the area in pixels for every depth and region
void genStereoSequ::calcPixAreaPerDepth()
{
	int32_t regArea = (int32_t)imgSize.area() / 9;
	areaPRegNear.resize(depthsPerRegion.size(), Mat::zeros(3, 3, CV_32SC1));
	areaPRegMid.resize(depthsPerRegion.size(), Mat::zeros(3, 3, CV_32SC1));
	areaPRegFar.resize(depthsPerRegion.size(), Mat::zeros(3, 3, CV_32SC1));

	for (size_t i = 0; i < depthsPerRegion.size(); i++)
	{
		for (size_t y = 0; y < 3; y++)
		{
			for (size_t x = 0; x < 3; x++)
			{
				int32_t tmp = (int32_t)round(depthsPerRegion[i][y][x].near * (double)regArea);
				if ((tmp != 0) && (tmp < minDArea))
					tmp = minDArea;
				areaPRegNear[i].at<int32_t>(y, x) = tmp;

				tmp = (int32_t)round(depthsPerRegion[i][y][x].mid * (double)regArea);
				if ((tmp != 0) && (tmp < minDArea))
					tmp = minDArea;
				areaPRegMid[i].at<int32_t>(y, x) = tmp;

				tmp = (int32_t)round(depthsPerRegion[i][y][x].far * (double)regArea);
				if ((tmp != 0) && (tmp < minDArea))
					tmp = minDArea;
				areaPRegFar[i].at<int32_t>(y, x) = tmp;
			}
		}
	}
}

/*Backproject 3D points (generated one or more frames before) found to be possibly visible in the 
current stereo rig position to the stereo image planes and check if they are visible or produce 
outliers in the first or second stereo image.
*/
void genStereoSequ::backProject3D()
{
	if (!actCorrsImg2TNFromLast.empty())
		actCorrsImg2TNFromLast.release();
	if (!actCorrsImg2TNFromLast_Idx.empty())
		actCorrsImg2TNFromLast_Idx.clear();
	if (!actCorrsImg1TNFromLast.empty())
		actCorrsImg1TNFromLast.release();
	if (!actCorrsImg1TNFromLast_Idx.empty())
		actCorrsImg1TNFromLast_Idx.clear();
	if (!actCorrsImg1TPFromLast.empty())
		actCorrsImg1TPFromLast.release();
	if (!actCorrsImg2TPFromLast.empty())
		actCorrsImg2TPFromLast.release();
	if (!actCorrsImg12TPFromLast_Idx.empty())
		actCorrsImg12TPFromLast_Idx.clear();

	if (actImgPointCloudFromLast.empty())
		return;

	struct imgWH {
		double width;
		double height;
		double maxDist;
	} dimgWH;
	dimgWH.width = (double)(imgSize.width - 1);
	dimgWH.height = (double)(imgSize.height - 1);
	dimgWH.maxDist = maxFarDistMultiplier * actDepthFar;

	std::vector<cv::Point3d> actImgPointCloudFromLast_tmp;
	size_t idx1 = 0;
	for (auto pt : actImgPointCloudFromLast)
	{
		if ((pt.z < actDepthNear) ||
			(pt.z > dimgWH.maxDist))
		{
			continue;
		}

		Mat X = Mat(pt).reshape(1, 3);
		Mat x1 = K1 * X;
		x1 /= x1.at<double>(2);

		//Check if the point is within the area of a moving object
		if (combMovObjLabelsAll.at<unsigned char>((int)round(x1.at<double>(1)), (int)round(x1.at<double>(0))) > 0)
			continue;

		bool outOfR[2] = { false, false };
		if ((x1.at<double>(0) < 0) || (x1.at<double>(0) > dimgWH.width) ||
			(x1.at<double>(1) < 0) || (x1.at<double>(0) > dimgWH.height))//Not visible in first image
		{
			outOfR[0] = true;
		}

		Mat x2 = K2 * (actR * X + actT);
		x2 /= x2.at<double>(2);

		if ((x2.at<double>(0) < 0) || (x2.at<double>(0) > dimgWH.width) ||
			(x2.at<double>(1) < 0) || (x2.at<double>(0) > dimgWH.height))//Not visible in second image
		{
			outOfR[1] = true;
		}

		//Check if the point is within the area of a moving object in the second image
		if (movObjMask2All.at<unsigned char>((int)round(x2.at<double>(1)), (int)round(x2.at<double>(0))) > 0)
			outOfR[1] = true;

		if (outOfR[0] && outOfR[1])
		{
			continue;
		}
		else if (outOfR[0])
		{
			actCorrsImg2TNFromLast.push_back(x2.t());
			actCorrsImg2TNFromLast_Idx.push_back(idx1);
		}
		else if (outOfR[1])
		{
			actCorrsImg1TNFromLast.push_back(x1.t());
			actCorrsImg1TNFromLast_Idx.push_back(idx1);
		}
		else
		{
			actCorrsImg1TPFromLast.push_back(x1.t());
			actCorrsImg2TPFromLast.push_back(x2.t());
			actCorrsImg12TPFromLast_Idx.push_back(idx1);
		}
		actImgPointCloudFromLast_tmp.push_back(pt);
		idx1++;
	}
	actImgPointCloudFromLast = actImgPointCloudFromLast_tmp;
	if (!actCorrsImg1TNFromLast.empty())
		actCorrsImg1TNFromLast = actCorrsImg1TNFromLast.t();
	if (!actCorrsImg2TNFromLast.empty())
		actCorrsImg2TNFromLast = actCorrsImg2TNFromLast.t();
	if (!actCorrsImg1TPFromLast.empty())
	{
		actCorrsImg1TPFromLast = actCorrsImg1TPFromLast.t();
		actCorrsImg2TPFromLast = actCorrsImg2TPFromLast.t();
	}
}

//Generate seeds for generating depth areas and include the seeds found by backprojection of the 3D points of the last frames
void genStereoSequ::checkDepthSeeds()
{
	seedsNear = std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>>(3, std::vector<std::vector<cv::Point3_<int32_t>>>(3));
	seedsMid = std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>>(3, std::vector<std::vector<cv::Point3_<int32_t>>>(3));
	seedsFar = std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>>(3, std::vector<std::vector<cv::Point3_<int32_t>>>(3));

	int posadd1 = max((int)ceil(pars.minKeypDist), (int)sqrt(minDArea));
	int sqrSi1 = 2 * posadd1;
	cv::Mat filtInitPts = Mat::zeros(imgSize.width + sqrSi1, imgSize.height + sqrSi1, CV_8UC1);
	sqrSi1++;
	Mat csurr1 = Mat::ones(sqrSi1, sqrSi1, CV_8UC1);
	//int maxSum1 = sqrSi1 * sqrSi1;
	
	cv::Size regSi = Size(imgSize.width / 3, imgSize.height / 3);
	if (!actCorrsImg1TPFromLast.empty())//Take seeding positions from backprojected coordinates
	{
		std::vector<cv::Point3_<int32_t>> seedsNear_tmp, seedsNear_tmp1;
		std::vector<cv::Point3_<int32_t>> seedsMid_tmp, seedsMid_tmp1;
		std::vector<cv::Point3_<int32_t>> seedsFar_tmp, seedsFar_tmp1;
		//Identify depth categories
		for (size_t i = 0; i < actCorrsImg12TPFromLast_Idx.size(); i++)
		{
			if (actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[i]].z >= actDepthFar)
			{
				seedsFar_tmp.push_back(cv::Point3_<int32_t>((int32_t)round(actCorrsImg1TPFromLast.at<double>(0, i)), (int32_t)round(actCorrsImg1TPFromLast.at<double>(1, i)), (int32_t)i));
			}
			else if (actImgPointCloudFromLast[actCorrsImg12TPFromLast_Idx[i]].z >= actDepthMid)
			{
				seedsMid_tmp.push_back(cv::Point3_<int32_t>((int32_t)round(actCorrsImg1TPFromLast.at<double>(0, i)), (int32_t)round(actCorrsImg1TPFromLast.at<double>(1, i)), (int32_t)i));
			}
			else
			{
				seedsNear_tmp.push_back(cv::Point3_<int32_t>((int32_t)round(actCorrsImg1TPFromLast.at<double>(0, i)), (int32_t)round(actCorrsImg1TPFromLast.at<double>(1, i)), (int32_t)i));
			}
		}

		//Check if the seeds are too near to each other
		int posadd = max((int)ceil(pars.minKeypDist), 1);
		int sqrSi = 2 * posadd;
		//cv::Mat filtInitPts = Mat::zeros(imgSize.width + sqrSi, imgSize.height + sqrSi, CV_8UC1);
		sqrSi++;//sqrSi = 2 * (int)floor(pars.minKeypDist) + 1;
		//csurr = Mat::ones(sqrSi, sqrSi, CV_8UC1);
		//int maxSum = sqrSi * sqrSi;
		int sqrSiDiff2 = (sqrSi1 - sqrSi) / 2;
		int hlp2 = sqrSi + sqrSiDiff2;

		vector<size_t> delListCorrs, delList3D;
		if (!seedsNear_tmp.empty())
		{
			for (size_t i = 0; i < seedsNear_tmp.size(); i++)
			{
				Mat s_tmp = filtInitPts(Range(seedsNear_tmp[i].y + sqrSiDiff2, seedsNear_tmp[i].y + hlp2), Range(seedsNear_tmp[i].x + sqrSiDiff2, seedsNear_tmp[i].x + hlp2));
				if (s_tmp.at<unsigned char>(posadd, posadd) > 0)
				{
					delListCorrs.push_back((size_t)seedsNear_tmp[i].z);
					delList3D.push_back(actCorrsImg12TPFromLast_Idx[delListCorrs.back()]);
					continue;
				}
				csurr.copyTo(s_tmp);
				seedsNear_tmp1.push_back(seedsNear_tmp[i]);
			}
		}
		if (!seedsMid_tmp.empty())
		{
			for (size_t i = 0; i < seedsMid_tmp.size(); i++)
			{
				Mat s_tmp = filtInitPts(Range(seedsMid_tmp[i].y + sqrSiDiff2, seedsMid_tmp[i].y + hlp2), Range(seedsMid_tmp[i].x + sqrSiDiff2, seedsMid_tmp[i].x + hlp2));
				if (s_tmp.at<unsigned char>(posadd, posadd) > 0)
				{
					delListCorrs.push_back((size_t)seedsMid_tmp[i].z);
					delList3D.push_back(actCorrsImg12TPFromLast_Idx[delListCorrs.back()]);
					continue;
				}
				csurr.copyTo(s_tmp);
				seedsMid_tmp1.push_back(seedsMid_tmp[i]);
			}
		}
		if (!seedsFar_tmp.empty())
		{
			for (size_t i = 0; i < seedsFar_tmp.size(); i++)
			{
				Mat s_tmp = filtInitPts(Range(seedsFar_tmp[i].y + sqrSiDiff2, seedsFar_tmp[i].y + hlp2), Range(seedsFar_tmp[i].x + sqrSiDiff2, seedsFar_tmp[i].x + hlp2));
				if (s_tmp.at<unsigned char>(posadd, posadd) > 0)
				{
					delListCorrs.push_back((size_t)seedsFar_tmp[i].z);
					delList3D.push_back(actCorrsImg12TPFromLast_Idx[delListCorrs.back()]);
					continue;
				}
				csurr.copyTo(s_tmp);
				seedsFar_tmp1.push_back(seedsFar_tmp[i]);
			}
		}
		filtInitPts(Rect(sqrSiDiff2, sqrSiDiff2, imgSize.width + 2 * posadd, imgSize.height + 2 * posadd)).copyTo(corrsIMG);
		
		//Delete correspondences and 3D points that were to near to each other in the image
		if (!delListCorrs.empty())
		{
			std::vector<cv::Point3d> actImgPointCloudFromLast_tmp;
			cv::Mat actCorrsImg1TPFromLast_tmp, actCorrsImg2TPFromLast_tmp;

			sort(delList3D.begin(), delList3D.end(), [](size_t first, size_t second) {return first < second; });//Ascending order

			if (!actCorrsImg1TNFromLast_Idx.empty())//Adapt the indices for TN (single keypoints without a match)
			{
				adaptIndicesNoDel(actCorrsImg1TNFromLast_Idx, delList3D);
			}
			if (!actCorrsImg2TNFromLast_Idx.empty())//Adapt the indices for TN (single keypoints without a match)
			{
				adaptIndicesNoDel(actCorrsImg2TNFromLast_Idx, delList3D);
			}
			adaptIndicesNoDel(actCorrsImg12TPFromLast_Idx, delList3D);
			deleteVecEntriesbyIdx(actImgPointCloudFromLast, delList3D);

			sort(delListCorrs.begin(), delListCorrs.end(), [](size_t first, size_t second) {return first < second; });
			if (!seedsNear_tmp1.empty())
				adaptIndicesCVPtNoDel(seedsNear_tmp1, delListCorrs);
			if (!seedsMid_tmp1.empty())
				adaptIndicesCVPtNoDel(seedsMid_tmp1, delListCorrs);
			if (!seedsFar_tmp1.empty())
				adaptIndicesCVPtNoDel(seedsFar_tmp1, delListCorrs);
			deleteVecEntriesbyIdx(actCorrsImg12TPFromLast_Idx, delListCorrs);
			deleteMatEntriesByIdx(actCorrsImg1TPFromLast, delListCorrs, false);
			deleteMatEntriesByIdx(actCorrsImg2TPFromLast, delListCorrs, false);
		}

		//Add the seeds to their regions
		for (size_t i = 0; i < seedsNear_tmp1.size(); i++)
		{
			int32_t ix = seedsNear_tmp1[i].x / regSi.width;
			int32_t iy = seedsNear_tmp1[i].y / regSi.height;
			seedsNear[iy][ix].push_back(seedsNear_tmp1[i]);
		}
		
		//Add the seeds to their regions
		for (size_t i = 0; i < seedsMid_tmp1.size(); i++)
		{
			int32_t ix = seedsMid_tmp1[i].x / regSi.width;
			int32_t iy = seedsMid_tmp1[i].y / regSi.height;
			seedsMid[iy][ix].push_back(seedsMid_tmp1[i]);
		}

		//Add the seeds to their regions
		for (size_t i = 0; i < seedsFar_tmp1.size(); i++)
		{
			int32_t ix = seedsFar_tmp1[i].x / regSi.width;
			int32_t iy = seedsFar_tmp1[i].y / regSi.height;
			seedsFar[iy][ix].push_back(seedsFar_tmp1[i]);
		}
	}

	//Generate new seeds
	Point3_<int32_t> pt;
	pt.z = -1;
	for (int32_t y = 0; y < 3; y++)
	{
		int32_t mmy[2];
		mmy[0] = y * regSi.height;
		mmy[1] = mmy[0] + regSi.height - 1;
		std::uniform_int_distribution<int32_t> distributionY(mmy[0], mmy[1]);
		for (int32_t x = 0; x < 3; x++)
		{				
			int32_t mmx[2];
			mmx[0] = x * regSi.width;
			mmx[1] = mmx[0] + regSi.width - 1;
			std::uniform_int_distribution<int32_t> distributionX(mmx[0], mmx[1]);
			int32_t diffNr = nrDepthAreasPRegNear[actCorrsPRIdx].at<int32_t>(y, x) - (int32_t)seedsNear[y][x].size();
			while (diffNr > 0)//Generate seeds for near depth areas
			{
				pt.x = distributionX(rand_gen);
				pt.y = distributionY(rand_gen);
				Mat s_tmp = filtInitPts(Range(pt.y, pt.y + sqrSi1), Range(pt.x, pt.x + sqrSi1));
				if (s_tmp.at<unsigned char>(posadd1, posadd1) > 0)
				{
					continue;
				}
				else
				{
					csurr1.copyTo(s_tmp);
					seedsNear[y][x].push_back(pt);
					diffNr--;
				}
			}
			diffNr = nrDepthAreasPRegMid[actCorrsPRIdx].at<int32_t>(y, x) - (int32_t)seedsMid[y][x].size();
			while (diffNr > 0)//Generate seeds for mid depth areas
			{
				pt.x = distributionX(rand_gen);
				pt.y = distributionY(rand_gen);
				Mat s_tmp = filtInitPts(Range(pt.y, pt.y + sqrSi1), Range(pt.x, pt.x + sqrSi1));
				if (s_tmp.at<unsigned char>(posadd1, posadd1) > 0)
				{
					continue;
				}
				else
				{
					csurr1.copyTo(s_tmp);
					seedsMid[y][x].push_back(pt);
					diffNr--;
				}
			}
			diffNr = nrDepthAreasPRegFar[actCorrsPRIdx].at<int32_t>(y, x) - (int32_t)seedsFar[y][x].size();
			while (diffNr > 0)//Generate seeds for far depth areas
			{
				pt.x = distributionX(rand_gen);
				pt.y = distributionY(rand_gen);
				Mat s_tmp = filtInitPts(Range(pt.y, pt.y + sqrSi1), Range(pt.x, pt.x + sqrSi1));
				if (s_tmp.at<unsigned char>(posadd1, posadd1) > 0)
				{
					continue;
				}
				else
				{
					csurr1.copyTo(s_tmp);
					seedsFar[y][x].push_back(pt);
					diffNr--;
				}
			}
		}
	}
}

//Deletes some entries within a vector using a vector with indices that point to the entries to delete.
//The deletion vector containing the indices must be sorted in ascending order.
template<typename T, typename A, typename T1, typename A1>
void deleteVecEntriesbyIdx(std::vector<T, A> &editVec, std::vector<T1, A1> const& delVec)
{
	size_t nrToDel = delVec.size();
	CV_Assert(nrToDel <= editVec.size());
	size_t n_new = editVec.size() - nrToDel;
	std::vector<T, A> editVecNew(n_new);
	T1 old_idx = 0;
	int startRowNew = 0;
	for (size_t i = 0; i < nrToDel; i++)
	{
		if (old_idx == delVec[i])
		{
			old_idx = delVec[i] + 1;
			continue;
		}
		const int nr_new_cpy_elements = (int)delVec[i] - (int)old_idx;
		const int endRowNew = startRowNew + nr_new_cpy_elements;
		std::copy(editVec.begin() + old_idx, editVec.begin() + delVec[i], editVecNew.begin() + startRowNew);

		startRowNew = endRowNew;
		old_idx = delVec[i] + 1;
	}
	if (old_idx < editVec.size())
	{
		std::copy(editVec.begin() + old_idx, editVec.end(), editVecNew.begin() + startRowNew);
	}
	editVec = editVecNew;
}

//Deletes some entries within a Mat using a vector with indices that point to the entries to delete.
//The deletion vector containing the indices must be sorted in ascending order.
//It can be specified if the entries in Mat are colum ordered (rowOrder=false) or row ordered (rowOrder=true).
template<typename T, typename A>
void deleteMatEntriesByIdx(cv::Mat &editMat, std::vector<T, A> const& delVec, bool rowOrder)
{
	size_t nrToDel = delVec.size();
	size_t nrData;
	if (rowOrder)
		nrData = (size_t)editMat.rows;
	else
		nrData = (size_t)editMat.cols;

	CV_Assert(nrToDel <= nrData);

	size_t n_new = nrData - nrToDel;
	cv::Mat editMatNew;
	if (rowOrder)
		editMatNew = (n_new, editMat.cols, editMat.type());
	else
		editMatNew = (editMat.rows, n_new, editMat.type());

	T old_idx = 0;
	int startRowNew = 0;
	for (size_t i = 0; i < nrToDel; i++)
	{
		if (old_idx == delVec[i])
		{
			old_idx = delVec[i] + 1;
			continue;
		}
		const int nr_new_cpy_elements = (int)delVec[i] - (int)old_idx;
		const int endRowNew = startRowNew + nr_new_cpy_elements;
		if (rowOrder)
			editMat.rowRange((int)old_idx, (int)delVec[i]).copyTo(editMatNew.rowRange(startRowNew, endRowNew));
		else
			editMat.colRange((int)old_idx, (int)delVec[i]).copyTo(editMatNew.colRange(startRowNew, endRowNew));

		startRowNew = endRowNew;
		old_idx = delVec[i] + 1;
	}
	if ((size_t)old_idx < nrData)
	{
		if (rowOrder)
			editMat.rowRange((int)old_idx, editMat.rows).copyTo(editMatNew.rowRange(startRowNew, n_new));
		else
			editMat.colRange((int)old_idx, editMat.rows).copyTo(editMatNew.colRange(startRowNew, n_new));
	}
	editMatNew.copyTo(editMat);
}

//Wrapper function for function adaptIndicesNoDel
void genStereoSequ::adaptIndicesCVPtNoDel(std::vector<cv::Point3_<int32_t>> &seedVec, std::vector<size_t> &delListSortedAsc)
{
	std::vector<size_t> seedVecIdx;
	seedVecIdx.reserve(seedVec.size());
	for (auto sV : seedVec)
	{
		seedVecIdx.push_back((size_t)sV.z);
	}
	adaptIndicesNoDel(seedVecIdx, delListSortedAsc);
	for (size_t i = 0; i < seedVecIdx.size(); i++)
	{
		seedVec[i].z = seedVecIdx[i];
	}
}

//Adapt the indices of a not continious vector for which a part of the target data where the indices point to was deleted (no data points the indices point to were deleted).
void genStereoSequ::adaptIndicesNoDel(std::vector<size_t> &idxVec, std::vector<size_t> &delListSortedAsc)
{
	std::vector<pair<size_t, size_t>> idxVec_tmp(idxVec.size());
	for (size_t i = 0; i < idxVec.size(); i++)
	{
		idxVec_tmp[i] = make_pair(idxVec[i], i);
	}
	sort(idxVec_tmp.begin(), idxVec_tmp.end(), [](pair<size_t, size_t> first, pair<size_t, size_t> second) {return first.first < second.first; });
	size_t idx = 0;
	size_t maxIdx = delListSortedAsc.size() - 1;
	for (size_t i = 0; i < idxVec_tmp.size(); i++)
	{
		if (idxVec_tmp[i].first < delListSortedAsc[idx])
		{
			idxVec_tmp[i].first -= idx;
		}
		else
		{
			while ((idxVec_tmp[i].first > delListSortedAsc[idx]) && (idx < maxIdx))
			{
				idx++;
			}
			idxVec_tmp[i].first -= idx;
		}
	}
	sort(idxVec_tmp.begin(), idxVec_tmp.end(), [](pair<size_t, size_t> first, pair<size_t, size_t> second) {return first.second < second.second; });
	for (size_t i = 0; i < idxVec_tmp.size(); i++)
	{
		idxVec[i] = idxVec_tmp[i].first;
	}
}

//Initialize region ROIs and masks
void genStereoSequ::genRegMasks()
{
	//Construct valid areas for every region
	regmasks = vector<vector<Mat>>(3, vector<Mat>(3, Mat::zeros(imgSize, CV_8UC1)));
	regmasksROIs = vector<vector<cv::Rect>>(3, vector<cv::Rect>(3));
	regROIs = vector<vector<cv::Rect>>(3, vector<cv::Rect>(3));
	Size imgSi13 = Size(imgSize.width / 3, imgSize.height / 3);
	Mat validRect = Mat::ones(imgSize, CV_8UC1);
	const float overSi = 1.25f;//Allows the expension of created areas outside its region by a given percentage
	for (size_t y = 0; y < 3; y++)
	{
		cv::Point2i pl1, pr1, pl2, pr2;
		pl1.y = y * imgSi13.height;
		pl2.y = pl1.y;
		if (y < 2)
		{
			pr1.y = pl1.y + (int)(overSi * (float)imgSi13.height);
			pr2.y = pl2.y + imgSi13.height;
		}
		else
		{
			pr1.y = imgSize.height;
			pr2.y = imgSize.height;
		}
		if (y > 0)
		{
			pl1.y -= (int)((overSi - 1.f) * (float)imgSi13.height);
		}
		for (size_t x = 0; x < 3; x++)
		{
			pl1.x = x * imgSi13.width;
			pl2.x = pl1.x;
			if (x < 2)
			{
				pr1.x = pl1.x + (int)(overSi * (float)imgSi13.width);
				pr2.x = pl2.x + imgSi13.width;
			}
			else
			{
				pr1.x = imgSize.width;
				pr2.x = imgSize.width;
			}
			if (x > 0)
			{
				pl1.x -= (int)((overSi - 1.f) * (float)imgSi13.width);
			}
			Rect vROI = Rect(pl1, pr1);
			regmasksROIs[y][x] = vROI;
			regmasks[y][x](vROI) |= validRect(vROI);
			Rect vROIo = Rect(pl2, pr2);
			regROIs[y][x] = vROIo;
		}
	}
}

//Generates a depth map with the size of the image where each pixel value corresponds to the depth
void genStereoSequ::genDepthMaps()
{
	int minSi = (int)sqrt(minDArea);
	cv::Mat noGenMaskB = Mat::ones(imgSize.height + 2 * minSi, imgSize.width + 2 * minSi, CV_8UC1);
	Mat noGenMaskB2 = noGenMaskB.clone();
	cv::Mat noGenMask = noGenMaskB(Range(minSi, imgSize.height + minSi), Range(minSi, imgSize.width + minSi));
	Mat noGenMask2 = noGenMaskB2(Range(minSi, imgSize.height + minSi), Range(minSi, imgSize.width + minSi));
	minSi = 2 * minSi + 1;
	Mat minArea = Mat::zeros(minSi, minSi, CV_8UC1);

	//Get an ordering of the different depth area sizes for every region
	cv::Mat beginDepth = cv::Mat(3, 3, CV_32SC3);
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			int32_t maxAPReg[3];
			maxAPReg[0] = areaPRegNear[actCorrsPRIdx].at<int32_t>(y, x);
			maxAPReg[1] = areaPRegMid[actCorrsPRIdx].at<int32_t>(y, x);
			maxAPReg[2] = areaPRegFar[actCorrsPRIdx].at<int32_t>(y, x);
			std::ptrdiff_t pdiff = min_element(maxAPReg, maxAPReg + 3) - maxAPReg;
			beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[0] = pdiff;
			if (maxAPReg[(pdiff + 1) % 3] < maxAPReg[(pdiff + 2) % 3])
			{
				beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[1] = (pdiff + 1) % 3;
				beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[2] = (pdiff + 2) % 3;
			}
			else
			{
				beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[2] = (pdiff + 1) % 3;
				beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[1] = (pdiff + 2) % 3;
			}
		}
	}

	//Reserve a little bit of space for depth areas generated later on (as they are larger)
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			for (int i = 2; i >= 1; i--)
			{
				switch (beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[i])
				{
				case 0:
					for (auto pt : seedsNear[y][x])
					{
						Mat part;
						if(i == 2)
							part = noGenMaskB(Range(pt.y, pt.y + minSi), Range(pt.x, pt.x + minSi));
						else
							part = noGenMaskB2(Range(pt.y, pt.y + minSi), Range(pt.x, pt.x + minSi));
						part &= minArea;
					}
					break;
				case 1:
					for (auto pt : seedsMid[y][x])
					{
						Mat part;
						if (i == 2)
							part = noGenMaskB(Range(pt.y, pt.y + minSi), Range(pt.x, pt.x + minSi));
						else
							part = noGenMaskB2(Range(pt.y, pt.y + minSi), Range(pt.x, pt.x + minSi));
						part &= minArea;
					}
					break;
				case 2:
					for (auto pt : seedsFar[y][x])
					{
						Mat part;
						if (i == 2)
							part = noGenMaskB(Range(pt.y, pt.y + minSi), Range(pt.x, pt.x + minSi));
						else
							part = noGenMaskB2(Range(pt.y, pt.y + minSi), Range(pt.x, pt.x + minSi));
						part &= minArea;
					}
					break;
				default:
					break;
				}
			}
			noGenMaskB &= noGenMaskB2;
		}
	}

	//Create first layer of depth areas
	std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> actPosSeedsNear(3, std::vector<std::vector<cv::Point_<int32_t>>>(3));
	std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> actPosSeedsMid(3, std::vector<std::vector<cv::Point_<int32_t>>>(3));
	std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> actPosSeedsFar(3, std::vector<std::vector<cv::Point_<int32_t>>>(3));
	std::vector<std::vector<std::vector<size_t>>> nrIterPerSeedNear(3, std::vector<std::vector<size_t>>(3));
	std::vector<std::vector<std::vector<size_t>>> nrIterPerSeedMid(3, std::vector<std::vector<size_t>>(3));
	std::vector<std::vector<std::vector<size_t>>> nrIterPerSeedFar(3, std::vector<std::vector<size_t>>(3));
	/*std::vector<std::vector<std::vector<int32_t>>> areaPerSeedNear(3, std::vector<std::vector<int32_t>>(3));
	std::vector<std::vector<std::vector<int32_t>>> areaPerSeedMid(3, std::vector<std::vector<int32_t>>(3));
	std::vector<std::vector<std::vector<int32_t>>> areaPerSeedFar(3, std::vector<std::vector<int32_t>>(3));*/
	std::vector<std::vector<int32_t>> actAreaNear(3, vector<int32_t>(3, 0));
	std::vector<std::vector<int32_t>> actAreaMid(3, vector<int32_t>(3, 0));
	std::vector<std::vector<int32_t>> actAreaFar(3, vector<int32_t>(3, 0));
	std::vector<std::vector<unsigned char>> dilateOpNear(3, vector<unsigned char>(3, 0));
	std::vector<std::vector<unsigned char>> dilateOpMid(3, vector<unsigned char>(3, 0));
	std::vector<std::vector<unsigned char>> dilateOpFar(3, vector<unsigned char>(3, 0));
	depthAreaMap = Mat::zeros(imgSize, CV_8UC1);
	Mat actUsedAreaNear = Mat::zeros(imgSize, CV_8UC1);
	Mat actUsedAreaMid = Mat::zeros(imgSize, CV_8UC1);
	Mat actUsedAreaFar = Mat::zeros(imgSize, CV_8UC1);
	//Init actual positions
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			if (!seedsNear[y][x].empty())
			{
				actPosSeedsNear[y][x].resize(seedsNear[y][x].size());
				nrIterPerSeedNear[y][x].resize(seedsNear[y][x].size(), 0);
				//areaPerSeedNear[y][x].resize(seedsNear[y][x].size(), 0);
				for (size_t i = 0; i < seedsNear[y][x].size(); i++)
				{
					int ix = seedsNear[y][x][i].x;
					int iy = seedsNear[y][x][i].y;
					actPosSeedsNear[y][x][i].x = ix;
					actPosSeedsNear[y][x][i].y = iy;
					depthAreaMap.at<unsigned char>(iy, ix) = 1;
					actUsedAreaNear.at<unsigned char>(iy, ix) = 1;
					actAreaNear[y][x]++;
				}
			}
			if (!seedsMid[y][x].empty())
			{
				actPosSeedsMid[y][x].resize(seedsMid[y][x].size());
				nrIterPerSeedMid[y][x].resize(seedsMid[y][x].size(), 0);
				//areaPerSeedMid[y][x].resize(seedsMid[y][x].size(), 0);
				for (size_t i = 0; i < seedsMid[y][x].size(); i++)
				{
					int ix = seedsMid[y][x][i].x;
					int iy = seedsMid[y][x][i].y;
					actPosSeedsMid[y][x][i].x = ix;
					actPosSeedsMid[y][x][i].y = iy;
					depthAreaMap.at<unsigned char>(iy, ix) = 2;
					actUsedAreaMid.at<unsigned char>(iy, ix) = 2;
					actAreaMid[y][x]++;
				}
			}
			if (!seedsFar[y][x].empty())
			{
				actPosSeedsFar[y][x].resize(seedsFar[y][x].size());
				nrIterPerSeedFar[y][x].resize(seedsFar[y][x].size(), 0);
				//areaPerSeedFar[y][x].resize(seedsFar[y][x].size(), 0);
				for (size_t i = 0; i < seedsFar[y][x].size(); i++)
				{
					int ix = seedsFar[y][x][i].x;
					int iy = seedsFar[y][x][i].y;
					actPosSeedsFar[y][x][i].x = ix;
					actPosSeedsFar[y][x][i].y = iy;
					depthAreaMap.at<unsigned char>(iy, ix) = 3;
					actUsedAreaFar.at<unsigned char>(iy, ix) = 3;
					actAreaFar[y][x]++;
				}
			}
		}
	}

	//Create depth areas beginning with the smallest areas (near, mid, or far) per region
	//Also create depth areas for the second smallest areas
	Size imgSiM1 = Size(imgSize.width - 1, imgSize.height - 1);
	for (int j = 0; j < 2; j++)
	{
		if (j > 0)
		{
			noGenMask = noGenMask2;
		}
		bool areasNFinish[3][3] = { true, true, true, true, true, true, true, true, true };
		while (areasNFinish[0][0] || areasNFinish[0][1] || areasNFinish[0][2] ||
			areasNFinish[1][0] || areasNFinish[1][1] || areasNFinish[1][2] ||
			areasNFinish[2][0] || areasNFinish[2][1] || areasNFinish[2][2])
		{
			for (size_t y = 0; y < 3; y++)
			{
				for (size_t x = 0; x < 3; x++)
				{
					if (!areasNFinish[y][x]) continue;
					switch (beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[j])
					{
					case 0:
						if (!actPosSeedsNear[y][x].empty())
						{
							for (size_t i = 0; i < actPosSeedsNear[y][x].size(); i++)
							{
								areasNFinish[y][x] = addAdditionalDepth(1,
									depthAreaMap,
									actUsedAreaNear,
									noGenMask,
									regmasks[y][x],
									actPosSeedsNear[y][x][i],
									actPosSeedsNear[y][x][i],
									actAreaNear[y][x],
									areaPRegNear[actCorrsPRIdx].at<int32_t>(y, x),
									imgSiM1,
									cv::Point_<int32_t>(seedsNear[y][x][i].x, seedsNear[y][x][i].y),
									regmasksROIs[y][x],
									nrIterPerSeedNear[y][x][i],
									dilateOpNear[y][x]);
							}
						}
						else
							areasNFinish[y][x] = false;
						break;
					case 1:
						if (!actPosSeedsMid[y][x].empty())
						{
							for (size_t i = 0; i < actPosSeedsMid[y][x].size(); i++)
							{
								areasNFinish[y][x] = addAdditionalDepth(2,
									depthAreaMap,
									actUsedAreaMid,
									noGenMask,
									regmasks[y][x],
									actPosSeedsMid[y][x][i],
									actPosSeedsMid[y][x][i],
									actAreaMid[y][x],
									areaPRegMid[actCorrsPRIdx].at<int32_t>(y, x),
									imgSiM1,
									cv::Point_<int32_t>(seedsMid[y][x][i].x, seedsMid[y][x][i].y),
									regmasksROIs[y][x],
									nrIterPerSeedMid[y][x][i],
									dilateOpMid[y][x]);
							}
						}
						else
							areasNFinish[y][x] = false;
						break;
					case 2:
						if (!actPosSeedsFar[y][x].empty())
						{
							for (size_t i = 0; i < actPosSeedsFar[y][x].size(); i++)
							{
								areasNFinish[y][x] = addAdditionalDepth(3,
									depthAreaMap,
									actUsedAreaFar,
									noGenMask,
									regmasks[y][x],
									actPosSeedsFar[y][x][i],
									actPosSeedsFar[y][x][i],
									actAreaFar[y][x],
									areaPRegFar[actCorrsPRIdx].at<int32_t>(y, x),
									imgSiM1,
									cv::Point_<int32_t>(seedsFar[y][x][i].x, seedsFar[y][x][i].y),
									regmasksROIs[y][x],
									nrIterPerSeedFar[y][x][i],
									dilateOpFar[y][x]);
							}
						}
						else
							areasNFinish[y][x] = false;
						break;
					default:
						break;
					}
				}
			}
		}
	}

	//Fill the remaining areas
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			Mat fillMask = (depthAreaMap(regROIs[y][x]) == 0) & Mat::ones(regROIs[y][x].height, regROIs[y][x].width, CV_8UC1);
			switch (beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[2])
			{
			case 0:
				actUsedAreaNear(regROIs[y][x]) |= fillMask;
				//depthAreaMap(regROIs[y][x]) |= fillMask;
				break;
			case 1:
				actUsedAreaMid(regROIs[y][x]) |= fillMask;
				/*fillMask *= 2;
				depthAreaMap(regROIs[y][x]) |= fillMask;*/
				break;
			case 2:
				actUsedAreaMid(regROIs[y][x]) |= fillMask;
				/*fillMask *= 3;
				depthAreaMap(regROIs[y][x]) |= fillMask;*/
				break;
			default:
				break;
			}
		}
	}

	//Get final depth values for each depth region
	Mat depthMapNear, depthMapMid, depthMapFar;
	getDepthMaps(depthMapNear, actUsedAreaNear, actDepthNear, actDepthMid, seedsNear, 0);
	getDepthMaps(depthMapMid, actUsedAreaMid, actDepthMid, actDepthFar, seedsMid, 1);
	getDepthMaps(depthMapFar, actUsedAreaFar, actDepthFar, maxFarDistMultiplier * actDepthFar, seedsFar, 2);

	//Combine the 3 depth maps to a single depth map
	depthMap = depthMapNear + depthMapMid + depthMapFar;
}

//Generate depth values (for every pixel) for the given areas of depth regions taking into account the depth values from backprojected 3D points
void genStereoSequ::getDepthMaps(cv::Mat &dout, cv::Mat &din, double dmin, double dmax, std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> &initSeeds, int dNr)
{
	std::vector<cv::Point3_<int32_t>> initSeedInArea;

	switch (dNr)
	{
	case 0:
		seedsNearFromLast = std::vector<std::vector<std::vector<cv::Point_<int32_t>>>>(3, std::vector<std::vector<cv::Point_<int32_t>>>(3));
			break;
	case 1:
		seedsMidFromLast = std::vector<std::vector<std::vector<cv::Point_<int32_t>>>>(3, std::vector<std::vector<cv::Point_<int32_t>>>(3));
		break;
	case 2:
		seedsFarFromLast = std::vector<std::vector<std::vector<cv::Point_<int32_t>>>>(3, std::vector<std::vector<cv::Point_<int32_t>>>(3));
		break;
	default:
		break;
	}

	//Check, if there are depth seeds available that were already backprojected from 3D
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			for (size_t i = 0; i < initSeeds[y][x].size(); i++)
			{
				if (initSeeds[y][x][i].z >= 0)
				{
					initSeedInArea.push_back(initSeeds[y][x][i]);
					switch (dNr)
					{
					case 0:
						seedsNearFromLast[y][x].push_back(cv::Point_<int32_t>(initSeedInArea.back().x, initSeedInArea.back().y));
						break;
					case 1:
						seedsMidFromLast[y][x].push_back(cv::Point_<int32_t>(initSeedInArea.back().x, initSeedInArea.back().y));
						break;
					case 2:
						seedsFarFromLast[y][x].push_back(cv::Point_<int32_t>(initSeedInArea.back().x, initSeedInArea.back().y));
						break;
					default:
						break;
					}
				}
			}
		}
	}
	getDepthVals(dout, din, dmin, dmax, initSeedInArea);
}

//Generate depth values (for every pixel) for the given areas of depth regions
void genStereoSequ::getDepthVals(cv::Mat &dout, cv::Mat &din, double dmin, double dmax, std::vector<cv::Point3_<int32_t>> &initSeedInArea)
{
	Mat actUsedAreaLabel;
	Mat actUsedAreaStats;
	Mat actUsedAreaCentroids;
	int nrLabels;
	vector<std::vector<double>> funcPars;
	uint16_t nL = 0;

	//Get connected areas
	nrLabels = connectedComponentsWithStats(din, actUsedAreaLabel, actUsedAreaStats, actUsedAreaCentroids, 4, CV_16U);
	nL = (uint16_t)(nrLabels + 1);
	getRandDepthFuncPars(funcPars, (size_t)nL);
	cv::ConnectedComponentsTypes::CC_STAT_HEIGHT;

	dout.release();
	dout = Mat::zeros(imgSize, CV_64FC1);
	for (uint16_t i = 0; i < nL; i++)
	{
		Rect labelBB = Rect(actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_LEFT),
			actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_TOP),
			actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_WIDTH),
			actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_HEIGHT));
		Mat laMat = actUsedAreaLabel(labelBB);
		Mat doutSlice = dout(labelBB);

		double dmin_tmp = getRandDoubleValRng(dmin, dmin + 0.6 * (dmax - dmin));
		double dmax_tmp = getRandDoubleValRng(dmin_tmp + 0.1 * (dmax - dmin), dmax);
		double drange = dmax_tmp - dmin_tmp;
		double rXr = getRandDoubleValRng(1.5, 3.0);
		double rYr = getRandDoubleValRng(1.5, 3.0);
		double h2 = (double)actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_HEIGHT);
		h2 *= h2;
		double w2 = (double)actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_WIDTH);
		w2 *= w2;
		double scale = sqrt(h2 + w2) / 2.0;
		double rXrSc = rXr / scale;
		double rYrSc = rYr / scale;
		double cx = actUsedAreaCentroids.at<double>(i, 0) - (double)actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_LEFT);
		double cy = actUsedAreaCentroids.at<double>(i, 1) - (double)actUsedAreaStats.at<int32_t>(i, cv::ConnectedComponentsTypes::CC_STAT_TOP);

		//If an initial seed was backprojected from 3D to this component, the depth range of the current component must be similar
		if (!initSeedInArea.empty())
		{
			int32_t minX = labelBB.x;
			int32_t maxX = minX + labelBB.width;
			int32_t minY = labelBB.y;
			int32_t maxY = minY + labelBB.height;
			vector<double> initDepths;
			for (size_t j = 0; j < initSeedInArea.size(); j++)
			{
				if ((initSeedInArea[j].x >= minX) && (initSeedInArea[j].x < maxX) &&
					(initSeedInArea[j].y >= minY) && (initSeedInArea[j].y < maxY))
				{
					if (actUsedAreaLabel.at<uint16_t>(initSeedInArea[j].y, initSeedInArea[j].x) == i)
					{
						initDepths.push_back(actImgPointCloudFromLast[initSeedInArea[j].z].z);
					}
				}
			}
			if (!initDepths.empty())
			{
				if (initDepths.size() == 1)
				{
					double tmp = getRandDoubleValRng(0.05, 0.5);
					dmin_tmp = initDepths[0] - tmp * (dmax - dmin);
					dmax_tmp = initDepths[0] + tmp * (dmax - dmin);
				}
				else
				{
					auto minMaxD = std::minmax_element(initDepths.begin(), initDepths.end());
					double range1 = *minMaxD.second - *minMaxD.first;
					if (range1 < 0.05 * (dmax - dmin))
					{
						double dmid_tmp = *minMaxD.first + range1 / 2.0;
						double tmp = getRandDoubleValRng(0.05, 0.5);
						dmin_tmp = dmid_tmp - tmp * (dmax - dmin);
						dmax_tmp = dmid_tmp + tmp * (dmax - dmin);
					}
					else
					{
						dmin_tmp = *minMaxD.first - range1 / 2.0;
						dmax_tmp = *minMaxD.second + range1 / 2.0;
					}
				}
				dmin_tmp = std::max(dmin_tmp, dmin);
				dmax_tmp = std::min(dmax_tmp, dmax);
				drange = dmax_tmp - dmin_tmp;
			}
		}

		double minVal = DBL_MAX, maxVal = -DBL_MAX;
		int32_t lareaCnt = 0, lareaNCnt = 2 * labelBB.width;
		for (int y = 0; y < labelBB.height; y++)
		{
			for (int x = 0; x < labelBB.width; x++)
			{
				if (laMat.at<uint16_t>(y, x) == i)
				{
					if (din.at<unsigned char>(y, x) == 0)
					{
						lareaNCnt--;
						if ((lareaCnt == 0) && (lareaNCnt < 0))
						{
							lareaCnt = -1;
							y = labelBB.height;
							break;
						}
						continue;
					}
					lareaCnt++;
					double val = getDepthFuncVal(funcPars[i], ((double)x - cx) * rXrSc, ((double)y - cy) * rYrSc);
					doutSlice.at<double>(y, x) = val;
					if (val > maxVal)
						maxVal = val;
					if (val < minVal)
						minVal = val;
				}
			}
		}
		if (lareaCnt > 0)
		{
			double ra = maxVal - minVal;
			scale = drange / ra;
			for (int y = 0; y < labelBB.height; y++)
			{
				for (int x = 0; x < labelBB.width; x++)
				{
					if (laMat.at<uint16_t>(y, x) == i)
					{
						double val = doutSlice.at<double>(y, x);
						val -= minVal;
						val *= scale;
						val += dmin_tmp;
						doutSlice.at<double>(y, x) = val;
					}
				}
			}
		}
	}
}

/*Calculates a depth value using the function
z = p1 * (p2 - x)^2 * e^(-x^2 - (y - p3)^2) - 10 * (x / p4 - x^p5 - y^p6) * e^(-x^2 - y^2) - p7 / 3 * e^(-(x + 1)^2 - y^2)
*/
inline double genStereoSequ::getDepthFuncVal(std::vector<double> &pars1, double x, double y)
{
	double tmp = pars1[1] - x;
	tmp *= tmp;
	double z = pars1[0] * tmp;
	tmp = y - pars1[2];
	tmp *= -tmp;
	tmp -= x * x;
	z *= exp(tmp);
	/*double tmp1[4];
	tmp1[0] = x / pars1[3];
    tmp1[1] = std::pow(x, pars1[4]);
    tmp1[2] = std::pow(y, pars1[5]);
    tmp1[3] = exp(-x * x - y * y);
    z -= 10.0 * (tmp1[0] - tmp1[1] - tmp1[2]) * tmp1[3];*/
    z -= 10.0 * (x / pars1[3] - std::pow(x, pars1[4]) - std::pow(y, pars1[5])) * exp(-x * x - y * y);
	tmp = x + 1.0;
	tmp *= -tmp;
	z -= pars1[6] / 3.0 * exp(tmp - y * y);
	return z;
}

/*Calculate random parameters for the function generating depth values
There are 7 random paramters p:
z = p1 * (p2 - x)^2 * e^(-x^2 - (y - p3)^2) - 10 * (x / p4 - x^p5 - y^p6) * e^(-x^2 - y^2) - p7 / 3 * e^(-(x + 1)^2 - y^2)
*/
void genStereoSequ::getRandDepthFuncPars(std::vector<std::vector<double>> &pars1, size_t n_pars)
{
	pars1 = std::vector<std::vector<double>>(n_pars, std::vector<double>(7, 0));

	//p1:
	std::uniform_real_distribution<double> distribution(0, 10.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars1[i][0] = distribution(rand_gen);
	}
	//p2:
	distribution = std::uniform_real_distribution<double>(0, 2.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars1[i][1] = distribution(rand_gen);
	}
	//p3:
	distribution = std::uniform_real_distribution<double>(0, 4.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars1[i][2] = distribution(rand_gen);
	}
	//p4:
	distribution = std::uniform_real_distribution<double>(0.5, 5.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars1[i][3] = distribution(rand_gen);
	}
	//p5 & p6:
	distribution = std::uniform_real_distribution<double>(2.0, 7.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars1[i][4] = round(distribution(rand_gen));
		pars1[i][5] = round(distribution(rand_gen));
	}
	//p7:
	distribution = std::uniform_real_distribution<double>(1.0, 40.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars1[i][6] = distribution(rand_gen);
	}
}

/* Adds a few random depth pixels near a given position (no actual depth value, but a part of a mask indicating the depth range (near, mid, far)
unsigned char pixVal	In: Value assigned to the random pixel positions
cv::Mat &imgD			In/Out: Image holding all depth ranges where the new random depth pixels should be added
cv::Mat &imgSD			In/Out: Image holding only one specific depth range where the new random depth pixels should be added
cv::Mat &mask			In: Mask for imgD and imgSD (marks backprojected moving objects (with a 1))
cv::Point_<int32_t> &startpos		In: Start position (excluding this single location) from where to start adding new depth pixels
cv::Point_<int32_t> &endpos			Out: End position where the last depth pixel was set
int32_t &addArea		In/Out: Adds the number of newly inserted pixels to the given number
int32_t &maxAreaReg		In: Maximum number of specific depth pixels per image region (9x9)
cv::Size &siM1			In: Image size -1
cv::Point_<int32_t> &initSeed	In: Initial position of the seed
cv::Rect &vROI			In: ROI were it is actually allowed to add new pixels
size_t &nrAdds			In/Out: Number of times this function was called for this depth area (including preceding calls to this function)
*/
bool genStereoSequ::addAdditionalDepth(unsigned char pixVal, 
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
	unsigned char &usedDilate)
{
	const size_t max_iter = 10000;
	const size_t midDilateCnt = 300;
    //get possible directions for expanding (max. 8 possibilities) by checking the masks
	vector<int32_t> directions;
	if((nrAdds <= max_iter) && !usedDilate && ((nrAdds % midDilateCnt != 0) || (nrAdds < midDilateCnt)))
    {
        directions = getPossibleDirections(startpos, mask, regMask, imgD, siM1, imgSD, true);
    }
	if (directions.empty() || (nrAdds > max_iter) || usedDilate || ((nrAdds % midDilateCnt == 0) && (nrAdds >= midDilateCnt)))//Dilate the label if no direction was found or there were already to many iterations
	{
		int strElmSi = 3, cnt = 0, maxCnt = 20, strElmSiAdd = 0, strElmSiDir[2] = {0, 0};
        int32_t siAfterDil = 0;
		while ((siAfterDil == 0) && (cnt < maxCnt))
		{
			cnt++;
			Mat element;
            int elSel = rand() % 3;
            strElmSiAdd = rand() % 3;
            strElmSiDir[0] = rand() % 2;
            strElmSiDir[1] = rand() % 2;
            switch (elSel)
            {
                case 0:
                    element = cv::getStructuringElement(MORPH_ELLIPSE, Size(strElmSi + strElmSiDir[0] * strElmSiAdd, strElmSi + strElmSiDir[1] * strElmSiAdd));
                    break;
                case 1:
                    element = cv::getStructuringElement(MORPH_RECT, Size(strElmSi + strElmSiDir[0] * strElmSiAdd, strElmSi + strElmSiDir[1] * strElmSiAdd));
                    break;
                case 2:
                    element = cv::getStructuringElement(MORPH_CROSS, Size(strElmSi + strElmSiDir[0] * strElmSiAdd, strElmSi + strElmSiDir[1] * strElmSiAdd));
                    break;
                default:
                    element = cv::getStructuringElement(MORPH_ELLIPSE, Size(strElmSi, strElmSi));
                    break;
            }


			strElmSi++;
			Mat imgSDdilate;

            /*Mat dilImgTh2;
            cv::threshold( imgSD(vROI), dilImgTh2, 0, 255,0 );
            namedWindow( "Dilated2", WINDOW_AUTOSIZE );
            imshow("Dilated2", dilImgTh2);
            waitKey(0);
            destroyWindow("Dilated2");*/

			dilate(imgSD(vROI), imgSDdilate, element);
//			imgSDdilate &= mask(vROI) & ((imgD(vROI) == 0) | imgSD(vROI));
            imgSDdilate &= (mask(vROI) == 0) & ((imgD(vROI) == 0) | imgSD(vROI));

            /*Mat dilImgTh3;
            cv::threshold( imgSDdilate, dilImgTh3, 0, 255,0 );
            namedWindow( "Dilated3", WINDOW_AUTOSIZE );
            imshow("Dilated3", dilImgTh3);
            waitKey(0);
            destroyWindow("Dilated3");*/

			siAfterDil = (int32_t)sum(imgSDdilate)[0];
            static size_t visualizeMask = 0;
            if(visualizeMask % 50 == 0)
            {
                Mat colorMapImg;
                // Apply the colormap:
                applyColorMap(imgD * 20, colorMapImg, cv::COLORMAP_RAINBOW);
                namedWindow( "combined ObjLabels1", WINDOW_AUTOSIZE );
                imshow("combined ObjLabels1", colorMapImg);

                Mat dilImgTh;
                cv::threshold( imgSDdilate, dilImgTh, 0, 255,0 );
                namedWindow( "Dilated1", WINDOW_AUTOSIZE );
                imshow("Dilated1", dilImgTh);

                Mat onlyDil = (imgSDdilate ^ imgSD(vROI)) * 20 + imgSD(vROI);
                applyColorMap(onlyDil, colorMapImg, cv::COLORMAP_HOT);
                namedWindow( "Dilated", WINDOW_AUTOSIZE );
                imshow("Dilated", onlyDil);

                waitKey(0);
                destroyWindow("combined ObjLabels1");
                destroyWindow("Dilated");
                destroyWindow("Dilated1");
            }
            visualizeMask++;

			if (siAfterDil > maxAreaReg)
			{
				int32_t diff = siAfterDil - maxAreaReg;
				imgSDdilate ^= imgSD(vROI);
				for (int y = 0; y < vROI.height; y++)
				{
					for (int x = 0; x < vROI.width; x++)
					{
						if (imgSDdilate.at<unsigned char>(y, x))
						{
							imgSDdilate.at<unsigned char>(y, x) = 0;
							diff--;
							if (diff <= 0)
								break;
						}
					}
					if (diff <= 0)
						break;
				}
				imgSD(vROI) |= imgSDdilate;
				imgSDdilate *= pixVal;
				imgD(vROI) |= imgSDdilate;
				addArea = maxAreaReg;
				nrAdds++;
				usedDilate = 1;

				return false;
			}
			else if (siAfterDil > 0)
			{
				imgSDdilate.copyTo(imgSD(vROI));
                imgD(vROI) &= (imgSDdilate == 0);
				imgSDdilate *= pixVal;
                imgD(vROI) |= imgSDdilate;
                //imgSDdilate.copyTo(imgD(vROI));
				if((directions.empty() && ((nrAdds % midDilateCnt != 0) || (nrAdds < midDilateCnt))) || (nrAdds > max_iter)) {
                    usedDilate = 1;
                }
                nrAdds++;
				if(addArea == siAfterDil)
                {
				    return false;
                }
                addArea = siAfterDil;

				return true;
			}
		}
		if (cnt >= maxCnt)
		{
			return false;
		}
	}
	else
	{
		//Get a random direction where to add a pixel
		int diri = rand() % (int)directions.size();
		endpos = startpos;
        nextPosition(endpos, directions[diri]);
		//Set the pixel
		imgD.at<unsigned char>(endpos) = pixVal;
		imgSD.at<unsigned char>(endpos) = 1;
		addArea++;
		nrAdds++;
		if (addArea >= maxAreaReg)
		{
			return false;
		}
		//Add additional pixels in the local neighbourhood (other possible directions) of the actual added pixel
		//and prevent adding new pixels in similar directions compared to the added one
		vector<int32_t> extension = getPossibleDirections(endpos, mask, regMask, imgD, siM1, imgSD, false);
		if (extension.size() > 1)//Check if we can add addition pixels without blocking the way for the next iteration
		{
            //Prevent adding additional pixels to the main direction and its immediate neighbor directions
			int32_t noExt[3];
			noExt[0] = (directions[diri] + 1) % 8;
			noExt[1] = directions[diri];
			noExt[2] = (directions[diri] + 7) % 8;
            //vector<int32_t>::reverse_iterator itr = extension.rbegin();
            for(vector<int32_t>::reverse_iterator itr = extension.rbegin(); itr != extension.rend(); itr++)
            {
                if ((*itr == noExt[0]) ||
                    (*itr == noExt[1]) ||
                    (*itr == noExt[2]))
                {
                    extension.erase(std::next(itr).base());
                }
            }
			/*for (int i = extension.size()-1; i >= 0; i--)
			{
				if ((extension[i] == noExt[0]) ||
					(extension[i] == noExt[1]) || 
					(extension[i] == noExt[2]))
				{
					extension.erase(extension.begin() + i);
					if(i > 0)
						i++;
				}
			}*/
			if (extension.size() > 1)
			{
				//Choose a random number of additional pixels to add (based on possible directions of expansion)
				int addsi = rand() % ((int)extension.size() + 1);
				if (addsi)
				{
					if ((addsi + addArea) > maxAreaReg)
					{
						addsi = maxAreaReg - addArea;
					}
					const int beginExt = rand() % (int)extension.size();
					for (int i = 0; i < addsi; i++)
					{
						cv::Point_<int32_t> singleExt = endpos;
						const int pos = (beginExt + i) % (int)extension.size();
                        nextPosition(singleExt, extension[pos]);
						//Set the pixel
						imgD.at<unsigned char>(singleExt) = pixVal;
						imgSD.at<unsigned char>(singleExt) = 1;
						addArea++;
					}
				}
				if (addArea >= maxAreaReg)
				{
					return false;
				}
			}
		}
	}

	return true;
}

//Get valid directions to expand the depth area given a start position
std::vector<int32_t> genStereoSequ::getPossibleDirections(cv::Point_<int32_t> &startpos, cv::Mat &mask, cv::Mat &regMask, cv::Mat &imgD, cv::Size &siM1, cv::Mat &imgSD, bool escArea)
{
	static int maxFixDirChange = 8;
	int fixDirChange = 0;
    Mat directions;
	unsigned char atBorderX = 0, atBorderY = 0;
    Mat directions_dist;
    vector<int32_t> dirs;
    int32_t fixedDir = 0;
    bool dirFixed = false;
    bool inOwnArea = false;
    do {
        directions = Mat::ones(3, 3, CV_8UC1);
        if (startpos.x <= 0) {
            directions.col(0) = Mat::zeros(3, 1, CV_8UC1);
            atBorderX = 0x1;
        }
        if (startpos.x >= siM1.width) {
            directions.col(2) = Mat::zeros(3, 1, CV_8UC1);
            atBorderX = 0x2;
        }
        if (startpos.y <= 0) {
            directions.row(0) = Mat::zeros(1, 3, CV_8UC1);
            atBorderY = 0x1;
        }
        if (startpos.y >= siM1.height) {
            directions.row(2) = Mat::zeros(1, 3, CV_8UC1);
            atBorderY = 0x2;
        }

        Range irx, iry, drx, dry;
        if (atBorderX) {
            const unsigned char atBorderXn = ~atBorderX;
            const unsigned char v1 = (atBorderXn & 0x1);//results in 0x0 (for atBorderX=0x1) or 0x1 (for atBorderX=0x2)
            const unsigned char v2 = (atBorderXn & 0x2) + ((atBorderX & 0x2)
                    >> 1);//results in 0x2 (for atBorderX=0x1) or 0x1 (for atBorderX=0x2)
            irx = Range(startpos.x - (int32_t) v1, startpos.x + (int32_t) v2);
            drx = Range((int32_t) (~v1 & 0x1), 1 + (int32_t) v2);
            if (atBorderY) {
                const unsigned char atBorderYn = ~atBorderY;
                const unsigned char v3 = (atBorderYn & 0x1);
                const unsigned char v4 = (atBorderYn & 0x2) + ((atBorderY & 0x2) >> 1);
                iry = Range(startpos.y - (int32_t) v3, startpos.y + (int32_t) v4);
                dry = Range((int32_t) (~v3 & 0x1), 1 + (int32_t) v4);
            } else {
                iry = Range(startpos.y - 1, startpos.y + 2);
                dry = Range::all();
            }
        } else if (atBorderY) {
            unsigned char atBorderYn = ~atBorderY;
            const unsigned char v3 = (atBorderYn & 0x1);
            const unsigned char v4 = (atBorderYn & 0x2) + ((atBorderY & 0x2) >> 1);
            iry = Range(startpos.y - (int32_t) v3, startpos.y + (int32_t) v4);
            irx = Range(startpos.x - 1, startpos.x + 2);
            drx = Range::all();
            dry = Range((int32_t) (~v3 & 0x1), 1 + (int32_t) v4);
        } else {
            irx = Range(startpos.x - 1, startpos.x + 2);
            iry = Range(startpos.y - 1, startpos.y + 2);
            drx = Range::all();
            dry = Range::all();
        }


        directions.copyTo(directions_dist);
        directions(dry, drx) &= (imgD(iry, irx) == 0) & (mask(iry, irx) == 0) & regMask(iry, irx);
        if ((sum(directions)[0] == 0) && escArea) {
            directions_dist(dry, drx) &=
                    ((imgD(iry, irx) == 0) | imgSD(iry, irx)) & (mask(iry, irx) == 0) & regMask(iry, irx);
            if (sum(directions_dist)[0] != 0) {
                if(!dirFixed) {
                    directions_dist.copyTo(directions);
                    inOwnArea = true;
                } else
                {
                    cv::Point_<int32_t> localPos = cv::Point_<int32_t>(1, 1);
                    nextPosition(localPos, fixedDir);
                    if(directions_dist.at<unsigned char>(localPos) == 0)
                    {
                        if(fixDirChange > maxFixDirChange) {
                            inOwnArea = false;
                            dirFixed = false;
                        } else
                        {
                            inOwnArea = true;
                            dirFixed = false;
                            directions_dist.copyTo(directions);
                        }
                        fixDirChange++;
                    }
                }
            } else {
                inOwnArea = false;
                dirFixed = false;
            }
        } else
        {
            dirFixed = false;
            inOwnArea = false;
        }

        if(!dirFixed) {
            dirs.clear();
            for (int32_t i = 0; i < 9; i++) {
                if (directions.at<bool>(i)) {
                    switch (i) {
                        case 0:
                            dirs.push_back(0);
                            break;
                        case 1:
                            dirs.push_back(1);
                            break;
                        case 2:
                            dirs.push_back(2);
                            break;
                        case 3:
                            dirs.push_back(7);
                            break;
                        case 5:
                            dirs.push_back(3);
                            break;
                        case 6:
                            dirs.push_back(6);
                            break;
                        case 7:
                            dirs.push_back(5);
                            break;
                        case 8:
                            dirs.push_back(4);
                            break;
                        default:
                            break;
                    }
                }
            }
        }

        if(inOwnArea && !dirs.empty())
        {
            if(!dirFixed) {
                if (dirs.size() == 1) {
                    fixedDir = dirs[0];
                } else {
                    //Get a random direction where to go next
                    fixedDir = dirs[rand() % (int) dirs.size()];
                }
                dirFixed = true;
            }
            nextPosition(startpos, fixedDir);
        }
    }
    while(inOwnArea);

	return dirs;
}

void genStereoSequ::nextPosition(cv::Point_<int32_t> &position, int32_t direction)
{
    switch (direction)
    {
        case 0://direction left up
            position.x--;
            position.y--;
            break;
        case 1://direction up
            position.y--;
            break;
        case 2://direction right up
            position.x++;
            position.y--;
            break;
        case 3://direction right
            position.x++;
            break;
        case 4://direction right down
            position.x++;
            position.y++;
            break;
        case 5://direction down
            position.y++;
            break;
        case 6://direction left down
            position.x--;
            position.y++;
            break;
        case 7://direction left
            position.x--;
            break;
        default:
            break;
    }
}

//Generates correspondences and 3D points in the camera coordinate system (including false matches) from static scene elements
void genStereoSequ::getKeypoints()
{
	int32_t kSi = csurr.rows;
	int32_t posadd = (kSi - 1) / 2;
	
	//Mark used areas (by correspondences, TN, and moving objects) in the second image
	Mat cImg2 = Mat::zeros(imgSize.width + kSi - 1, imgSize.height + kSi - 1, CV_8UC1);
	for (int i = 0; i < actCorrsImg2TPFromLast.cols; i++)
	{
		Point_<int32_t> pt((int32_t)round(actCorrsImg2TPFromLast.at<double>(0, i)), (int32_t)round(actCorrsImg2TPFromLast.at<double>(1, i)));
		Mat s_tmp = cImg2(Rect(pt, Size(kSi, kSi)));
		csurr.copyTo(s_tmp);
	}
	const int nrBPTN2 = actCorrsImg2TNFromLast.cols;
	int nrBPTN2cnt = 0;
	for (int i = 0; i < nrBPTN2; i++)
	{
		Point_<int32_t> pt((int32_t)round(actCorrsImg2TNFromLast.at<double>(0, i)), (int32_t)round(actCorrsImg2TNFromLast.at<double>(1, i)));
		Mat s_tmp = cImg2(Rect(pt, Size(kSi, kSi)));
		csurr.copyTo(s_tmp);
	}
	cImg2(Rect(Point(posadd, posadd), imgSize)) |= movObjMask2All;

	//Get regions of backprojected TN in first image and mark their positions; add true negatives from backprojection to the new outlier data
	vector<vector<vector<Point_<int32_t>>>> x1pTN(3, vector<vector<Point_<int32_t>>>(3));
	Size rSl(imgSize.width / 3, imgSize.height / 3);
	for (int i = 0; i < actCorrsImg1TNFromLast.cols; i++)
	{
		Point_<int32_t> pt((int32_t)round(actCorrsImg1TNFromLast.at<double>(0, i)), (int32_t)round(actCorrsImg1TNFromLast.at<double>(1, i)));
		Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
		csurr.copyTo(s_tmp);

		x1pTN[pt.y / rSl.height][pt.x / rSl.width].push_back(pt);
	}

	vector<vector<vector<Point_<int32_t>>>> corrsAllD(3, vector<vector<Point_<int32_t>>>(3));
	vector<vector<vector<Point2d>>> corrsAllD2(3, vector<vector<Point2d>>(3));
	Point_<int32_t> pt;
	Point2d pt2;
	Point3d pCam;
	vector<vector<vector<Point3d>>> p3DTPnew(3, vector<vector<Point3d>>(3));
	vector<vector<vector<Point2d>>> x1TN(3, vector<vector<Point2d>>(3));
	vector<vector<vector<Point2d>>> x2TN(3, vector<vector<Point2d>>(3));
	vector<vector<vector<double>>> x2TNdistCorr(3, vector<vector<double>>(3));
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			if (movObjHasArea[y][x])
				continue;

			int32_t nrNear = (int32_t)floor(depthsPerRegion[actCorrsPRIdx][y][x].near * (double)nrTruePosRegs[actFrameCnt].at<int32_t>(y, x));
			int32_t nrFar = (int32_t)floor(depthsPerRegion[actCorrsPRIdx][y][x].far * (double)nrTruePosRegs[actFrameCnt].at<int32_t>(y, x));
			int32_t nrMid = nrTruePosRegs[actFrameCnt].at<int32_t>(y, x) - nrNear - nrFar;

			int32_t nrTN = nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x) - (int32_t)x1pTN[y][x].size();

			int32_t maxSelect = max(3 * nrTruePosRegs[actFrameCnt].at<int32_t>(y, x), 1000);
			int32_t maxSelect2 = 50;
			int32_t maxSelect3 = 50;
			int32_t maxSelect4 = 50;
			std::uniform_int_distribution<int32_t> distributionX(regROIs[y][x].x, regROIs[y][x].x + regROIs[y][x].width - 1);
			std::uniform_int_distribution<int32_t> distributionY(regROIs[y][x].y, regROIs[y][x].y + regROIs[y][x].height - 1);

			vector<Point_<int32_t>> corrsNearR, corrsMidR, corrsFarR;
			vector<Point2d> corrsNearR2, corrsMidR2, corrsFarR2;
			//vector<Point3d> p3DTPnewR, p3DTNnewR;
			vector<Point3d> p3DTPnewRNear, p3DTPnewRMid, p3DTPnewRFar;
			//vector<Point2d> x1TNR;
			corrsNearR.reserve(nrNear);
			corrsMidR.reserve(nrMid);
			corrsFarR.reserve(nrFar);
			p3DTPnew[y][x].reserve(nrNear + nrMid + nrFar);
			corrsAllD[y][x].reserve(nrNear + nrMid + nrFar);
			p3DTPnewRNear.reserve(nrNear);
			p3DTPnewRMid.reserve(nrNear);
			p3DTPnewRFar.reserve(nrFar);
			x1TN[y][x].reserve(nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x));
			x2TN[y][x].reserve(nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x));
			x2TNdistCorr[y][x].reserve(nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x));

			//Ckeck for backprojected correspondences
			nrNear -= (int32_t)seedsNearFromLast[y][x].size();
			nrFar -= (int32_t)seedsFarFromLast[y][x].size();
			nrMid -= (int32_t)seedsMidFromLast[y][x].size();
			if (nrNear < 0)
				nrFar += nrNear;
			if (nrFar < 0)
				nrMid += nrFar;

			while (((nrNear > 0) || (nrFar > 0) || (nrMid > 0)) && (maxSelect2 > 0) && (maxSelect3 > 0) && (maxSelect4 > 0))
			{
				pt.x = distributionX(rand_gen);
				pt.y = distributionY(rand_gen);

				if (depthAreaMap.at<unsigned char>(pt) == 1)
				{
					maxSelect--;
					if ((nrNear <= 0) && (maxSelect >= 0)) continue;
					//Check if coordinate is too near to existing keypoint
					Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
					if ((s_tmp.at<unsigned char>(posadd, posadd) > 0) || (combMovObjLabelsAll.at<unsigned char>(pt) > 0))
					{
						maxSelect++;
						maxSelect2--;
						continue;
					}
					maxSelect2 = 50;
					//Check if it is also an inlier in the right image
					bool isInl = checkLKPInlier(pt, pt2, pCam, depthMap);
					Mat s_tmp1 = cImg2(Rect((int)round(pt2.x), (int)round(pt2.y), kSi, kSi));
					if (s_tmp1.at<unsigned char>(posadd, posadd) > 0)
					{
						maxSelect++;
						maxSelect4--;
						continue;
					}
					s_tmp1.at<unsigned char>(posadd, posadd) = 1;//The minimum distance between keypoints in the second image is fixed to 1 for new correspondences
					maxSelect4 = 50;
					s_tmp += csurr;
					if (!isInl)
					{
						if (nrTN > 0)
						{
							x1TN[y][x].push_back(Point2d((double)pt.x, (double)pt.y));
							nrTN--;
						}
						else
						{
							maxSelect++;
							maxSelect3--;
							s_tmp -= csurr;
						}
						continue;
					}
					maxSelect3 = 50;
					nrNear--;
					corrsNearR.push_back(pt);
					corrsNearR2.push_back(pt2);
					p3DTPnewRNear.push_back(pCam);
				}
				else if (depthAreaMap.at<unsigned char>(pt) == 2)
				{
					maxSelect--;
					if ((nrMid <= 0) && (maxSelect >= 0)) continue;
					//Check if coordinate is too near to existing keypoint
					Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
					if ((s_tmp.at<unsigned char>(posadd, posadd) > 0) || (combMovObjLabelsAll.at<unsigned char>(pt) > 0))
					{
						maxSelect++;
						maxSelect2--;
						continue;
					}
					maxSelect2 = 50;
					//Check if it is also an inlier in the right image
					bool isInl = checkLKPInlier(pt, pt2, pCam, depthMap);
					Mat s_tmp1 = cImg2(Rect((int)round(pt2.x), (int)round(pt2.y), kSi, kSi));
					if (s_tmp1.at<unsigned char>(posadd, posadd) > 0)
					{
						maxSelect++;
						maxSelect4--;
						continue;
					}
					s_tmp1.at<unsigned char>(posadd, posadd) = 1;//The minimum distance between keypoints in the second image is fixed to 1 for new correspondences
					maxSelect4 = 50;
					s_tmp += csurr;
					if (!isInl)
					{
						if (nrTN > 0)
						{
							x1TN[y][x].push_back(Point2d((double)pt.x, (double)pt.y));
							nrTN--;
						}
						else
						{
							maxSelect++;
							maxSelect3--;
							s_tmp -= csurr;
						}
						continue;
					}
					maxSelect3 = 50;
					nrMid--;
					corrsMidR.push_back(pt);
					corrsMidR2.push_back(pt2);
					p3DTPnewRMid.push_back(pCam);
				}
				else if (depthAreaMap.at<unsigned char>(pt) == 3)
				{
					maxSelect--;
					if ((nrFar <= 0) && (maxSelect >= 0)) continue;
					//Check if coordinate is too near to existing keypoint
					Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
					if ((s_tmp.at<unsigned char>(posadd, posadd) > 0) || (combMovObjLabelsAll.at<unsigned char>(pt) > 0))
					{
						maxSelect++;
						maxSelect2--;
						continue;
					}
					maxSelect2 = 50;
					//Check if it is also an inlier in the right image
					bool isInl = checkLKPInlier(pt, pt2, pCam, depthMap);
					Mat s_tmp1 = cImg2(Rect((int)round(pt2.x), (int)round(pt2.y), kSi, kSi));
					if (s_tmp1.at<unsigned char>(posadd, posadd) > 0)
					{
						maxSelect++;
						maxSelect4--;
						continue;
					}
					s_tmp1.at<unsigned char>(posadd, posadd) = 1;//The minimum distance between keypoints in the second image is fixed to 1 for new correspondences
					maxSelect4 = 50;
					s_tmp += csurr;
					if (!isInl)
					{
						if (nrTN > 0)
						{
							x1TN[y][x].push_back(Point2d((double)pt.x, (double)pt.y));
							nrTN--;
						}
						else
						{
							maxSelect++;
							maxSelect3--;
							s_tmp -= csurr;
						}
						continue;
					}
					maxSelect3 = 50;
					nrFar--;
					corrsFarR.push_back(pt);
					corrsFarR2.push_back(pt2);
					p3DTPnewRFar.push_back(pCam);
				}
				else
				{
					cout << "Depth area not defined! This should not happen!" << endl;
				}
			}

			//Copy 3D points and correspondences
			if (!p3DTPnewRNear.empty())
			{
				//std::copy(p3DTPnewRNear.begin(), p3DTPnewRNear.end(), p3DTPnew[y][x].end());
				p3DTPnew[y][x].insert(p3DTPnew[y][x].end(), p3DTPnewRNear.begin(), p3DTPnewRNear.end());
			}
			if (!p3DTPnewRMid.empty())
			{
				//std::copy(p3DTPnewRMid.begin(), p3DTPnewRMid.end(), p3DTPnew[y][x].end());
				p3DTPnew[y][x].insert(p3DTPnew[y][x].end(), p3DTPnewRMid.begin(), p3DTPnewRMid.end());
			}
			if (!p3DTPnewRFar.empty())
			{
				//std::copy(p3DTPnewRFar.begin(), p3DTPnewRFar.end(), p3DTPnew[y][x].end());
				p3DTPnew[y][x].insert(p3DTPnew[y][x].end(), p3DTPnewRFar.begin(), p3DTPnewRFar.end());
			}

			if (!corrsNearR.empty())
			{
				//std::copy(corrsNearR.begin(), corrsNearR.end(), corrsAllD[y][x].end());
				corrsAllD[y][x].insert(corrsAllD[y][x].end(), corrsNearR.begin(), corrsNearR.end());
			}
			if (!corrsMidR.empty())
			{
				//std::copy(corrsMidR.begin(), corrsMidR.end(), corrsAllD[y][x].end());
				corrsAllD[y][x].insert(corrsAllD[y][x].end(), corrsMidR.begin(), corrsMidR.end());
			}
			if (!corrsFarR.empty())
			{
				//std::copy(corrsFarR.begin(), corrsFarR.end(), corrsAllD[y][x].end());
				corrsAllD[y][x].insert(corrsAllD[y][x].end(), corrsFarR.begin(), corrsFarR.end());
			}

			if (!corrsNearR2.empty())
			{
				//std::copy(corrsNearR2.begin(), corrsNearR2.end(), corrsAllD2[y][x].end());
				corrsAllD2[y][x].insert(corrsAllD2[y][x].end(), corrsNearR2.begin(), corrsNearR2.end());
			}
			if (!corrsMidR2.empty())
			{
				//std::copy(corrsMidR2.begin(), corrsMidR2.end(), corrsAllD2[y][x].end());
				corrsAllD2[y][x].insert(corrsAllD2[y][x].end(), corrsMidR2.begin(), corrsMidR2.end());
			}
			if (!corrsFarR2.empty())
			{
				//std::copy(corrsFarR2.begin(), corrsFarR2.end(), corrsAllD2[y][x].end());
				corrsAllD2[y][x].insert(corrsAllD2[y][x].end(), corrsFarR2.begin(), corrsFarR2.end());
			}

			//Select for true negatives in image 1 true negatives in image 2
			size_t selTN2 = 0;
			if (nrBPTN2cnt < nrBPTN2)//First take backprojected TN from the second image
			{
				for (size_t i = 0; i < x1TN[y][x].size(); i++)
				{
					pt2.x = actCorrsImg2TNFromLast.at<double>(0, nrBPTN2cnt);
					pt2.y = actCorrsImg2TNFromLast.at<double>(1, nrBPTN2cnt);
					x2TN[y][x].push_back(pt2);
					x2TNdistCorr[y][x].push_back(50.0);
					nrBPTN2cnt++;
					selTN2++;
					if (nrBPTN2cnt >= nrBPTN2)
						break;
				}
			}
			std::uniform_int_distribution<int32_t> distributionX2(0, imgSize.width - 1);
			std::uniform_int_distribution<int32_t> distributionY2(0, imgSize.height - 1);
			if (selTN2 < x1TN[y][x].size())//Select the rest randomly
			{
				for (size_t i = selTN2; i < x1TN[y][x].size(); i++)
				{
					int max_try = 10;
					while (max_try > 0)
					{
						pt.x = distributionX2(rand_gen);
						pt.y = distributionY2(rand_gen);
						Mat s_tmp = cImg2(Rect(pt, Size(kSi, kSi)));
						if (s_tmp.at<unsigned char>(posadd, posadd) > 0)
						{
							max_try--;
							continue;
						}
						csurr.copyTo(s_tmp);
						x2TN[y][x].push_back(Point2d((double)pt.x, (double)pt.y));
						x2TNdistCorr[y][x].push_back(50.0);
						break;
					}
				}
				while (x1TN[y][x].size() > x2TN[y][x].size())
				{
					x1TN[y][x].pop_back();
					nrTN++;
				}
			}
			if ((nrTN > 0) && (nrBPTN2cnt < nrBPTN2))//Take backprojected TN from the second image if available
			{
				int32_t nrTN_tmp = nrTN;
				for (int32_t i = 0; i < nrTN_tmp; i++)
				{
					int max_try = 10;
					while (max_try > 0)
					{
						pt.x = distributionX(rand_gen);
						pt.y = distributionY(rand_gen);
						Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
						if ((s_tmp.at<unsigned char>(posadd, posadd) > 0) || (combMovObjLabelsAll.at<unsigned char>(pt) > 0))
						{
							max_try--;
							continue;
						}
						csurr.copyTo(s_tmp);
						x1TN[y][x].push_back(Point2d((double)pt.x, (double)pt.y));
						pt2.x = actCorrsImg2TNFromLast.at<double>(0, nrBPTN2cnt);
						pt2.y = actCorrsImg2TNFromLast.at<double>(1, nrBPTN2cnt);
						nrBPTN2cnt++;
						x2TN[y][x].push_back(pt2);
						x2TNdistCorr[y][x].push_back(50.0);
						nrTN--;
						break;
					}
					if (nrBPTN2cnt >= nrBPTN2)
						break;
				}
			}
			
			//Get the rest of TN correspondences
			if (nrTN > 0)
			{
				std::vector<Point2d> x1TN_tmp, x2TN_tmp;
				std::vector<double> x2TNdistCorr_tmp;
				Mat maskImg1;
				copyMakeBorder(combMovObjLabelsAll, maskImg1, posadd, posadd, posadd, posadd, BORDER_CONSTANT, Scalar(0));
				maskImg1 |= corrsIMG;
				nrTN = genTrueNegCorrs(nrTN, distributionX, distributionY, distributionX2, distributionY2, x1TN_tmp, x2TN_tmp, x2TNdistCorr_tmp, maskImg1, cImg2, depthMap);
				if (!x1TN_tmp.empty())
				{
					corrsIMG(Rect(Point(posadd, posadd), imgSize)) |= maskImg1(Rect(Point(posadd, posadd), imgSize)) & (combMovObjLabelsAll == 0);
					//copy(x1TN_tmp.begin(), x1TN_tmp.end(), x1TN[y][x].end());
					x1TN[y][x].insert(x1TN[y][x].end(), x1TN_tmp.begin(), x1TN_tmp.end());
					//copy(x2TN_tmp.begin(), x2TN_tmp.end(), x2TN[y][x].end());
					x2TN[y][x].insert(x2TN[y][x].end(), x2TN_tmp.begin(), x2TN_tmp.end());
					//copy(x2TNdistCorr_tmp.begin(), x2TNdistCorr_tmp.end(), x2TNdistCorr[y][x].end());
					x2TNdistCorr[y][x].insert(x2TNdistCorr[y][x].end(), x2TNdistCorr_tmp.begin(), x2TNdistCorr_tmp.end());
				}
			}
		}
	}

	//Store correspondences
	actImgPointCloud.clear();
	distTNtoReal.clear();
	size_t nrTPCorrs = 0, nrTNCorrs = 0;
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			nrTPCorrs += corrsAllD[y][x].size();
			nrTNCorrs += x1TN[y][x].size();
		}
	}
	actCorrsImg1TP = Mat::ones(3, nrTPCorrs, CV_64FC1);
	actCorrsImg2TP = Mat::ones(3, nrTPCorrs, CV_64FC1);
	actCorrsImg1TN = Mat::ones(3, nrTNCorrs, CV_64FC1);
	actCorrsImg2TN = Mat::ones(3, nrTNCorrs, CV_64FC1);

	size_t cnt = 0, cnt2 = 0;
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			if (!p3DTPnew[y][x].empty())
			{
				//std::copy(p3DTPnew[y][x].begin(), p3DTPnew[y][x].end(), actImgPointCloud.end());
				actImgPointCloud.insert(actImgPointCloud.end(), p3DTPnew[y][x].begin(), p3DTPnew[y][x].end());
			}
			if (!x2TNdistCorr[y][x].empty())
			{
				//std::copy(x2TNdistCorr[y][x].begin(), x2TNdistCorr[y][x].end(), distTNtoReal.end());
				distTNtoReal.insert(distTNtoReal.end(), x2TNdistCorr[y][x].begin(), x2TNdistCorr[y][x].end());
			}
			
			for (size_t i = 0; i < corrsAllD[y][x].size(); i++)
			{
				actCorrsImg1TP.at<double>(0, cnt) = (double)corrsAllD[y][x][i].x;
				actCorrsImg1TP.at<double>(1, cnt) = (double)corrsAllD[y][x][i].y;
				actCorrsImg2TP.at<double>(0, cnt) = corrsAllD2[y][x][i].x;
				actCorrsImg2TP.at<double>(1, cnt) = corrsAllD2[y][x][i].y;
				cnt++;
			}

			for (size_t i = 0; i < x1TN[y][x].size(); i++)
			{
				actCorrsImg1TN.at<double>(0, cnt2) = x1TN[y][x][i].x;
				actCorrsImg1TN.at<double>(1, cnt2) = x1TN[y][x][i].y;
				actCorrsImg2TN.at<double>(0, cnt2) = x2TN[y][x][i].x;
				actCorrsImg2TN.at<double>(1, cnt2) = x2TN[y][x][i].y;
				cnt2++;
			}
		}
	}

}

/*Generates a number of nrTN true negative correspondences with a given x- & y- distribution (including the range) in both
images. True negative correspondences are only created in areas where the values around the selected locations in the masks 
for images 1 and 2 (img1Mask and img2Mask) are zero (indicates that there is no near neighboring correspondence).
nrTN			In: Number of true negative correspondences to create
distributionX	In: x-distribution and value range in the first image
distributionY	In: y-distribution and value range in the first image
distributionX2	In: x-distribution and value range in the second image
distributionY2	In: y-distribution and value range in the second image
x1TN			Out: True negative keypoints in the first image
x2TN			Out: True negative keypoints in the second image
x2TNdistCorr	Out: Distance of a TN keypoint in the second image to its true positive location. If the value is larger 50.0, the TN was generated completely random.
img1Mask		In/Out: Mask marking not usable regions / areas around already selected correspondences in camera 1
img2Mask		In/Out: Mask marking not usable regions / areas around already selected correspondences in camera 2

Return value: Number of true negatives that could not be selected due to area restrictions.
*/
int32_t genStereoSequ::genTrueNegCorrs(int32_t nrTN, 
	std::uniform_int_distribution<int32_t> &distributionX,
	std::uniform_int_distribution<int32_t> &distributionY,
	std::uniform_int_distribution<int32_t> &distributionX2,
	std::uniform_int_distribution<int32_t> &distributionY2,
	std::vector<cv::Point2d> &x1TN,
	std::vector<cv::Point2d> &x2TN,
	std::vector<double> &x2TNdistCorr,
	cv::Mat &img1Mask,
	cv::Mat &img2Mask,
	cv::Mat &usedDepthMap)/*,
	cv::InputArray labelMask)*/
{
	int32_t kSi = csurr.rows;
	int32_t posadd = (kSi - 1) / 2;
	std::normal_distribution<double> distributionNX2(0, max(imgSize.width / 48, 10));
	std::normal_distribution<double> distributionNY2(0, max(imgSize.width / 48, 10));
	int maxSelect2 = 75;
	int maxSelect3 = max(3 * nrTN, 500);
	Point pt;
	Point2d pt2;
	Point3d pCam;

	/*Mat optLabelMask;//Optional mask for moving objects to select only TN on moving objects in the first image
	if(labelMask.empty())
    {
        optLabelMask = Mat::ones(imgSize, CV_8UC1);
    } else
    {
        optLabelMask = labelMask.getMat();
    }*/

	while ((nrTN > 0) && (maxSelect2 > 0) && (maxSelect3 > 0))
	{
		pt.x = distributionX(rand_gen);
		pt.y = distributionY(rand_gen);

		Mat s_tmp = img1Mask(Rect(pt, Size(kSi, kSi)));
		if ((s_tmp.at<unsigned char>(posadd, posadd) > 0))// || (optLabelMask.at<unsigned char>(pt) == 0))
		{
			maxSelect2--;
			continue;
		}
		maxSelect2 = 75;
		s_tmp += csurr;
		x1TN.push_back(Point2d((double)pt.x, (double)pt.y));
		int max_try = 10;
		double perfDist = 50.0;
		if (!checkLKPInlier(pt, pt2, pCam, usedDepthMap))//Take a random corresponding point in the second image if the reprojection is not visible to get a TN
		{
			while (max_try > 0)
			{
				pt.x = distributionX2(rand_gen);
				pt.y = distributionY2(rand_gen);
				Mat s_tmp1 = img2Mask(Rect(pt, Size(kSi, kSi)));
				if (s_tmp1.at<unsigned char>(posadd, posadd) > 0)
				{
					max_try--;
					continue;
				}
				//s_tmp1 += csurr;
                s_tmp1.at<unsigned char>(posadd, posadd)++;
				break;
			}
			pt2 = Point2d((double)pt.x, (double)pt.y);
		}
		else//Distort the reprojection in the second image to get a TN
		{
			Point2d ptd;
			while (max_try > 0)
			{
			    int maxAtBorder = 10;
                do {
                    do {
                        ptd.x = distributionNX2(rand_gen);
                        ptd.x += 0.75 * ptd.x / abs(ptd.x);
                        ptd.x *= 1.5;
                        ptd.y = distributionNY2(rand_gen);
                        ptd.y += 0.75 * ptd.y / abs(ptd.y);
                        ptd.y *= 1.5;
                    } while ((abs(ptd.x) < 1.5) && (abs(ptd.y) < 1.5));
                    pt2 += ptd;
                    maxAtBorder--;
                } while (((pt2.x < 0) || (pt2.x > (double)(imgSize.width - 1)) ||
                         (pt2.y < 0) || (pt2.y > (double)(imgSize.height - 1))) && (maxAtBorder > 0));

                if (maxAtBorder <= 0)
                {
                    max_try = 0;
                    break;
                }

				Mat s_tmp1 = img2Mask(Rect((int)round(pt2.x), (int)round(pt2.y), kSi, kSi));
				if (s_tmp1.at<unsigned char>(posadd, posadd) > 0)
				{
					max_try--;
					continue;
				}
				//s_tmp1 += csurr;
                s_tmp1.at<unsigned char>(posadd, posadd)++;
				perfDist = norm(ptd);
				break;
			}
		}
		if (max_try <= 0)
		{
			maxSelect3--;
			x1TN.pop_back();
			s_tmp -= csurr;
			continue;
		}
		x2TN.push_back(pt2);
		x2TNdistCorr.push_back(perfDist);
		nrTN--;
	}

	return nrTN;
}

//Check, if the given point in the first camera is also visible in the second camera
//Calculates the 3D-point in the camera coordinate system and the corresponding point in the second image
bool genStereoSequ::checkLKPInlier(cv::Point_<int32_t> pt, cv::Point2d &pt2, cv::Point3d &pCam, cv::Mat &usedDepthMap)
{
	Mat x = (Mat_<double>(3, 1) << (double)pt.x, (double)pt.y, 1.0);

	double depth = usedDepthMap.at<double>(pt);
	x = K1i * x;
	x *= depth / x.at<double>(2);
	pCam = Point3d(x);

	Mat x2 = K2 * (actR * x + actT);
	x2 /= x2.at<double>(2);
	pt2 = Point2d(x2.rowRange(0, 2));

	if ((pt2.x < 0) || (pt2.x > (double)(imgSize.width - 1)) ||
		(pt2.y < 0) || (pt2.y > (double)(imgSize.height - 1)))
	{
		return false;
	}

	return true;
}

//Calculate the initial number, size, and positions of moving objects in the image
void genStereoSequ::getNrSizePosMovObj()
{
	//size_t nrMovObjs;//Number of moving objects in the scene
	//cv::InputArray startPosMovObjs;//Possible starting positions of moving objects in the image (must be 3x3 boolean (CV_8UC1))
	//std::pair<double, double> relAreaRangeMovObjs;//Relative area range of moving objects. Area range relative to the image area at the beginning.

	if (pars.nrMovObjs == 0)
	{
		return;
	}

	if (pars.startPosMovObjs.empty() || (cv::sum(pars.startPosMovObjs)[0] == 0))
	{
		startPosMovObjs = Mat::zeros(3, 3, CV_8UC1);
		while(cv::sum(startPosMovObjs)[0] == 0) {
            for (size_t y = 0; y < 3; y++) {
                for (size_t x = 0; x < 3; x++) {
                    startPosMovObjs.at<unsigned char>(y, x) = (unsigned char) (rand() % 2);
                }
            }
        }
	}
	else
	{
		startPosMovObjs = pars.startPosMovObjs;
	}

	//Check, if the input paramters are valid and if not, adapt them
	int nrStartA = 0;
	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			if (startPosMovObjs.at<unsigned char>(y,x))
			{
				nrStartA++;
			}
		}
	}

	int imgArea = imgSize.area();
	maxOPerReg = (int)ceil((float)pars.nrMovObjs / (float)nrStartA);
	int area23 = 2 * imgArea / 3;//The moving objects should not be larger than that
	minOArea = (int)round(pars.relAreaRangeMovObjs.first * (double)imgArea);
	maxOArea = (int)round(pars.relAreaRangeMovObjs.second * (double)imgArea);

	//The maximum image area coverd with moving objects should not exeed 2/3 of the image
	if (minOArea * (int)pars.nrMovObjs > area23)
	{
		pars.nrMovObjs = (size_t)(area23 / minOArea);
		maxOArea = minOArea;
		minOArea = minOArea / 2;
	}

	//If more than 2 seeds for moving objects are within an image region (9x9), then the all moving objects in a region should cover not more than 2/3 of the region
	//This helps to reduce the propability that during the generation of the moving objects (beginning at the seed positions) one objects blocks the generation of an other
	//For less than 3 objects per region, there shouldnt be a problem as they can grow outside an image region and the propability of blocking a different moving object is not that high
	if (maxOPerReg > 2)
	{
		int areaPerReg23 = area23 / 9;
		if (maxOPerReg * minOArea > areaPerReg23)
		{
			if (minOArea > areaPerReg23)
			{
				maxOArea = areaPerReg23;
				minOArea = maxOArea / 2;
				maxOPerReg = 1;
			}
			else
			{
				maxOPerReg = areaPerReg23 / minOArea;
				maxOArea = minOArea;
				minOArea = minOArea / 2;
			}
			pars.nrMovObjs = (size_t)(maxOPerReg * nrStartA);
		}
	}
	else
	{
		maxOPerReg = 2;
	}

	//Get the number of moving object seeds per region
	int nrMovObjs_tmp = (int)pars.nrMovObjs;
	Mat nrPerReg = Mat::zeros(3, 3, CV_8UC1);
	while (nrMovObjs_tmp > 0)
	{
		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				if (startPosMovObjs.at<unsigned char>(y, x) && (maxOPerReg > (int)nrPerReg.at<unsigned char>(y, x)))
				{
					int addit = rand() % 2;
					if (addit)
					{
						nrPerReg.at<unsigned char>(y, x)++;
						nrMovObjs_tmp--;
						if (nrMovObjs_tmp == 0)
							break;
					}
				}
			}
			if (nrMovObjs_tmp == 0)
				break;
		}
	}

	//Get the area for each moving object
	int maxObjsArea = min(area23, maxOArea * (int)pars.nrMovObjs);
	maxOArea = maxObjsArea / (int)pars.nrMovObjs;
	std::uniform_int_distribution<int32_t> distribution((int32_t)minOArea, (int32_t)maxOArea);
	movObjAreas = vector<vector<vector<int32_t>>>(3, vector<vector<int32_t>>(3));
	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			int nr_tmp = (int)nrPerReg.at<unsigned char>(y, x);
			for(int i = 0; i < nr_tmp; i++)
			{
				movObjAreas[y][x].push_back(distribution(rand_gen));
			}
		}
	}

	//Get seed positions
	minODist = imgSize.height / (3 * (maxOPerReg + 1));
	movObjSeeds = vector<vector<vector<cv::Point_<int32_t>>>>(3, vector<vector<cv::Point_<int32_t>>>(3));
	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			int nr_tmp = (int)nrPerReg.at<unsigned char>(y, x);
			if (nr_tmp > 0)
			{
				rand_gen = std::default_random_engine((unsigned int)std::rand());//Prevent getting the same starting positions for equal ranges
				std::uniform_int_distribution<int> distributionX(regROIs[y][x].x, regROIs[y][x].x + regROIs[y][x].width - 1);
				std::uniform_int_distribution<int> distributionY(regROIs[y][x].y, regROIs[y][x].y + regROIs[y][x].height - 1);
				movObjSeeds[y][x].push_back(cv::Point_<int32_t>(distributionX(rand_gen), distributionY(rand_gen)));
				nr_tmp--;
				if (nr_tmp > 0)
				{
					vector<int> xposes, yposes;
					xposes.push_back(movObjSeeds[y][x].back().x);
					yposes.push_back(movObjSeeds[y][x].back().y);
					while (nr_tmp > 0)
					{
						sort(xposes.begin(), xposes.end());
						sort(yposes.begin(), yposes.end());
						vector<double> xInterVals, yInterVals;
						vector<double> xWeights, yWeights;

						//Get possible selection ranges for x-values
						int start = max(xposes[0] - minODist, regROIs[y][x].x);
						int maxEnd = regROIs[y][x].x + regROIs[y][x].width;
						int xyend = min(xposes[0] + minODist, maxEnd);
						if (start == regROIs[y][x].x)
						{
							xInterVals.push_back((double)start);
							xInterVals.push_back((double)(xposes[0] + minODist));
							xWeights.push_back(0);
						}
						else
						{
							xInterVals.push_back((double)regROIs[y][x].x);
							xInterVals.push_back((double)start);
							xWeights.push_back(1.0);
							if (xyend != maxEnd)
							{
								xInterVals.push_back((double)xyend);
								xWeights.push_back(0);
							}
						}
						if (xyend != maxEnd)
						{
							for (size_t i = 1; i < xposes.size(); i++)
							{
								start = max(xposes[i] - minODist, (int)floor(xInterVals.back()));
								if (start != (int)floor(xInterVals.back()))
								{
									xInterVals.push_back((double)(xposes[i] - minODist));
									xWeights.push_back(1.0);
								}								
								xyend = min(xposes[i] + minODist, maxEnd);
								if (xyend != maxEnd)
								{
									xInterVals.push_back((double)xyend);
									xWeights.push_back(0);
								}
							}
						}
						if (xyend != maxEnd)
						{
							xInterVals.push_back((double)maxEnd);
							xWeights.push_back(1.0);
						}

						//Get possible selection ranges for y-values
						start = max(yposes[0] - minODist, regROIs[y][x].y);
						maxEnd = regROIs[y][x].y + regROIs[y][x].height;
						xyend = min(yposes[0] + minODist, maxEnd);
						if (start == regROIs[y][x].y)
						{
							yInterVals.push_back((double)start);
							yInterVals.push_back((double)(yposes[0] + minODist));
							yWeights.push_back(0);
						}
						else
						{
							yInterVals.push_back((double)regROIs[y][x].y);
							yInterVals.push_back((double)start);
							yWeights.push_back(1.0);
							if (xyend != maxEnd)
							{
								yInterVals.push_back((double)xyend);
								yWeights.push_back(0);
							}
						}
						if (xyend != maxEnd)
						{
							for (size_t i = 1; i < yposes.size(); i++)
							{
								start = max(yposes[i] - minODist, (int)floor(yInterVals.back()));
								if (start != (int)floor(yInterVals.back()))
								{
									yInterVals.push_back((double)(yposes[i] - minODist));
									yWeights.push_back(1.0);
								}
								xyend = min(yposes[i] + minODist, maxEnd);
								if (xyend != maxEnd)
								{
									yInterVals.push_back((double)xyend);
									yWeights.push_back(0);
								}
							}
						}
						if (xyend != maxEnd)
						{
							yInterVals.push_back((double)maxEnd);
							yWeights.push_back(1.0);
						}

						//Create piecewise uniform distribution and get a random seed
						piecewise_constant_distribution<double> distrPieceX(xInterVals.begin(), xInterVals.end(), xWeights.begin());
						piecewise_constant_distribution<double> distrPieceY(yInterVals.begin(), yInterVals.end(), yWeights.begin());
						movObjSeeds[y][x].push_back(cv::Point_<int32_t>((int32_t)floor(distrPieceX(rand_gen)), (int32_t)floor(distrPieceY(rand_gen))));
						xposes.push_back(movObjSeeds[y][x].back().x);
						yposes.push_back(movObjSeeds[y][x].back().y);
						nr_tmp--;
					}
				}
			}
		}
	}
}

//Generates labels of moving objects within the image and calculates the percentage of overlap for each region
//Moreover, the number of static correspondences per region is adapted and the number of correspondences on the moving objects is calculated
//mask is used to exclude areas from generating labels and must have the same size as the image; mask holds the areas from backprojected moving objects
//seeds must hold the seeding positions for generating the labels
//areas must hold the desired area for every label
//corrsOnMovObjLF must hold the number of correspondences on moving objects that were backprojected (thus the objects were created one or more frames beforehand)
//					from 3D (including TN calculated using the inlier ratio).
void genStereoSequ::generateMovObjLabels(cv::Mat &mask, std::vector<cv::Point_<int32_t>> &seeds, std::vector<int32_t> &areas, int32_t corrsOnMovObjLF)
{
	CV_Assert(seeds.size() == areas.size());

	size_t nr_movObj = areas.size();

	movObjLabels.clear();
	if(nr_movObj) {
        //movObjLabels.resize(nr_movObj, cv::Mat::zeros(imgSize, CV_8UC1));
        for (size_t i = 0; i < nr_movObj; i++)
        {
            movObjLabels.push_back(cv::Mat::zeros(imgSize, CV_8UC1));
        }
    }
	combMovObjLabels = cv::Mat::zeros(imgSize, CV_8UC1);
	//Set seeding positions in mov. obj. label images
	for (size_t i = 0; i < nr_movObj; i++)
	{
		movObjLabels[i].at<unsigned char>(seeds[i]) = 1;
		combMovObjLabels.at<unsigned char>(seeds[i]) = (unsigned char)(i + 1);
	}
	Size siM1(imgSize.width - 1, imgSize.height - 1);

#define MOV_OBJ_ONLY_IN_REGIONS 0
#if MOV_OBJ_ONLY_IN_REGIONS
	vector<cv::Point_<int32_t>> objRegionIndices(nr_movObj);
	for (size_t i = 0; i < nr_movObj; i++)
    {
        objRegionIndices[i].x = seeds[i].x / (imgSize.width / 3);
        objRegionIndices[i].y = seeds[i].y / (imgSize.height / 3);
    }
#else
	Rect imgArea = Rect(Point(0, 0), imgSize);//Is also useless as it covers the whole image
	Mat regMask = cv::Mat::ones(imgSize, CV_8UC1);//is currently not really used (should mark the areas where moving objects can grow)
#endif
	std::vector<cv::Point_<int32_t>> startposes = seeds;
	vector<int32_t> actArea(nr_movObj, 1);
	vector<size_t> nrIterations(nr_movObj, 0);
	vector<unsigned char> dilateOps(nr_movObj, 0);
	vector<bool> objNFinished(nr_movObj, true);
	int remainObj = (int)nr_movObj;

	//Generate labels
    size_t visualizeMask = 0;
	while (remainObj > 0)
	{
		for (size_t i = 0; i < nr_movObj; i++)
		{
			if (objNFinished[i])
			{
				if (!addAdditionalDepth((unsigned char)(i + convhullPtsObj.size() + 1),
					combMovObjLabels,
					movObjLabels[i],
					mask,
#if MOV_OBJ_ONLY_IN_REGIONS
					regmasks[objRegionIndices[i].y][objRegionIndices[i].x],
#else
					regMask,
#endif
					startposes[i],
					startposes[i],
					actArea[i],
					areas[i],
					siM1,
					seeds[i],
#if MOV_OBJ_ONLY_IN_REGIONS
					regmasksROIs[objRegionIndices[i].y][objRegionIndices[i].x],
#else
					imgArea,
#endif
					nrIterations[i],
					dilateOps[i]))
				{
					objNFinished[i] = false;
					remainObj--;
				}
			}
            /*Mat dilImgTh4;
            cv::threshold( movObjLabels[i], dilImgTh4, 0, 255,0 );
            namedWindow( "Dilated4", WINDOW_AUTOSIZE );
            imshow("Dilated4", dilImgTh4);
            waitKey(0);
            destroyWindow("Dilated4");*/
        }
		if(visualizeMask % 100 == 0)
        {
		    Mat colorMapImg;
            unsigned char clmul = 255 / nr_movObj;
            // Apply the colormap:
            applyColorMap(combMovObjLabels * clmul, colorMapImg, cv::COLORMAP_RAINBOW);
            namedWindow( "combined ObjLabels", WINDOW_AUTOSIZE );
            imshow("combined ObjLabels", colorMapImg);

            waitKey(0);
            destroyWindow("combined ObjLabels");
        }
        visualizeMask++;
	}

	//Get bounding rectangles for the areas
	if (nr_movObj > 0)
	{
		movObjLabelsROIs.resize(nr_movObj);
		for (size_t i = 0; i < nr_movObj; i++)
		{
			Mat mask_tmp = movObjLabels[i].clone();
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			size_t dilTries = 0;
			cv::findContours(mask_tmp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			while((contours.size() > 1) && (dilTries < 5))//Prevent the detection of multiple objects if connections between parts are too small
            {
                Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
                dilate(mask_tmp, mask_tmp, element);
                contours.clear();
                hierarchy.clear();
                cv::findContours(mask_tmp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
                dilTries++;
            }
			if(dilTries >= 5)
            {
                vector<vector<Point> > contours_new(1);
                for(vector<vector<Point> >::reverse_iterator cit = contours.rbegin(); cit != contours.rend(); cit++)
                {
                    contours_new[0].insert(contours_new[0].end(), cit->begin(), cit->end());
                }
                contours = contours_new;
            }
			movObjLabelsROIs[i] = cv::boundingRect(contours[0]);
		}
	}

	//Get overlap of regions and the portion of correspondences that is covered by the moving objects
	vector<vector<double>> movObjOverlap(3, vector<double>(3, 0));
	movObjHasArea = vector<vector<bool>>(3, vector<bool>(3, false));
	vector<vector<int32_t>> movObjCorrsFromStatic(3, vector<int32_t>(3, 0));
	vector<vector<int32_t>> movObjCorrsFromStaticInv(3, vector<int32_t>(3, 0));
	int32_t absNrCorrsFromStatic = 0;
	Mat statCorrsPRegNew = Mat::zeros(3, 3, CV_32SC1);
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			movObjOverlap[y][x] = (double)(cv::countNonZero(combMovObjLabels(regROIs[y][x])) + cv::countNonZero(mask(regROIs[y][x]))) / (double)(regROIs[y][x].area());
			if (movObjOverlap[y][x] > 0.9)
			{
				movObjHasArea[y][x] = true;
				movObjCorrsFromStatic[y][x] = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x);
				movObjCorrsFromStaticInv[y][x] = 0;
				absNrCorrsFromStatic += movObjCorrsFromStatic[y][x];
			}
			else if (nearZero(movObjOverlap[y][x]))
			{
				statCorrsPRegNew.at<int32_t>(y, x) = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x);
			}
			else
			{
				movObjCorrsFromStatic[y][x] = (int32_t)round((double)nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) * movObjOverlap[y][x]);
				movObjCorrsFromStaticInv[y][x] = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) - movObjCorrsFromStatic[y][x];
				absNrCorrsFromStatic += movObjCorrsFromStatic[y][x];
				statCorrsPRegNew.at<int32_t>(y, x) = movObjCorrsFromStaticInv[y][x];
			}
		}
	}

	////Get area of backprojected moving objects
	//double moAreaOld = 0;
	//for (size_t i = 0; i < convhullPtsObj.size; i++)
	//{
	//	moAreaOld += cv::contourArea(convhullPtsObj[i]);
	//}
	
	if (nr_movObj == 0)
		actCorrsOnMovObj = corrsOnMovObjLF;
	else
		actCorrsOnMovObj = (int32_t)round(pars.CorrMovObjPort * (double)nrCorrs[actFrameCnt]);// -corrsOnMovObjLF;
	//actCorrsOnMovObj = actCorrsOnMovObj > 0 ? actCorrsOnMovObj : 0;
	if (actCorrsOnMovObj == 0)
	{
		seeds.clear();
		areas.clear();
		movObjLabels.clear();
		//movObjLabels.resize(nr_movObj, cv::Mat::zeros(imgSize, CV_8UC1));
		combMovObjLabels = cv::Mat::zeros(imgSize, CV_8UC1);
		actCorrsOnMovObj = 0;
		actTruePosOnMovObj = 0;
		actTrueNegOnMovObj = 0;
		return;
	}
	//Check if there are too many correspondences on the moving objects
	int32_t maxCorrs = 0;
	int32_t areassum = 0;
	if (nr_movObj > 0)
	{
		for (auto i : actArea)
		{
			areassum += i;
		}
        //reduce the initial area by reducing the radius of a circle with corresponding area by 1: are_new = area - 2*sqrt(pi)*sqrt(area)+pi
//		maxCorrs = max((int32_t)(((double)(areassum)-3.545 * sqrt((double)(areassum)+3.15)) / (1.5 * pars.minKeypDist * pars.minKeypDist)), 1);
        //reduce the initial area by reducing the radius of a circle with corresponding area by reduceRadius
        double reduceRadius = 5.0;
        double tmp = sqrt((double)(areassum) * M_PI) - reduceRadius;
		maxCorrs = max((int32_t)((tmp * tmp) / (1.5 * M_PI * pars.minKeypDist * pars.minKeypDist)), 1);
		if ((actCorrsOnMovObj - corrsOnMovObjLF) > maxCorrs)
		{
			actCorrsOnMovObj = maxCorrs + corrsOnMovObjLF;
		}
	}

	double areaFracStaticCorrs = (double)absNrCorrsFromStatic / (double)nrCorrs[actFrameCnt];//Fraction of correspondences which the moving objects should take because of their area
	//double r_CorrMovObjPort = round(pars.CorrMovObjPort * 100.0) / 100.0;//Fraction of correspondences the user specified for the moving objects
	double r_areaFracStaticCorrs = round(areaFracStaticCorrs * 100.0) / 100.0;
	double r_effectiveFracMovObj = round((double)actCorrsOnMovObj / (double)nrCorrs[actFrameCnt] * 100.0) / 100.0;//Effective not changable fraction of correspondences on moving objects
	if (r_effectiveFracMovObj > r_areaFracStaticCorrs)//Remove additional static correspondences and add them to the moving objects
	{
		int32_t remStat = actCorrsOnMovObj - absNrCorrsFromStatic;
		int32_t actStatCorrs = nrCorrs[actFrameCnt] - absNrCorrsFromStatic;
		int32_t remStatrem = remStat;
		for (size_t y = 0; y < 3; y++)
		{
			for (size_t x = 0; x < 3; x++)
			{
				if (!movObjHasArea[y][x] && (remStatrem > 0))
				{
					int32_t val = (int32_t)round((double)movObjCorrsFromStaticInv[y][x] / (double)actStatCorrs * (double)remStat);
					int32_t newval = movObjCorrsFromStaticInv[y][x] - val;
					if (newval > 0)
					{
						remStatrem -= val;
						if (remStatrem < 0)
						{
							val += remStatrem;
							newval = movObjCorrsFromStaticInv[y][x] - val;
							remStatrem = 0;
						}
						statCorrsPRegNew.at<int32_t>(y, x) = newval;
					}
					else
					{
						remStatrem -= val + newval;
						if (remStatrem < 0)
						{
							statCorrsPRegNew.at<int32_t>(y, x) = -remStatrem;
							remStatrem = 0;
						}
					}
				}
			}
		}
		if (remStatrem > 0)
		{
			vector<pair<size_t, int32_t>> movObjCorrsFromStaticInv_tmp(9);
			for (size_t y = 0; y < 3; y++)
			{
				for (size_t x = 0; x < 3; x++)
				{
					const size_t idx = y * 3 + x;
					movObjCorrsFromStaticInv_tmp[idx] = make_pair(idx, statCorrsPRegNew.at<int32_t>(y, x));
				}
			}
			sort(movObjCorrsFromStaticInv_tmp.begin(), movObjCorrsFromStaticInv_tmp.end(), [](pair<size_t, int32_t> first, pair<size_t, int32_t> second) {return first.second > second.second; });
			int maxIt = remStatrem;
			while ((remStatrem > 0) && (maxIt > 0))
			{
				for (size_t i = 0; i < 9; i++)
				{
					if (movObjCorrsFromStaticInv_tmp[i].second > 0)
					{
						size_t y = movObjCorrsFromStaticInv_tmp[i].first / 3;
						size_t x = movObjCorrsFromStaticInv_tmp[i].first - y * 3;
						statCorrsPRegNew.at<int32_t>(y, x)--;
						remStatrem--;
						movObjCorrsFromStaticInv_tmp[i].second--;
						if (remStatrem == 0)
						{
							break;
						}
					}
				}
				maxIt--;
			}
		}
	}
	else if (r_effectiveFracMovObj < r_areaFracStaticCorrs)//Distribute a part of the correspondences from moving objects over the static elements not covered by moving objects
	{
		int32_t remMov = absNrCorrsFromStatic - actCorrsOnMovObj;

		int32_t actStatCorrs = nrCorrs[actFrameCnt] - absNrCorrsFromStatic;
		int32_t remMovrem = remMov;
		vector<vector<int32_t>> cmaxreg(3, vector<int32_t>(3, 0));
		for (size_t y = 0; y < 3; y++)
		{
			for (size_t x = 0; x < 3; x++)
			{
				if (!movObjHasArea[y][x] && (remMovrem > 0))
				{
					int32_t val = (int32_t)round((double)movObjCorrsFromStaticInv[y][x] / (double)actStatCorrs * (double)remMov);
					int32_t newval = movObjCorrsFromStaticInv[y][x] + val;
					//Get the maximum # of correspondences per area using the minimum distance between keypoints
					cmaxreg[y][x] = (int32_t)((double)((regROIs[y][x].width - 1) * (regROIs[y][x].height - 1)) * (1.0 - movObjOverlap[y][x]) / (1.5 * pars.minKeypDist * pars.minKeypDist));
					if (newval <= cmaxreg[y][x])
					{
						remMovrem -= val;
						if (remMovrem < 0)
						{
							val += remMovrem;
							newval = movObjCorrsFromStaticInv[y][x] + val;
							remMovrem = 0;
						}
						statCorrsPRegNew.at<int32_t>(y, x) = newval;
						cmaxreg[y][x] -= newval;
					}
					else
					{
						statCorrsPRegNew.at<int32_t>(y, x) = cmaxreg[y][x];
						remMovrem -= cmaxreg[y][x] - movObjCorrsFromStaticInv[y][x];
						if (remMovrem < 0)
						{
							statCorrsPRegNew.at<int32_t>(y, x) += remMovrem;
							remMovrem = 0;							
						}
						cmaxreg[y][x] -= statCorrsPRegNew.at<int32_t>(y, x);
					}
				}
			}
		}
		if (remMovrem > 0)
		{
			vector<pair<size_t, int32_t>> movObjCorrsFromStaticInv_tmp(9);
			for (size_t y = 0; y < 3; y++)
			{
				for (size_t x = 0; x < 3; x++)
				{
					const size_t idx = y * 3 + x;
					movObjCorrsFromStaticInv_tmp[idx] = make_pair(idx, cmaxreg[y][x]);
				}
			}
			sort(movObjCorrsFromStaticInv_tmp.begin(), movObjCorrsFromStaticInv_tmp.end(), [](pair<size_t, int32_t> first, pair<size_t, int32_t> second) {return first.second > second.second; });
			int maxIt = remMovrem;
			while ((remMovrem > 0) && (maxIt > 0))
			{
				for (size_t i = 0; i < 9; i++)
				{
					size_t y = movObjCorrsFromStaticInv_tmp[i].first / 3;
					size_t x = movObjCorrsFromStaticInv_tmp[i].first - y * 3;
					if ((movObjCorrsFromStaticInv_tmp[i].second > 0) && (statCorrsPRegNew.at<int32_t>(y, x) > 0))
					{					
						statCorrsPRegNew.at<int32_t>(y, x)++;
						remMovrem--;
						movObjCorrsFromStaticInv_tmp[i].second--;
						if (remMovrem == 0)
						{
							break;
						}
					}
				}
				maxIt--;
			}
		}
	}

	//Set new number of static correspondences
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			if (nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) > 0)
			{
				int32_t TPnew = (int32_t)round((double)statCorrsPRegNew.at<int32_t>(y, x) * (double)nrTruePosRegs[actFrameCnt].at<int32_t>(y, x) / (double)nrCorrsRegs[actFrameCnt].at<int32_t>(y, x));
				int32_t TNnew = statCorrsPRegNew.at<int32_t>(y, x) - TPnew;
				nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x) = TNnew;
				nrTruePosRegs[actFrameCnt].at<int32_t>(y, x) = TPnew;
				nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) = statCorrsPRegNew.at<int32_t>(y, x);
			}
		}
	}

	//Calculate number of correspondences on newly created moving objects
	if (nr_movObj > 0)
	{
		actCorrsOnMovObj -= corrsOnMovObjLF;
		actTruePosOnMovObj = (int32_t)round(inlRat[actFrameCnt] * (double)actCorrsOnMovObj);
		actTrueNegOnMovObj = actCorrsOnMovObj - actTruePosOnMovObj;
		actTPPerMovObj.resize(nr_movObj, 0);
		actTNPerMovObj.resize(nr_movObj, 0);
		if (nr_movObj > 1)
		{
			//First sort the areas and begin with the smallest as rounding for the number of TP and TN for every area can lead to a larger rest of correspondences that must be taken from the last area. Thus, it should be the largest.
			vector<pair<size_t, int32_t>> actAreaIdx(nr_movObj);
			for (size_t i = 0; i < nr_movObj; i++)
			{
				actAreaIdx[i] = make_pair(i, actArea[i]);
			}
			sort(actAreaIdx.begin(), actAreaIdx.end(), [](pair<size_t, int32_t> first, pair<size_t, int32_t> second) {return first.second < second.second; });
			int32_t sumTP = 0, sumTN = 0;
			for (size_t i = 0; i < nr_movObj - 1; i++)
			{
				int32_t actTP = (int32_t)round((double)actTruePosOnMovObj * (double)actAreaIdx[i].second / (double)areassum);
				int32_t actTN = (int32_t)round((double)actTrueNegOnMovObj * (double)actAreaIdx[i].second / (double)areassum);
				actTPPerMovObj[actAreaIdx[i].first] = actTP;
				actTNPerMovObj[actAreaIdx[i].first] = actTN;
				sumTP += actTP;
				sumTN += actTN;
			}
			int32_t restTP = actTruePosOnMovObj - sumTP;
			int32_t restTN = actTrueNegOnMovObj - sumTN;
			bool isValid = true;
			if (restTP <= 0)
			{
				int idx = 0;
				while ((restTP <= 0) && (idx < nr_movObj - 1))
				{
					if (actTPPerMovObj[actAreaIdx[idx].first] > 1)
					{
						actTPPerMovObj[actAreaIdx[idx].first]--;
						restTP++;
					}
				}
				if (restTP <= 0)
				{
					seeds.erase(seeds.begin() + actAreaIdx[nr_movObj - 1].first);
					areas.erase(areas.begin() + actAreaIdx[nr_movObj - 1].first);
					movObjLabels.erase(movObjLabels.begin() + actAreaIdx[nr_movObj - 1].first);
					actTPPerMovObj.erase(actTPPerMovObj.begin() + actAreaIdx[nr_movObj - 1].first);
					actTNPerMovObj.erase(actTNPerMovObj.begin() + actAreaIdx[nr_movObj - 1].first);
					movObjLabelsROIs.erase(movObjLabelsROIs.begin() + actAreaIdx[nr_movObj - 1].first);
					nr_movObj--;
					isValid = false;
				}
				else
				{
					actTPPerMovObj[actAreaIdx[nr_movObj - 1].first] = restTP;
				}
			}
			else
			{
				actTPPerMovObj[actAreaIdx[nr_movObj - 1].first] = restTP;
			}
			if (isValid)
			{
				if (restTN <= 0)
				{
					int idx = 0;
					while ((restTN <= 0) && (idx < nr_movObj - 1))
					{
						if (actTNPerMovObj[actAreaIdx[idx].first] > 1)
						{
							actTNPerMovObj[actAreaIdx[idx].first]--;
							restTN++;
						}
					}
					if (restTN <= 0)
					{
						seeds.erase(seeds.begin() + actAreaIdx[nr_movObj - 1].first);
						areas.erase(areas.begin() + actAreaIdx[nr_movObj - 1].first);
						movObjLabels.erase(movObjLabels.begin() + actAreaIdx[nr_movObj - 1].first);
						actTPPerMovObj.erase(actTPPerMovObj.begin() + actAreaIdx[nr_movObj - 1].first);
						actTNPerMovObj.erase(actTNPerMovObj.begin() + actAreaIdx[nr_movObj - 1].first);
						movObjLabelsROIs.erase(movObjLabelsROIs.begin() + actAreaIdx[nr_movObj - 1].first);
						nr_movObj--;
					}
					else
					{
						actTNPerMovObj[actAreaIdx[nr_movObj - 1].first] = restTN;
					}
				}
				else
				{
					actTNPerMovObj[actAreaIdx[nr_movObj - 1].first] = restTN;
				}
			}
		}
		else
		{
			actTPPerMovObj[0] = actTruePosOnMovObj;
			actTNPerMovObj[0] = actTrueNegOnMovObj;
		}
	}
	else
	{
		actCorrsOnMovObj = 0;
		actTruePosOnMovObj = 0;
		actTrueNegOnMovObj = 0;
		actTPPerMovObj.clear();
		actTNPerMovObj.clear();
		movObjLabels.clear();
		combMovObjLabels = cv::Mat::zeros(imgSize, CV_8UC1);
		movObjLabelsROIs.clear();
	}

	//Combine existing and new labels of moving objects
	combMovObjLabelsAll = combMovObjLabels | movObjMaskFromLast;
}

//Assign a depth category to each new object label and calculate all depth values for each label
void genStereoSequ::genNewDepthMovObj()
{
	if (pars.nrMovObjs == 0)
		return;

	//Get the depth classes that should be used for the new generated moving objects
	if (pars.movObjDepth.empty())
	{
		pars.movObjDepth.push_back(depthClass::MID);
	}
	if (pars.movObjDepth.size() == pars.nrMovObjs)//Take for every moving object its corresponding depth
	{
		if ((movObjDepthClass.size() > 0) && (movObjDepthClass.size() < pars.nrMovObjs))
		{
			vector<bool> usedDepths(pars.movObjDepth.size(), false);
			for (size_t i = 0; i < pars.movObjDepth.size(); i++)
			{
				for (size_t j = 0; j < movObjDepthClass.size(); j++)
				{
					if ((pars.movObjDepth[i] == movObjDepthClass[j]) && !usedDepths[i])
					{
						usedDepths[i] = true;
						break;
					}
				}
			}
			movObjDepthClassNew.resize(movObjLabels.size());
			for (size_t i = 0; i < movObjLabels.size(); i++)
			{
				for (size_t j = 0; j < usedDepths.size(); j++)
				{
					if (!usedDepths[j])
					{
						usedDepths[j] = true;
						movObjDepthClassNew[i] = pars.movObjDepth[j];
						break;
					}
				}
			}
		}
		else if (movObjDepthClass.size() == 0)
		{
			movObjDepthClassNew.clear();
			//copy(pars.movObjDepth.begin(), pars.movObjDepth.begin() + movObjLabels.size(), movObjDepthClassNew.begin());
			movObjDepthClassNew.insert(movObjDepthClassNew.end(), pars.movObjDepth.begin(), pars.movObjDepth.begin() + movObjLabels.size());
		}
		else
		{
			cout << "No new moving objects! This should not happen!" << endl;
			return;
		}
	}
	else if ((pars.movObjDepth.size() == 1))//Always take this depth class
	{
		movObjDepthClassNew.clear();
		movObjDepthClassNew.resize(movObjLabels.size(), pars.movObjDepth[0]);
	}
	else if ((pars.movObjDepth.size() < pars.nrMovObjs) && (pars.movObjDepth.size() > 1) && (pars.movObjDepth.size() < 4))//Randomly choose a depth for every single object
	{
		movObjDepthClassNew.clear();
		movObjDepthClassNew.resize(movObjLabels.size());
		for (size_t i = 0; i < movObjLabels.size(); i++)
		{
			int rval = rand() % (int)pars.movObjDepth.size();
			movObjDepthClassNew[i] = pars.movObjDepth[rval];
		}
	}
	else//Use the given distribution of depth classes based on the number of given depth classes
	{
		movObjDepthClassNew.clear();
		movObjDepthClassNew.resize(movObjLabels.size());
		std::array<double, 3> depthDist = { 0,0,0 };
		for (size_t i = 0; i < pars.movObjDepth.size(); i++)
		{
			switch (pars.movObjDepth[i])
			{
			case depthClass::NEAR:
				depthDist[0]++;
				break;
			case depthClass::MID:
				depthDist[1]++;
				break;
			case depthClass::FAR:
				depthDist[2]++;
				break;
			default:
				break;
			}
		}
		std::discrete_distribution<int> distribution(depthDist.begin(), depthDist.end());
		for (size_t i = 0; i < movObjLabels.size(); i++)
		{
			int usedDepthClass = distribution(rand_gen);
			switch (usedDepthClass)
			{
			case 0:
				movObjDepthClassNew[i] = depthClass::NEAR;
				break;
			case 1:
				movObjDepthClassNew[i] = depthClass::MID;
				break;
			case 2:
				movObjDepthClassNew[i] = depthClass::FAR;
				break;
			default:
				break;
			}
		}
	}

	//Get random parameters for the depth function
	std::vector<std::vector<double>> depthFuncPars;
	getRandDepthFuncPars(depthFuncPars, movObjDepthClassNew.size());

	//Get depth values for every pixel position inside the new labels
	combMovObjDepths = Mat::zeros(imgSize, CV_64FC1);
	for (size_t i = 0; i < movObjDepthClassNew.size(); i++)
	{
		double dmin = actDepthNear, dmax = actDepthMid;
		switch (movObjDepthClassNew[i])
		{
		case depthClass::NEAR:
			dmin = actDepthNear;
			dmax = actDepthMid;
			break;
		case depthClass::MID:
			dmin = actDepthMid;
			dmax = actDepthFar;
			break;
		case depthClass::FAR:
			dmin = actDepthFar;
			dmax = maxFarDistMultiplier * actDepthFar;
			break;
		default:
			break;
		}
		
		double dmin_tmp = getRandDoubleValRng(dmin, dmin + 0.6 * (dmax - dmin));
		double dmax_tmp = getRandDoubleValRng(dmin_tmp + 0.1 * (dmax - dmin), dmax);
		double drange = dmax_tmp - dmin_tmp;
		double rXr = getRandDoubleValRng(1.5, 3.0);
		double rYr = getRandDoubleValRng(1.5, 3.0);
		double h2 = (double)movObjLabelsROIs[i].height;
		h2 *= h2;
		double w2 = (double)movObjLabelsROIs[i].width;
		w2 *= w2;
		double scale = sqrt(h2 + w2) / 2.0;
		double rXrSc = rXr / scale;
		double rYrSc = rYr / scale;
		double cx = (double)movObjLabelsROIs[i].width / 2.0;
		double cy = (double)movObjLabelsROIs[i].height / 2.0;

		double minVal = DBL_MAX, maxVal = -DBL_MAX;
		Mat objArea = movObjLabels[i](movObjLabelsROIs[i]);
		Mat objAreaDepths = combMovObjDepths(movObjLabelsROIs[i]);
		for (int y = 0; y < movObjLabelsROIs[i].height; y++)
		{
			for (int x = 0; x < movObjLabelsROIs[i].width; x++)
			{
				if (objArea.at<unsigned char>(y, x) != 0)
				{
					double val = getDepthFuncVal(depthFuncPars[i], ((double)x - cx) * rXrSc, ((double)y - cy) * rYrSc);
					objAreaDepths.at<double>(y, x) = val;
					if (val > maxVal)
						maxVal = val;
					if (val < minVal)
						minVal = val;
				}
			}
		}
		double ra = maxVal - minVal;
		scale = drange / ra;
		for (int y = 0; y < movObjLabelsROIs[i].height; y++)
		{
			for (int x = 0; x < movObjLabelsROIs[i].width; x++)
			{
				if (objArea.at<unsigned char>(y, x) != 0)
				{
					double val = objAreaDepths.at<double>(y, x);
					val -= minVal;
					val *= scale;
					val += dmin_tmp;
					objAreaDepths.at<double>(y, x) = val;
				}
			}
		}
	}

    //Visualize the depth values
    Mat normalizedDepth, labelMask = cv::Mat::zeros(imgSize, CV_8UC1);
    for(auto dl : movObjLabels)
    {
        labelMask |= (dl != 0);
    }
    cv::normalize(combMovObjDepths, normalizedDepth, 0.1, 1.0, cv::NORM_MINMAX, -1, labelMask);
    namedWindow( "Normalized Moving Obj Depth", WINDOW_AUTOSIZE );
    imshow("Normalized Moving Obj Depth", normalizedDepth);
    waitKey(0);
    destroyWindow("Normalized Moving Obj Depth");

	//Add new depth classes to existing ones
	//copy(movObjDepthClassNew.begin(), movObjDepthClassNew.end(), movObjDepthClass.end());
	movObjDepthClass.insert(movObjDepthClass.end(), movObjDepthClassNew.begin(), movObjDepthClassNew.end());
}

//Generate correspondences and TN for newly generated moving objects
void genStereoSequ::getMovObjCorrs()
{
	size_t nr_movObj = actTPPerMovObj.size();
	int32_t kSi = csurr.rows;
	int32_t posadd = (kSi - 1) / 2;
	Point_<int32_t> pt;
	Point2d pt2;
	Point3d pCam;
	Mat corrsSet = Mat::zeros(imgSize.height + kSi - 1, imgSize.width + kSi - 1, CV_8UC1);
	cv::copyMakeBorder(movObjMaskFromLast2, movObjMask2All, posadd, posadd, posadd, posadd, BORDER_CONSTANT, Scalar(0));//movObjMask2All must be reduced to image size at the end
	//int maxIt = 20;

	//For visualization
	int dispit = 0;
	const int dispit_interval = 10;

	//Generate TP correspondences
	movObjCorrsImg1TP.clear();
	movObjCorrsImg1TP.resize(nr_movObj);
	movObjCorrsImg2TP.clear();
	movObjCorrsImg2TP.resize(nr_movObj);
	movObjCorrsImg1TN.clear();
	movObjCorrsImg1TN.resize(nr_movObj);
	movObjCorrsImg2TN.clear();
	movObjCorrsImg2TN.resize(nr_movObj);
	movObj3DPtsCamNew.clear();
	movObj3DPtsCamNew.resize(nr_movObj);
	movObjDistTNtoRealNew.clear();
	movObjDistTNtoRealNew.resize(nr_movObj);
	for (size_t i = 0; i < nr_movObj; i++)
	{
		std::uniform_int_distribution<int32_t> distributionX(movObjLabelsROIs[i].x, movObjLabelsROIs[i].x + movObjLabelsROIs[i].width - 1);
		std::uniform_int_distribution<int32_t> distributionY(movObjLabelsROIs[i].y, movObjLabelsROIs[i].y + movObjLabelsROIs[i].height - 1);
		//int cnt1 = 0;
		int nrTN = actTNPerMovObj[i];
		int nrTP = actTPPerMovObj[i];
		vector<Point2d> x1TN, x2TN;
		vector<Point2d> x1TP, x2TP;
		int32_t maxSelect = 50;
		int32_t maxSelect2 = 50;
		int32_t maxSelect3 = 50;

		while ((nrTP > 0) && (maxSelect > 0) && (maxSelect2 > 0) && (maxSelect3 > 0))
		{
			pt.x = distributionX(rand_gen);
			pt.y = distributionY(rand_gen);

			Mat s_tmp = corrsSet(Rect(pt, Size(kSi, kSi)));
			if ((movObjLabels[i].at<unsigned char>(pt) == 0) || (s_tmp.at<unsigned char>(posadd, posadd) > 0))
			{
				maxSelect--;
				continue;
			}
			maxSelect = 50;

			//Check if it is also an inlier in the right image
			bool isInl = checkLKPInlier(pt, pt2, pCam, combMovObjDepths);
			if(isInl) {
                Mat s_tmp1 = movObjMask2All(Rect((int) round(pt2.x), (int) round(pt2.y), kSi, kSi));
                if (s_tmp1.at<unsigned char>(posadd, posadd) > 0) {
                    maxSelect2--;
                    continue;
                }
                s_tmp1.at<unsigned char>(posadd,
                                         posadd) = 1;//The minimum distance between keypoints in the second image is fixed to approx. 1 for new correspondences
                maxSelect2 = 50;
            }
			s_tmp += csurr;
			if (!isInl)
			{
				if (nrTN > 0)
				{
					x1TN.push_back(Point2d((double)pt.x, (double)pt.y));
					nrTN--;
				}
				else
				{
					maxSelect3--;
					s_tmp -= csurr;
				}
				continue;
			}
			maxSelect3 = 50;
			nrTP--;
			x1TP.push_back(Point2d((double)pt.x, (double)pt.y));
			x2TP.push_back(pt2);
			movObj3DPtsCamNew[i].push_back(pCam);

			//Visualize the masks
			if(dispit % dispit_interval == 0)
            {
                namedWindow( "Move Corrs mask img1", WINDOW_AUTOSIZE );
                imshow("Move Corrs mask img1", (corrsSet > 0));
                namedWindow( "Move Corrs mask img2", WINDOW_AUTOSIZE );
                imshow("Move Corrs mask img2", (movObjMask2All > 0));
                waitKey(0);
                destroyWindow("Move Corrs mask img1");
                destroyWindow("Move Corrs mask img2");
            }
            dispit++;
		}
		//If there are still correspondences missing, try to use them in the next object
		/*if ((nrTP > 0) && (i < (nr_movObj - 1)))
		{
			actTPPerMovObj[i + 1] += nrTP;
		}*/

		std::uniform_int_distribution<int32_t> distributionX2(0, imgSize.width - 1);
		std::uniform_int_distribution<int32_t> distributionY2(0, imgSize.height - 1);
		//Find TN keypoints in the second image for already found TN in the first image
		size_t corrsNotVisible = x1TN.size();
		if (!x1TN.empty())
		{
            //Generate mask for visualization before adding keypoints
            Mat dispMask = (movObjMask2All > 0);

			for (size_t j = 0; j < corrsNotVisible; j++)
			{
				int max_try = 10;
				while (max_try > 0)
				{
					pt.x = distributionX2(rand_gen);
					pt.y = distributionY2(rand_gen);
					Mat s_tmp = movObjMask2All(Rect(pt, Size(kSi, kSi)));
					if (s_tmp.at<unsigned char>(posadd, posadd) > 0)
					{
						max_try--;
						continue;
					}
					//csurr.copyTo(s_tmp);
                    s_tmp.at<unsigned char>(posadd, posadd) = 1;
					x2TN.push_back(Point2d((double)pt.x, (double)pt.y));
					movObjDistTNtoRealNew[i].push_back(50.0);
					break;
				}
			}
			while (x1TN.size() > x2TN.size())
			{
				x1TN.pop_back();
				nrTN++;
			}

			//Visualize the mask afterwards
            namedWindow( "Move TN Corrs mask img2", WINDOW_AUTOSIZE );
			Mat dispMask2 = (movObjMask2All > 0);
            vector<Mat> channels;
            Mat b = Mat::zeros(dispMask2.size(), CV_8UC1);
            channels.push_back(b);
            channels.push_back(dispMask);
            channels.push_back(dispMask2);
            Mat img3c;
            merge(channels, img3c);
            imshow("Move TN Corrs mask img2", img3c);
            waitKey(0);
            destroyWindow("Move TN Corrs mask img2");
		}

		//Get the rest of TN correspondences
		if (nrTN > 0)
		{
			std::vector<Point2d> x1TN_tmp, x2TN_tmp;
			std::vector<double> x2TNdistCorr_tmp;
			Mat maskImg1;
			copyMakeBorder(movObjLabels[i], maskImg1, posadd, posadd, posadd, posadd, BORDER_CONSTANT, Scalar(0));
			maskImg1 = (maskImg1 == 0) | corrsSet;

            //Generate mask for visualization before adding keypoints
            Mat dispMaskImg2 = (movObjMask2All > 0);
            Mat dispMaskImg1 = (maskImg1 > 0);

            //Generate a depth map for generating TN based on the depth of the actual moving object
            double dmin = actDepthNear, dmax = actDepthMid;
            switch (movObjDepthClassNew[i])
            {
                case depthClass::NEAR:
                    dmin = actDepthNear;
                    dmax = actDepthMid;
                    break;
                case depthClass::MID:
                    dmin = actDepthMid;
                    dmax = actDepthFar;
                    break;
                case depthClass::FAR:
                    dmin = actDepthFar;
                    dmax = maxFarDistMultiplier * actDepthFar;
                    break;
                default:
                    break;
            }
            Mat randDepth(imgSize,CV_64FC1);
            randu(randDepth, Scalar(dmin), Scalar(dmax + 0.001));

			nrTN = genTrueNegCorrs(nrTN, distributionX, distributionY, distributionX2, distributionY2, x1TN_tmp, x2TN_tmp, x2TNdistCorr_tmp, maskImg1, movObjMask2All, randDepth);//, movObjLabels[i]);

            //Visualize the mask afterwards
            Mat dispMask2Img2 = (movObjMask2All > 0);
            Mat dispMask2Img1 = (maskImg1 > 0);
            vector<Mat> channels, channels1;
            Mat b = Mat::zeros(dispMask2Img2.size(), CV_8UC1);
            channels.push_back(b);
            channels.push_back(dispMaskImg2);
            channels.push_back(dispMask2Img2);
            channels1.push_back(b);
            channels1.push_back(dispMaskImg1);
            channels1.push_back(dispMask2Img1);
            Mat img3c, img3c1;
            merge(channels, img3c);
            merge(channels1, img3c1);
            namedWindow( "Move rand TN Corrs mask img1", WINDOW_AUTOSIZE );
            imshow("Move rand TN Corrs mask img1", img3c1);
            namedWindow( "Move rand TN Corrs mask img2", WINDOW_AUTOSIZE );
            imshow("Move rand TN Corrs mask img2", img3c);
            waitKey(0);
            destroyWindow("Move rand TN Corrs mask img1");
            destroyWindow("Move rand TN Corrs mask img2");

			if (!x1TN_tmp.empty())
			{
				corrsSet(Rect(Point(posadd, posadd), imgSize)) |= (maskImg1(Rect(Point(posadd, posadd), imgSize)) & (movObjLabels[i] > 0));
				//copy(x1TN_tmp.begin(), x1TN_tmp.end(), x1TN.end());
				x1TN.insert(x1TN.end(), x1TN_tmp.begin(), x1TN_tmp.end());
				//copy(x2TN_tmp.begin(), x2TN_tmp.end(), x2TN.end());
				x2TN.insert(x2TN.end(), x2TN_tmp.begin(), x2TN_tmp.end());
				//copy(x2TNdistCorr_tmp.begin(), x2TNdistCorr_tmp.end(), movObjDistTNtoRealNew[i].end());
				movObjDistTNtoRealNew[i].insert(movObjDistTNtoRealNew[i].end(), x2TNdistCorr_tmp.begin(), x2TNdistCorr_tmp.end());
			}
		}

        //Adapt the number of TP and TN in the next objects based on the remaining number of TP and TN of the current object
        if (((nrTP > 0) || (nrTN > 0)) && (i < (nr_movObj - 1)))
        {
            double reductionFactor = (double)(actTPPerMovObj[i] - nrTP + actTNPerMovObj[i] - nrTN) / (double)(actTPPerMovObj[i] + actTNPerMovObj[i]);
            //incorporate fraction of not visible (in cam2) features
            reductionFactor *= (double)(corrsNotVisible + x1TP.size()) / (double)x1TP.size();
            reductionFactor = reductionFactor > 1.0 ? 1.0:reductionFactor;
            for (int j = i + 1; j < nr_movObj; ++j) {
                actTPPerMovObj[j] = (int32_t)round((double)actTPPerMovObj[j] * reductionFactor);
                actTNPerMovObj[j] = (int32_t)round((double)actTNPerMovObj[j] * reductionFactor);
            }
            //Change the number of TP and TN to correct the overall inlier ratio of moving objects (as the desired inlier ratio of the current object is not reached)
            int32_t next_corrs = actTNPerMovObj[i + 1] + actTPPerMovObj[i + 1];
            actTPPerMovObj[i + 1] += nrTP;
            actTNPerMovObj[i + 1] += nrTN;
            reductionFactor = (double)next_corrs / (double)(actTNPerMovObj[i + 1] + actTPPerMovObj[i + 1]);
            actTPPerMovObj[i + 1] = (int32_t)round((double)actTPPerMovObj[i + 1] * reductionFactor);
            actTNPerMovObj[i + 1] = (int32_t)round((double)actTNPerMovObj[i + 1] * reductionFactor);
            if(actTPPerMovObj[i + 1] < nrTP)
            {
                actTPPerMovObj[i + 1] = nrTP;
            }
            if(actTNPerMovObj[i + 1] < nrTN)
            {
                actTNPerMovObj[i + 1] = nrTN;
            }
        }

		//Store correspondences
		if (!x1TP.empty())
		{
			movObjCorrsImg1TP[i] = Mat::ones(3, (int)x1TP.size(), CV_64FC1);
			movObjCorrsImg2TP[i] = Mat::ones(3, (int)x1TP.size(), CV_64FC1);
			movObjCorrsImg1TP[i].rowRange(0, 2) = Mat(x1TP).reshape(1).t();
			movObjCorrsImg2TP[i].rowRange(0, 2) = Mat(x2TP).reshape(1).t();
		}
		if (!x1TN.empty())
		{
			movObjCorrsImg1TN[i] = Mat::ones(3, (int)x1TN.size(), CV_64FC1);
			movObjCorrsImg1TN[i].rowRange(0, 2) = Mat(x1TN).reshape(1).t();
		}
		if (!x2TN.empty())
		{
			movObjCorrsImg2TN[i] = Mat::ones(3, (int)x2TN.size(), CV_64FC1);
			movObjCorrsImg2TN[i].rowRange(0, 2) = Mat(x2TN).reshape(1).t();
		}
	}
	movObjMask2All = movObjMask2All(Rect(Point(posadd, posadd), imgSize));
}

//Generate (backproject) correspondences from existing moving objects and generate hulls of the objects in the image
//Moreover, as many true negatives as needed by the inlier ratio are generated
void genStereoSequ::backProjectMovObj()
{
	convhullPtsObj.clear();
	actTNPerMovObjFromLast.clear();
	movObjLabelsFromLast.clear();
	movObjCorrsImg1TPFromLast.clear();
	movObjCorrsImg2TPFromLast.clear();
	movObjCorrsImg12TPFromLast_Idx.clear();
	movObjCorrsImg1TNFromLast.clear();
	movObjCorrsImg2TNFromLast.clear();
	movObjDistTNtoReal.clear();
	movObjMaskFromLast = Mat::zeros(imgSize, CV_8UC1);
	movObjMaskFromLast2 = Mat::zeros(imgSize, CV_8UC1);
	actCorrsOnMovObjFromLast = 0;
	actTruePosOnMovObjFromLast = 0;
	actTrueNegOnMovObjFromLast = 0;

	if (pars.nrMovObjs == 0)
	{
		return;
	}

	vector<size_t> delList;
	size_t actNrMovObj = movObj3DPtsCam.size();

	if (movObj3DPtsCam.empty())
		return;

	movObjCorrsImg1TPFromLast.resize(actNrMovObj);
	movObjCorrsImg2TPFromLast.resize(actNrMovObj);
	movObjCorrsImg12TPFromLast_Idx.resize(actNrMovObj);
	movObjCorrsImg1TNFromLast.resize(actNrMovObj);
	movObjCorrsImg2TNFromLast.resize(actNrMovObj);

	struct imgWH {
		double width;
		double height;
		double maxDist;
	} dimgWH;
	dimgWH.width = (double)(imgSize.width - 1);
	dimgWH.height = (double)(imgSize.height - 1);
	dimgWH.maxDist = maxFarDistMultiplier * actDepthFar;
	int sqrSi = csurr.rows;
	int posadd = (sqrSi - 1) / 2;
	std::vector<Mat> movObjMaskFromLastLarge(actNrMovObj);
	std::vector<Mat> movObjMaskFromLastLarge2(actNrMovObj);
	std::vector<std::vector<cv::Point>> movObjPt1(actNrMovObj), movObjPt2(actNrMovObj);

	for (size_t i = 0; i < actNrMovObj; i++)
	{
		movObjMaskFromLastLarge[i] = Mat::zeros(imgSize.height + sqrSi - 1, imgSize.width + sqrSi - 1, CV_8UC1);
		movObjMaskFromLastLarge2[i] = Mat::zeros(imgSize.height + sqrSi - 1, imgSize.width + sqrSi - 1, CV_8UC1);
		size_t oor = 0;
		size_t i2 = 0;
		for (auto pt : movObj3DPtsCam[i])
		{
			i2++;
			if ((pt.z < actDepthNear) ||
				(pt.z > dimgWH.maxDist))
			{
				oor++;
				continue;
			}

			Mat X = Mat(pt).reshape(1, 3);
			Mat x1 = K1 * X;
			x1 /= x1.at<double>(2);

			bool outOfR[2] = { false, false };
			if ((x1.at<double>(0) < 0) || (x1.at<double>(0) > dimgWH.width) ||
				(x1.at<double>(1) < 0) || (x1.at<double>(0) > dimgWH.height))//Not visible in first image
			{
				outOfR[0] = true;
			}

			Mat x2 = K2 * (actR * X + actT);
			x2 /= x2.at<double>(2);

			if ((x2.at<double>(0) < 0) || (x2.at<double>(0) > dimgWH.width) ||
				(x2.at<double>(1) < 0) || (x2.at<double>(0) > dimgWH.height))//Not visible in second image
			{
				outOfR[1] = true;
			}

			if (outOfR[0] || outOfR[1])
				oor++;

			Point ptr1 = Point((int)round(x1.at<double>(0)), (int)round(x1.at<double>(1)));
			Point ptr2 = Point((int)round(x2.at<double>(0)), (int)round(x2.at<double>(1)));

			//Check if the point is too near to an other correspondence of a moving object
			if (movObjMaskFromLast.at<unsigned char>(ptr1) > 0)
				outOfR[0] = true;

			//Check if the point is too near to an other correspondence of a moving object in the second image
			if (movObjMaskFromLast2.at<unsigned char>(ptr2) > 0)
				outOfR[1] = true;

			//Check if the point is too near to an other correspondence of this moving object
			if (!outOfR[0])
			{
				Mat s_tmp = movObjMaskFromLastLarge[i](Rect(ptr1, Size(sqrSi, sqrSi)));
				if (s_tmp.at<unsigned char>(posadd, posadd) > 0)
					outOfR[0] = true;
				else
					csurr.copyTo(s_tmp);
			}
			//Check if the point is too near to an other correspondence of this moving object in the second image
			if (!outOfR[1])
			{
				Mat s_tmp = movObjMaskFromLastLarge2[i](Rect(ptr2, Size(sqrSi, sqrSi)));
				if (s_tmp.at<unsigned char>(posadd, posadd) > 0)
					outOfR[1] = true;
				else
					csurr.copyTo(s_tmp);
			}

			if (outOfR[0] && outOfR[1])
			{
				continue;
			}
			else if (outOfR[0])
			{
				if (movObjCorrsImg2TNFromLast[i].empty())
				{
					movObjCorrsImg2TNFromLast[i] = x2.t();
				}
				else
				{
					movObjCorrsImg2TNFromLast[i].push_back(x2.t());
				}
				movObjPt2[i].push_back(ptr2);
			}
			else if (outOfR[1])
			{
				if (movObjCorrsImg1TNFromLast[i].empty())
				{
					movObjCorrsImg1TNFromLast[i] = x1.t();
				}
				else
				{
					movObjCorrsImg1TNFromLast[i].push_back(x1.t());
				}
				movObjPt1[i].push_back(ptr1);
			}
			else
			{
				if (movObjCorrsImg1TPFromLast[i].empty())
				{
					movObjCorrsImg1TPFromLast[i] = x1.t();
					movObjCorrsImg2TPFromLast[i] = x2.t();
				}
				else
				{
					movObjCorrsImg1TPFromLast[i].push_back(x1.t());
					movObjCorrsImg2TPFromLast[i].push_back(x2.t());
				}
				movObjCorrsImg12TPFromLast_Idx[i].push_back(i2-1);
				movObjPt1[i].push_back(ptr1);
				movObjPt2[i].push_back(ptr2);
			}
		}

		double actGoodPortion = (double)(movObj3DPtsCam[i].size() - oor) / (double)movObj3DPtsCam[i].size();
		if ((actGoodPortion < pars.minMovObjCorrPortion) || nearZero(actGoodPortion))
		{
			delList.push_back(i);
		}
		else
		{
			if (!movObjCorrsImg1TNFromLast[i].empty())
				movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].t();
			if (!movObjCorrsImg2TNFromLast[i].empty())
				movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].t();
			if (!movObjCorrsImg1TPFromLast[i].empty())
			{
				movObjCorrsImg1TPFromLast[i] = movObjCorrsImg1TPFromLast[i].t();
				movObjCorrsImg2TPFromLast[i] = movObjCorrsImg2TPFromLast[i].t();
			}
			movObjMaskFromLast |= movObjMaskFromLastLarge[i](Rect(Point(posadd, posadd), imgSize));
			movObjMaskFromLast2 |= movObjMaskFromLastLarge2[i](Rect(Point(posadd, posadd), imgSize));
		}
	}

	if (!delList.empty())
	{
		for (int i = (int)delList.size() - 1; i >= 0; i--)
		{
			movObjCorrsImg1TNFromLast.erase(movObjCorrsImg1TNFromLast.begin() + delList[i]);
			movObjCorrsImg2TNFromLast.erase(movObjCorrsImg2TNFromLast.begin() + delList[i]);
			movObjCorrsImg1TPFromLast.erase(movObjCorrsImg1TPFromLast.begin() + delList[i]);
			movObjCorrsImg2TPFromLast.erase(movObjCorrsImg2TPFromLast.begin() + delList[i]);
			movObjCorrsImg12TPFromLast_Idx.erase(movObjCorrsImg12TPFromLast_Idx.begin() + delList[i]);
			movObjMaskFromLastLarge.erase(movObjMaskFromLastLarge.begin() + delList[i]);
			movObjMaskFromLastLarge2.erase(movObjMaskFromLastLarge2.begin() + delList[i]);
			movObjPt1.erase(movObjPt1.begin() + delList[i]);
			movObjPt2.erase(movObjPt2.begin() + delList[i]);

			movObj3DPtsCam.erase(movObj3DPtsCam.begin() + delList[i]);
			movObj3DPtsWorld.erase(movObj3DPtsWorld.begin() + delList[i]);
			movObjWorldMovement.erase(movObjWorldMovement.begin() + delList[i]);
			movObjDepthClass.erase(movObjDepthClass.begin() + delList[i]);
		}
		actNrMovObj = movObj3DPtsCam.size();
	}
	movObjDistTNtoReal.resize(actNrMovObj);

	//Generate hulls of the objects in the image
	movObjLabelsFromLast.resize(actNrMovObj);
	convhullPtsObj.resize(actNrMovObj);
	Mat movObjMaskFromLastOld = movObjMaskFromLast.clone();
	movObjMaskFromLast = Mat::zeros(imgSize, CV_8UC1);
	//vector<double> actAreaMovObj(actNrMovObj, 0);
	for (size_t i = 0; i < actNrMovObj; i++)
	{
		Mat movObjMaskFromLastLargePiece = movObjMaskFromLastLarge[i](Rect(Point(posadd, posadd), imgSize));
		genMovObjHulls(movObjMaskFromLastLargePiece, movObjPt1[i], movObjLabelsFromLast[i]);
		movObjMaskFromLast |= movObjLabelsFromLast[i];
	}

	//Enlarge the object areas if they are too small
	Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	const int maxCnt = 20;
	for (size_t i = 0; i < actNrMovObj; i++)
	{
		int areaMO = cv::countNonZero(movObjLabelsFromLast[i]);
		int cnt = 0;
		while ((areaMO < minOArea) && (cnt < maxCnt))
		{
			Mat imgSDdilate;
			dilate(movObjLabelsFromLast[i], imgSDdilate, element);
			imgSDdilate &= ((movObjMaskFromLast == 0) | movObjLabelsFromLast[i]);
			int areaMO2 = cv::countNonZero(imgSDdilate);
			if (areaMO2 > areaMO)
			{
				areaMO = areaMO2;
				imgSDdilate.copyTo(movObjLabelsFromLast[i]);
			}
			else
			{
				break;
			}
			cnt++;
		}
	}

	//Get missing TN correspondences in second image for found TN keypoints in first image using found TN keypoints in second image
	vector<int> missingCImg2(actNrMovObj, 0);
	vector<int> missingCImg1(actNrMovObj, 0);
	for (size_t i = 0; i < actNrMovObj; i++)
	{
		if (!movObjCorrsImg1TNFromLast[i].empty() && !movObjCorrsImg1TNFromLast[i].empty())
		{
			if (movObjCorrsImg1TNFromLast[i].cols > movObjCorrsImg2TNFromLast[i].cols)
			{
				missingCImg2[i] = movObjCorrsImg1TNFromLast[i].cols - movObjCorrsImg2TNFromLast[i].cols;
			}
			else if (movObjCorrsImg1TNFromLast[i].cols < movObjCorrsImg2TNFromLast[i].cols)
			{
				missingCImg1[i] = movObjCorrsImg2TNFromLast[i].cols - movObjCorrsImg1TNFromLast[i].cols;
			}
		}
		else if (!movObjCorrsImg1TNFromLast[i].empty())
		{
			missingCImg2[i] = movObjCorrsImg1TNFromLast[i].cols;
		}
		else if (!movObjCorrsImg2TNFromLast[i].empty())
		{
			missingCImg1[i] = movObjCorrsImg2TNFromLast[i].cols;
		}
	}

	//Get ROIs of moving objects
	vector<Rect> objROIs(actNrMovObj);
	for (size_t i = 0; i < actNrMovObj; i++)
	{
		vector<Point> hull;
		genHullFromMask(movObjLabelsFromLast[i], hull);
		objROIs[i] = boundingRect(hull);
	}

	//Get missing TN correspondences for found keypoints
	std::uniform_int_distribution<int32_t> distributionX2(0, imgSize.width - 1);
	std::uniform_int_distribution<int32_t> distributionY2(0, imgSize.height - 1);
	for (size_t i = 0; i < actNrMovObj; i++)
	{
		if (missingCImg2[i] > 0)
		{
			//Enlarge mask
			Mat movObjMaskFromLast2Border(movObjMaskFromLastLarge2[i].size(), movObjMaskFromLastLarge2[i].type());
			cv::copyMakeBorder(movObjMaskFromLast2, movObjMaskFromLast2Border, posadd, posadd, posadd, posadd, BORDER_CONSTANT, cv::Scalar(0));
			Mat elemnew = Mat::ones(missingCImg2[i], 3, CV_64FC1);
			int cnt1 = 0;
			for (int j = 0; j < missingCImg2[i]; j++)
			{
				int cnt = 0;
				while (cnt < maxCnt)
				{
					Point_<int32_t> pt = Point_<int32_t>(distributionX2(rand_gen), distributionY2(rand_gen));
					Mat s_tmp = movObjMaskFromLast2Border(Rect(pt, Size(sqrSi, sqrSi)));
					if (s_tmp.at<unsigned char>(posadd, posadd) == 0)
					{
						csurr.copyTo(s_tmp);
						elemnew.at<double>(j, 0) = (double)pt.x;
						elemnew.at<double>(j, 1) = (double)pt.y;
						break;
					}
					cnt++;
				}
				if (cnt == maxCnt)
					break;
				cnt1++;
			}
			if (cnt1 > 0)
			{
				movObjMaskFromLast2 |= movObjMaskFromLast2Border(Rect(Point(posadd, posadd), imgSize));
				movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].t();
				movObjCorrsImg2TNFromLast[i].push_back(elemnew.rowRange(0, cnt1));
				movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].t();
				missingCImg2[i] -= cnt1;
				movObjDistTNtoReal[i] = vector<double>(cnt1, 50.0);
			}
			if (missingCImg2[i] > 0)
			{
				movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].colRange(0, movObjCorrsImg1TNFromLast[i].cols - missingCImg2[i]);
			}
		}
		else if (missingCImg1[i] > 0)
		{
			Mat elemnew = Mat::ones(missingCImg1[i], 3, CV_64FC1);
			int cnt1 = 0;
			std::uniform_int_distribution<int32_t> distributionX(objROIs[i].x, objROIs[i].x + objROIs[i].width - 1);
			std::uniform_int_distribution<int32_t> distributionY(objROIs[i].y, objROIs[i].y + objROIs[i].height - 1);
			//Enlarge mask
			Mat movObjMaskFromLastBorder(movObjMaskFromLastLarge[i].size(), movObjMaskFromLastLarge[i].type());
			cv::copyMakeBorder(movObjMaskFromLastOld, movObjMaskFromLastBorder, posadd, posadd, posadd, posadd, BORDER_CONSTANT, cv::Scalar(0));			
			for (int j = 0; j < missingCImg1[i]; j++)
			{
				int cnt = 0;
				while (cnt < maxCnt)
				{
					Point_<int32_t> pt = Point_<int32_t>(distributionX(rand_gen), distributionY(rand_gen));
					Mat s_tmp = movObjMaskFromLastBorder(Rect(pt, Size(sqrSi, sqrSi)));
					if ((s_tmp.at<unsigned char>(posadd, posadd) == 0) && (movObjLabelsFromLast[i].at<unsigned char>(pt) > 0))
					{
						csurr.copyTo(s_tmp);
						elemnew.at<double>(j, 0) = (double)pt.x;
						elemnew.at<double>(j, 1) = (double)pt.y;
						break;
					}
					cnt++;
				}
				if (cnt == maxCnt)
					break;
				cnt1++;
			}
			if (cnt1 > 0)
			{
				movObjMaskFromLastOld |= movObjMaskFromLastBorder(Rect(Point(posadd, posadd), imgSize));
				movObjLabelsFromLast[i] |= movObjMaskFromLastBorder(Rect(Point(posadd, posadd), imgSize));
				movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].t();
				movObjCorrsImg1TNFromLast[i].push_back(elemnew.rowRange(0, cnt1));
				movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].t();
				missingCImg1[i] -= cnt1;
				movObjDistTNtoReal[i] = vector<double>(cnt1, 50.0);
			}
			if (missingCImg1[i] > 0)
			{
				movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].colRange(0, movObjCorrsImg2TNFromLast[i].cols - missingCImg1[i]);
			}
		}
	}

	//Additionally add TN using the inlier ratio
	for (size_t i = 0; i < actNrMovObj; i++)
	{
		if (missingCImg2[i] == 0)//Skip adding TN if adding TN in the second image failed already before
		{
			missingCImg2[i] = (int)round((double)movObjCorrsImg1TPFromLast[i].cols * (1.0 - inlRat[actFrameCnt])) - movObjCorrsImg1TNFromLast[i].cols;
		}
		else
		{
			missingCImg2[i] = -1;
		}
	}
	element = cv::getStructuringElement(MORPH_RECT, Size(sqrSi, sqrSi));
	//Enlarge mask for image 2
	Mat movObjMaskFromLast2Border(movObjMaskFromLastLarge2[0].size(), movObjMaskFromLastLarge2[0].type());
	cv::copyMakeBorder(movObjMaskFromLast2, movObjMaskFromLast2Border, posadd, posadd, posadd, posadd, BORDER_CONSTANT, cv::Scalar(0));
	//Enlarge mask for image 1
	Mat movObjMaskFromLastBorder(movObjMaskFromLastLarge[0].size(), movObjMaskFromLastLarge[0].type());
	cv::copyMakeBorder(movObjMaskFromLastOld, movObjMaskFromLastBorder, posadd, posadd, posadd, posadd, BORDER_CONSTANT, cv::Scalar(0));
	for (size_t i = 0; i < actNrMovObj; i++)
	{
	    //Generate a depth map for generating TN based on the depth of the back-projected 3D points
	    double minDepth = DBL_MAX, maxDepth = DBL_MIN;
	    for(size_t j = 0; j < movObjCorrsImg12TPFromLast_Idx[i].size(); i++)
        {
	        double sDepth = movObj3DPtsCam[i][movObjCorrsImg12TPFromLast_Idx[i][j]].z;
	        if(sDepth < minDepth)
	            minDepth = sDepth;
	        if(sDepth > maxDepth)
                maxDepth = sDepth;
        }
	    Mat randDepth(imgSize,CV_64FC1);
	    randu(randDepth, Scalar(minDepth), Scalar(maxDepth + 0.001));

		std::uniform_int_distribution<int32_t> distributionX(objROIs[i].x, objROIs[i].x + objROIs[i].width - 1);
		std::uniform_int_distribution<int32_t> distributionY(objROIs[i].y, objROIs[i].y + objROIs[i].height - 1);
		int areaMO = cv::countNonZero(movObjLabelsFromLast[i]);
		int cnt2 = 0;
		while (((areaMO < maxOArea) || (cnt2 == 0)) && (missingCImg2[i] > 0) && (cnt2 < maxCnt))//If not all elements could be selected, try to enlarge the area
		{
			//Generate label mask for image 1
			Mat movObjLabelsFromLastN;
			cv::copyMakeBorder(movObjLabelsFromLast[i], movObjLabelsFromLastN, posadd, posadd, posadd, posadd, BORDER_CONSTANT, cv::Scalar(0));
			movObjLabelsFromLastN = (movObjLabelsFromLastN == 0);
			movObjLabelsFromLastN |= movObjMaskFromLastBorder;

			std::vector<cv::Point2d> x1TN;
			std::vector<cv::Point2d> x2TN;
			int32_t remainingTN = genTrueNegCorrs(missingCImg2[i],
				distributionX,
				distributionY,
				distributionX2,
				distributionY2,
				x1TN,
				x2TN,
				movObjDistTNtoReal[i],
				movObjLabelsFromLastN,
				movObjMaskFromLast2Border,
				randDepth);

			cnt2++;
			if (remainingTN != missingCImg2[i])
			{
				movObjLabelsFromLastN(Rect(Point(posadd, posadd), imgSize)) &= movObjLabelsFromLast[i];
				movObjMaskFromLastBorder(Rect(Point(posadd, posadd), imgSize)) |= movObjLabelsFromLastN(Rect(Point(posadd, posadd), imgSize));

				int32_t nelem = (int32_t)x1TN.size();
				Mat elemnew = Mat::ones(nelem, 3, CV_64FC1);
				Mat elemnew2 = Mat::ones(nelem, 3, CV_64FC1);
				for (int32_t j = 0; j < nelem; j++)
				{
					elemnew.at<double>(j, 0) = x1TN[j].x;
					elemnew.at<double>(j, 1) = x1TN[j].y;
					elemnew2.at<double>(j, 0) = x2TN[j].x;
					elemnew2.at<double>(j, 1) = x2TN[j].y;
				}
				movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].t();
				movObjCorrsImg1TNFromLast[i].push_back(elemnew);
				movObjCorrsImg1TNFromLast[i] = movObjCorrsImg1TNFromLast[i].t();
				movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].t();
				movObjCorrsImg2TNFromLast[i].push_back(elemnew2);
				movObjCorrsImg2TNFromLast[i] = movObjCorrsImg2TNFromLast[i].t();
			}
			if (remainingTN > 0)
			{			
				//Perform dilation
				Mat imgSDdilate;
				dilate(movObjLabelsFromLast[i], imgSDdilate, element);
				imgSDdilate &= ((movObjMaskFromLast == 0) | movObjLabelsFromLast[i]);
				int areaMO2 = cv::countNonZero(imgSDdilate);
				if (areaMO2 > areaMO)
				{
					areaMO = areaMO2;
					if (areaMO < maxOArea)
					{
						imgSDdilate.copyTo(movObjLabelsFromLast[i]);
						objROIs[i] = Rect(max(objROIs[i].x - sqrSi, 0), max(objROIs[i].y - sqrSi, 0), objROIs[i].width + 2 * sqrSi, objROIs[i].height + 2 * sqrSi);
						objROIs[i] = Rect(objROIs[i].x, objROIs[i].y, (objROIs[i].x + objROIs[i].width) > imgSize.width ? (imgSize.width - objROIs[i].x) : objROIs[i].width, (objROIs[i].y + objROIs[i].height) > imgSize.height ? (imgSize.height - objROIs[i].y) : objROIs[i].height);
					}
					else
					{
						break;
					}
				}
				else
				{
					break;
				}				
			}
			missingCImg2[i] = remainingTN;
		}
	}
	movObjMaskFromLastBorder(Rect(Point(posadd, posadd), imgSize)).copyTo(movObjMaskFromLastOld);	

	movObjMaskFromLast = Mat::zeros(imgSize, CV_8UC1);
	actTNPerMovObjFromLast.resize(actNrMovObj);
	for (size_t i = 0; i < actNrMovObj; i++)
	{
		genHullFromMask(movObjLabelsFromLast[i], convhullPtsObj[i]);
		movObjMaskFromLast |= (unsigned char)(i + 1) *  movObjLabelsFromLast[i];
		//actAreaMovObj[i] = contourArea(convhullPtsObj[i]);

		actTNPerMovObjFromLast[i] = movObjCorrsImg1TNFromLast[i].cols;
		
		actTruePosOnMovObjFromLast += movObjCorrsImg1TPFromLast[i].cols;
		actTrueNegOnMovObjFromLast += actTNPerMovObjFromLast[i];
	}
	actCorrsOnMovObjFromLast = actTrueNegOnMovObjFromLast + actTruePosOnMovObjFromLast;
}

void genStereoSequ::genHullFromMask(cv::Mat &mask, std::vector<cv::Point> &finalHull)
{
	//Get the contour of the mask
	vector<vector<Point>> contours;
	Mat finalMcopy = mask.clone();
	findContours(finalMcopy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	//Simplify the contour
	double epsilon = 0.01 * cv::arcLength(contours[0], true);//1% of the overall contour length
	approxPolyDP(contours[0], finalHull, epsilon, true);
}

void genStereoSequ::genMovObjHulls(cv::Mat &corrMask, std::vector<cv::Point> &kps, cv::Mat &finalMask)
{
	int sqrSi = csurr.rows;

	//Get the convex hull of the keypoints
	vector<vector<Point>> hull(1);
	convexHull(kps, hull[0]);

	//Get bounding box
	Rect hullBB = boundingRect(hull[0]);
	hullBB = Rect(max(hullBB.x - sqrSi, 0), max(hullBB.y - sqrSi, 0), hullBB.width + 2 * sqrSi, hullBB.height + 2 * sqrSi);
	hullBB = Rect(hullBB.x, hullBB.y, (hullBB.x + hullBB.width) > imgSize.width ? (imgSize.width - hullBB.x) : hullBB.width, (hullBB.y + hullBB.height) > imgSize.height ? (imgSize.height - hullBB.y) : hullBB.height);

	//Invert the mask
	Mat ncm = (corrMask(hullBB) == 0);

	//draw the filled convex hull with enlarged borders
	Mat hullMat1 = Mat::zeros(imgSize, CV_8UC3);
	vector<Mat> hullMat1C;
	//with filled contour:
	drawContours(hullMat1, hull, -1, Scalar(255, 255, 255), CV_FILLED);
	//enlarge borders:
	drawContours(hullMat1, hull, -1, Scalar(255, 255, 255), sqrSi);
	split(hullMat1, hullMat1C);
	hullMat1 = hullMat1C[0](hullBB);

	//Combine convex hull and inverted mask
	Mat icm = ncm & hullMat1;

	//Apply distance transform algorithm
	Mat distImg;
	distanceTransform(icm, distImg, DIST_L2, DIST_MASK_PRECISE, CV_32FC1);

	//Get the largest distance from white pixels to black pixels
	double minVal, maxVal;
	minMaxLoc(distImg, &minVal, &maxVal);

	//Calculate the kernel size for closing
	int kSize = (int)ceil(maxVal) * 2 + 1;

	//Get structering element
	Mat element = getStructuringElement(MORPH_RECT, Size(kSize, kSize));

	//Perform closing to generate the final mask
	morphologyEx(corrMask, finalMask, MORPH_CLOSE, element);
}

//Calculate the seeding position and area for every new moving object
//This function should not be called for the first frame
void genStereoSequ::getSeedsAreasMovObj()
{
	//Get number of possible seeds per region
	Mat nrPerRegMax = Mat::zeros(3, 3, CV_8UC1);
	int minOArea23 = 2 * min(minOArea, maxOArea - minOArea) / 3;
	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			if (startPosMovObjs.at<unsigned char>(y, x) == 0)
				continue;
			int moarea = countNonZero(movObjMaskFromLast(regROIs[y][x]));
			if (moarea < minOArea23)
			{
				nrPerRegMax.at<unsigned char>(y, x) = (unsigned char)maxOPerReg;
			}
			else if ((moarea <= maxOArea) && (maxOPerReg > 1))
			{
				nrPerRegMax.at<unsigned char>(y, x) = (unsigned char)(maxOPerReg - 1);
			}
			else if(moarea <= (maxOPerReg - 1) * maxOArea)
			{
				Mat actUsedAreaLabel;
				int nrLabels = connectedComponents(movObjMaskFromLast(regROIs[y][x]), actUsedAreaLabel, 4, CV_16U);
				if (nrLabels < maxOPerReg)
					nrPerRegMax.at<unsigned char>(y, x) = (unsigned char)(maxOPerReg - nrLabels);
			}
		}
	}

	//Get the number of moving object seeds per region
	int nrMovObjs_tmp = (int)pars.nrMovObjs - (int)movObj3DPtsWorld.size();
	Mat nrPerReg = Mat::zeros(3, 3, CV_8UC1);
	while (nrMovObjs_tmp > 0)
	{
		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				if (nrPerRegMax.at<unsigned char>(y, x) > 0)
				{
					int addit = rand() % 2;
					if (addit)
					{
						nrPerReg.at<unsigned char>(y, x)++;
						nrPerRegMax.at<unsigned char>(y, x)--;
						nrMovObjs_tmp--;
						if (nrMovObjs_tmp == 0)
							break;
					}
				}
			}
			if (nrMovObjs_tmp == 0)
				break;
		}
	}

	//Get the area for each moving object
	std::uniform_int_distribution<int32_t> distribution((int32_t)minOArea, (int32_t)maxOArea);
	movObjAreas.clear();
	movObjAreas = vector<vector<vector<int32_t>>>(3, vector<vector<int32_t>>(3));
	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			int nr_tmp = (int)nrPerReg.at<unsigned char>(y, x);
			for (int i = 0; i < nr_tmp; i++)
			{
				movObjAreas[y][x].push_back(distribution(rand_gen));
			}
		}
	}

	//Get seed positions
	minODist = imgSize.height / (3 * (maxOPerReg + 1));
	movObjSeeds.clear();
	movObjSeeds = vector<vector<vector<cv::Point_<int32_t>>>>(3, vector<vector<cv::Point_<int32_t>>>(3));
	int maxIt = 20;
	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			int nr_tmp = (int)nrPerReg.at<unsigned char>(y, x);
			if (nr_tmp > 0)
			{
				rand_gen = std::default_random_engine((unsigned int)std::rand());//Prevent getting the same starting positions for equal ranges
				std::uniform_int_distribution<int> distributionX(regROIs[y][x].x, regROIs[y][x].x + regROIs[y][x].width - 1);
				std::uniform_int_distribution<int> distributionY(regROIs[y][x].y, regROIs[y][x].y + regROIs[y][x].height - 1);
				cv::Point_<int32_t> pt;
				int cnt = 0;
				while (cnt < maxIt)
				{
					pt = cv::Point_<int32_t>(distributionX(rand_gen), distributionY(rand_gen));
					if (movObjMaskFromLast.at<unsigned char>(pt) == 0)
					{
						break;
					}
					cnt++;
				}
				if (cnt == maxIt)
				{
					movObjAreas[y][x].clear();
					nrPerReg.at<unsigned char>(y, x) = 0;
					break;
				}
				movObjSeeds[y][x].push_back(pt);
				nr_tmp--;
				if (nr_tmp > 0)
				{
					vector<int> xposes, yposes;
					xposes.push_back(movObjSeeds[y][x].back().x);
					yposes.push_back(movObjSeeds[y][x].back().y);
					while (nr_tmp > 0)
					{
						sort(xposes.begin(), xposes.end());
						sort(yposes.begin(), yposes.end());
						vector<double> xInterVals, yInterVals;
						vector<double> xWeights, yWeights;

						//Get possible selection ranges for x-values
						int start = max(xposes[0] - minODist, regROIs[y][x].x);
						int maxEnd = regROIs[y][x].x + regROIs[y][x].width - 1;
						int xyend = min(xposes[0] + minODist, maxEnd);
						if (start == regROIs[y][x].x)
						{
							xInterVals.push_back((double)start);
							xInterVals.push_back((double)(xposes[0] + minODist));
							xWeights.push_back(0);
						}
						else
						{
							xInterVals.push_back((double)regROIs[y][x].x);
							xInterVals.push_back((double)start);
							xWeights.push_back(1.0);
							if (xyend != maxEnd)
							{
								xInterVals.push_back((double)xyend);
								xWeights.push_back(0);
							}
						}
						if (xyend != maxEnd)
						{
							for (size_t i = 1; i < xposes.size(); i++)
							{
								start = max(xposes[i] - minODist, (int)floor(xInterVals.back()));
								if (start != (int)floor(xInterVals.back()))
								{
									xInterVals.push_back((double)(xposes[i] - minODist));
									xWeights.push_back(1.0);
								}
								xyend = min(xposes[i] + minODist, maxEnd);
								if (xyend != maxEnd)
								{
									xInterVals.push_back((double)xyend);
									xWeights.push_back(0);
								}
							}
						}
						if (xyend != maxEnd)
						{
							xInterVals.push_back((double)maxEnd);
							xWeights.push_back(1.0);
						}

						//Get possible selection ranges for y-values
						start = max(yposes[0] - minODist, regROIs[y][x].y);
						maxEnd = regROIs[y][x].y + regROIs[y][x].height - 1;
						xyend = min(yposes[0] + minODist, maxEnd);
						if (start == regROIs[y][x].y)
						{
							yInterVals.push_back((double)start);
							yInterVals.push_back((double)(yposes[0] + minODist));
							yWeights.push_back(0);
						}
						else
						{
							yInterVals.push_back((double)regROIs[y][x].y);
							yInterVals.push_back((double)start);
							yWeights.push_back(1.0);
							if (xyend != maxEnd)
							{
								yInterVals.push_back((double)xyend);
								yWeights.push_back(0);
							}
						}
						if (xyend != maxEnd)
						{
							for (size_t i = 1; i < yposes.size(); i++)
							{
								start = max(yposes[i] - minODist, (int)floor(yInterVals.back()));
								if (start != (int)floor(yInterVals.back()))
								{
									yInterVals.push_back((double)(yposes[i] - minODist));
									yWeights.push_back(1.0);
								}
								xyend = min(yposes[i] + minODist, maxEnd);
								if (xyend != maxEnd)
								{
									yInterVals.push_back((double)xyend);
									yWeights.push_back(0);
								}
							}
						}
						if (xyend != maxEnd)
						{
							yInterVals.push_back((double)maxEnd);
							yWeights.push_back(1.0);
						}

						//Create piecewise uniform distribution and get a random seed
						piecewise_constant_distribution<double> distrPieceX(xInterVals.begin(), xInterVals.end(), xWeights.begin());
						piecewise_constant_distribution<double> distrPieceY(yInterVals.begin(), yInterVals.end(), yWeights.begin());

						cnt = 0;
						while (cnt < maxIt)
						{
							pt = cv::Point_<int32_t>((int32_t)floor(distrPieceX(rand_gen)), (int32_t)floor(distrPieceY(rand_gen)));
							if (movObjMaskFromLast.at<unsigned char>(pt) == 0)
							{
								break;
							}
							cnt++;
						}
						if (cnt == maxIt)
						{
							for (size_t i = 0; i < (movObjAreas[y][x].size() - movObjSeeds[y][x].size()); i++)
							{
								movObjAreas[y][x].pop_back();
							}
							nrPerReg.at<unsigned char>(y, x) = (unsigned char)movObjSeeds[y][x].size();
							break;
						}

						movObjSeeds[y][x].push_back(pt);
						xposes.push_back(movObjSeeds[y][x].back().x);
						yposes.push_back(movObjSeeds[y][x].back().y);
						nr_tmp--;
					}
				}
			}
		}
	}
}

//Extracts the areas and seeding positions for new moving objects from the region structure
bool genStereoSequ::getSeedAreaListFromReg(std::vector<cv::Point_<int32_t>> &seeds, std::vector<int32_t> &areas)
{
	seeds.clear();
	areas.clear();
	seeds.reserve(pars.nrMovObjs);
	areas.reserve(pars.nrMovObjs);
	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			if (!movObjAreas[y][x].empty())
			{
				//copy(movObjAreas[y][x].begin(), movObjAreas[y][x].end(), areas.end());
				areas.insert(areas.end(), movObjAreas[y][x].begin(), movObjAreas[y][x].end());
			}
			if (!movObjSeeds[y][x].empty())
			{
				//copy(movObjSeeds[y][x].begin(), movObjSeeds[y][x].end(), seeds.end());
				seeds.insert(seeds.end(), movObjSeeds[y][x].begin(), movObjSeeds[y][x].end());
			}
		}
	}

	CV_Assert(seeds.size() == areas.size());
	if (seeds.empty())
		return false;

	return true;
}

//Check, if it is necessary to calculate new moving objects
//This function should not be called during the first frame
//The function backProjectMovObj() must be called before
bool genStereoSequ::getNewMovObjs()
{
	if (pars.minNrMovObjs == 0)
	{
		return false;
	}
	if (movObj3DPtsWorld.size() >= pars.minNrMovObjs)
	{
		return false;
	}

	return true;
}

//Combines correspondences from static and moving objects
void genStereoSequ::combineCorrespondences()
{
	//Get number of TP correspondences
	combNrCorrsTP = actCorrsImg1TPFromLast.cols + actCorrsImg1TP.cols;
	for (auto i : movObjCorrsImg1TPFromLast)
	{
		combNrCorrsTP += i.cols;
	}
	for (auto i : movObjCorrsImg1TP)
	{
		combNrCorrsTP += i.cols;
	}
	//Get number of TN correspondences
	combNrCorrsTN = actCorrsImg1TN.cols;
	for (auto i : movObjCorrsImg1TNFromLast)
	{
		combNrCorrsTN += i.cols;
	}
	for (auto i : movObjCorrsImg1TN)
	{
		combNrCorrsTN += i.cols;
	}

	combCorrsImg1TP = Mat(3, combNrCorrsTP, CV_64FC1);
	combCorrsImg2TP = Mat(3, combNrCorrsTP, CV_64FC1);
	comb3DPts.clear();
	comb3DPts.reserve(combNrCorrsTP);

	//Copy all TP keypoints of first image
	int actColNr = 0;
	int actColNr2 = actCorrsImg1TPFromLast.cols;
	actCorrsImg1TPFromLast.copyTo(combCorrsImg1TP.colRange(actColNr, actColNr2));
	actColNr = actColNr2;
	actColNr2 = actColNr + actCorrsImg1TP.cols;
	actCorrsImg1TP.copyTo(combCorrsImg1TP.colRange(actColNr, actColNr2));
	for (auto i : movObjCorrsImg1TPFromLast)
	{
		actColNr = actColNr2;
		actColNr2 = actColNr + i.cols;
		i.copyTo(combCorrsImg1TP.colRange(actColNr, actColNr2));
	}
	for (auto i : movObjCorrsImg1TP)
	{
		actColNr = actColNr2;
		actColNr2 = actColNr + i.cols;
		i.copyTo(combCorrsImg1TP.colRange(actColNr, actColNr2));
	}

	//Copy all 3D points
	for (auto i : actCorrsImg12TPFromLast_Idx)
	{
		comb3DPts.push_back(actImgPointCloudFromLast[i]);
	}
	//copy(actImgPointCloud.begin(), actImgPointCloud.end(), comb3DPts.end());
	comb3DPts.insert(comb3DPts.end(), actImgPointCloud.begin(), actImgPointCloud.end());
	for (size_t i = 0; i < movObjCorrsImg12TPFromLast_Idx.size(); i++)
	{
		for (auto j : movObjCorrsImg12TPFromLast_Idx[i])
		{
			comb3DPts.push_back(movObj3DPtsCam[i][j]);
		}
	}
	for (auto i : movObj3DPtsCamNew)
	{
		//copy(i.begin(), i.end(), comb3DPts.end());
		comb3DPts.insert(comb3DPts.end(), i.begin(), i.end());
	}

	CV_Assert(combCorrsImg1TP.cols == (int)comb3DPts.size());

	//Copy all TP keypoints of second image
	actColNr = 0;
	actColNr2 = actCorrsImg2TPFromLast.cols;
	actCorrsImg2TPFromLast.copyTo(combCorrsImg2TP.colRange(actColNr, actColNr2));
	actColNr = actColNr2;
	actColNr2 = actColNr + actCorrsImg2TP.cols;
	actCorrsImg2TP.copyTo(combCorrsImg2TP.colRange(actColNr, actColNr2));
	for (auto i : movObjCorrsImg2TPFromLast)
	{
		actColNr = actColNr2;
		actColNr2 = actColNr + i.cols;
		i.copyTo(combCorrsImg2TP.colRange(actColNr, actColNr2));
	}
	for (auto i : movObjCorrsImg2TP)
	{
		actColNr = actColNr2;
		actColNr2 = actColNr + i.cols;
		i.copyTo(combCorrsImg2TP.colRange(actColNr, actColNr2));
	}

	combCorrsImg1TN = Mat(3, combNrCorrsTN, CV_64FC1);
	combCorrsImg2TN = Mat(3, combNrCorrsTN, CV_64FC1);

	//Copy all TN keypoints of first image
	actColNr = 0;
	actColNr2 = actCorrsImg1TN.cols;
	actCorrsImg1TN.copyTo(combCorrsImg1TN.colRange(actColNr, actColNr2));
	for (auto i : movObjCorrsImg1TNFromLast)
	{
		actColNr = actColNr2;
		actColNr2 = actColNr + i.cols;
		i.copyTo(combCorrsImg1TN.colRange(actColNr, actColNr2));
	}
	for (auto i : movObjCorrsImg1TN)
	{
		actColNr = actColNr2;
		actColNr2 = actColNr + i.cols;
		i.copyTo(combCorrsImg1TN.colRange(actColNr, actColNr2));
	}

	//Copy all TN keypoints of second image
	actColNr = 0;
	actColNr2 = actCorrsImg2TN.cols;
	actCorrsImg2TN.copyTo(combCorrsImg2TN.colRange(actColNr, actColNr2));
	for (auto i : movObjCorrsImg2TNFromLast)
	{
		actColNr = actColNr2;
		actColNr2 = actColNr + i.cols;
		i.copyTo(combCorrsImg2TN.colRange(actColNr, actColNr2));
	}
	for (auto i : movObjCorrsImg2TN)
	{
		actColNr = actColNr2;
		actColNr2 = actColNr + i.cols;
		i.copyTo(combCorrsImg2TN.colRange(actColNr, actColNr2));
	}

	//Copy distances of TN locations to their real matching position
	combDistTNtoReal.reserve(combNrCorrsTN);
	//copy(distTNtoReal.begin(), distTNtoReal.end(), combDistTNtoReal.end());
	combDistTNtoReal.insert(combDistTNtoReal.end(), distTNtoReal.begin(), distTNtoReal.end());
	for (auto i : movObjDistTNtoReal)
	{
		//copy(i.begin(), i.end(), combDistTNtoReal.end());
		combDistTNtoReal.insert(combDistTNtoReal.end(), i.begin(), i.end());
	}
	for (auto i : movObjDistTNtoRealNew)
	{
		//copy(i.begin(), i.end(), combDistTNtoReal.end());
		combDistTNtoReal.insert(combDistTNtoReal.end(), i.begin(), i.end());
	}

	CV_Assert((size_t)combCorrsImg1TN.cols == combDistTNtoReal.size());
}

//Get the paramters and indices for the actual frame. This function must be called before simulating a new stereo frame
void genStereoSequ::updateFrameParameters()
{
	if (((actFrameCnt % (pars.nFramesPerCamConf)) == 0) && (actFrameCnt > 0))
	{
		actStereoCIdx++;
	}
	actR = R[actStereoCIdx];
	actT = t[actStereoCIdx];
	actDepthNear = depthNear[actStereoCIdx];
	actDepthMid = depthMid[actStereoCIdx];
	actDepthFar = depthFar[actStereoCIdx];
	
	if (((actFrameCnt % (pars.corrsPerRegRepRate)) == 0) && (actFrameCnt > 0))
	{
		actCorrsPRIdx++;
		if (actCorrsPRIdx >= pars.corrsPerRegion.size())
		{
			actCorrsPRIdx = 0;
		}
	}
}

//Insert new generated 3D points into the world coordinate point cloud
void genStereoSequ::transPtsToWorld()
{
	if (actImgPointCloud.empty())
	{
		return;
	}

	size_t nrPts = actImgPointCloud.size();

	staticWorld3DPts.reserve(staticWorld3DPts.size() + nrPts);

	for (size_t i = 0; i < nrPts; i++)
	{
		Mat ptm = absCamCoordinates[actFrameCnt].R * Mat(actImgPointCloud[i]).reshape(1).t() + absCamCoordinates[actFrameCnt].t;
		staticWorld3DPts.push_back(pcl::PointXYZ((float)ptm.at<double>(0), (float)ptm.at<double>(1), (float)ptm.at<double>(2)));
	}
}

//Transform new generated 3D points from new generated moving objects into world coordinates
void genStereoSequ::transMovObjPtsToWorld()
{
	if (movObj3DPtsCamNew.empty())
	{
		return;
	}

	size_t nrNewObjs = movObj3DPtsCamNew.size();
	size_t nrOldObjs = movObj3DPtsWorld.size();
	movObj3DPtsWorld.resize(nrOldObjs + nrNewObjs);
	movObjWorldMovement.resize(nrOldObjs + nrNewObjs);
	//Mat trans_c2w;//Translation vector for transferring 3D points from camera to world coordinates
	//trans_c2w = -1.0 * absCamCoordinates[actFrameCnt].R * absCamCoordinates[actFrameCnt].t;//Get the C2W-translation from the position of the camera centre in world coordinates
	for (size_t i = 0; i < nrNewObjs; i++)
	{
		size_t idx = nrOldObjs + i;
		movObj3DPtsWorld[idx].reserve(movObj3DPtsCamNew[i].size());
		for (size_t j = 0; j < movObj3DPtsCamNew[i].size(); j++)
		{
			Mat ptm = absCamCoordinates[actFrameCnt].R * Mat(movObj3DPtsCamNew[i][j]).reshape(1).t() + absCamCoordinates[actFrameCnt].t;
			movObj3DPtsWorld[idx].push_back(pcl::PointXYZ((float)ptm.at<double>(0), (float)ptm.at<double>(1), (float)ptm.at<double>(2)));
		}
		double velocity = 0;
		if (nearZero(pars.relMovObjVelRange.first - pars.relMovObjVelRange.second))
		{
			velocity = absCamVelocity * pars.relMovObjVelRange.first;
		}
		else
		{
			double relV = getRandDoubleValRng(pars.relMovObjVelRange.first, pars.relMovObjVelRange.second);
			velocity = absCamVelocity * relV;
		}
		Mat tdiff;
		if ((actFrameCnt + 1) < totalNrFrames)
		{
			//Get direction of camera from actual to next frame
			tdiff = absCamCoordinates[actFrameCnt + 1].t - absCamCoordinates[actFrameCnt].t;
		}
		else
		{
			//Get direction of camera from last to actual frame
			tdiff = absCamCoordinates[actFrameCnt].t - absCamCoordinates[actFrameCnt - 1].t;
		}
		tdiff /= norm(tdiff);
		//Add the movement direction of the moving object
		tdiff += movObjDir;
		tdiff /= norm(tdiff);
		tdiff *= velocity;
		movObjWorldMovement[i] = tdiff.clone();
	}
}

//Get the relative movement direction (compared to the camera movement) for every moving object
void genStereoSequ::checkMovObjDirection()
{
	if (pars.movObjDir.empty())
	{
		Mat newMovObjDir(3, 1, CV_64FC1);
		cv::randu(newMovObjDir, Scalar(0), Scalar(1.0));
		newMovObjDir /= norm(newMovObjDir);
		newMovObjDir.copyTo(movObjDir);
	}
	else
	{
		movObjDir = pars.movObjDir;
		movObjDir /= norm(movObjDir);
	}
}

//Updates the actual position of moving objects and their corresponding 3D points according to their moving direction and velocity.
void genStereoSequ::updateMovObjPositions()
{
	if (movObj3DPtsWorld.empty())
	{
		return;
	}

	for (size_t i = 0; i < movObj3DPtsWorld.size(); i++)
	{
		Eigen::Affine3f obj_transform = Eigen::Affine3f::Identity();
		obj_transform.translation() << (float)movObjWorldMovement[i].at<double>(0),
			(float)movObjWorldMovement[i].at<double>(1),
			(float)movObjWorldMovement[i].at<double>(2);
		pcl::transformPointCloud(movObj3DPtsWorld[i], movObj3DPtsWorld[i], obj_transform);
	}
}

//Get 3D-points of moving objects that are visible in the camera and transform them from the world coordinate system into camera coordinate system
void genStereoSequ::getMovObjPtsCam()
{
	if (movObj3DPtsWorld.empty())
	{
		return;
	}

	vector<int> delList;
	movObj3DPtsCam.clear();
	movObj3DPtsCam.resize(movObj3DPtsWorld.size());
	for (size_t i = 0; i < movObj3DPtsWorld.size(); i++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_movObj3DPtsWorld(&movObj3DPtsWorld[i]);
		pcl::PointCloud<pcl::PointXYZ>::Ptr camFilteredPts(new pcl::PointCloud<pcl::PointXYZ>());
		bool success = getVisibleCamPointCloud(ptr_movObj3DPtsWorld, camFilteredPts);
		if (!success)
		{
			delList.push_back(i);
			continue;
		}
		pcl::PointCloud<pcl::PointXYZ>::Ptr filteredOccludedPts(new pcl::PointCloud<pcl::PointXYZ>());
		success = filterNotVisiblePts(camFilteredPts, filteredOccludedPts);
		if (!success)
		{
			filteredOccludedPts->clear();
			success = filterNotVisiblePts(camFilteredPts, filteredOccludedPts, true);
			if (!success)
			{
				if (filteredOccludedPts->size() < 5)
				{
					delList.push_back(i);
					continue;
				}
			}
		}
		for (size_t j = 0; j < filteredOccludedPts->size(); j++)
		{
			cv::Point3d pt = Point3d((double)filteredOccludedPts->at(j).x, (double)filteredOccludedPts->at(j).y, (double)filteredOccludedPts->at(j).z);
			Mat ptm = Mat(pt, false).reshape(1, 3);
			ptm = absCamCoordinates[actFrameCnt].R.t() * (ptm - absCamCoordinates[actFrameCnt].t);//does this work???????? -> physical memory of pt and ptm must be the same
			movObj3DPtsCam[i].push_back(pt);
		}
	}
	if (!delList.empty())
	{
		for (int i = (int)delList.size() - 1; i >= 0; i--)
		{
			movObj3DPtsCam.erase(movObj3DPtsCam.begin() + delList[i]);
			movObj3DPtsWorld.erase(movObj3DPtsWorld.begin() + delList[i]);
		}
	}
}

void genStereoSequ::getCamPtsFromWorld()
{
	if (staticWorld3DPts.empty())
	{
		return;
	}

	actImgPointCloudFromLast.clear();
	pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_actImgPointCloudFromLast(&staticWorld3DPts);
	pcl::PointCloud<pcl::PointXYZ>::Ptr camFilteredPts(new pcl::PointCloud<pcl::PointXYZ>());
	bool success = getVisibleCamPointCloud(ptr_actImgPointCloudFromLast, camFilteredPts);
	if (!success)
	{
		return;
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr filteredOccludedPts(new pcl::PointCloud<pcl::PointXYZ>());
	success = filterNotVisiblePts(camFilteredPts, filteredOccludedPts);
	if (!success)
	{
		filteredOccludedPts->clear();
		success = filterNotVisiblePts(camFilteredPts, filteredOccludedPts, true);
		if (!success)
		{
			if (filteredOccludedPts->empty())
			{
				return;
			}
		}
	}
	for (size_t j = 0; j < filteredOccludedPts->size(); j++)
	{
		cv::Point3d pt = Point3d((double)filteredOccludedPts->at(j).x, (double)filteredOccludedPts->at(j).y, (double)filteredOccludedPts->at(j).z);
		Mat ptm = Mat(pt, false).reshape(1, 3);
		ptm = absCamCoordinates[actFrameCnt].R.t() * (ptm - absCamCoordinates[actFrameCnt].t);//does this work???????? -> physical memory of pt and ptm must be the same
		actImgPointCloudFromLast.push_back(pt);
	}
}

//Calculates the actual camera pose in camera coordinates in a different camera coordinate system (X forward, Y is up, and Z is right) to use the PCL filter FrustumCulling
void genStereoSequ::getActEigenCamPose()
{
	//Get actual camera pose in Eigen representation (code below could be much shorter, but this is a kind of docu for using this functions)
	Eigen::Vector3d trans;
	Eigen::Matrix3d rot;
	Eigen::Vector4d quat;
	//Mat trans_c2w;//Translation vector for transferring 3D points from camera to world coordinates
	//trans_c2w = -1.0 * absCamCoordinates[actFrameCnt].R * absCamCoordinates[actFrameCnt].t;//Get the C2W-translation from the position of the camera centre in world coordinates
	cv::cv2eigen(absCamCoordinates[actFrameCnt].t, trans);//Pose of the camera in world coordinates
	cv::cv2eigen(absCamCoordinates[actFrameCnt].R.t(), rot);//From world to cam -> Thus, w.r.t. origin
	MatToQuat(rot, quat);
	actCamRot = Eigen::Quaternionf(quat.cast<float>());
	Eigen::Affine3f cam_pose(actCamRot);
	cam_pose.translation() = trans.cast<float>();
	//cam_pose.rotate(actCamRot);

	Eigen::Matrix4f pose_orig = cam_pose.matrix();
	Eigen::Matrix4f cam2robot;
	cam2robot << 0, 0, 1, 0,//To convert from the traditional camera coordinate system (X right, Y down, Z forward) to (X is forward, Y is up, and Z is right)
		0, -1, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 1;
	actCamPose = pose_orig * cam2robot;
}

//Get part of a pointcloud visible in a camera
bool genStereoSequ::getVisibleCamPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut)
{
	pcl::FrustumCulling<pcl::PointXYZ> fc;
	fc.setInputCloud(cloudIn);
	fc.setVerticalFOV(2.f * 180.f * std::atan((float)imgSize.height / (2.f * (float)K1.at<double>(1,1))) / (float)M_PI);
	fc.setHorizontalFOV(2.f * 180.f * std::atan((float)imgSize.width / (2.f * (float)K1.at<double>(0, 0))) / (float)M_PI);
	fc.setNearPlaneDistance((float)actDepthNear);
	fc.setFarPlaneDistance((float)(maxFarDistMultiplier * actDepthFar));
	fc.setCameraPose(actCamPose);

	fc.filter(*cloudOut);

	if (cloudOut->empty())
		return false;

	return true;
}

//Filters occluded 3D points based on a voxel size corresponding to 1 pixel (when projected to the image plane) at near_depth + (medium depth - near_depth) / 2
//Returns false if more than 33% are occluded
bool genStereoSequ::filterNotVisiblePts(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut, bool useNearLeafSize)
{
	cloudIn->sensor_origin_ = Eigen::Vector4f(actCamPose(0), actCamPose(1), actCamPose(2), 1.f);
	cloudIn->sensor_orientation_ = actCamRot;

	pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> voxelFilter;
	voxelFilter.setInputCloud(cloudIn);
	float leaf_size;
	if (useNearLeafSize)
	{
		leaf_size = (float)(actDepthNear / K1.at<double>(0, 0));
	}
	else
	{
		leaf_size = (float)((actDepthNear + (actDepthMid - actDepthNear) / 2.0) / K1.at<double>(0, 0));
	}
	voxelFilter.setLeafSize(leaf_size, leaf_size, leaf_size);//1 pixel (when projected to the image plane) at near_depth + (medium depth - near_depth) / 2
	voxelFilter.initializeVoxelGrid();

	for (size_t i = 0; i < cloudIn->size(); i++)
	{
		Eigen::Vector3i grid_coordinates = voxelFilter.getGridCoordinates(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
		int grid_state;
		int ret = voxelFilter.occlusionEstimation(grid_state, grid_coordinates);
		if ((ret == 0) && (grid_state == 0))
		{
			cloudOut->push_back(cloudIn->points[i]);
		}
	}

	float fracOcc = (float)(cloudOut->size()) / (float)(cloudIn->size());
	if (fracOcc < 0.67)
		return false;

	return true;
}

//Perform the whole procedure of generating correspondences, new static, and dynamic 3D elements
void genStereoSequ::getNewCorrs()
{
	updateFrameParameters();

	//Get pose of first camera in camera coordinates using a different coordinate system where X is forward, Y is up, and Z is right
	getActEigenCamPose();

	if (pars.nrMovObjs > 0)
	{
		cv::Mat movObjMask;
		int32_t corrsOnMovObjLF = 0;
		bool calcNewMovObj = true;
		if (actFrameCnt == 0)
		{
			movObjMask = Mat::zeros(imgSize, CV_8UC1);
		}
		else
		{
			// Update the 3D world coordinates of movObj3DPtsWorld based on direction and velocity
			updateMovObjPositions();

			//Calculate movObj3DPtsCam from movObj3DPtsWorld: Get 3D-points of moving objects that are visible in the camera and transform them from the world coordinate system into camera coordinate system
			getMovObjPtsCam();

			//Generate maps (masks) of moving objects by backprojection from 3D for the first and second stereo camera: movObjMaskFromLast, movObjMaskFromLast2; create convex hulls: convhullPtsObj; and
			//Check if some moving objects should be deleted
			backProjectMovObj();
			movObjMask = movObjMaskFromLast;
			corrsOnMovObjLF = actCorrsOnMovObjFromLast;

			//Generate seeds and areas for new moving objects
			calcNewMovObj = getNewMovObjs();
			if (calcNewMovObj)
			{
				getSeedsAreasMovObj();
			}
		}
		if (calcNewMovObj)
		{
			std::vector<cv::Point_<int32_t>> seeds;
			std::vector<int32_t> areas;
			if (getSeedAreaListFromReg(seeds, areas))
			{
				//Generate new moving objects and adapt the number of static correspondences per region
				generateMovObjLabels(movObjMask, seeds, areas, corrsOnMovObjLF);

				//Assign a depth category to each new moving object label and calculate all depth values for each label
				genNewDepthMovObj();

				//Generate correspondences and 3D points for new moving objects
				getMovObjCorrs();

				//Insert new 3D points (from moving objects) into world coordinate system
				transMovObjPtsToWorld();
			}
		}
		else
		{
			//Set the global mask for moving objects
			combMovObjLabelsAll = movObjMaskFromLast;
			movObjMask2All = movObjMaskFromLast2;
		}
	}
																																	   
	//Get 3D points of static elements and store them to actImgPointCloudFromLast
	getCamPtsFromWorld();

	//Backproject static 3D points
	backProject3D();

	//Generate seeds for generating depth areas and include the seeds found by backprojection of the 3D points of the last frames
	checkDepthSeeds();

	//Generate depth areas for the current image and static elements
	genDepthMaps();

	//Generates correspondences and 3D points in the camera coordinate system (including false matches) from static scene elements
	getKeypoints();

	//Combine correspondences of static and moving objects
	combineCorrespondences();

	//Insert new 3D coordinates into the world coordinate system
	transPtsToWorld();
}

//Start calculating the whole sequence
void genStereoSequ::startCalc()
{
	actFrameCnt = 0;
	actCorrsPRIdx = 0;
	actStereoCIdx = 0;

	while(actFrameCnt < totalNrFrames)
	{
		getNewCorrs();
		actFrameCnt++;
	}
}