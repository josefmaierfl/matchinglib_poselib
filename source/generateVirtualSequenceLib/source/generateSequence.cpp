/**********************************************************************************************************
FILE: generateSequence.cpp

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: March 2018

LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for generating stereo sequences with correspondences given
a view restrictions like depth ranges, moving objects, ...
**********************************************************************************************************/

#include "generateSequence.h"
#include "helper_funcs.h"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

/* -------------------------- Functions -------------------------- */

genStereoSequ::genStereoSequ(cv::Size imgSize_, cv::Mat K1_, cv::Mat K2_, std::vector<cv::Mat> R_, std::vector<cv::Mat> t_, StereoSequParameters pars_) :
	imgSize(imgSize_), K1(K1_), K2(K2_), R(R_), t(t_), pars(pars_)
{
	CV_Assert((K1.rows == 3) && (K2.rows == 3) && (K1.cols == 3) && (K2.cols == 3) && (K1.type() == CV_64FC1) && (K2.type() == CV_64FC1));
	CV_Assert((imgSize.area() > 0) && (R.size() == t.size()) && (R.size() > 0));

	randSeed(rand_gen);

	//Calculate inverse of camera matrices
	K1i = K1.inv();
	K2i = K2.inv();

	//Number of stereo configurations
	nrStereoConfs = R.size();

	//Construct the camera path
	constructCamPath();

	//Calculate the thresholds for the depths near, mid, and far for every camera configuration
	getDepthRanges();

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

	//Calculate the initial number, size, and positions of moving objects in the image
	getNrSizePosMovObj();
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
				}
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
		double areaCorrs = maxCorr * pars.minKeypDist * pars.minKeypDist;
		
		if (areaCorrs > regA)
		{
			cout << "There are too many keypoints per region when demanding a minimum keypoint distance of " << pars.minKeypDist << ". Changing it!" << endl;
			double mKPdist = round(10.0 * sqrt(regA / maxCorr)) / 10.0;
			if (mKPdist <= 1.414214)
			{
				cout << "Changed the minimum keypoint distance to 1.0. There are still too many keypoints. Changing the number of keypoints!" << endl;
				pars.minKeypDist = 1.0;
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
				}
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
		size_t initTruePos = (size_t)round(getRandDoubleValRng((double)pars.truePosRange.first, (double)pars.truePosRange.second));
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
			std::uniform_real_distribution<size_t> distribution(pars.truePosRange.first, pars.truePosRange.second);
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
				size_t minTruePos = nrTruePos[i - 1] - rangeVal;
				minTruePos = minTruePos < pars.truePosRange.first ? pars.truePosRange.first : minTruePos;
				std::uniform_real_distribution<size_t> distribution(minTruePos, maxTruePos);
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
				inlRat[i] = distribution(rand_gen);
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
				inlRat[i] = getRandDoubleValRng(minInlrat, maxInlrat);
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
		R0 = pars.R.getMat();
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
		vector<Mat> diffTrack = vector<Mat>(nrTracks);
		vector<double> tdiffNorms = vector<double>(nrTracks);
		double trackNormSum = norm(pars.camTrack[0]);
		diffTrack[0] = pars.camTrack[0].clone();// / trackNormSum;
		tdiffNorms[0] = trackNormSum;
		for (size_t i = 1; i < nrTracks; i++)
		{
			Mat tdiff = pars.camTrack[i] - pars.camTrack[i - 1];
			double tdiffnorm = norm(tdiff);
			trackNormSum += tdiffnorm;
			diffTrack[i] = tdiff.clone();// / tdiffnorm;
			tdiffNorms[i] = tdiffnorm;
		}

		//Calculate a new scaling for the path based on the original path length, total number of frames and camera velocity
		double trackScale = (double)totalNrFrames * absCamVelocity / trackNormSum;
		//Rescale track diffs
		for (size_t i = 0; i < nrTracks; i++)
		{
			diffTrack[i] *= trackScale;
			tdiffNorms[i] *= trackScale;
		}

		//Get camera positions
		Mat R1 = R0 * getTrackRot(diffTrack[0]);
		absCamCoordinates[0] = Poses(R1.clone(), t1.clone());
		double actDiffLength = 0;
		size_t actTrackNr = 0, lastTrackNr = 0;
		for (size_t i = 1; i < totalNrFrames; i++)
		{
			bool firstAdd = true;
			Mat multTracks = Mat::zeros(3, 1, CV_64FC1);
			double usedLength = 0;
			while ((actDiffLength < absCamVelocity) && (actTrackNr < nrTracks))
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

			R1 = R0 * getTrackRot(diffTrack[lastTrackNr]);
			t1 += multTracks;
			absCamCoordinates[i] = Poses(R1.clone(), t1.clone());
			actDiffLength -= absCamVelocity;
		}
	}
}

/*Calculates a rotation for every differential vector of a track segment to ensure that the camera looks always in the direction perpendicular to the track segment.
* If the track segment equals the x-axis, the camera faces into positive z-direction (if the initial rotaion equals the identity).
*/
cv::Mat genStereoSequ::getTrackRot(cv::Mat tdiff)
{
	CV_Assert((tdiff.rows == 3) && (tdiff.cols == 1) && (tdiff.type() == CV_64FC1));

	tdiff /= norm(tdiff);

	double cy2 = tdiff.at<double>(0) * tdiff.at<double>(0) + tdiff.at<double>(2) * tdiff.at<double>(2);
	double cy = sqrt(cy2);
	double cosy = tdiff.at<double>(0) / cy;
	double siny = tdiff.at<double>(2) / cy;

	Mat Ry = (Mat_<double>(3, 3) <<
		cosy, 0, siny,
		0, 1.0, 0,
		-siny, 0, cosy);

	double cz = sqrt(cy2 + tdiff.at<double>(1) * tdiff.at<double>(1));
	double cosz = cy / cz;
	double sinz = -1.0 * tdiff.at<double>(1) / cz;

	Mat Rz = (Mat_<double>(3, 3) <<
		cosz, -sinz, 0,
		sinz, cosz, 0,
		0, 0, 1.0);

	Mat Rt_W2C = Rz * Ry;//Rotation from world to camera

	return Rt_W2C.t();//return rotation from camera to world
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
		if (!nearZero(c1))
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
		std::uniform_real_distribution<size_t> distribution(1, maxElems + 1);
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
							std::ptrdiff_t pdiff = max_element(maxAPRegd, maxAPRegd + 2) - maxAPRegd;
							while ((diffap < 0) && (cnt < 3))
							{
								if (maxAPReg[pdiff] > 1)
								{
									maxAPReg[pdiff]--;
									diffap++;
								}
								if (diffap < 0)
								{
									if ((maxAPReg[(pdiff + 1) % 3] > 1) && (maxAPRegd[(pdiff + 1) % 3] > maxAPRegd[(pdiff + 2) % 3]))
									{
										maxAPReg[(pdiff + 1) % 3]--;
										diffap++;
									}
									else if ((maxAPReg[(pdiff + 2) % 3] > 1) && (maxAPRegd[(pdiff + 2) % 3] > maxAPRegd[(pdiff + 1) % 3]))
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
							std::ptrdiff_t pdiff = min_element(maxAPRegd, maxAPRegd + 2) - maxAPRegd;
							while (diffap > 0)
							{
								maxAPReg[pdiff]++;
								diffap--;
								if (diffap > 0)
								{
									if (maxAPRegd[(pdiff + 1) % 3] < maxAPRegd[(pdiff + 2) % 3])
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
	if (!actCorrsImg2TNFromLast_Idx.empty)
		actCorrsImg2TNFromLast_Idx.clear();
	if (!actCorrsImg1TNFromLast.empty())
		actCorrsImg1TNFromLast.release();
	if (!actCorrsImg1TNFromLast_Idx.empty)
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
		actCorrsImg1TNFromLast.t();
	if (!actCorrsImg2TNFromLast.empty())
		actCorrsImg2TNFromLast.t();
	if (!actCorrsImg1TPFromLast.empty())
	{
		actCorrsImg1TPFromLast.t();
		actCorrsImg2TPFromLast.t();
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
	int maxSum1 = sqrSi1 * sqrSi1;
	
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
		csurr = Mat::ones(sqrSi, sqrSi, CV_8UC1);
		int maxSum = sqrSi * sqrSi;
		int sqrSiDiff2 = (sqrSi1 - sqrSi) / 2;
		int hlp2 = sqrSi + sqrSiDiff2;

		vector<size_t> delListCorrs, delList3D;
		if (!seedsNear_tmp.empty())
		{
			for (size_t i = 0; i < seedsNear_tmp.size(); i++)
			{
				Mat s_tmp = filtInitPts(Range(seedsNear_tmp[i].y + sqrSiDiff2, seedsNear_tmp[i].y + hlp2), Range(seedsNear_tmp[i].x + sqrSiDiff2, seedsNear_tmp[i].x + hlp2));
				s_tmp += csurr;
				if (sum(s_tmp)[0] > maxSum)
				{
					s_tmp -= csurr;
					delListCorrs.push_back((size_t)seedsNear_tmp[i].z);
					delList3D.push_back(actCorrsImg12TPFromLast_Idx[delListCorrs.back()]);
					continue;
				}
				seedsNear_tmp1.push_back(seedsNear_tmp[i]);
			}
		}
		if (!seedsMid_tmp.empty())
		{
			for (size_t i = 0; i < seedsMid_tmp.size(); i++)
			{
				Mat s_tmp = filtInitPts(Range(seedsMid_tmp[i].y + sqrSiDiff2, seedsMid_tmp[i].y + hlp2), Range(seedsMid_tmp[i].x + sqrSiDiff2, seedsMid_tmp[i].x + hlp2));
				s_tmp += csurr;
				if (sum(s_tmp)[0] > maxSum)
				{
					s_tmp -= csurr;
					delListCorrs.push_back((size_t)seedsMid_tmp[i].z);
					delList3D.push_back(actCorrsImg12TPFromLast_Idx[delListCorrs.back()]);
					continue;
				}
				seedsMid_tmp1.push_back(seedsMid_tmp[i]);
			}
		}
		if (!seedsFar_tmp.empty())
		{
			for (size_t i = 0; i < seedsFar_tmp.size(); i++)
			{
				Mat s_tmp = filtInitPts(Range(seedsFar_tmp[i].y + sqrSiDiff2, seedsFar_tmp[i].y + hlp2), Range(seedsFar_tmp[i].x + sqrSiDiff2, seedsFar_tmp[i].x + hlp2));
				s_tmp += csurr;
				if (sum(s_tmp)[0] > maxSum)
				{
					s_tmp -= csurr;
					delListCorrs.push_back((size_t)seedsFar_tmp[i].z);
					delList3D.push_back(actCorrsImg12TPFromLast_Idx[delListCorrs.back()]);
					continue;
				}
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
		mmy[1] += regSi.height;
		std::uniform_real_distribution<int32_t> distributionY(mmy[0], mmy[1]);
		for (int32_t x = 0; x < 3; x++)
		{				
			int32_t mmx[2];
			mmx[0] = x * regSi.width;
			mmx[1] += regSi.width;
			std::uniform_real_distribution<int32_t> distributionX(mmx[0], mmx[1]);
			int32_t diffNr = nrDepthAreasPRegNear[actCorrsPRIdx].at<int32_t>(y, x) - (int32_t)seedsNear[y][x].size();
			while (diffNr > 0)//Generate seeds for near depth areas
			{
				pt.x = distributionX(rand_gen);
				pt.y = distributionY(rand_gen);
				Mat s_tmp = filtInitPts(Range(pt.y, pt.y + sqrSi1), Range(pt.x, pt.x + sqrSi1));
				s_tmp += csurr1;
				if (sum(s_tmp)[0] > maxSum1)
				{
					s_tmp -= csurr1;
					continue;
				}
				else
				{
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
				s_tmp += csurr1;
				if (sum(s_tmp)[0] > maxSum1)
				{
					s_tmp -= csurr1;
					continue;
				}
				else
				{
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
				s_tmp += csurr1;
				if (sum(s_tmp)[0] > maxSum1)
				{
					s_tmp -= csurr1;
					continue;
				}
				else
				{
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
void deleteVecEntriesbyIdx(std::vector<T, A> const& editVec, std::vector<T1, A1> const& delVec)
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
			std::ptrdiff_t pdiff = min_element(maxAPReg, maxAPReg + 2) - maxAPReg;
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
inline double genStereoSequ::getDepthFuncVal(std::vector<double> &pars, double x, double y)
{
	double tmp = pars[1] - x;
	tmp *= tmp;
	double z = pars[0] * tmp;
	tmp = y - pars[2];
	tmp *= -tmp;
	tmp -= x * x;
	z *= exp(tmp);
	z -= 10.0 * (x / pars[3] - std::pow(x, pars[4]) - std::pow(y, pars[5])) * exp(-x * x - y * y);
	tmp = x + 1.0;
	tmp *= -tmp;
	z -= pars[6] / 3.0 * exp(tmp - y * y);
	return z;
}

/*Calculate random parameters for the function generating depth values
There are 7 random paramters p:
z = p1 * (p2 - x)^2 * e^(-x^2 - (y - p3)^2) - 10 * (x / p4 - x^p5 - y^p6) * e^(-x^2 - y^2) - p7 / 3 * e^(-(x + 1)^2 - y^2)
*/
void genStereoSequ::getRandDepthFuncPars(std::vector<std::vector<double>> &pars, size_t n_pars)
{
	pars = std::vector<std::vector<double>>(n_pars, std::vector<double>(7, 0));

	//p1:
	std::uniform_real_distribution<double> distribution(0, 10.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars[i][0] = distribution(rand_gen);
	}
	//p2:
	distribution = std::uniform_real_distribution<double>(0, 2.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars[i][1] = distribution(rand_gen);
	}
	//p3:
	distribution = std::uniform_real_distribution<double>(0, 4.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars[i][2] = distribution(rand_gen);
	}
	//p4:
	distribution = std::uniform_real_distribution<double>(0.5, 5.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars[i][3] = distribution(rand_gen);
	}
	//p5 & p6:
	distribution = std::uniform_real_distribution<double>(2.0, 7.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars[i][4] = distribution(rand_gen);
		pars[i][5] = distribution(rand_gen);
	}
	//p7:
	distribution = std::uniform_real_distribution<double>(1.0, 40.0);
	for (size_t i = 0; i < n_pars; i++)
	{
		pars[i][6] = distribution(rand_gen);
	}
}

/* Adds a few random depth pixels near a given position (no actual depth value, but a part of a mask indicating the depth range (near, mid, far)
unsigned char pixVal	In: Value assigned to the random pixel positions
cv::Mat &imgD			In/Out: Image holding all depth ranges where the new random depth pixels should be added
cv::Mat &imgSD			In/Out: Image holding only one specific depth range where the new random depth pixels should be added
cv::Mat &mask			In: Mask for imgD and imgSD
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
	const size_t max_iter = 100;
	vector<int32_t> directions = getPossibleDirections(startpos, mask, regMask, imgD, siM1);
	if (directions.empty() || (nrAdds > max_iter) || usedDilate)
	{
		int siAfterDil = 0, strElmSi = 3, cnt = 0, maxCnt = 20;
		while ((siAfterDil == 0) && (cnt < maxCnt))
		{
			cnt++;
			Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(strElmSi, strElmSi));
			strElmSi++;
			Mat imgSDdilate;
			dilate(imgSD(vROI), imgSDdilate, element);
			imgSDdilate &= mask(vROI) & ((imgD(vROI) == 0) | imgSD(vROI));
			siAfterDil = sum(imgSDdilate)[0];
			if (siAfterDil > maxAreaReg)
			{
				int diff = siAfterDil - maxAreaReg;
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
				imgSDdilate *= pixVal;
				imgSDdilate.copyTo(imgD(vROI));
				addArea += siAfterDil;
				nrAdds++;
				usedDilate = 1;

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
		switch (directions[diri])
		{
		case 0://direction left up
			endpos.x--;
			endpos.y--;
			break;
		case 1://direction up
			endpos.y--;
			break;
		case 2://direction right up
			endpos.x++;
			endpos.y--;
			break;
		case 3://direction right
			endpos.x++;
			break;
		case 4://direction right down
			endpos.x++;
			endpos.y++;
			break;
		case 5://direction down
			endpos.y++;
			break;
		case 6://direction left down
			endpos.x--;
			endpos.y++;
			break;
		case 7://direction left
			endpos.x--;
			break;
		default:
			break;
		}
		imgD.at<unsigned char>(endpos) = pixVal;
		imgSD.at<unsigned char>(endpos) = 1;
		addArea++;
		nrAdds++;
		if (addArea >= maxAreaReg)
		{
			return false;
		}
		vector<int32_t> extension = getPossibleDirections(endpos, mask, regMask, imgD, siM1);
		if (extension.size() > 1)//Check if we can add addition pixels without blocking the way for the next iteration
		{
			int32_t noExt[3];
			noExt[0] = (directions[diri] + 1) % 8;
			noExt[1] = directions[diri];
			noExt[2] = (directions[diri] + 7) % 8;
			//Prevent adding additional pixels to the main direction and its immediate neighbor directions
			for (int i = extension.size()-1; i >= 0; i--)
			{
				if ((extension[i] == noExt[0]) ||
					(extension[i] == noExt[1]) || 
					(extension[i] == noExt[2]))
				{
					extension.erase(extension.begin() + i);
					if(i > 0)
						i++;
				}
			}
			if (extension.size() > 1)
			{
				//Choose a random number of additional pixels to add
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
						switch (extension[pos])
						{
						case 0://direction left up
							singleExt.x--;
							singleExt.y--;
							break;
						case 1://direction up
							singleExt.y--;
							break;
						case 2://direction right up
							singleExt.x++;
							singleExt.y--;
							break;
						case 3://direction right
							singleExt.x++;
							break;
						case 4://direction right down
							singleExt.x++;
							singleExt.y++;
							break;
						case 5://direction down
							singleExt.y++;
							break;
						case 6://direction left down
							singleExt.x--;
							singleExt.y++;
							break;
						case 7://direction left
							singleExt.x--;
							break;
						default:
							break;
						}
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
std::vector<int32_t> genStereoSequ::getPossibleDirections(cv::Point_<int32_t> &startpos, cv::Mat &mask, cv::Mat &regMask, cv::Mat &imgD, cv::Size &siM1)
{
	Mat directions = Mat::ones(3, 3, CV_8UC1);
	unsigned char atBorderX = 0, atBorderY = 0;
	if (startpos.x <= 0)
	{
		directions.col(0) = Mat::zeros(3, 1, CV_8UC1);
		atBorderX = 0x1;
	}
	if (startpos.x >= siM1.width)
	{
		directions.col(2) = Mat::zeros(3, 1, CV_8UC1);
		atBorderX = 0x2;
	}
	if (startpos.y <= 0)
	{
		directions.row(0) = Mat::zeros(1, 3, CV_8UC1);
		atBorderY = 0x1;
	}
	if (startpos.y >= siM1.height)
	{
		directions.row(2) = Mat::zeros(1, 3, CV_8UC1);
		atBorderY = 0x2;
	}

	Range irx, iry, drx, dry;
	if (atBorderX)
	{
		const unsigned char atBorderXn = ~atBorderX;
		const unsigned char v1 = (atBorderXn & 0x1);
		const unsigned char v2 = (atBorderXn & 0x2) + ((atBorderX & 0x2) >> 1);
		irx = Range(startpos.x - (int32_t)v1, startpos.x + (int32_t)v2);
		drx = Range((int32_t)(~v1), 1 + (int32_t)v2);
		if (atBorderY)
		{ 
			const unsigned char atBorderYn = ~atBorderY;
			const unsigned char v3 = (atBorderYn & 0x1);
			const unsigned char v4 = (atBorderYn & 0x2) + ((atBorderY & 0x2) >> 1);
			iry = Range(startpos.y - (int32_t)v3, startpos.y + (int32_t)v4);
			dry = Range((int32_t)(~v3), 1 + (int32_t)v4);
		}
		else
		{
			iry = Range(startpos.y - 1, startpos.y + 2);
			dry = Range::all();
		}
	}
	else if (atBorderY)
	{
		unsigned char atBorderYn = ~atBorderY;
		iry = Range(startpos.y - (atBorderYn & 0x1), startpos.y + (atBorderYn & 0x2) + ((atBorderY & 0x2) >> 1));
		irx = Range(startpos.x - 1, startpos.x + 2);
		drx = Range::all();
	}
	else
	{
		irx = Range(startpos.x - 1, startpos.x + 2);
		iry = Range(startpos.y - 1, startpos.y + 2);
		drx = Range::all();
		dry = Range::all();
	}

	directions(dry, drx) &= (imgD(iry, irx) == 0) & mask(iry, irx) & regMask(iry, irx);

	vector<int32_t> dirs;
	for (int32_t i = 0; i < 9; i++)
	{
		if (directions.at<bool>(i))
		{
			switch (i)
			{
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

	return dirs;
}

//Generates correspondences and 3D points in the camera coordinate system (including false matches)
void genStereoSequ::getKeypoints()
{
	int32_t kSi = csurr.rows;
	
	//Mark used areas (by correspondences and TN) in the second image
	Mat cImg2 = Mat::zeros(imgSize.width + kSi - 1, imgSize.height + kSi - 1, CV_8UC1);
	for (int i = 0; i < actCorrsImg2TPFromLast.cols; i++)
	{
		Point_<int32_t> pt((int32_t)round(actCorrsImg2TPFromLast.at<double>(0, i)), (int32_t)round(actCorrsImg2TPFromLast.at<double>(1, i)));
		Mat s_tmp = cImg2(Rect(pt, Size(kSi, kSi)));
		s_tmp += csurr;
	}
	for (int i = 0; i < actCorrsImg2TNFromLast.cols; i++)
	{
		Point_<int32_t> pt((int32_t)round(actCorrsImg2TNFromLast.at<double>(0, i)), (int32_t)round(actCorrsImg2TNFromLast.at<double>(1, i)));
		Mat s_tmp = cImg2(Rect(pt, Size(kSi, kSi)));
		s_tmp += csurr;
	}

	//Get regions of backprojected TN in first image and mark their positions
	vector<vector<vector<Point_<int32_t>>>> x1pTN(3, vector<vector<Point_<int32_t>>>(3));
	Size rSl(imgSize.width / 3, imgSize.height / 3);
	for (int i = 0; i < actCorrsImg1TNFromLast.cols; i++)
	{
		Point_<int32_t> pt((int32_t)round(actCorrsImg1TNFromLast.at<double>(0, i)), (int32_t)round(actCorrsImg1TNFromLast.at<double>(1, i)));
		Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
		s_tmp += csurr;

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
			int32_t nrNear = (int32_t)floor(depthsPerRegion[actCorrsPRIdx][y][x].near * (double)nrTruePosRegs[actFrameCnt].at<int32_t>(y, x));
			int32_t nrFar = (int32_t)floor(depthsPerRegion[actCorrsPRIdx][y][x].far * (double)nrTruePosRegs[actFrameCnt].at<int32_t>(y, x));
			int32_t nrMid = nrTruePosRegs[actFrameCnt].at<int32_t>(y, x) - nrNear - nrFar;

			int32_t nrTN = nrTrueNegRegs[actFrameCnt].at<int32_t>(y, x) - (int32_t)x1pTN[y][x].size();

			int32_t maxSelect = max(3 * nrTruePosRegs[actFrameCnt].at<int32_t>(y, x), 1000);
			int32_t maxSelect2 = 50;
			int32_t maxSelect3 = 50;
			std::uniform_real_distribution<int32_t> distributionX(regROIs[y][x].x, regROIs[y][x].x + regROIs[y][x].width);
			std::uniform_real_distribution<int32_t> distributionY(regROIs[y][x].y, regROIs[y][x].y + regROIs[y][x].height);

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

			while (((nrNear > 0) || (nrFar > 0) || (nrMid > 0)) && (maxSelect2 > 0) && (maxSelect3 > 0))
			{
				pt.x = distributionX(rand_gen);
				pt.y = distributionY(rand_gen);

				if (depthAreaMap.at<unsigned char>(pt) == 1)
				{
					maxSelect--;
					if ((nrNear <= 0) && (maxSelect >= 0)) continue;
					//Check if coordinate is too near to existing keypoint
					Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
					if (sum(s_tmp)[0] > 0)
					{
						maxSelect++;
						maxSelect2--;
						continue;
					}
					maxSelect2 = 50;
					s_tmp += csurr;
					//Check if it is also an inlier in the right image
					if (!checkLKPInlier(pt, pt2, pCam))
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
					if (sum(s_tmp)[0] > 0)
					{
						maxSelect++;
						maxSelect2--;
						continue;
					}
					maxSelect2 = 50;
					s_tmp += csurr;
					//Check if it is also an inlier in the right image
					if (!checkLKPInlier(pt, pt2, pCam))
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
					if (sum(s_tmp)[0] > 0)
					{
						maxSelect++;
						maxSelect2--;
						continue;
					}
					maxSelect2 = 50;
					s_tmp += csurr;
					//Check if it is also an inlier in the right image
					if (!checkLKPInlier(pt, pt2, pCam))
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
			if(!p3DTPnewRNear.empty())
				std::copy(p3DTPnewRNear.begin(), p3DTPnewRNear.end(), p3DTPnew[y][x].end());
			if (!p3DTPnewRMid.empty())
				std::copy(p3DTPnewRMid.begin(), p3DTPnewRMid.end(), p3DTPnew[y][x].end());
			if (!p3DTPnewRFar.empty())
				std::copy(p3DTPnewRFar.begin(), p3DTPnewRFar.end(), p3DTPnew[y][x].end());

			if (!corrsNearR.empty())
				std::copy(corrsNearR.begin(), corrsNearR.end(), corrsAllD[y][x].end());
			if (!corrsMidR.empty())
				std::copy(corrsMidR.begin(), corrsMidR.end(), corrsAllD[y][x].end());
			if (!corrsFarR.empty())
				std::copy(corrsFarR.begin(), corrsFarR.end(), corrsAllD[y][x].end());

			if (!corrsNearR2.empty())
				std::copy(corrsNearR2.begin(), corrsNearR2.end(), corrsAllD2[y][x].end());
			if (!corrsMidR2.empty())
				std::copy(corrsMidR2.begin(), corrsMidR2.end(), corrsAllD2[y][x].end());
			if (!corrsFarR2.empty())
				std::copy(corrsFarR2.begin(), corrsFarR2.end(), corrsAllD2[y][x].end());
			

			//Select for true negatives in image 1 true negatives in image 2
			std::uniform_real_distribution<int32_t> distributionX2(0, imgSize.width);
			std::uniform_real_distribution<int32_t> distributionY2(0, imgSize.height);
			for (size_t i = 0; i < x1TN[y][x].size(); i++)
			{
				int max_try = 10;
				while (max_try > 0)
				{
					pt.x = distributionX2(rand_gen);
					pt.y = distributionY2(rand_gen);
					Mat s_tmp = cImg2(Rect(pt, Size(kSi, kSi)));
					if (sum(s_tmp)[0] > 0)
					{
						max_try--;
						continue;
					}
					s_tmp += csurr;
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
			
			//Get the rest of TN correspondences
			std::normal_distribution<double> distributionNX2(0, max(imgSize.width / 48, 10));
			std::normal_distribution<double> distributionNY2(0, max(imgSize.width / 48, 10));
			maxSelect2 = 50;
			maxSelect3 = max(3 * nrTN, 500);
			while ((nrTN > 0) && (maxSelect2 > 0) && (maxSelect3 > 0))
			{
				pt.x = distributionX(rand_gen);
				pt.y = distributionY(rand_gen);

				Mat s_tmp = corrsIMG(Rect(pt, Size(kSi, kSi)));
				if (sum(s_tmp)[0] > 0)
				{
					maxSelect2--;
					continue;
				}
				maxSelect2 = 50;
				s_tmp += csurr;
				x1TN[y][x].push_back(Point2d((double)pt.x, (double)pt.y));
				int max_try = 10;
				double perfDist = 50.0;
				if (!checkLKPInlier(pt, pt2, pCam))//Take a random corresponding point in the second image if the reprojection is not visible to get a TN
				{
					while (max_try > 0)
					{
						pt.x = distributionX2(rand_gen);
						pt.y = distributionY2(rand_gen);
						Mat s_tmp1 = cImg2(Rect(pt, Size(kSi, kSi)));
						if (sum(s_tmp1)[0] > 0)
						{
							max_try--;
							continue;
						}
						s_tmp1 += csurr;
						break;
					}
					pt2 = Point2d((double)pt.x, (double)pt.y);
				}
				else//Distort the reprojection in the second image to get a TN
				{
					Point2d ptd;
					while (max_try > 0)
					{						
						do
						{
							ptd.x = distributionNX2(rand_gen);
							ptd.x += 0.75 * ptd.x / abs(ptd.x);
							ptd.x *= 1.5;
							ptd.y = distributionNY2(rand_gen);
							ptd.y += 0.75 * ptd.y / abs(ptd.y);
							ptd.y *= 1.5;
						} while ((abs(ptd.x) < 1.5) && (abs(ptd.y) < 1.5));
						pt2 += ptd;
						
						Mat s_tmp1 = cImg2(Rect((int)round(pt2.x), (int)round(pt2.y), kSi, kSi));
						if (sum(s_tmp1)[0] > 0)
						{
							max_try--;
							continue;
						}
						s_tmp1 += csurr;
						perfDist = norm(ptd);
						break;
					}
				}
				if (max_try <= 0)
				{
					maxSelect3--;
					x1TN[y][x].pop_back();
					s_tmp -= csurr;
					continue;
				}
				x2TN[y][x].push_back(pt2);
				x2TNdistCorr[y][x].push_back(perfDist);
				nrTN--;
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
				std::copy(p3DTPnew[y][x].begin(), p3DTPnew[y][x].end(), actImgPointCloud.end());
			if (!x2TNdistCorr[y][x].empty())
				std::copy(x2TNdistCorr[y][x].begin(), x2TNdistCorr[y][x].end(), distTNtoReal.end());
			
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

//Check, if the given point in the first camera is also visible in the second camera
//Calculates the 3D-point in the camera coordinate system and the corresponding point in the second image
bool genStereoSequ::checkLKPInlier(cv::Point_<int32_t> pt, cv::Point2d &pt2, cv::Point3d &pCam)
{
	Mat x = (Mat_<double>(3, 1) << (double)pt.x, (double)pt.y, 1.0);

	double depth = depthMap.at<double>(pt);
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

	if (pars.startPosMovObjs.empty())
	{
		startPosMovObjs = Mat::zeros(3, 3, CV_8UC1);
		for (size_t y = 0; y < 3; y++)
		{
			for (size_t x = 0; x < 3; x++)
			{
				startPosMovObjs.at<unsigned char>(y, x) = (unsigned char)(rand() % 2);
			}
		}
	}
	else
	{
		startPosMovObjs = pars.startPosMovObjs.getMat();
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
	int maxOPerReg = (int)ceil((float)pars.nrMovObjs / (float)nrStartA);
	int area23 = 2 * imgArea / 3;//The moving objects should
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
	std::uniform_real_distribution<int32_t> distribution((int32_t)minOArea, (int32_t)maxOArea);
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
				std::uniform_real_distribution<int> distributionX(regROIs[y][x].x, regROIs[y][x].x + regROIs[y][x].width);
				std::uniform_real_distribution<int> distributionY(regROIs[y][x].y, regROIs[y][x].y + regROIs[y][x].height);
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
						vector<int> xInterVals, yInterVals;
						vector<double> xWeights, yWeights;

						//Get possible selection ranges for x-values
						int start = max(xposes[0] - minODist, regROIs[y][x].x);
						int maxEnd = regROIs[y][x].x + regROIs[y][x].width;
						int xyend = min(xposes[0] + minODist, maxEnd);
						if (start == regROIs[y][x].x)
						{
							xInterVals.push_back(start);
							xInterVals.push_back(xposes[0] + minODist);
							xWeights.push_back(0);
						}
						else
						{
							xInterVals.push_back(regROIs[y][x].x);
							xInterVals.push_back(start);
							xWeights.push_back(1.0);
							if (xyend != maxEnd)
							{
								xInterVals.push_back(xyend);
								xWeights.push_back(0);
							}
						}
						if (xyend != maxEnd)
						{
							for (size_t i = 1; i < xposes.size(); i++)
							{
								start = max(xposes[i] - minODist, xInterVals.back());
								if (start != xInterVals.back())
								{
									xInterVals.push_back(xposes[i] - minODist);
									xWeights.push_back(1.0);
								}								
								xyend = min(xposes[i] + minODist, maxEnd);
								if (xyend != maxEnd)
								{
									xInterVals.push_back(xyend);
									xWeights.push_back(0);
								}
							}
						}
						if (xyend != maxEnd)
						{
							xInterVals.push_back(maxEnd);
							xWeights.push_back(1.0);
						}

						//Get possible selection ranges for y-values
						start = max(yposes[0] - minODist, regROIs[y][x].y);
						maxEnd = regROIs[y][x].y + regROIs[y][x].height;
						int xyend = min(yposes[0] + minODist, maxEnd);
						if (start == regROIs[y][x].y)
						{
							yInterVals.push_back(start);
							yInterVals.push_back(yposes[0] + minODist);
							yWeights.push_back(0);
						}
						else
						{
							yInterVals.push_back(regROIs[y][x].y);
							yInterVals.push_back(start);
							yWeights.push_back(1.0);
							if (xyend != maxEnd)
							{
								yInterVals.push_back(xyend);
								yWeights.push_back(0);
							}
						}
						if (xyend != maxEnd)
						{
							for (size_t i = 1; i < yposes.size(); i++)
							{
								start = max(yposes[i] - minODist, yInterVals.back());
								if (start != yInterVals.back())
								{
									yInterVals.push_back(yposes[i] - minODist);
									yWeights.push_back(1.0);
								}
								xyend = min(yposes[i] + minODist, maxEnd);
								if (xyend != maxEnd)
								{
									yInterVals.push_back(xyend);
									yWeights.push_back(0);
								}
							}
						}
						if (xyend != maxEnd)
						{
							yInterVals.push_back(maxEnd);
							yWeights.push_back(1.0);
						}

						//Create piecewise uniform distribution and get a random seed
						piecewise_constant_distribution<int> distrPieceX(xInterVals.begin(), xInterVals.end(), xWeights.begin());
						piecewise_constant_distribution<int> distrPieceY(yInterVals.begin(), yInterVals.end(), yWeights.begin());
						movObjSeeds[y][x].push_back(cv::Point_<int32_t>(distrPieceX(rand_gen), distrPieceY(rand_gen)));
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
//mask is used to exclude areas from generating labels and must have the same size as the image
//seeds must hold the seeding positions for generating the labels
//areas must hold the desired area for every label
void genStereoSequ::generateMovObjLabels(cv::Mat &mask, std::vector<cv::Point_<int32_t>> &seeds, std::vector<int32_t> &areas)
{
	CV_Assert(seeds.size() == areas.size());

	size_t nr_movObj = areas.size();

	movObjLabels.clear();
	movObjLabels.resize(nr_movObj, cv::Mat::zeros(imgSize, CV_8UC1));
	Mat combLabels = cv::Mat::zeros(imgSize, CV_8UC1);
	//Set seeding positions in mov. obj. label images
	for (size_t i = 0; i < nr_movObj; i++)
	{
		movObjLabels[i].at<unsigned char>(seeds[i]) = 1;
		combLabels.at<unsigned char>(seeds[i]) = (unsigned char)i;
	}
	Size siM1(imgSize.width - 1, imgSize.height - 1);
	Rect imgArea = Rect(Point(0, 0), imgSize);
	Mat regMask = cv::Mat::ones(imgSize, CV_8UC1);
	std::vector<cv::Point_<int32_t>> startposes = seeds;
	vector<int32_t> actArea(nr_movObj, 1);
	vector<size_t> nrIterations(nr_movObj, 0);
	vector<unsigned char> dilateOps(nr_movObj, 0);
	vector<bool> objNFinished(nr_movObj, true);
	int remainObj = (int)nr_movObj;

	//Generate labels
	while (remainObj > 0)
	{
		for (size_t i = 0; i < nr_movObj; i++)
		{
			if (objNFinished[i])
			{
				if (!addAdditionalDepth((unsigned char)i,
					combLabels,
					movObjLabels[i],
					mask,
					regMask,
					startposes[i],
					startposes[i],
					actArea[i],
					areas[i],
					siM1,
					seeds[i],
					imgArea,
					nrIterations[i],
					dilateOps[i]))
				{
					objNFinished[i] = false;
					remainObj--;
				}
			}
		}
	}

	//Get overlap of regions and the portion of correspondences that is covered by the moving objects
	vector<vector<double>> movObjOverlap(3, vector<double>(3, 0));
	vector<vector<bool>> movObjHasArea(3, vector<bool>(3, false));
	vector<vector<int32_t>> movObjCorrsFromStatic(3, vector<int32_t>(3, 0));
	vector<vector<int32_t>> movObjCorrsFromStaticInv(3, vector<int32_t>(3, 0));
	int32_t absNrCorrsFromStatic = 0;
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			movObjOverlap[y][x] = (double)cv::countNonZero(combLabels(regROIs[y][x])) / (double)(regROIs[y][x].area());
			if (movObjOverlap[y][x] > 0.9)
			{
				movObjHasArea[y][x] = true;
				movObjCorrsFromStatic[y][x] = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x);
				movObjCorrsFromStaticInv[y][x] = 0;
				absNrCorrsFromStatic += movObjCorrsFromStatic[y][x];
			}
			else
			{
				movObjCorrsFromStatic[y][x] = (int32_t)round((double)nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) * movObjOverlap[y][x]);
				movObjCorrsFromStaticInv[y][x] = nrCorrsRegs[actFrameCnt].at<int32_t>(y, x) - movObjCorrsFromStatic[y][x];
				absNrCorrsFromStatic += movObjCorrsFromStatic[y][x];
			}
		}
	}

	double areaFracStaticCorrs = (double)absNrCorrsFromStatic / (double)nrCorrs[actFrameCnt];
	double r_CorrMovObjPort = round(pars.CorrMovObjPort * 100.0) / 100.0;
	double r_areaFracStaticCorrs = round(areaFracStaticCorrs * 100.0) / 100.0;
	Mat statCorrsPRegNew = Mat::zeros(3, 3, CV_32SC1);
	int32_t actCorrsOnMovObj = (int32_t)round(pars.CorrMovObjPort * (double)nrCorrs[actFrameCnt]);
	if (r_CorrMovObjPort > r_areaFracStaticCorrs)//Remove additional static correspondences and add them to the moving objects
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
			while (remStatrem > 0)
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
			}
		}
	}
	else if (r_CorrMovObjPort < r_areaFracStaticCorrs)//Distribute a part of the correspondences from moving objects over the static elements not covered by moving objects
	{
		int32_t remMov = absNrCorrsFromStatic - actCorrsOnMovObj;
	}

	//Set new static correspondences
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
}