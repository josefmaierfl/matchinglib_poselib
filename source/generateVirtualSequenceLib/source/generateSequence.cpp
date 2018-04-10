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

	//Number of stereo configurations
	nrStereoConfs = R.size();

	//Construct the camera path
	constructCamPath();

	//Calculate the thresholds for the depths near, mid, and far for every camera configuration
	getDepthRanges();

	//Used inlier ratios
	genInlierRatios();

	//Number of correspondences per image
	genNrCorrsImg();

	//Correspondences per image regions
	initFracCorrImgReg();

	//Depths per image region
	adaptDepthsPerRegion();

	//Check if the given ranges of connected depth areas per image region are correct and initialize them for every definition of depths per image region
	checkDepthAreas();

	//Calculate the area in pixels for every depth and region
	calcPixAreaPerDepth();
}

//Initialize fraction of correspondences per image region and calculate the absolute number of TP/TN correspondences per image region
void genStereoSequ::initFracCorrImgReg()
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

		//Get number of tru positives per region
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
		inlRat.resize(totalNrFrames, pars.inlRatRange.first);
	}
	else
	{
		double initInlRat = getRandDoubleValRng(pars.inlRatRange.first, pars.inlRatRange.second, rand_gen);
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

/* As the user can specify portions of different depths (neaer, mid, far) globally for the whole image and also for regions within the image,
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

void genStereoSequ::checkDepthSeeds()
{
	seedsNear = std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>>(3, std::vector<std::vector<cv::Point3_<int32_t>>>(3));
	seedsMid = std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>>(3, std::vector<std::vector<cv::Point3_<int32_t>>>(3));
	seedsFar = std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>>(3, std::vector<std::vector<cv::Point3_<int32_t>>>(3));

	int posadd1 = max((int)floor(pars.minKeypDist), (int)sqrt(minDArea));
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
		int posadd = max((int)floor(pars.minKeypDist), 1);
		int sqrSi = 2 * posadd;
		//cv::Mat filtInitPts = Mat::zeros(imgSize.width + sqrSi, imgSize.height + sqrSi, CV_8UC1);
		sqrSi++;//sqrSi = 2 * (int)floor(pars.minKeypDist) + 1;
		Mat csurr = Mat::ones(sqrSi, sqrSi, CV_8UC1);
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

void genStereoSequ::genDepthMaps()
{
	/*
	std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> seedsNear;//Holds the actual near seeds for every region; Size 3x3xn; Point3 holds the seed coordinate (x,y) and a possible index (=z) to actCorrsImg12TPFromLast_Idx if the seed was generated from an existing 3D point (otherwise z=-1).
	std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> seedsMid;//Holds the actual mid seeds for every region; Size 3x3xn; Point3 holds the seed coordinate (x,y) and a possible index (=z) to actCorrsImg12TPFromLast_Idx if the seed was generated from an existing 3D point (otherwise z=-1).
	std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> seedsFar*/
	/*
	std::vector<cv::Mat> areaPRegNear;//Type CV_32SC1, same size as depthsPerRegion
	std::vector<cv::Mat> areaPRegMid;//Type CV_32SC1, same size as depthsPerRegion
	std::vector<cv::Mat> areaPRegFar;//Type CV_32SC1, same size as depthsPerRegion*/
	//actCorrsPRIdx

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

	//Construct valid areas for every region
	vector<vector<Mat>> regmasks(3, vector<Mat>(3, Mat::zeros(imgSize, CV_8UC1)));
	vector<vector<cv::Rect>> regmasksROIs(3, vector<cv::Rect>(3));
	vector<vector<cv::Rect>> regROIs(3, vector<cv::Rect>(3));
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
	Mat actUsedAreas = Mat::zeros(imgSize, CV_8UC1);
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
					actUsedAreas.at<unsigned char>(iy, ix) = 1;
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
					actUsedAreas.at<unsigned char>(iy, ix) = 2;
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
					actUsedAreas.at<unsigned char>(iy, ix) = 3;
					actUsedAreaFar.at<unsigned char>(iy, ix) = 3;
					actAreaFar[y][x]++;
				}
			}
		}
	}

	//Create depth areas beginning with the smallest areas (near, mid, or far) per region
	//Also create depth areas for the second smallest areas
	Size imgSiM1 = Size(imgSi13.width - 1, imgSiM1.height - 1);
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
									actUsedAreas,
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
									actUsedAreas,
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
									actUsedAreas,
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
			Mat fillMask = (actUsedAreas(regROIs[y][x]) == 0) & Mat::ones(regROIs[y][x].height, regROIs[y][x].width, CV_8UC1);
			switch (beginDepth.at<cv::Vec<int32_t, 3>>(y, x)[2])
			{
			case 0:
				actUsedAreaNear(regROIs[y][x]) |= fillMask;
				//actUsedAreas(regROIs[y][x]) |= fillMask;
				break;
			case 1:
				actUsedAreaMid(regROIs[y][x]) |= fillMask;
				/*fillMask *= 2;
				actUsedAreas(regROIs[y][x]) |= fillMask;*/
				break;
			case 2:
				actUsedAreaMid(regROIs[y][x]) |= fillMask;
				/*fillMask *= 3;
				actUsedAreas(regROIs[y][x]) |= fillMask;*/
				break;
			default:
				break;
			}
		}
	}

	//Get final depth values for each depth region
	Mat depthMapNear, depthMapMid, depthMapFar;
	getDepthMaps(depthMapNear, actUsedAreaNear, actDepthNear, actDepthMid, seedsNear);
	getDepthMaps(depthMapMid, actUsedAreaMid, actDepthMid, actDepthFar, seedsMid);
	getDepthMaps(depthMapFar, actUsedAreaFar, actDepthFar, maxFarDistMultiplier * actDepthFar, seedsFar);

	//Combine the 3 depth maps to a single depth map
	depthMap = depthMapNear + depthMapMid + depthMapFar;
}

//Generate depth values (for every pixel) for the given areas of depth regions taking into account the depth values from backprojected 3D points
void genStereoSequ::getDepthMaps(cv::Mat &dout, cv::Mat &din, double dmin, double dmax, std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> &initSeeds)
{
	std::vector<cv::Point3_<int32_t>> initSeedInArea;
	//Check, if there are depth seeds available that were already backprojected from 3D
	for (size_t y = 0; y < 3; y++)
	{
		for (size_t x = 0; x < 3; x++)
		{
			for (size_t i = 0; i < initSeeds[y][x].size(); i++)
			{
				if (initSeeds[y][x][i].z >= 0)
					initSeedInArea.push_back(initSeeds[y][x][i]);
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

/* Adds a few random depth pixels near a given position
unsigned char pixVal	Value assigned to the random pixel positions
cv::Mat &imgD			Image holding all depth ranges where the new random depth pixels should be added
cv::Mat &imgSD			Image holding only one specific depth range where the new random depth pixels should be added
cv::Mat &mask			Mask for imgD and imgSD
cv::Point_<int32_t> &startpos		Start position (excluding this single location) from where to start adding new depth pixels
cv::Point_<int32_t> &endpos			End position where the last depth pixel was set
int32_t &addArea		Adds the number of newly inserted pixels to the given number
int32_t &maxAreaReg		Maximum number of specific depth pixels per image region (9x9)
cv::Size &siM1			Image size -1
cv::Point_<int32_t> &initSeed	Initial position of the seed
cv::Rect &vROI			ROI were it is actually allowed to add new pixels
size_t &nrAdds			Number of pixels that were added to this depth area (including preceding calls to this function)
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