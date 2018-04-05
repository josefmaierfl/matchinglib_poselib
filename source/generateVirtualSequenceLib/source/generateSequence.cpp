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
	size_t maxElems = 3 * imgSize.area() / (9 * (16 + 2 * 4));//3 -> from (near, mid, far) depths, 9 is the nr of regions, 16 is the min area and 2*4 = 2*sqrt(16) is the gap between areas;
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
				if ((tmp != 0) && (tmp < 16))
					tmp = 16;
				areaPRegNear[i].at<int32_t>(y, x) = tmp;

				tmp = (int32_t)round(depthsPerRegion[i][y][x].mid * (double)regArea);
				if ((tmp != 0) && (tmp < 16))
					tmp = 16;
				areaPRegMid[i].at<int32_t>(y, x) = tmp;

				tmp = (int32_t)round(depthsPerRegion[i][y][x].far * (double)regArea);
				if ((tmp != 0) && (tmp < 16))
					tmp = 16;
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
	dimgWH.maxDist = 100.0 * actDepthFar;

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
	
	if (!actCorrsImg1TPFromLast.empty())//Take seeding postions from backprojected coordinates
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
		int posadd = (int)floor(pars.minKeypDist);
		int sqrSi = 2 * posadd;
		cv::Mat filtInitPts = Mat::zeros(imgSize.width + sqrSi, imgSize.height + sqrSi, CV_8UC1);
		sqrSi++;//sqrSi = 2 * (int)floor(pars.minKeypDist) + 1;
		Mat csurr = Mat::ones(sqrSi, sqrSi, CV_8UC1);
		int maxSum = sqrSi * sqrSi;
		vector<size_t> delListCorrs, delList3D;
		if (!seedsNear_tmp.empty())
		{
			for (size_t i = 0; i < seedsNear_tmp.size(); i++)
			{
				Mat s_tmp = filtInitPts(Range(seedsNear_tmp[i].y, seedsNear_tmp[i].y + sqrSi), Range(seedsNear_tmp[i].x, seedsNear_tmp[i].x + sqrSi));
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
				Mat s_tmp = filtInitPts(Range(seedsMid_tmp[i].y, seedsMid_tmp[i].y + sqrSi), Range(seedsMid_tmp[i].x, seedsMid_tmp[i].x + sqrSi));
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
				Mat s_tmp = filtInitPts(Range(seedsFar_tmp[i].y, seedsFar_tmp[i].y + sqrSi), Range(seedsFar_tmp[i].x, seedsFar_tmp[i].x + sqrSi));
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
		if (!delListCorrs.empty())
		{
			std::vector<cv::Point3d> actImgPointCloudFromLast_tmp;
			cv::Mat actCorrsImg1TPFromLast_tmp, actCorrsImg2TPFromLast_tmp;
			std::vector<size_t> actCorrsImg12TPFromLast_Idx_tmp;
			//Correct actCorrsImg1TNFromLast_Idx and actCorrsImg2TNFromLast_Idx

			sort(delList3D.begin(), delList3D.end(), [](size_t first, size_t second) {return first > second; });

			std::vector<pair<size_t, size_t>> actCorrsImg1TNFromLast_Idx_tmp(actCorrsImg1TNFromLast_Idx.size());
			for (size_t i = 0; i < actCorrsImg1TNFromLast_Idx.size(); i++)
			{
				actCorrsImg1TNFromLast_Idx_tmp[i] = make_pair(actCorrsImg1TNFromLast_Idx[i], i);
			}
			sort(actCorrsImg1TNFromLast_Idx_tmp.begin(), actCorrsImg1TNFromLast_Idx_tmp.end(), [](pair<size_t, size_t> first, pair<size_t, size_t> second) {return first.first > second.first; });
			size_t idx = 0;
			for (size_t i = 0; i < actCorrsImg1TNFromLast_Idx_tmp.size(); i++)
			{
				if (actCorrsImg1TNFromLast_Idx_tmp[i].first > delList3D[idx])
				{
					actCorrsImg1TNFromLast_Idx_tmp[i].first -= idx;
				}
				else
				{
					while (actCorrsImg1TNFromLast_Idx_tmp[i].first > delList3D[idx])
					{
						idx++;///////////////////////////////
					}
				}
			}
		}
	}
}



void genStereoSequ::combineDepthMaps()
{
	std::vector<std::vector<depthPortion>> actDepthsPerRegion;
	if (actImgPointCloud.empty())
	{

	}
	else
	{

	}
}

void genDepthMaps(cv::Mat map, std::vector<cv::Point3d> mapSeed, bool noSeedOrder, cv::InputArray mask)
{
	//mapSeed: x,y image coordinates and 3D distance
	if (mapSeed.empty())
	{
		//Generate depth seeds
	}
	else
	{
		if (noSeedOrder)
		{

		}
		else
		{

		}
	}
}