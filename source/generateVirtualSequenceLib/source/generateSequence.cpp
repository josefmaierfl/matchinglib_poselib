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

	
	adaptDepthsPerRegion();
}

//Initialize fraction of correspondences per image region
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

	//Generate absolute number of correspondences per image region and frame

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
	//Check if the sum of fractions is 1.0
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			pars.depthsPerRegion[i][j].sumTo1();
		}
	}
	pars.corrsPerDepth.sumTo1();
	
	depthsPerRegion = std::vector<std::vector<std::vector<depthPortion>>>(pars.corrsPerRegion.size(), pars.depthsPerRegion);

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