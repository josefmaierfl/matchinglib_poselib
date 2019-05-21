/**********************************************************************************************************
FILE: getStereoCameraExtr.cpp

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: February 2018

LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for generating optimal camera paramters given a disired
overlap area ratio between the views and some restrictions on the camera parameters
**********************************************************************************************************/

#include "getStereoCameraExtr.h"
#include "helper_funcs.h"
#include "opencv2/imgproc/imgproc.hpp"
//#include <iomanip>

using namespace std;
using namespace cv;

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

/* -------------------------- Functions -------------------------- */

GenStereoPars::GenStereoPars(std::vector<std::vector<double>> tx,
	std::vector<std::vector<double>> ty,
	std::vector<std::vector<double>> tz,
	std::vector<std::vector<double>> roll,
	std::vector<std::vector<double>> pitch,
	std::vector<std::vector<double>> yaw,
	double approxImgOverlap, cv::Size imgSize):
	tx_(move(tx)),
	ty_(move(ty)),
	tz_(move(tz)),
	roll_(move(roll)),
	pitch_(move(pitch)),
	yaw_(move(yaw)),
	approxImgOverlap_(approxImgOverlap),
	imgSize_(imgSize)
{
	//srand(time(NULL));
	randSeed(rand_generator);

	//Check if the number of constraints is equal for all paramters
	nrConditions = tx_.size();
	if ((nrConditions != ty_.size()) ||
		(nrConditions != tz_.size()) ||
		(nrConditions != roll_.size()) ||
		(nrConditions != pitch_.size()) ||
		(nrConditions != yaw_.size()))
	{
		throw InvalidDataStructureException("Number of constraints not equal");
	}

	//Check the right format (first value must be smaller than second)
	checkParameterFormat(tx_, "tx");
	checkParameterFormat(ty_, "ty");
	checkParameterFormat(tz_, "tz");
	checkParameterFormat(roll_, "roll");
	checkParameterFormat(pitch_, "pitch");
	checkParameterFormat(yaw_, "yaw");

	//Check for the sign of the potential largest baseline direction x or y (must be negative)
	//and check if tx and ty components are not smaller tz
	for (size_t i = 0; i < nrConditions; i++)
	{
		double maxX, maxY, maxZ;
		if (tx_[i].size() > 1)
		{
			if (((tx_[i][0] >= 0) && (tx_[i][1] > 0)) ||
				(abs(tx_[i][1]) > abs(tx_[i][0])))
			{
				maxX = tx_[i][1];
			}
			else 
			{
				maxX = tx_[i][0];
			}
		}
		else
		{
			maxX = tx_[i][0];
		}
		if (ty_[i].size() > 1)
		{
			if (((ty_[i][0] >= 0) && (ty_[i][1] > 0)) ||
				(abs(ty_[i][1]) > abs(ty_[i][0])))
			{
				maxY = ty_[i][1];
			}
			else
			{
				maxY = ty_[i][0];
			}
		}
		else
		{
			maxY = ty_[i][0];
		}
		if (tz_[i].size() > 1)
		{
			if (((tz_[i][0] >= 0) && (tz_[i][1] > 0)) ||
				(abs(tz_[i][1]) > abs(tz_[i][0])))
			{
				maxZ = tz_[i][1];
			}
			else
			{
				maxZ = tz_[i][0];
			}
		}
		else
		{
			maxZ = tz_[i][0];
		}

		if (((abs(maxX) >= abs(maxY)) && (maxX > 0)) ||
			((abs(maxY) >= abs(maxX)) && (maxY > 0)))
		{
			throw InvalidDataStructureException("The largest translation element must be negative");
		}

		double maxTelem = abs(maxX) > abs(maxY) ? abs(maxX) : abs(maxY);
		if (abs(maxZ) >= maxTelem)
		{
			throw InvalidDataStructureException("The z-translation element must be smaller than the largest absolute value from tx and ty.");
		}
	}

	//Check if the ranges are equal
	checkEqualRanges(tx_, txRangeEqual);
	checkEqualRanges(ty_, tyRangeEqual);
	checkEqualRanges(tz_, tzRangeEqual);
	checkEqualRanges(roll_, rollRangeEqual);
	checkEqualRanges(pitch_, pitchRangeEqual);
	checkEqualRanges(yaw_, yawRangeEqual);
}

void GenStereoPars::getNewRandSeed(){
    randSeed(rand_generator);
}

inline double GenStereoPars::getRandDoubleVal(double lowerBound, double upperBound)
{
	rand_generator = std::default_random_engine((unsigned int)std::rand());
	std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
	return distribution(rand_generator);
}

cv::Mat GenStereoPars::getNormalDistributionVals(int sizeV, double mean, double stddev)
{
	cv::Mat distr = Mat(sizeV, 1, CV_64FC1);
	std::normal_distribution<double> distribution(mean, stddev);
	for (int i = 0; i < sizeV; i++)
	{
		distr.at<double>(i) = distribution(rand_generator);
	}
	return distr;
}

void GenStereoPars::checkParameterFormat(std::vector<std::vector<double>> par, std::string name)
{
	for (size_t i = 0; i < nrConditions; i++)
	{
		//double maxX, maxY;
		if (par[i].size() == 2)
		{
			if (par[i][0] >= par[i][1])
			{
				std::stringstream ss;
				ss << "Wrong order of " << name << " range parameters";
				throw InvalidDataStructureException(ss.str());
			}
		}
		else if (par[i].size() != 1)
		{
			std::stringstream ss;
			ss << name << " is not a range nor a fixed parameter";
			throw InvalidDataStructureException(ss.str());
		}
	}
}

void GenStereoPars::checkEqualRanges(std::vector<std::vector<double>> par, bool& areEqual)
{
	areEqual = true;
	if (nrConditions == 1)
	{
		return;
	}
	const bool fixedPar = (par[0].size() == 1) ? true : false;
	double minPar = par[0][0];
	double maxPar = fixedPar ? minPar : par[0][1];
	for (size_t i = 1; i < nrConditions; i++)
	{
		bool fixedPar1 = (par[i].size() == 1) ? true : false;
		if ((fixedPar ^ fixedPar1) ||
			!nearZero(minPar - par[i][0]) ||
			!nearZero(maxPar - ((par[i].size() == 1) ? par[i][0] : par[i][1])))
		{
			areEqual = false;
			return;
		}
	}
}

void GenStereoPars::initRandPars(std::vector<std::vector<double>>& parIn, bool& rangeEqual, std::vector<double>& parOut)
{
	if (rangeEqual)
	{
		double par_use_single;
		if (parIn[0].size() == 1)
		{
			par_use_single = parIn[0][0];
		}
		else
		{
			par_use_single = getRandDoubleVal(parIn[0][0], parIn[0][1]);
		}
		parOut = vector<double>(nrConditions, par_use_single);
	}
	else
	{
		for (size_t i = 0; i < nrConditions; i++)
		{
			if (parIn[i].size() == 1)
			{
				parOut.push_back(parIn[i][0]);
			}
			else
			{
				parOut.push_back(getRandDoubleVal(parIn[i][0], parIn[i][1]));
			}
		}
	}
}

int GenStereoPars::optimizeRtf(int verbose)
{
    tx_use.clear();
    ty_use.clear();
    pitch_use.clear();
    roll_use.clear();
    yaw_use.clear();
	//Set the rotation about the z axis (yaw)
	initRandPars(yaw_, yawRangeEqual, yaw_use);
	//Set the virtual image widths
	double perc = 0;
	virtWidth = vector<double>(nrConditions);
	for (size_t i = 0; i < nrConditions; i++)
	{
		getRotRectDiffArea(yaw_use[i], perc, virtWidth[i]);
	}

	//Set the z-component of the translation vector
	initRandPars(tz_, tzRangeEqual, tz_use);

	//Set the focal length to a default value
	f = (double)imgSize_.width / (2.0 * tan(PI / 4.0));

	//Camera matrix 1
	K1 = Mat::eye(3, 3, CV_64FC1);
	K1.at<double>(0, 0) = f;
	K1.at<double>(1, 1) = f;
	K1.at<double>(0, 2) = (double)imgSize_.width / 2.0;
	K1.at<double>(1, 2) = (double)imgSize_.height / 2.0;
	//Camera matrix 2
	K2 = K1.clone();

	//Set all remaining parameters to a random or given value
	initRandPars(tx_, txRangeEqual, tx_use);
	initRandPars(ty_, tyRangeEqual, ty_use);
	initRandPars(pitch_, pitchRangeEqual, pitch_use);
	initRandPars(roll_, rollRangeEqual, roll_use);

	//Check if tx is larger ty and if the larger value is negative
	//If the larger value is not negative, try to find new random values
	//And check, if the camera alignment is the same for all configurations
	size_t cnt = 0;
	const size_t maxit = 200;
	if (txRangeEqual && tyRangeEqual)
	{
		while ((((std::abs(tx_use[0]) >= std::abs(ty_use[0])) && (tx_use[0] > 0)) ||
			((std::abs(tx_use[0]) < std::abs(ty_use[0])) && (ty_use[0] > 0))) &&
			(cnt < maxit))
		{
			tx_use.clear();
			initRandPars(tx_, txRangeEqual, tx_use);
			if (((std::abs(tx_use[0]) >= std::abs(ty_use[0])) && (tx_use[0] > 0)) ||
				((std::abs(tx_use[0]) < std::abs(ty_use[0])) && (ty_use[0] > 0)))
			{
				ty_use.clear();
				initRandPars(ty_, tyRangeEqual, ty_use);
			}
			cnt++;
		}
		if (cnt >= maxit)
		{
			return -1;//Not able to generate a valid random camera configuration. Ranges of tx and ty should be adapted.
		}
	}
	else
	{
		int alignCnt = 0;
		size_t cnt2 = 0;
		const size_t maxit1 = maxit * maxit;
		for (int i = 0; i < (int)nrConditions; i++)
		{
			cnt = 0;
			int i_tmp = i;
			while ((((std::abs(tx_use[i]) >= std::abs(ty_use[i])) && (tx_use[i] > 0)) ||
				((std::abs(tx_use[i]) < std::abs(ty_use[i])) && (ty_use[i] > 0))) &&
				(cnt < maxit))
			{
				if(!helpNewRandEquRangeVals(i_tmp, maxit, 0))
					return -1;//Not able to generate a valid random camera configuration. Ranges of tx and ty should be adapted.
				cnt++;
			}
			if (cnt >= maxit)
			{
				return -1;//Not able to generate a valid random camera configuration. Ranges of tx and ty should be adapted.
			}
			if (std::abs(tx_use[i]) >= std::abs(ty_use[i]))
			{
				alignCnt++;
			}
			else
			{
				alignCnt--;
			}
			cnt2++;
			if (cnt2 >= maxit1)
			{
				return -1;//Not able to generate a valid random camera configuration. Ranges of tx and ty should be adapted.
			}
			if (i_tmp != i)
			{
				alignCnt = 0;
				i = -1;
			}
		}
		if (std::abs(alignCnt) != (int)nrConditions)//The camera alignment changes between configurations but must be the same
		{
			alignCnt = alignCnt == 0 ? 1 : alignCnt;//If both alignment options are equal often covered choose horizontal alignment
			cnt2 = 0;
			for (int i = 0; i < (int)nrConditions; i++)
			{
				cnt = 0;
				int i_tmp[2];
				i_tmp[0] = i_tmp[1] = i;
				//Generate a new random configuration for wrong alignments
				while ((((std::abs(tx_use[i]) >= std::abs(ty_use[i])) && (alignCnt < 0)) ||
					((std::abs(tx_use[i]) < std::abs(ty_use[i])) && (alignCnt > 0))) &&
					(cnt < maxit))
				{
					if (!helpNewRandEquRangeVals(i_tmp[0], maxit, alignCnt))
						return -1;//Not able to generate a valid random camera configuration. Ranges of tx and ty should be adapted.
					size_t cnt1 = 0;
					while ((((std::abs(tx_use[i]) >= std::abs(ty_use[i])) && (tx_use[i] > 0)) ||
						((std::abs(tx_use[i]) < std::abs(ty_use[i])) && (ty_use[i] > 0))) &&
						(cnt1 < maxit))
					{
						if (!helpNewRandEquRangeVals(i_tmp[1], maxit, alignCnt))
							return -1;//Not able to generate a valid random camera configuration. Ranges of tx and ty should be adapted.
						cnt1++;
					}
					if (cnt1 >= maxit)
					{
						return -1;//Not able to generate a valid random camera configuration. Ranges of tx and ty should be adapted.
					}
					cnt++;
				}
				if (cnt >= maxit)
				{
					return -1;//Not able to generate a valid random camera configuration. Ranges of tx and ty should be adapted.
				}
				cnt2++;
				if (cnt2 >= maxit1)
				{
					return -1;//Not able to generate a valid random camera configuration. Ranges of tx and ty should be adapted.
				}
				if ((i_tmp[0] != i) ||
					(i_tmp[1] != i))
				{
					i = -1;
				}
			}
		}
	}
	horizontalCamAlign = false;
	if (std::abs(tx_use[0]) >= std::abs(ty_use[0]))
	{
		horizontalCamAlign = true;
	}

	//Set the range of the focal length
	fRange[0] = (double)imgSize_.width / (2.0 * tan(angView[1] / 2.0));
	fRange[1] = (double)imgSize_.width / (2.0 * tan(angView[0] / 2.0));

	int err = optParLM(verbose);
	if (err)
		return err-1; //Not able to reach desired result (-2) or paramters are not usable (-3)

	return 0;	
}

int GenStereoPars::optParLM(int verbose)
{
	CV_Assert(nr_residualsPCond == 9);//If the number of residuals per condition changes, then fixedFuncTol must be adapted

	//Set image coordinates for optimization
	setCoordsForOpti();

	//Generate parameter vector and tolerance vectors
	Mat p = Mat((int)(nrConditions * 4 + 1), 1, CV_64FC1);
	Mat xTol = Mat(p.size(), CV_64FC1);
	Mat funcTol = Mat::ones((int)(nrConditions * nr_residualsPCond), 1, CV_64FC1) * 1e-4;
	const double fixedFuncTol[] = { 1e-3, 0.05, 0.05, 0.05, 0.05, 1e-4, 0.35, 0.5, 0.5 };
	for (int i = 0; i < (int)nrConditions; i++)
	{
		p.at<double>(i * 4) = pitch_use[i];
		p.at<double>(i * 4 + 1) = roll_use[i];
		p.at<double>(i * 4 + 2) = tx_use[i];
		p.at<double>(i * 4 + 3) = ty_use[i];

		xTol.at<double>(i * 4) = pitchTol_;
		xTol.at<double>(i * 4 + 1) = rollTol_;
		xTol.at<double>(i * 4 + 2) = txTol_;
		xTol.at<double>(i * 4 + 3) = tyTol_;

		memcpy(funcTol.data + i * nr_residualsPCond * sizeof(double), fixedFuncTol, nr_residualsPCond * sizeof(double));
	}
	p.at<double>((int)(nrConditions * 4)) = f;
	xTol.at<double>((int)(nrConditions * 4)) = fTol_;

	//Call Levenberg Marquardt
	cv::Mat p_new, residuals;
	bool aliChange = true;
	bool nZ = true;
	const size_t lmCntMax = 10;
	size_t cnt = 0;
	Mat p_new_save, residuals_save;
	double ssq_old;
	vector<Mat> p_history, r_history;
	vector<double> ssq_history;	
	while (aliChange && (cnt < lmCntMax))
	{
		aliChange = false;
		int cntit = LMFsolve(p, p_new, residuals, funcTol, xTol, 100, verbose);

		double ssq = residuals.dot(residuals);
		if (cntit < 0)
			cout << "LM was maybe not successful!" << endl;
		else
			cout << "LM finsihed after " << cntit << " iterations with sum of squared residuals " << ssq << endl;

		ssq_history.push_back(ssq);
		p_history.push_back(p_new.clone());
		r_history.push_back(residuals.clone());

		//Check if the alignment has changed
		int aliSum = 0;
		for (int i = 0; i < (int)nrConditions; i++)
		{
			if (std::abs(p_new.at<double>(i * 4 + 2)) >= std::abs(p_new.at<double>(i * 4 + 3)))
				aliSum++;
			else
				aliSum--;
		}

		double alihelp = (double)aliSum / (double)nrConditions;
		if ((((alihelp > 0) && !horizontalCamAlign) ||
			((alihelp < 0) && horizontalCamAlign)) &&
			!nearZero(alihelp))
		{
			aliChange = true;
			nZ = true;
			cout << "Camera alignment changed! Starting new optimization with this alignment." << endl;
		}
		else if (nearZero(alihelp))
		{
			if (nZ)
			{
				aliChange = true;
				nZ = false;
				cout << "Maybe better results can be achieved by changing the alignment. Starting new optimization." << endl;
				p_new.copyTo(p_new_save);
				residuals.copyTo(residuals_save);
				ssq_old = ssq;
			}
			else if(ssq_old < ssq)
			{
				cout << "The last alignment achieved smaller costs. Changing alignment." << endl;
				horizontalCamAlign = !horizontalCamAlign;
				p_new_save.copyTo(p_new);
				residuals_save.copyTo(residuals);
			}
		}
		if (aliChange)
		{
			p_new.copyTo(p);
			horizontalCamAlign = !horizontalCamAlign;
			setCoordsForOpti();
			//Store last valid parameters to history before alignment changed
			ssq_history.pop_back();
			ssq_history.push_back(r_before_aliC.dot(r_before_aliC));
			p_history.pop_back();
			p_history.push_back(x_before_aliC);
			r_history.pop_back();
			r_history.push_back(r_before_aliC);
		}		
		cnt++;
	}

	int err = 0;
	double ssq = ssq_history.back();
	if (cnt >= lmCntMax)
	{
		//Get configuration with smallest error
		auto idxMin = std::distance(ssq_history.begin(), min_element(ssq_history.begin(), ssq_history.end()));
		p_history[idxMin].copyTo(p_new);
		r_history[idxMin].copyTo(residuals);
		ssq = ssq_history[idxMin];
		cout << "Taking best result over the last 10 minimizations with changing alignments. Sum of squared residuals: " << ssq << endl;
		int aliSum = 0;
		for (int i = 0; i < (int)nrConditions; i++)
		{
			if (std::abs(p_new.at<double>(i * 4 + 2)) >= std::abs(p_new.at<double>(i * 4 + 3)))
				aliSum++;
			else
				aliSum--;
		}

		double alihelp = (double)aliSum / (double)nrConditions;
		if (!nearZero(1.0 - std::abs(alihelp)))
		{
			cout << "Camera alignment not consistent but best solution so far!" << endl;
			err = -2;
		}
		if (((alihelp > 0) && !horizontalCamAlign) ||
			((alihelp < 0) && horizontalCamAlign))
		{
			horizontalCamAlign = !horizontalCamAlign;
		}
	}

	double meanOvLapError = 0;
	for (int i = 0; i < (int)nrConditions; i++)
	{
		meanOvLapError += std::abs(residuals.at<double>(i * nr_residualsPCond + 5));
	}
	meanOvLapError /= (double)nrConditions;
	cout << "Approx. overlap error: " << meanOvLapError << endl;
	if (meanOvLapError > 0.05)
	{
		cout << "Unable to reach desired result! Try different ranges and/or parameters." << endl;
		err = -1;
	}
	if (ssq > 10.0)
	{
		cout << "Resulting paramters are not usable! Sum of squared residuals: " << ssq << endl;
		err = -2;
	}

	//Store parameters back to vectors
	for (int i = 0; i < (int)nrConditions; i++)
	{
		pitch_use[i] = p_new.at<double>(i * 4);
		roll_use[i] = p_new.at<double>(i * 4 + 1);
		tx_use[i] = p_new.at<double>(i * 4 + 2);
		ty_use[i] = p_new.at<double>(i * 4 + 3);
	}
	f = p_new.at<double>((int)(nrConditions * 4));
	K1.at<double>(0, 0) = f;
	K1.at<double>(1, 1) = f;
	K1.copyTo(K2);

	//Generate rotation matrizes and translation vectors
	for (size_t i = 0; i < nrConditions; i++)
	{
		Mat R = eulerAnglesToRotationMatrix(roll_use[i] * PI / 180.0, pitch_use[i] * PI / 180.0, yaw_use[i] * PI / 180.0);
		Mat t = (Mat_<double>(3, 1) << tx_use[i], ty_use[i], tz_use[i]);
		Ris.push_back(R.clone());
		tis.push_back(t.clone());
	}

	return err;
}

bool GenStereoPars::getCamPars(std::vector<cv::Mat>& Rv, std::vector<cv::Mat>& tv, cv::Mat& K_1, cv::Mat& K_2)
{
	if (Ris.empty())
	{
		return false;
	}

	Rv = Ris;
	tv = tis;
	K_1 = K1.clone();
	K_2 = K2.clone();

	return true;
}

bool GenStereoPars::getEulerAngles(std::vector<double>& roll, std::vector<double>& pitch, std::vector<double>& yaw)
{
	if (Ris.empty())
	{
		return false;
	}

	roll = roll_use;
	pitch = pitch_use;
	yaw = yaw_use;

	return true;
}

/*Set the tolerance for the parameters estimated by the Levengergh Marquardt algorithm. It is used to escape the
* optimization loop if the change of all paramters is smaller than these thresholds. Moreover, they are used to 
* calculate the step size for numerical differentiation (for calculating the partial derivatives of the Jacobi matrix).
*/
void GenStereoPars::setLMTolerance(double rollTol, double pitchTol, double txTol, double tyTol, double fTol)
{
	rollTol_ = rollTol;
	pitchTol_ = pitchTol;
	txTol_ = txTol;
	tyTol_ = tyTol;
	fTol_ = fTol;
}

/**  LMFSOLVE  Solve a Set of Nonlinear Equations in Least - Squares Sense.
*  Changed by JM->now no general solver
*
*  A solution is obtained by a shortened Fletcher version of the
*  Levenberg - Maquardt algoritm for minimization of a sum of squares
*   of equation residuals.
* 
* [Xf, Ssq, CNT] = LMFsolve(p, xf, residuals, funcTol, xTol, maxIter)
*  Input:
*  p       is a vector of initial guesses of solution,
*  p must hold n * 4 + 1  paramters (rows) where n is the number of camera configurations
*  The order of parameters must be [pitch, roll, tx, ty, ..., f]
* 
*  Options: 
* 
*     Name   Values{ default }         Description
*  'verbose'     integer     Display iteration information
* { 0 }  no display
*                              k   display initial and every k - th iteration;
*  'funcTol'      {1e-7}      norm(FUN(x), 1) stopping tolerance;
*  'xTol'        {1e-7}      norm(x - xold, 1) stopping tolerance;
*  'MaxIter'     {100}       Maximum number of iterations;
*  Not defined fields of the Options are filled by default values.
* 
*  Output Arguments :
*  xf        final solution approximation
*  residuals residuals
*  Cnt       >0          count of iterations
*            -MaxIter, did not converge in MaxIter iterations
*/
int GenStereoPars::LMFsolve(cv::Mat p,
	cv::Mat& xf,
	cv::Mat& residuals,
	cv::InputArray funcTol,
	cv::InputArray xTol,
	size_t maxIter,
	int verbose)
{
	cv::Mat funcTol_;
	if (funcTol.empty())
	{
		funcTol_ = Mat::ones((int)nrConditions * nr_residualsPCond, 1, p.type()) * 1e-7;
	}
	else
	{
		funcTol_ = funcTol.getMat();
		if (funcTol_.rows != ((int)nrConditions * nr_residualsPCond))
		{
			cout << "Number of tolerance values for the functions to optimize is not correct! Taking default values!" << endl;
			funcTol_ = Mat::ones((int)nrConditions * nr_residualsPCond, 1, p.type()) * 1e-7;
		}
	}

	cv::Mat xTol_;
	if (xTol.empty())
	{
		xTol_ = Mat::ones(p.size(), p.type()) * 1e-4;
	}
	else
	{
		xTol_ = xTol.getMat();
		if (xTol_.rows != p.rows)
		{
			cout << "Number of tolerance values for the parameters to optimize is not correct! Taking default values!" << endl;
			xTol_ = Mat::ones(p.size(), p.type()) * 1e-4;
		}
	}

	//Set up vector for optimization
	Mat x = p.clone();

	//Calculate residuals at starting point
	Mat r = LMfunc(x);

	double S = r.dot(r);

	Mat J = finjac(r, x, xTol_);

	//Number of function and Jacobi calculations
	size_t nfJ = 2;

	//System matrix
	Mat A = J.t() * J;
	Mat v = J.t() * r;

	double Rlo = 0.25;
	double Rhi = 0.75;
	double lamda = 1.0;//Starting value for learning rate
	double lc = 0.75;//Below lc, lamda is set to zero to get a normal Newton algorithm
	Mat D = getD(A.diag(0), x, lamda);
	int cnt = 0;
	printit(verbose, -1, nfJ, 0, x, x, lamda, lc);                //   Table header
	Mat d = xTol_.clone(); //vector for the first cycle
	Mat deltaValid = Mat::ones(x.size(), x.type());
	Mat xd_old = x.clone();
	Mat r_old = Mat::ones(r.size(), r.type()) * 100.0;
	size_t sameXdCnt[2] = { 0, 0};
	bool noAliChange = true;

	//MAIN ITERATION CYCLE
	while ((cnt < (int)maxIter) &&
		any_vec_cv(cv::abs(d.mul(deltaValid)) >= xTol_) &&
		any_vec_cv(cv::abs(r) >= funcTol_) && 
		!nearZero(cv::sum(cv::abs(r_old - r))[0]) &&
		(sameXdCnt[1] < 4))
	{
		Mat A1 = A + lamda * D;
		bool solA = solveLinEqu(A1, v, d);

		bool randVec = false;
		if (!solA || /*any_vec_cv(d != d) || */ !isfinite_vec_cv(d))//checks for NaN
		{
			d = getNormalDistributionVals(d.rows, 0, 0.02);
			randVec = true;
		}
		Mat xd = x - d;
		adaptParVec(xd, xd, deltaValid);//Adapt solution
		if (nearZero(cv::sum(deltaValid)[0]))
		{
			cout << "Unable to perform optimization using LM with the given ranges" << endl;
		}
		if (nearZero(cv::sum(cv::abs(xd_old - xd))[0]))
		{
			d = getNormalDistributionVals(d.rows, 0, 0.05);
			lamda = 1.0;
			xd = x - d;
			adaptParVec(xd, xd, deltaValid);
			randVec = true;
			sameXdCnt[0]++;
		}
		if (sameXdCnt[0] > 2)
			sameXdCnt[1]++;
		xd.copyTo(xd_old);
		Mat rd = LMfunc(xd);
		if (nearZero(100.0 * cv::sum(cv::abs(rd))[0]))//If the alignment changed, all residuals are zero. This breaks the while-loop. But the last residuals not zero should be returned.
		{
			noAliChange = false;
		}

		nfJ++;
		double Sd = rd.dot(rd);
		double R;
		if (randVec)
		{
			R = Rhi * 2.0;
			if (abs(lamda) < 1e-5)
			{
				lamda = 1.0;
			}
		}
		else
		{
			double dS = d.dot(2.0 * v - A * d);
			R = (S - Sd) / dS;
		}

		if (R > Rhi)//halve lambda if R too high
		{
			lamda = lamda / 2.0;
			D = getD(A.diag(0), x, lamda);
			if (lamda < lc)
				lamda = 0;
		}
		else if (R < Rlo)//find new nu if R too low
		{
			double nu = (Sd - S) / d.dot(v) + 2.0;
			if (nu < 2.0)
			{
				nu = 2.0;
			}
			else if (nu > 10.0)
			{
				nu = 10.0;
			}
			if (nearZero(lamda))
			{
				double detA = cv::determinant(A);
				Mat Ai;
				if (nearZero(detA))//Check if matrix A is singular or near singular
				{
					Ai = A.inv(DECOMP_SVD);
				}
				else
				{
					Ai = A.inv(DECOMP_LU);
				}
				double maxAid;
				Mat Aida = cv::abs(Ai.diag(0));
				cv::minMaxLoc(Aida, nullptr, &maxAid);
				if (!isfinite(maxAid))
					maxAid = 1.5;
				if (nearZero(100.0 * maxAid))
					maxAid = getRandDoubleVal(0, 1.5);
				if (maxAid > (1.0 / DBL_EPSILON))
					maxAid = 1.0 / DBL_EPSILON;
				lc = 1.0 / maxAid;
				lamda = lc;
				nu = nu / 2.0;
			}
			lamda = nu * lamda;
			D = getD(A.diag(0), x, lamda);
		}

		cnt++;
		//Print iteration?
		if (verbose && (((cnt % abs(verbose)) == 0) || (cnt == 1)))
		{
			printit(verbose, cnt, nfJ, S, x, d, lamda, lc);
		}

		if (Sd < S)
		{
			if (!noAliChange)
			{
				x.copyTo(x_before_aliC);
				r.copyTo(r_before_aliC);
			}
			S = Sd;
			x = xd.clone();
			r.copyTo(r_old);
			r = rd.clone();
			J = finjac(r, x, xTol_);
			//       ~~~~~~~~~~~~~~~~~~~~~~~~~
			nfJ++;
			A = J.t() * J;
			v = J.t() * r;
			D = getD(A.diag(0), x, lamda);
		}
	}

	//final solution
	x.copyTo(xf);
	if (cnt == (int)maxIter)
	{
		cnt *= -1;
	}
	Mat rd = LMfunc(xf, true);
	rd.copyTo(residuals);
	nfJ++;

	double Sd = rd.dot(rd);
	if (verbose)
	{
		cout << endl;
		printit(verbose < 0 ? -1:1, (cnt < 0) ? -cnt:cnt, nfJ, Sd, xf, d, lamda, lc);
	}

	return cnt;
}

/*         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Printing of intermediate results
*   ipr <  0  do not print lambda columns
*		= 0  do not print at all
*       >  0  print every(ipr)th iteration
*   cnt = -1  print out the header
*		   0  print out second row of results
*         >0  print out first row of results
*/
void GenStereoPars::printit(int ipr, int cnt, size_t nfJ, double ss, cv::Mat x, cv::Mat dx, double lamda, double lamda_c)
{
	if (ipr)
	{
		if (cnt < 0)//Table Header
		{
			cout << endl;
			for (size_t i = 0; i < 75; i++)
			{
				cout << "*";
			}
			cout << endl;
			cout << "  itr  nfJ   SUM(r^2)        x           dx";
			if (ipr > 0)
			{
				cout << "           lamda       lc";
			}
			cout << endl;
			for (size_t i = 0; i < 75; i++)
			{
				cout << "*";
			}
			cout << endl;
		}
		else if(((cnt % abs(ipr)) == 0) || (cnt==1))
		{
			if (ipr > 0)
			{
				printf("%4d %4d %12.4e %12.4e %12.4e %12.4e %12.4e \n", cnt, (int)nfJ, ss, x.at<double>(0), dx.at<double>(0), lamda, lamda_c);
			}
			else
			{
				printf("%4d %4d %12.4e %12.4e %12.4e \n", cnt, (int)nfJ, ss, x.at<double>(0), dx.at<double>(0));
			}
			int lx = x.rows;
			for (int i = 1; i < lx; i++)
			{
				for (size_t j = 0; j < 23; j++)
				{
					cout << " ";
				}
				printf("%12.4e %12.4e \n", x.at<double>(i), dx.at<double>(i));
			}
		}
	}
}

/* Adapts the new parameter vector x. For fixed values in the parameter vector, the new x is set to the fixed value. In
* addition, if a specific parameter value is found to be outside its allowed range, it is set to its nearest range parameter. 
*    parVec holds the function paramters after the last LM iteration.Form:
* [pitch; roll; tx; ty; f]
* 
*    deltaValid indicates which x value is valid
*/
void GenStereoPars::adaptParVec(cv::Mat& parVec, cv::Mat& parVecOut, cv::Mat& deltaValid)
{
	parVec.copyTo(parVecOut);
	deltaValid = Mat::ones(parVec.size(), parVec.type());

	for (int i = 0; i < parVec.rows; i++)
	{
		bool validR = true;
		int spar = i % 4;
		int cond_n = (i - spar) / 4;
		pair<double, double> parRange;
		if (cond_n >= (int)nrConditions)
		{
			parRange = make_pair(fRange[0], fRange[1]);
		}
		else
		{
			switch (spar)
			{
			case 0:
				if (pitch_[cond_n].size() == 1)
				{
					validR = false;
					if (!nearZero(pitch_[cond_n][0] - parVec.at<double>(i)))
					{
						parVecOut.at<double>(i) = pitch_[cond_n][0];
						deltaValid.at<double>(i) = 0;
					}
				}
				else
				{
					parRange = make_pair(pitch_[cond_n][0], pitch_[cond_n][1]);
				}
				break;
			case 1:
				if (roll_[cond_n].size() == 1)
				{
					validR = false;
					if (!nearZero(roll_[cond_n][0] - parVec.at<double>(i)))
					{
						parVecOut.at<double>(i) = roll_[cond_n][0];
						deltaValid.at<double>(i) = 0;
					}
				}
				else
				{
					parRange = make_pair(roll_[cond_n][0], roll_[cond_n][1]);
				}
				break;
			case 2:
				if (tx_[cond_n].size() == 1)
				{
					validR = false;
					if (!nearZero(tx_[cond_n][0] - parVec.at<double>(i)))
					{
						parVecOut.at<double>(i) = tx_[cond_n][0];
						deltaValid.at<double>(i) = 0;
					}
				}
				else
				{
					parRange = make_pair(tx_[cond_n][0], tx_[cond_n][1]);
				}
				break;
			case 3:
				if (ty_[cond_n].size() == 1)
				{
					validR = false;
					if (!nearZero(ty_[cond_n][0] - parVec.at<double>(i)))
					{
						parVecOut.at<double>(i) = ty_[cond_n][0];
						deltaValid.at<double>(i) = 0;
					}
				}
				else
				{
					parRange = make_pair(ty_[cond_n][0], ty_[cond_n][1]);
				}
				break;
			default:
				validR = false;
				break;
			}
		}

		if (validR)
		{
			if (parVec.at<double>(i) < parRange.first)
			{
				parVecOut.at<double>(i) = parRange.first;
				deltaValid.at<double>(i) = 0;
			}
			else if (parVec.at<double>(i) > parRange.second)
			{
				parVecOut.at<double>(i) = parRange.second;
				deltaValid.at<double>(i) = 0;
			}
		}
	}

}

cv::Mat GenStereoPars::getD(cv::Mat DvecLTL, cv::Mat parVec, double lamda)
{
	int p_n = DvecLTL.rows;
	CV_Assert((p_n >= DvecLTL.cols) && (p_n == parVec.rows));

	Mat D = Mat::eye(p_n, p_n, CV_64FC1);
	Mat Ddiag = D.diag(0);
	Mat mIs0 = Mat::zeros(p_n, 1, CV_64FC1);

	for (int i = 0; i < p_n; i++)
	{
		Ddiag.at<double>(i) = getDampingF(parVec.at<double>(i), i);
		if (nearZero(100.0 * Ddiag.at<double>(i)))
		{
			mIs0.at<double>(i) = 1.0;
		}
	}

	if (lamda > 0)
	{
		Mat negA = -1.0 * DvecLTL + DvecLTL / cv::abs(DvecLTL) * (2.0 * DBL_EPSILON);
		Mat Dtmp = mIs0.mul(negA, 1.0 / lamda);
		cv::multiply(Ddiag, DvecLTL, Ddiag);
		Ddiag += Dtmp;
	}
	else
	{
		//Ddiag = Ddiag.mul(DvecLTL);
		cv::multiply(Ddiag, DvecLTL, Ddiag);
	}

	return D;//CHECK IF THE DIAGONAL ENTRIES ARE SET CORRECTLY!!!!!!!!!!!!!!!!!!!!!!
}

double GenStereoPars::getDampingF(double par, int rangeIdx0)
{
	double damp = -1.0;
	int spar = rangeIdx0 % 4;
	int cond_n = (rangeIdx0 - spar) / 4;
	pair<double, double> parRange;
	if (cond_n >= (int)nrConditions)
	{
		parRange = make_pair(fRange[0], fRange[1]);
	}
	else
	{
		switch (spar)
		{
		case 0:
			if(pitch_[cond_n].size() == 1)
				damp = 0;
			else
				parRange = make_pair(pitch_[cond_n][0], pitch_[cond_n][1]);
			break;
		case 1:
			if (roll_[cond_n].size() == 1)
				damp = 0;
			else
				parRange = make_pair(roll_[cond_n][0], roll_[cond_n][1]);
			break;
		case 2:
			if (tx_[cond_n].size() == 1)
				damp = 0;
			else
				parRange = make_pair(tx_[cond_n][0], tx_[cond_n][1]);
			break;
		case 3:
			if (ty_[cond_n].size() == 1)
				damp = 0;
			else
				parRange = make_pair(ty_[cond_n][0], ty_[cond_n][1]);
			break;
		default:
			damp = 1.0;
			break;
		}
	}
	if (damp < 0)
	{
		double pd = (parRange.second - parRange.first) / 2.0;
		double pm = parRange.first + pd;
		damp = 1.0 - abs(par - pm) / pd;
		if (damp < 0)
			damp = 0;
	}

	return damp;
}

//Calculate Jacobi matrix
cv::Mat GenStereoPars::finjac(cv::Mat& residuals, cv::Mat& x, cv::Mat& xTol)
{
	CV_Assert((residuals.rows >= residuals.cols) && (x.rows >= x.cols) && (xTol.rows >= xTol.cols) && (x.rows == xTol.rows));

	int lx = x.rows;
	Mat J = Mat::zeros(residuals.rows, lx, CV_64FC1);
	for (int k = 0; k < lx; k++)
	{
		double dx = 0.25 * xTol.at<double>(k);
		Mat xd = x.clone();
		xd.at<double>(k) += dx;
		Mat rd = LMfunc(xd);
		J.col(k) = (rd - residuals) / dx;
	}
	return J;
}

//Calculate residuals
cv::Mat GenStereoPars::LMfunc(cv::Mat p, bool noAlignCheck)
{
	//Check if the alignment has changend
	int aliSum = 0;
	for (int i = 0; i < (int)nrConditions; i++)
	{
		if (std::abs(p.at<double>(i * 4 + 2)) >= std::abs(p.at<double>(i * 4 + 3)))
			aliSum++;
		else
			aliSum--;
	}

	double alihelp = (double)aliSum / (double)nrConditions;
	//Only abort the optimization (to start a new one with changed alignment) if more than the half of alignments have changed
	if (!noAlignCheck && (((alihelp > 0) && !horizontalCamAlign) ||
		((alihelp < 0) && horizontalCamAlign)) &&
		!nearZero(alihelp))
	{
		return Mat::zeros((int)nrConditions * nr_residualsPCond, 1, CV_64FC1);
	}

	cv::Mat r = Mat::zeros((int)nrConditions * nr_residualsPCond, 1, CV_64FC1);

	Mat K1p = K1.clone();
	K1p.at<double>(0, 0) = K1p.at<double>(1, 1) = p.at<double>((int)nrConditions * 4);
	Mat K2p = K1p.clone();

	for (int i = 0; i < (int)nrConditions; i++)
	{
		//Generate residuals for achieving no large positive tx or ty values
		r.at<double>(i * nr_residualsPCond + 7) = alignFunctionNoLargePos(p.at<double>(i * 4 + 2), p.at<double>(i * 4 + 3));

		//Evaluate function to achieve the same orientation for all camera configurations
		if (horizontalCamAlign)
		{
			r.at<double>(i * nr_residualsPCond + 8) = 0.1 * alignFunctionHori(p.at<double>(i * 4 + 2), p.at<double>(i * 4 + 3));
		}
		else
		{
			r.at<double>(i * nr_residualsPCond + 8) = 0.1 * alignFunctionVert(p.at<double>(i * 4 + 2), p.at<double>(i * 4 + 3));
		}

		//Calculate extrinsics
		Mat R, t;
		CalcExtrFromPar(p.at<double>(i * 4 + 1), p.at<double>(i * 4), p.at<double>(i * 4 + 2), p.at<double>(i * 4 + 3), i, t, R);
		double bl = norm(t);

		//Get maximal useable depth
		double zmax = sqrt(K1p.at<double>(0, 0) * bl * bl / 0.15);//0.15 corresponds to the approx. typ. correspondence accuracy in pixels
		double zm = zmax / 2.0;

		//Calculate min distance for 3D points visible in both images
		Mat b1 = getLineCam1(K1p, x_lb_max1[i]);
		Mat a2, b2;
		getLineCam2(R, t, K2p, x_rt_min1[i], a2, b2);
		double z = getLineIntersect(b1, a2, b2);
		double ovLapZ = (zmax - z) / zmax;
		double r_tmp = 0.66 * (ovLapZ - 1.0);
		pow2mult(r_tmp, 4);
		r.at<double>(i * nr_residualsPCond) = r_tmp;//Penalty for z-values larger zmax (ovLap gets negative) -> The field of view of both cameras is shared above zmax.

		//Calculate overlap at medium z-distance in the direction of the main camera displacement
		//First overlap (left or bottom side)
		b1 = getLineCam1(K1p, x_lb_min1[i]);
		getLineCam2(R, t, K2p, x_rt_min1[i], a2, b2);
		double dist1 = getDistance2LinesPlane(cv::Mat::zeros(3,1,CV_64FC1), b1, a2, b2, zm);
		double ovLapCam1Zm = 1.0;
		if (!nearZero(dist1))
		{
			Mat b11 = getLineCam1(K1p, x_lb_max1[i]);
			double distCam1 = getDistance2LinesPlane(cv::Mat::zeros(3, 1, CV_64FC1), b1, cv::Mat::zeros(3, 1, CV_64FC1), b11, zm);
			ovLapCam1Zm = 1.0 - dist1 / distCam1;
		}
		r_tmp = 0.85 * (ovLapCam1Zm - 1.0);
		pow2mult(r_tmp, 4);
		r.at<double>(i * nr_residualsPCond + 1) = r_tmp;//Penalty if the 2 cameras do not overlap at the medium distance or the order of rays is wrong along the main displacement (direction 1)

		//Second overlap (right or top side)
		b1 = getLineCam1(K1p, x_lb_max1[i]);
		getLineCam2(R, t, K2p, x_rt_max1[i], a2, b2);
		double dist2 = getDistance2LinesPlane(cv::Mat::zeros(3, 1, CV_64FC1), b1, a2, b2, zm);
		double ovLapCam2Zm = 1.0;
		if (!nearZero(dist2))
		{
			Mat a21, b21;
			getLineCam2(R, t, K2p, x_rt_min1[i], a21, b21);
			double distCam2 = getDistance2LinesPlane(a2, b2, a21, b21, zm);
			ovLapCam2Zm = 1.0 - dist2 / distCam2;
		}
		r_tmp = 0.85 * (ovLapCam2Zm - 1.0);
		pow2mult(r_tmp, 4);
		r.at<double>(i * nr_residualsPCond + 2) = r_tmp;//Penalty if the 2 cameras do not overlap at the medium distance or the order of rays is wrong along the main displacement (direction 2)
		//Get mean overlap
		double ovLapZm = (ovLapCam1Zm + ovLapCam2Zm) / 2.0;

		//Calculate overlap at medium z - distance in the direction perpendicular to the main camera displacement
		//First overlap (left or bottom side)
		b1 = getLineCam1(K1p, x_lb_min2[i]);
		Mat b11 = getLineCam1(K1p, x_lb_max2[i]);
		getLineCam2(R, t, K2p, x_rt_min2[i], a2, b2);
		Mat a21, b21;
		getLineCam2(R, t, K2p, x_rt_max2[i], a21, b21);
		dist1 = getDistance2LinesPlane(cv::Mat::zeros(3, 1, CV_64FC1), b1, a2, b2, zm);
		double distCam1 = getDistance2LinesPlane(cv::Mat::zeros(3, 1, CV_64FC1), b1, cv::Mat::zeros(3, 1, CV_64FC1), b11, zm);
		double distCam2 = getDistance2LinesPlane(a2, b2, a21, b21, zm);
		//Calculate mean first overlap over camera 1 and 2
		double ovLapCam1ZmP = 1.0 - (dist1 / distCam1 + dist1 / distCam2) / 2.0;
		r_tmp = 0.85 * (ovLapCam1ZmP - 1.0);
		pow2mult(r_tmp, 4);
		r.at<double>(i * nr_residualsPCond + 3) = r_tmp;//Penalty if the 2 cameras do not overlap at the medium distance or the order of rays is wrong perpendicular to the main displacement (direction 1)

		//Second overlap (right or top side)
		b1 = getLineCam1(K1p, x_lb_max2[i]);
		getLineCam2(R, t, K2p, x_rt_max2[i], a2, b2);
		dist2 = getDistance2LinesPlane(cv::Mat::zeros(3, 1, CV_64FC1), b1, a2, b2, zm);
		//Calculate mean second overlap over camera 1 and 2
		double ovLapCam2ZmP = 1.0 - (dist2 / distCam1 + dist2 / distCam2) / 2.0;
		r_tmp = 0.85 * (ovLapCam2ZmP - 1.0);
		pow2mult(r_tmp, 4);
		r.at<double>(i * nr_residualsPCond + 4) = r_tmp;//Penalty if the 2 cameras do not overlap at the medium distance or the order of rays is wrong perpendicular to the main displacement (direction 2)
		//Calculate overall mean overlap in the direction perpendicular to the main camera displacement
		double ovLapZmP = (ovLapCam1ZmP + ovLapCam2ZmP) / 2.0;

		//Optimize overlapping view in 3D in terms of z
		double ovLapZ3D = (z - 0.75 * bl) / zmax; //The optimal overlap in z - direction between the 2 cameras begins at 0.75 times the baseline length
		r_tmp = 0.9 * (ovLapZ3D - 1.0);
		pow2mult(r_tmp, 5);
		r_tmp = 1.0 / (r_tmp + 1.0);
		ovLapZ3D *= r_tmp;

		//Calculate main residuals
		r.at<double>(i * nr_residualsPCond + 5) = approxImgOverlap_ - ovLapZ * ovLapZm * ovLapZmP;//Get the desired image overlap
		r.at<double>(i * nr_residualsPCond + 6) = 0.35 * ovLapZ3D;//Optimize overlapping area
	}

	return r;
}

/*Calculate the distance between 2 points on a plane with a specific
* z - coordinate(all over the plane) whereas the points are intersections of
* lines with the plane.
* The lines are represented by the join of 2 points 'a' and direction 'b'.
*/
double GenStereoPars::getDistance2LinesPlane(cv::Mat a1, cv::Mat b1, cv::Mat a2, cv::Mat b2, double z)
{
	double s1 = (z - a1.at<double>(2)) / b1.at<double>(2);
	Mat X1 = a1 + s1 * b1;
	double s2 = (z - a2.at<double>(2)) / b2.at<double>(2);
	Mat X2 = a2 + s2 * b2;

	return std::abs(cv::norm(X1 - X2));
}

void GenStereoPars::CalcExtrFromPar(double roll, double pitch, double tx, double ty, int cnf, cv::Mat& t, cv::Mat& R)
{
	roll *= PI / 180.0;
	pitch *= PI / 180.0;
	R = eulerAnglesToRotationMatrix(roll, pitch, 0);
	t = (Mat_<double>(3, 1) << tx, ty, tz_use[cnf]);
}

double GenStereoPars::alignFunctionNoLargePos(double tx, double ty)
{
	double baseval = 1.0 + (tx + ty) / std::sqrt(2.0 * (tx * tx + ty * ty));
	pow2mult(baseval, 4);
	return baseval;
}

double GenStereoPars::alignFunctionHori(double tx, double ty)
{
	double baseval = 1.0 + (tx - ty) / std::sqrt(2.0 * (tx * tx + ty * ty));
	pow2mult(baseval, 4);
	return baseval;
}

double GenStereoPars::alignFunctionVert(double tx, double ty)
{
	double baseval = 1.0 + (ty - tx) / std::sqrt(2.0 * (tx * tx + ty * ty));
	pow2mult(baseval, 4);
	return baseval;
}

inline void GenStereoPars::pow2mult(double& val, size_t num)
{
	for (size_t i = 0; i < num; i++)
	{
		val *= val;
	}
	if (!isfinite(val))
	{
		val = DBL_MAX;
	}
}

void GenStereoPars::setCoordsForOpti()
{
	if (x_lb_max1.empty())
		x_lb_max1 = std::vector<cv::Mat>(nrConditions);
	if (x_lb_min1.empty())
		x_lb_min1 = std::vector<cv::Mat>(nrConditions);
	if (x_rt_max1.empty())
		x_rt_max1 = std::vector<cv::Mat>(nrConditions);
	if (x_rt_min1.empty())
		x_rt_min1 = std::vector<cv::Mat>(nrConditions);

	if (x_lb_max2.empty())
		x_lb_max2 = std::vector<cv::Mat>(nrConditions);
	if (x_lb_min2.empty())
		x_lb_min2 = std::vector<cv::Mat>(nrConditions);
	if (x_rt_max2.empty())
		x_rt_max2 = std::vector<cv::Mat>(nrConditions);
	if (x_rt_min2.empty())
		x_rt_min2 = std::vector<cv::Mat>(nrConditions);

	if (horizontalCamAlign)
	{
		for (size_t i = 0; i < nrConditions; i++)
		{
			x_lb_max1[i] = (Mat_<double>(3, 1) << ((double)imgSize_.width + virtWidth[i]) / 2.0, (double)imgSize_.height / 2.0, 1.0);
			x_lb_min1[i] = (Mat_<double>(3, 1) << ((double)imgSize_.width - virtWidth[i]) / 2.0, (double)imgSize_.height / 2.0, 1.0);
			x_rt_max1[i] = (Mat_<double>(3, 1) << (double)imgSize_.width, (double)imgSize_.height / 2.0, 1.0);
			x_rt_min1[i] = (Mat_<double>(3, 1) << 0, (double)imgSize_.height / 2.0, 1.0);
		}

		x_lb_max2 = std::vector<cv::Mat>(nrConditions,(Mat_<double>(3, 1) << (double)imgSize_.width / 2.0, (double)imgSize_.height, 1.0));
		x_lb_min2 = std::vector<cv::Mat>(nrConditions,(Mat_<double>(3, 1) << (double)imgSize_.width / 2.0, 0, 1.0));
		x_rt_max2 = std::vector<cv::Mat>(nrConditions,(Mat_<double>(3, 1) << (double)imgSize_.width / 2.0, (double)imgSize_.height, 1.0));
		x_rt_min2 = std::vector<cv::Mat>(nrConditions,(Mat_<double>(3, 1) << (double)imgSize_.width / 2.0, 0, 1.0));
	}
	else
	{
		x_lb_max1 = std::vector<cv::Mat>(nrConditions, (Mat_<double>(3, 1) << (double)imgSize_.width / 2.0, (double)imgSize_.height, 1.0));
		x_lb_min1 = std::vector<cv::Mat>(nrConditions, (Mat_<double>(3, 1) << (double)imgSize_.width / 2.0, 0, 1.0));
		x_rt_max1 = std::vector<cv::Mat>(nrConditions, (Mat_<double>(3, 1) << (double)imgSize_.width / 2.0, (double)imgSize_.height, 1.0));
		x_rt_min1 = std::vector<cv::Mat>(nrConditions, (Mat_<double>(3, 1) << (double)imgSize_.width / 2.0, 0, 1.0));

		for (size_t i = 0; i < nrConditions; i++)
		{
			x_lb_max2[i] = (Mat_<double>(3, 1) << ((double)imgSize_.width + virtWidth[i]) / 2.0, (double)imgSize_.height / 2.0, 1.0);
			x_lb_min2[i] = (Mat_<double>(3, 1) << ((double)imgSize_.width - virtWidth[i]) / 2.0, (double)imgSize_.height / 2.0, 1.0);
			x_rt_max2[i] = (Mat_<double>(3, 1) << (double)imgSize_.width, (double)imgSize_.height / 2.0, 1.0);
			x_rt_min2[i] = (Mat_<double>(3, 1) << 0, (double)imgSize_.height / 2.0, 1.0);
		}
	}
}

bool GenStereoPars::helpNewRandEquRangeVals(int& idx, const int maxit, int align)
{
	int cnt = 0;
	int idx_tmp = idx;
	bool additionalAdaption = true;
	if (txRangeEqual && (tx_[0].size() > 1))
	{
		//Check if the tx value must be changed as changing the ty values has no effect (tx value out of range compared to ty value range)
		while ((((std::abs(tx_use[idx_tmp]) >= std::abs(ty_use[idx_tmp])) && ((tx_use[idx_tmp] > 0) || (align < 0)) &&
			((std::abs(tx_use[idx_tmp]) >= std::abs(ty_[idx_tmp][0])) ||
			(std::abs(tx_use[idx_tmp]) >= std::abs(ty_[idx_tmp][1])))) ||
				((std::abs(tx_use[idx_tmp]) < std::abs(ty_use[idx_tmp])) && ((ty_use[idx_tmp] > 0) || (align > 0)) &&
			((std::abs(tx_use[idx_tmp]) < std::abs(ty_[idx_tmp][0])) ||
					(std::abs(tx_use[idx_tmp]) < std::abs(ty_[idx_tmp][1]))))) &&
				(cnt < maxit))
		{
			tx_use.clear();
			initRandPars(tx_, txRangeEqual, tx_use);
			if ((ty_[idx_tmp].size() > 1) && (((std::abs(tx_use[idx_tmp]) >= std::abs(ty_use[idx_tmp])) && ((tx_use[idx_tmp] > 0) || (align < 0))) ||
				((std::abs(tx_use[idx_tmp]) < std::abs(ty_use[idx_tmp])) && ((ty_use[idx_tmp] > 0) || (align > 0)))))
			{
				ty_use[idx_tmp] = getRandDoubleVal(ty_[idx_tmp][0], ty_[idx_tmp][1]);
			}
			idx = -1;
			cnt++;
		}
		if (cnt >= maxit)
		{
			return false;//Not able to generate a valid random camera configuration. Ranges of tx and ty should be adapted.
		}
		additionalAdaption = false;
	}
	else if(tx_[idx].size() > 1)
	{
		tx_use[idx] = getRandDoubleVal(tx_[idx][0], tx_[idx][1]);
		if (!(((std::abs(tx_use[idx_tmp]) >= std::abs(ty_use[idx_tmp])) && ((tx_use[idx_tmp] > 0) || (align < 0))) ||
			((std::abs(tx_use[idx_tmp]) < std::abs(ty_use[idx_tmp])) && ((ty_use[idx_tmp] > 0) || (align > 0)))))
		{
			additionalAdaption = false;
		}
	}

	if (additionalAdaption)
	{
		if (tyRangeEqual && (ty_[idx].size() > 1))
		{
			cnt = 0;
			//Check if the ty value must be changed as changing the tx values has no effect (ty value out of range compared to tx value range)
			while ((((std::abs(tx_use[idx_tmp]) >= std::abs(ty_use[idx_tmp])) && ((tx_use[idx_tmp] > 0) || (align < 0)) &&
				((std::abs(tx_[idx_tmp][0]) >= std::abs(ty_use[idx_tmp])) ||
				(std::abs(tx_[idx_tmp][1]) >= std::abs(ty_use[idx_tmp])))) ||
					((std::abs(tx_use[idx_tmp]) < std::abs(ty_use[idx_tmp])) && ((ty_use[idx_tmp] > 0) || (align > 0)) &&
				((std::abs(tx_[idx_tmp][0]) < std::abs(ty_use[idx_tmp])) ||
						(std::abs(tx_[idx_tmp][1]) < std::abs(ty_use[idx_tmp]))))) &&
					(cnt < maxit))
			{
				ty_use.clear();
				initRandPars(ty_, tyRangeEqual, ty_use);
				if ((tx_[idx_tmp].size() > 1) && (((std::abs(tx_use[idx_tmp]) >= std::abs(ty_use[idx_tmp])) && ((tx_use[idx_tmp] > 0) || (align < 0))) ||
					((std::abs(tx_use[idx_tmp]) < std::abs(ty_use[idx_tmp])) && ((ty_use[idx_tmp] > 0) || (align > 0)))))
				{
					tx_use[idx_tmp] = getRandDoubleVal(tx_[idx_tmp][0], tx_[idx_tmp][1]);
				}
				idx = -1;
				cnt++;
			}
			if (cnt >= maxit)
			{
				return false;//Not able to generate a valid random camera configuration. Ranges of tx and ty should be adapted.
			}
		}
		else if (ty_[idx].size() > 1)
		{
			ty_use[idx_tmp] = getRandDoubleVal(ty_[idx_tmp][0], ty_[idx_tmp][1]);
		}
	}

	return true;
}

//Calculates the intersection area between a rectangle and the same
//rectangle which is rotated by angle about its center and returns the ratio
// (between 0 and 1) of the intersecting area compared to the full rectangle
//area in perc.
//virtWidth1 specifies the virtual image width(proportional to the loss in
//area) that remains due to this rotation about the z - axis of camera 2.
void GenStereoPars::getRotRectDiffArea(double yaw_angle, double& perc, double& virtWidth1)
{
	//transform into radians and use only a negative angle as the result for
	//positive angles is the same
	yaw_angle = PI * -abs(yaw_angle) / 180.0;

	//Rotate rectangle
	Mat trans = (Mat_<double>(2,1) << (double)imgSize_.width / -2.0, (double)imgSize_.height / -2.0);
	Mat R = (Mat_<double>(3,3) << cos(yaw_angle), -sin(yaw_angle), 0, sin(yaw_angle), cos(yaw_angle), 0, 0, 0, 1.0);

	//Translate center to origin
	Mat bl1 = (Mat_<double>(3, 1) << 0, 0, 1.0);
	bl1.rowRange(0,2) += trans;
	Mat br1 = (Mat_<double>(3, 1) << (double)imgSize_.width, 0, 1.0);
	br1.rowRange(0, 2) += trans;
	Mat tr1 = (Mat_<double>(3, 1) << (double)imgSize_.width, (double)imgSize_.height, 1.0);
	tr1.rowRange(0, 2) += trans;
	Mat tl1 = (Mat_<double>(3, 1) << 0, (double)imgSize_.height, 1.0);
	tl1.rowRange(0, 2) += trans;

	Mat bl2 = R * bl1;
	Mat br2 = R * br1;
	Mat tr2 = R * tr1;
	Mat tl2 = R * tl1;

	//Calculate border angle
	Mat x1 = bl1;
	Mat x2 = tl1;
	double phi = abs(180.0 * acos((x1.at<double>(0)*x2.at<double>(0) + x1.at<double>(1)*x2.at<double>(1)) / 
		(x1.at<double>(0) * x1.at<double>(0) + x1.at<double>(1) * x1.at<double>(1))) / PI);

	//Get lines
	Mat lb1 = bl1.cross(br1);
	Mat lt1 = tl1.cross(tr1);
	Mat ll1 = bl1.cross(tl1);
	Mat lr1 = br1.cross(tr1);

	Mat lb2 = bl2.cross(br2);
	Mat lt2 = tl2.cross(tr2);
	Mat ll2 = bl2.cross(tl2);
	Mat lr2 = br2.cross(tr2);

	if (nearZero(round(yaw_angle * 100.0) / 100.0))
	{
		perc = 1.0;
		virtWidth1 = (double)imgSize_.width;
	}
	else
	{
		/*Mat cax;
		Mat cay;*/
		vector<Point2f> contour;
		if (nearZero(round((phi + yaw_angle) * 100.0) / 100.0))
		{
			Mat c1 = tl1;
			Mat c2 = lt1.cross(lt2);
			c2 /= c2.at<double>(2);
			Mat c3 = br1;
			Mat c4 = lb1.cross(lb2);
			c4 /= c4.at<double>(2);
			contour.emplace_back(Point2f((float)c1.at<double>(0), (float)c1.at<double>(1)));
			contour.emplace_back(Point2f((float)c2.at<double>(0), (float)c2.at<double>(1)));
			contour.emplace_back(Point2f((float)c3.at<double>(0), (float)c3.at<double>(1)));
			contour.emplace_back(Point2f((float)c4.at<double>(0), (float)c4.at<double>(1)));
			//cax = (Mat_<double>(1, 4) << c1.at<double>(0), c2.at<double>(0), c3.at<double>(0), c4.at<double>(0));
			//cay = (Mat_<double>(1, 4) << c1.at<double>(1), c2.at<double>(1), c3.at<double>(1), c4.at<double>(1));
			//Calc virtual remaining image width
			Mat w1 = c4 - bl1;
			double loss_area = norm(w1.rowRange(0,2)) * (double)imgSize_.height / 2.0;
			double reduceW = 2.0 * loss_area / (double)imgSize_.height;
			virtWidth1 = (double)imgSize_.width - reduceW;
		}
		else if ((phi + yaw_angle) > 0)
		{
			Mat c1 = ll1.cross(lb2);
			Mat c2 = ll1.cross(ll2);
			Mat c3 = lt1.cross(ll2);
			Mat c4 = lt1.cross(lt2);
			Mat c5 = lr1.cross(lt2);
			Mat c6 = lr1.cross(lr2);
			Mat c7 = lb1.cross(lr2);
			Mat c8 = lb1.cross(lb2);
			c1 /= c1.at<double>(2);
			c2 /= c2.at<double>(2);
			c3 /= c3.at<double>(2);
			c4 /= c4.at<double>(2);
			c5 /= c5.at<double>(2);
			c6 /= c6.at<double>(2);
			c7 /= c7.at<double>(2);
			c8 /= c8.at<double>(2);
			
			//cax = (Mat_<double>(1, 8) << c1.at<double>(0), c2.at<double>(0), c3.at<double>(0), c4.at<double>(0), c5.at<double>(0), c6.at<double>(0), c7.at<double>(0), c8.at<double>(0));
			//cay = (Mat_<double>(1, 8) << c1.at<double>(1), c2.at<double>(1), c3.at<double>(1), c4.at<double>(1), c5.at<double>(1), c6.at<double>(1), c7.at<double>(1), c8.at<double>(1));
			contour.emplace_back(Point2f((float)c1.at<double>(0), (float)c1.at<double>(1)));
			contour.emplace_back(Point2f((float)c2.at<double>(0), (float)c2.at<double>(1)));
			contour.emplace_back(Point2f((float)c3.at<double>(0), (float)c3.at<double>(1)));
			contour.emplace_back(Point2f((float)c4.at<double>(0), (float)c4.at<double>(1)));
			contour.emplace_back(Point2f((float)c5.at<double>(0), (float)c5.at<double>(1)));
			contour.emplace_back(Point2f((float)c6.at<double>(0), (float)c6.at<double>(1)));
			contour.emplace_back(Point2f((float)c7.at<double>(0), (float)c7.at<double>(1)));
			contour.emplace_back(Point2f((float)c8.at<double>(0), (float)c8.at<double>(1)));
			//Calc virtual remaining image width
			double h1 = norm(c1.rowRange(0, 2) - bl1.rowRange(0, 2));
			double h2 = norm(tl1.rowRange(0, 2) - c2.rowRange(0, 2));
			double w1 = norm(c8.rowRange(0, 2) - bl1.rowRange(0, 2));
			double w2 = norm(c3.rowRange(0, 2) - tl1.rowRange(0, 2));
			double loss_area1 = h1 * w1 / 2.0;
			double loss_area2 = h2 * w2 / 2.0;
			double reduceW = 2.0 * (loss_area1 + loss_area2) / (double)imgSize_.height;
			virtWidth1 = (double)imgSize_.width - reduceW;
		}
		else
		{
			Mat c1 = lt1.cross(lb2);
			Mat c2 = lt1.cross(lt2);
			Mat c3 = lb1.cross(lt2);
			Mat c4 = lb1.cross(lb2);
			c1 /= c1.at<double>(2);
			c2 /= c2.at<double>(2);
			c3 /= c3.at<double>(2);
			c4 /= c4.at<double>(2);
			//cax = (Mat_<double>(1, 4) << c1.at<double>(0), c2.at<double>(0), c3.at<double>(0), c4.at<double>(0));
			//cay = (Mat_<double>(1, 4) << c1.at<double>(1), c2.at<double>(1), c3.at<double>(1), c4.at<double>(1));
			contour.emplace_back(Point2f((float)c1.at<double>(0), (float)c1.at<double>(1)));
			contour.emplace_back(Point2f((float)c2.at<double>(0), (float)c2.at<double>(1)));
			contour.emplace_back(Point2f((float)c3.at<double>(0), (float)c3.at<double>(1)));
			contour.emplace_back(Point2f((float)c4.at<double>(0), (float)c4.at<double>(1)));
			//Calc virtual remaining image width
			vector<Point2f> contour1;
			contour1.emplace_back(Point2f((float)bl1.at<double>(0), (float)bl1.at<double>(1)));
			contour1.emplace_back(Point2f((float)tl1.at<double>(0), (float)tl1.at<double>(1)));
			contour1.emplace_back(Point2f((float)c1.at<double>(0), (float)c1.at<double>(1)));
			contour1.emplace_back(Point2f((float)c4.at<double>(0), (float)c4.at<double>(1)));
			double loss_area = cv::contourArea(contour1);
			double reduceW = 2.0 * loss_area / (double)imgSize_.height;
			virtWidth1 = (double)imgSize_.width - reduceW;
		}
		double parea = cv::contourArea(contour);
		perc = parea / (double)(imgSize_.width * imgSize_.height);
	}
}