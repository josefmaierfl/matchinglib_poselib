/**********************************************************************************************************
FILE: getStereoCameraExtr.h

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: Bebruary 2018

LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for generating optimal camera paramters given a disired
overlap area ratio between the views and some restrictions on the camera paramters
**********************************************************************************************************/

#pragma once

#include "glob_includes.h"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include <stdexcept>
#include <random>

#include "generateVirtualSequenceLib\generateVirtualSequenceLib_api.h"

/* --------------------------- Defines --------------------------- */

#define PI 3.14159265

/* --------------------------- Classes --------------------------- */

class InvalidDataStructureException : public std::runtime_error
{
public:
	InvalidDataStructureException(std::string mess) : std::runtime_error(mess) {}
};

class GENERATEVIRTUALSEQUENCELIB_API GenStereoPars
{
public:
	//Ranges must be in the signed form [lower_bound, upper_bound] or [single_value]:
	//e.g.: [-5, -1] or [1, 5] or [3]
	//For tx and ty: the largest absolute value must be negative
	GenStereoPars(std::vector<std::vector<double>> tx, 
		std::vector<std::vector<double>> ty, 
		std::vector<std::vector<double>> tz, 
		std::vector<std::vector<double>> roll, 
		std::vector<std::vector<double>> pitch, 
		std::vector<std::vector<double>> yaw, 
		double approxImgOverlap, cv::Size imgSize);

	int optimizeRtf(int verbose = 0);
	void setLMTolerance(double rollTol, double pitchTol, double txTol, double tyTol, double fTol);
	bool getCamPars(std::vector<cv::Mat>& Rv, std::vector<cv::Mat>& tv, cv::Mat& K_1, cv::Mat& K_2);
	bool getEulerAngles(std::vector<double>& roll, std::vector<double>& pitch, std::vector<double>& yaw);

public:
	std::vector<cv::Mat> Ris;//Final Rotation matrizes
	std::vector<cv::Mat> tis;//Final translation vectors
	cv::Mat K1, K2;//Camera matrices

private:
	std::default_random_engine rand_generator;

	std::vector<std::vector<double>> tx_;
	std::vector<std::vector<double>> ty_;
	std::vector<std::vector<double>> tz_;
	std::vector<std::vector<double>> roll_;
	std::vector<std::vector<double>> pitch_;
	std::vector<std::vector<double>> yaw_;
	double approxImgOverlap_;
	cv::Size imgSize_;
	size_t nrConditions;

	bool txRangeEqual;
	bool tyRangeEqual;
	bool tzRangeEqual;
	bool rollRangeEqual;
	bool pitchRangeEqual;
	bool yawRangeEqual;

	std::vector<double> tx_use;
	std::vector<double> ty_use;
	std::vector<double> tz_use;
	std::vector<double> roll_use;
	std::vector<double> pitch_use;
	std::vector<double> yaw_use;
	double f;
	double fRange[2];

	const double angView[2] = { 45.0 * PI / 180.0, 120.0 * PI / 180.0 };//Min and max viewing angles of the cameras

	std::vector<double> virtWidth;
	bool horizontalCamAlign = true;
	bool aliChange = true;

	//Virtual coordinates within the images for optimization
	std::vector<cv::Mat> x_lb_max1;
	std::vector<cv::Mat> x_lb_min1;
	std::vector<cv::Mat> x_rt_max1;
	std::vector<cv::Mat> x_rt_min1;

	std::vector<cv::Mat> x_lb_max2;
	std::vector<cv::Mat> x_lb_min2;
	std::vector<cv::Mat> x_rt_max2;
	std::vector<cv::Mat> x_rt_min2;

	//Number of residuals per condition
	const int nr_residualsPCond = 9;

	double rollTol_ = 1e-4;
	double pitchTol_ = 1e-4;
	double txTol_ = 1e-6;
	double tyTol_ = 1e-6;
	double fTol_ = 1e-3;

private:
	void checkParameterFormat(std::vector<std::vector<double>> par, std::string name);
	void checkEqualRanges(std::vector<std::vector<double>> par, bool& areEqual);
	//Get a random floating point number between 2 ranges
	inline double getRandDoubleVal(double lowerBound, double upperBound);
	void initRandPars(std::vector<std::vector<double>>& parIn, bool& rangeEqual, std::vector<double>& parOut);
	void getRotRectDiffArea(double yaw_angle, double& perc, double& virtWidth);
	bool helpNewRandEquRangeVals(int& idx, const int maxit);
	void setCoordsForOpti();
	int LMFsolve(cv::Mat p,
		cv::Mat& xf,
		cv::Mat& residuals,
		cv::InputArray funcTol = cv::noArray(),
		cv::InputArray xTol = cv::noArray(),
		size_t maxIter = 100,
		int verbose = 0);
	cv::Mat LMfunc(cv::Mat p);
	double alignFunctionNoLargePos(double tx, double ty);
	double alignFunctionHori(double tx, double ty);
	double alignFunctionVert(double tx, double ty);
	void CalcExtrFromPar(double roll, double pitch, double tx, double ty, int cnf, cv::Mat& t, cv::Mat& R);
	cv::Mat getLineCam1(cv::Mat K, cv::Mat x);
	void getLineCam2(cv::Mat R, cv::Mat t, cv::Mat K, cv::Mat x, cv::Mat& a, cv::Mat& b);
	double getLineIntersect(cv::Mat b1, cv::Mat a2, cv::Mat b2);
	bool solveLinEqu(cv::Mat& A, cv::Mat& b, cv::Mat& x);
	inline void pow2mult(double& val, size_t num);
	double getDistance2LinesPlane(cv::Mat a1, cv::Mat b1, cv::Mat a2, cv::Mat b2, double z);
	cv::Mat finjac(cv::Mat& residuals, cv::Mat& x, cv::Mat& xTol);
	cv::Mat getD(cv::Mat DvecLTL, cv::Mat parVec, double lamda);
	double getDampingF(double par, int rangeIdx0);
	void printit(int ipr, int cnt, size_t nfJ, double ss, cv::Mat x, cv::Mat dx, double lamda, double lamda_c);
	cv::Mat getNormalDistributionVals(int sizeV, double mean, double stddev);
	void adaptParVec(cv::Mat& parVec, cv::Mat& parVecOut, cv::Mat& deltaValid);
	int optParLM(int verbose);
};

/* --------------------- Function prototypes --------------------- */

cv::Mat eulerAnglesToRotationMatrix(double x, double y, double z);
bool any_vec_cv(cv::Mat bin);
bool isfinite_vec_cv(cv::Mat bin);

/* -------------------------- Functions -------------------------- */

//Checks, if determinants, etc. are too close to 0
inline bool nearZero(double d)
{
	//Decide if determinants, etc. are too close to 0 to bother with
	const double EPSILON = 1e-4;
	return (d<EPSILON) && (d>-EPSILON);
}
