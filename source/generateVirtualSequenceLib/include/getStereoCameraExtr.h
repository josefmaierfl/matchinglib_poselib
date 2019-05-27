/**********************************************************************************************************
FILE: getStereoCameraExtr.h

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: February 2018

LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for generating optimal camera paramters given a disired
overlap area ratio between the views and some restrictions on the camera parameters
**********************************************************************************************************/

#pragma once

#include "glob_includes.h"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include <stdexcept>
#include <random>

#include "generateVirtualSequenceLib/generateVirtualSequenceLib_api.h"

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

    GenStereoPars(){}

	//Copy constructor
    GenStereoPars(const GenStereoPars &gsp):
            Ris(gsp.Ris),
            tis(gsp.tis),
            K1(gsp.K1),
            K2(gsp.K2),
            tx_(gsp.tx_),
            ty_(gsp.ty_),
            tz_(gsp.tz_),
            roll_(gsp.roll_),
            pitch_(gsp.pitch_),
            yaw_(gsp.yaw_),
            approxImgOverlap_(gsp.approxImgOverlap_),
            meanOvLapError_sv(gsp.meanOvLapError_sv),
            negMaxOvLapError_sv(gsp.negMaxOvLapError_sv),
            posMaxOvLapError_sv(gsp.posMaxOvLapError_sv),
            sumSqrRes(gsp.sumSqrRes),
            imgSize_(gsp.imgSize_),
            nrConditions(gsp.nrConditions),
            txRangeEqual(gsp.txRangeEqual),
            tyRangeEqual(gsp.tyRangeEqual),
            tzRangeEqual(gsp.tzRangeEqual),
            rollRangeEqual(gsp.rollRangeEqual),
            pitchRangeEqual(gsp.pitchRangeEqual),
            yawRangeEqual(gsp.yawRangeEqual),
            tx_use(gsp.tx_use),
            ty_use(gsp.ty_use),
            tz_use(gsp.tz_use),
            roll_use(gsp.roll_use),
            pitch_use(gsp.pitch_use),
            yaw_use(gsp.yaw_use),
            f(gsp.f),
            fRange{gsp.fRange[0], gsp.fRange[1]},
            iOvLapMult(gsp.iOvLapMult){
        meanOvLapError = 0;
        negMaxOvLapError = DBL_MAX;
        posMaxOvLapError = DBL_MIN;
        virtWidth.clear();
        x_lb_max1.clear();
        x_lb_min1.clear();
        x_rt_max1.clear();
        x_rt_min1.clear();
        x_lb_max2.clear();
        x_lb_min2.clear();
        x_rt_max2.clear();
        x_rt_min2.clear();
    }

    //Copy assignment operator
    GenStereoPars& operator=(const GenStereoPars& gsp){
        Ris = gsp.Ris;
        tis = gsp.tis;
        K1 = gsp.K1;
        K2 = gsp.K2;
        tx_ = gsp.tx_;
        ty_ = gsp.ty_;
        tz_ = gsp.tz_;
        roll_ = gsp.roll_;
        pitch_ = gsp.pitch_;
        yaw_ = gsp.yaw_;
        approxImgOverlap_ = gsp.approxImgOverlap_;
        imgSize_ = gsp.imgSize_;
        nrConditions = gsp.nrConditions;
        txRangeEqual = gsp.txRangeEqual;
        tyRangeEqual = gsp.tyRangeEqual;
        tzRangeEqual = gsp.tzRangeEqual;
        rollRangeEqual = gsp.rollRangeEqual;
        pitchRangeEqual = gsp.pitchRangeEqual;
        yawRangeEqual = gsp.yawRangeEqual;
        tx_use = gsp.tx_use;
        ty_use = gsp.ty_use;
        tz_use = gsp.tz_use;
        roll_use = gsp.roll_use;
        pitch_use = gsp.pitch_use;
        yaw_use = gsp.yaw_use;
        f = gsp.f;
        fRange[0] = gsp.fRange[0];
        fRange[1] = gsp.fRange[1];
        iOvLapMult = gsp.iOvLapMult;
        meanOvLapError_sv = gsp.meanOvLapError_sv;
        negMaxOvLapError_sv = gsp.negMaxOvLapError_sv;
        posMaxOvLapError_sv = gsp.posMaxOvLapError_sv;
        sumSqrRes = gsp.sumSqrRes;
        meanOvLapError = 0;
        negMaxOvLapError = DBL_MAX;
        posMaxOvLapError = DBL_MIN;
        virtWidth.clear();
        x_lb_max1.clear();
        x_lb_min1.clear();
        x_rt_max1.clear();
        x_rt_min1.clear();
        x_lb_max2.clear();
        x_lb_min2.clear();
        x_rt_max2.clear();
        x_rt_min2.clear();
        return *this;
    }

	int optimizeRtf(int verbose = 0);
	void setLMTolerance(double rollTol, double pitchTol, double txTol, double tyTol, double fTol);
	bool getCamPars(std::vector<cv::Mat>& Rv, std::vector<cv::Mat>& tv, cv::Mat& K_1, cv::Mat& K_2);
	bool getEulerAngles(std::vector<double>& roll, std::vector<double>& pitch, std::vector<double>& yaw);
	void getNewRandSeed();
	double getMeanOverlapError(){
	    return meanOvLapError;
	}
    double getNegMaxOvLapError(){
        return negMaxOvLapError;
    }
    double getPosMaxOvLapError(){
        return posMaxOvLapError;
    }
    double getSavedMeanOverlapError(){
        return meanOvLapError_sv;
    }
    double getSavedNegMaxOvLapError(){
        return negMaxOvLapError_sv;
    }
    double getSavedPosMaxOvLapError(){
        return posMaxOvLapError_sv;
    }
    double getSumofSquredResiduals(){
	    return sumSqrRes;
	}

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
	double approxImgOverlap_ = 0;
    double meanOvLapError = 0;
    double negMaxOvLapError = DBL_MAX;
    double posMaxOvLapError = DBL_MIN;
    double meanOvLapError_sv = 0;
    double negMaxOvLapError_sv = DBL_MAX;
    double posMaxOvLapError_sv = DBL_MIN;
    double sumSqrRes = 0;
	cv::Size imgSize_ = cv::Size(0,0);
	size_t nrConditions = 0;

	bool txRangeEqual = false;
	bool tyRangeEqual = false;
	bool tzRangeEqual = false;
	bool rollRangeEqual = false;
	bool pitchRangeEqual = false;
	bool yawRangeEqual = false;

	std::vector<double> tx_use;
	std::vector<double> ty_use;
	std::vector<double> tz_use;
	std::vector<double> roll_use;
	std::vector<double> pitch_use;
	std::vector<double> yaw_use;
	double f = 0;
	double fRange[2] = {0,0};

	const double angView[2] = { 45.0 * PI / 180.0, 120.0 * PI / 180.0 };//Min and max viewing angles of the cameras

	std::vector<double> virtWidth;
	bool horizontalCamAlign = true;
	//bool aliChange = true;

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

	//Multiplication factor for the quadratic image overlap difference to adapt to the residual range
	double iOvLapMult = 1.0;

	//Save optimization result before alignment changes
	cv::Mat r_before_aliC, x_before_aliC;

private:
	void checkParameterFormat(std::vector<std::vector<double>> par, std::string name);
	void checkEqualRanges(std::vector<std::vector<double>> par, bool& areEqual);
	//Get a random floating point number between 2 ranges
	inline double getRandDoubleVal(double lowerBound, double upperBound);
	void initRandPars(std::vector<std::vector<double>>& parIn, bool& rangeEqual, std::vector<double>& parOut);
	void getRotRectDiffArea(double yaw_angle, double& perc, double& virtWidth1);
	bool helpNewRandEquRangeVals(int& idx, const int maxit, int align);
	void setCoordsForOpti();
	int LMFsolve(cv::Mat p,
		cv::Mat& xf,
		cv::Mat& residuals,
		cv::InputArray funcTol = cv::noArray(),
		cv::InputArray xTol = cv::noArray(),
		size_t maxIter = 100,
		int verbose = 0);
	cv::Mat LMfunc(cv::Mat p, bool noAlignCheck = false);
	double alignFunctionNoLargePos(double tx, double ty);
	double alignFunctionHori(double tx, double ty);
	double alignFunctionVert(double tx, double ty);
	void CalcExtrFromPar(double roll, double pitch, double tx, double ty, int cnf, cv::Mat& t, cv::Mat& R);
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

/* -------------------------- Functions -------------------------- */
