/**********************************************************************************************************
FILE: generateSequence.h

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: March 2018

LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides some helper functions.
**********************************************************************************************************/

#pragma once

#include "glob_includes.h"
#include "opencv2/highgui/highgui.hpp"
#include <random>

#include <Eigen/Dense>

//#include "generateVirtualSequenceLib\generateVirtualSequenceLib_api.h"

/* --------------------------- Defines --------------------------- */

/* --------------------------- Classes --------------------------- */

/* --------------------- Function prototypes --------------------- */

//Initializes the random number generator with a seed based on the current time
void randSeed(std::default_random_engine& rand_generator);

//Get a random number within a given range
double getRandDoubleValRng(double lowerBound, double upperBound, std::default_random_engine rand_generator = std::default_random_engine((unsigned int)std::rand()));

//construct a rotation matrix from angles given in RAD
cv::Mat eulerAnglesToRotationMatrix(double x, double y, double z);

//Returns true, if any element of the boolean (CV_8UC1) Mat vector is also true
bool any_vec_cv(cv::Mat bin);

//Returns true, if every element of the double (CV_64FC1) Mat vector is a finite number (no element is infinity nor NaN)
bool isfinite_vec_cv(cv::Mat bin);

//Generates the 3D direction vector for a camera at the origin and a given pixel coordinate
cv::Mat getLineCam1(cv::Mat K, cv::Mat x);

//Generates a 3D line (start coordinate & direction vector) for a second camera within a stereo alignment (cam not at the origin) and a given pixel coordinate
void getLineCam2(cv::Mat R, cv::Mat t, cv::Mat K, cv::Mat x, cv::Mat& a, cv::Mat& b);

//Calculate the z - distance of the intersection of 2 3D lines or the mean z - distance at the shortest perpendicular between 2 skew lines in 3D.
double getLineIntersect(cv::Mat b1, cv::Mat a2, cv::Mat b2);

//Solves a linear equation of th form Ax=b
bool solveLinEqu(cv::Mat& A, cv::Mat& b, cv::Mat& x);

//Converts a (Rotation) matrix to a (Rotation) quaternion
void MatToQuat(const Eigen::Matrix3d & rot, Eigen::Vector4d & quat);

/* -------------------------- Functions -------------------------- */

//Checks, if determinants, etc. are too close to 0
inline bool nearZero(double d)
{
	//Decide if determinants, etc. are too close to 0 to bother with
	const double EPSILON = 1e-4;
	return (d<EPSILON) && (d>-EPSILON);
}