/**********************************************************************************************************
FILE: generateSequence.h

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: March 2018

LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides some helper functions.
**********************************************************************************************************/

#pragma once

#include "glob_includes.h"
#include "opencv2/highgui/highgui.hpp"
#include <random>

#include <Eigen/Dense>

#include "generateVirtualSequenceLib/generateVirtualSequenceLib_api.h"

/* --------------------------- Defines --------------------------- */

/* --------------------------- Classes --------------------------- */

/* --------------------- Function prototypes --------------------- */

//Initializes the random number generator with a seed based on the current time
long int randSeed(std::default_random_engine& rand_generator);

//Initializes the random number generator with a given seed based on the current time
void randSeed(std::mt19937& rand_generator, long int seed);

//Get a random number within a given range
double getRandDoubleValRng(double lowerBound, double upperBound, std::default_random_engine rand_generator = std::default_random_engine((unsigned int)std::rand()));

//construct a rotation matrix from angles given in RAD
cv::Mat GENERATEVIRTUALSEQUENCELIB_API eulerAnglesToRotationMatrix(double x, double y, double z);

//Returns true, if any element of the boolean (CV_8UC1) Mat vector is also true
bool any_vec_cv(const cv::Mat& bin);

//Returns true, if every element of the double (CV_64FC1) Mat vector is a finite number (no element is infinity nor NaN)
bool isfinite_vec_cv(const cv::Mat& bin);

//Generates the 3D direction vector for a camera at the origin and a given pixel coordinate
cv::Mat getLineCam1(const cv::Mat& K, const cv::Mat& x);

//Generates a 3D line (start coordinate & direction vector) for a second camera within a stereo alignment (cam not at the origin) and a given pixel coordinate
void getLineCam2(const cv::Mat& R, const cv::Mat& t, const cv::Mat& K, const cv::Mat& x, cv::Mat& a, cv::Mat& b);

//Calculate the z - distance of the intersection of 2 3D lines or the mean z - distance at the shortest perpendicular between 2 skew lines in 3D.
double getLineIntersect(const cv::Mat& b1, const cv::Mat& a2, const cv::Mat& b2);

//Solves a linear equation of th form Ax=b
bool solveLinEqu(const cv::Mat& A, const cv::Mat& b, cv::Mat& x);

//Converts a (Rotation) matrix to a (Rotation) quaternion
void MatToQuat(const Eigen::Matrix3d & rot, Eigen::Vector4d & quat);

//Checks if a 3x3 matrix is a rotation matrix
bool isMatRotationMat(const cv::Mat& R);

//Checks if a 3x3 matrix is a rotation matrix
bool isMatRotationMat(Eigen::Matrix3d R);

//Calculates the difference (rotation angle) between two rotation quaternions.
double rotDiff(Eigen::Vector4d & R1, Eigen::Vector4d & R2);

//Calculates the difference (rotation angle) between two rotation matrices.
double rotDiff(const cv::Mat& R1, const cv::Mat& R2);

//Calculates the difference (rotation angle) between two rotation matrices.
double rotDiff(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2);

//Calculates the difference (rotation angle) between two camera projection matrices.
double rotDiff(const Eigen::Matrix4d& R1, const Eigen::Matrix4d& R2);

//Calculates the difference (rotation angle) between two camera projection matrices.
double rotDiff(const Eigen::Matrix4f& R1, const Eigen::Matrix4f& R2);

//Calculates the product of a quaternion and a conjugated quaternion.
void quatMultConj(const Eigen::Vector4d & Q1, const Eigen::Vector4d & Q2, Eigen::Vector4d & Qres);

//Normalizes the provided quaternion.
void quatNormalise(Eigen::Vector4d & Q);

//Calculates the angle of a quaternion.
double quatAngle(Eigen::Vector4d & Q);

//Round every entry of a matrix to its nearest integer
cv::Mat roundMat(const cv::Mat& m);

/* -------------------------- Functions -------------------------- */

//Checks, if determinants, etc. are too close to 0
inline bool nearZero(double d)
{
	//Decide if determinants, etc. are too close to 0 to bother with
	const double EPSILON = 1e-4;
	return (d<EPSILON) && (d>-EPSILON);
}