/**********************************************************************************************************
 FILE: pose_helper.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: May 2016

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides helper functions for the estimation and optimization of poses between
			  two camera views (images).
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "glob_includes.h"
#include <Eigen/Core>

#include "poselib\poselib_api.h"

namespace poselib
{

/* --------------------------- Defines --------------------------- */

#define PI 3.14159265

/*
 * medErr ...	median of the reprojection errors masked as inliers
 * arithErr ... arithmetic mean value of the reprojection errors masked as inliers
 * arithStd	... standard deviation of the reprojection errors masked as inliers
 * medStd ... standard deviation of the reprojection errors masked as inliers using the median instead of the mean value -> median absolute deviation (MAD)
*/
typedef struct statVals {
		double medErr, arithErr, arithStd, medStd;
} statVals;

/* --------------------- Function prototypes --------------------- */

//Calculates the Sampson L1-distance for a point correspondence
void POSELIB_API SampsonL1(const cv::Mat x1, const cv::Mat x2, const cv::Mat E, double & denom1, double & num);
//Calculates the closest essential matrix
int POSELIB_API getClosestE(Eigen::Matrix3d & E);
//Validate the Essential/Fundamental matrix with the oriented epipolar constraint
bool POSELIB_API validateEssential(const cv::Mat p1, const cv::Mat p2, const Eigen::Matrix3d E, bool EfullCheck = false, cv::InputOutputArray _mask = cv::noArray());
//Checks, if determinants, etc. are too close to 0
inline bool POSELIB_API nearZero(double d);
//Calculates statistical parameters for the given values in the vector
void POSELIB_API getStatsfromVec(const std::vector<double> vals, statVals *stats, bool rejQuartiles = false);
//Extracts the 3D translation vector from the translation essential matrix.
cv::Mat POSELIB_API getTfromTransEssential(cv::Mat Et);
//Calculates the vector norm.
double POSELIB_API normFromVec(cv::Mat vec);
//Calculates the vector norm.
double POSELIB_API normFromVec(std::vector<double> vec);
//Calculates the reprojection errors for all correspondences and/or their statistics  
void POSELIB_API getReprojErrors(cv::Mat Essential, 
					 cv::InputArray p1, 
					 cv::InputArray p2, 
					 bool takeImageCoords, 
					 statVals* qp = NULL, 
					 std::vector<double> *repErr = NULL, 
					 cv::InputArray K1 = cv::noArray(), 
					 cv::InputArray K2 = cv::noArray(), 
					 bool EisF = false);
//Computes the Sampson distance (first-order geometric error) for the provided point correspondences in the form 2 rows x n columns.
void POSELIB_API computeReprojError1(cv::Mat X1, cv::Mat X2, cv::Mat E, std::vector<double> & error, double *error1 = NULL);
//Computes the Sampson distance (first-order geometric error) for the provided point correspondences in the form n rows x 2 columns.
void POSELIB_API computeReprojError2(cv::Mat X1, cv::Mat X2, cv::Mat E, std::vector<double> & error, double *error1 = NULL);
//Calculates the euler angles from a given rotation matrix.
void POSELIB_API getAnglesRotMat(cv::InputArray R, double & roll, double & pitch, double & yaw, bool useDegrees = true);
//Calculates the difference (roation angle) between two rotation quaternions and the distance between two 3D translation vectors.
void POSELIB_API getRTQuality(Eigen::Vector4d & R, Eigen::Vector4d & Rcalib, Eigen::Vector3d & T, 
				  Eigen::Vector3d & Tcalib, double* rdiff, double* tdiff);
//Calculates the essential matrix from the rotation matrix R and the translation
cv::Mat POSELIB_API getEfromRT(cv::Mat R, cv::Mat t);
//Generates a 3x3 skew-symmetric matrix from a 3-vector
cv::Mat POSELIB_API getSkewSymMatFromVec(cv::Mat t);
//Converts a (Rotation) Quaternion to a (Rotation) matrix
void POSELIB_API quatToMatrix(cv::Mat & R, cv::Mat q);
//Converts a (Rotation) matrix to a (Rotation) quaternion
void POSELIB_API MatToQuat(const Eigen::Matrix3d & rot, Eigen::Vector4d & quat);
//Converts a quaternion to axis angle representation
void POSELIB_API QuatToAxisAngle(Eigen::Vector4d quat, Eigen::Vector3d axis, double & angle);
//Calculates the product of a quaternion and a conjugated quaternion.
void POSELIB_API quatMultConj(const Eigen::Vector4d & Q1, const Eigen::Vector4d & Q2, Eigen::Vector4d & Qres);
//Normalizes the provided quaternion.
void POSELIB_API quatNormalise(Eigen::Vector4d & Q);
//Calculates the angle of a quaternion.
double POSELIB_API quatAngle(Eigen::Vector4d & Q);
//Multiplies a quaternion with a 3D-point (e.g. translation vector)
Eigen::Vector3d POSELIB_API quatMult3DPt(const Eigen::Vector4d & q, const Eigen::Vector3d & p);
//Multiplies a quaternion with a vector
void POSELIB_API quatMultByVec(const Eigen::Vector4d & Q1, const Eigen::Vector3d & vec, Eigen::Vector4d & Qres);
//Multiplies a quaternion with a conj. quaternion
void POSELIB_API quatMultConjIntoVec(const Eigen::Vector4d & Q1, const Eigen::Vector4d & Q2, Eigen::Vector3d & Qres);
//Calculates the difference (roation angle) between two rotation quaternions.
double POSELIB_API rotDiff(Eigen::Vector4d & R, Eigen::Vector4d & Rcalib);
//Calculates the transponse (inverse rotation) of a quaternion
Eigen::Vector4d POSELIB_API quatConj(const Eigen::Vector4d & Q);
//Normalizes the image coordinates and transfers the image coordinates into cameracoordinates, respectively.
void POSELIB_API ImgToCamCoordTrans(std::vector<cv::Point2f>& points, cv::Mat K);
//Transfers coordinates from the camera coordinate system into the the image coordinate system.
void POSELIB_API CamToImgCoordTrans(std::vector<cv::Point2f>& points, cv::Mat K);
//Transfers coordinates from the camera coordinate system into the the image coordinate system.
void POSELIB_API CamToImgCoordTrans(cv::Mat& points, cv::Mat K);
//This function removes the lens distortion for corresponding points
bool POSELIB_API Remove_LensDist(std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, cv::Mat dist1, cv::Mat dist2);

void POSELIB_API compareRTs(cv::Mat R1, cv::Mat R2, cv::Mat t1, cv::Mat t2, double *rdiff, double *tdiff, bool printDiff = false);

}