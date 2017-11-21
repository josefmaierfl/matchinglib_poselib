/**********************************************************************************************************
 FILE: pose_helper.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: May 2016

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides helper functions for the estimation and optimization of poses between
              two camera views (images).
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "poselib/glob_includes.h"
#include <Eigen/Core>

#include "poselib/poselib_api.h"

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
bool POSELIB_API validateEssential(const cv::Mat p1, const cv::Mat p2, const Eigen::Matrix3d E, bool EfullCheck = false, cv::InputOutputArray _mask = cv::noArray(), bool tryOrientedEpipolar = false);
//Checks, if determinants, etc. are too close to 0
inline bool POSELIB_API nearZero(double d)
{
    //Decide if determinants, etc. are too close to 0 to bother with
    const double EPSILON = 1e-3;
    return (d<EPSILON) && (d>-EPSILON);
}
//Calculates statistical parameters for the given values in the vector
void POSELIB_API getStatsfromVec(const std::vector<double> vals, statVals *stats, bool rejQuartiles = false, bool roundStd = true);
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
void POSELIB_API getRTQuality(cv::Mat & R, cv::Mat & Rcalib, cv::Mat & T,
                  cv::Mat & Tcalib, double* rdiff, double* tdiff);
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
//Calculates the difference (roation angle) between two rotation quaternions and the distance between two 3D translation vectors
void POSELIB_API compareRTs(cv::Mat R1, cv::Mat R2, cv::Mat t1, cv::Mat t2, double *rdiff, double *tdiff, bool printDiff = false);
//Calculation of the rectifying matrices based on the extrinsic and intrinsic camera parameters.
int POSELIB_API getRectificationParameters(cv::InputArray R,
                              cv::InputArray t,
                              cv::InputArray K1,
                              cv::InputArray K2,
                              cv::InputArray distcoeffs1,
                              cv::InputArray distcoeffs2,
                              cv::Size imageSize,
                              cv::OutputArray Rect1,
                              cv::OutputArray Rect2,
                              cv::OutputArray K1new,
                              cv::OutputArray K2new,
                              double alpha = -1,
                              bool globRectFunct = true,
                              cv::Size newImgSize = cv::Size(),
                              cv::Rect *roi1 = NULL,
                              cv::Rect *roi2 = NULL,
                              cv::OutputArray P1new = cv::noArray(),
                              cv::OutputArray P2new = cv::noArray());
//Estimates the optimal scale for the focal length of the virtuel camera.
double POSELIB_API estimateOptimalFocalScale(double alpha, cv::Mat K1, cv::Mat K2, cv::Mat R1, cv::Mat R2, cv::Mat P1, cv::Mat P2,
                                 cv::Mat dist1, cv::Mat dist2, cv::Size imageSize, cv::Size newImgSize);
//Estimates the vergence (shift of starting point) for correspondence search in the stereo engine.
int POSELIB_API estimateVergence(cv::Mat R, cv::Mat RR1, cv::Mat RR2, cv::Mat PR1, cv::Mat PR2);
//This function shows the rectified images.
int POSELIB_API ShowRectifiedImages(cv::InputArray img1, cv::InputArray img2, cv::InputArray mapX1, cv::InputArray mapY1, cv::InputArray mapX2, cv::InputArray mapY2, cv::InputArray t, std::string path, cv::Size newImgSize = cv::Size());
// This function return the rectified images
int POSELIB_API GetRectifiedImages(cv::InputArray img1, cv::InputArray img2, cv::InputArray mapX1, cv::InputArray mapY1, cv::InputArray mapX2, cv::InputArray mapY2, cv::InputArray t, cv::OutputArray outImg1, cv::OutputArray outImg2, cv::Size newImgSize = cv::Size());
//This function estimates an initial delta value for the SPRT test used within USAC.
double estimateSprtDeltaInit(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2, double th, cv::Size imgSize);
//This function estimates an initial epsilon value for the SPRT test used within USAC.
double estimateSprtEpsilonInit(std::vector<cv::DMatch> matches, unsigned int nrMatchesVfcFiltered);
//This function generates an index of the matches with the lowest matching costs first.
void getSortedMatchIdx(std::vector<cv::DMatch> matches, std::vector<unsigned int> & sortedMatchIdx);
//Checks if a 3x3 matrix is a rotation matrix
bool POSELIB_API isMatRoationMat(cv::Mat R);
//Checks if a 3x3 matrix is a rotation matrix
bool POSELIB_API isMatRoationMat(Eigen::Matrix3d R);
//Calculates the Sampson L2 error for 1 correspondence
double POSELIB_API getSampsonL2Error(cv::InputArray E, cv::InputArray x1, cv::InputArray x2);
//Calculates the Sampson L2 error for 1 correspondence
double POSELIB_API getSampsonL2Error(Eigen::Matrix3d E, Eigen::Vector3d x1, Eigen::Vector3d x2);
//Checks for a given vector of error values if they are inliers or not in respect to threshold th.
size_t POSELIB_API getInlierMask(std::vector<double> error, double th, cv::Mat & mask);
}
