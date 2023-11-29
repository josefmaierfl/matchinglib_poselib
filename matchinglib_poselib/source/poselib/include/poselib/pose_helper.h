//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2019 AIT Austrian Institute of Technology GmbH
//
//Permission is hereby granted, free of charge, to any person obtaining
//a copy of this software and associated documentation files (the "Software"),
//to deal in the Software without restriction, including without limitation
//the rights to use, copy, modify, merge, publish, distribute, sublicense,
//and/or sell copies of the Software, and to permit persons to whom the
//Software is furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included
//in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//Author: Josef Maier (josefjohann-dot-maier-at-gmail-dot-at)
/**********************************************************************************************************
 FILE: pose_helper.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: May 2016

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides helper functions for the estimation and optimization of poses between
              two camera views (images).
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "poselib/glob_includes.h"
#include <Eigen/Core>
#include <unordered_map>
#include <unordered_set>

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
void POSELIB_API SampsonL1(const cv::Mat &x1, const cv::Mat &x2, const cv::Mat &E, double & denom1, double & num);
//Calculates the closest essential matrix
int POSELIB_API getClosestE(Eigen::Matrix3d & E);
//Validate the Essential/Fundamental matrix with the oriented epipolar constraint
bool POSELIB_API validateEssential(const cv::Mat &p1,
                                   const cv::Mat &p2,
                                   Eigen::Matrix3d E,
                                   bool EfullCheck = false,
                                   cv::InputOutputArray _mask = cv::noArray(),
                                   bool tryOrientedEpipolar = false);
//Checks, if determinants, etc. are too close to 0
template <class T>
inline bool POSELIB_API nearZero(const T d, const double EPSILON = 1e-4)
{
    return (static_cast<double>(d) < EPSILON) && (static_cast<double>(d) > -EPSILON);
}
template <class T>
inline void POSELIB_API hash_combine(std::size_t &seed, const T &v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct POSELIB_API pair_hash
{
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &x) const
    {
        std::size_t seed = 0;
        hash_combine(seed, x.first);
        hash_combine(seed, x.second);
        return seed;
        return std::hash<T1>()(x.first) ^ std::hash<T2>()(x.second);
    }
};

struct POSELIB_API pair_EqualTo
{
    template <class T1, class T2>
    bool operator()(const std::pair<T1, T2> &x1, const std::pair<T1, T2> &x2) const
    {
        const bool equal1 = (x1.first == x2.first);
        const bool equal2 = (x1.second == x2.second);
        return equal1 && equal2;
    }
};

inline void POSELIB_API convertPrecision(double &inOut, const double &preci = FLT_EPSILON)
{
    if (nearZero(preci - 1.0) || preci > 1.0)
    {
        return;
    }
    else if (nearZero(preci / FLT_EPSILON - 1.0))
    {
        const float tmp = static_cast<float>(inOut);
        inOut = static_cast<double>(tmp);
    }
    else if (preci > DBL_EPSILON)
    {
        inOut = std::round(inOut / preci) * preci;
    }
}

inline double POSELIB_API convertPrecisionRet(const double &in, const double &preci = FLT_EPSILON)
{
    if (nearZero(preci - 1.0) || preci > 1.0)
    {
        return in;
    }
    else if (nearZero(preci / FLT_EPSILON - 1.0))
    {
        const float tmp = static_cast<float>(in);
        return static_cast<double>(tmp);
    }
    else if (preci > DBL_EPSILON)
    {
        return std::round(in / preci) * preci;
    }
    return in;
}

inline void POSELIB_API convertPrecisionMat(cv::InputOutputArray mat, const double &preci = FLT_EPSILON)
{
    CV_Assert(mat.type() == CV_64FC1);
    cv::Mat mat_ = mat.getMat();
    for (int y = 0; y < mat_.rows; y++)
    {
        for (int x = 0; x < mat_.cols; x++)
        {
            convertPrecision(mat_.at<double>(y, x), preci);
        }
    }
}

template <class T>
T POSELIB_API getMedian(std::vector<T> &measurements)
{
    const size_t length = measurements.size();
    std::sort(measurements.begin(), measurements.end());
    T median;
    if (length % 2)
    {
        median = measurements[(length - 1) / 2];
    }
    else
    {
        median = (measurements[length / 2] + measurements[length / 2 - 1]) / static_cast<T>(2.0);
    }
    return median;
}

template <class T>
T POSELIB_API getMean(const std::vector<T> &measurements)
{
    T mean = 0;
    T n_d = static_cast<T>(measurements.size());
    for (const auto &val : measurements)
    {
        mean += val;
    }
    mean /= n_d;

    return mean;
}

template <class T>
void POSELIB_API getMeanStandardDeviation(const std::vector<T> &measurements, T &mean, T &sd)
{
    mean = 0;
    T err2sum = 0;
    T n_d = static_cast<T>(measurements.size());
    for (const auto &val : measurements)
    {
        mean += val;
        err2sum += val * val;
    }
    mean /= n_d;

    T hlp = err2sum - n_d * mean * mean;
    sd = std::sqrt(hlp / (n_d - static_cast<T>(1.0)));
}

template <class T>
std::pair<T, T> POSELIB_API getMeanStandardDeviation(const std::vector<T> &measurements)
{
    T mean, sd;
    getMeanStandardDeviation(measurements, mean, sd);
    return std::make_pair(mean, sd);
}
//Calculates statistical parameters for the given values in the vector
void POSELIB_API getStatsfromVec(const std::vector<double> &vals, statVals *stats, bool rejQuartiles = false, bool roundStd = true);
//Extracts the 3D translation vector from the translation essential matrix.
cv::Mat POSELIB_API getTfromTransEssential(cv::Mat Et);
//Calculates the vector norm.
double POSELIB_API normFromVec(const cv::Mat& vec);
//Calculates the vector norm.
double POSELIB_API normFromVec(std::vector<double> vec);

//Calculates the reprojection errors for all correspondences and/or their statistics
void POSELIB_API getReprojErrors(const cv::Mat& Essential,
                     cv::InputArray p1,
                     cv::InputArray p2,
                     bool takeImageCoords,
                     statVals* qp = nullptr,
                     std::vector<double> *repErr = nullptr,
                     cv::InputArray K1 = cv::noArray(),
                     cv::InputArray K2 = cv::noArray(),
                     bool EisF = false);
//Computes the Sampson distance (first-order geometric error) for the provided point correspondences in the form 2 rows x n columns.
void POSELIB_API computeReprojError1(cv::Mat X1, cv::Mat X2, const cv::Mat& E, std::vector<double> & error, double *error1 = nullptr);
//Computes the Sampson distance (first-order geometric error) for the provided point correspondences in the form n rows x 2 columns.
void POSELIB_API computeReprojError2(cv::Mat X1, cv::Mat X2, const cv::Mat& E, std::vector<double> & error, double *error1 = nullptr);
//Computes the Sampson distance (first-order geometric error) for the provided point correspondence
double POSELIB_API calculateSampsonError(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &F);
//Computes the squared distance to the epipolar line in the second image for the provided point correspondence
double POSELIB_API calculateEpipolarLineDistanceImg2Squared(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &F);
//Computes the squared distance to the epipolar line in the first image for the provided point correspondence
double POSELIB_API calculateEpipolarLineDistanceImg1Squared(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &F);
//Computes the squared distance to the epipolar line in the second image for the provided (single) point correspondence
double POSELIB_API calculateEpipolarLineDistanceImg2Squared(const cv::Mat &p1, const cv::Mat &p2, const cv::Mat &F);
//Computes the squared distance to the epipolar line in the first image for the provided (single) point correspondence
double POSELIB_API calculateEpipolarLineDistanceImg1Squared(const cv::Mat &p1, const cv::Mat &p2, const cv::Mat &F);
//Computes sqrt distances for every provided point correspondence of mean 1/2 * (e1 + e2) squared distances e1 and e2 to epipolar lines in first and second images
void POSELIB_API calculateEpipolarLineDistanceImg12(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2, const cv::Mat &F, std::vector<double> &errors);
//Computes sqrt distances for every provided point correspondence of mean 1/2 * (e1 + e2) squared distances e1 and e2 to epipolar lines in first and second images
void POSELIB_API calculateEpipolarLineDistanceImg12(const cv::Mat &points1, const cv::Mat &points2, const cv::Mat &F, std::vector<double> &errors);
//Computes sqrt mean distance over all provided point correspondence of mean 1/2 * (e1 + e2) squared distances e1 and e2 to epipolar lines in first and second images
double POSELIB_API calculateMeanEpipolarLineDistanceImg12(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2, const cv::Mat &F);
//Computes sqrt mean distance over all provided point correspondence of mean 1/2 * (e1 + e2) squared distances e1 and e2 to epipolar lines in first and second images
double POSELIB_API calculateMeanEpipolarLineDistanceImg12(const cv::Mat &points1, const cv::Mat &points2, const cv::Mat &F);
//Compute the epipolar error of a single point correspondence in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) utilizing the Essential matrix E
double POSELIB_API calculateEpipolarError(const Eigen::Vector3d &x1h, const Eigen::Vector3d &x2h, const Eigen::Matrix3d &E);
//Compute the epipolar error of a single point correspondence in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y] utilizing the Essential matrix E
double POSELIB_API calculateEpipolarError(const Eigen::Vector2d &x1, const Eigen::Vector2d &x2, const Eigen::Matrix3d &E);

template <typename T>
double calculateEpipolarLineDistanceImg12SquaredMean(const T &p1, const T &p2, const cv::Mat &F)
{
    const double e2 = calculateEpipolarLineDistanceImg2Squared(p1, p2, F);
    const double e1 = calculateEpipolarLineDistanceImg1Squared(p2, p1, F.t());
    return (e1 + e2) / 2.0;
}

//Reject 3D points and image projections with high error values
bool POSELIB_API rejectHighErr3D(cv::Mat &Q, 
                                 std::vector<cv::Mat> &map3D, 
                                 std::vector<cv::Mat> &pts2D, 
                                 const std::vector<cv::Mat> &Rs, 
                                 const std::vector<cv::Mat> &ts, 
                                 const std::vector<cv::Mat> &Ks, 
                                 const int &n_cams, 
                                 const double &th, 
                                 const float minRemainRatio = 0.33f, 
                                 cv::InputArray dists = cv::noArray(), 
                                 cv::InputArray cheirality_mask = cv::noArray(), 
                                 cv::InputOutputArray Q2 = cv::noArray(), 
                                 const bool forceFiltering = false, 
                                 std::vector<std::vector<cv::Point2f>> *pts2Df = nullptr);
//Clone all parameters and 3D points for n cameras
void POSELIB_API cloneParameters(const std::vector<cv::Mat> &Rs_in, const std::vector<cv::Mat> &ts_in,
                                 std::vector<cv::Mat> &Rs_out, std::vector<cv::Mat> &ts_out,
                                 cv::InputArray Qs_in = cv::noArray(), cv::OutputArray Qs_out = cv::noArray(),
                                 cv::InputArray Ks_in = cv::noArray(), std::vector<cv::Mat> *Ks_out = nullptr,
                                 cv::InputArray dists_in = cv::noArray(), std::vector<cv::Mat> *dists_out = nullptr);
//Project 3D point into image plane (no distortion) resulting in a homogeneous coordinate [u v 1]
cv::Mat_<double> POSELIB_API project3D(const cv::Vec3d &Q, const cv::Matx33d &R, const cv::Vec3d &t, const cv::Matx33d &K);
//Calculates squared distance between a given image coordinate and a 3D point projected into the image plane (no distortion)
double POSELIB_API reproject3DDist2(const cv::Vec3d &Q, const cv::Matx33d &R, const cv::Vec3d &t, const cv::Matx33d &K, const cv::Vec2d &x);
//Returns true, if the squared distance between a given image coordinate and a 3D point projected into the image plane (no distortion) is below a threshold (provide the squared threshold to compare against the squared distance)
bool POSELIB_API reproject3DTh2(const cv::Vec3d &Q, const cv::Matx33d &R, const cv::Vec3d &t, const cv::Matx33d &K, const cv::Vec2d &x, const double &th);

//Calculates the euler angles from a given rotation matrix.
void POSELIB_API getAnglesRotMat(cv::InputArray R, double & roll, double & pitch, double & yaw, bool useDegrees = true);
//Calculates the difference (roation angle) between two rotation quaternions and the distance between two 3D translation vectors.
void POSELIB_API getRTQuality(cv::Mat & R, cv::Mat & Rcalib, cv::Mat & T,
                  cv::Mat & Tcalib, double* rdiff, double* tdiff);
//Calculates the difference (roation angle) between two rotation quaternions and the distance between two 3D translation vectors.
void POSELIB_API getRTQuality(Eigen::Vector4d & R, Eigen::Vector4d & Rcalib, Eigen::Vector3d & T,
                  Eigen::Vector3d & Tcalib, double* rdiff, double* tdiff);
//Calculates the essential matrix from the rotation matrix R and the translation
cv::Mat POSELIB_API getEfromRT(const cv::Mat& R, const cv::Mat& t);
//Compute Essential matrix E from rotation matrix R and translation vector t
Eigen::Matrix3d POSELIB_API getEfromRT(const Eigen::Matrix3d &R, const Eigen::Vector3d &t);
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

//Return the skew-symmetric matrix from a translation vector
template <typename T>
Eigen::Matrix<T, 3, 3> POSELIB_API getSkewSymMatFromVec(const Eigen::Matrix<T, 3, 1> &t)
{
    Eigen::Matrix<T, 3, 3> tx = Eigen::Matrix<T, 3, 3>::Zero();
    tx(0, 1) = -1. * t.z();
    tx(0, 2) = t.y();
    tx(1, 0) = t.z();
    tx(1, 2) = -1. * t.x();
    tx(2, 0) = -1. * t.y();
    tx(2, 1) = t.x();
    return tx;
}

//Computes the Lie-Logarithm for a rotation matrix R
Eigen::Vector3cd POSELIB_API compute_Lie_log(const Eigen::Matrix3d &R);
//Calculate a mean rotation matrix from multiple rotation matrices
cv::Mat POSELIB_API getMeanRotation(const std::vector<cv::Mat> &Rs);
//Calculate a mean translation vector from multiple translation vectors
cv::Mat POSELIB_API getMeanTranslation(const std::vector<cv::Mat> &ts);
//Calculate the camera center from rotation matrix R and translation vector t
cv::Mat POSELIB_API getCamCenterFromRt(const cv::Mat &R, const cv::Mat &t);

//Normalizes the image coordinates and transfers the image coordinates into cameracoordinates, respectively.
void POSELIB_API ImgToCamCoordTrans(std::vector<cv::Point2f>& points, cv::Mat K);
//Transfers coordinates from the camera coordinate system into the the image coordinate system.
void POSELIB_API CamToImgCoordTrans(std::vector<cv::Point2f>& points, cv::Mat K);
//Transfers coordinates from the camera coordinate system into the the image coordinate system.
void POSELIB_API CamToImgCoordTrans(cv::Mat& points, cv::Mat K);
//This function removes the lens distortion for single corresponding points in the camera coordinate system (K^-1 * [u v 1]^T)
bool POSELIB_API LensDist_Oulu(const cv::Mat &distorted, cv::Mat &corrected, const cv::Mat &dist, cv::InputArray R_tau = cv::noArray(), int iters = 10);
//This function removes the lens distortion for single corresponding points in the camera coordinate system (K^-1 * [u v 1]^T)
bool POSELIB_API LensDist_Oulu(const cv::Point2f &distorted, cv::Point2f &corrected, const cv::Mat &dist, cv::InputArray R_tau = cv::noArray(), int iters = 10);
//This function removes the lens distortion for single corresponding points in the camera coordinate system (K^-1 * [u v 1]^T)
bool POSELIB_API LensDist_Oulu(const cv::Point2d &distorted, cv::Point2d &corrected, const cv::Mat &dist, cv::InputArray R_tau = cv::noArray(), int iters = 10);
//Get the distortion (rotation) matrix for the tilted sensor model distortion with parameters tau_x and tau_y
cv::Mat POSELIB_API get_Rtau_distortionMat(const double &tx, const double &ty);
//Distorts a single homogeneous point provided in the image coordinate system ([u v 1]^T). Returns distorted 2D point in camera coordinate system: 1/z * [x y]
cv::Mat POSELIB_API distort(const cv::Mat &undistorted, cv::Mat &dist, cv::Mat &K_inv);
//Distort a single 2D point in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y]
cv::Mat POSELIB_API distort(const cv::Mat &undistorted, cv::Mat &dist);
//Distort a single 2D point in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y]
cv::Point2d POSELIB_API distort(const cv::Point2d &undistorted, cv::Mat &dist);
//Convert a single point in the image coordinate system ([u v]^T) into camera coordinate system (K^-1 * [u v 1]^T) and optionally undistort it
bool POSELIB_API toCamCoordsUndistort(const cv::Point2d &p_img, const cv::Mat Kinv, const bool undistort, const cv::Mat distortion, cv::Point2d &p_out);
//This function removes the lens distortion for multiple corresponding points in the camera coordinate system (K^-1 * [u v 1]^T)
bool POSELIB_API Remove_LensDist(std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, const cv::Mat& dist1, const cv::Mat& dist2);
//This function removes the lens distortion for multiple corresponding points. Input must be points in the camera coordinate system [x y z]^T = (K^-1 * [u v 1]^T) with z-component normalized to 1 -> 1/z * [x y].
bool POSELIB_API Remove_LensDist2(std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, const cv::Mat &dist1, const cv::Mat &dist2, cv::OutputArray mask = cv::noArray());
//Calculates the difference (roation angle) between two rotation quaternions and the distance between two 3D translation vectors
void POSELIB_API compareRTs(const cv::Mat& R1, const cv::Mat& R2, cv::Mat t1, cv::Mat t2, double *rdiff, double *tdiff, bool printDiff = false);
//Get R & t from Essential matrix E decomposition which is most similar to given R & t
void POSELIB_API getNearestRTfromEDecomposition(const cv::Mat &E, const cv::Mat &R_ref, const cv::Mat &t_ref, cv::Mat &R_out, cv::Mat &t_out);
//Calculation of the rectifying matrices based on the extrinsic and intrinsic camera parameters.
int POSELIB_API getRectificationParameters(cv::InputArray R,
                              cv::InputArray t,
                              cv::InputArray K1,
                              cv::InputArray K2,
                              cv::InputArray distcoeffs1,
                              cv::InputArray distcoeffs2,
                              const cv::Size& imageSize,
                              cv::OutputArray Rect1,
                              cv::OutputArray Rect2,
                              cv::OutputArray K1new,
                              cv::OutputArray K2new,
                              double alpha = -1,
                              bool globRectFunct = true,
                              const cv::Size& newImgSize = cv::Size(),
                              cv::Rect *roi1 = nullptr,
                              cv::Rect *roi2 = nullptr,
                              cv::OutputArray P1new = cv::noArray(),
                              cv::OutputArray P2new = cv::noArray());
//Estimates the optimal scale for the focal length of the camera.
double POSELIB_API estimateOptimalFocalScale(double alpha, cv::Mat K1, cv::Mat K2, cv::Mat R1, cv::Mat R2, cv::Mat P1, cv::Mat P2,
                                 cv::Mat dist1, cv::Mat dist2, const cv::Size& imageSize, cv::Size newImgSize);
//Estimates the vergence (shift of starting point) for correspondence search in the stereo engine.
int POSELIB_API estimateVergence(const cv::Mat& R, const cv::Mat& RR1, const cv::Mat& RR2, const cv::Mat& PR1, const cv::Mat& PR2);
//This function shows the rectified images.
int POSELIB_API ShowRectifiedImages(cv::InputArray img1,
                                    cv::InputArray img2,
                                    cv::InputArray mapX1,
                                    cv::InputArray mapY1,
                                    cv::InputArray mapX2,
                                    cv::InputArray mapY2,
                                    cv::InputArray t,
                                    const std::string& path,
                                    const std::string& imgName1 = "",
                                    const std::string& imgName2 = "",
                                    bool showResult = true,
                                    cv::Size newImgSize = cv::Size());
// This function returns the rectified images
int POSELIB_API GetRectifiedImages(cv::InputArray img1,
                                   cv::InputArray img2,
                                   cv::InputArray mapX1,
                                   cv::InputArray mapY1,
                                   cv::InputArray mapX2,
                                   cv::InputArray mapY2,
                                   cv::InputArray t,
                                   cv::OutputArray outImg1,
                                   cv::OutputArray outImg2,
                                   cv::Size newImgSize = cv::Size());
//This function estimates an initial delta value for the SPRT test used within USAC.
double estimateSprtDeltaInit(const std::vector<cv::DMatch> &matches,
                             const std::vector<cv::KeyPoint> &kp1,
                             const std::vector<cv::KeyPoint> &kp2,
                             const double &th,
                             const cv::Size &imgSize);
//This function estimates an initial epsilon value for the SPRT test used within USAC.
double estimateSprtEpsilonInit(const std::vector<cv::DMatch> &matches, const unsigned int &nrMatchesVfcFiltered);
//This function generates an index of the matches with the lowest matching costs first.
void getSortedMatchIdx(std::vector<cv::DMatch> matches, std::vector<unsigned int> & sortedMatchIdx);
//Checks if a 3x3 matrix is a rotation matrix
bool POSELIB_API isMatRoationMat(const cv::Mat& R);
//Checks if a 3x3 matrix is a rotation matrix
bool POSELIB_API isMatRoationMat(Eigen::Matrix3d R);
//Calculates the Sampson L2 error for 1 correspondence
double POSELIB_API getSampsonL2Error(cv::InputArray E, cv::InputArray x1, cv::InputArray x2);
//Calculates the Sampson L2 error for 1 correspondence
double POSELIB_API getSampsonL2Error(Eigen::Matrix3d E, const Eigen::Vector3d& x1, Eigen::Vector3d x2);
//Checks for a given vector of error values if they are inliers or not in respect to threshold th.
size_t POSELIB_API getInlierMask(const std::vector<double> &error, const double &th, cv::Mat & mask);
//Calculates the angle between two vectors
double POSELIB_API getAnglesBetwVectors(cv::Mat v1, cv::Mat v2, bool degree = true);

bool POSELIB_API searchAlternativeImgComb(const int &c1, 
                                          const int &c2, 
                                          const std::unordered_map<int, std::unordered_set<int>> &av_img_combs, 
                                          std::vector<std::vector<std::pair<int, int>>> &track, 
                                          const size_t &track_idx);
bool POSELIB_API getAvailablePairCombinationsForMissing(const std::unordered_set<std::pair<int, int>, pair_hash, pair_EqualTo> &missed_img_combs, 
                                                        const std::unordered_set<std::pair<int, int>, pair_hash, pair_EqualTo> &available_img_combs, 
                                                        std::unordered_map<std::pair<int, int>, std::vector<std::vector<std::pair<int, int>>>, pair_hash, pair_EqualTo> &missing_restore_sequences);
}
