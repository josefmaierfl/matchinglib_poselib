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
 FILE: pose_helper.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: May 2016

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides helper functions for the estimation and optimization of poses between
              two camera views (images).
**********************************************************************************************************/

#include "poselib/pose_helper.h"

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <chrono>
#include <thread>
#include <map>

#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Eigenvalues>

using namespace cv;
using namespace std;

namespace poselib
{

/* --------------------------- Defines --------------------------- */



/* --------------------- Function prototypes --------------------- */

//This function undistorts an image point
bool LensDist_Oulu(const cv::Point2f& distorted, cv::Point2f& corrected, cv::Mat dist, int iters = 10);
//Calculation of the rectifying matrices
int rectifyFusiello(cv::InputArray K1, cv::InputArray K2, cv::InputArray R, cv::InputArray t,
                    cv::InputArray distcoeffs1, cv::InputArray distcoeffs2, const cv::Size& imageSize,
                    cv::OutputArray Rect1, cv::OutputArray Rect2, cv::OutputArray K12new,
                    double alpha = -1, cv::Size newImgSize=cv::Size(), cv::Rect *roi1 = nullptr, cv::OutputArray P1new = cv::noArray(),
                    cv::OutputArray P2new = cv::noArray());
//OpenCV interface function for cvStereoRectify. This code was copied from the OpenCV without changes.
void stereoRectify2( InputArray cameraMatrix1, InputArray distCoeffs1,
                               InputArray cameraMatrix2, InputArray distCoeffs2,
                               Size imageSize, InputArray _Rmat, InputArray T,
                               OutputArray _Rmat1, OutputArray _Rmat2,
                               OutputArray _Pmat1, OutputArray _Pmat2,
                               OutputArray _Qmat, int flags=CALIB_ZERO_DISPARITY,
                               double alpha=-1, Size newImageSize=Size(),
                               CV_OUT Rect* validPixROI1=nullptr, CV_OUT Rect* validPixROI2=nullptr );
//Slightly changed version of the OpenCV rectification function cvStereoRectify.
void cvStereoRectify2( const cv::Mat* _cameraMatrix1, const cv::Mat* _cameraMatrix2,
                      const cv::Mat* _distCoeffs1, const cv::Mat* _distCoeffs2,
                      const cv::Size& imageSize, const cv::Mat* matR, const cv::Mat* matT,
                      cv::Mat* _R1, cv::Mat* _R2, cv::Mat* _P1, cv::Mat* _P2,
                      cv::Mat* matQ, int flags, double alpha, cv::Size newImgSize,
                      cv::Rect* roi1, cv::Rect* roi2 );
//Slightly changed version of the OpenCV undistortion function cvUndistortPoints.
void cvUndistortPoints2(const cv::Mat& src_, cv::Mat& dst_, const cv::Mat& cameraMatrix_,
                        const cv::Mat* distCoeffs_,
                        const cv::Mat* matR, const cv::Mat* matP, cv::OutputArray mask );
// Estimates the inner rectangle of a distorted image containg only valid/available image information and an outer rectangle countaing all image information
void icvGetRectanglesV0( const cv::Mat* cameraMatrix, const cv::Mat* distCoeffs,
                 const cv::Mat* R, const cv::Mat* newCameraMatrix, const cv::Size& imgSize,
                 cv::Rect_<float>& inner, cv::Rect_<float>& outer );
// Helping function - Takes some actions on a mouse move
void on_mouse_move(int event, int x, int y, int flags, void* param);

/* --------------------- Functions --------------------- */

/* Calculates the Sampson L1-distance for a point correspondence and returns the invers of the
 * denominator (in denom1) and the numerator of the Sampson L1-distance. To calculate the
 * Sampson distance, simply multiply these two. For the Sampson error, multiply and square them.
 *
 * Mat x1							Input  -> Image projection of the lweft image
 * Mat x2							Input  -> Image projection of the right image
 * Mat E							Input  -> Essential matrix
 * double & denom1					Output -> invers of the denominator of the Sampson distance
 * double & num						Output -> numerator of the Sampson distance
 *
 * Return value:					none
 */
void SampsonL1(const cv::Mat &x1, const cv::Mat &x2, const cv::Mat &E, double & denom1, double & num)
{
    Mat X1, X2;
    if(x1.rows > x1.cols)
    {
        X1 = (Mat_<double>(3, 1) << x1.at<double>(0,0), x1.at<double>(1,0), 1.0);
        X2 = (Mat_<double>(3, 1) << x2.at<double>(0,0), x2.at<double>(1,0), 1.0);
    }
    else
    {
        X1 = (Mat_<double>(3, 1) << x1.at<double>(0,0), x1.at<double>(0,1), 1.0);
        X2 = (Mat_<double>(3, 1) << x2.at<double>(0,0), x2.at<double>(0,1), 1.0);
    }
    Mat xpE = X2.t() * E;
    xpE = xpE.t();
    num = xpE.dot(X1);
    //num = X2.dot(E * X1);
    Mat Ex1 = E * X1;
    //Ex1 /= Ex1.at<double>(2);
    //xpE /= xpE.at<double>(2);
    //Mat Etx2 = E.t() * X2;
    double a = Ex1.at<double>(0,0) * Ex1.at<double>(0,0);
    double b = Ex1.at<double>(1,0) * Ex1.at<double>(1,0);
    double c = xpE.at<double>(0,0) * xpE.at<double>(0,0);
    double d = xpE.at<double>(1,0) * xpE.at<double>(1,0);

    denom1 = 1 / (std::sqrt(a + b + c + d) + 1e-8);
}

/* Calculates the closest essential matrix by enforcing the singularity constraint (third
 * singular value is zero).
 *
 * Mat x1							Input & Output  -> Essential matrix
 *
 * Return value:					0:		  Everything ok
 *									-1:		  E is no essential matrix
 */
int getClosestE(Eigen::Matrix3d & E)
{
    //double avgSingVal;
    Eigen::JacobiSVD<Eigen::Matrix3d> svdE(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    if(!nearZero(svdE.singularValues()[2])){
        return -1; // E is no essential matrix
    }
    if((svdE.singularValues()[0]/svdE.singularValues()[1] > 1.5) ||
            (svdE.singularValues()[0]/svdE.singularValues()[1] < 0.66)){
        return -1; // E is no essential matrix
    }

    Eigen::Matrix3d D;
    D.setZero();
    /*avgSingVal = svdE.singularValues().segment(0,2).sum()/2;
    D(0,0) = D(1,1) = avgSingVal;*/
    D(0,0) = svdE.singularValues()[0];
    D(1,1) = svdE.singularValues()[1];

    E = svdE.matrixU() * D * svdE.matrixV().transpose();

    return 0;
}


/* Validate the Essential/Fundamental matrix with the oriented epipolar constraint (this should
 * be extensively tested if it makes sence) and optionally checks the correctness of the singular
 * values of the essential matrix.
 *
 * Mat p1							Input  -> Image projections (n rows) of the left image
 * Mat p2							Input  -> Corresponding image projections of the right image
 * Eigen::Matrix3d E				Input  -> Essential matrix
 * bool EfullCheck					Input  -> If true, the correctness of the singular values of
 *											  the essential matrix is checked
 * InputOutputArray _mask			I/O    -> If provided, a mask marking invalid correspondences
 *											  is returned
 * bool tryOrientedEpipolar			Input  -> Optional input [DEFAULT = false] to specify if a essential matrix
 *											  should be evaluated by the oriented epipolar constraint. Maybe this
 *											  is only possible for a fundamental matrix?
 *
 * Return value:					true:		  Essential/Fundamental matrix is valid
 *									false:		  Essential/Fundamental matrix is invalid
 */
bool validateEssential(const cv::Mat &p1,
                       const cv::Mat &p2,
                       Eigen::Matrix3d E,
                       bool EfullCheck,
                       cv::InputOutputArray _mask,
                       bool tryOrientedEpipolar)
{
    //Eigen::Matrix3d E;
    Eigen::Vector3d e2, x1, x2;

    Mat _p1, _p2;
    if(p1.channels() == 2)
    {
        if(p1.cols > p1.rows)
        {
            _p1 = p1.clone();
            _p2 = p2.clone();
            _p1 = _p1.t();
            _p2 = _p2.t();
            _p1 = _p1.reshape(1);
            _p2 = _p2.reshape(1);
        }
        else
        {
            _p1 = p1.reshape(1);
            _p2 = p2.reshape(1);
        }
    }
    else
    {
        if(p1.cols > p1.rows)
        {
            _p1 = p1.clone();
            _p2 = p2.clone();
            _p1 = _p1.t();
            _p2 = _p2.t();
        }
        else
        {
            _p1 = p1;
            _p2 = p2;
        }
    }

    int cnt = 0, n = _p1.rows;
    float badPtsRatio;
    Mat mask = Mat::ones(1,n,CV_8UC1);

    //cv2eigen(Ecv, E);

//tryagain:
    bool exitfor = true;
    while (exitfor)
    {
        if (EfullCheck)
        {
            Eigen::Matrix3d V;
            Eigen::JacobiSVD<Eigen::Matrix3d > svdE(E.transpose(), Eigen::ComputeFullV);

            if (svdE.singularValues()(0) / svdE.singularValues()(1) > 1.2)
                return false;
            if (!nearZero(0.01*svdE.singularValues()(2) / svdE.singularValues()(1)))
                return false;

            V = svdE.matrixV();
            e2 = V.col(2);
        }
        else
        {
            Eigen::MatrixXd ker = E.transpose().fullPivLu().kernel();
            if (ker.cols() != 1)
                return false;
            e2 = ker.col(0);
        }

        //Does this improve something or only need time?
        exitfor = false;
        if (tryOrientedEpipolar)
        {
            for (int i = 0; i < n; i++)
            {
                Eigen::Vector3d e2_line1, e2_line2;
                x1 << _p1.at<double>(i, 0),
                    _p1.at<double>(i, 1),
                    1.0;
                x2 << _p2.at<double>(i, 0),
                    _p2.at<double>(i, 1),
                    1.0;
                e2_line1 = e2.cross(x2);
                e2_line2 = E * x1;
                for (int j = 0; j < 3; j++)
                {
                    if (nearZero(0.1*e2_line1(j)) || nearZero(0.1*e2_line2(j)))
                        continue;
                    if (e2_line1(j)*e2_line2(j) < 0)
                    {
                        if (cnt < 3)
                            cnt++;
                        else if ((cnt == 3) && (i == cnt))
                        {
                            E *= -1.0;
                            for (int k = 0; k < cnt; k++)
                                mask.at<bool>(k) = true;
                            cnt++;
                            //goto tryagain;
                            exitfor = true;
                            break;
                        }
                        mask.at<bool>(i) = false;
                        break;
                    }
                }
                if (exitfor)
                    break;
            }
        }
    }
    badPtsRatio = 1.0f - (float)cv::countNonZero(mask) / (float)n;
    if (badPtsRatio > 0.4f)
        return false;

    if(_mask.needed())
    {
        Mat mask1 = _mask.getMat();
        if(mask1.empty())
        {
            _mask.create(1, n, CV_8UC1, -1, true);
            mask1 = _mask.getMat();
            mask1 = Mat::ones(1, n, CV_8UC1);
        }
        bitwise_and(mask, mask1, mask1);
    }

    return true;
}

/* Checks, if determinants, etc. are too close to 0
 *
 * double d							Input  -> The value which should be checked
 *
 * Return value:					TRUE:  Value is too close to zero
 *									FALSE: Value is ok.
 */
//inline bool nearZero(double d)
//{
//    //Decide if determinants, etc. are too close to 0 to bother with
//    const double EPSILON = 1e-3;
//    return (d<EPSILON) && (d>-EPSILON);
//}


/* Calculates statistical parameters for the given values in the vector. The following parameters
 * are calculated: median, arithmetic mean value, standard deviation and median absolute deviation (MAD).
 *
 * vector<double> vals		Input  -> Input vector from which the statistical parameters should be calculated
 * statVals* stats			Output -> Structure holding the statistical parameters
 * bool rejQuartiles		Input  -> If true [Default=false], the lower and upper quartiles are rejected before calculating
 *									  the parameters
 * bool roundStd			Input  -> If true [Default], an standard deviation below 1e-6 is set to 0
 *
 * Return value:		 none
 */
void getStatsfromVec(const std::vector<double> &vals, statVals *stats, bool rejQuartiles, bool roundStd)
{
    if(vals.empty())
    {
        stats->arithErr = 0;
        stats->arithStd = 0;
        stats->medErr = 0;
        stats->medStd = 0;
        return;
    }
    int n = (int)vals.size();
    int qrt_si = (int)floor(0.25 * (double)n);
    std::vector<double> vals_tmp(vals);

    std::sort(vals_tmp.begin(),vals_tmp.end(),[](double const & first, double const & second){
        return first < second;});

    if(n % 2)
        stats->medErr = vals_tmp[(n-1)/2];
    else
        stats->medErr = (vals_tmp[n/2] + vals_tmp[n/2-1]) / 2.0;

    stats->arithErr = 0.0;
    double err2sum = 0.0;
    //double medstdsum = 0.0;
    double hlp;
    std::vector<double> madVec;
    for(int i = rejQuartiles ? qrt_si:0; i < (rejQuartiles ? (n-qrt_si):n); i++)
    {
        stats->arithErr += vals_tmp[i];
        err2sum += vals_tmp[i] * vals_tmp[i];

        madVec.push_back(std::abs(vals_tmp[i] - stats->medErr));

        //medstdsum += hlp * hlp;
    }
    if(rejQuartiles)
        n -= 2 * qrt_si;
    stats->arithErr /= (double)n;

    std::sort(madVec.begin(),madVec.end(),[](double const & first, double const & second){
        return first < second;});

    if(n % 2)
        stats->medStd = 1.4826 * madVec[(n-1)/2]; //1.4826 corresponds to a scale factor for transform the MAD to approximately
                                                    //the standard deviation for a standard normal distribution, see https://en.wikipedia.org/wiki/Median_absolute_deviation
    else
        stats->medStd = 1.4826 * (madVec[n/2] + madVec[n/2-1]) / 2.0;

    hlp = err2sum - (double)n * (stats->arithErr) * (stats->arithErr);

    if(roundStd && std::abs(hlp) < 1e-6)
        stats->arithStd = 0.0;
    else
        stats->arithStd = std::sqrt(hlp/((double)n - 1.0));
}

/* Extracts the 3D translation vector from the translation essential matrix. It is possible that the
 * resulting vector points in the opposite direction.
 *
 * Mat Et								Input  -> The translation essential matrix
 *
 * Return value:						The 3D translation vector (+-)
 */
cv::Mat getTfromTransEssential(cv::Mat Et)
{
    CV_Assert(!Et.empty() && (Et.type() == CV_64FC1) && nearZero(Et.at<double>(0,1) / Et.at<double>(1,0) + 1.0)
                && nearZero(Et.at<double>(0,2) / Et.at<double>(2,0) + 1.0) && nearZero(Et.at<double>(1,2) / Et.at<double>(2,1) + 1.0));

    Mat t = (Mat_<double>(3, 1) << Et.at<double>(1,2), Et.at<double>(2,0), Et.at<double>(0,1));
    double t_norm = normFromVec(t);
    if(std::abs(t_norm - 1.0) > 1e-3)
        t /= t_norm;

    return t;
}

/* Calculates the vector norm.
 *
 * cv::Mat vec						Input  -> Vector for which the norm should be calculated
 *											  (size must be 1 x n or n x 1)
 *
 * Return value:					Vector norm
 */
double normFromVec(const cv::Mat& vec)
{
    int n;
    double norm = 0;
    Mat tmp;
    if(vec.type() != CV_64FC1)
        vec.convertTo(tmp,CV_64FC1);
    else
        tmp = vec;

    n = tmp.rows > tmp.cols ? tmp.rows : tmp.cols;

    for(int i = 0; i < n; i++)
        norm += tmp.at<double>(i) * tmp.at<double>(i);

    return std::sqrt(norm);
}

/* Calculates the vector norm.
 *
 * vector<double> vec				Input  -> Vector for which the norm should be calculated
 *
 * Return value:					Vector norm
 */
double normFromVec(std::vector<double> vec)
{
    size_t n = vec.size();
    double norm = 0;

    for(size_t i = 0; i < n; i++)
        norm += vec[i] * vec[i];

    return std::sqrt(norm);
}

/* Calculates the statistics on the reprojection errors for the given correspondences and a given
 * essential matrix. If a (normalized) fundamental matrix is used, EisF and takeImageCoords must be true
 * and the correspondences must be normalized. If "takeImageCoords" is true and EisF=false [Default], the
 * correspondences which are in the camera (or world) coordinate system are transferred into the image
 * coordinate system (Thus, K1 and K2 must be provided). The following parameters are calculated from
 * the correspondences (if qp != NULL): median, arithmetic mean value, standard deviation and median
 * absolute deviation (MAD) which is scaled to match the standard deviation of a standard normal
 * distribution.
 *
 * Mat Essential			Input  -> Essential matrix
 * bool takeImageCoords		Input  -> If true, the image coordinate system is used instead of the camera
 *									  coordinate system.
 * qualityParm* qp			Output -> If this pointer is not NULL, the result is stored here
 * vector<double> *repErr	Output -> If this pointer is not NULL, only the error-vector is returned and
 *									  no quality parameters are calculated
 * InputArray p1			Input  -> Image projections of the first image (n rows x 2 cols)
 * InputArray p2			Input  -> Image projections of the second image (n rows x 2 cols)
 * InputArray K1			Input  -> Camera matrix of the first camera (must be provided if an essential matrix
 *									  is provided and an error in pixel units should be calculated
 *									  (takeImageCoords=true, EisF=false))
 * InputArray K2			Input  -> Camera matrix of the second camera (must be provided if an essential matrix
 *									  is provided and an error in pixel units should be calculated
 *									  (takeImageCoords=true, EisF=false))
 * bool EisF				Input  -> If true [Default=false], a fundamental matrix is given instead of an
 *									  essential matrix (takeImageCoords must be set to true)
 *
 * Return value:		 none
 */
void getReprojErrors(const cv::Mat& Essential, cv::InputArray p1, cv::InputArray p2, bool takeImageCoords, statVals* qp, std::vector<double> *repErr, cv::InputArray K1, cv::InputArray K2, bool EisF)
{
    CV_Assert(!p1.empty() && !p2.empty());
    CV_Assert(!(takeImageCoords && !K1.empty() && !K2.empty()) || EisF);
    CV_Assert(!((qp == nullptr) && (repErr == nullptr)));

    if(EisF && !takeImageCoords)
        takeImageCoords = true;

    std::vector<double> error;
    int n;
    n = p1.getMat().rows;

    Mat x1, x2, FE, x1_tmp, x2_tmp, K1_, K2_;

    if(!K1.empty() && !K2.empty())
    {
        K1_ = K1.getMat();
        K2_ = K2.getMat();
    }

    if(takeImageCoords)
    {
        if(EisF)
        {
            x1 = p1.getMat();
            x2 = p2.getMat();
            FE = Essential;
        }
        else
        {
            x1 = Mat::ones(3,n,CV_64FC1);
            x1_tmp = p1.getMat().t();
            x1_tmp.copyTo(x1.rowRange(0,2));
            x1 = K1_*x1;
            x1.row(0) /= x1.row(2);
            x1.row(1) /= x1.row(2);
            x1 = x1.rowRange(0,2).t();

            x2 = Mat::ones(3,n,CV_64FC1);
            x2_tmp = p2.getMat().t();
            x2_tmp.copyTo(x2.rowRange(0,2));
            x2 = K2_*x2;
            x2.row(0) /= x2.row(2);
            x2.row(1) /= x2.row(2);
            x2 = x2.rowRange(0,2).t();

            FE = K2_.inv().t()*Essential*K1_.inv();
        }
    }
    else
    {
        x1 = p1.getMat();
        x2 = p2.getMat();

        FE = Essential;
    }

    if(repErr != nullptr)
        computeReprojError2(x1, x2, FE, *repErr);

    if(qp != nullptr)
    {
        if(repErr == nullptr)
        {
            computeReprojError2(x1, x2, FE, error);
            getStatsfromVec(error, qp);
        }
        else
        {
            getStatsfromVec(*repErr, qp);
        }
    }
}

/* Computes the Sampson distance (first-order geometric error) for the provided point correspondences.
 * If the fundamental matrix is used, the homogeneous points have to be in (normalized) camera
 * coordinate system units. If the essential matrix is used for computing the error, the homogeneous
 * points have to be in world coordinate system units (K^-1 * x).
 *
 * Mat X1					Input  -> Points in the left (first) camera of the form 2 rows x n cols
 * Mat X2					Input  -> Points in the right (second) camera of the form 2 rows x n cols
 * Mat E					Input  -> Essential matrix or fundamental matrix -> depends on coordinate
 *									  system
 * vector<double> error		Output -> Vector of errors corresponding to the point correspondences if
 *									  the pointer to error1 equals NULL
 * double *error1			Output -> If this pointer is not NULL and X1 & X2 hold only 1 correspondence,
 *									  then the error is returned here and NOT in the vector
 *
 * Return value:		none
 */
void computeReprojError1(cv::Mat X1, cv::Mat X2, const cv::Mat& E, std::vector<double> & error, double *error1)
{
    CV_Assert((X1.cols >= X1.rows) || ((X1.cols == 1) && (X1.rows == 2)));
    int n = X1.cols;
    Mat Et = E.t();

    for (int i = 0; i < n; i++)
    {
        Mat x1 = (Mat_<double>(3, 1) << X1.at<double>(0, i), X1.at<double>(1, i), 1.0);
        Mat x2 = (Mat_<double>(3, 1) << X2.at<double>(0, i), X2.at<double>(1, i), 1.0);
        //Mat x1 = X1.col(i);
        //Mat x2 = X2.col(i);
        double x2tEx1 = x2.dot(E * x1);
        Mat Ex1 = E * x1;
        Mat Etx2 = Et * x2;
        double a = Ex1.at<double>(0) * Ex1.at<double>(0);
        double b = Ex1.at<double>(1) * Ex1.at<double>(1);
        double c = Etx2.at<double>(0) * Etx2.at<double>(0);
        double d = Etx2.at<double>(1) * Etx2.at<double>(1);

        if(error1 && (n == 1))
            *error1 = x2tEx1 * x2tEx1 / (a + b + c + d);
        else
            error.push_back(x2tEx1 * x2tEx1 / (a + b + c + d));
    }
}

/* Computes the Sampson distance (first-order geometric error) for the provided point correspondences.
 * If the fundamental matrix is used, the homogeneous points have to be in (normalized) camera
 * coordinate system units. If the essential matrix is used for computing the error, the homogeneous
 * points have to be in world coordinate system units (K^-1 * x).
 *
 * Mat X1					Input  -> Points in the left (first) camera of the form n rows x 2 cols
 * Mat X2					Input  -> Points in the right (second) camera of the form n rows x 2 cols
 * Mat E					Input  -> Essential matrix or fundamental matrix -> depends on coordinate
 *									  system
 * vector<double> error		Output -> Vector of errors corresponding to the point correspondences if
 *									  the pointer to error1 equals NULL
 * double *error1			Output -> If this pointer is not NULL and X1 & X2 hold only 1 correspondence,
 *									  then the error is returned here and NOT in the vector
 *
 * Return value:		none
 */
void computeReprojError2(cv::Mat X1, cv::Mat X2, const cv::Mat& E, std::vector<double> & error, double *error1)
{
    CV_Assert((X1.cols <= X1.rows) || ((X1.cols == 2) && (X1.rows == 1)));
    int n = X1.rows;
    Mat Et = E.t();

    for (int i = 0; i < n; i++)
    {
        Mat x1 = (Mat_<double>(3, 1) << X1.at<double>(i, 0), X1.at<double>(i, 1), 1.0);
        Mat x2 = (Mat_<double>(3, 1) << X2.at<double>(i, 0), X2.at<double>(i, 1), 1.0);
        //Mat x1 = X1.col(i);
        //Mat x2 = X2.col(i);
        double x2tEx1 = x2.dot(E * x1);
        Mat Ex1 = E * x1;
        Mat Etx2 = Et * x2;
        double a = Ex1.at<double>(0) * Ex1.at<double>(0);
        double b = Ex1.at<double>(1) * Ex1.at<double>(1);
        double c = Etx2.at<double>(0) * Etx2.at<double>(0);
        double d = Etx2.at<double>(1) * Etx2.at<double>(1);

        if(error1 && (n == 1))
            *error1 = x2tEx1 * x2tEx1 / (a + b + c + d);
        else
            error.push_back(x2tEx1 * x2tEx1 / (a + b + c + d));
    }
}

/* Computes the Sampson distance (first-order geometric error) for the provided point correspondence.
 *
 * Point2f p1				Input  -> Point in the left (first) camera
 * Point2f p2				Input  -> Point in the right (second) camera
 * Mat F					Input  -> Essential matrix or fundamental matrix -> depends on coordinate
 *									  system
 *
 * Return value:		    (Squared) sampson error
 */
double calculateSampsonError(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &F)
{
    cv::Mat x1 = (cv::Mat_<double>(3, 1) << static_cast<double>(p1.x), static_cast<double>(p1.y), 1.0);
    cv::Mat x2 = (cv::Mat_<double>(3, 1) << static_cast<double>(p2.x), static_cast<double>(p2.y), 1.0);
    double x2tEx1 = x2.dot(F * x1);
    cv::Mat Ex1 = F * x1;
    cv::Mat Etx2 = F.t() * x2;
    double a = Ex1.at<double>(0) * Ex1.at<double>(0);
    double b = Ex1.at<double>(1) * Ex1.at<double>(1);
    double c = Etx2.at<double>(0) * Etx2.at<double>(0);
    double d = Etx2.at<double>(1) * Etx2.at<double>(1);

    return x2tEx1 * x2tEx1 / (a + b + c + d);
}

/* Computes the squared distance to the epipolar line in the second image for the provided point correspondence
 *
 * Point2f p1				Input  -> Point in the left (first) camera
 * Point2f p2				Input  -> Point in the right (second) camera
 * Mat F					Input  -> Essential matrix or fundamental matrix -> depends on coordinate
 *									  system
 *
 * Return value:		    Squared distance to epipolar line in the second image
 */
double calculateEpipolarLineDistanceImg2Squared(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &F)
{
    cv::Mat x1 = (cv::Mat_<double>(3, 1) << static_cast<double>(p1.x), static_cast<double>(p1.y), 1.0);
    cv::Mat l2 = F * x1;

    const double a = l2.at<double>(0) * static_cast<double>(p2.x) + l2.at<double>(1) * static_cast<double>(p2.y) + l2.at<double>(2);
    const double b = l2.at<double>(0) * l2.at<double>(0) + l2.at<double>(1) * l2.at<double>(1);

    return a * a / b;
}

/* Computes the squared distance to the epipolar line in the second image for the provided point correspondence
 *
 * Mat p1				    Input  -> Point in the left (first) camera
 * Mat p2				    Input  -> Point in the right (second) camera
 * Mat F					Input  -> Essential matrix or fundamental matrix -> depends on coordinate
 *									  system
 *
 * Return value:		    Squared distance to epipolar line in the second image
 */
double calculateEpipolarLineDistanceImg2Squared(const cv::Mat &p1, const cv::Mat &p2, const cv::Mat &F)
{
    cv::Mat x1 = (cv::Mat_<double>(3, 1) << p1.at<double>(0), p1.at<double>(1), 1.0);
    cv::Mat l2 = F * x1;

    const double a = l2.at<double>(0) * p2.at<double>(0) + l2.at<double>(1) * p2.at<double>(1) + l2.at<double>(2);
    const double b = l2.at<double>(0) * l2.at<double>(0) + l2.at<double>(1) * l2.at<double>(1);

    return a * a / b;
}

/* Computes the squared distance to the epipolar line in the first image for the provided point correspondence
 *
 * Point2f p1				Input  -> Point in the left (first) camera
 * Point2f p2				Input  -> Point in the right (second) camera
 * Mat F					Input  -> Essential matrix or fundamental matrix -> depends on coordinate
 *									  system
 *
 * Return value:		    Squared distance to epipolar line in the first image
 */
double calculateEpipolarLineDistanceImg1Squared(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &F)
{
    cv::Mat x2 = (cv::Mat_<double>(3, 1) << static_cast<double>(p2.x), static_cast<double>(p2.y), 1.0);
    cv::Mat l1 = F.t() * x2;

    const double a = l1.at<double>(0) * static_cast<double>(p1.x) + l1.at<double>(1) * static_cast<double>(p1.y) + l1.at<double>(2);
    const double b = l1.at<double>(0) * l1.at<double>(0) + l1.at<double>(1) * l1.at<double>(1);

    return a * a / b;
}

/* Computes the squared distance to the epipolar line in the first image for the provided point correspondence
 *
 * Mat p1				    Input  -> Point in the left (first) camera
 * Mat p2				    Input  -> Point in the right (second) camera
 * Mat F					Input  -> Essential matrix or fundamental matrix -> depends on coordinate
 *									  system
 *
 * Return value:		    Squared distance to epipolar line in the first image
 */
double calculateEpipolarLineDistanceImg1Squared(const cv::Mat &p1, const cv::Mat &p2, const cv::Mat &F)
{
    cv::Mat x2 = (cv::Mat_<double>(3, 1) << p2.at<double>(0), p2.at<double>(1), 1.0);
    cv::Mat l1 = F.t() * x2;

    const double a = l1.at<double>(0) * p1.at<double>(0) + l1.at<double>(1) * p1.at<double>(1) + l1.at<double>(2);
    const double b = l1.at<double>(0) * l1.at<double>(0) + l1.at<double>(1) * l1.at<double>(1);

    return a * a / b;
}    

/* Computes sqrt distances for every provided point correspondence of mean 1/2 * (e1 + e2) squared 
 * distances e1 and e2 to epipolar lines in first and second images
 *
 * vector<Point2f> points1		Input  -> Corresponding points in the left (first) camera
 * vector<Point2f> points2		Input  -> Corresponding points in the right (second) camera
 * Mat F					    Input  -> Essential matrix or fundamental matrix -> depends on coordinate
 *			    						  system
 * vector<double> errors	    Output -> sqrt mean (1st and 2nd camera) epipolar distances for every provided point correspondence
 *
 * Return value:		        none
 */
void calculateEpipolarLineDistanceImg12(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2, const cv::Mat &F, std::vector<double> &errors)
{
    const size_t si = points1.size();
    errors.clear();
    errors.reserve(si);
    for (size_t i = 0; i < si; ++i)
    {
        errors.emplace_back(sqrt(calculateEpipolarLineDistanceImg12SquaredMean(points1.at(i), points2.at(i), F)));
    }
}

/* Computes sqrt distances for every provided point correspondence of mean 1/2 * (e1 + e2) squared 
 * distances e1 and e2 to epipolar lines in first and second images
 *
 * Mat points1		            Input  -> Corresponding points in the left (first) camera. One coordinate per row
 * Mat points2		            Input  -> Corresponding points in the right (second) camera. One coordinate per row
 * Mat F					    Input  -> Essential matrix or fundamental matrix -> depends on coordinate
 *			    						  system
 * vector<double> errors	    Output -> sqrt mean (1st and 2nd camera) epipolar distances for every provided point correspondence
 *
 * Return value:		        none
 */
void calculateEpipolarLineDistanceImg12(const cv::Mat &points1, const cv::Mat &points2, const cv::Mat &F, std::vector<double> &errors)
{
    const int si = points1.rows;
    errors.clear();
    errors.reserve(si);
    for (int i = 0; i < si; ++i)
    {
        errors.emplace_back(sqrt(calculateEpipolarLineDistanceImg12SquaredMean(points1.row(i), points2.row(i), F)));
    }
}

/* Computes sqrt mean distance over all provided point correspondence of mean 1/2 * (e1 + e2) 
 * squared distances e1 and e2 to epipolar lines in first and second images
 *
 * vector<Point2f> points1		Input  -> Corresponding points in the left (first) camera
 * vector<Point2f> points2		Input  -> Corresponding points in the right (second) camera
 * Mat F					    Input  -> Essential matrix or fundamental matrix -> depends on coordinate
 *			    						  system
 *
 * Return value:		        sqrt mean epipolar distances over all provided point correspondences
 */
double calculateMeanEpipolarLineDistanceImg12(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2, const cv::Mat &F)
{
    const size_t si = points1.size();
    double error = 0;
    for (size_t i = 0; i < si; ++i)
    {
        error += calculateEpipolarLineDistanceImg12SquaredMean(points1.at(i), points2.at(i), F);
    }
    return sqrt(error / static_cast<double>(si));
}

/* Computes sqrt mean distance over all provided point correspondence of mean 1/2 * (e1 + e2) 
 * squared distances e1 and e2 to epipolar lines in first and second images
 *
 * Mat points1		            Input  -> Corresponding points in the left (first) camera. One coordinate per row
 * Mat points2		            Input  -> Corresponding points in the right (second) camera. One coordinate per row
 * Mat F					    Input  -> Essential matrix or fundamental matrix -> depends on coordinate
 *			    						  system
 *
 * Return value:		        sqrt mean epipolar distances over all provided point correspondences
 */
double calculateMeanEpipolarLineDistanceImg12(const cv::Mat &points1, const cv::Mat &points2, const cv::Mat &F)
{
    const int si = points1.rows;
    double error = 0;
    for (int i = 0; i < si; ++i)
    {
        error += calculateEpipolarLineDistanceImg12SquaredMean(points1.row(i), points2.row(i), F);
    }
    return sqrt(error / static_cast<double>(si));
}

/* Compute the epipolar error of a single point correspondence in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) utilizing the Essential matrix E
 *
 * Vector2d x1				    Input  -> Point in the left (first) camera
 * Vector2d x2				    Input  -> Point in the right (second) camera
 * Matrix3d E					Input  -> Essential matrix
 *
 * Return value:		        Epipolar error [x2 y2 1] * E * [x1 y1 1]^T
 */
double calculateEpipolarError(const Eigen::Vector2d &x1, const Eigen::Vector2d &x2, const Eigen::Matrix3d &E)
{
    Eigen::Vector3d x1h(x1.x(), x1.y(), 1.0);
    Eigen::Vector3d x2h(x2.x(), x2.y(), 1.0);
    return calculateEpipolarError(x1h, x2h, E);
}

/* Compute the epipolar error of a single  homogeneous point correspondence in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) utilizing the Essential matrix E
 *
 * Vector3d x1				    Input  -> Point in the left (first) camera
 * Vector3d x2				    Input  -> Point in the right (second) camera
 * Matrix3d E					Input  -> Essential matrix
 *
 * Return value:		        Epipolar error [x2 y2 1] * E * [x1 y1 1]^T
 */
double calculateEpipolarError(const Eigen::Vector3d &x1h, const Eigen::Vector3d &x2h, const Eigen::Matrix3d &E)
{
    Eigen::Vector3d l2 = E * x1h;
    return x2h.dot(l2);
}

/* Reject 3D points and image projections with high error values
 *
 * Mat Q				            I/O    -> 3D coordinates. 1 coordinate per row -> n rows
 * vector<Mat> map3D		        I/O    -> Corresponding 3D point mask (size of n bytes) for every camera: map3D[c].at<char>(i) for camera c and i'th 3D point in Q
 * vector<Mat> pts2D		        I/O    -> 2D image measurements for every camera ordered corresponding to Q. 
 *                                            E.g. Q = [q0 q1 q2 q3 q4 q5 ...]^T, map3D[0] = [0 0 1 1 0 1 ...] -> pts2D[0] = [x2 x3 x5]
 * vector<Mat> Rs		            Input  -> Rotation matrix for every camera
 * vector<Mat> ts		            Input  -> Translation vector for every camera
 * vector<Mat> Ks		            Input  -> Camera matrix for every camera
 * int n_cams		                Input  -> Number of cameras
 * double th		                Input  -> Reprojection error threshold in pixels
 * float minRemainRatio		        Input  -> Minimum ratio (0 ... 1) of remaining 3D points. If below this ratio and forceFiltering=false, filtering is aborted. Default=0.33
 * InputArray dists		            Input  -> Optional distortion coeffitients for every camera
 * InputArray cheirality_mask		Input  -> Optional inlier mask for Q
 * InputOutputArray Q2		        I/O    -> Optional 2nd array of 3D points corresponding to Q (e.g. from different estimation)
 * bool forceFiltering		        Input  -> Forces filtering of Q and pts2D even if the ratio of remaining 3D points is below minRemainRatio. 
 *                                            Still, more than 10 3D points must survive -> otherwise, abort.
 * vector<vector<Point2f>> *pts2Df	I/O    -> Optional second array of 2D image measurements (e.g. keypoint locations). Filtering is performed based on pts2D (same index)
 *
 * Return value:		            Success
 */
bool rejectHighErr3D(cv::Mat &Q, 
                     std::vector<cv::Mat> &map3D, 
                     std::vector<cv::Mat> &pts2D, 
                     const std::vector<cv::Mat> &Rs, 
                     const std::vector<cv::Mat> &ts, 
                     const std::vector<cv::Mat> &Ks, 
                     const int &n_cams, 
                     const double &th, 
                     const float minRemainRatio, 
                     cv::InputArray dists, 
                     cv::InputArray cheirality_mask, 
                     cv::InputOutputArray Q2, 
                     const bool forceFiltering, 
                     std::vector<std::vector<cv::Point2f>> *pts2Df)
{
    const int Q_nr = Q.rows;
    bool haveDists = false;
    vector<cv::Mat> dist_vec;
    if (!dists.empty())
    {
        dists.getMatVector(dist_vec);
        haveDists = true;
    }
    vector<cv::Mat> Ps;
    for (int j = 0; j < n_cams; ++j)
    {
        if (pts2D.at(j).rows < 10)
        {
            return false;
        }
        cv::Mat Pc;
        cv::hconcat(Rs[j], ts[j], Pc);
        if (!haveDists)
        {
            Pc = Ks[j] * Pc;
        }
        Ps.emplace_back(Pc.clone());
    }
    cv::Mat Q_inlier_mask = cv::Mat::ones(Q_nr, 1, CV_8UC1);
    if (cheirality_mask.empty())
    {
        Q_inlier_mask = cv::Mat::ones(Q_nr, 1, CV_8UC1);
    }
    else
    {
        Q_inlier_mask = cheirality_mask.getMat();
    }
    vector<int> cam_proj_indices(n_cams, 0);
    for (int j = 0; j < Q_nr; j++)
    {
        bool invalid_found = false;
        for (int c1 = 0; c1 < n_cams; ++c1)
        {
            if (map3D[c1].at<char>(j))
            {
                if (invalid_found){
                    cam_proj_indices[c1]++;
                    continue;
                }
                cv::Mat Qh = (cv::Mat_<double>(4, 1) << Q.at<double>(j, 0), Q.at<double>(j, 1), Q.at<double>(j, 2), 1.0);
                cv::Mat qc = Ps[c1] * Qh;
                if (haveDists)
                {
                    qc /= qc.at<double>(2);
                    qc.resize(2);
                    qc = distort(qc, dist_vec[c1]);
                    qc.resize(3, 1.0);
                    qc = Ks[c1] * qc;
                    qc /= qc.at<double>(2);
                    qc.resize(2);
                }
                else
                {
                    qc /= qc.at<double>(2);
                    qc.resize(2);
                }
                int &img_p_i = cam_proj_indices[c1];
                cv::Mat x_img = pts2D[c1].row(img_p_i++).t();
                cv::Mat x_diff = qc - x_img;
                const double repr_e_c = cv::norm(x_diff);
                if (repr_e_c > th)
                {
                    Q_inlier_mask.at<unsigned char>(j) = 0;
                    invalid_found = true;
                }
            }
        }
    }

    int nr_Q_remaining = cv::countNonZero(Q_inlier_mask);
    float rem_part = static_cast<float>(nr_Q_remaining) / static_cast<float>(Q_nr);
    if (!forceFiltering && ((nr_Q_remaining < 100 && nr_Q_remaining < Q_nr) || rem_part < minRemainRatio))
    {
        return false;
    }
    else if (forceFiltering && nr_Q_remaining < 10)
    {
        return false;
    }
    if (nr_Q_remaining < Q_nr)
    {
        cv::Mat Q_new, Q_new2, Q2_;
        bool sec_Q = !Q2.empty();
        if (sec_Q)
        {
            Q2_ = Q2.getMat();
        }
        std::vector<cv::Mat> map3D_new, pts2D_new(n_cams, cv::Mat());
        std::vector<std::vector<cv::Point2f>> pts2Df_new(n_cams, std::vector<cv::Point2f>());
        for (int j = 0; j < n_cams; ++j)
        {
            map3D_new.emplace_back(cv::Mat::zeros(nr_Q_remaining, 1, CV_8SC1));
        }
        cam_proj_indices = vector<int>(n_cams, 0);
        int q_idx = -1;
        for (int j = 0; j < Q_nr; j++)
        {
            if (Q_inlier_mask.at<unsigned char>(j))
            {
                Q_new.push_back(Q.row(j));
                if (sec_Q)
                {
                    Q_new2.push_back(Q2_.row(j));
                }
                q_idx++;
            }
            for (int c1 = 0; c1 < n_cams; ++c1)
            {
                if (map3D[c1].at<char>(j))
                {
                    if (Q_inlier_mask.at<unsigned char>(j))
                    {
                        map3D_new[c1].at<char>(q_idx) = 1;
                        pts2D_new[c1].push_back(pts2D[c1].row(cam_proj_indices[c1]));
                        if (pts2Df){
                            pts2Df_new[c1].emplace_back(pts2Df->at(c1).at(cam_proj_indices[c1]));
                        }
                    }
                    cam_proj_indices[c1]++;
                }
            }
        }
        for (int j = 0; j < n_cams; ++j)
        {
            if (pts2D_new.at(j).rows < 10)
            {
                return false;
            }
        }
        Q_new.copyTo(Q);
        if (sec_Q)
        {
            Q2.create(Q_new.size(), Q_new.type());
            cv::Mat Q_ref = Q2.getMat();
            Q_new2.copyTo(Q_ref);
        }
        map3D = move(map3D_new);
        pts2D = move(pts2D_new);
        if (pts2Df)
        {
            *pts2Df = move(pts2Df_new);
        }
    }
    return true;
}

/* Clone all parameters and 3D points for n cameras
 *
 * vector<Mat> Rs_in		        Input  -> Rotation matrix for every camera
 * vector<Mat> ts_in		        Input  -> Translation vector for every camera
 * vector<Mat> Rs_out		        Output -> Rotation matrix for every camera
 * vector<Mat> ts_out		        Output -> Translation vector for every camera
 * InputArray Qs_in				    Input  -> Optional 3D coordinates. 1 coordinate per row -> n rows
 * InputArray Qs_out				Output -> Optional 3D coordinates. 1 coordinate per row -> n rows
 * InputArray Ks_in		            Input  -> Camera matrix for every camera as vector<Mat>
 * vector<Mat> *Ks_out		        Output -> Camera matrix for every camera
 * InputArray dists_in		        Input  -> Optional distortion coeffitients for every camera
 * vector<Mat> *dists_out		    Output -> Optional distortion coeffitients for every camera
 *
 * Return value:		            None
 */
void cloneParameters(const std::vector<cv::Mat> &Rs_in, const std::vector<cv::Mat> &ts_in,
                     std::vector<cv::Mat> &Rs_out, std::vector<cv::Mat> &ts_out,
                     cv::InputArray Qs_in, cv::OutputArray Qs_out,
                     cv::InputArray Ks_in, std::vector<cv::Mat> *Ks_out,
                     cv::InputArray dists_in, std::vector<cv::Mat> *dists_out)
{
    const size_t nr_elems = Rs_in.size();
    Rs_out.clear();
    ts_out.clear();
    for (size_t i = 0; i < nr_elems; ++i)
    {
        Rs_out.emplace_back(Rs_in.at(i).clone());
        ts_out.emplace_back(ts_in.at(i).clone());
    }
    if (!Qs_in.empty() && Qs_out.needed())
    {
        cv::Mat Qs = Qs_in.getMat();
        if (Qs_out.empty())
        {
            Qs_out.create(Qs.size(), Qs.type());
        }
        cv::Mat Qsc = Qs_out.getMat();
        Qs.copyTo(Qsc);
    }
    if (!Ks_in.empty() && Ks_out)
    {
        vector<cv::Mat> Ks;
        Ks_in.getMatVector(Ks);
        Ks_out->clear();
        for (size_t i = 0; i < nr_elems; ++i)
        {
            Ks_out->emplace_back(Ks.at(i).clone());
        }
    }
    if (!dists_in.empty() && dists_out)
    {
        vector<cv::Mat> dists, distsc;
        dists_in.getMatVector(dists);
        dists_out->clear();
        for (size_t i = 0; i < nr_elems; ++i)
        {
            dists_out->emplace_back(dists.at(i).clone());
        }
    }
}

/* Project a 3D point on the image plane
 *
 * Vec3d Q			Input  -> 3D point
 * Matx33d R		Input  -> Rotation matrix
 * Vec3d t			Input  -> Translation vector
 * Matx33d K		Input  -> Camera matrix
 *
 * Return value:    Homogeneous normalized image projection [u v 1]
 */
cv::Mat_<double> project3D(const cv::Vec3d &Q, const cv::Matx33d &R, const cv::Vec3d &t, const cv::Matx33d &K)
{
    cv::Vec3d x = K * (R * Q + t);
    x /= x[2];
    return cv::Mat(x);
}

/* Calculates squared distance between a given image coordinate and a 3D point projected into the image plane (no distortion)
 *
 * Vec3d Q			Input  -> 3D point
 * Matx33d R		Input  -> Rotation matrix
 * Vec3d t			Input  -> Translation vector
 * Matx33d K		Input  -> Camera matrix
 * Vec2d x	    	Input  -> 2D image measurement
 *
 * Return value:    Squared distance between x and projected Q
 */
double reproject3DDist2(const cv::Vec3d &Q, const cv::Matx33d &R, const cv::Vec3d &t, const cv::Matx33d &K, const cv::Vec2d &x)
{
    cv::Mat xr = project3D(Q, R, t, K);
    xr.resize(2);
    cv::Vec2d xrv(xr);
    cv::Vec2d d = x - xrv;
    return d[0] * d[0] + d[1] * d[1];
}

/* Returns true, if the squared distance between a given image coordinate and a 3D point projected into the image plane (no distortion) 
 * is below a threshold (provide the squared threshold to compare against the squared distance)
 *
 * Vec3d Q			Input  -> 3D point
 * Matx33d R		Input  -> Rotation matrix
 * Vec3d t			Input  -> Translation vector
 * Matx33d K		Input  -> Camera matrix
 * Vec2d x	    	Input  -> 2D image measurement
 * double th    	Input  -> Squared pixel threshold
 *
 * Return value:    Squared distance between x and projected Q
 */
bool reproject3DTh2(const cv::Vec3d &Q, const cv::Matx33d &R, const cv::Vec3d &t, const cv::Matx33d &K, const cv::Vec2d &x, const double &th){
    double d = reproject3DDist2(Q, R, t, K, x);
    return d < th;
}

/* Calculates the euler angles from a given rotation matrix. As default the angles are returned in degrees.
 *
 * InputArray R							Input  -> Rotation matrix
 * double roll							Output -> Roll angle or Bank (rotation about x-axis)
 * double pitch							Output -> Pitch angle or Heading (rotation about y-axis)
 * double yaw							Output -> Yaw angle or Attitude (rotation about z-axis)
 * bool useDegrees						Input  -> If true (default), the angles are returned in degrees. Otherwise in radians.
 *
 * Return value:						none
 */
void getAnglesRotMat(cv::InputArray R, double & roll, double & pitch, double & yaw, bool useDegrees)
{
    Mat m = R.getMat();
    const double radDegConv = 180.0 / PI;

    /** this conversion uses conventions as described on page:
*   http://www.euclideanspace.com/maths/geometry/rotations/euler/index.htm
*   Coordinate System: right hand
*   Positive angle: right hand
*   Order of euler angles: pitch first, then yaw, then roll
*   matrix row column ordering:
*   [m00 m01 m02]
*   [m10 m11 m12]
*   [m20 m21 m22]*/

    // Assuming the angles are in radians.
    if (m.at<double>(1,0) > 0.998) { // singularity at north pole
        pitch = std::atan2(m.at<double>(0,2),m.at<double>(2,2));
        yaw = PI/2;
        roll = 0;
    }
    else if (m.at<double>(1,0) < -0.998) { // singularity at south pole
        pitch = std::atan2(m.at<double>(0,2),m.at<double>(2,2));
        yaw = -PI/2;
        roll = 0;
    }
    else
    {
        pitch = std::atan2(-m.at<double>(2,0),m.at<double>(0,0));
        roll = std::atan2(-m.at<double>(1,2),m.at<double>(1,1));
        yaw = std::asin(m.at<double>(1,0));
    }
    if(useDegrees)
    {
        pitch *= radDegConv;
        roll *= radDegConv;
        yaw *= radDegConv;
        pitch = round(1e6 * pitch) / 1e6;
        roll = round(1e6 * roll) / 1e6;
        yaw = round(1e6 * yaw) / 1e6;
    }
}

/* Calculates the difference (roation angle) between two rotation quaternions and the distance between
 * two 3D translation vectors back-rotated by the matrices R and Rcalib (therefore, this error represents
 * the full error caused by the different rotations and translations)
 *
 * Mat R                    Input  -> First rotation quaternion (e.g. result from pose estimation)
 * Mat Rcalib               Input  -> Second rotation quaternion (e.g. from offline calibration)
 * Mat T                    Input  -> First 3D (translation) vector (e.g. result from pose estimation)
 * Mat Tcalib               Input  -> Second 3D (translation) vector (e.g. from offline calibration)
 * double rdiff				Output -> Rotation angle (from Angle-axis-representation) between the two rotations
 * double tdiff				Output -> Distance between the two translation vectors back-rotated by the matrices
 *									  R and Rcalib
 *
 * Return value:			none
 */
void getRTQuality(cv::Mat & R, cv::Mat & Rcalib, cv::Mat & T,
                  cv::Mat & Tcalib, double* rdiff, double* tdiff)
{
    CV_Assert((R.rows == 4) && (R.cols == 1) && (Rcalib.rows == 4) && (Rcalib.cols == 1) &&
              (T.rows == 3) && (T.cols == 1) && (Tcalib.rows == 3) && (Tcalib.cols == 1) &&
              (R.type() == CV_64FC1) && (Rcalib.type() == CV_64FC1) && (T.type() == CV_64FC1) && (Tcalib.type() == CV_64FC1) &&
              rdiff && tdiff);
    Eigen::Vector4d Re, Rcalibe;
    Eigen::Vector3d Te, Tcalibe;
    cv::cv2eigen(R, Re);
    cv::cv2eigen(Rcalib, Rcalibe);
    cv::cv2eigen(T, Te);
    cv::cv2eigen(Tcalib, Tcalibe);
    getRTQuality(Re, Rcalibe, Te, Tcalibe, rdiff, tdiff);
}

/* Calculates the difference (roation angle) between two rotation quaternions and the distance between
 * two 3D translation vectors back-rotated by the matrices R and Rcalib (therefore, this error represents
 * the full error caused by the different rotations and translations)
 *
 * Eigen::Vector4d R		Input  -> First rotation quaternion (e.g. result from pose estimation)
 * Eigen::Vector4d Rcalib	Input  -> Second rotation quaternion (e.g. from offline calibration)
 * Eigen::Vector3d T		Input  -> First 3D (translation) vector (e.g. result from pose estimation)
 * Eigen::Vector3d Tcalib	Input  -> Second 3D (translation) vector (e.g. from offline calibration)
 * double rdiff				Output -> Rotation angle (from Angle-axis-representation) between the two rotations
 * double tdiff				Output -> Distance between the two translation vectors back-rotated by the matrices
 *									  R and Rcalib
 *
 * Return value:			none
 */
void getRTQuality(Eigen::Vector4d & R, Eigen::Vector4d & Rcalib, Eigen::Vector3d & T,
                  Eigen::Vector3d & Tcalib, double* rdiff, double* tdiff)
{
    Eigen::Vector4d t1, t2;

    *rdiff = rotDiff(R, Rcalib);

    Eigen::Vector3d Tdiff1;
    Tdiff1 = quatMult3DPt(quatConj(R), T);
    Tdiff1 -= quatMult3DPt(quatConj(Rcalib), Tcalib); //Error vecot includes both, the error from R and T

    *tdiff = std::sqrt(Tdiff1(0)*Tdiff1(0) + Tdiff1(1)*Tdiff1(1) + Tdiff1(2)*Tdiff1(2));
}

/* Calculates the essential matrix from the rotation matrix R and the translation
 * vector t: E = [t]x * R
 *
 * cv::Mat R						Input  -> Rotation matrix R
 * cv::Mat t						Input  -> Translation vector t
 *
 * Return value:					Essential matrix
 */
cv::Mat getEfromRT(const cv::Mat& R, const cv::Mat& t)
{
    return getSkewSymMatFromVec(t/normFromVec(t)) * R;
}

/* Calculates the essential matrix from the rotation matrix R and the translation
 * vector t: E = [t]x * R
 *
 * Matrix3d R						Input  -> Rotation matrix R
 * Vector3d t						Input  -> Translation vector t
 *
 * Return value:					Essential matrix
 */
Eigen::Matrix3d getEfromRT(const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
{
    return getSkewSymMatFromVec<double>(t / t.norm()) * R;
}

/* Generates a 3x3 skew-symmetric matrix from a 3-vector (allows multiplication
 * instead of cross-product)
 *
 * Eigen::Vector4d & Q1				Input  -> Quaternion in the form [w,x,y,z]
 *
 * Return value:					The resulting quaternion in the form [w,x,y,z]
 */
cv::Mat getSkewSymMatFromVec(cv::Mat t)
{
    if(t.type() != CV_64FC1)
        t.convertTo(t,CV_64FC1);

    return (Mat_<double>(3, 3) << 0, -t.at<double>(2), t.at<double>(1),
                                  t.at<double>(2), 0, -t.at<double>(0),
                                  -t.at<double>(1), t.at<double>(0), 0);
}

/* Converts a (Rotation) Quaternion to a (Rotation) matrix
 *
 * Mat q							Input  -> Quaternion vector of the form [w,x,y,z]
 * Mat R							Output -> 3x3 Rotation matrix
 *
 * Return value:					none
 */
void quatToMatrix(cv::Mat & R, cv::Mat q)
{
    R.create(3,3,CV_64FC1);
    double sqw = q.at<double>(0)*q.at<double>(0);
    double sqx = q.at<double>(1)*q.at<double>(1);
    double sqy = q.at<double>(2)*q.at<double>(2);
    double sqz = q.at<double>(3)*q.at<double>(3);

    // invs (inverse square length) is only required if quaternion is not already normalised
    double invs = 1 / (sqx + sqy + sqz + sqw);
    R.at<double>(0,0) = ( sqx - sqy - sqz + sqw)*invs ; // since sqw + sqx + sqy + sqz =1/invs*invs
    R.at<double>(1,1) = (-sqx + sqy - sqz + sqw)*invs ;
    R.at<double>(2,2) = (-sqx - sqy + sqz + sqw)*invs ;

    double tmp1 = q.at<double>(1)*q.at<double>(2);
    double tmp2 = q.at<double>(3)*q.at<double>(0);
    R.at<double>(1,0) = 2.0 * (tmp1 + tmp2)*invs ;
    R.at<double>(0,1) = 2.0 * (tmp1 - tmp2)*invs ;

    tmp1 = q.at<double>(1)*q.at<double>(3);
    tmp2 = q.at<double>(2)*q.at<double>(0);
    R.at<double>(2,0) = 2.0 * (tmp1 - tmp2)*invs ;
    R.at<double>(0,2) = 2.0 * (tmp1 + tmp2)*invs ;
    tmp1 = q.at<double>(2)*q.at<double>(3);
    tmp2 = q.at<double>(1)*q.at<double>(0);
    R.at<double>(2,1) = 2.0 * (tmp1 + tmp2)*invs ;
    R.at<double>(1,2) = 2.0 * (tmp1 - tmp2)*invs ;
}


/* Converts a (Rotation) matrix to a (Rotation) quaternion
 *
 * Matrix3d rot						Input  -> 3x3 Rotation matrix
 * Vector4d quat					Output -> Quaternion vector of the form [w,x,y,z]
 *
 * Return value:					none
 */
void MatToQuat(const Eigen::Matrix3d & rot, Eigen::Vector4d & quat) {
    /*    double trace = rot.trace();

        MAT_TO_QUAT(ACCESS_EIGENMAT_AS_MAT)

        normalise();
        lengthOk();

        BROKEN -- try this from boost instead
     */

    double fTrace = rot.trace();
    double fRoot;

    //From http://www.geometrictools.com/LibFoundation/Mathematics/Wm4Quaternion.inl
    double m_afTuple[4];
    if (fTrace > (double) 0.0) //0 is w
    {
        // |w| > 1/2, may as well choose w > 1/2
        fRoot = sqrt(fTrace + (double) 1.0); // 2w
        m_afTuple[0] = ((double) 0.5) * fRoot;
        fRoot = ((double) 0.5) / fRoot; // 1/(4w)
        m_afTuple[1] = (rot(2, 1) - rot(1, 2)) * fRoot;
        m_afTuple[2] = (rot(0, 2) - rot(2, 0)) * fRoot;
        m_afTuple[3] = (rot(1, 0) - rot(0, 1)) * fRoot;
    } else {
        // |w| <= 1/2
        int i = 0;
        if (rot(1, 1) > rot(0, 0)) {
            i = 1;
        }
        if (rot(2, 2) > rot(i, i)) {
            i = 2;
        }
        //        int j = ms_iNext[i];
        //        int k = ms_iNext[j];
        int j = (i + 1);
        j %= 3;
        int k = (j + 1);
        k %= 3;

        fRoot = sqrt(rot(i, i) - rot(j, j) - rot(k, k)+(double) 1.0);
        //double* apfQuat[3] = { &m_afTuple[1], &m_afTuple[2], &m_afTuple[3] };
        m_afTuple[i + 1] = ((double) 0.5) * fRoot;
        fRoot = ((double) 0.5) / fRoot;
        m_afTuple[0] = (rot(k, j) - rot(j, k)) * fRoot;
        m_afTuple[j + 1] = (rot(j, i) + rot(i, j)) * fRoot;
        m_afTuple[k + 1] = (rot(k, i) + rot(i, k)) * fRoot;
    }

    quat(0) = m_afTuple[0];
    quat(1) = m_afTuple[1];
    quat(2) = m_afTuple[2];
    quat(3) = m_afTuple[3];
}

/* Converts a quaternion to axis angle representation.
 *
 * Vector4d quat					Input  -> Quaternion vector of the form [w,x,y,z]
 * Vector3d axis					Output -> Rotation axis [x,y,z]
 * double angle						Output -> Rotation angle
 *
 * Return value:					none
 */
void QuatToAxisAngle(Eigen::Vector4d quat, Eigen::Vector3d axis, double & angle)
{
    //From http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm

    Eigen::Vector4d quat_n = quat;
    if(quat_n(0) > 1.0) // if w>1 acos and sqrt will produce errors, this cant happen if quaternion is normalised
        quat_n.normalize();
    angle = 2.0 * std::acos(quat_n(0));
    double s = std::sqrt(1.0 - quat_n(0) * quat_n(0)); // assuming quaternion normalised then w is less than 1, so term always positive.
    if (s < 0.001) { // test to avoid divide by zero, s is always positive due to sqrt
     // if s close to zero then direction of axis not important
     axis(0) = quat_n(1); // if it is important that axis is normalised then replace with x=1; y=z=0;
     axis(1) = quat_n(2);
     axis(2) = quat_n(3);
   } else {
     axis(0) = quat_n(1) / s; // normalise axis
     axis(1) = quat_n(2) / s;
     axis(2) = quat_n(3) / s;
   }
}

/* Calculates the product of a quaternion and a conjugated quaternion. This is used e.g. to calculate the
 * angular difference between two rotation quaternions
 *
 * Eigen::Vector4d Q1				Input  -> The first quaternion in the form [w,x,y,z]
 * Eigen::Vector4d Q2				Input  -> The second quaternion in the form [w,x,y,z]
 * Eigen::Vector4d & Qres			Output -> The resulting quaternion in the form [w,x,y,z]
 *
 * Return value:					none
 */
void quatMultConj(const Eigen::Vector4d & Q1, const Eigen::Vector4d & Q2, Eigen::Vector4d & Qres)
{
    //v(4)=dotproduct(a,quatConj(b));
    //v.rows(1,3)=crossproduct(a_vec,b_vec)   +   a(4)*b_vec    +   b(4)*a_vec;

    Qres(1) = ((Q1(3) * Q2(2) - Q1(2) * Q2(3)) - Q1(0) * Q2(1)) + Q2(0) * Q1(1);
    Qres(2) = ((Q1(1) * Q2(3) - Q1(3) * Q2(1)) - Q1(0) * Q2(2)) + Q2(0) * Q1(2);
    Qres(3) = ((Q1(2) * Q2(1) - Q1(1) * Q2(2)) - Q1(0) * Q2(3)) + Q2(0) * Q1(3);

    Qres(0) = Q1(1) * Q2(1) + Q1(2) * Q2(2) + Q1(3) * Q2(3) + Q1(0) * Q2(0); //just dot prod
}

/* Normalizes the provided quaternion.
 *
 * Eigen::Vector4d Q1				Input & Output  -> A quaternion in the form [w,x,y,z] must be provided.
 *													   The normalized quaternion is also returned here.
 *
 * Return value:					none
 */
void quatNormalise(Eigen::Vector4d & Q)
{
    double length = Q(0) * Q(0) + Q(1) * Q(1) + Q(2) * Q(2) + Q(3) * Q(3);
    double check = length - 1;
    if (check > 0.0000001 || check < -0.0000001) {
        double scale = 1.0 / sqrt(length);
        Q(0) *= scale;
        Q(1) *= scale;
        Q(2) *= scale;
        Q(3) *= scale;
    }
}

/* Calculates the angle of a quaternion.
 *
 * Eigen::Vector4d Q1				Input  -> Quaternion in the form [w,x,y,z]
 *
 * Return value:					The angle in RAD.
 */
double quatAngle(Eigen::Vector4d & Q)
{
    double cosAng = fabs(Q(0));
    if (cosAng > 1.0) cosAng = 1.0;
    double ang = 2 * acos(cosAng);
    if(ang < 0)
        cout << "acos returning val less than 0" << endl;
    //if(isnan(ang))
    //	cout << "acos returning nan" << endl;
    if (ang > PI) ang -= 2 * PI;
    return ang;
}

/* Multiplies a quaternion with a 3D-point (e.g. translation vector)
 *
 * Eigen::Vector4d q				Input  -> Quaternion in the form [w,x,y,z]
 * Eigen::Vector3d p				Input  -> 3D-point in the form [x,y,z]
 *
 * Return value:					The new 3D-point or translation vector
 */
Eigen::Vector3d quatMult3DPt(const Eigen::Vector4d & q, const Eigen::Vector3d & p)
{
    Eigen::Vector3d multPoint;

    //v=q*v*q_conjugate

    Eigen::Vector4d temp;
    quatMultByVec(q, p, temp);
    quatMultConjIntoVec(temp, q, multPoint);

    return multPoint;
}

/* Multiplies a quaternion with a vector
 *
 * Eigen::Vector4d & Q1				Input  -> Quaternion in the form [w,x,y,z]
 * Eigen::Vector3d & vec			Input  -> Vector in the form [x,y,z]
 * Eigen::Vector4d & Qres			Output -> The resulting quaternion vector in the form [w,x,y,z]
 *
 * Return value:					none
 */
void quatMultByVec(const Eigen::Vector4d & Q1, const Eigen::Vector3d & vec, Eigen::Vector4d & Qres)
{
    //v(4)=dotproduct(a,quatConj(b));
    //v.rows(1,3)=crossproduct(a_vec,b_vec)   +   a(4)*b_vec    +   b(4)*a_vec;

    Qres(1) = Q1(2) * vec(2) - Q1(3) * vec(1) + Q1(0) * vec(0);
    Qres(2) = Q1(3) * vec(0) - Q1(1) * vec(2) + Q1(0) * vec(1);
    Qres(3) = Q1(1) * vec(1) - Q1(2) * vec(0) + Q1(0) * vec(2);

    Qres(0) = -(Q1(1) * vec(0) + Q1(2) * vec(1) + Q1(3) * vec(2));
}

/* Multiplies a quaternion with a conj. quaternion
 *
 * Eigen::Vector4d & Q1				Input  -> Quaternion in the form [w,x,y,z]
 * Eigen::Vector4d & Q2				Input  -> Quaternion in the form [w,x,y,z]
 * Eigen::Vector3d & Qres			Output -> The resulting vector or 3D-point in the form [x,y,z]
 *
 * Return value:					none
 */
void quatMultConjIntoVec(const Eigen::Vector4d & Q1, const Eigen::Vector4d & Q2, Eigen::Vector3d & Qres)
{

    //v(4)=dotproduct(a,quatConj(b));
    //v.rows(1,3)=crossproduct(a_vec,b_vec)   +   a(4)*b_vec    +   b(4)*a_vec;

    Qres(0) = ((Q1(3) * Q2(2) - Q1(2) * Q2(3)) - Q1(0) * Q2(1)) + Q2(0) * Q1(1);
    Qres(1) = ((Q1(1) * Q2(3) - Q1(3) * Q2(1)) - Q1(0) * Q2(2)) + Q2(0) * Q1(2);
    Qres(2) = ((Q1(2) * Q2(1) - Q1(1) * Q2(2)) - Q1(0) * Q2(3)) + Q2(0) * Q1(3);

    if(!nearZero(Q1(0) * Q2(0) + Q1(1) * Q2(1) + Q1(2) * Q2(2) + Q1(3) * Q2(3)))
        cout << "Bad rotation (probably scale overflow creating a massive vector)" << endl; //just dot prod
}

/* Calculates the difference (roation angle) between two rotation quaternions.
 *
 * Eigen::Vector4d R		Input  -> First rotation quaternion (e.g. result from pose estimation)
 * Eigen::Vector4d Rcalib	Input  -> Second rotation quaternion (e.g. from offline calibration)
 *
 * Return value:			Rotation angle (from Angle-axis-representation) between the two rotations
 */
double rotDiff(Eigen::Vector4d & R, Eigen::Vector4d & Rcalib)
{
    Eigen::Vector4d Rdiff1;
    quatMultConj(R,Rcalib,Rdiff1);
    quatNormalise(Rdiff1);
    return quatAngle(Rdiff1);
}

/* Calculates the transponse (inverse rotation) of a quaternion
 *
 * Eigen::Vector4d & Q1				Input  -> Quaternion in the form [w,x,y,z]
 *
 * Return value:					The resulting quaternion in the form [w,x,y,z]
 */
Eigen::Vector4d quatConj(const Eigen::Vector4d & Q) //'transpose' -- inverse rotation
{
    Eigen::Vector4d invertedRot;

    for (int i = 1; i < 4; i++)
        invertedRot(i) = -Q(i);

    invertedRot(0) = Q(0);

    return invertedRot;
}

/* Computes the Lie-Logarithm for a rotation matrix R
 *
 * Eigen::Matrix3d R		Input  -> Rotation matrix
 *
 * Return value:			The resulting Lie-Logarithm of R
 */
Eigen::Vector3cd compute_Lie_log(const Eigen::Matrix3d &R)
{
    //Computes the Lie-Logarithm
    //R = expm([lie_log(R)]_x) ... with [*]_x a skew-smmetric matrix (cross product -> matrix multiplication)
    //see page 18 of http://users.cecs.anu.edu.au/~hongdong/rotationaveraging.pdf
    //and https://de.mathworks.com/matlabcentral/answers/240322-how-to-find-rotation-matrix-mean

    //Calculate eigen values and eigen vectors
    Eigen::EigenSolver<Eigen::Matrix3d> es;
    es.compute(R, true);

    Eigen::Vector3cd eig_vals_compl = es.eigenvalues();
    Eigen::Matrix3cd eig_vecs_compl = es.eigenvectors();

    Eigen::Vector3d eig_vals_phase = eig_vals_compl.array().arg();

    int idx_max, idx_min;
    double ang = eig_vals_phase.maxCoeff(&idx_max);

    Eigen::Vector3cd eig_vals_m1 = eig_vals_compl - Eigen::Vector3d::Ones();

    Eigen::Vector3d eig_vals_magnitude = eig_vals_m1.array().abs2();
    double mag = eig_vals_magnitude.minCoeff(&idx_min);

    Eigen::Vector3cd v1 = ang * eig_vecs_compl.col(idx_min);
    Eigen::Vector3cd v2 = -1. * ang * eig_vecs_compl.col(idx_min);

    Eigen::Matrix3cd v1_x = getSkewSymMatFromVec(v1);
    Eigen::Matrix3cd v2_x = getSkewSymMatFromVec(v2);

    Eigen::Matrix3d R1 = v1_x.exp().real();
    Eigen::Matrix3d R2 = v2_x.exp().real();

    Eigen::Matrix3d RR1 = R * R1.transpose(); // R1.adjoint();
    Eigen::Matrix3d RR2 = R * R2.transpose(); // R2.adjoint();

    Eigen::EigenSolver<Eigen::Matrix3d> es1;
    es.compute(RR1, false);
    Eigen::Vector3cd eig_vals_compl1 = es1.eigenvalues();
    Eigen::Vector3d eig_vals_phase1 = eig_vals_compl1.array().arg();
    double ang1 = eig_vals_phase1.maxCoeff();

    es.compute(RR2, false);
    Eigen::Vector3cd eig_vals_compl2 = es1.eigenvalues();
    Eigen::Vector3d eig_vals_phase2 = eig_vals_compl2.array().arg();
    double ang2 = eig_vals_phase2.maxCoeff();

    if (ang2 > ang1){
        return v1;
    }
    return v2;
}

/* Calculate a mean rotation matrix from multiple rotation matrices
 * See page 18 of http://users.cecs.anu.edu.au/~hongdong/rotationaveraging.pdf
 *
 * vector<Mat> Rs		Input  -> Rotation matrices
 *
 * Return value:        Average rotation matrix
 */
cv::Mat getMeanRotation(const std::vector<cv::Mat> &Rs)
{
    //see page 18 of http://users.cecs.anu.edu.au/~hongdong/rotationaveraging.pdf
    if(Rs.empty()){
        return cv::Mat();
    }
    if (Rs.size() == 1)
    {
        return Rs[0];
    }
    #define USE_EXPONENTIAL 0

    cv::Mat R_avg = Rs[0].clone();
    const int max_it = 1000;
    const double tol_r = 1e-10;
    for (int i = 0; i < max_it; i++)
    {
        std::vector<cv::Mat> axis_angle;
        //Get axis-angle representations of rotation differences
        for (const auto &Rx : Rs)
        {
            cv::Mat v;
            cv::Rodrigues(R_avg.t() * Rx, v);
            axis_angle.emplace_back(v);
        }

        //Calculate mean axis-angle difference
        cv::Mat v_mean = cv::Mat::zeros(axis_angle[0].size(), axis_angle[0].type());
        for (const auto &vx : axis_angle)
        {
            v_mean += vx;
        }
        v_mean /= static_cast<double>(axis_angle.size());

#if USE_EXPONENTIAL
        Eigen::Vector3d v_eig;
        cv::cv2eigen(v_mean, v_eig);
        //Calculate rotation matrix using matrix exponential of skew-symmetric matrix from axis-angle
        //See pages 583-585 in "Multiple View Geometry in computer vision" from R. Hartley and A. Zisserman
        Eigen::Matrix3d V_x = getSkewSymMatFromVec(v_eig);
        Eigen::Matrix3d R_eig = V_x.exp();
        cv::Mat R_diff;
        cv::eigen2cv(R_eig, R_diff);
#else
        cv::Mat R_diff;
        cv::Rodrigues(v_mean, R_diff);
#endif
        R_avg = R_avg * R_diff;

        const double v_norm = cv::norm(v_mean);
        if(v_norm < tol_r){
            break;
        }
    }

    return R_avg;
}

/* Calculate a mean translation vector from multiple translation vectors
 *
 * vector<Mat> ts		Input  -> Translation vectors
 *
 * Return value:        Average translation vector
 */
cv::Mat getMeanTranslation(const std::vector<cv::Mat> &ts)
{
    if (ts.empty())
    {
        return cv::Mat();
    }
    if (ts.size() == 1)
    {
        return ts[0];
    }

    cv::Mat t_mean = cv::Mat::zeros(ts[0].size(), ts[0].type());
    // std::cout << "Translations for AVG:" << std::endl;
    for(const auto &t: ts){
        t_mean += t;
        // printCvMat(t.t(), "t");
    }
    t_mean /= static_cast<double>(ts.size());

    return t_mean;
}

/* Calculate the camera center from rotation matrix R and translation vector t
 *
 * Mat R		Input  -> Rotation matrix
 * Mat t		Input  -> Translation vector
 *
 * Return value:        Camera center
 */
cv::Mat getCamCenterFromRt(const cv::Mat &R, const cv::Mat &t)
{
    return -1.0 * R.t() * t;
}

/* Normalizes the image coordinates and transfers the image coordinates into
 * cameracoordinates, respectively.
 *
 * vector<Point2f> points			I/O    -> Input: Image coordinates (in pixels)
 *											  Output: Camera coordinates
 * Mat K							Input  -> Camera matrix
 *
 * Return value:					none
 */
void ImgToCamCoordTrans(std::vector<cv::Point2f>& points, cv::Mat K)
{
    size_t n = points.size();

    for(size_t i = 0; i < n; i++)
    {
        points[i].x = (float)(((double)points[i].x - K.at<double>(0,2))/K.at<double>(0,0));
        points[i].y = (float)(((double)points[i].y - K.at<double>(1,2))/K.at<double>(1,1));
    }
}

/* Transfers coordinates from the camera coordinate system into the the image coordinate system.
 *
 * vector<Point2f> points			I/O    -> Input: Camera coordinates
 *											  Output: Image coordinates (in pixels)
 * Mat K							Input  -> Camera matrix
 *
 * Return value:					none
 */
void CamToImgCoordTrans(std::vector<cv::Point2f>& points, cv::Mat K)
{
    size_t n = points.size();

    for(size_t i = 0; i < n; i++)
    {
        points[i].x = (float)((double)points[i].x * K.at<double>(0,0) + K.at<double>(0,2));
        points[i].y = (float)((double)points[i].y * K.at<double>(1,1) + K.at<double>(1,2));
    }
}

/* Transfers coordinates from the camera coordinate system into the the image coordinate system.
 *
 * Mat points						I/O    -> Input: Camera coordinates (n rows x 2 cols)
 *											  Output: Image coordinates (in pixels)
 * Mat K							Input  -> Camera matrix
 *
 * Return value:					none
 */
void CamToImgCoordTrans(cv::Mat& points, cv::Mat K)
{
    CV_Assert(((points.rows > points.cols) || ((points.rows <= 2) && (points.cols == 2))) && (points.type() == CV_64FC1));
    int n = points.rows;

    for(int i = 0; i < n; i++)
    {
        points.at<double>(i,0) = points.at<double>(i,0) * K.at<double>(0,0) + K.at<double>(0,2);
        points.at<double>(i,1) = points.at<double>(i,1) * K.at<double>(1,1) + K.at<double>(1,2);
    }
}

/* This function removes the lens distortion for corresponding points in the first (left) and second (right) image.
 * Moreover, correspondences for which undistortion with subsequent distortion does not lead to the original
 * coordinate, are removed. Thus, the point set might be smaller. The same lens distortion models as within the OpenCV
 * library are supported (max. 8 coefficients including 2 for tangential distortion).
 *
 * vector<Point2f> points1			I/O	   -> Points in the first (left) image using the camera coordinate system
 * vector<Point2f> points2			I/O	   -> Points in the second (right) image using the camera coordinate system
 * cv::Mat dist1					Input  -> Distortion coefficients of the first (left) image. The ordering of the
 *											  coefficients is compliant with the OpenCV library. A number of 8
 *											  coefficients is required. If higher order coefficients are not
 *											  available, set them to 0.
 * cv::Mat dist2					Input  -> Distortion coefficients of the second (right) image. The ordering of the
 *											  coefficients is compliant with the OpenCV library. A number of 8
 *											  coefficients is required. If higher order coefficients are not
 *											  available, set them to 0.
 *
 * Return value:					true:  Everything ok.
 *									false: Undistortion failed
 */
bool Remove_LensDist(std::vector<cv::Point2f>& points1,
                     std::vector<cv::Point2f>& points2,
                     const cv::Mat& dist1,
                     const cv::Mat& dist2)
{
    CV_Assert(points1.size() == points2.size());

    if(nearZero(sum(dist1)[0]) && nearZero(sum(dist2)[0])){
        return true;
    }

    vector<Point2f> distpoints1, distpoints2;
    int n1, n = (int)points1.size();
    cv::Mat mask = cv::Mat::ones(1, n, CV_8UC1);

    //Remove the lens distortion on the normalized coordinates (camera coordinate system)
    distpoints1 = points1;
    distpoints2 = points2;
    for(int i = 0;i < n;i++)
    {
        if(!LensDist_Oulu(distpoints1[i], points1[i], dist1, cv::noArray()))
        {
            mask.at<bool>(i) = false;
            continue;
        }
        if(!LensDist_Oulu(distpoints2[i], points2[i], dist2, cv::noArray()))
            mask.at<bool>(i) = false;
    }

    n1 = cv::countNonZero(mask);

    if(n1 < 16)
        return false;

    if((float)n1/(float)n < 0.75f)
        cout << "There is a problem with the distortion parameters! Check your internal calibration parameters!" << endl;

    //Remove invalid correspondences
    if(n1 < n)
    {
        vector<Point2f> validPoints1, validPoints2;
        for(int i = 0;i < n;i++)
        {
            if(mask.at<bool>(i))
            {
                validPoints1.push_back(points1[i]);
                validPoints2.push_back(points2[i]);
            }
        }
        points1 = validPoints1;
        points2 = validPoints2;
    }

    return true;
}


/* This function undistorts a point in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y] 
 * using the methode from the Oulu University.
 *
 * Point2f* distorted				Input  -> Distorted point
 * Point2f* corrected				Output -> Corrected (undistorted) point
 * cv::Mat dist						Input  -> Distortion coefficients. The ordering of the coefficients
 *											  is compliant with the OpenCV library.
 * int iters						Input  -> Number of iterations used to correct the point:
 *									the higher the number, the better the solution (use a number
 *									between 3 and 20). The number 3 typically results in an error
 *									of 0.1 pixels.
 *
 * Return value:					true:  Everything ok.
 *									false: Undistortion failed
 */
bool LensDist_Oulu(const cv::Point2f& distorted, cv::Point2f& corrected, cv::Mat dist, int iters)
{
    CV_Assert(dist.cols == 8);

    double r2, _2xy, rad_corr, delta[2], k1, k2, k3, k4, k5, k6, p1, p2;
    k1 = dist.at<double>(0);
    k2 = dist.at<double>(1);
    p1 = dist.at<double>(2);
    p2 = dist.at<double>(3);
    k3 = dist.at<double>(4);
    k4 = dist.at<double>(5);
    k5 = dist.at<double>(6);
    k6 = dist.at<double>(7);

    for(int i = 0;i < iters;i++)
    {
        r2 = (double)corrected.x * (double)corrected.x + (double)corrected.y * (double)corrected.y;
        _2xy = 2.0 * (double)corrected.x * (double)corrected.y;
        rad_corr = (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2);
        delta[0] = p1 * _2xy + p2 * (r2 + 2.0 * (double)corrected.x * (double)corrected.x);
        delta[1] = p1 * (r2 + 2.0 * (double)corrected.y * (double)corrected.y) + p2 * _2xy;
        corrected.x = (float)(((double)distorted.x - delta[0]) / rad_corr);
        corrected.y = (float)(((double)distorted.y - delta[1]) / rad_corr);
    }

    //proof
    Point2f proofdist;
    r2 = (double)corrected.x * (double)corrected.x + (double)corrected.y * (double)corrected.y;
    _2xy = 2.0 * (double)corrected.x * (double)corrected.y;
    rad_corr = (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2);
    delta[0] = p1 * _2xy + p2 * (r2 + 2.0 * (double)corrected.x * (double)corrected.x);
    delta[1] = p1 * (r2 + 2.0 * (double)corrected.y * (double)corrected.y) + p2 * _2xy;
    proofdist.x = (float)((double)corrected.x * rad_corr + delta[0] - (double)distorted.x);
    proofdist.y = (float)((double)corrected.y * rad_corr + delta[1] - (double)distorted.y);
    if( std::sqrt(proofdist.x * proofdist.x + proofdist.y * proofdist.y) > 0.25f)
        return false;

    return true;
}

/* This function undistorts a point in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y] 
 * using the methode from the Oulu University. Different distortion models are suppoted including
 * thin prism distortion and the distortion (rotation) matrix R_tau for the tilted sensor model distortion
 *
 * cv::Mat distorted				Input  -> Distorted point
 * cv::Mat corrected				Output -> Corrected (undistorted) point
 * cv::Mat dist						Input  -> Distortion coefficients. The ordering of the coefficients
 *											  is compliant with the OpenCV library: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4]]]) or (k1,k2,k3)
 * InputArray R_tau				    Input  -> Distortion (rotation) matrix for the tilted sensor model distortion
 * int iters						Input  -> Number of iterations used to correct the point:
 *									the higher the number, the better the solution (use a number
 *									between 3 and 20). The number 3 typically results in an error
 *									of 0.1 pixels.
 *
 * Return value:					true:  Everything ok.
 *									false: Undistortion failed
 */
bool LensDist_Oulu(const cv::Mat &distorted, cv::Mat &corrected, const cv::Mat &dist, cv::InputArray R_tau, int iters)
{
    CV_Assert(distorted.rows * distorted.cols == 2 && distorted.type() == CV_64FC1);
    if (corrected.empty())
    {
        corrected = distorted.clone();
    }
    const cv::Point2d distorted_d(distorted.at<double>(0), distorted.at<double>(1));
    cv::Point2d corrected_d(corrected.at<double>(0), corrected.at<double>(0));
    const bool ret = LensDist_Oulu(distorted_d, corrected_d, dist, R_tau, iters);
    corrected = cv::Mat(corrected_d, true).reshape(1, distorted.rows);
    return ret;
}

/* This function undistorts a point in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y] 
 * using the methode from the Oulu University. Different distortion models are suppoted including
 * thin prism distortion and the distortion (rotation) matrix R_tau for the tilted sensor model distortion
 *
 * cv::Point2f distorted			Input  -> Distorted point
 * cv::Point2f corrected			Output -> Corrected (undistorted) point
 * cv::Mat dist						Input  -> Distortion coefficients. The ordering of the coefficients
 *											  is compliant with the OpenCV library: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4]]]) or (k1,k2,k3)
 * InputArray R_tau				    Input  -> Distortion (rotation) matrix for the tilted sensor model distortion
 * int iters						Input  -> Number of iterations used to correct the point:
 *									the higher the number, the better the solution (use a number
 *									between 3 and 20). The number 3 typically results in an error
 *									of 0.1 pixels.
 *
 * Return value:					true:  Everything ok.
 *									false: Undistortion failed
 */
bool LensDist_Oulu(const cv::Point2f &distorted, cv::Point2f &corrected, const cv::Mat &dist, cv::InputArray R_tau, int iters)
{
    const cv::Point2d distorted_d(static_cast<double>(distorted.x), static_cast<double>(distorted.y));
    cv::Point2d corrected_d(static_cast<double>(corrected.x), static_cast<double>(corrected.y));
    const bool ret = LensDist_Oulu(distorted_d, corrected_d, dist, R_tau, iters);
    corrected = cv::Point2f(static_cast<float>(corrected_d.x), static_cast<float>(corrected_d.y));
    return ret;
}

/* This function undistorts a point in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y] 
 * using the methode from the Oulu University. Different distortion models are suppoted including
 * thin prism distortion and the distortion (rotation) matrix R_tau for the tilted sensor model distortion
 *
 * cv::Point2d distorted			Input  -> Distorted point
 * cv::Point2d corrected			Output -> Corrected (undistorted) point
 * cv::Mat dist						Input  -> Distortion coefficients. The ordering of the coefficients
 *											  is compliant with the OpenCV library: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4]]]) or (k1,k2,k3)
 * InputArray R_tau				    Input  -> Distortion (rotation) matrix for the tilted sensor model distortion
 * int iters						Input  -> Number of iterations used to correct the point:
 *									the higher the number, the better the solution (use a number
 *									between 3 and 20). The number 3 typically results in an error
 *									of 0.1 pixels.
 *
 * Return value:					true:  Everything ok.
 *									false: Undistortion failed
 */
bool LensDist_Oulu(const cv::Point2d &distorted, cv::Point2d &corrected, const cv::Mat &dist, cv::InputArray R_tau, int iters)
{
    CV_Assert(dist.cols <= 14 && dist.rows <= 14);
    const int distSi = dist.cols * dist.rows;
    CV_Assert(distSi != 4 && distSi != 6 && distSi != 7 && distSi != 9 && distSi != 10 && distSi != 11 && distSi != 13);
    const cv::Point2d distorted_d = distorted; //For the case, the same variable was provided for distorted and corrected
    corrected = distorted_d;

    double r2, r4, _2xy, rad_corr, delta[2], k1 = 0., k2 = 0., k3 = 0., k4 = 0., k5 = 0., k6 = 0., p1 = 0., p2 = 0.;
    double s1 = 0., s2 = 0., s3 = 0., s4 = 0.;
    double tx = 0., ty = 0.;
    cv::Mat R, Ri;
    if (distSi >= 1)
    {
        k1 = dist.at<double>(0);
    }
    if (distSi >= 2)
    {
        k2 = dist.at<double>(1);
    }
    if (distSi == 3)
    {
        k3 = dist.at<double>(4);
    }
    else if (distSi >= 5)
    {
        k3 = dist.at<double>(4);
        p1 = dist.at<double>(2);
        p2 = dist.at<double>(3);
    }
    if (distSi >= 8)
    {
        k4 = dist.at<double>(5);
        k5 = dist.at<double>(6);
        k6 = dist.at<double>(7);
    }
    if (distSi >= 12)
    {
        s1 = dist.at<double>(8);
        s2 = dist.at<double>(9);
        s3 = dist.at<double>(10);
        s4 = dist.at<double>(11);
    }
    if (distSi == 14)
    {
        tx = dist.at<double>(12);
        ty = dist.at<double>(13);
        if(R_tau.empty()){
            R = get_Rtau_distortionMat(tx, ty);
        }
        else
        {
            R = R_tau.getMat();
        }
        Ri = R.inv();
    }

    for (int i = 0; i < iters; i++)
    {
        if (!R.empty())
        {
            cv::Mat uvh = (cv::Mat_<double>(3, 1) << corrected.x, corrected.y, 1.);
            uvh = Ri * uvh;
            uvh /= uvh.at<double>(2);
            corrected.x = uvh.at<double>(0);
            corrected.y = uvh.at<double>(1);
        }
        r2 = corrected.x * corrected.x + corrected.y * corrected.y;
        r4 = r2 * r2;
        _2xy = 2.0 * corrected.x * corrected.y;
        rad_corr = (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2);
        delta[0] = p1 * _2xy + p2 * (r2 + 2.0 * corrected.x * corrected.x) + s1 * r2 + s2 * r4;
        delta[1] = p1 * (r2 + 2.0 * corrected.y * corrected.y) + p2 * _2xy + s3 * r2 + s4 * r4;
        corrected.x = (distorted_d.x - delta[0]) / rad_corr;
        corrected.y = (distorted_d.y - delta[1]) / rad_corr;
    }

    //proof
    cv::Point2d proofdist;
    r2 = corrected.x * corrected.x + corrected.y * corrected.y;
    r4 = r2 * r2;
    _2xy = 2.0 * corrected.x * corrected.y;
    rad_corr = (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2);
    delta[0] = p1 * _2xy + p2 * (r2 + 2.0 * corrected.x * corrected.x) + s1 * r2 + s2 * r4;
    delta[1] = p1 * (r2 + 2.0 * corrected.y * corrected.y) + p2 * _2xy + s3 * r2 + s4 * r4;
    if (!R.empty())
    {
        proofdist.x = corrected.x * rad_corr + delta[0];
        proofdist.y = corrected.y * rad_corr + delta[1];
        cv::Mat uvh = (cv::Mat_<double>(3, 1) << proofdist.x, proofdist.y, 1.);
        uvh = R * uvh;
        uvh /= uvh.at<double>(2);
        proofdist.x = uvh.at<double>(0) - distorted_d.x;
        proofdist.y = uvh.at<double>(1) - distorted_d.y;
    }else{
        proofdist.x = corrected.x * rad_corr + delta[0] - distorted_d.x;
        proofdist.y = corrected.y * rad_corr + delta[1] - distorted_d.y;
    }

    if (std::sqrt(proofdist.x * proofdist.x + proofdist.y * proofdist.y) > 0.25)
    {
        corrected = distorted_d;
        return false;
    }

    return true;
}

/* Get the distortion (rotation) matrix for the tilted sensor model distortion with parameters tau_x and tau_y.
 * In some cases, the image sensor may be tilted in order to focus an oblique plane in front of the camera (Scheimpflug principle). 
 * This can be useful for particle image velocimetry (PIV) or triangulation with a laser fan. 
 * The tilt causes a perspective distortion of distorted coordinates (radial, tangential, thin prism) in the camera coordinate system (K^-1 * [u v 1]^T).
 * For details see: https://docs.opencv.org/4.8.0/d9/d0c/group__calib3d.html#gga6ff34ec31356df0b3cef388a831bf4daac87f64a9ebe7c605bc8e4dd6e355528d
 *
 * double tx			Input  -> Distortion parameter tau_x
 * double ty			Input  -> Distortion parameter tau_y
 *
 * Return value:		Distortion (rotation) matrix for the tilted sensor model distortion
 */
cv::Mat get_Rtau_distortionMat(const double &tx, const double &ty){
    cv::Mat R_tau;
    const double tx_cos = std::cos(tx);
    const double ty_cos = std::cos(ty);
    const double tx_sin = std::sin(tx);
    const double ty_sin = std::sin(ty);
    R_tau = (cv::Mat_<double>(3, 3) << ty_cos, (tx_sin * ty_sin), (-ty_sin * tx_cos),
                0, tx_cos, tx_sin,
                ty_sin, (-ty_cos * tx_sin), (ty_cos * tx_cos));
    cv::Mat R1 = cv::Mat::eye(3, 3, CV_64FC1);
    R1.at<double>(0, 0) = R_tau.at<double>(2, 2);
    R1.at<double>(0, 2) = -R_tau.at<double>(0, 2);
    R1.at<double>(1, 1) = R_tau.at<double>(2, 2);
    R1.at<double>(1, 2) = -R_tau.at<double>(1, 2);
    R_tau = R1 * R_tau;
    return R_tau;
}

/* Distort a single homogeneous point provided in the image coordinate system ([u v 1]^T)
 *
 * Mat undistorted			Input  -> Homogeneous point coordinate in the image coordinate system
 * Mat dist			        Input  -> Distortion coefficients. The ordering of the coefficients
 *									  is compliant with the OpenCV library: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4]]]) or (k1,k2,k3)
 * Mat K_inv    			Input  -> Inverse camera matrix
 *
 * Return value:	        Distorted 2D coordinate in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y]
 */
cv::Mat distort(const cv::Mat &undistorted, cv::Mat &dist, cv::Mat &K_inv)
{
    CV_Assert(undistorted.rows * undistorted.cols == 3 && undistorted.type() == CV_64FC1);
    cv::Mat undistorted_tmp;
    if (undistorted.cols > undistorted.rows){
        undistorted_tmp = undistorted.t();
    }else{
        undistorted_tmp = undistorted;
    }
    cv::Mat x_cam = K_inv * undistorted_tmp;
    x_cam /= x_cam.at<double>(2);
    const cv::Point2d undistorted_d(x_cam.at<double>(0), x_cam.at<double>(1));
    cv::Point2d distorted = distort(undistorted_d, dist);
    return cv::Mat(distorted, true).reshape(1, 2);
}

/* Distort a single point 1/z * [x y] in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y]
 *
 * Mat undistorted			Input  -> 2D coordinate in the camera coordinate system
 * Mat dist			        Input  -> Distortion coefficients. The ordering of the coefficients
 *									  is compliant with the OpenCV library: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4]]]) or (k1,k2,k3)
 *
 * Return value:	        Distorted 2D coordinate in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y]
 */
cv::Mat distort(const cv::Mat &undistorted, cv::Mat &dist)
{
    CV_Assert(undistorted.rows * undistorted.cols == 2 && undistorted.type() == CV_64FC1);
    // const cv::Point2d undistorted_d(undistorted.at<double>(0), undistorted.at<double>(1));
    const cv::Point2d undistorted_d(undistorted);
    cv::Point2d distorted = distort(undistorted_d, dist);
    return cv::Mat(distorted, true).reshape(1, undistorted.rows);
}

/* Distort a single point 1/z * [x y] in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y]
 *
 * Point2d undistorted		Input  -> 2D coordinate in the image coordinate system
 * Mat dist			        Input  -> Distortion coefficients. The ordering of the coefficients
 *									  is compliant with the OpenCV library: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4]]]) or (k1,k2,k3)
 *
 * Return value:	        Distorted 2D coordinate in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y]
 */
cv::Point2d distort(const cv::Point2d &undistorted, cv::Mat &dist)
{
    CV_Assert(dist.cols <= 14 && dist.rows <= 14);
    const int distSi = dist.cols * dist.rows;
    CV_Assert(distSi != 4 && distSi != 6 && distSi != 7);

    double k1 = 0., k2 = 0., k3 = 0., k4 = 0., k5 = 0., k6 = 0., tan_corr_x = 0., tan_corr_y = 0.;
    double s1 = 0., s2 = 0., s3 = 0., s4 = 0.;
    double tx = 0., ty = 0.;
    cv::Mat R;
    const double r2 = undistorted.x * undistorted.x + undistorted.y * undistorted.y;
    if (distSi >= 1)
    {
        k1 = dist.at<double>(0);
    }
    if (distSi >= 2)
    {
        k2 = dist.at<double>(1);
    }
    if (distSi == 3)
    {
        k3 = dist.at<double>(4);
    }
    else if (distSi >= 5)
    {
        k3 = dist.at<double>(4);
        const double p1 = dist.at<double>(2);
        const double p2 = dist.at<double>(3);
        const double _2xy = 2.0 * undistorted.x * undistorted.y;
        tan_corr_x = p1 * _2xy + p2 * (r2 + 2.0 * undistorted.x * undistorted.x);
        tan_corr_y = p1 * (r2 + 2.0 * undistorted.y * undistorted.y) + p2 * _2xy;
    }
    if (distSi >= 8)
    {
        k4 = dist.at<double>(5);
        k5 = dist.at<double>(6);
        k6 = dist.at<double>(7);
    }
    const double r4 = r2 * r2;
    if (distSi >= 12)//thin prism distortion
    {
        s1 = dist.at<double>(8);
        s2 = dist.at<double>(9);
        s3 = dist.at<double>(10);
        s4 = dist.at<double>(11);
        tan_corr_x += s1 * r2 + s2 * r4;
        tan_corr_y += s3 * r2 + s4 * r4;
    }
    if (distSi == 14)//tilted sensor model distortion
    {
        tx = dist.at<double>(12);
        ty = dist.at<double>(13);
        R = get_Rtau_distortionMat(tx, ty);
    }

    const double r6 = r4 * r2;
    const double rad_corr = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);

    cv::Point2d p_dist(undistorted.x * rad_corr + tan_corr_x, undistorted.y * rad_corr + tan_corr_y);
    if (!R.empty())
    {
        cv::Mat uvh = (cv::Mat_<double>(3, 1) << p_dist.x, p_dist.y, 1.);
        uvh = R * uvh;
        uvh /= uvh.at<double>(2);
        p_dist.x = uvh.at<double>(0);
        p_dist.y = uvh.at<double>(1);
    }

    return p_dist;
}

/* Convert a single point in the image coordinate system ([u v]^T) into camera coordinate system (K^-1 * [u v 1]^T) and optionally undistort it
 *
 * Point2d p_img		    Input  -> 2D coordinate in the image coordinate system [u v]
 * Mat Kinv		            Input  -> Inverse camera matrix
 * bool undistort		    Input  -> Specify, if undistortion should be performed
 * Mat distortion			Input  -> Distortion coefficients. The ordering of the coefficients
 *									  is compliant with the OpenCV library: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4]]]) or (k1,k2,k3)
 * Point2d p_out            Output -> (Undistorted) 2D coordinate in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y]
 *
 * Return value:	        true:  Everything ok.
 *							false: Undistortion failed
 */
bool toCamCoordsUndistort(const cv::Point2d &p_img, const cv::Mat Kinv, const bool undistort, const cv::Mat distortion, cv::Point2d &p_out)
{
    cv::Mat ph = (cv::Mat_<double>(3, 1) << p_img.x, p_img.y, 1.0);
    ph = Kinv * ph;
    ph /= ph.at<double>(2);
    ph.resize(2);
    if (undistort)
    {
        if (!LensDist_Oulu(ph, ph, distortion))
        {
            return false;
        }
    }
    p_out = cv::Point2d(ph);
    return true;
}

/* This function undistorts multiple corresponding points from 2 cameras in the camera coordinate system [x y z] = (K^-1 * [u v 1]^T) -> 1/z * [x y] 
 * using the methode from the Oulu University. A mask is provided for correspondences not successfully undistorted.
 *
 * vector<Point2f> points1		I/O    -> Distorted points of cam 1 input and undistorted points output
 * vector<Point2f> points2		I/O    -> Distorted points of cam 1 input and undistorted points output
 * cv::Mat dist1				Input  -> Distortion coefficients for cam 1. The ordering of the coefficients
 *									      is compliant with the OpenCV library: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4]]]) or (k1,k2,k3)
 * cv::Mat dist1				Input  -> Distortion coefficients for cam 2. The ordering of the coefficients
 *									      is compliant with the OpenCV library: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4]]]) or (k1,k2,k3)
 * OutputArray mask				Output  -> Optional mask with successful/failed undistorted points
 *
 * Return value:				true:  Everything ok.
 *								false: Undistortion failed for too many correspondences
 */
bool Remove_LensDist2(std::vector<cv::Point2f> &points1,
                      std::vector<cv::Point2f> &points2,
                      const cv::Mat &dist1,
                      const cv::Mat &dist2,
                      cv::OutputArray mask)
{
    CV_Assert(points1.size() == points2.size());

    if (nearZero(sum(dist1)[0]) && nearZero(sum(dist2)[0]))
    {
        return true;
    }

    cv::Mat R_tau;
    int nrDistPars = dist1.cols * dist1.rows;
    if(nrDistPars == 14){
        const double tx = dist1.at<double>(12);
        const double ty = dist1.at<double>(13);
        R_tau = get_Rtau_distortionMat(tx, ty);
    }

    vector<cv::Point2f> distpoints1, distpoints2;
    int n1, n = static_cast<int>(points1.size());
    cv::Mat mask0 = cv::Mat::ones(1, n, CV_8UC1);

    //Remove the lens distortion on the normalized coordinates (camera coordinate system)
    distpoints1 = points1;
    distpoints2 = points2;
    for (int i = 0; i < n; i++)
    {
        if (!LensDist_Oulu(distpoints1.at(i), points1.at(i), dist1, R_tau))
        {
            mask0.at<bool>(i) = false;
            continue;
        }
        if (!LensDist_Oulu(distpoints2.at(i), points2.at(i), dist2, R_tau))
            mask0.at<bool>(i) = false;
    }

    if(mask.needed()){
        mask.create(1, n, CV_8UC1);
        cv::Mat mask_ = mask.getMat();
        mask0.copyTo(mask_);
    }

    n1 = cv::countNonZero(mask);

    float rat = static_cast<float>(n1) / static_cast<float>(n);

    if (rat < 0.1f)
        return false;

    if (rat < 0.75f)
        cout << "There is a problem with the distortion parameters! Check your internal calibration parameters!" << endl;

    return true;
}

/* Calculates the difference (roation angle) between two rotation matrices and the distance between
 * two 3D translation vectors back-rotated by the matrices R1 and R2 (therefore, this error represents
 * the full error caused by the different rotations and translations)
 *
 * Mat R1				Input  -> First rotation matrix (e.g. result from pose estimation)
 * Mat R2				Input  -> Second rotation matrix (e.g. from offline calibration)
 * Mat t1				Input  -> First 3D (translation) vector (e.g. result from pose estimation)
 * Mat t2				Input  -> Second 3D (translation) vector (e.g. from offline calibration)
 * double rdiff			Output -> Rotation angle (from Angle-axis-representation) between the two rotations
 * double tdiff			Output -> Distance between the two translation vectors back-rotated by the matrices
 *								  R and Rcalib
 * bool printDiff		Input  -> If true, the results are printed to std::out [Default=false]
 *
 * Return value:		none
 */
void compareRTs(const cv::Mat& R1, const cv::Mat& R2, cv::Mat t1, cv::Mat t2, double *rdiff, double *tdiff, bool printDiff)
{
    Eigen::Matrix3d R1e, R2e;
    Eigen::Vector4d r1quat, r2quat;
    Eigen::Vector3d t1e, t2e;

    cv::cv2eigen(R1,R1e);
    cv::cv2eigen(R2,R2e);

    MatToQuat(R1e, r1quat);
    MatToQuat(R2e, r2quat);

    t1e << t1.at<double>(0), t1.at<double>(1), t1.at<double>(2);
    t2e << t2.at<double>(0), t2.at<double>(1), t2.at<double>(2);

    getRTQuality(r1quat, r2quat, t1e, t2e, rdiff, tdiff);

    if(printDiff)
    {
        cout << "Angle between rotation matrices: " << *rdiff / PI * 180.0 << char(248) << endl;
        cout << "Distance between translation vectors: " << *tdiff << endl;
    }
}

/* Return rotation R_out and translation t_out resulting from Essential matrix decomposition which is most similar to given pose R_ref & t_ref
 *
 * Mat E				Input  -> Essential matrix
 * Mat R_ref			Input  -> Given rotation matrix
 * Mat t_ref			Input  -> Given translation vector
 * Mat R_out			Output -> Resulting rotation matrix from E decomposition
 * Mat t_out			Output -> Resulting translation vector from E decomposition
 *
 * Return value:		none
 */
void getNearestRTfromEDecomposition(const cv::Mat &E, const cv::Mat &R_ref, const cv::Mat &t_ref, cv::Mat &R_out, cv::Mat &t_out)
{
    cv::Mat R12_1, R12_2, t12;
    cv::decomposeEssentialMat(E, R12_1, R12_2, t12);
    double norm_t = cv::norm(t12);
    if (!nearZero(1.0 - norm_t))
    {
        t12 /= norm_t;
    }
    std::vector<std::pair<cv::Mat, cv::Mat>> possible_RTs;
    possible_RTs.emplace_back(std::make_pair(R12_1, t12));
    possible_RTs.emplace_back(std::make_pair(R12_1, -1. * t12));
    possible_RTs.emplace_back(std::make_pair(R12_2, t12));
    possible_RTs.emplace_back(std::make_pair(R12_2, -1. * t12));
    std::map<double, size_t> rt_diffs;
    for (size_t i = 0; i < 4; i++)
    {
        double r_diff, t_diff;
        compareRTs(R_ref, possible_RTs.at(i).first, t_ref, possible_RTs.at(i).second, &r_diff, &t_diff);
        r_diff /= M_PI;
        t_diff /= 1.7320508; //squareroot of 3
        rt_diffs.emplace(r_diff + t_diff, i);
    }
    const size_t bestFit_idx = rt_diffs.begin()->second;
    possible_RTs.at(bestFit_idx).first.copyTo(R_out);
    possible_RTs.at(bestFit_idx).second.copyTo(t_out);
}

/* Calculation of the rectifying matrices based on the extrinsic and intrinsic camera parameters. There are 2 methods available
 * for calculating the rectifying matrices: 1st method (Default: globRectFunct=true): A. Fusiello, E. Trucco and A. Verri: "A compact
 * algorithm for rectification of stereo pairs", 2000. This methode can be used for the rectification of cameras with a general form
 * of the extrinsic parameters. 2nd method (globRectFunct=false): A slightly changed version (to be more robust) of the OpenCV
 * stereoRectify-function for stereo cameras with no or only a small differnce in the vertical position and small rotations (the cameras
 * should be nearly parallel). Moreover, an new camera matrix is calculated based on the image areas. Therefore, alpha specifies if all
 * valid pixels, only valid pixels (no black areas) or something inbetween should be present in the rectified images.
 *
 * InputArray R							Input  -> Rotation matrix
 * InputArray t							Input  -> Translation matrix
 * InputArray K1						Input  -> Input camera matrix of the left camera
 * InputArray K2						Input  -> Input camera matrix of the right camera
 * InputArray distcoeffs1				Input  -> Distortion coeffitients of the left camera
 * InputArray distcoeffs2				Input  -> Distortion coeffitients of the right camera
 * Size imageSize						Input  -> Size of the input image
 * OutputArray Rect1					Output -> Rectification matrix for the left camera
 * OutputArray Rect2					Output -> Rectification matrix for the right camera
 * OutputArray K1new					Output -> New camera matrix for the left camera (equal to K2new if globRectFunct=true)
 * OutputArray K2new					Output -> New camera matrix for the right camera (equal to K1new if globRectFunct=true)
 * double alpha							Input  -> Free scaling parameter. If it is -1 or absent [Default=-1], the function performs the default
 *												  scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
 *												  images are zoomed and shifted so that only valid pixels are visible (no black areas after
 *												  rectification). alpha=1 means that the rectified image is decimated and shifted so that all
 *												  the pixels from the original images from the cameras are retained in the rectified images
 *												  (no source image pixels are lost). Obviously, any intermediate value yields an intermediate
 *												  result between those two extreme cases.
 * bool globRectFunct					Input  -> Used method for rectification [Default=true]. If true, the method from A. Fusiello, E. Trucco
 *												  and A. Verri: "A compact algorithm for rectification of stereo pairs", 2000. This methode can
 *												  be used for the rectification of cameras with a general form of the extrinsic parameters.
 *												  If false, a slightly changed version (to be more robust) of the OpenCV stereoRectify-function
 *												  is used. This method can be used for stereo cameras with only a small differnce in the
 *												  vertical position and small rotations only (the cameras should be nearly parallel).
 * Size newImgSize						Input  -> Optional new image resolution after rectification. The same size should be passed to
 *												  initUndistortRectifyMap() (see the stereo_calib.cpp sample in OpenCV samples directory).
 *												  When (0,0) is passed (default), it is set to the original imageSize . Setting it to larger
 *												  value can help you preserve details in the original image, especially when there is a big
 *												  radial distortion.
 * Rect *roi1							Output -> Optional output rectangles inside the rectified images where all the pixels are valid.
 *												  If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller.
 * Rect *roi2							Output -> Optional output rectangles inside the rectified images where all the pixels are valid.
 *												  If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller.
 * OutputArray P1new					Output -> Optional new projection matrix for the left camera (only available if globRectFunct=true)
 * OutputArray P2new					Output -> Optional new projection matrix for the right camera (only available if globRectFunct=true)
 *
 * Return value:						0 :		Everything ok
 */
int getRectificationParameters(cv::InputArray R,
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
                              double alpha,
                              bool globRectFunct,
                              const cv::Size& newImgSize,
                              cv::Rect *roi1,
                              cv::Rect *roi2,
                              cv::OutputArray P1new,
                              cv::OutputArray P2new)
{
    CV_Assert(!R.empty() && !t.empty() && !K1.empty() && !K2.empty() &&
              (!P1new.needed() || globRectFunct) && (!P2new.needed() || globRectFunct));

    Mat _R, _t;
    Mat R1, R2, P1, P2, Q, t_tmp;

    _R = R.getMat();
    _t = t.getMat();

    double t_norm = normFromVec(_t);
    t_tmp = _t.clone();
    if(std::abs(t_norm-1.0) > 1e-4)
        t_tmp /= t_norm;

    if(globRectFunct)
    {
        //rectifyFusiello(K1, K2, _R.t(), -1.0 * _R.t() * t_tmp, distcoeffs1, distcoeffs2, imageSize, Rect1, Rect2, K1new, alpha, newImgSize, roi1, P1new, P2new);
        rectifyFusiello(K1, K2, _R, t_tmp, distcoeffs1, distcoeffs2, imageSize, Rect1, Rect2, K1new, alpha, newImgSize, roi1, P1new, P2new);

        Mat _K2new, _K1new;
        if(K2new.empty())
        {
            K2new.create(3, 3, K1new.type());
        }
        _K2new = K2new.getMat();
        _K1new = K1new.getMat();
        _K1new.copyTo(_K2new);

        roi2 = roi1;
    }
    else
    {
        //stereoRectify2(K1, distcoeffs1, K2, distcoeffs2, imageSize, _R.t(), -1.0 * _R.t() * t_tmp, Rect1, Rect2, K1new, K2new, Q, /*0*/CV_CALIB_ZERO_DISPARITY, alpha, newImgSize, roi1, roi2);
        stereoRectify2(K1, distcoeffs1, K2, distcoeffs2, imageSize, _R, t_tmp, Rect1, Rect2, K1new, K2new, Q, /*0*/cv::CALIB_ZERO_DISPARITY, alpha, newImgSize, roi1, roi2);
    }

    return 0;
}

/* Calculation of the rectifying matrices based on the extrinsic and intrinsic camera parameters based on the methode from
 * A. Fusiello, E. Trucco and A. Verri: "A compact algorithm for rectification of stereo pairs", 2000. This methode can be used
 * for the rectification of cameras with a general form of the extrinsic parameters. Moreover, an new camera matrix is calculated
 * based on the image areas. Therefore, alpha specifies if all valid pixels, only valid pixels (no black areas) or something
 * inbetween should be present in the rectified images.
 *
 * InputArray K1						Input  -> Input camera matrix of the left camera
 * InputArray K2						Input  -> Input camera matrix of the right camera
 * InputArray R							Input  -> Rotation matrix
 * InputArray t							Input  -> Translation matrix
 * InputArray distcoeffs1				Input  -> Distortion coeffitients of the left camera
 * InputArray distcoeffs2				Input  -> Distortion coeffitients of the right camera
 * Size imageSize						Input  -> Size of the input image
 * OutputArray Rect1					Output -> Rectification matrix for the left camera
 * OutputArray Rect2					Output -> Rectification matrix for the right camera
 * OutputArray K12new					Output -> New camera matrix for both cameras
 * double alpha							Input  -> Free scaling parameter. If it is -1 or absent, the function performs the default scaling.
 *												  Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images
 *												  are zoomed and shifted so that only valid pixels are visible (no black areas after
 *												  rectification). alpha=1 means that the rectified image is decimated and shifted so that all
 *												  the pixels from the original images from the cameras are retained in the rectified images
 *												  (no source image pixels are lost). Obviously, any intermediate value yields an intermediate
 *												  result between those two extreme cases.
 * Size newImgSize						Input  -> New image resolution after rectification. The same size should be passed to
 *												  initUndistortRectifyMap() (see the stereo_calib.cpp sample in OpenCV samples directory).
 *												  When (0,0) is passed (default), it is set to the original imageSize . Setting it to larger
 *												  value can help you preserve details in the original image, especially when there is a big
 *												  radial distortion.
 * Rect *roi1							Output -> Optional output rectangles inside the rectified images where all the pixels are valid.
 *												  If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller.
 * OutputArray P1new					Output -> Optional new projection matrix for the left camera
 * OutputArray P2new					Output -> Optional new projection matrix for the right camera
 *
 * Return value:						0 :		Everything ok
 */
int rectifyFusiello(cv::InputArray K1, cv::InputArray K2, cv::InputArray R, cv::InputArray t,
                    cv::InputArray distcoeffs1, cv::InputArray distcoeffs2, const cv::Size& imageSize,
                    cv::OutputArray Rect1, cv::OutputArray Rect2, cv::OutputArray K12new,
                    double alpha, cv::Size newImgSize, cv::Rect *roi1, cv::OutputArray P1new,
                    cv::OutputArray P2new)
{
    Mat c1, c2, v1, v2, v3, Rv, Pn1, Pn2, Rect1_, Rect2_, K1_, K2_, R_, t_, dk1_, dk2_;

    K1_ = K1.getMat();
    K2_ = K2.getMat();
    R_ = R.getMat();
    t_ = t.getMat();
    dk1_ = distcoeffs1.getMat();
    dk2_ = distcoeffs2.getMat();
    auto nx = (double)imageSize.width, ny = (double)imageSize.height;


    Mat Po1, Po2;
    //Calculate projection matrix of camera 1
    Po1 = cv::Mat::zeros(3,4,K1_.type());
    Po1.colRange(0,3) = cv::Mat::eye(3,3,K1_.type());
    Po1 = K1_ * Po1;

    //Calculate projection matrix of camera 2
    Po2 = cv::Mat(3,4,K2_.type());
    R_.copyTo(Po2.colRange(0,3));
    t_.copyTo(Po2.col(3));
    Po2 = K2_ * Po2;

    /*Mat Q, Ucv, Bcv, R1, t1, K11, R2, t2, K21;
    Eigen::Matrix3d U, B, Qe;
    Eigen::MatrixXd test, P;

    //Recalculate extrinsic and intrinsic paramters from camera 1
    cv::invert(Po1.colRange(0,3),Q);
    cv::cv2eigen(Q,Qe);
    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(Qe);
    //Eigen::HouseholderQR<Eigen::Matrix3d> qr(Qe);
    U = qr.householderQ();
    B = U.inverse()*Qe;
    cv::eigen2cv(U,Ucv);
    cv::eigen2cv(B,Bcv);
    invert(Ucv,R1);
    t1 = Bcv*Po1.col(3);
    invert(Bcv, K11);
    K11 = K11 / K11.at<double>(2,2);

    //Recalculate extrinsic and intrinsic paramters from camera 2
    cv::invert(Po2.colRange(0,3),Q);
    cv::cv2eigen(Q,Qe);
    qr.compute(Qe);
    B = qr.matrixQR().triangularView<Eigen::Upper>();
    U = qr.householderQ();
    P = qr.colsPermutation(); //Permutation matrix must be integrated before it works
    test = U*B;
    cv::eigen2cv(U,Ucv);
    cv::eigen2cv(B,Bcv);
    invert(Ucv,R2);
    t2 = Bcv*Po2.col(3);
    invert(Bcv, K21);
    K21 = K21 / K21.at<double>(2,2);*/

    unsigned int idx = fabs(t_.at<double>(0)) > fabs(t_.at<double>(1)) ? 0 : 1;
    auto fc_new = DBL_MAX;
    vector<cv::Point2d> cc_new = vector<cv::Point2d>(2, cv::Point2d(0,0));

    for(int k = 0; k < 2; k++ ) {
        Mat A = k == 0 ? K1_ : K2_;
        Mat Dk = k == 0 ? dk1_ : dk2_;
        double dk1 = Dk.empty() ? 0 : Dk.at<double>(0);
        double fc = A.at<double>(idx^1,idx^1);
        if( dk1 < 0 ) {
            fc *= 1.0 + dk1 * (nx * nx + ny * ny) / (4.0 * fc * fc);
        }
        fc_new = std::min(fc_new, fc);
    }

    newImgSize = newImgSize.width * newImgSize.height != 0 ? newImgSize : imageSize;
    cc_new[0].x = cc_new[1].x = (nx - 1.0) / 2.0;
    cc_new[0].y = cc_new[1].y = (ny - 1.0) / 2.0;
    cc_new[0].x = cc_new[1].x = (double)newImgSize.width * cc_new[0].x / (double)imageSize.width;
    cc_new[0].y = cc_new[1].y = (double)newImgSize.height * cc_new[0].y/ (double)imageSize.height;

    Mat Knew = (Mat_<double>(3, 3) << fc_new, 0, cc_new[0].x,
                                      0, fc_new, cc_new[0].y,
                                      0, 0, 1);

    //The rectification is done using the algorithm from A. Fusiello, E. Trucco and A. Verri: "A compact algorithm for rectification of stereo pairs", 2000

    //Calculate optical centers from unchanged cameras
    c1 = -1.0 * K1_.inv() * Po1.col(3);
    c2 = -1.0 * R_.t() * K2_.inv() * Po2.col(3);

    /*c1 = -1.0 * Po1.colRange(0,3).inv() * Po1.col(3);
    c2 = -1.0 * Po2.colRange(0,3).inv() * Po2.col(3);*/

    //New x axis (=direction of baseline)
    v1 = c2 - c1;//c1 - c2;

    //New y axis (orthogonal to new x and old z)
    Mat r1_tmp = (Mat_<double>(3, 1) << 0, 0, 1.0);
    v2 = r1_tmp.cross(v1);

    //New z axis (orthogonal to baseline and y)
    v3 = v1.cross(v2);

    //New extrinsic parameters (translation is left unchanged)
    Rv = Mat(3,3,CV_64FC1);
    Rv.row(0) = v1.t()/cv::norm(v1);
    Rv.row(1) = v2.t()/cv::norm(v2);
    Rv.row(2) = v3.t()/cv::norm(v3);

    //Calc new camera matrices
    Mat K1new, K2new;
    K2_.copyTo(K1new);
    K2_.copyTo(K2new);
    K1new.at<double>(0,1) = 0.0;
    K2new.at<double>(0,1) = 0.0;

    //New projection matrices
    Mat Ptmp = Mat(3,4,CV_64FC1);
    Rv.copyTo(Ptmp.colRange(0,3));
    Ptmp.col(3) = -1.0 * Rv * c1;
    Pn1 = K1new * Ptmp;
    Ptmp.col(3) = -1.0 * Rv * c2;
    Pn2 = K2new * Ptmp;

    //Rectifying image transformation
    Rect1_ = Pn1.colRange(0,3) * Po1.colRange(0,3).inv();
    Rect2_ = Pn2.colRange(0,3) * Po2.colRange(0,3).inv();

    //Center left image
    double _imc[3] = {(nx - 1.0) / 2.0, (ny - 1.0) / 2.0, 1.0};
    double _imcr1[3], _imcr2[3];
    Mat imcr1 = Mat(3,1,CV_64FC1,_imcr1);
    Mat imcr2 = Mat(3,1,CV_64FC1,_imcr2);
    Mat imc = Mat(3,1,CV_64FC1,_imc);
    imcr1 = Rect1_ * imc;
    imcr1 = imcr1 / imcr1.at<double>(2);
    Mat d_imc1 = imc - imcr1;

    //Center right image
    imcr2 = Rect2_ * imc;
    imcr2 = imcr2 / imcr2.at<double>(2);
    Mat d_imc2 = imc - imcr2;
    //d_imc1.at<double>(1) = d_imc2.at<double>(1);

    //Take the mean from the shifts
    d_imc1 = (d_imc1 + d_imc2) / 2.0;

    //Recalculate camera matrix
    Knew.rowRange(0,2).col(2) = Knew.rowRange(0,2).col(2) + d_imc1.rowRange(0,2);
    //K1new.rowRange(0,2).col(2) = K1new.rowRange(0,2).col(2) + d_imc1.rowRange(0,2);
    //K2new.rowRange(0,2).col(2) = K2new.rowRange(0,2).col(2) + d_imc2.rowRange(0,2);

    //New projection matrices
    Ptmp.col(3) = -1.0 * Rv * c1;
    Pn1 = Knew * Ptmp;
    Ptmp.col(3) = -1.0 * Rv * c2;
    Pn2 = Knew * Ptmp;

    //Rectifying image transformation
    Rect1_ = Pn1.colRange(0,3) * Po1.colRange(0,3).inv();
    Rect2_ = Pn2.colRange(0,3) * Po2.colRange(0,3).inv();

    //Calculate new camera matrix like transformation matrix to asure the right image size
    imcr1 = Rect1_ * imc;
    imcr1 = imcr1 / imcr1.at<double>(2);
    imcr2 = Rect2_ * imc;
    imcr2 = imcr2 / imcr2.at<double>(2);

    //double scaler = 0, ro2 = _imc[0]*_imc[0]+_imc[1]*_imc[1];
    //double xmin = DBL_MAX, xmax = -1.0*DBL_MAX, ymin = DBL_MAX, ymax = -1.0*DBL_MAX;
    //for(int i = 0; i < 4; i++ )
    //{
    //	int j = (i<2) ? 0 : 1;
    //	double r_tmp1, r_tmp2, _corners[3], _corners1[3];
    //	Mat corners = Mat(3,1,CV_64FC1,_corners);
    //	Mat corners1 = Mat(3,1,CV_64FC1,_corners1);
    //	_corners[0] = (double)((i % 2)*(nx-1));
    //	_corners[1] = (double)(j*(ny-1));
    //	_corners[2] = 1.0;
    //	corners1 = Rect1_ * corners;
    //	corners1 /= _corners1[2];
    //	xmin = std::min(xmin,_corners1[0]);
    //	ymin = std::min(ymin,_corners1[1]);
    //	xmax = std::max(xmax,_corners1[0]);
    //	ymax = std::max(ymax,_corners1[1]);
    //	r_tmp1 = _corners1[0] - _imcr1[0];
    //	r_tmp1 *= r_tmp1;
    //	r_tmp2 = _corners1[1] - _imcr1[1];
    //	r_tmp2 *= r_tmp2;
    //	scaler += std::sqrt(ro2/(r_tmp1 + r_tmp2));
    //	corners1 = Rect2_ * corners;
    //	corners1 /= _corners1[2];
    //	xmin = std::min(xmin,_corners1[0]);
    //	ymin = std::min(ymin,_corners1[1]);
    //	xmax = std::max(xmax,_corners1[0]);
    //	ymax = std::max(ymax,_corners1[1]);
    //	r_tmp1 = _corners1[0] - _imcr2[0];
    //	r_tmp1 *= r_tmp1;
    //	r_tmp2 = _corners1[1] - _imcr2[1];
    //	r_tmp2 *= r_tmp2;
    //	scaler += std::sqrt(ro2/(r_tmp1 + r_tmp2));
    //}
    //scaler /= 8;

    ////Scale the rectifying matrix
    //Mat Kst = (Mat_<double>(3, 3) << scaler, 0, 0,
    //								0, scaler, 0,
    //								0, 0, 1);

    //Rect1_ = Kst * Rect1_;
    //Rect2_ = Kst * Rect2_;

    //Recalculate camera matrix
    /*Knew.at<double>(0,2) = scaler*(xmax-xmin)/2.0;
    Knew.at<double>(1,2) = scaler*(ymax-ymin)/2.0;*/

    //Take the 2D rectifying transformations into 3D space
    Rect1_ = Knew.inv() * (Rect1_ * K1_);
    Rect2_ = Knew.inv() * (Rect2_ * K2_);

    //Calculate optimal new camera matrix (extracted from OpenCV)
//    CvPoint2D64f cc_tmp = {DBL_MAX, DBL_MAX};
    cv::Mat _cameraMatrix1 = K1_, _cameraMatrix2 = K2_;
    cv::Mat _distCoeffs1 = dk1_, _distCoeffs2 = dk2_;
    double _z[3] = {0,0,0}, _pp[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    cv::Mat Z = cv::Mat(3, 1, CV_64FC1, _z);
    cv::Rect_<float> inner1, inner2, outer1, outer2;
    cv::Mat pp  = cv::Mat(3, 3, CV_64FC1, _pp);
    double cx1_0, cy1_0, cx1, cy1, s;

    vector<cv::Point2f> _pts1 = vector<cv::Point2f>(4, cv::Point2f(0, 0));
    vector<cv::Point2f> _pts2 = vector<cv::Point2f>(4, cv::Point2f(0, 0));
    cv::Mat pts1 = cv::Mat(_pts1, false);
    cv::Mat pts2 = cv::Mat(_pts2, false);
    for(int k = 0; k < 2; k++ )
    {
        const cv::Mat A = k == 0 ? _cameraMatrix1 : _cameraMatrix2;
        const cv::Mat Dk = k == 0 ? _distCoeffs1 : _distCoeffs2;
        vector<cv::Point2f>& _pts = k == 0 ? _pts1 : _pts2;
        cv::Mat& pts = k == 0 ? pts1 : pts2;

        for(int i = 0; i < 4; i++ )
        {
            int j = (i<2) ? 0 : 1;
            _pts[i].x = (float)((i % 2)*(nx-1));
            _pts[i].y = (float)(j*(ny-1));
        }
        {
            int valid_nr;
            double reduction = 0.01;
            cv::Point2d imgCent = cv::Point2d(nx / 2.0, ny / 2.0);
            vector<cv::Point2f> _pts_sv = _pts;
            do
            {
                Mat mask;
                cvUndistortPoints2( pts, pts, A, &Dk, nullptr, nullptr, mask );
                valid_nr = cv::countNonZero(mask);
                if(valid_nr < 4)
                {
                    for(int i = 0; i < 4; i++ )
                    {
                        int j = (i<2) ? 0 : 1;
                        if(!mask.at<bool>(i))
                        {
                            _pts_sv[i].x = _pts[i].x = (float)((floor((1.0 - reduction) * ((i % 2) * nx - imgCent.x)) + imgCent.x - 1.0));
                            _pts_sv[i].y = _pts[i].y = (float)((floor((1.0 - reduction) * (j * ny - imgCent.y)) + imgCent.y - 1.0));
                        }
                        else
                        {
                            _pts[i].x = _pts_sv[i].x;
                            _pts[i].y = _pts_sv[i].y;
                        }
                    }
                    reduction += 0.01;
                }
            }while((valid_nr < 4) && (reduction < 0.25));

            if(reduction >= 0.25)
            {
                Mat mask;
                cvUndistortPoints2( pts, pts, A, nullptr, nullptr, nullptr, mask );
            }
        }
    }

    cv::Mat R1_ = Rect1_.clone(), R2_ = Rect2_.clone();
    for(int k = 0; k < 2; k++ )
    {
//        vector<cv::Point3f> _pts_3 = vector<cv::Point3f>(4, cv::Point3f(0, 0, 0));
//        vector<cv::Point2f> _pts_tmp = vector<cv::Point2f>(4, cv::Point2f(0, 0));
        cv::Mat pts_3, pts12; //= cv::Mat(_pts_3, false);
        cv::Mat& pts = k == 0 ? pts1 : pts2;
        cv::Mat pts_tmp;// = cv::Mat(_pts_tmp, false);
        pts.convertTo(pts12, CV_64FC1);
        cv::convertPointsToHomogeneous( pts12, pts_3 );

        //Change camera matrix to have cc=[0,0] and fc = fc_new
        double _a_tmp[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        cv::Mat A_tmp  = cv::Mat(3, 3, CV_64FC1, _a_tmp);
        _a_tmp[0][0]=fc_new;
        _a_tmp[1][1]=fc_new;
        _a_tmp[0][2]=0.0;
        _a_tmp[1][2]=0.0;
        _a_tmp[2][2]=1.0;
        cv::projectPoints(pts_3, k == 0 ? R1_ : R2_, Z, A_tmp, cv::noArray(), pts_tmp );
        cv::Scalar avg = cv::mean(pts_tmp);
        cc_new[k].x = (nx-1)/2 - avg[0];
        cc_new[k].y = (ny-1)/2 - avg[1];
    }

    // For simplicity, set the principal points for both cameras to be the average
    // of the two principal points
    cc_new[0].x = cc_new[1].x = (cc_new[0].x + cc_new[1].x)*0.5;
    cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;

    pp.setTo(cv::Scalar::all(0));
    _pp[0][0] = _pp[1][1] = fc_new;
    _pp[0][2] = cc_new[0].x;
    _pp[1][2] = cc_new[0].y;
    _pp[2][2] = 1;

    alpha = std::min(alpha, 1.);

    icvGetRectanglesV0(&_cameraMatrix1, &_distCoeffs1, &R1_, &pp, imageSize, inner1, outer1 );
    icvGetRectanglesV0(&_cameraMatrix2, &_distCoeffs2, &R2_, &pp, imageSize, inner2, outer2 );

    cx1_0 = cc_new[0].x;
    cy1_0 = cc_new[0].y;
    cx1 = (double)newImgSize.width * cx1_0 / (double)imageSize.width;
    cy1 = (double)newImgSize.height * cy1_0/ (double)imageSize.height;
    s = 1.;

    if( alpha >= 0 )
    {
        double s0 = std::max(std::max(std::max((double)cx1/(cx1_0 - inner1.x), (double)cy1/(cy1_0 - inner1.y)),
                                        (double)(newImgSize.width - cx1)/(inner1.x + inner1.width - cx1_0)),
                                (double)(newImgSize.height - cy1)/(inner1.y + inner1.height - cy1_0));

        double s1 = std::min(std::min(std::min((double)cx1/(cx1_0 - outer1.x), (double)cy1/(cy1_0 - outer1.y)),
                                        (double)(newImgSize.width - cx1)/(outer1.x + outer1.width - cx1_0)),
                                (double)(newImgSize.height - cy1)/(outer1.y + outer1.height - cy1_0));

        s = s0*(1.0 - alpha) + s1*alpha;
        if((s > 2.0) || (s < 0.5)) //added to OpenCV function
            s = 1.0;
    }

    fc_new *= s;
    cc_new[0] = cv::Point2d(cx1, cy1);

    Knew = (Mat_<double>(3, 3) << fc_new, 0, cc_new[0].x,
                                    0, fc_new, cc_new[0].y,
                                    0, 0, 1.0);

    if(roi1)
    {
        //Intersection of rectangles
        *roi1 = cv::Rect((int)std::ceil(((double)inner1.x - cx1_0) * s + cx1),
                         (int)std::ceil(((double)inner1.y - cy1_0) * s + cy1),
                         (int)std::floor((double)inner1.width * s),
                         (int)std::floor((double)inner1.height * s))
            & cv::Rect(0, 0, newImgSize.width, newImgSize.height);
    }

    Rect1.create(3,3,CV_64FC1);
    Mat Rect1_tmp = Rect1.getMat();
    Rect1_.copyTo(Rect1_tmp);

    Rect2.create(3,3,CV_64FC1);
    Rect1_tmp = Rect2.getMat();
    Rect2_.copyTo(Rect1_tmp);

    K12new.create(3,3,CV_64FC1);
    Rect1_tmp = K12new.getMat();
    Knew.copyTo(Rect1_tmp);

    if(P1new.needed())
    {
        Ptmp.col(3) = -1.0 * Rv * c1;
        Pn1 = Knew * Ptmp;
        P1new.create(3,4,CV_64FC1);
        Rect1_tmp = P1new.getMat();
        Pn1.copyTo(Rect1_tmp);
    }

    if(P2new.needed())
    {
        Ptmp.col(3) = -1.0 * Rv * c2;
        Pn2 = Knew * Ptmp;
        P2new.create(3,4,CV_64FC1);
        Rect1_tmp = P2new.getMat();
        Pn2.copyTo(Rect1_tmp);
    }

    return 0;
}

/* OpenCV interface function for cvStereoRectify. This code was copied from the OpenCV without changes. Check the OpenCV documentation
 * for more information. This function was copied to be able to change a few details on the core functionality of the rectification -
 * especiallly the undistortion functionality to estimate the new virtual cameras.
 *
 * InputArray _cameraMatrix1			Input  -> Camera matrix of the first (left) camera
 * InputArray _distCoeffs1				Input  -> Distortion parameters of the first camera
 * InputArray _cameraMatrix2			Input  -> Camera matrix of the second (right) camera
 * InputArray _distCoeffs2				Input  -> Distortion parameters of the second camera
 * Size imageSize						Input  -> Size of the original image
 * InputArray _Rmat						Input  -> Rotation matrix to specify the 3D rotation from the first to the second camera
 * InputArray _Tmat						Input  -> Translation vector specifying the translational direction from the first to the
 *												  second camera (the norm of the vector must be 1.0)
 * OutputArray _Rmat1					Output -> Rectification transform (rotation matrix) for the first camera
 * OutputArray _Rmat2					Output -> Rectification transform (rotation matrix) for the second camera
 * OutputArray _Pmat1					Output -> Projection matrix in the new (rectified) coordinate systems for the first camera
 * OutputArray _Pmat2					Output -> Projection matrix in the new (rectified) coordinate systems for the second camera
 * OutputArray _Qmat					Output -> Output 4x4 disparity-to-depth mapping matrix (see reprojectImageTo3D() ).
 * int flags							Input  -> Operation flags that may be zero or CV_CALIB_ZERO_DISPARITY . If the flag is set,
 *												  the function makes the principal points of each camera have the same pixel
 *												  coordinates in the rectified views. And if the flag is not set, the function may
 *												  still shift the images in the horizontal or vertical direction (depending on the
 *												  orientation of epipolar lines) to maximize the useful image area.
 * double alpha							Input  -> Free scaling parameter. If it is -1 or absent, the function performs the default scaling.
 *												  Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images
 *												  are zoomed and shifted so that only valid pixels are visible (no black areas after
 *												  rectification). alpha=1 means that the rectified image is decimated and shifted so that all
 *												  the pixels from the original images from the cameras are retained in the rectified images
 *												  (no source image pixels are lost). Obviously, any intermediate value yields an intermediate
 *												  result between those two extreme cases.
 * Size newImageSize					Input  -> New image resolution after rectification. The same size should be passed to
 *												  initUndistortRectifyMap() (see the stereo_calib.cpp sample in OpenCV samples directory).
 *												  When (0,0) is passed (default), it is set to the original imageSize . Setting it to larger
 *												  value can help you preserve details in the original image, especially when there is a big
 *												  radial distortion.
 * Rect* validPixROI1					Output -> Optional output rectangles inside the rectified images where all the pixels are valid.
 *												  If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller.
 * Rect* validPixROI2					Output -> Optional output rectangles inside the rectified images where all the pixels are valid.
 *												  If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller.
 *
 * Return value:						none
 */
void stereoRectify2( InputArray _cameraMatrix1, InputArray _distCoeffs1,
                        InputArray _cameraMatrix2, InputArray _distCoeffs2,
                        Size imageSize, InputArray _Rmat, InputArray _Tmat,
                        OutputArray _Rmat1, OutputArray _Rmat2,
                        OutputArray _Pmat1, OutputArray _Pmat2,
                        OutputArray _Qmat, int flags,
                        double alpha, Size newImageSize,
                        Rect* validPixROI1, Rect* validPixROI2 )
{
    Mat cameraMatrix1 = _cameraMatrix1.getMat(), cameraMatrix2 = _cameraMatrix2.getMat();
    Mat distCoeffs1 = _distCoeffs1.getMat(), distCoeffs2 = _distCoeffs2.getMat();
    Mat Rmat = _Rmat.getMat(), Tmat = _Tmat.getMat();
    cv::Mat c_cameraMatrix1 = cameraMatrix1;
    cv::Mat c_cameraMatrix2 = cameraMatrix2;
    cv::Mat c_distCoeffs1 = distCoeffs1;
    cv::Mat c_distCoeffs2 = distCoeffs2;
    cv::Mat c_R = Rmat, c_T = Tmat;

    int rtype = CV_64F;
    _Rmat1.create(3, 3, rtype);
    _Rmat2.create(3, 3, rtype);
    _Pmat1.create(3, 4, rtype);
    _Pmat2.create(3, 4, rtype);
    cv::Mat c_R1 = _Rmat1.getMat(), c_R2 = _Rmat2.getMat(), c_P1 = _Pmat1.getMat(), c_P2 = _Pmat2.getMat();
    cv::Mat c_Q, *p_Q = nullptr;

    if( _Qmat.needed() )
    {
        _Qmat.create(4, 4, rtype);
        p_Q = &(c_Q = _Qmat.getMat());
    }

    cvStereoRectify2( &c_cameraMatrix1, &c_cameraMatrix2, &c_distCoeffs1, &c_distCoeffs2,
        imageSize, &c_R, &c_T, &c_R1, &c_R2, &c_P1, &c_P2, p_Q, flags, alpha,
        newImageSize, (cv::Rect*)validPixROI1, (cv::Rect*)validPixROI2);
}


/* Slightly changed version of the OpenCV rectification function cvStereoRectify. Check the OpenCV documentation
 * for more information. This function was copied to be able to change a few details on the core functionality of the rectification -
 * especiallly the undistortion functionality to estimate the new virtual cameras.
 *
 * CvMat* _cameraMatrix1				Input  -> Camera matrix of the first (left) camera
 * CvMat* _cameraMatrix2				Input  -> Camera matrix of the second (right) camera
 * CvMat* _distCoeffs1					Input  -> Distortion parameters of the first camera
 * CvMat* _distCoeffs2					Input  -> Distortion parameters of the second camera
 * CvSize imageSize						Input  -> Size of the original image
 * CvMat* matR							Input  -> Rotation matrix to specify the 3D rotation from the first to the second camera
 * CvMat* matT							Input  -> Translation vector specifying the translational direction from the first to the
 *												  second camera (the norm of the vector must be 1.0)
 * CvMat* _R1							Output -> Rectification transform (rotation matrix) for the first camera
 * CvMat* _R2							Output -> Rectification transform (rotation matrix) for the second camera
 * CvMat* _P1							Output -> Projection matrix in the new (rectified) coordinate systems for the first camera
 * CvMat* _P2							Output -> Projection matrix in the new (rectified) coordinate systems for the second camera
 * CvMat* matQ							Output -> Output 4x4 disparity-to-depth mapping matrix (see reprojectImageTo3D() ).
 * int flags							Input  -> Operation flags that may be zero or CV_CALIB_ZERO_DISPARITY . If the flag is set,
 *												  the function makes the principal points of each camera have the same pixel
 *												  coordinates in the rectified views. And if the flag is not set, the function may
 *												  still shift the images in the horizontal or vertical direction (depending on the
 *												  orientation of epipolar lines) to maximize the useful image area.
 * double alpha							Input  -> Free scaling parameter. If it is -1 or absent, the function performs the default scaling.
 *												  Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images
 *												  are zoomed and shifted so that only valid pixels are visible (no black areas after
 *												  rectification). alpha=1 means that the rectified image is decimated and shifted so that all
 *												  the pixels from the original images from the cameras are retained in the rectified images
 *												  (no source image pixels are lost). Obviously, any intermediate value yields an intermediate
 *												  result between those two extreme cases.
 * CvSize newImgSize					Input  -> New image resolution after rectification. The same size should be passed to
 *												  initUndistortRectifyMap() (see the stereo_calib.cpp sample in OpenCV samples directory).
 *												  When (0,0) is passed (default), it is set to the original imageSize . Setting it to larger
 *												  value can help you preserve details in the original image, especially when there is a big
 *												  radial distortion.
 * CvRect* roi1							Output -> Optional output rectangles inside the rectified images where all the pixels are valid.
 *												  If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller.
 * CvRect* roi2							Output -> Optional output rectangles inside the rectified images where all the pixels are valid.
 *												  If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller.
 *
 * Return value:						none
 */
void cvStereoRectify2( const cv::Mat* _cameraMatrix1, const cv::Mat* _cameraMatrix2,
                      const cv::Mat* _distCoeffs1, const cv::Mat* _distCoeffs2,
                      const cv::Size& imageSize, const cv::Mat* matR, const cv::Mat* matT,
                      cv::Mat* _R1, cv::Mat* _R2, cv::Mat* _P1, cv::Mat* _P2,
                      cv::Mat* matQ, int flags, double alpha, cv::Size newImgSize,
                      cv::Rect* roi1, cv::Rect* roi2 )
{
    double _om[3], _t[3], _uu[3]={0,0,0}, _r_r[3][3], _pp[3][4];
    double _ww[3], _wr[3][3], _z[3] = {0,0,0}, _ri[3][3];
    cv::Rect_<float> inner1, inner2, outer1, outer2;

    cv::Mat om  = cv::Mat(3, 1, CV_64F, _om);
    cv::Mat t   = cv::Mat(3, 1, CV_64F, _t);
    cv::Mat uu  = cv::Mat(3, 1, CV_64F, _uu);
    cv::Mat r_r = cv::Mat(3, 3, CV_64F, _r_r);
    cv::Mat pp  = cv::Mat(3, 4, CV_64F, _pp);
    cv::Mat ww  = cv::Mat(3, 1, CV_64F, _ww); // temps
    cv::Mat wR  = cv::Mat(3, 3, CV_64F, _wr);
    cv::Mat Z   = cv::Mat(3, 1, CV_64F, _z);
    cv::Mat Ri  = cv::Mat(3, 3, CV_64F, _ri);
    double nx = imageSize.width, ny = imageSize.height;
    int i, k;

    if( matR->rows == 3 && matR->cols == 3 )
        cv::Rodrigues(*matR, om);// get vector rotation
    else
        matR->convertTo(om, CV_64F);// it's already a rotation vector
    om.convertTo(om, CV_64F, -0.5);// get average rotation
    cv::Rodrigues(om, r_r);// rotate cameras to same orientation by averaging
    t = r_r * *matT;
//    cvConvertScale(&om, &om, -0.5); // get average rotation
//    cvRodrigues2(&om, &r_r);        // rotate cameras to same orientation by averaging
//    cvMatMul(&r_r, matT, &t);

    unsigned int idx = fabs(_t[0]) > fabs(_t[1]) ? 0 : 1;
    double c = _t[idx], nt = cv::norm(t, cv::NORM_L2);
    _uu[idx] = c > 0 ? 1 : -1;

    // calculate global Z rotation
//    cvCrossProduct(&t,&uu,&ww);
    ww = t.cross(uu);
//    double nw = cvNorm(&ww, 0, CV_L2);
    double nw = cv::norm(ww, cv::NORM_L2);
    if (nw > 0.0) {
        ww.convertTo(ww, CV_64F, acos(fabs(c) / nt) / nw);
//        cvConvertScale(&ww, &ww, acos(fabs(c) / nt) / nw);
    }
    cv::Rodrigues(ww, wR);
//    cvRodrigues2(&ww, &wR);

    // apply to both views
    Ri = wR * r_r.t();
    Ri.convertTo(*_R1, _R1->type());
    Ri = wR * r_r;
    Ri.convertTo(*_R2, _R2->type());
    t = Ri * *matT;
//    cvGEMM(&wR, &r_r, 1, 0, 0, &Ri, CV_GEMM_B_T);
//    cvConvert( &Ri, _R1 );
//    cvGEMM(&wR, &r_r, 1, 0, 0, &Ri, 0);
//    cvConvert( &Ri, _R2 );
//    cvMatMul(&Ri, matT, &t);

    // calculate projection/camera matrices
    // these contain the relevant rectified image internal params (fx, fy=fx, cx, cy)
    auto fc_new = DBL_MAX;
    std::vector<cv::Point2d> cc_new = std::vector<cv::Point2d>(2, cv::Point2d(0, 0));
//    CvPoint2D64f cc_new[2] = {{0,0}, {0,0}};

    for( k = 0; k < 2; k++ ) {
        const cv::Mat* A = k == 0 ? _cameraMatrix1 : _cameraMatrix2;
        const cv::Mat* Dk = k == 0 ? _distCoeffs1 : _distCoeffs2;
        double dk1 = Dk ? Dk->at<double>(0) : 0;
        double fc = A->at<double>(idx^1,idx^1);
        if( dk1 < 0 ) {
            fc *= 1 + dk1*(nx*nx + ny*ny)/(4*fc*fc);
        }
        fc_new = MIN(fc_new, fc);
    }

    for( k = 0; k < 2; k++ )
    {
        const cv::Mat* A = k == 0 ? _cameraMatrix1 : _cameraMatrix2;
        const cv::Mat* Dk = k == 0 ? _distCoeffs1 : _distCoeffs2;
        std::vector<cv::Point2f> _pts(4), _pts_3(4);
//        CvPoint2D32f _pts[4];
//        CvPoint3D32f _pts_3[4];
        cv::Mat pts = cv::Mat(_pts);
        cv::Mat pts_3 = cv::Mat(_pts_3);

        for( i = 0; i < 4; i++ )
        {
            int j = (i<2) ? 0 : 1;
            _pts[i].x = (float)((i % 2)*(nx-1));
            _pts[i].y = (float)(j*(ny-1));
        }
        { //From OpenCV deviating implementation starts here
            int valid_nr;
            double reduction = 0.01;
            cv::Point2d imgCent = cv::Point2d(nx / 2.0, ny / 2.0);
            std::vector<cv::Point2f> _pts_sv = _pts;
//            CvPoint2D64f imgCent = cvPoint2D64f(nx / 2.0, ny / 2.0);
//            CvPoint2D32f _pts_sv[4];
//            memcpy(_pts_sv, _pts, 4 * sizeof(CvPoint2D32f));
            //cv::Mat pts_sv = cv::Mat(1, 4, CV_32FC2, _pts_sv);
            //pts_sv = cvCloneMat(&pts);
            do
            {
                Mat mask;
                cvUndistortPoints2( pts, pts, *A, Dk, nullptr, nullptr, mask );
                valid_nr = cv::countNonZero(mask);
                if(valid_nr < 4)
                {
                    for( i = 0; i < 4; i++ )
                    {
                        int j = (i<2) ? 0 : 1;
                        if(!mask.at<bool>(i))
                        {
                            _pts_sv[i].x = _pts[i].x = (float)((floor((1.0 - reduction) * ((i % 2) * nx - imgCent.x)) + imgCent.x - 1));
                            _pts_sv[i].y = _pts[i].y = (float)((floor((1.0 - reduction) * (j * ny - imgCent.y)) + imgCent.y - 1));
                        }
                        else
                        {
                            _pts[i].x = _pts_sv[i].x;
                            _pts[i].y = _pts_sv[i].y;
                        }
                    }
                    reduction += 0.01;
                }
            }while((valid_nr < 4) && (reduction < 0.25));

            if(reduction >= 0.25)
            {
                Mat mask;
                cvUndistortPoints2( pts, pts, *A, nullptr, nullptr, nullptr, mask );
            }
        } //From OpenCV deviating implementation ends here


//        cvConvertPointsHomogeneous( &pts, &pts_3 );
        cv::convertPointsHomogeneous( pts, pts_3 );

        //Change camera matrix to have cc=[0,0] and fc = fc_new
        double _a_tmp[3][3];
        cv::Mat A_tmp  = cv::Mat(3, 3, CV_64F, _a_tmp);
        _a_tmp[0][0]=fc_new;
        _a_tmp[1][1]=fc_new;
        _a_tmp[0][2]=0.0;
        _a_tmp[1][2]=0.0;
//        cvProjectPoints2( &pts_3, k == 0 ? _R1 : _R2, &Z, &A_tmp, 0, &pts );
        cv::projectPoints( pts_3, k == 0 ? *_R1 : *_R2, Z, A_tmp, 0, pts );
//        CvScalar avg = cvAvg(&pts);
        cv::Scalar avg = cv::mean(pts);
        cc_new[k].x = (nx - 1.)/2. - avg[0];
        cc_new[k].y = (ny - 1.)/2. - avg[1];
    }

    // vertical focal length must be the same for both images to keep the epipolar constraint
    // (for horizontal epipolar lines -- TBD: check for vertical epipolar lines)
    // use fy for fx also, for simplicity

    // For simplicity, set the principal points for both cameras to be the average
    // of the two principal points (either one of or both x- and y- coordinates)
    if( flags & cv::CALIB_ZERO_DISPARITY )
    {
        cc_new[0].x = cc_new[1].x = (cc_new[0].x + cc_new[1].x)*0.5;
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;
    }
    else if( idx == 0 ) // horizontal stereo
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;
    else // vertical stereo
        cc_new[0].x = cc_new[1].x = (cc_new[0].x + cc_new[1].x)*0.5;

//    cvZero( &pp );
    pp.setTo(cv::Scalar::all(0));
    _pp[0][0] = _pp[1][1] = fc_new;
    _pp[0][2] = cc_new[0].x;
    _pp[1][2] = cc_new[0].y;
    _pp[2][2] = 1.;
    pp.convertTo(*_P1, _P1->type());
//    cvConvert(&pp, _P1);

    _pp[0][2] = cc_new[1].x;
    _pp[1][2] = cc_new[1].y;
    _pp[idx][3] = _t[idx]*fc_new; // baseline * focal length
//    cvConvert(&pp, _P2);
    pp.convertTo(*_P2, _P2->type());

    alpha = std::min(alpha, 1.);

    icvGetRectanglesV0( _cameraMatrix1, _distCoeffs1, _R1, _P1, imageSize, inner1, outer1 );
    icvGetRectanglesV0( _cameraMatrix2, _distCoeffs2, _R2, _P2, imageSize, inner2, outer2 );

    {
    newImgSize = newImgSize.width*newImgSize.height != 0 ? newImgSize : imageSize;
    double cx1_0 = cc_new[0].x;
    double cy1_0 = cc_new[0].y;
    double cx2_0 = cc_new[1].x;
    double cy2_0 = cc_new[1].y;
    double cx1 = newImgSize.width*cx1_0/imageSize.width;
    double cy1 = newImgSize.height*cy1_0/imageSize.height;
    double cx2 = newImgSize.width*cx2_0/imageSize.width;
    double cy2 = newImgSize.height*cy2_0/imageSize.height;
    double s = 1.;

    if( alpha >= 0 )
    {
        double s0 = std::max(std::max(std::max((double)cx1/(cx1_0 - inner1.x), (double)cy1/(cy1_0 - inner1.y)),
                            (double)(newImgSize.width - cx1)/(inner1.x + inner1.width - cx1_0)),
                        (double)(newImgSize.height - cy1)/(inner1.y + inner1.height - cy1_0));
        s0 = std::max(std::max(std::max(std::max((double)cx2/(cx2_0 - inner2.x), (double)cy2/(cy2_0 - inner2.y)),
                         (double)(newImgSize.width - cx2)/(inner2.x + inner2.width - cx2_0)),
                     (double)(newImgSize.height - cy2)/(inner2.y + inner2.height - cy2_0)),
                 s0);

        double s1 = std::min(std::min(std::min((double)cx1/(cx1_0 - outer1.x), (double)cy1/(cy1_0 - outer1.y)),
                            (double)(newImgSize.width - cx1)/(outer1.x + outer1.width - cx1_0)),
                        (double)(newImgSize.height - cy1)/(outer1.y + outer1.height - cy1_0));
        s1 = std::min(std::min(std::min(std::min((double)cx2/(cx2_0 - outer2.x), (double)cy2/(cy2_0 - outer2.y)),
                         (double)(newImgSize.width - cx2)/(outer2.x + outer2.width - cx2_0)),
                     (double)(newImgSize.height - cy2)/(outer2.y + outer2.height - cy2_0)),
                 s1);

        s = s0*(1. - alpha) + s1*alpha;
        if((s > 2.) || (s < 0.5)) //added to OpenCV function
            s = 1.0;
    }

    fc_new *= s;
//    cc_new[0] = cvPoint2D64f(cx1, cy1);
//    cc_new[1] = cvPoint2D64f(cx2, cy2);
    cc_new[0] = cv::Point2d(cx1, cy1);
    cc_new[1] = cv::Point2d(cx2, cy2);

    _P1->at<double>(0,0) = fc_new;
    _P1->at<double>(1,1) = fc_new;
    _P1->at<double>(0,2) = cx1;
    _P1->at<double>(1,2) = cy1;
//    cvmSet(_P1, 0, 0, fc_new);
//    cvmSet(_P1, 1, 1, fc_new);
//    cvmSet(_P1, 0, 2, cx1);
//    cvmSet(_P1, 1, 2, cy1);

    _P2->at<double>(0,0) = fc_new;
    _P2->at<double>(1,1) = fc_new;
    _P2->at<double>(0,2) = cx2;
    _P2->at<double>(1,2) = cy2;
    _P2->at<double>(idx,3) = s * _P2->at<double>(idx, 3);
//    cvmSet(_P2, 0, 0, fc_new);
//    cvmSet(_P2, 1, 1, fc_new);
//    cvmSet(_P2, 0, 2, cx2);
//    cvmSet(_P2, 1, 2, cy2);
//    cvmSet(_P2, idx, 3, s*cvmGet(_P2, idx, 3));

    if(roi1)
    {
        *roi1 = cv::Rect((int)std::ceil(((double)inner1.x - cx1_0)*s + cx1),
                         (int)std::ceil(((double)inner1.y - cy1_0)*s + cy1),
                         (int)std::floor((double)inner1.width*s),
                         (int)std::floor((double)inner1.height*s))
            & cv::Rect(0, 0, newImgSize.width, newImgSize.height);
    }

    if(roi2)
    {
        *roi2 = cv::Rect((int)std::ceil(((double)inner2.x - cx2_0)*s + cx2),
                         (int)std::ceil(((double)inner2.y - cy2_0)*s + cy2),
                         (int)std::floor((double)inner2.width*s),
                         (int)std::floor((double)inner2.height*s))
            & cv::Rect(0, 0, newImgSize.width, newImgSize.height);
    }
    }

    if( matQ )
    {
        double q[] =
        {
            1, 0, 0, -cc_new[0].x,
            0, 1, 0, -cc_new[0].y,
            0, 0, 0, fc_new,
            0, 0, -1./_t[idx],
            (idx == 0 ? cc_new[0].x - cc_new[1].x : cc_new[0].y - cc_new[1].y)/_t[idx]
        };
        cv::Mat Q = cv::Mat(4, 4, CV_64F, q);
        Q.convertTo(*matQ, matQ->type());
//        cvConvert( &Q, matQ );
    }
}

/* Slightly changed version of the OpenCV undistortion function cvUndistortPoints. Check the OpenCV documentation
 * for more information. Here a check was added to identify errors during undistortion. Therefore a mask is provided
 * which marks coordinates for which the undistortion was not possible due to a too large error.
 *
 * CvMat* _src							Input  -> Observed point coordinates (distorted), 1xN or Nx1 2-channel (CV_32FC2 or CV_64FC2).
 * CvMat* dst_							Output -> Output ideal point coordinates after undistortion and reverse perspective
 *												  transformation. If matrix P is identity or omitted, dst will contain normalized
 *												  point coordinates.
 * CvMat* _cameraMatrix					Input  -> Camera matrix
 * CvMat* _distCoeffs					Input  -> Input vector of distortion coefficients (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])
 *												  of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients
 *												  are assumed.
 * CvMat* matR							Input  -> Rectification transform (rotation matrix) in the object space (3x3 matrix). R1 or R2
 *												  computed by stereoRectify() can be passed here. If the matrix is empty, the identity
 *												  transformation is used.
 * CvMat* matP							Input  -> New camera matrix (3x3) or new projection matrix (3x4). P1 or P2 computed by
 *												  stereoRectify() can be passed here. If the matrix is empty, the identity new camera
 *												  matrix is used.
 * OutputArray mask						Output -> Mask marking coordinates for which the undistortion was not possible due to a too
 *												  large error
 *
 * Return value:						none
 */
void cvUndistortPoints2(const cv::Mat& src_, cv::Mat& dst_, const cv::Mat& cameraMatrix_,
                        const cv::Mat* distCoeffs_,
                        const cv::Mat* matR, const cv::Mat* matP, cv::OutputArray mask ) //the mask was added here
{
    double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, RR[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, k[8]={0,0,0,0,0,0,0,0}, fx, fy, ifx, ify, cx, cy;
    cv::Mat matA=cv::Mat(3, 3, CV_64F, A), Dk_;
    cv::Mat RR_ =cv::Mat(3, 3, CV_64F, RR);
    std::vector<cv::Point2d> srcd;
    int stype, dtype;
    int sstep;
    int i, j, n, iters = 1;

    CV_Assert((src_.rows == 1 || src_.cols == 1) &&
              (dst_.rows == 1 || dst_.cols == 1) &&
              src_.cols + src_.rows - 1 == dst_.rows + dst_.cols - 1 &&
              (src_.type() == CV_32FC2 || src_.type() == CV_64FC2) &&
              (dst_.type() == CV_32FC2 || dst_.type() == CV_64FC2));

    CV_Assert(cameraMatrix_.rows == 3 && cameraMatrix_.cols == 3 );

    cameraMatrix_.convertTo(matA, CV_64F);

    if( distCoeffs_ ){
        CV_Assert((distCoeffs_->rows == 1 || distCoeffs_->cols == 1) &&
                  (distCoeffs_->rows * distCoeffs_->cols == 4 ||
                   distCoeffs_->rows * distCoeffs_->cols == 5 ||
                   distCoeffs_->rows * distCoeffs_->cols == 8));

        Dk_ = cv::Mat(distCoeffs_->rows, distCoeffs_->cols, CV_64FC(distCoeffs_->channels()), k);

        distCoeffs_->convertTo(Dk_, CV_64F);
        iters = 5;
    }

    if( matR ){
        CV_Assert( matR->rows == 3 && matR->cols == 3 );
        matR->convertTo(RR_, CV_64F);
    }
    else{
        RR_ = cv::Mat::eye(3, 3, CV_64FC1);
    }

    if( matP ){
        double PP[3][3];
        cv::Mat PP_=cv::Mat(3, 3, CV_64F, PP);
        CV_Assert( matP->rows == 3 && (matP->cols == 3 || matP->cols == 4));
        matP->colRange(0,3).convertTo(PP_, CV_64F);
        RR_ = PP_ * RR_;
    }

    stype = src_.type();
    sstep = src_.rows == 1 ? 1 : src_.step / CV_ELEM_SIZE(stype);
    src_.reshape(2).convertTo(srcd, CV_64F);
    if(dst_.empty()){
        src_.copyTo(dst_);
    }
    dtype = dst_.type();

    n = static_cast<int>(srcd.size());

    fx = A[0][0];
    fy = A[1][1];
    ifx = 1./fx;
    ify = 1./fy;
    cx = A[0][2];
    cy = A[1][2];

    //Generate a mask to check if the undistortion generates valid results
    mask.create(1,n,CV_8UC1);
    Mat _mask = mask.getMat();
    _mask = cv::Mat::ones(1,n,CV_8UC1);

    for( i = 0; i < n; i++ )
    {
        double x, y, x0, y0;
        x = srcd[i].x;
        y = srcd[i].y;

        x0 = x = (x - cx)*ifx;
        y0 = y = (y - cy)*ify;

        // compensate distortion iteratively
        for( j = 0; j < iters; j++ )
        {
            double r2 = x*x + y*y;
            double icdist = (1. + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1. + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
            double deltaX = 2. * k[2]*x*y + k[3]*(r2 + 2. * x*x);
            double deltaY = k[2]*(r2 + 2. * y*y) + 2. * k[3]*x*y;
            x = (x0 - deltaX)*icdist;
            y = (y0 - deltaY)*icdist;
        }

        //check the error of the undistortion
        {
            Point2f proofdist;
            double r2 = x*x + y*y;
            double icdist = (1. + ((k[4]*r2 + k[1])*r2 + k[0])*r2)/(1. + ((k[7]*r2 + k[6])*r2 + k[5])*r2);
            double deltaX = 2. * k[2]*x*y + k[3]*(r2 + 2. * x*x);
            double deltaY = k[2]*(r2 + 2. * y*y) + 2. * k[3]*x*y;
            proofdist.x = x * icdist + deltaX - x0;
            proofdist.y = y * icdist + deltaY - y0;
            if( std::sqrt(proofdist.x * proofdist.x + proofdist.y * proofdist.y) > 0.25)
                _mask.at<bool>(i) = false;
        }

        double xx = RR[0][0]*x + RR[0][1]*y + RR[0][2];
        double yy = RR[1][0]*x + RR[1][1]*y + RR[1][2];
        double ww = 1./(RR[2][0]*x + RR[2][1]*y + RR[2][2]);
        x = xx*ww;
        y = yy*ww;

        if( dtype == CV_32FC2 )
        {
            dst_.at<float>(i * 2) = static_cast<float>(x);
            dst_.at<float>(i * 2 + 1) = static_cast<float>(y);
        }
        else
        {
            dst_.at<double>(i * 2) = x;
            dst_.at<double>(i * 2 + 1) = y;
        }
    }
}

/* Estimates the inner rectangle of a distorted image containg only valid/available image information and an outer rectangle countaing all
 * image information. This function was copied from the OpenCV (calibration.cpp) and modified such that the image coordinates used for
 * undistortion are checked afterwards to be valid. If not, the initial coordinates are changed as long as only valid undistorted
 * coordinates are used.
 *
 * CvMat* cameraMatrix					Input  -> Original camera matrix
 * CvMat* distCoeffs					Input  -> Distortion parameters of the camera
 * CvMat* R								Input  -> Rectification transform (rotation matrix) for the camera
 * CvMat* newCameraMatrix				Input  -> Camera matrix of the new (virtual) camera
 * CvSize imgSize						Input  -> Size of the original image
 * Rect_<float>& inner					Output -> Inner rectangle containing only valid image information
 * Rect_<float>& outer					Output -> Outer rectangle containing all the image information
 *
 * Return value:						none
 */
void icvGetRectanglesV0( const cv::Mat* cameraMatrix, const cv::Mat* distCoeffs,
                 const cv::Mat* R, const cv::Mat* newCameraMatrix, const cv::Size& imgSize,
                 cv::Rect_<float>& inner, cv::Rect_<float>& outer )
{
    const int N = 9;
    int x, y, k;
//    cv::Ptr<cv::Mat> _pts = cvCreateMat(1, N*N, CV_32FC2);
    cv::Mat _pts = Mat(1, N*N, CV_32FC2);
    std::vector<cv::Point2f> pts = (std::vector<cv::Point2f>)(_pts.reshape(2));
//    CvPoint2D32f* pts = (CvPoint2D32f*)(_pts->data.ptr);

    for( y = k = 0; y < N; y++ )
        for( x = 0; x < N; x++ )
            pts[k++] = cv::Point2f((float)x*imgSize.width/(N-1),
                                   (float)y*imgSize.height/(N-1));

    { //From OpenCV deviating implementation starts here
        int valid_nr;
        float reduction = 0.01f;
        cv::Point2f imgCent = cv::Point2f((float)imgSize.width / 2.f,(float)imgSize.height / 2.f);
        std::vector<cv::Point2f> pts_sv = pts;
//        memcpy(pts_sv, pts, N * N * sizeof(CvPoint2D32f));
        do
        {
            Mat mask;
            cvUndistortPoints2(_pts, _pts, *cameraMatrix, distCoeffs, R, newCameraMatrix, mask);
            valid_nr = cv::countNonZero(mask);
            if(valid_nr < N*N)
            {
                for( y = k = 0; y < N; y++ )
                    for( x = 0; x < N; x++ )
                    {
                        if(!mask.at<bool>(k))
                            pts_sv[k] = pts[k] = cv::Point2f((1.0 - reduction) * ((float)x*imgSize.width/(N-1) - imgCent.x) + imgCent.x,
                                                              (1.0 - reduction) * ((float)y*imgSize.height/(N-1) - imgCent.y) + imgCent.y);
                        else
                            pts[k] = pts_sv[k];
                        k++;
                    }
                reduction += 0.01f;
            }
        }while((valid_nr < N*N) && (reduction < 0.25));

        if(reduction >= 0.25)
        {
            Mat mask;
            cvUndistortPoints2(_pts, _pts, *cameraMatrix, nullptr, R, newCameraMatrix, mask);
        }
    } //From OpenCV deviating implementation ends here

    float iX0=-FLT_MAX, iX1=FLT_MAX, iY0=-FLT_MAX, iY1=FLT_MAX;
    float oX0=FLT_MAX, oX1=-FLT_MAX, oY0=FLT_MAX, oY1=-FLT_MAX;
    // find the inscribed rectangle.
    // the code will likely not work with extreme rotation matrices (R) (>45%)
    for( y = k = 0; y < N; y++ )
        for( x = 0; x < N; x++ )
        {
            cv::Point2f p = pts[k++];
            oX0 = min(oX0, p.x);
            oX1 = max(oX1, p.x);
            oY0 = min(oY0, p.y);
            oY1 = max(oY1, p.y);

            if( x == 0 )
                iX0 = max(iX0, p.x);
            if( x == N-1 )
                iX1 = min(iX1, p.x);
            if( y == 0 )
                iY0 = max(iY0, p.y);
            if( y == N-1 )
                iY1 = min(iY1, p.y);
        }
    inner = cv::Rect_<float>(iX0, iY0, iX1-iX0, iY1-iY0);
    outer = cv::Rect_<float>(oX0, oY0, oX1-oX0, oY1-oY0);
}

/* Estimates the vergence (shift of starting point) for correspondence search in the stereo engine. To get the right values, the
 * first camera centre must be at the orign of the coordinate system.
 *
 * Mat R								Input  -> Rotation matrix between the cameras.
 * Mat RR1								Input  -> Rectification transform (rotation matrix) for the first camera
 * Mat RR2								Input  -> Rectification transform (rotation matrix) for the second camera
 * Mat PR1								Input  -> Camera (Projection) matrix in the new (rectified) coordinate systems for the first camera
 * Mat PR2								Input  -> Camera (Projection) matrix in the new (rectified) coordinate systems for the second camera
 *
 * Return value:						Vergence
 */
int estimateVergence(const cv::Mat& R, const cv::Mat& RR1, const cv::Mat& RR2, const cv::Mat& PR1, const cv::Mat& PR2)
{
    Mat a = R.row(2).t();
    Mat K1 = PR1.colRange(0,3);
    Mat K2 = PR2.colRange(0,3);
    Mat ar1 = K1 * RR1 * a;
    Mat ar2 = K2 * RR2.col(2);
    ar1 = ar1 / ar1.at<double>(2);
    ar2 = ar2 / ar2.at<double>(2);
    double vergence = ar1.at<double>(0) - ar2.at<double>(0);
    if(nearZero(vergence))
        return 0;
    vergence = std::ceil(1.1 * vergence);

    if(vergence < 0.0)
        cout << "Vergence is negative!" << endl;

    return (int)vergence;
}


/* Estimates the optimal scale for the focal length of the virtuel camera. This is a slightly changed version of the same functionality
 * implemented in the function cvStereoRectify of the OpenCV. In contrast to the original OpenCV function, the result of the undistortion
 * is checked to be valid.
 *
 * double alpha							Input  -> Free scaling parameter. If it is -1 or absent, the function performs the default scaling.
 *												  Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images
 *												  are zoomed and shifted so that only valid pixels are visible (no black areas after
 *												  rectification). alpha=1 means that the rectified image is decimated and shifted so that all
 *												  the pixels from the original images from the cameras are retained in the rectified images
 *												  (no source image pixels are lost). Obviously, any intermediate value yields an intermediate
 *												  result between those two extreme cases.
 * Mat K1								Input  -> Camera matrix of the first (left) camera
 * Mat K2								Input  -> Camera matrix of the second (right) camera
 * Mat R1								Input  -> Rectification transform (rotation matrix) for the first camera
 * Mat R2								Input  -> Rectification transform (rotation matrix) for the second camera
 * Mat P1								Input  -> Camera (Projection) matrix in the new (rectified) coordinate systems for the first camera
 * Mat P2								Input  -> Camera (Projection) matrix in the new (rectified) coordinate systems for the second camera
 * Mat dist1							Input  -> Distortion parameters of the first camera
 * Mat dist2							Input  -> Distortion parameters of the second camera
 * Size imageSize						Input  -> Size of the original image
 * Size newImageSize					Input  -> Size of the new image (from the virtual camera)
 *
 * Return value:						Scaling parameter for the focal length
 */
double estimateOptimalFocalScale(double alpha, cv::Mat K1, cv::Mat K2, cv::Mat R1, cv::Mat R2, cv::Mat P1, cv::Mat P2,
                                 cv::Mat dist1, cv::Mat dist2, const cv::Size& imageSize, cv::Size newImgSize)
{
    alpha = MIN(alpha, 1.);

    cv::Mat _cameraMatrix1 = K1;
    cv::Mat _cameraMatrix2 = K2;
    cv::Mat _distCoeffs1 = dist1;
    cv::Mat _distCoeffs2 = dist2;
    cv::Mat _R1 = R1;
    cv::Mat _R2 = R2;
    cv::Mat _P1 = P1;
    cv::Mat _P2 = P2;

    cv::Rect_<float> inner1, inner2, outer1, outer2;

    icvGetRectanglesV0( &_cameraMatrix1, &_distCoeffs1, &_R1, &_P1, imageSize, inner1, outer1 );
    icvGetRectanglesV0( &_cameraMatrix2, &_distCoeffs2, &_R2, &_P2, imageSize, inner2, outer2 );

    newImgSize = newImgSize.width*newImgSize.height != 0 ? newImgSize : imageSize;
    double cx1_0 = K1.at<double>(0,2);
    double cy1_0 = K1.at<double>(1,2);
    double cx2_0 = K2.at<double>(0,2);
    double cy2_0 = K2.at<double>(1,2);
    double cx1 = newImgSize.width*cx1_0/imageSize.width;
    double cy1 = newImgSize.height*cy1_0/imageSize.height;
    double cx2 = newImgSize.width*cx2_0/imageSize.width;
    double cy2 = newImgSize.height*cy2_0/imageSize.height;
    double s = 1.;

    if( alpha >= 0 )
    {
        double s0 = std::max(std::max(std::max((double)cx1/(cx1_0 - inner1.x), (double)cy1/(cy1_0 - inner1.y)),
                            (double)(newImgSize.width - cx1)/(inner1.x + inner1.width - cx1_0)),
                        (double)(newImgSize.height - cy1)/(inner1.y + inner1.height - cy1_0));
        s0 = std::max(std::max(std::max(std::max(cx2/(cx2_0 - (double)inner2.x), cy2/(cy2_0 - (double)inner2.y)),
                         ((double)newImgSize.width - cx2)/((double)inner2.x + (double)inner2.width - cx2_0)),
                     ((double)newImgSize.height - cy2)/((double)inner2.y + (double)inner2.height - cy2_0)),
                 s0);

        double s1 = std::min(std::min(std::min((double)cx1/(cx1_0 - outer1.x), (double)cy1/(cy1_0 - outer1.y)),
                            (double)(newImgSize.width - cx1)/(outer1.x + outer1.width - cx1_0)),
                        (double)(newImgSize.height - cy1)/(outer1.y + outer1.height - cy1_0));
        s1 = std::min(std::min(std::min(std::min((double)cx2/(cx2_0 - outer2.x), (double)cy2/(cy2_0 - outer2.y)),
                         (double)(newImgSize.width - cx2)/(outer2.x + outer2.width - cx2_0)),
                     (double)(newImgSize.height - cy2)/(outer2.y + outer2.height - cy2_0)),
                 s1);

        s = s0*(1 - alpha) + s1*alpha;
    }

    return s;
}

/* This function shows the rectified images
 *
 * InputArray img1				Input  -> Image from the first camera
 * InputArray img2				Input  -> Image from the second camera
 * InputArray mapX1				Input  -> Rectification map for the x-coordinates of the first image
 * InputArray mapY1				Input  -> Rectification map for the y-coordinates of the first image
 * InputArray mapX2				Input  -> Rectification map for the x-coordinates of the second image
 * InputArray mapY2				Input  -> Rectification map for the y-coordinates of the second image
 * InputArray t						Input  -> Translation vector of the pose. Take translation vector for
 *											  mapping a position of the left camera x to the a position of
 *											  the right camera x' (x' = R^T * x - R^T * t0) with t0 the
 *											  translation vector after pose estimation and t = -1 * R^T * t0
 *											  the translation vector that should be provided.
 * Size newImgSize				Input  -> Size of the new image (must be the same as specified at the
 *											  rectification function and initUndistortRectifyMap()). If not
 *											  specified, the same size from the input images is used.
 * string path            Input -> output path for rectified images (e.g.: c:\temp\results)
 *                        if "", no images are saved
 *
 * Return:							0:		  Success
 */
int ShowRectifiedImages(cv::InputArray img1,
                        cv::InputArray img2,
                        cv::InputArray mapX1,
                        cv::InputArray mapY1,
                        cv::InputArray mapX2,
                        cv::InputArray mapY2,
                        cv::InputArray t,
                        const std::string& path,
                        const std::string& imgName1,
                        const std::string& imgName2,
                        bool showResult,
                        cv::Size newImgSize)
{
    CV_Assert(!img1.empty() && !img2.empty() && !mapX1.empty() && !mapY1.empty() && !mapX2.empty() && !mapY2.empty() && !t.empty());
    CV_Assert((img1.rows() == img2.rows()) && (img1.cols() == img2.cols()));

    if(newImgSize == cv::Size())
    {
        newImgSize = img1.size();
    }

    Mat _t;
    _t = t.getMat();

    Mat imgRect1, imgRect2, composed, comCopy;
    remap(img1, imgRect1, mapX1, mapY1, cv::BORDER_CONSTANT);
    remap(img2, imgRect2, mapX2, mapY2, cv::BORDER_CONSTANT);

    cv::namedWindow("Rectification");

    // save rectified images
    if (!path.empty())
    {
        string path1;
        if(path.rfind('/') + 1 == path.size()){
            path1 = path;
        }else{
            path1 = path + "/";
        }
        std::string namel, namer;
        if(imgName1.empty() || imgName2.empty()) {
            static int count = 0;
            char buffer[15];
            sprintf(buffer, "left_%04d.jpg", count);
            namel = path1 + buffer;
            sprintf(buffer, "right_%04d.jpg", count);
            namer = path1 + buffer;
            count++;
        }else{
            namel = path1 + imgName1 + ".jpg";
            namer = path1 + imgName2 + ".jpg";
        }

        cv::imwrite(namel, imgRect1);
        cv::imwrite(namer, imgRect2);
    }

    if(showResult) {
        int maxHorImgSize, maxVerImgSize;
        int r, c;
        int rc;
        if (std::abs(_t.at<double>(0)) > std::abs(_t.at<double>(1))) {
            r = 1;
            c = 2;
            rc = 1;
            if (_t.at<double>(0) < 0) {
                Mat imgRect1_tmp;
                imgRect1_tmp = imgRect1.clone();
                imgRect1 = imgRect2.clone();
                imgRect2 = imgRect1_tmp.clone();
            }
        } else {
            r = 2;
            c = 1;
            rc = 0;
            if (_t.at<double>(1) > 0) {
                Mat imgRect1_tmp;
                imgRect1_tmp = imgRect1.clone();
                imgRect1 = imgRect2.clone();
                imgRect2 = imgRect1_tmp.clone();
            }
        }

        //Allocate memory for composed image
        maxHorImgSize = 800;
        if (newImgSize.width > maxHorImgSize) {
            maxVerImgSize = (int) ((float) maxHorImgSize * (float) newImgSize.height / (float) newImgSize.width);
            composed = cv::Mat(cv::Size(maxHorImgSize * c, maxVerImgSize * r), CV_8UC3);
            comCopy = cv::Mat(cv::Size(maxHorImgSize, maxVerImgSize), CV_8UC3);
        } else {
            composed = cv::Mat(cv::Size(newImgSize.width * c, newImgSize.height * r), CV_8UC3);
            comCopy = cv::Mat(cv::Size(newImgSize.width, newImgSize.height), CV_8UC3);
        }

        // create images to display
        string str;
        vector<cv::Mat> show_rect(2);
        cv::cvtColor(imgRect1, show_rect[0], cv::COLOR_GRAY2RGB);
        cv::cvtColor(imgRect2, show_rect[1], cv::COLOR_GRAY2RGB);
        for (int j = 0; j < 2; j++) {
            cv::resize(show_rect[j], comCopy, cv::Size(comCopy.cols, comCopy.rows));
            if (j == 0) str = "CAM 1";
            else str = "CAM 2";
            cv::putText(comCopy, str, cv::Point2d(25, 25), cv::FONT_HERSHEY_SIMPLEX | cv::FONT_ITALIC, 0.7,
                        cv::Scalar(0, 0, 255));
            comCopy.copyTo(
                    composed(cv::Rect(j * rc * comCopy.cols, j * (rc ^ 1) * comCopy.rows, comCopy.cols, comCopy.rows)));
        }

        cv::setMouseCallback("Rectification", on_mouse_move, (void *) (&composed));
        cv::imshow("Rectification", composed);
        cv::waitKey(0);
        cv::destroyWindow("Rectification");
    }

    return 0;
}

/* This function returns the rectified images
 *
 * InputArray img1				Input  -> Image from the first camera
 * InputArray img2				Input  -> Image from the second camera
 * InputArray mapX1				Input  -> Rectification map for the x-coordinates of the first image
 * InputArray mapY1				Input  -> Rectification map for the y-coordinates of the first image
 * InputArray mapX2				Input  -> Rectification map for the x-coordinates of the second image
 * InputArray mapY2				Input  -> Rectification map for the y-coordinates of the second image
 * InputArray t						Input  -> Translation vector of the pose. Take translation vector for
 *											  mapping a position of the left camera x to the a position of
 *											  the right camera x' (x' = R^T * x - R^T * t0) with t0 the
 *											  translation vector after pose estimation and t = -1 * R^T * t0
 *											  the translation vector that should be provided.
 * OutputArray outImg1			Output -> Rectified Image from the first camera
 * OutputArray outImg2			Output -> Rectified Image from the second camera
 * Size newImgSize				Input  -> Size of the new image (must be the same as specified at the
 *											  rectification function and initUndistortRectifyMap()). If not
 *											  specified, the same size from the input images is used.
 *
 * Return:							0:		  Success
 */
int GetRectifiedImages(cv::InputArray img1, cv::InputArray img2, cv::InputArray mapX1, cv::InputArray mapY1, cv::InputArray mapX2, cv::InputArray mapY2, cv::InputArray t, cv::OutputArray outImg1, cv::OutputArray outImg2, cv::Size newImgSize)
{
    CV_Assert(!img1.empty() && !img2.empty() && !mapX1.empty() && !mapY1.empty() && !mapX2.empty() && !mapY2.empty() && !t.empty());
    CV_Assert((img1.rows() == img2.rows()) && (img1.cols() == img2.cols()));

    if(newImgSize == cv::Size())
    {
        newImgSize = img1.size();
    }

    Mat _t;
    _t = t.getMat();

    Mat composed, comCopy;
    remap(img1, outImg1, mapX1, mapY1, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    remap(img2, outImg2, mapX2, mapY2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    return 0;
}

/*------------------------------------------------------------------------------------------
Functionname: on_mouse_move
Parameters: refer to OpenCV documentation (cvSetMouseCallback())
Return: none
Description: draws crosslines over images saved in Mat* composed from int ShowRectifiedImages(...)
------------------------------------------------------------------------------------------*/
void on_mouse_move(int event, int x, int y, int flags, void* param)
{
    Mat composed = *((Mat*)param);
    Mat tmpCopy = cv::Mat(cv::Size(composed.cols, composed.rows), CV_8UC3);
    composed.copyTo(tmpCopy);
    cv::line(tmpCopy, cv::Point2d(0, y), cv::Point2d(tmpCopy.cols, y), cv::Scalar(0, 0, 255));
    cv::line(tmpCopy, cv::Point2d(x, 0), cv::Point2d(x, tmpCopy.rows), cv::Scalar(0, 0, 255));
    cv::imshow("Rectification", tmpCopy);
    bool isAlive = cv::getWindowProperty("Rectification", WND_PROP_VISIBLE) >= 1. - DBL_EPSILON;
    if(isAlive) {
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
//    cv::waitKey(4);
}

/* This function estimates an initial delta value for the SPRT test used within USAC.
* It estimates the initial propability of a keypoint to be classified as an inlier of
* an invalid model (e.g. essential matrix). This initial value is estimated dividing the
* area around the longest possible epipolar line e (length of e * 2 * inlier threshold) by
* the area of the convex hull defined by the found correspondences.
*
* vector<DMatch> matches		Input  -> Matches (ascending queries)
* vector<KeyPoint> kp1			Input  -> Keypoints in the left/first image
* vector<KeyPoint> kp2			Input  -> Keypoints in the right/second image
* double th						Input  -> Inlier threshold in pixels
* Size imgSize					Input  -> Size of the image
*
* Return:						initial delta for SPRT
*/
double estimateSprtDeltaInit(const std::vector<cv::DMatch> &matches,
                             const std::vector<cv::KeyPoint> &kp1,
                             const std::vector<cv::KeyPoint> &kp2,
                             const double &th,
                             const cv::Size &imgSize)
{
    //Extract coordinates from keypoints
    vector<cv::Point2f> points1, points2;
    vector<cv::Point2f> hull1, hull2;
    double area[2] = { 0, 0 };
    double maxEpipoleArea = 0, sprt_delta = 0;
    for (auto matche : matches)
    {
        points1.push_back(kp1[matche.queryIdx].pt);
        points2.push_back(kp2[matche.trainIdx].pt);
    }

    //Convex hull of the keypoints
    cv::convexHull(points1, hull1);
    cv::convexHull(points2, hull2);

    //Area of the convex hull
    area[0] = cv::contourArea(hull1);
    area[1] = cv::contourArea(hull2);
    area[0] = area[0] > area[1] ? area[1] : area[0];

    //max length of an epipolar line within image
    maxEpipoleArea = std::sqrt((double)(imgSize.width * imgSize.width + imgSize.height * imgSize.height));
    //max area for a keypoint to be classified as inlier
    maxEpipoleArea *= 2 * th;
    area[0] = area[0] < (6 * maxEpipoleArea) ? (6 * maxEpipoleArea) : area[0];
    sprt_delta = maxEpipoleArea / area[0];
    sprt_delta = sprt_delta < 0.001 ? 0.001 : sprt_delta;

    return sprt_delta;
}

/* This function estimates an initial epsilon value for the SPRT test used within USAC.
* It estimates the probability that a data point is consistent with a good model (e.g. essential matrix)
* which corresponds approximately to the inlier ratio. To estimate the inlier ratio, VFC has to be applied
* to the matches which should reject most false matches (additionally removes matches at borders) as the
* filtering only allows smooth changes of the optical flow. VFC does not work well for low inlier ratios.
* Thus, the inlier ratio is bounded between 0.1 and 0.75.
*
* vector<DMatch> matches				Input  -> Matches
* unsigned int nrMatchesVfcFiltered		Input  -> Number of remaining matches after filtering with VFC
*
* Return:						initial epsilon for SPRT
*/
double estimateSprtEpsilonInit(const std::vector<cv::DMatch> &matches, const unsigned int &nrMatchesVfcFiltered)
{
    auto nrMatches = (double)matches.size();
    double epsilonInit = 0.8 * (double)nrMatchesVfcFiltered / nrMatches;
    epsilonInit = epsilonInit > 0.4 ? 0.4 : epsilonInit;
    epsilonInit = epsilonInit < 0.1 ? 0.1 : epsilonInit;

    return epsilonInit;
}

/* This function generates an index of the matches with the lowest matching costs first. It is used
* within PROSAC of the USAC framework. To work correctly, the order of the queries within "matches"
* should be ascending. The used correspondences within USAC must be in the same order than "matches".
*
* vector<DMatch> matches				Input  -> Matches
* unsigned int nrMatchesVfcFiltered		Output -> Indices of matches sorted corresponding to the
*												  matching costs in ascending order
*
* Return:						none
*/
void getSortedMatchIdx(std::vector<cv::DMatch> matches, std::vector<unsigned int> & sortedMatchIdx)
{
    size_t i = 0;
    for (i = 0; i < matches.size(); i++)
    {
        if (matches[i].queryIdx != static_cast<int>(i))
            break;
    }
    if (i < matches.size())
    {
        for (i = 0; i < matches.size(); i++)
        {
            matches[i].queryIdx = static_cast<int>(i);
        }
    }

    std::sort(matches.begin(), matches.end(), [](cv::DMatch const & first, cv::DMatch const & second) {
        return first.distance < second.distance; });

    sortedMatchIdx.resize(matches.size());
    for (i = 0; i < matches.size(); i++)
    {
        sortedMatchIdx[i] = (unsigned int)matches[i].queryIdx;
    }
}

/* Checks if a 3x3 matrix is a rotation matrix
*
* cv::Mat R				Input  -> Rotation matrix
*
* Return:				true or false
*/
bool isMatRoationMat(const cv::Mat& R)
{
    CV_Assert(!R.empty());

    Eigen::Matrix3d Re;
    cv::cv2eigen(R, Re);

    return isMatRoationMat(Re);
}

/* Checks if a 3x3 matrix is a rotation matrix
*
* Matrix3d R			Input  -> Rotation matrix
*
* Return:				true or false
*/
bool isMatRoationMat(Eigen::Matrix3d R)
{
    //Check if R is a rotation matrix
    Eigen::Matrix3d R_check = (R.transpose() * R) - Eigen::Matrix3d::Identity();
    double r_det = R.determinant() - 1.0;

    return R_check.isZero(1e-3) && poselib::nearZero(r_det);
}

/* Calculates the Sampson L2 error for 1 correspondence
*
* InputArray E			Input  -> Essential matrix
* InputArray x1			Input  -> First point correspondence
* InputArray x2			Input  -> Second point correspondence
*
* Return:				Sampson L2 error
*/
double getSampsonL2Error(cv::InputArray E, cv::InputArray x1, cv::InputArray x2)
{
    CV_Assert(x1.rows() == x2.rows() && x1.cols() == x2.cols() && ((x1.cols() < 4 && x1.rows() == 1) || (x1.cols() == 1 && x1.rows() < 4)) && x1.type() == CV_64FC1 && x2.type() == CV_64FC1);
    Eigen::Vector3d x1e, x2e;
    Eigen::Matrix3d Ee;
    cv::Mat x1_ = cv::Mat::ones(3, 1, CV_64FC1);
    cv::Mat x2_ = cv::Mat::ones(3, 1, CV_64FC1);
    if (x1.cols() > x1.rows())
    {
        if (x1.cols() == 2)
        {
            x1_.rowRange(0, 2) = x1.getMat().t();
            x2_.rowRange(0, 2) = x2.getMat().t();
        }
        else
        {
            x1_ = x1.getMat().t();
            x2_ = x2.getMat().t();
        }
    }
    else
    {
        if (x1.rows() == 2)
        {
            x1_.rowRange(0, 2) = x1.getMat();
            x2_.rowRange(0, 2) = x2.getMat();
        }
        else
        {
            x1_ = x1.getMat();
            x2_ = x2.getMat();
        }
    }
    cv::cv2eigen(E.getMat(), Ee);
    cv::cv2eigen(x1_, x1e);
    cv::cv2eigen(x2_, x2e);
    return getSampsonL2Error(Ee, x1e, x2e);
}

/* Calculates the Sampson L2 error for 1 correspondence
*
* Matrix3d E			Input  -> Essential matrix
* Vector3d x1			Input  -> First point correspondence
* Vector3d x2			Input  -> Second point correspondence
*
* Return:				Sampson L2 error
*/
double getSampsonL2Error(Eigen::Matrix3d E, const Eigen::Vector3d& x1, Eigen::Vector3d x2)
{
    double r, rx, ry, temp_err;
    Eigen::Vector3d x2E = x2.transpose() * E;
    r = x2E.dot(x1);
    rx = E.row(0).dot(x1);
    ry = E.row(1).dot(x1);
    temp_err = r*r / (x2E(0)*x2E(0) + x2E(1)*x2E(1) + rx*rx + ry*ry);
    return temp_err;
}

/* Checks for a given vector of error values if they are inliers or not in respect to threshold th.
*
* vector<double> error		Input  -> Error values
* double th					Input  -> Threshold (should be squared beforehand to fit for L2)
* Mat inliers				Output -> Inlier mask
*
* Return value:		number of inliers
*/
size_t getInlierMask(const std::vector<double> &error, const double &th, cv::Mat & mask)
{
    size_t n = error.size(), nr_inliers = 0;
    mask = cv::Mat::zeros(1, n, CV_8UC1);
    for (size_t i = 0; i < n; i++)
    {
        if (error[i] < th)
        {
            mask.at<bool>(i) = true;
            nr_inliers++;
        }
    }

    return nr_inliers;
}

/* Calculates the angle between two vectors
*
* Mat v1					Input  -> First vector
* Mat v1					Input  -> Second vector
* bool degree				Input  -> If true [Default], the angle is returned in degrees. Otherwise in rad.
*
* Return value:				Angle
*/
double getAnglesBetwVectors(cv::Mat v1, cv::Mat v2, bool degree)
{
    CV_Assert(v1.type() == v2.type());
    if (v1.cols > v1.rows)
        v1 = v1.t();
    if (v2.cols > v2.rows)
        v2 = v2.t();
    CV_Assert((v1.cols == v2.cols) && (v1.rows == v2.rows));
    double angle = v1.dot(v2);// std::acos(v1.dot(v2) / (cv::norm(v1) * cv::norm(v2)));
    angle /= cv::norm(v1) * cv::norm(v2);
    if(poselib::nearZero(1e5 * (angle - 1.0))){
        return 0;
    }
    angle = std::acos(angle);
    if (degree)
        angle *= 180.0 / PI;
    return angle;
}

bool searchAlternativeImgComb(const int &c1, const int &c2, const std::unordered_map<int, std::unordered_set<int>> &av_img_combs, std::vector<std::vector<std::pair<int, int>>> &track, const size_t &track_idx)
{
    if (av_img_combs.find(c1) == av_img_combs.end())
    {
        return false;
    }
    if (av_img_combs.find(c2) == av_img_combs.end())
    {
        return false;
    }
    const std::unordered_set<int> &cams_c1 = av_img_combs.at(c1);
    if (cams_c1.find(c2) != cams_c1.end())
    {
        if (track.empty())
        {
            track.emplace_back(std::vector<std::pair<int, int>>(1, std::make_pair(c1, c2)));
        }
        else
        {
            track.at(track_idx).emplace_back(std::make_pair(c1, c2));
        }
        return true;
    }
    size_t track_idx2 = 0;
    std::vector<std::vector<std::pair<int, int>>> track2;
    for (const auto &c3 : cams_c1)
    {
        if (track.empty())
        {
            track2.emplace_back(std::vector<std::pair<int, int>>(1, std::make_pair(c1, c3)));
        }
        else
        {
            const std::vector<std::pair<int, int>> &track_tmp = track.at(track_idx);
            const std::pair<int, int> &tr_back = track_tmp.back();
            if(tr_back.first == c3){
                continue;
            }
            bool cc_found = false;
            for (const auto &tr : track_tmp)
            {
                if(tr.first == c1 && tr.second == c3){
                    cc_found = true;
                    break;
                }
            }
            if (cc_found){
                continue;
            }

            track2.emplace_back(track_tmp);
            track2.back().emplace_back(std::make_pair(c1, c3));
        }

        const size_t nr_before = track2.size();
        if (searchAlternativeImgComb(c3, c2, av_img_combs, track2, track_idx2))
        {
            const size_t nr_after = track2.size();
            track_idx2 += nr_after - nr_before + 1;
        }
        else
        {
            track2.pop_back();
        }
    }
    if (!track2.empty())
    {
        size_t i = 0;
        if (!track.empty())
        {
            track.at(track_idx) = track2.at(0);
            i++;
        }
        for (; i < track2.size(); i++)
        {
            track.emplace_back(track2.at(i));
        }
    }
    return track_idx2 > 0;
}

bool getAvailablePairCombinationsForMissing(const std::unordered_set<std::pair<int, int>, pair_hash, pair_EqualTo> &missed_img_combs, const std::unordered_set<std::pair<int, int>, pair_hash, pair_EqualTo> &available_img_combs, std::unordered_map<std::pair<int, int>, std::vector<std::vector<std::pair<int, int>>>, pair_hash, pair_EqualTo> &missing_restore_sequences)
{
    if (missed_img_combs.empty()){
        return true;
    }
    if (available_img_combs.empty())
    {
        return false;
    }
    //Check if the missing combination can be restored from the rest
    std::unordered_map<int, std::unordered_set<int>> available_img_combs2;
    for (const auto &av : available_img_combs)
    {
        if (available_img_combs2.find(av.first) != available_img_combs2.end())
        {
            available_img_combs2.at(av.first).emplace(av.second);
        }
        else
        {
            std::unordered_set<int> tmp;
            tmp.emplace(av.second);
            available_img_combs2.emplace(av.first, std::move(tmp));
        }
        if (available_img_combs2.find(av.second) != available_img_combs2.end())
        {
            available_img_combs2.at(av.second).emplace(av.first);
        }
        else
        {
            std::unordered_set<int> tmp;
            tmp.emplace(av.first);
            available_img_combs2.emplace(av.second, std::move(tmp));
        }
    }
    for (const auto &miss : missed_img_combs)
    {
        int c1, c2;
        if (miss.first < miss.second)
        {
            c1 = miss.first;
            c2 = miss.second;
        }
        else
        {
            c1 = miss.second;
            c2 = miss.first;
        }
        std::vector<std::vector<std::pair<int, int>>> track;
        if (!searchAlternativeImgComb(c1, c2, available_img_combs2, track, 0))
        {
            return false;
        }
        missing_restore_sequences.emplace(miss, std::move(track));
    }
    return true;
}
}
