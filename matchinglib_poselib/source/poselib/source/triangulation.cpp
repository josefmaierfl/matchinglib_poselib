//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2021 Josef Maier
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

#include "poselib/triangulation.h"
// #include <utils_common.h>
// #include <utils_cv.h>
#include <iostream>
#include <numeric>
#include <map>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/sfm.hpp>
#include "poselib/pose_helper.h"
#include "five-point-nister/five-point.hpp"

using namespace std;

namespace poselib
{
    cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point2d u,  //image point (u,v)
                                                    cv::Matx34d P,  //projection 1 matrix
                                                    cv::Point2d u1, //image point in 2nd camera
                                                    cv::Matx34d P1, //projection 2 matrix
                                                    cv::Mat K1,
                                                    cv::Mat K2)
    {
        cv::Mat K1i = K1.inv();
        cv::Mat K2i = K2.inv();
        cv::Point3d u11(u.x, u.y, 1.0), u12(u1.x, u1.y, 1.0);
        cv::Mat_<double> um1 = K1i * cv::Mat_<double>(u11);
        cv::Point2d uc1, uc2;
        uc1.x = um1(0);
        uc1.y = um1(1);

        um1 = K2i * cv::Mat_<double>(u12);
        uc2.x = um1(0);
        uc2.y = um1(1);

        return IterativeLinearLSTriangulation(uc1, P, uc2, P1);
    }

    cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point2d u,  //image point (u,v)
                                                    cv::Matx34d P,  //projection 1 matrix
                                                    cv::Point2d u1, //image point in 2nd camera
                                                    cv::Matx34d P1  //projection 2 matrix
    )
    {
        /* From https://stackoverflow.com/questions/30493928/opencv-triangulatepoints-strange-result
    * which originates from https://github.com/MasteringOpenCV/code/tree/master/Chapter4_StructureFromMotion and
    * https://github.com/MasteringOpenCV/code/blob/master/Chapter4_StructureFromMotion/Triangulation.cpp , respectively
    * Input correspondences should be normalized (in the camera coordinate system)
  */

        double wi = 1, wi1 = 1;
        cv::Mat_<double> X(4, 1);
        const static float EPSILON = 0.0001f;

        for (int i = 0; i < 10; i++)
        { //Hartley suggests 10 iterations at most
            cv::Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
            X(0) = X_(0);
            X(1) = X_(1);
            X(2) = X_(2);
            X(3) = 1.0;
            //recalculate weights
            double p2x = cv::Mat_<double>(cv::Mat_<double>(P).row(2) * X)(0);
            double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2) * X)(0);

            //breaking point
            if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON)
                break;

            wi = p2x;
            wi1 = p2x1;

            //reweight equations and solve
            cv::Matx43d A((u.x * P(2, 0) - P(0, 0)) / wi, (u.x * P(2, 1) - P(0, 1)) / wi, (u.x * P(2, 2) - P(0, 2)) / wi,
                          (u.y * P(2, 0) - P(1, 0)) / wi, (u.y * P(2, 1) - P(1, 1)) / wi, (u.y * P(2, 2) - P(1, 2)) / wi,
                          (u1.x * P1(2, 0) - P1(0, 0)) / wi1, (u1.x * P1(2, 1) - P1(0, 1)) / wi1, (u1.x * P1(2, 2) - P1(0, 2)) / wi1,
                          (u1.y * P1(2, 0) - P1(1, 0)) / wi1, (u1.y * P1(2, 1) - P1(1, 1)) / wi1, (u1.y * P1(2, 2) - P1(1, 2)) / wi1);
            cv::Mat_<double> B = (cv::Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)) / wi,
                                  -(u.y * P(2, 3) - P(1, 3)) / wi,
                                  -(u1.x * P1(2, 3) - P1(0, 3)) / wi1,
                                  -(u1.y * P1(2, 3) - P1(1, 3)) / wi1);

            cv::solve(A, B, X_, cv::DECOMP_SVD);
            X(0) = X_(0);
            X(1) = X_(1);
            X(2) = X_(2);
            X(3) = 1.0;
        }

        return X;
    }

    cv::Mat_<double> LinearLSTriangulation(cv::Point2d u,  //image point (u,v)
                                           cv::Matx34d P,  //projection 1 matrix
                                           cv::Point2d u1, //image point in 2nd camera
                                           cv::Matx34d P1  //projection 2 matrix
    )
    {
        /* From https://stackoverflow.com/questions/30493928/opencv-triangulatepoints-strange-result
    * which originates from https://github.com/MasteringOpenCV/code/tree/master/Chapter4_StructureFromMotion and
    * https://github.com/MasteringOpenCV/code/blob/master/Chapter4_StructureFromMotion/Triangulation.cpp , respectively
  */
        //build matrix A for homogenous equation system Ax = 0
        //assume X = (x,y,z,1), for Linear-LS method
        //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
        cv::Matx43d A(u.x * P(2, 0) - P(0, 0), u.x * P(2, 1) - P(0, 1), u.x * P(2, 2) - P(0, 2),
                      u.y * P(2, 0) - P(1, 0), u.y * P(2, 1) - P(1, 1), u.y * P(2, 2) - P(1, 2),
                      u1.x * P1(2, 0) - P1(0, 0), u1.x * P1(2, 1) - P1(0, 1), u1.x * P1(2, 2) - P1(0, 2),
                      u1.y * P1(2, 0) - P1(1, 0), u1.y * P1(2, 1) - P1(1, 1), u1.y * P1(2, 2) - P1(1, 2));
        cv::Mat_<double> B = (cv::Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)),
                              -(u.y * P(2, 3) - P(1, 3)),
                              -(u1.x * P1(2, 3) - P1(0, 3)),
                              -(u1.y * P1(2, 3) - P1(1, 3)));

        cv::Mat_<double> X;
        cv::solve(A, B, X, cv::DECOMP_SVD);

        return X;
    }

    /* Recovers the rotation and translation from an essential matrix and triangulates the given correspondences to form 3D coordinates.
    * If the given essential matrix corresponds to a translational essential matrix, set "translatE" to true. Moreover 3D coordintes with
    * a z-value lager than "dist" are marked as invalid within "mask" due to their numerical instability (such 3D points are also not
    * considered in the returned number of valid 3D points.
    *
    * InputArray E							Input  -> Essential matrix
    * InputArray p1						Input  -> Observed point coordinates of the left image in the camera coordinate system
    *												  (n rows, 2 cols)
    * InputArray p2						Input  -> Observed point coordinates of the right image in the camera coordinate system
    *												  (n rows, 2 cols)
    * OutputArray R						Output -> Rotation matrix
    * OutputArray t						Output -> Translation vector (3 rows x 1 column)
    * OutputArray Q						Output -> Triangulated 3D-points including invalid points (n rows x 3 columns)
    * InputOutputArray mask				I/O	   -> Inlier mask / Valid 3D points [Default=noArray()]
    * double dist							Input  -> Threshold on the distance of the normalized 3D coordinates to the camera [Default=50]
    * bool translatE						Input  -> Should be true, if a translational essential matrix is given (R corresponds to identity)
    *												  [Default=false]
    *
    * Return value:						>=0:	Number of valid 3D points
    *										-1:		R, t, or Q are mandatory output variables (one or more are missing)
    */
    int getPoseTriangPts(cv::InputArray E,
            cv::InputArray p1,
            cv::InputArray p2,
            cv::OutputArray R,
            cv::OutputArray t,
            cv::OutputArray Q,
            cv::InputOutputArray mask,
            const double dist,
            bool translatE)
    {
        int n;
        Mat R_, t_, Q_;
        if(!R.needed() || !t.needed() || !Q.needed())
            return -1;

        n = recoverPose( E.getMat(), p1.getMat(), p2.getMat(), R_, t_, Q_, mask, dist, translatE ? getTfromTransEssential(E.getMat()):(cv::noArray()));

        if(R.empty())
        {
            R.create(3, 3, CV_64F);
        }
        R_.copyTo(R.getMat());

        if(t.empty())
        {
            t.create(3, 1, CV_64F);
        }
        t_.copyTo(t.getMat());

        Q.create(Q_.size(),Q_.type());
        Q_.copyTo(Q.getMat());

        return n;
    }

    /* Triangulates 3D-points from correspondences with provided R and t. The world coordinate
    * system is located in the left camera centre.
    *
    * InputArray R							Input  -> Rotation matrix R
    * InputArray t							Input  -> Translation vector t
    * InputArray _points1				Input  -> Image projections in the left camera
    *											  in camera coordinates (1 projection per row)
    * InputArray _points2				Input  -> Image projections in the right camera
    *											  in camera coordinates (1 projection per row)
    * Mat & Q3D						Output -> Triangulated 3D points (1 coordinate per row)
    * Mat & mask						Output -> Mask marking points near infinity (mask(i) = 0)
    * double dist						Input  -> Optional threshold (Default: 50.0) for far points (near infinity)
    *
    * Return value:					>= 0:	Valid triangulated 3D-points
    *									  -1:	The matrix for the 3D points must be provided
    */
    int triangPts3D(cv::InputArray R, cv::InputArray t, cv::InputArray _points1, cv::InputArray _points2, cv::OutputArray Q3D, cv::InputOutputArray mask, const double dist)
    {
        Mat points1, points2;
        points1 = _points1.getMat();
        points2 = _points2.getMat();
        Mat R_ = R.getMat();
        Mat t_ = t.getMat();
        Mat mask_;
    //	int npoints = points1.checkVector(2);
        if(mask.needed())
        {
            mask_ = mask.getMat();
        }

        points1 = points1.t();
        points2 = points2.t();

        Mat P0 = Mat::eye(3, 4, R_.type());
        Mat P1(3, 4, R_.type());
        P1(Range::all(), Range(0, 3)) = R_ * 1.0;
        P1.col(3) = t_ * 1.0;

        // Notice here a threshold dist is used to filter
        // out far away points (i.e. infinite points) since
        // there depth may vary between postive and negtive.
        //const double dist = 50.0;
        Mat Q1,q1;
        triangulatePoints(P0, P1, points1, points2, Q1);
        if(Q1.empty() || (Q1.cols == 0)){
            return -1;
        }

        q1 = P1 * Q1;
        if(mask_.empty())
        {
            mask_ = (q1.row(2).mul(Q1.row(3)) > 0);
        }
        else
        {
            mask_ = (q1.row(2).mul(Q1.row(3)) > 0) & mask_;
        }

        Q1.row(0) /= Q1.row(3);
        Q1.row(1) /= Q1.row(3);
        Q1.row(2) /= Q1.row(3);
        mask_ = (Q1.row(2) < dist) & (Q1.row(2) > 0) & mask_;

        if(!Q3D.needed())
            return -1; //The matrix for the 3D points must be provided

        Q3D.create(Q1.cols, 3, Q1.type());
        Mat Q3D_tmp = Q3D.getMat();
        Q3D_tmp = Q1.rowRange(0,3).t();

        /*points1 = points1.t();
        points2 = points2.t();

        double scale = 0;
        double scale1 = 0;
        Mat Qs;
        for(int i = 0; i < points1.rows;i++)
        {
            scale = points1.at<double>(i,0) * Q3D.at<double>(i,2) / Q3D.at<double>(i,0);
            scale += points1.at<double>(i,1) * Q3D.at<double>(i,2) / Q3D.at<double>(i,1);
            scale /= 2;
            if(abs(scale - 1.0) > 0.001)
            {
                Q3D.row(i) *= scale;
                scale = points1.at<double>(i,0) * Q3D.at<double>(i,2) / Q3D.at<double>(i,0);
                scale += points1.at<double>(i,1) * Q3D.at<double>(i,2) / Q3D.at<double>(i,1);
                scale /= 2;
            }
            Qs = R_*Q3D.row(i).t() + t_;
            scale1 = points2.at<double>(i,0) * Qs.at<double>(2) / Qs.at<double>(0);
            scale1 += points2.at<double>(i,1) * Qs.at<double>(2) / Qs.at<double>(1);
            scale1 /= 2;
        }
        //scale /= points1.rows * 2;*/

        return countNonZero(mask_);
    }

    //For correspondence in the image coordinate sytem
    cv::Mat_<double> triangulatePoints_img(const cv::Mat &R, const cv::Mat &t, const cv::Mat &K1, const cv::Mat &K2, const cv::Mat &p1, const cv::Mat &p2, const double &th, cv::InputOutputArray mask)
    {
        cv::Matx34d P1 = cv::Matx34d::eye();
        cv::Matx34d P2(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
                       R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
                       R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2));
        cv::Mat p1c, p2c;
        imgToCamCordinatesAndMeasPerRow(p1, p2, K1, K2, p1c, p2c);

        return triangulatePoints(P1, P2, p1c, p2c, th, mask);
    }

    //For correspondence in camera coordinates
    cv::Mat_<double> triangulatePoints_cam(const cv::Mat &R, const cv::Mat &t, const cv::Mat &p1, const cv::Mat &p2, const double &th, cv::InputOutputArray mask)
    {
        cv::Matx34d P1 = cv::Matx34d::eye();
        cv::Matx34d P2(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
                       R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
                       R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2));
        return triangulatePoints(P1, P2, p1, p2, th, mask);
    }

    //For correspondence in the image coordinate sytem
    cv::Mat_<double> triangulatePoints_img(const cv::Mat &R1, const cv::Mat &t1, const cv::Mat &R2, const cv::Mat &t2, const cv::Mat &K1, const cv::Mat &K2, const cv::Mat &p1, const cv::Mat &p2, const double &th, cv::InputOutputArray mask)
    {
        cv::Matx34d P1(R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0),
                       R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1),
                       R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2));
        cv::Matx34d P2(R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), t2.at<double>(0),
                       R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), t2.at<double>(1),
                       R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), t2.at<double>(2));
        cv::Mat p1c, p2c;
        imgToCamCordinatesAndMeasPerRow(p1, p2, K1, K2, p1c, p2c);
        return triangulatePoints(P1, P2, p1c, p2c, th, mask);
    }

    //For correspondence in camera coordinates
    cv::Mat_<double> triangulatePoints_cam(const cv::Mat &R1, const cv::Mat &t1, const cv::Mat &R2, const cv::Mat &t2, const cv::Mat &p1, const cv::Mat &p2, const double &th, cv::InputOutputArray mask)
    {
        cv::Matx34d P1(R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0),
                       R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1),
                       R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2));
        cv::Matx34d P2(R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), t2.at<double>(0),
                       R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), t2.at<double>(1),
                       R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), t2.at<double>(2));
        return triangulatePoints(P1, P2, p1, p2, th, mask);
    }

    cv::Mat_<double> triangulatePoints(const cv::Matx34d &P1, const cv::Matx34d &P2, const cv::Mat &p1, const cv::Mat &p2, const double &th, cv::InputOutputArray mask)
    {
        int nr_cols = p1.cols;
        int nr_rows = p1.rows;
        int max_size = max(nr_cols, nr_rows);
        int min_size = min(nr_cols, nr_rows);
        const bool mask_needed = mask.needed() && (th > DBL_EPSILON);
        cv::Mat mask_;
        if (mask_needed)
        {
            if (mask.empty() && !(max_size == 2 && min_size == 1))
            {
                mask.create(1, max_size, CV_8UC1);
            }
            else if (mask.empty())
            {
                mask.create(1, 1, CV_8UC1);
                mask.setTo(1);
            }
            else if (mask.type() != CV_8UC1)
            {
                throw runtime_error("Triangulation mask is of wrong type.");
            }
            mask_ = mask.getMat();
        }
        if (p1.type() != CV_64FC1)
        {
            throw runtime_error("Triangulation needs 2D image coordinates of type double.");
        }
        if (max_size == 2 && min_size == 1)
        {
            cv::Point2d p1d(p1);
            cv::Point2d p2d(p2);
            cv::Mat Q = IterativeLinearLSTriangulation(p1d, P1, p2d, P2);
            return Q.rowRange(0, 3).t();
        }
        if (min_size != 2)
        {
            throw runtime_error("Triangulation needs 2D image coordinates, not homogeneous coordinates");
        }
        bool per_row = nr_cols > nr_rows ? false : true;
        if (nr_cols == nr_rows)
        {
            cout << "WARNING! Only 2 measurements for triangulation available. Assuming 1 measurement per row." << endl;
        }
        cv::Mat Q;
        for (int i = 0; i < max_size; ++i)
        {
            cv::Point2d p1d, p2d;
            if (per_row)
            {
                p1d = cv::Point2d(p1.row(i));
                p2d = cv::Point2d(p2.row(i));
            }
            else
            {
                p1d = cv::Point2d(p1.col(i));
                p2d = cv::Point2d(p2.col(i));
            }
            cv::Mat X = IterativeLinearLSTriangulation(p1d, P1, p2d, P2);
            if (mask_needed)
            {
                const bool is_inl = isInlier(P1, X, p1d, th) && isInlier(P2, X, p2d, th);
                mask_.at<unsigned char>(i) = is_inl ? 1 : 0;
            }
            Q.push_back(X.rowRange(0, 3).t());
        }
        return Q;
    }

    void imgToCamCordinatesAndMeasPerRow(cv::InputArray p1, cv::InputArray p2, cv::InputArray K1, cv::InputArray K2, cv::OutputArray p1c, cv::OutputArray p2c)
    {
        cv::Mat p1_ = p1.getMat();
        cv::Mat p2_ = p2.getMat();
        CV_Assert(p1_.rows == p2_.rows && p1_.cols == p2_.cols);
        CV_Assert(p1_.type() == CV_64FC1 && p1_.type() == p2_.type());
        cv::Mat Ki1 = K1.getMat().inv();
        cv::Mat Ki2 = K2.getMat().inv();
        cv::Mat Ki1t = Ki1.t();
        cv::Mat Ki2t = Ki2.t();
        int nr_cols = p1_.cols;
        int nr_rows = p1_.rows;
        int max_size = max(nr_cols, nr_rows);
        int min_size = min(nr_cols, nr_rows);
        bool per_row = true;
        if ((max_size == 2 && min_size == 1 && nr_cols < nr_rows) || (max_size > 2 && nr_cols > nr_rows))
        {
            per_row = false;
        }
        if (max_size == 2 && min_size == 1)
        {
            max_size = 1;
            min_size = 2;
        }
        if (nr_cols == nr_rows)
        {
            cout << "WARNING! Only 2 measurements for triangulation available. Assuming 1 measurement per row." << endl;
        }
        if (!p1c.empty())
        {
            p1c.release();
        }
        p1c.create(max_size, 2, CV_64FC1);
        if (!p2c.empty())
        {
            p2c.release();
        }
        p2c.create(max_size, 2, CV_64FC1);
        cv::Mat p1c_ = p1c.getMat(), p2c_ = p2c.getMat();
        for (int i = 0; i < max_size; ++i)
        {
            if (per_row)
            {
                cv::Mat p1i = (cv::Mat_<double>(1, 3) << p1_.at<double>(i, 0), p1_.at<double>(i, 1), 1.0);
                p1i = p1i * Ki1t;
                p1i.colRange(0, 2).copyTo(p1c_.row(i));
                cv::Mat p2i = (cv::Mat_<double>(1, 3) << p2_.at<double>(i, 0), p2_.at<double>(i, 1), 1.0);
                p2i = p2i * Ki2t;
                p2i.colRange(0, 2).copyTo(p2c_.row(i));
            }
            else
            {
                cv::Mat p1i = (cv::Mat_<double>(1, 3) << p1_.at<double>(0, i), p1_.at<double>(1, i), 1.0);
                p1i = p1i * Ki1t;
                p1i.colRange(0, 2).copyTo(p1c_.row(i));
                cv::Mat p2i = (cv::Mat_<double>(1, 3) << p2_.at<double>(0, i), p2_.at<double>(1, i), 1.0);
                p2i = p2i * Ki2t;
                p2i.colRange(0, 2).copyTo(p2c_.row(i));
            }
        }
    }

    // x1 and x2 must be in the camera coordinate system (not in the image coordinate system) if K1 and K2 are not provided
    cv::Mat_<double> getPose_triangulatePoints(cv::InputArray E, cv::InputArray x1, cv::InputArray x2, cv::OutputArray R, cv::OutputArray t, cv::InputOutputArray mask, cv::InputArray K1, cv::InputArray K2, const bool noTriangulation, const double &depth_th, cv::InputArray R_hint, cv::InputArray t_hint, std::vector<cv::Mat> *Rs_possible, std::vector<cv::Mat> *ts_possible, const double &th)
    {
        cv::Mat p1, p2;
        cv::Mat R1, R2, t1;
        const bool check_inl = th > DBL_EPSILON;
        cv::decomposeEssentialMat(E, R1, R2, t1);
        double norm_t = cv::norm(t1);
        if (!nearZero(1.0 - norm_t))
        {
            t1 /= norm_t;
        }

        cv::Matx34d P1 = cv::Matx34d::eye();
        cv::Matx34d P2x[4];
        P2x[0] = cv::Matx34d(R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0),
                             R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1),
                             R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2));
        P2x[1] = cv::Matx34d(R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), -1. * t1.at<double>(0),
                             R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), -1. * t1.at<double>(1),
                             R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), -1. * t1.at<double>(2));
        P2x[2] = cv::Matx34d(R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), t1.at<double>(0),
                             R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), t1.at<double>(1),
                             R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), t1.at<double>(2));
        P2x[3] = cv::Matx34d(R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), -1. * t1.at<double>(0),
                             R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), -1. * t1.at<double>(1),
                             R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), -1. * t1.at<double>(2));
        if (!K1.empty() && !K2.empty())
        {
            imgToCamCordinatesAndMeasPerRow(x1, x2, K1, K2, p1, p2);
        }
        else
        {
            p1 = x1.getMat();
            p2 = x2.getMat();
        }
        int nr_cols = p1.cols;
        int nr_rows = p1.rows;
        int max_size = max(nr_cols, nr_rows);
        int min_size = min(nr_cols, nr_rows);
        if (p1.type() != CV_64FC1)
        {
            throw runtime_error("Triangulation needs 2D image coordinates of type double.");
        }

        // Do the cheirality check.
        // Notice here a threshold dist is used to filter
        // out far away points (i.e. infinite points) since
        // there depth may vary between positive and negative.
        std::array<int, 4> v{{0, 0, 0, 0}};
        if (max_size == 2 && min_size == 1)
        {
            cv::Point2d p1d(p1);
            cv::Point2d p2d(p2);
            cv::Mat Xs[4];
            std::array<std::array<double, 2>, 4> depths;

            for (int i = 0; i < 4; ++i)
            {
                Xs[i] = IterativeLinearLSTriangulation(p1d, P1, p2d, P2x[i]);
                depths[i][0] = CalculateDepth(P1, Xs[i]);
                depths[i][1] = CalculateDepth(P2x[i], Xs[i]);
                v[i] = (depths[i][0] > DBL_EPSILON && depths[i][1] > DBL_EPSILON) ? 1 : 0;
            }
            if (accumulate(v.begin(), v.end(), 0) > 1)
            {
                for (int i = 0; i < 4; ++i)
                {
                    v[i] -= (depths[i][0] > depth_th || depths[i][1] > depth_th) ? 1 : 0;
                }
                if (accumulate(v.begin(), v.end(), 0) != 1)
                {
                    throw runtime_error("Unable to determine correct R & t as only 1 correspondence was provided");
                }
            }
            auto it = max_element(v.begin(), v.end());
            auto pos = std::distance(v.begin(), it);
            if (R.empty())
            {
                R.create(3, 3, CV_64FC1);
            }
            cv::Mat R_ = R.getMat();
            cv::Mat P2 = cv::Mat(P2x[pos]);
            P2(cv::Range::all(), cv::Range(0, 3)).copyTo(R_);
            if (t.empty())
            {
                t.create(3, 1, CV_64FC1);
            }
            cv::Mat t_ = t.getMat();
            P2(cv::Range::all(), cv::Range(3, 4)).copyTo(t_);
            return Xs[pos].rowRange(0, 3).t();
        }
        if (min_size != 2)
        {
            throw runtime_error("Triangulation needs 2D image coordinates, not homogeneous coordinates");
        }
        bool per_row = nr_cols > nr_rows ? false : true;
        if (nr_cols == nr_rows)
        {
            cout << "WARNING! Only 2 measurements for triangulation available. Assuming 1 measurement per row." << endl;
        }

        cv::Mat Qx[4];
        for (int i = 0; i < max_size; ++i)
        {
            cv::Point2d p1d, p2d;
            if (per_row)
            {
                p1d = cv::Point2d(p1.row(i));
                p2d = cv::Point2d(p2.row(i));
            }
            else
            {
                p1d = cv::Point2d(p1.col(i));
                p2d = cv::Point2d(p2.col(i));
            }

            for (int j = 0; j < 4; ++j)
            {
                cv::Mat Xj = IterativeLinearLSTriangulation(p1d, P1, p2d, P2x[j]);
                const double d1 = CalculateDepth(P1, Xj);
                const double d2 = CalculateDepth(P2x[j], Xj);
                v[j] += ((d1 > DBL_EPSILON && d2 > DBL_EPSILON) && (d1 < depth_th && d2 < depth_th)) ? 1 : 0;
                Qx[j].push_back(Xj.rowRange(0, 3).t());
            }

            if (i > 10)
            {
                vector<int> vv(v.begin(), v.end());
                sort(vv.begin(), vv.end());
                if (vv[3] == 0)
                {
                    continue;
                }
                float fact = static_cast<float>(vv[2]) / static_cast<float>(vv[3]);
                if (fact < 0.33 || (i > 50 && fact < 0.5))
                {
                    break;
                }
            }
        }
        vector<int> vv(v.begin(), v.end());
        sort(vv.begin(), vv.end());
        int v_th = static_cast<int>(round(0.9f * static_cast<float>(vv[3])));
        if (vv[2] >= v_th)
        {
            bool use_th_check = true;
            if (!R_hint.empty() && !t_hint.empty())
            {
                std::array<double, 4> err_rt{{0, 0, 0, 0}};
                cv::Mat R_hint_ = R_hint.getMat();
                cv::Mat t_hint_ = t_hint.getMat();
                for (int j = 0; j < 4; ++j)
                {
                    double err_R, err_t;
                    cv::Mat P_tmp(P2x[j]);
                    cv::Mat R_tmp = P_tmp(cv::Range::all(), cv::Range(0, 3));
                    cv::Mat t_tmp = P_tmp(cv::Range::all(), cv::Range(3, 4));
                    poselib::compareRTs(R_tmp, R_hint_, t_tmp, t_hint_, &err_R, &err_t);
                    err_rt.at(j) = std::abs(err_R) + std::abs(err_t);
                }
                auto it = min_element(err_rt.begin(), err_rt.end());
                auto pos = std::distance(err_rt.begin(), it);
                v[pos] *= 2;
                vv = vector<int>(v.begin(), v.end());
                sort(vv.begin(), vv.end());
                if (vv[2] != vv[3])
                {
                    use_th_check = false;
                }
            }
            else if (Rs_possible && ts_possible)
            {
                if (!Rs_possible->empty() || !ts_possible->empty())
                {
                    Rs_possible->clear();
                    ts_possible->clear();
                }
                auto it = max_element(v.begin(), v.end());
                auto pos = std::distance(v.begin(), it);
                size_t pos2 = 4, hcnt = 0;
                for (size_t j = 0; j < 4; ++j)
                {
                    if (static_cast<long int>(j) == pos)
                    {
                        continue;
                    }
                    if (v[j] >= v_th)
                    {
                        hcnt++;
                        pos2 = j;
                    }
                }
                if (pos2 < 4 && hcnt == 1)
                {
                    Rs_possible->emplace_back(cv::Mat(P2x[pos])(cv::Range::all(), cv::Range(0, 3)).clone());
                    Rs_possible->emplace_back(cv::Mat(P2x[pos2])(cv::Range::all(), cv::Range(0, 3)).clone());
                    ts_possible->emplace_back(cv::Mat(P2x[pos])(cv::Range::all(), cv::Range(3, 4)).clone());
                    ts_possible->emplace_back(cv::Mat(P2x[pos2])(cv::Range::all(), cv::Range(3, 4)).clone());
                }
            }
            if (use_th_check)
            {
                const double depth_th1 = 0.55 * depth_th;
                if (depth_th1 > 40.)
                {
                    return getPose_triangulatePoints(E, x1, x2, R, t, mask, K1, K2, noTriangulation, depth_th1);
                }
                throw runtime_error("Unable to determine correct R & t as too less correspondences were provided");
            }
        }
        auto it = max_element(v.begin(), v.end());
        auto pos = std::distance(v.begin(), it);
        cv::Matx34d P2 = P2x[pos];
        if (R.empty())
        {
            R.create(3, 3, CV_64FC1);
        }
        cv::Mat R_ = R.getMat(), P2_ = cv::Mat(P2);
        P2_(cv::Range::all(), cv::Range(0, 3)).copyTo(R_);
        if (t.empty())
        {
            t.create(3, 1, CV_64FC1);
        }
        cv::Mat t_ = t.getMat();
        P2_(cv::Range::all(), cv::Range(3, 4)).copyTo(t_);

        cv::Mat Q = Qx[pos];
        if (!noTriangulation)
        {
            int start = Q.rows;
            for (int i = start; i < max_size; ++i)
            {
                cv::Point2d p1d, p2d;
                if (per_row)
                {
                    p1d = cv::Point2d(p1.row(i));
                    p2d = cv::Point2d(p2.row(i));
                }
                else
                {
                    p1d = cv::Point2d(p1.col(i));
                    p2d = cv::Point2d(p2.col(i));
                }
                cv::Mat X = IterativeLinearLSTriangulation(p1d, P1, p2d, P2);
                Q.push_back(X.rowRange(0, 3).t());
            }
        }

        if (mask.needed())
        {
            bool transpose = false;
            if (mask.empty())
            {
                mask.create(1, Q.rows, CV_8UC1);
                mask.getMat().setTo(UCHAR_MAX);
            }
            cv::Mat mask_ = mask.getMat();
            if (mask_.rows > mask_.cols)
            {
                mask_ = mask_.t();
                transpose = true;
            }
            vector<double> d1, d2;
            cv::Mat inliers(1, Q.rows, CV_8UC1);
            for (int j = 0; j < Q.rows; ++j)
            {
                cv::Mat Q1 = (cv::Mat_<double>(4, 1) << Q.at<double>(j, 0), Q.at<double>(j, 1), Q.at<double>(j, 2), 1.);
                d1.emplace_back(CalculateDepth(P1, Q1));
                d2.emplace_back(CalculateDepth(P2, Q1));
                if (check_inl)
                {
                    cv::Mat p1x, p2x;
                    if (per_row)
                    {
                        p1x = p1.row(j).t();
                        p2x = p2.row(j).t();
                    }
                    else
                    {
                        p1x = p1.col(j);
                        p2x = p2.col(j);
                    }
                    const bool ii = isInlier(P1, Q1, p1x, th) && isInlier(P2, Q1, p2x, th);
                    inliers.at<unsigned char>(j) = ii ? __UINT8_MAX__ : 0;
                }
            }
            cv::Mat d1_cv(d1);
            cv::Mat d2_cv(d2);
            if (d1_cv.rows > d1_cv.cols)
            {
                d1_cv = d1_cv.t();
                d2_cv = d2_cv.t();
            }
            mask_ = (d1_cv > DBL_EPSILON) & (d2_cv > DBL_EPSILON) & mask_;
            mask_ = (d1_cv < depth_th) & (d2_cv < depth_th) & mask_;
            mask_ &= inliers;
            if (transpose)
            {
                mask.getMat() = mask_.t();
            }
        }

        return Q;
    }

    double CalculateDepth(const cv::Matx34d &P, const cv::Mat &point3Dh)
    {
        cv::Mat Pr2 = cv::Mat(P.row(2)).t();
        const double proj_z = Pr2.dot(point3Dh);
        return proj_z * cv::norm(P.col(2));
    }

    bool isInlier(const cv::Matx34d &P, const cv::Mat &point3Dh, const cv::Point2d &point2D, const double &th)
    {
        return isInlier(P, point3Dh, cv::Mat(point2D, false).reshape(1, 2), th);
    }

    bool isInlier(const cv::Matx34d &P, const cv::Mat &point3Dh, const cv::Mat &point2D, const double &th)
    {
        cv::Mat x = P * point3Dh;
        x /= x.at<double>(2);
        cv::Mat x_diff = x.rowRange(0, 2) - point2D;
        const double err = cv::norm(x_diff);
        return err < th;
    }

    bool triangulateMultProjections(const int &imgId, const std::vector<cv::Point2d> &ps_ud, const std::vector<int> &camIdxs, const std::vector<cv::Mat> &Pi, const std::vector<cv::Mat> &Ri, const std::vector<cv::Mat> &ti, const std::vector<cv::Mat> &Ki, const std::vector<cv::Point2d> &ps, cv::Mat &Qs, std::vector<std::pair<int, int>> &projectionInvalids, const double &th, const bool invalid_return)
    {
        const double th2 = th * th;
        cv::Mat X;
        bool larger2 = false;
        bool cont = false;
        const size_t si = ps_ud.size();
        if (si == 2)
        {
            X = IterativeLinearLSTriangulation(ps_ud.at(0), Pi.at(0), ps_ud.at(1), Pi.at(1));
            X.resize(3);
            Qs.push_back(X.t());
        }
        else
        {
            vector<cv::Mat> corrs;
            for (size_t i = 0; i < ps_ud.size(); i++)
            {
                corrs.emplace_back(cv::Mat(ps_ud[i]));
            }
            cv::sfm::triangulatePoints(corrs, Pi, X);
            Qs.push_back(X.t());
            larger2 = true;
        }
        if (X.at<double>(2) < 0. || X.at<double>(2) > 50.)
        {
            cont = true;
        }
        else
        {
            for (size_t i = 0; i < ps.size(); i++)
            {
                if (!reproject3DTh2(X, Ri[i], ti[i], Ki[i], ps.at(i), th2))
                {
                    cont = true;
                    break;
                }
            }
        }
        if (cont && larger2)
        {
            vector<cv::Mat> Q;
            for (size_t i = 0; i < ps_ud.size() - 1; i++)
            {
                for (size_t j = i + 1; j < ps_ud.size(); j++)
                {
                    cv::Mat qi = IterativeLinearLSTriangulation(ps_ud.at(i), Pi.at(i), ps_ud.at(j), Pi.at(j));
                    Q.emplace_back(qi.rowRange(0, 3));
                }
            }
            std::map<double, std::pair<size_t, std::vector<size_t>>> errs_mean;
            for (size_t j = 0; j < Q.size(); j++)
            {
                vector<double> errs;
                std::vector<size_t> iv, valids;
                for (size_t i = 0; i < ps.size(); i++)
                {
                    const double err = reproject3DDist2(Q[j], Ri[i], ti[i], Ki[i], ps.at(i));
                    if (err < th2 && Q[j].at<double>(2) > 0)
                    {
                        errs.emplace_back(err);
                        valids.emplace_back(i);
                    }
                    else
                    {
                        iv.emplace_back(i);
                    }
                }
                const size_t nr_errs = errs.size();
                if (nr_errs == ps.size())
                {
                    Qs.pop_back();
                    Qs.push_back(Q[j].t());
                    cont = false;
                    break;
                }
                else if (nr_errs > 1)
                {
                    const double nr_errs_d = static_cast<double>(nr_errs);
                    double mean_err = std::accumulate(errs.begin(), errs.end(), 0.0);
                    // if (nr_errs == 2){
                    //     if (mean_err > 1.0){
                    //         mean_err /= nr_errs_d;
                    //     }
                    //     else if (mean_err < 0.5){
                    //         mean_err *= 1.5;
                    //     }
                    // }
                    // else{
                    const double mult = (1.0 - std::pow(0.95, nr_errs)) * nr_errs_d;
                    // const double mult = std::pow(1.2, nr_errs);
                    mean_err /= mult * nr_errs_d * nr_errs_d;
                    // }
                    if (nr_errs > 2)
                    {
                        cv::Mat Q_new2;
                        double errs_mean_new2;
                        std::vector<size_t> iv_new2;
                        reTriangulateMultProjections(ps_ud, Pi, Ri, ti, Ki, ps, valids, Q_new2, iv_new2, errs_mean_new2, th2);
                        if (errs_mean_new2 < mean_err)
                        {
                            errs_mean.emplace(errs_mean_new2, make_pair(j, move(iv_new2)));
                            Q_new2.copyTo(Q[j]);
                        }
                        else
                        {
                            errs_mean.emplace(mean_err, make_pair(j, move(iv)));
                        }
                    }
                    else
                    {
                        errs_mean.emplace(mean_err, make_pair(j, move(iv)));
                    }
                }
            }
            if (cont && !errs_mean.empty())
            {
                std::pair<size_t, std::vector<size_t>> &best = errs_mean.begin()->second;
                Qs.pop_back();
                Qs.push_back(Q[best.first].t());
                cont = false;
                for (const auto &i : best.second)
                {
                    projectionInvalids.emplace_back(make_pair(camIdxs.at(i), imgId));
                }
            }
            else if (!invalid_return && cont)
            {
                Qs.pop_back();
            }
        }
        else if (!invalid_return && cont)
        {
            Qs.pop_back();
        }
        return !cont;
    }

    bool triangulateMultProjections(const std::vector<cv::Point2d> &ps_ud, const std::vector<std::pair<int, int>> &ciIdxs, const std::vector<cv::Mat> &Pi, const std::vector<cv::Mat> &Ki, const std::vector<cv::Point2d> &ps, cv::Mat &Qs, std::vector<std::pair<int, int>> &projectionInvalids, const double &th, const bool invalid_return)
    {
        vector<int> indices(ciIdxs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::vector<std::pair<int, int>> projectionInvalids_tmp;
        vector<cv::Mat> Ri, ti;
        for (const auto &pi : Pi)
        {
            Ri.emplace_back(pi.colRange(0, 3));
            ti.emplace_back(pi.col(3));
        }
        bool res = triangulateMultProjections(0, ps_ud, indices, Pi, Ri, ti, Ki, ps, Qs, projectionInvalids_tmp, th, invalid_return);
        if (!res)
        {
            return false;
        }
        if (!projectionInvalids_tmp.empty())
        {
            projectionInvalids.clear();
            for (const auto &i : projectionInvalids_tmp)
            {
                projectionInvalids.emplace_back(ciIdxs.at(i.first));
            }
        }
        return true;
    }

    void triangulateMultProjectionsArray(const std::vector<std::unordered_map<int, cv::Point2f>> &pts2D, const std::vector<cv::Mat> &R, const std::vector<cv::Mat> &t, const std::vector<cv::Mat> &K, const std::vector<cv::Mat> &distortion, std::vector<bool> &Qs_mask, cv::Mat &Qs, const double &th)
    {
        vector<bool> undistort;
        std::vector<cv::Mat> Kinvs, Ps, dist;
        Qs_mask.clear();
        for (size_t c = 0; c < K.size(); c++)
        {
            if (!distortion.empty()){
                undistort.emplace_back(!nearZero(cv::sum(distortion.at(c))[0]));
                dist.emplace_back(distortion.at(c));
            }
            else
            {
                undistort.emplace_back(false);
                dist.emplace_back(cv::Mat::zeros(5, 1, CV_64FC1));
            }
            Kinvs.emplace_back(K.at(c).inv());
            cv::Mat P;
            cv::hconcat(R.at(c), t.at(c), P);
            Ps.emplace_back(P);
        }

        cv::Mat Q;
        for (size_t j = 0; j < pts2D.size(); j++)
        {
            vector<cv::Point2d> ps_ud, ps;
            vector<cv::Mat> Pi, Ki;
            vector<pair<int, int>> camIdxs;
            bool cont = false;
            for (const auto &p : pts2D.at(j))
            {
                cv::Point2d pd(static_cast<double>(p.second.x), static_cast<double>(p.second.y));
                cv::Point2d p_out;
                if (!toCamCoordsUndistort(pd, Kinvs.at(p.first), undistort.at(p.first), dist.at(p.first), p_out))
                {
                    cont = true;
                    break;
                }

                ps_ud.emplace_back(move(p_out));
                ps.emplace_back(move(pd));
                const pair<int, int> ci3(p.first, 0);
                Pi.emplace_back(Ps.at(p.first));
                Ki.emplace_back(K.at(p.first));
                camIdxs.emplace_back(ci3);
            }
            if (cont)
            {
                Q.push_back(cv::Mat::zeros(1, 3, CV_64FC1));
                Qs_mask.emplace_back(false);
                continue;
            }
            std::vector<std::pair<int, int>> projectionInvalids;
            cont = triangulateMultProjections(ps_ud, camIdxs, Pi, Ki, ps, Q, projectionInvalids, th, true);
            if(cont){
                Qs_mask.emplace_back(true);
            }else{
                Qs_mask.emplace_back(false);
            }
        }
        Q.copyTo(Qs);
    }

    size_t reTriangulateMultProjections(const std::vector<cv::Point2d> &ps_ud, const std::vector<cv::Mat> &Pi, const std::vector<cv::Mat> &Ri, const std::vector<cv::Mat> &ti, const std::vector<cv::Mat> &Ki, const std::vector<cv::Point2d> &ps, const std::vector<size_t> &valids, cv::Mat &Q_new, std::vector<size_t> &iv_new, double &errs_mean_new, const double &th2, size_t tries)
    {
        vector<cv::Mat> corrs;
        vector<cv::Mat> P_tmp;
        for (const auto &i : valids)
        {
            corrs.emplace_back(cv::Mat(ps_ud[i]));
            P_tmp.emplace_back(Pi.at(i));
        }
        cv::Mat Q_new1;
        cv::sfm::triangulatePoints(corrs, P_tmp, Q_new1);
        vector<double> errs_new1;
        std::vector<size_t> iv_new1, valids_new;
        // double err_sum = 0.;
        for (size_t i = 0; i < ps.size(); i++)
        {
            const double err = reproject3DDist2(Q_new1, Ri[i], ti[i], Ki[i], ps.at(i));
            if (err < th2 && Q_new1.at<double>(2) > 0)
            {
                errs_new1.emplace_back(err);
                valids_new.emplace_back(i);
                // err_sum += err;
            }
            else
            {
                iv_new1.emplace_back(i);
            }
        }
        // if (err_sum < 200.0){
        //     const double err_th = getMedian(errs_new1) + 4.0;
        //     vector<double> errs_new12;
        //     std::vector<size_t> valids_new12;
        //     for (size_t i = 0; i < errs_new1.size(); i++)
        //     {
        //         if (errs_new1.at(i) < err_th){
        //             errs_new12.emplace_back(errs_new1.at(i));
        //             valids_new12.emplace_back(valids_new.at(i));
        //         }else{
        //             iv_new1.emplace_back(valids_new.at(i));
        //         }
        //     }
        //     errs_new1 = move(errs_new12);
        //     valids_new = move(valids_new12);
        // }
        const size_t nr_errs = errs_new1.size();
        const double nr_errs_d = static_cast<double>(nr_errs);
        double mean_err = std::accumulate(errs_new1.begin(), errs_new1.end(), 0.0);
        // if (nr_errs == 2)
        // {
        //     if (mean_err > 1.0)
        //     {
        //         mean_err /= nr_errs_d;
        //     }
        //     else if (mean_err < 0.5)
        //     {
        //         mean_err *= 1.33;
        //     }
        // }
        // else
        // {
        const double mult = (1.0 - std::pow(0.95, nr_errs)) * nr_errs_d;
        // const double mult = std::pow(1.2, nr_errs);
        mean_err /= mult * nr_errs_d * nr_errs_d;
        // }
        if (nr_errs == valids.size() || nr_errs < 3 || tries > 5)
        {
            errs_mean_new = mean_err;
            iv_new = move(iv_new1);
            Q_new1.copyTo(Q_new);
            return nr_errs;
        }
        cv::Mat Q_new2;
        double errs_mean_new2;
        std::vector<size_t> iv_new2;
        tries++;
        const size_t nr_errs2 = reTriangulateMultProjections(ps_ud, Pi, Ri, ti, Ki, ps, valids_new, Q_new2, iv_new2, errs_mean_new2, th2, tries);
        if (errs_mean_new2 < mean_err)
        {
            errs_mean_new = errs_mean_new2;
            iv_new = move(iv_new2);
            Q_new2.copyTo(Q_new);
            return nr_errs2;
        }
        errs_mean_new = mean_err;
        iv_new = move(iv_new1);
        Q_new1.copyTo(Q_new);
        return nr_errs;
    }
}