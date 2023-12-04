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

#include <utils_cv.h>
#include "opencv2/imgproc.hpp"
#include <opencv2/calib3d/calib3d.hpp>

#include <ransac.h>

#include <map>

using namespace std;

namespace utilslib
{   
    double getUsedCorrespAreaRatio(const std::vector<cv::Point2f> &pts, const cv::Size &imgSize)
    {
        vector<cv::Point2f> convexHull; // Convex hull points
        vector<cv::Point2f> contour;    // Convex hull contour points
        const double epsilon = 0.5;     // Contour approximation accuracy

        // Calculate convex hull of original points (which points positioned on the boundary)
        cv::convexHull(pts, convexHull);
        // Approximating polygonal curve to convex hull
        cv::approxPolyDP(convexHull, contour, epsilon, true);
        const double area = cv::contourArea(contour);
        const double imgArea = static_cast<double>(imgSize.area());
        return area / imgArea;
    }

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

    cv::Mat_<double> calculatePlane(const cv::Vec3d &Q1, const cv::Vec3d &Q2, const cv::Vec3d &Q3)
    {
        cv::Vec3d Q12 = Q2 - Q1;
        cv::Vec3d Q13 = Q3 - Q1;
        cv::Vec3d normal = Q12.cross(Q13);
        normal /= cv::norm(normal);
        const double d = -1.0 * Q1.dot(normal);
        return (cv::Mat_<double>(4, 1) << normal[0], normal[1], normal[2], d);
    }

    double distanceToPlane(const cv::Vec3d &Q1, const cv::Mat &plane, const bool &absDist){
        CV_Assert(plane.rows == 4 && plane.cols == 1);
        cv::Mat qh = cv::Mat::ones(4, 1, CV_64FC1);
        qh.at<double>(0) = Q1[0];
        qh.at<double>(1) = Q1[1];
        qh.at<double>(2) = Q1[2];
        double d = qh.dot(plane);
        if(absDist){
            d = abs(d);
        }
        return d / cv::norm(plane.rowRange(0, 3));
    }

    cv::Mat_<double> calculatePlaneKnownNormal(const cv::Vec3d &Q, const cv::Vec3d &planeNormal)
    {
        const double d = -1.0 * Q.dot(planeNormal);
        return (cv::Mat_<double>(4, 1) << planeNormal[0], planeNormal[1], planeNormal[2], d);
    }

    double fitCircle(const std::vector<cv::Point2d> &data, cv::Point2d &mid, double &radius)
    {
        /*
        Adapted version from: https://people.cas.uab.edu/~mosya/cl/CPPcircle.html
        One more version available at: https://github.com/SohranEliassi/Circle-Fitting-Hyper-Fit
        Circle fit to a given set of data points (in 2D)
      
        This is an algebraic fit based on the journal article
     
        A. Al-Sharadqah and N. Chernov, "Error analysis for circle fitting algorithms",
        Electronic Journal of Statistics, Vol. 3, pages 886-911, (2009)
      
        It is an algebraic circle fit with "hyperaccuracy" (with zero essential bias). 
        The term "hyperaccuracy" first appeared in papers by Kenichi Kanatani around 2006

        This method combines the Pratt and Taubin fits to eliminate the essential bias.
        
        It works well whether data points are sampled along an entire circle or
        along a small arc. 
     
        Its statistical accuracy is theoretically higher than that of the Pratt fit 
        and Taubin fit, but practically they all return almost identical circles
        (unlike the Kasa fit that may be grossly inaccurate). 
     
        It provides a very good initial guess for a subsequent geometric fit. 
     
        Nikolai Chernov  (September 2012)
        */
        size_t iter;
        const size_t IterMAX = 99;

        double Xi, Yi;
        double Mz, Mxy, Mxx, Myy, Mxz, Myz, Mzz, Cov_xy, Var_z;
        double A0, A1, A2, A22;
        double Dy, xnew, x, ynew, y;
        double DET, Xcenter, Ycenter;

        // Compute x- and y- sample means
        double meanX = 0, meanY = 0;
        const double data_n_dbl = static_cast<double>(data.size());
        for (const auto &pt : data)
        {
            meanX += pt.x;
            meanY += pt.y;
        }
        meanX /= data_n_dbl;
        meanY /= data_n_dbl;

        //computing moments
        Mxx = Myy = Mxy = Mxz = Myz = Mzz = 0.;

        for (const auto &pt : data)
        {
            Xi = pt.x - meanX;           //  centered x-coordinates
            Yi = pt.y - meanY;           //  centered y-coordinates
            const double Zi = Xi * Xi + Yi * Yi;

            Mxy += Xi * Yi;
            Mxx += Xi * Xi;
            Myy += Yi * Yi;
            Mxz += Xi * Zi;
            Myz += Yi * Zi;
            Mzz += Zi * Zi;
        }
        Mxx /= data_n_dbl;
        Myy /= data_n_dbl;
        Mxy /= data_n_dbl;
        Mxz /= data_n_dbl;
        Myz /= data_n_dbl;
        Mzz /= data_n_dbl;

        //computing the coefficients of the characteristic polynomial
        Mz = Mxx + Myy;
        Cov_xy = Mxx * Myy - Mxy * Mxy;
        Var_z = Mzz - Mz * Mz;

        A2 = 4.0 * Cov_xy - 3.0 * Mz * Mz - Mzz;
        A1 = Var_z * Mz + 4.0 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz;
        A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy;
        A22 = A2 + A2;

        //finding the root of the characteristic polynomial
        //using Newton's method starting at x=0
        //(it is guaranteed to converge to the right root)
        for (x = 0., y = A0, iter = 0; iter < IterMAX; iter++) // usually, 4-6 iterations are enough
        {
            Dy = A1 + x * (A22 + 16. * x * x);
            xnew = x - y / Dy;
            if ((xnew == x) || (!isfinite(xnew)))
                break;
            ynew = A0 + xnew * (A1 + xnew * (A2 + 4.0 * xnew * xnew));
            if (abs(ynew) >= abs(y))
                break;
            x = xnew;
            y = ynew;
        }

        //computing paramters of the fitting circle
        DET = x * x - x * Mz + Cov_xy;
        Xcenter = (Mxz * (Myy - x) - Myz * Mxy) / DET / 2.0;
        Ycenter = (Myz * (Mxx - x) - Mxz * Mxy) / DET / 2.0;

        //assembling the output
        mid.x = Xcenter + meanX;
        mid.y = Ycenter + meanY;
        radius = sqrt(Xcenter * Xcenter + Ycenter * Ycenter + Mz - x - x);

        return Circle_Sigma(data, mid, radius);
    }

    double Circle_Sigma(const std::vector<cv::Point2d> &data, const cv::Point2d &mid, const double &radius)
    {
        //root mean square error
        double sum = 0.;

        for (const auto &pt : data)
        {
            const double dx = pt.x - mid.x;
            const double dy = pt.y - mid.y;
            const double err = sqrt(dx * dx + dy * dy) - radius;
            sum += err * err;
        }
        return sqrt(sum / static_cast<double>(data.size()));
    }

    int geometric_circle_fit(const std::vector<cv::Point2d> &data, const cv::Point2d &mid_init, const double &radius_init, cv::Point2d &mid, double &radius, const double &lambda_init)
    {
        /*
        Adapted version from: https://people.cas.uab.edu/~mosya/cl/CPPcircle.html
        Geometric circle fit to a given set of data points (in 2D)
		
       	        
        LambdaIni - the initial value of the control parameter "lambda"
                    for the Levenberg-Marquardt procedure
                    (common choice is a small positive number, e.g. 0.001)
		        
        Output:
	       integer function value is a code:
            0:  normal termination, the best fitting circle is 
                successfully found
            1:  the number of outer iterations exceeds the limit (99)
                (indicator of a possible divergence)
            2:  the number of inner iterations exceeds the limit (99)
                (another indicator of a possible divergence)
            3:  the coordinates of the center are too large
                (a strong indicator of divergence)
 		        
        Algorithm:  Levenberg-Marquardt running over the full parameter space (a,b,r)
                         
        See a detailed description in Section 4.5 of the book by Nikolai Chernov:
        "Circular and linear regression: Fitting circles and lines by least squares"
        Chapman & Hall/CRC, Monographs on Statistics and Applied Probability, volume 117, 2010.
         
		Nikolai Chernov,  February 2014
        */
        int code, iter, inner;
        const int IterMAX = 99;
        const double data_n_dbl = static_cast<double>(data.size());
        double meanX = 0, meanY = 0;
        for (const auto &pt : data)
        {
            meanX += pt.x;
            meanY += pt.y;
        }
        meanX /= data_n_dbl;
        meanY /= data_n_dbl;

        const double factorUp = 10., factorDown = 0.04, ParLimit = 1.e+6;
        double dx, dy, ri, u, v;
        double Mu, Mv, Muu, Mvv, Muv, Mr, UUl, VVl, Nl, F1, F2, F3, dX, dY, dR;
        const double epsilon = 3.e-8;
        double G11, G22, G33, G12, G13, G23, D1, D2, D3;

        struct Circle{
            cv::Point2d mid;
            double radius;
            double sigma;//Root mean square error
            double g;
        } Old, New;

        //starting with the given initial circle (initial guess)
        New.mid = mid_init;
        New.radius = radius_init;

        //compute the root-mean-square error via function Sigma
        New.sigma = Circle_Sigma(data, New.mid, New.radius);

        //initializing lambda, iteration counters, and the exit code
        double lambda = lambda_init;
        iter = inner = code = 0;
        do{
            Old = New;
            if (++iter > IterMAX)
            {
                code = 1;
                break;
            }

            //computing moments
            Mu = Mv = Muu = Mvv = Muv = Mr = 0.;
            for (const auto &pt : data)
            {
                dx = pt.x - Old.mid.x;
                dy = pt.y - Old.mid.y;
                ri = sqrt(dx * dx + dy * dy);
                u = dx / ri;
                v = dy / ri;
                Mu += u;
                Mv += v;
                Muu += u * u;
                Mvv += v * v;
                Muv += u * v;
                Mr += ri;
            }
            Mu /= data_n_dbl;
            Mv /= data_n_dbl;
            Muu /= data_n_dbl;
            Mvv /= data_n_dbl;
            Muv /= data_n_dbl;
            Mr /= data_n_dbl;

            //computing matrices
            F1 = Old.mid.x + Old.radius * Mu - meanX;
            F2 = Old.mid.y + Old.radius * Mv - meanY;
            F3 = Old.radius - Mr;

            Old.g = New.g = sqrt(F1 * F1 + F2 * F2 + F3 * F3);

            bool abort = false;
            do
            {
                UUl = Muu + lambda;
                VVl = Mvv + lambda;
                Nl = 1.0 + lambda;

                //Cholesly decomposition
                G11 = sqrt(UUl);
                G12 = Muv / G11;
                G13 = Mu / G11;
                G22 = sqrt(VVl - G12 * G12);
                G23 = (Mv - G12 * G13) / G22;
                G33 = sqrt(Nl - G13 * G13 - G23 * G23);

                D1 = F1 / G11;
                D2 = (F2 - G12 * D1) / G22;
                D3 = (F3 - G13 * D1 - G23 * D2) / G33;

                dR = D3 / G33;
                dY = (D2 - G23 * dR) / G22;
                dX = (D1 - G12 * dY - G13 * dR) / G11;

                if ((abs(dR) + abs(dX) + abs(dY)) / (1.0 + Old.radius) < epsilon){
                    abort = true;
                    break;
                }

                //updating the parameters
                New.mid.x = Old.mid.x - dX;
                New.mid.y = Old.mid.y - dY;

                if (abs(New.mid.x) > ParLimit || abs(New.mid.y) > ParLimit)
                {
                    code = 3;
                    abort = true;
                    break;
                }

                New.radius = Old.radius - dR;

                if (New.radius <= 0.)
                {
                    lambda *= factorUp;
                    if (++inner > IterMAX)
                    {
                        code = 2;
                        abort = true;
                        break;
                    }
                    continue;
                }

                //compute the root-mean-square error
                New.sigma = Circle_Sigma(data, New.mid, New.radius);

                //check if improvement is gained
                if (New.sigma < Old.sigma) //improvement
                {
                    lambda *= factorDown;
                    break;
                }
                else //no improvement
                {
                    if (++inner > IterMAX)
                    {
                        code = 2;
                        abort = true;
                        break;
                    }
                    lambda *= factorUp;
                }
            } while (inner <= IterMAX);
            if(abort){
                break;
            }            
        } while (iter < IterMAX);

        mid = Old.mid;
        radius = Old.radius;

        return code;
    }

    bool estimateCircle(const std::vector<cv::Point2d> &data2D, const double &th, std::mt19937 &mt, cv::Point2d &center, double &radius, std::vector<size_t> *inliers)
    {
        if (data2D.size() == 3)
        {
            // Estimate circle parameters (algebraic fit)
            double err = fitCircle(data2D, center, radius);
            if (!isfinite(center.x) || !isfinite(center.y) || !isfinite(radius) || err > 3.0 * th || radius < 0)
            {
                return false;
            }
            if (err > 0.1 * th)
            {
                cv::Point2d mid2;
                double radius2;
                // Refine the parameters using Levenberg-Marquardt
                if (geometric_circle_fit(data2D, center, radius, mid2, radius2) == 0)
                {
                    if (isfinite(mid2.x) && isfinite(mid2.y) && isfinite(radius2))
                    {
                        const double err2 = Circle_Sigma(data2D, mid2, radius2);
                        if (err > err2 && radius2 > 0)
                        {
                            err = err2;
                            center = mid2;
                            radius = radius2;
                        }
                    }
                }
                if (err > th)
                {
                    return false;
                }
            }
            if (inliers)
            {
                *inliers = std::vector<size_t>{0, 1, 2};
            }
        }
        else
        {
            size_t max_iterations = data2D.size() > 100 ? 10000UL : min(binomialCoeff(data2D.size(), 3), 100000UL);
            CircleRANSAC cr(mt, data2D, th, max_iterations);
            cv::Point3d model = cr.compute(inliers);
            if (!isfinite(model.x) || !isfinite(model.y) || !isfinite(model.z) || model.z < 0)
            {
                return false;
            }
            center.x = model.x;
            center.y = model.y;
            radius = model.z;
        }
        return true;
    }

    cv::Point2d projectToPlane(const cv::Mat &Q3, const cv::Mat &basisX, const cv::Mat &basisY, const cv::Mat &origin)
    {
        CV_Assert(Q3.rows == 3 && Q3.cols == 1);
        CV_Assert(basisX.rows == 3 && basisX.cols == 1);
        CV_Assert(basisY.rows == 3 && basisY.cols == 1);
        CV_Assert(origin.rows == 3 && origin.cols == 1);
        cv::Mat Q = Q3 - origin;
        return cv::Point2d(basisX.dot(Q), basisY.dot(Q));
    }

    void calculatePlaneBasisVectors(const cv::Mat &planeNormal, cv::Mat &axis1, cv::Mat &axis2)
    {
        CV_Assert(planeNormal.rows == 3 && planeNormal.cols == 1);
        // From: https://stackoverflow.com/questions/23472048/projecting-3d-points-to-2d-plane
        // and from: https://math.stackexchange.com/questions/152077/how-to-find-a-2d-basis-within-a-3d-plane-direct-matrix-method
        // Calculate orthogonal basis vecors for all planes
        int elem = 2;
        if (abs(planeNormal.at<double>(0)) < abs(planeNormal.at<double>(1)) && abs(planeNormal.at<double>(0)) < abs(planeNormal.at<double>(2)))
        {
            elem = 0;
        }
        else if (abs(planeNormal.at<double>(1)) < abs(planeNormal.at<double>(2)))
        {
            elem = 1;
        }
        axis1 = cv::Mat::zeros(3, 1, CV_64FC1);
        axis1.at<double>(elem) = 1.0;
        axis1 = axis1.cross(planeNormal);
        axis1 /= cv::norm(axis1);
        axis2 = axis1.cross(planeNormal);
        axis2 /= cv::norm(axis2);
    }

    cv::Mat_<double> CalculatePt3DMean(const cv::Mat_<double> &points)
    {
        cv::Mat result = points.row(0);
        for (int i = 1; i < points.rows; i++)
        {
            result += points.row(i);
        }
        result /= static_cast<double>(points.rows);
        return result;
    }

    cv::Mat_<double> estimateRigid3DTansformation(const cv::Mat_<double> &points1, const cv::Mat_<double> &points2, double *scaling)
    {
        /* From https://stackoverflow.com/questions/21206870/opencv-rigid-transformation-between-two-3d-point-clouds and
           http://nghiaho.com/?page_id=671 */
        /* Calculate centroids. */
        cv::Mat_<double> t1 = -1.0 * CalculatePt3DMean(points1);
        cv::Mat_<double> t2 = -1.0 * CalculatePt3DMean(points2);

        // cv::Mat_<double> T1 = cv::Mat_<double>::eye(4, 4);
        // T1(0, 3) = t1(0);
        // T1(1, 3) = t1(1);
        // T1(2, 3) = t1(2);

        // cv::Mat_<double> T2 = cv::Mat_<double>::eye(4, 4);
        // T2(0, 3) = -1.0 * t2(0);
        // T2(1, 3) = -1.0 * t2(1);
        // T2(2, 3) = -1.0 * t2(2);

        /* Calculate covariance matrix for input points. Also calculate RMS deviation from centroid
         * which is used for scale calculation.
         */
        cv::Mat_<double> C(3, 3, 0.0);
        double p1Rms = 0, p2Rms = 0;
        for (int ptIdx = 0; ptIdx < points1.rows; ptIdx++)
        {
            cv::Mat_<double> p1 = points1.row(ptIdx) + t1;
            cv::Mat_<double> p2 = points2.row(ptIdx) + t2;
            p1Rms += p1.dot(p1);
            p2Rms += p2.dot(p2);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    C(i, j) += p2(i) * p1(j);
                }
            }
        }

        cv::Mat_<double> u, s, vh;
        cv::SVD::compute(C, s, u, vh);

        cv::Mat_<double> R = u * vh;

        if (cv::determinant(R) < 0)
        {
            R -= u.col(2) * (vh.row(2) * 2.0);
        }

        double scale = sqrt(p2Rms / p1Rms);
        // R *= scale;
        if (scaling){
            *scaling = scale;
        }
        cv::Mat_<double> t = scale * R * t1.t() - t2.t();
        cv::Mat_<double> P;
        cv::hconcat(R, t, P);
        return P;

        // cv::Mat_<double> M = cv::Mat_<double>::eye(4, 4);
        // R.copyTo(M.colRange(0, 3).rowRange(0, 3));

        // cv::Mat_<double> result = T2 * M * T1;
        // result /= result(3, 3);

        // return result.rowRange(0, 3);
    }

    double getRigidTransformPt3DError(const cv::Mat_<double> &P, const cv::Mat_<double> &X1, const cv::Mat_<double> &X2)
    {
        CV_Assert(X1.rows == 3 && X2.rows == 3 && X1.cols == 1 && X2.cols == 1);
        CV_Assert(P.rows == 3 && P.cols == 4);
        cv::Mat_<double> X1_ = X1.clone();
        X1_.resize(4);
        X1_(3) = 1.0;
        cv::Mat_<double> X12 = P * X1_;
        cv::Mat_<double> diff = X12 - X2;
        return cv::norm(diff);
    }

    double getDescriptorDist(const cv::Mat &descr1, const cv::Mat &descr2)
    {
        if (descr1.type() == CV_8U)
        {
            return norm(descr1, descr2, cv::NORM_HAMMING);
        }
        return norm(descr1, descr2, cv::NORM_L2);
    }

    int getVectorMainDirIdx(const cv::Mat vec)
    {
        CV_Assert(vec.rows == 1 || vec.cols == 1);
        const int nr_elems = vec.rows * vec.cols;
        double val_max = abs(vec.at<double>(0));
        int mainDir = 0;
        for (int i = 1; i < nr_elems; i++)
        {
            double val = abs(vec.at<double>(i));
            if (val > val_max)
            {
                val_max = val;
                mainDir = i;
            }
        }
        return mainDir;
    }

    double getAngleBetwVecs(const cv::Mat vec1, const cv::Mat vec2)
    {
        CV_Assert((vec1.rows == 1 || vec1.cols == 1) && (vec2.rows == 1 || vec2.cols == 1) && (vec1.rows == vec2.rows));
        CV_Assert(vec1.rows * vec1.cols == vec2.rows * vec2.cols);
        return std::acos(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2))) * 180.0 / M_PI;
    }

    bool angleBetwVecsBelowTh(const cv::Mat vec1, const cv::Mat vec2, const double &th_deg, const bool ignoreMirror)
    {
        double angle = getAngleBetwVecs(vec1, vec2);
        if (ignoreMirror && angle > 90.)
        {
            angle = abs(angle - 180.);
        }
        return angle < th_deg;
    }

    /* Image Shadow / Highlight Correction. The same function as it in Photoshop / GIMP
     * adapted version of https://gist.github.com/HViktorTsoi/8e8b0468a9fb07842669aa368382a7df
     * img: input greyscale image
     * shadow_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
     * shadow_tone_percent [0.0 ~ 1.0]: Controls the range of tones (image brightness values, i.e. 0 to shadow_tone_percent * 255) in the shadows that are modified.
     * shadow_radius [>0]: Controls the size of the local neighborhood around each pixel
     * highlight_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
     * highlight_tone_percent [0.0 ~ 1.0]: Controls the range of tones (image brightness values, i.e. 255 - highlight_tone_percent * 255 to 255) in the highlights that are modified.
     * highlight_radius [>0]: Controls the size of the local neighborhood around each pixel
     * histEqual: If true, histogram equalization is performed afterwards
     */
    cv::Mat shadowHighlightCorrection(cv::InputArray img, const float &shadow_amount_percent, const float &shadow_tone_percent, const int &shadow_radius, const float &highlight_amount_percent, const float &highlight_tone_percent, const int &highlight_radius, const bool histEqual)
    {
        const cv::Mat img_in = img.getMat();
        CV_Assert(img_in.type() == CV_8UC1);
        CV_Assert(shadow_amount_percent >= 0 && shadow_amount_percent <= 1.f && shadow_tone_percent >= 0 && shadow_tone_percent <= 1.f);
        CV_Assert(highlight_amount_percent >= 0 && highlight_amount_percent <= 1.f && highlight_tone_percent >= 0 && highlight_tone_percent <= 1.f);
        CV_Assert(shadow_radius > 0 && shadow_radius < std::min(img_in.rows, img_in.cols) / 4);
        CV_Assert(highlight_radius > 0 && highlight_radius < std::min(img_in.rows, img_in.cols) / 4);
        const float shadow_tone = shadow_tone_percent * 255.f;//0...1 -> 0...255
        const float highlight_tone = 255.f - highlight_tone_percent * 255.f; // 0...1 -> 255...0

        const float shadow_gain = 1.f + shadow_amount_percent * 6.f;// 0...1 -> 1...7
        const float highlight_gain = 1.f + highlight_amount_percent * 6.f;//0...1 -> 1...7

        // Convert img to float
        cv::Mat img_flt;
        img_in.convertTo(img_flt, CV_32FC1);
        const cv::Size imgSi = img_in.size();

        // extract shadow
        // darkest regions get highest values, img values > shadow_tone -> 0, range: 0...shadow_tone -> 255...0
        cv::Mat shadow_map = 255.f - (img_flt * 255.f) / shadow_tone;
        // cv::Mat shadow_map = 255.f - img_flt / shadow_tone;
        for (int y = 0; y < imgSi.height; y++)
        {
            for (int x = 0; x < imgSi.width; x++)
            {
                if (img_flt.at<float>(y, x) >= shadow_tone)
                {
                    shadow_map.at<float>(y, x) = 0.f;
                }
            }
        }

        // extract highlight
        // brightest regions get highest values, img values < highlight_tone -> 0, range highlight_tone...255 -> 0...255
        cv::Mat highlight_map = 255.f - ((255.f - img_flt) * 255.f) / (255.f - highlight_tone);
        // cv::Mat highlight_map = 255.f - (255.f - img_flt) / (255.f - highlight_tone);
        for (int y = 0; y < imgSi.height; y++)
        {
            for (int x = 0; x < imgSi.width; x++)
            {
                if (img_flt.at<float>(y, x) <= highlight_tone)
                {
                    highlight_map.at<float>(y, x) = 0.f;
                }
            }
        }

        // Gaussian blur on tone map, for smoother transition
        if (shadow_amount_percent * static_cast<float>(shadow_radius) > 0.f)
        {
            cv::blur(shadow_map, shadow_map, cv::Size(shadow_radius, shadow_radius));
        }
        if (highlight_amount_percent * static_cast<float>(highlight_radius) > 0.f)
        {
            cv::blur(highlight_map, highlight_map, cv::Size(highlight_radius, highlight_radius));
        }

        // Tone LUT
        std::vector<float> t(256);
        std::iota(t.begin(), t.end(), 0);
        std::vector<float> lut_shadow, lut_highlight;
        const float m1 = 1.f / 255.f;
        for (size_t i = 0; i < 256; i++)
        {
            const float im1 = static_cast<float>(i) * m1;
            const float pwr = std::pow(1.f - im1, shadow_gain); //(1 - i / 255)^shadow_gain: 1...0
            float ls = (1.f - pwr) * 255.f;//0...255
            ls = std::max(0.f, std::min(255.f, std::round(ls)));
            lut_shadow.emplace_back(std::move(ls));

            float lh = std::pow(im1, highlight_gain) * 255.f; //(i / 255)^highlight_gain: 0...255
            lh = std::max(0.f, std::min(255.f, std::round(lh)));
            lut_highlight.emplace_back(std::move(lh));
        }

        // adjust tone
        shadow_map = shadow_map * m1;//0...1
        highlight_map = highlight_map * m1;//0...1

        cv::Mat shadow_map_tone1 = cv::Mat::zeros(imgSi, shadow_map.type());
        for (int y = 0; y < imgSi.height; y++)
        {
            for (int x = 0; x < imgSi.width; x++)
            {
                const unsigned char &vi = img_in.at<unsigned char>(y, x);
                const float &ls = lut_shadow.at(vi);
                shadow_map_tone1.at<float>(y, x) = ls * shadow_map.at<float>(y, x); //[0...255] * [0...1] -> [0...255]
            }
        }
        cv::Mat shadow_map_tone2 = 1.f - shadow_map;     // 1...0, 1 for all pixel values > shadow_tone
        shadow_map_tone2 = shadow_map_tone2.mul(img_flt); //[1...0] * [0...255], pixel values > shadow_tone remain untouched
        shadow_map_tone1 += shadow_map_tone2;

        cv::Mat highlight_map_tone1 = cv::Mat::zeros(imgSi, shadow_map.type());
        for (int y = 0; y < imgSi.height; y++)
        {
            for (int x = 0; x < imgSi.width; x++)
            {
                const unsigned char vi = static_cast<unsigned char>(std::max(0.f, std::min(std::round(shadow_map_tone1.at<float>(y, x)), 255.f)));
                const float &lh = lut_highlight.at(vi);
                highlight_map_tone1.at<float>(y, x) = lh * highlight_map.at<float>(y, x);
            }
        }
        cv::Mat highlight_map_tone2 = 1.f - highlight_map;
        highlight_map_tone2 = highlight_map_tone2.mul(shadow_map_tone1);
        shadow_map_tone1 = highlight_map_tone2 + highlight_map_tone1;
        cv::convertScaleAbs(shadow_map_tone1, shadow_map_tone1);
        // cv::imshow("other", shadow_map_tone1);

        cv::Mat histequ;
        if (histEqual)
        {
            cv::equalizeHist(shadow_map_tone1, histequ);
            // cv::imshow("other2", histequ);
        }
        else
        {
            histequ = shadow_map_tone1;
        }

        return histequ;
    }
}