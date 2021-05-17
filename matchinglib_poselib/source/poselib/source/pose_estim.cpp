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
 FILE: pose_estim.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: May 2016

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for the estimation and optimization of poses between
			  two camera views (images).
**********************************************************************************************************/

#include "poselib/pose_estim.h"
#include "poselib/pose_helper.h"
#include "five-point-nister/five-point.hpp"
#include "BA_driver.h"
#include "usac/usac_estimations.h"
#include <Eigen/StdVector>

using namespace cv;
using namespace std;

namespace poselib
{

/* --------------------------- Defines --------------------------- */



/* --------------------- Function prototypes --------------------- */



/* --------------------- Functions --------------------- */

/* Estimation of the Essential matrix based on the 5-pt Nister algorithm integrated in an ARRSAC framework with an automatic threshold
 * estimation based on the reprojection errors.
 *
 * InputArray p1						Input  -> Observed point coordinates of the left image in the camera coordinate system
 *												  (n rows, 2 cols)
 * InputArray p2						Input  -> Observed point coordinates of the right image in the camera coordinate system
 *												  (n rows, 2 cols)
 * OutputArray E						Output -> Essential matrix
 * OutputArray mask						Output -> Inlier mask
 * double *th							I/O	   -> Pointer to the inlier threshold (must be in the camera coordinate system)
 * int *nrgoodPts						Output -> Number of inliers to E
 *
 * Return value:						0 :		Everything ok
 *										-1:		Calculation of Essential matrix not possible (execute addFailedProps() within uncalibartedRectify)
 *										-2:		Mat for essential matrix missing
 */
int AutoThEpi::estimateEVarTH(cv::InputArray p1, cv::InputArray p2, cv::OutputArray E, cv::OutputArray mask, double *th, int *nrgoodPts)
{
	bool th_sem[3] = {true, true, false};
	int th_fail_cnt = 2;
	double th_failed = *th;
	double th_old;
	Mat p1_ = p1.getMat();
	Mat p2_ = p2.getMat();
	Mat mask_, E_;

	do
	{
		if(!mask_.empty()) mask_.release();
		th_old = *th;
		if(!findEssentialMat(E_, p1_, p2_, ARRSAC, 0.99, *th, mask_,true,robustEssentialRefine))
		{
			if((*th < PIX_MIN_GOOD_TH * pixToCamFact) && !th_stable && !th_sem[2])
			{
				if(!mask_.empty()) mask_.release();
				th_failed = *th;
				*th = PIX_MIN_GOOD_TH * pixToCamFact;
				th_old = *th;
				if(!findEssentialMat(E_, p1_, p2_, ARRSAC, 0.99, *th, mask_,true,robustEssentialRefine))
				{
					//addFailedProps();-------------------------------------------------------------------->execute this for a return value = -1 within  class StereoAutoCalib in project uncalibratedRectify, file RectStructMot.cpp
					return -1; //Calculation of Essential matrix not possible
				}
				th_sem[2] = true;
			}
			else if(th_sem[2])
			{
				th_fail_cnt *= 2;
			}
			else
			{
				//addFailedProps();
				return -1; //Calculation of Essential matrix not possible
			}
		}
		else
		{
			if(th_sem[2] && (th_fail_cnt > 2))
				corr_filt_min_pix_th = *th / pixToCamFact;
			th_sem[2] = false;
		}
		if((th_fail_cnt <= 2) || !th_sem[2])
			*nrgoodPts = cv::countNonZero(mask_);
		if(!th_stable)
		{
			if((th_fail_cnt <= 2) || !th_sem[2])
				*th = estimateThresh(p1_, p2_, E_);
			else
				*th = th_failed * (double)th_fail_cnt;
			if(th_sem[2] && (th_failed >= *th))
				*th = th_failed * (double)th_fail_cnt;
			if(!th_sem[2])
			{
				if(th_old / *th > 1.0 + 1e-6)
					th_sem[0] = false;
				else if(th_old / *th < 1.0 - 1e-6)
					th_sem[1] = false;
			}
		}
	}
	while(((th_old / *th < 0.9) || (*th/th_old < 0.9)) && ((((float)*nrgoodPts/(float)p1_.rows < 0.67) &&
			(th_sem[0] || th_sem[1])) || th_sem[2])); //Try to find a good threshold if the initial wasnt a good choice

	if(E.needed())
	{
		if(E.empty())
			E.create(3, 3, CV_64F);
		E_.copyTo(E.getMat());
	}
	else
		return -2;

	if(mask.needed())
	{
		Mat mask_tmp;
		mask.create(mask_.size(), mask_.type());
		mask_tmp = mask.getMat();
		mask_.copyTo(mask_tmp);
	}

	return 0;
}

/* Estimates a new threshold based on the correspondences and the given essential matrix.
 *
 * InputArray p1					Input  -> Left image projections (n rows x 2 columns)
 * InputArray p2					Input  -> Right image projections (n rows x 2 columns)
 * InputArray E						Input  -> Essential matrix
 * bool useImgCoordSystem			Input  -> Specifies if the input thresh and the return value should be
 *											  in camera (false) or pixel (true) units
 *											  (default: false -> camera coord. system)
 * bool storeGlobally				Input  -> If true (default: false), the threshold is stored globally in the
 *											  class member variable
 *
 * Return value:					Threshold in the camera coordinate system if
 *									useImgCoordSystem = false (default) and in the image coordinate system
 *									otherwise
 */
double AutoThEpi::estimateThresh(cv::InputArray p1, cv::InputArray p2, cv::InputArray E, bool useImgCoordSystem,
									   bool storeGlobally)
{
	statVals qp_act;
	Mat _E = E.getMat();
	Mat _p1 = p1.getMat();
	Mat _p2 = p2.getMat();
	std::vector<double> error;
	double th, th_tmp, maxInlDist;
	//double pixToCamFact = 4.0/(std::sqrt((double)2)*(this->K1.at<double>(0,0)+this->K1.at<double>(1,1)+this->K2.at<double>(0,0)+this->K2.at<double>(1,1)));

	getReprojErrors(_E,_p1,_p2,useImgCoordSystem,nullptr,&error);

	for(auto &i : error)
		i = std::sqrt(i);

	if(useImgCoordSystem)
	{
		th = this->corr_filt_pix_th;
		maxInlDist = 4.0 * th;
		maxInlDist = maxInlDist > 5.0 ? 5.0:maxInlDist;
	}
	else
	{
		th = this->corr_filt_cam_th;
		maxInlDist = 4.0 * th;
		maxInlDist = maxInlDist > 5.0 * pixToCamFact ? (5.0 * pixToCamFact):maxInlDist;
	}

	double r1 = *std::max_element(error.begin(),error.end()) - *std::min_element(error.begin(),error.end());
	if(r1 > maxInlDist)
	{
		getStatsfromVec(error, &qp_act,true);
	}
	else
	{
		getStatsfromVec(error, &qp_act);
	}

	if((qp_act.arithErr/qp_act.medErr > 2.0) || (qp_act.arithErr/qp_act.medErr < 0.5))
		th_tmp = qp_act.medErr + 3.0 * qp_act.medStd;
	else
		th_tmp = qp_act.arithErr + 3.0 * qp_act.arithStd;

	if((th_tmp < 5.0 * th) || (th_tmp < 4.0 * PIX_MIN_GOOD_TH))
		th = setCorrTH(th_tmp,useImgCoordSystem,storeGlobally);
	else
	{
		if(th < (useImgCoordSystem ? (MAX_PIX_TH/2):((MAX_PIX_TH/2) * pixToCamFact)))
			th = setCorrTH(th * 2.0,useImgCoordSystem,storeGlobally);
		else
		{
			th = setCorrTH(corr_filt_min_pix_th,true,storeGlobally);
			if(!useImgCoordSystem)
				th *= pixToCamFact;
		}
	}

	return th;
}

/* Sets a new threshold for marking correspondences as in- or outliers.
 *
 * double thresh					Input  -> New threshold (default: in the camera coordinate system)
 * bool useImgCoordSystem			Input  -> Specifies if the input thresh and the return value should be
 *											  in camera (false) or pixel (true) units
 *											  (default: false -> camera coord. system)
 * bool storeGlobally				Input  -> If true (default), the threshold is stored globally in the
 *											  class member variable
 *
 * Return value:					Threshold in the camera coordinate system if
 *									useImgCoordSystem = false (default) and in the image coordinate system
 *									otherwise
 */
double AutoThEpi::setCorrTH(double thresh, bool useImgCoordSystem, bool storeGlobally)
{
	//CV_Assert(!this->K1.empty() && !this->K2.empty());

	double pix_th_tmp, cam_th_tmp;
	//double pixToCamFact = 4.0/(std::sqrt((double)2)*(this->K1.at<double>(0,0)+this->K1.at<double>(1,1)+this->K2.at<double>(0,0)+this->K2.at<double>(1,1)));

	if(useImgCoordSystem)
		pix_th_tmp = thresh;
	else
		pix_th_tmp = thresh / pixToCamFact;

	if(pix_th_tmp < corr_filt_min_pix_th)
		pix_th_tmp = corr_filt_min_pix_th;
	else if(pix_th_tmp > MAX_PIX_TH)
		pix_th_tmp = MAX_PIX_TH;

	cam_th_tmp = pix_th_tmp * pixToCamFact;

	if(storeGlobally)
	{
		this->corr_filt_pix_th = pix_th_tmp;
		this->corr_filt_cam_th = cam_th_tmp;
	}

	if(useImgCoordSystem)
		return pix_th_tmp;

	return cam_th_tmp;
}

/* Refines the essential matrix E by using the 8-point-algorithm and SVD (currently the
 * solution is not found by SVD but by calculating the eigenvalues and eigenvectors of
 * A^T*A and selcting the eigenvector corresponding to the smallest eigenvalue which is
 * much faster than SVD) with a pseudo-huber cost function. Thus, this methode is very
 * robust and can refine E also in an iterative manner. If the input variable iters is
 * set to 0, the essential matrix is as long refined as the sum of squared errors reaches
 * a minimum value, its reduction is too small or the maximum number of iterations is
 * reached.
 *
 * InputArray points1				Input  -> Image projections in the left camera without
 *											  outliers (1 projection per column with 2 channels
 *											  or 1 projection per row with 2 columns)
 * InputArray points2				Input  -> Image projections in the right camera without
 *											  outliers (1 projection per column with 2 channels
 *											  or 1 projection per row with 2 columns)
 * InputArray E_init				Input  -> Initial estimate of the essential matrix
 * Mat & E_refined					Output -> Refined essential matrix
 * double th						Input  -> Threshold for the pseudo-huber cost function.
 *											  Can be really small (e.g. 0.005 for image
 *											  coordinate system)
 * unsigned int iters				Input  -> Number of iterations that should be performed.
 *											  If 0, the essential matrix is as long refined
 *											  as the sum of squared errors reaches a minimum
 *											  value, its reduction is too small or the
 *											  maximum number of iterations is reached.
 * bool makeClosestE				Input  -> Specifies if the closest essential matrix
 *											  should be computed by enforcing the singularity
 *											  constraint (first 2 singular values are equal
 *											  and third is zero).
 * unsigned int *sumSqrErr_init		Output -> Initial sum of squared errors (after first
 *											  refinement).
 * unsigned int *sumSqrErr			Output -> Final sum of squared errors
 * OutputArray errors				Output -> Final Sampson error for every correspondence
 * InputOutputArray mask			I/O    -> Inlier mask (input) and mask to exclude points which do not
 *											  correspond with the oriented epipolar constraint combined with
 *											  the input inlier mask (output).
 * int model						Input  -> Optional input (Default = 0) to specify the used
 *											  model (0 = Normal essential matrix, 1 = affine
 *											  essential matrix, 2 = translational essential matrix).
 * bool tryOrientedEpipolar			Input  -> Optional input [DEFAULT = false] to specify if a essential matrix
 *											  should be evaluated by the oriented epipolar constraint. Maybe this
 *											  is only possible for a fundamental matrix?
 * bool normalizeCorrs				Input  -> If true [DEFAULT=false], the coordinates are normalized. Normalization has
 *											  an impact on the properties of the essential matrix! Thus, after normalization,
 *											  the properties of the fundamental matrix are valid!
 *
 * Return value:					none
 */
void robustEssentialRefine(cv::InputArray points1, cv::InputArray points2, cv::InputArray E_init, cv::Mat & E_refined,
						  double th, unsigned int iters, bool makeClosestE, double *sumSqrErr_init,
						  double *sumSqrErr, cv::OutputArray errors, cv::InputOutputArray mask, int model, bool tryOrientedEpipolar, bool normalizeCorrs)
{

	Mat _points1 = points1.getMat(), _points2 = points2.getMat();
	Mat pointsRedu1, pointsRedu2;
	Mat E, mask_;
	double err = -9999.0, err_old = 1e12;//, err_min = 1e12;
    int npoints;
	if(_points1.channels() == 2)
		npoints = _points1.checkVector(2);
	else
		npoints = _points1.rows;

    CV_Assert( npoints >= 0 &&
			  ((_points2.checkVector(2) == npoints &&
              _points1.type() == CV_64FC2 &&
			  _points1.rows == 1 && _points1.cols > _points1.rows) ||
			  (_points1.rows == npoints &&
			  _points1.type() == CV_64FC1 &&
			  _points1.cols == 2 && _points1.rows > _points1.cols)) &&
              _points1.type() == _points2.type());

	if(mask.needed() && !mask.getMat().empty())
	{
		Mat pointsRedu1_tmp;
		Mat pointsRedu2_tmp;
		mask_ = mask.getMat();
		if(_points1.channels() == 2)
		{
			pointsRedu1_tmp = _points1.reshape(1,2).t();
			pointsRedu2_tmp = _points2.reshape(1,2).t();
		}
		else
		{
			pointsRedu1_tmp = _points1;
			pointsRedu2_tmp = _points2;
		}
		for(int i = 0; i < npoints; i++)
		{
			if(mask_.at<bool>(i))
			{
				pointsRedu1.push_back(pointsRedu1_tmp.row(i));
				pointsRedu2.push_back(pointsRedu2_tmp.row(i));
			}
		}
		npoints = pointsRedu1.rows;
		//pointsRedu1 = pointsRedu1.t();
		pointsRedu1 = pointsRedu1.reshape(2,1);
		//pointsRedu2 = pointsRedu2.t();
		pointsRedu2 = pointsRedu2.reshape(2,1);
	}
	else
	{
		if(_points1.channels() == 2)
		{
			pointsRedu1 = _points1;
			pointsRedu2 = _points2;
		}
		else
		{
			//pointsRedu1 = _points1.t();
			//pointsRedu1 = pointsRedu1.reshape(2, 1);
			pointsRedu1 = _points1.reshape(2,1);
			//pointsRedu2 = _points2.t();
			//pointsRedu2 = pointsRedu2.reshape(2,1);
			pointsRedu2 = _points2.reshape(2, 1);
		}
	}

	if(npoints < 50)
	{
		E_init.getMat().copyTo(E_refined);
		cout << "There are too less points for a refinement left!" << endl;
		return;
	}
	E = E_init.getMat().clone();

	//Thresholds for iterative refinement
	const double minSumSqrErrDiff = th/10;
	const double minSumSqrErr = th*th/100 * npoints;
	const unsigned int maxIters = 50;

    cv::Point2d m0c = cv::Point2d(0, 0);
    cv::Point2d m1c = cv::Point2d(0, 0);
//    CvPoint2D64f m0c = {0,0}, m1c = {0,0};
    double t, scale0 = 0, scale1 = 0;

    std::vector<Point2d> _m1, _m2;
    _m1 = (std::vector<Point2d>) pointsRedu1;
    _m2 = (std::vector<Point2d>) pointsRedu2;
//	const CvPoint2D64f* _m1 = (const CvPoint2D64f*)pointsRedu1.data;
//    const CvPoint2D64f* _m2 = (const CvPoint2D64f*)pointsRedu2.data;

    int i;

    // compute centers and average distances for each of the two point sets
	Mat H0, H1;
	if (normalizeCorrs)
	{
		for (i = 0; i < npoints; i++)
		{
			double x = _m1[i].x, y = _m1[i].y;
			m0c.x += x; m0c.y += y;

			x = _m2[i].x, y = _m2[i].y;
			m1c.x += x; m1c.y += y;
		}

		// calculate the normalizing transformations for each of the point sets:
		// after the transformation each set will have the mass center at the coordinate origin
		// and the average distance from the origin will be ~sqrt(2).
		t = 1. / npoints;
		if (model != 2)
		{
			m0c.x *= t; m0c.y *= t;
			m1c.x *= t; m1c.y *= t;
		}
		else
		{
			t *= 0.5;
			m1c.x = m0c.x = (m0c.x + m1c.x) * t;
			m1c.y = m0c.y = (m0c.y + m1c.y) * t;
		}

		for (i = 0; i < npoints; i++)
		{
			double x = _m1[i].x - m0c.x, y = _m1[i].y - m0c.y;
			scale0 += sqrt(x*x + y*y);

			x = _m2[i].x - m1c.x, y = _m2[i].y - m1c.y;
			scale1 += sqrt(x*x + y*y);
		}

		if (model != 2)
		{
			scale0 *= t;
			scale1 *= t;
		}
		else
		{
			scale1 = scale0 = (scale0 + scale1) * t;
		}

		if (scale0 < FLT_EPSILON || scale1 < FLT_EPSILON)
			return;

		scale0 = sqrt(2.) / scale0;
		scale1 = sqrt(2.) / scale1;

		H0 = (Mat_<double>(3, 3) << scale0, 0, -scale0*m0c.x,
			0, scale0, -scale0*m0c.y,
			0, 0, 1);

		H1 = (Mat_<double>(3, 3) << scale1, 0, -scale1*m1c.x,
			0, scale1, -scale1*m1c.y,
			0, 0, 1);
	}

	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic > TAMat;
	TAMat A1, A2;

	if(!model)
	{
		A1.resize(npoints, 9);
		A2.resize(9, 9);
	}
	else if(model == 1)
	{
		A1.resize(npoints, 5);
		A2.resize(5, 5);
	}
	else if(model == 2)
	{
		A1.resize(npoints, 3);
		A2.resize(3, 3);
	}

	Mat F3 = E.clone();

	unsigned int j;
	for(j = 0; j < (iters ? iters:maxIters); j++)
	{
		// form a linear system Ax=0: for each selected pair of points m1 & m2,
		// the row of A(=a) represents the coefficients of equation: (m2, 1)'*F*(m1, 1) = 0
		// to save computation time, we compute (At*A) instead of A and then solve (At*A)x=0.
		std::vector<double> weights(npoints), denom1s(npoints);
		for (i = 0; i < npoints; i++)
		{
			double denom1, num;
			SampsonL1(pointsRedu1.col(i).reshape(1, 2), pointsRedu2.col(i).reshape(1, 2), F3, denom1, num);
			const double weight = costPseudoHuber(num*denom1, th);
			weights[i] = weight;
			denom1s[i] = denom1;
		}
		double weightnorm = 0;
		for (i = 0; i < npoints; i++)
		{
			weightnorm += pow(weights[i]* denom1s[i], 2);
		}
		weightnorm = sqrt(weightnorm);

		for (i = 0; i < npoints; i++)
		{


			double x0;
			double y0;
			double x1;
			double y1;
			if (normalizeCorrs)
			{
				x0 = (_m1[i].x - m0c.x)*scale0;
				y0 = (_m1[i].y - m0c.y)*scale0;
				x1 = (_m2[i].x - m1c.x)*scale1;
				y1 = (_m2[i].y - m1c.y)*scale1;
			}
			else
			{
				x0 = _m1[i].x;
				y0 = _m1[i].y;
				x1 = _m2[i].x;
				y1 = _m2[i].y;
			}

			if(!model)
			{
				A1(i, 0) = x1 * x0;
				A1(i, 1) = x1 * y0;
				A1(i, 2) = x1;
				A1(i, 3) = y1 * x0;
				A1(i, 4) = y1 * y0;
				A1(i, 5) = y1;
				A1(i, 6) = x0;
				A1(i, 7) = y0;
				A1(i, 8) = 1;
			}
			else if(model == 1)
			{
				A1(i, 0) = x1;
				A1(i, 1) = y1;
				A1(i, 2) = x0;
				A1(i, 3) = y0;
				A1(i, 4) = 1.0;
			}
			else if(model == 2)
			{
				double ww = x1 * y0 - x0 * y1;
				if(fabs(ww) < DBL_EPSILON)
					continue;

				ww = 1./ww;

				A1(i, 0) = (y1 - y0) * ww;
				A1(i, 1) = (x0 - x1) * ww;
				A1(i, 2) = 1.0;
			}

			A1.row(i) *= denom1s[i] * weights[i] / weightnorm; /* multiply by 1/dominator of the Sampson L1-distance to eliminate
										* it from the dominator of the pseudo-huber weight value (which is
										* actually sqrt(cost_pseudo_huber)/Sampson_L1_distance). This is
										* necessary because the SVD calculates the least squared algebraic
										* error of the fundamental matrix (x2^T*F*x1)^2. This algebraic
										* error is the same as the numerator of the Sampson distance and
										* should be replaced by the pseudo-huber cost function during SVD.
										*/
		}

		A2 = A1.transpose()*A1;

		Eigen::Matrix3d F2;
		Eigen::Matrix<double, Eigen::Dynamic, 1> lastCol;
		if(!model)
		{
			Eigen::EigenSolver<Eigen::Matrix<double,9,9>> eigA(A2);
			for( i = 0; i < 9; i++ )
			{
				if( fabs(eigA.eigenvalues().real()[i]) < DBL_EPSILON )
					break;
			}
			if( i < 8 )
			{
				E.copyTo(E_refined);
				cout << "Refinement failed!" << endl;
				return;
			}

			double smallestEigVal = fabs(eigA.eigenvalues().real()[0]);
			int smallestEigValIdx = 0;
			for( i = 1; i < 9; i++ )
			{
				if( fabs(eigA.eigenvalues().real()[i]) < smallestEigVal )
				{
					smallestEigVal = fabs(eigA.eigenvalues().real()[i]);
					smallestEigValIdx = i;
				}
			}

			lastCol.resize(9,1);
			lastCol = eigA.eigenvectors().col(smallestEigValIdx).real();

			//Much slower then solving for eigenvectors of A^T*A
			/*Eigen::JacobiSVD<TAMat> svdA(A1, Eigen::ComputeFullV);
			lastCol = svdA.matrixV().col(8);*/

			//Convert to F

			F2 = Eigen::Matrix3d(lastCol.data());
			F2.transposeInPlace();
			if(makeClosestE)
			{
				if(getClosestE(F2))
				{
					cout << "E is no essential matrix - taking last valid E!" << endl;
					//E = H1.inv().t() * E * H0.inv();
					//cv::cv2eigen(E,F2);
					break;
				}
			}

			if(makeClosestE)
			{
				bool validE;
				if(mask.needed())
					validE = validateEssential(_points1, _points2, F2, false, mask, tryOrientedEpipolar);
				else
					validE = validateEssential(_points1, _points2, F2, false, cv::noArray(), tryOrientedEpipolar);
				if(!validE)
				{
					cout << "E is no valid essential matrix - taking last valid E!" << endl;
					break;
				}
			}
			else
			{
				bool validE;
				if(mask.needed())
					validE = validateEssential(_points1, _points2, F2, true, mask, tryOrientedEpipolar);
				else
					validE = validateEssential(_points1, _points2, F2, true, cv::noArray(), tryOrientedEpipolar);
				if(!validE)
				{
					cout << "E is no valid essential matrix - taking last valid E!" << endl;
					break;
				}
			}
		}
		else if(model == 1)
		{
			Eigen::EigenSolver<Eigen::Matrix<double,5,5>> eigA(A2);
			for( i = 0; i < 5; i++ )
			{
				if( fabs(eigA.eigenvalues().real()[i]) < DBL_EPSILON )
					break;
			}
			if( i < 5 )
				return;

			double smallestEigVal = fabs(eigA.eigenvalues().real()[0]);
			int smallestEigValIdx = 0;
			for( i = 1; i < 5; i++ )
			{
				if( fabs(eigA.eigenvalues().real()[i]) < smallestEigVal )
				{
					smallestEigVal = fabs(eigA.eigenvalues().real()[i]);
					smallestEigValIdx = i;
				}
			}

			lastCol.resize(3,1);
			lastCol = eigA.eigenvectors().col(smallestEigValIdx).real();

			//Much slower then solving for eigenvectors of A^T*A
			/*Eigen::JacobiSVD<TAMat> svdA(A1, Eigen::ComputeFullV);
			Eigen::Matrix<double, 5, 1 > lastCol = svdA.matrixV().col(4);*/

			//Convert to Fa
			F2 = Eigen::Matrix3d::Zero();
			F2(0,2) = lastCol(0);
			F2(1,2) = lastCol(1);
			F2(2,0) = lastCol(2);
			F2(2,1) = lastCol(3);
			F2(2,2) = lastCol(4);

			bool validE;
			if(mask.needed())
				validE = validateEssential(_points1, _points2, F2, false, mask, tryOrientedEpipolar);
			else
				validE = validateEssential(_points1, _points2, F2, false, cv::noArray(), tryOrientedEpipolar);
			if(!validE)
			{
				cout << "E is no valid essential matrix - taking last valid E!" << endl;
				break;
			}
		}
		else if(model == 2)
		{
			Eigen::EigenSolver<Eigen::Matrix<double,3,3>> eigA(A2);
			for( i = 0; i < 3; i++ )
			{
				if( fabs(eigA.eigenvalues().real()[i]) < DBL_EPSILON )
					break;
			}
			if( i < 3 )
				return;

			double smallestEigVal = fabs(eigA.eigenvalues().real()[0]);
			int smallestEigValIdx = 0;
			for( i = 1; i < 3; i++ )
			{
				if( fabs(eigA.eigenvalues().real()[i]) < smallestEigVal )
				{
					smallestEigVal = fabs(eigA.eigenvalues().real()[i]);
					smallestEigValIdx = i;
				}
			}

			//Much slower then solving for eigenvectors of A^T*A
			/*Eigen::JacobiSVD<TAMat> svdA(A1, Eigen::ComputeFullV);
			Eigen::Matrix<double, 5, 1 > lastCol = svdA.matrixV().col(4);*/

			lastCol.resize(3,1);
			lastCol = eigA.eigenvectors().col(smallestEigValIdx).real();

			lastCol /= lastCol.norm();

			//Convert to Ft
			F2 = Eigen::Matrix3d::Zero();
			F2(0,1) = lastCol(2);
			F2(0,2) = -1.0 * lastCol(1);
			F2(1,0) = -1.0 * lastCol(2);
			F2(1,2) = lastCol(0);
			F2(2,0) = lastCol(1);
			F2(2,1) = -1.0 * lastCol(0);

			bool validE;
			if(mask.needed())
				validE = validateEssential(_points1, _points2, F2, false, mask, tryOrientedEpipolar);
			else
				validE = validateEssential(_points1, _points2, F2, false, cv::noArray(), tryOrientedEpipolar);
			if(!validE)
			{
				cout << "E is no valid essential matrix - taking last valid E!" << endl;
				break;
			}
		}

		cv::eigen2cv(F2,F3);
		if(normalizeCorrs)
			F3 = H1.t() * F3 * H0;


		if(!iters)
		{
			err = (A1 * lastCol).squaredNorm();
			//if(err < err_min)
			//{
			//	E_refined = F3.clone();
			//	err_min = err;
			//}
			const double errdiff = abs(err_old - err);
			if((j > 1) && ((errdiff < minSumSqrErrDiff) ||
				(err < minSumSqrErr)))
			{
				break;
			}
			err_old = err;
		}
		else if((j == (iters-1)) && sumSqrErr)
		{
			err = (A1 * lastCol).squaredNorm();
		}

		if(!j && sumSqrErr_init)
		{
			if(!iters) *sumSqrErr_init = err;
			else *sumSqrErr_init = (A1 * lastCol).squaredNorm();
		}
	}

	if(sumSqrErr && j) *sumSqrErr = err;

	if(errors.needed())
	{
		errors.create(npoints,1, CV_64FC1);
		Mat sdist = errors.getMat();
		for(i = 0; i < npoints; i++)
		{
			double denom1, num;
			SampsonL1(_points1.col(i).reshape(1,2), _points2.col(i).reshape(1,2), F3, denom1, num);
			sdist.row(i) = denom1 * denom1 * num * num;
		}
	}

	E_refined = F3.clone();
	/*if(model != 2)
		E_refined /= E_refined.at<double>(2,2);
	else*/
		//cv::normalize(E_refined, E_refined);
}


/* Estimation of the Essential matrix based on the 5-pt Nister algorithm integrated in an ARRSAC, RANSAC or LMEDS framework. For
 * ARRSAC optional refinement (refine=true) is performed using a Pseudo Huber cost function. For RANSAC optional refinement
 * (refine=true) is performed using a least squares solution. For LMEDS no refinement is available.
 *
 * OutputArray E						Output -> Essential matrix
 * InputArray p1						Input  -> Observed point coordinates of the left image in the camera coordinate system
 *												  (n rows, 2 cols)
 * InputArray p2						Input  -> Observed point coordinates of the right image in the camera coordinate system
 *												  (n rows, 2 cols)
 * string method						Input  -> Name of preferred algorithm: ARRSAC, RANSAC, or LMEDS [Default = ARRSAC]
 * double threshold						Input  -> Threshold [Default=PIX_MIN_GOOD_TH]
 * bool refine							Input  -> If true [Default=true], the essential matrix is refined (except for LMEDS)
 * OutputArray mask						Output -> Inlier mask
 *
 * Return value:						true :	Success
 *										false:	Failed
 */
bool estimateEssentialMat(cv::OutputArray E,
        cv::InputArray p1,
        cv::InputArray p2,
        const std::string &method,
        double threshold,
        bool refine,
        cv::OutputArray mask)
{
	bool result = false;
	if(method == "ARRSAC")
	{
		result = findEssentialMat(E, p1, p2, ARRSAC, 0.999, threshold, mask, refine, robustEssentialRefine); //ARRSAC with optional robust refinement using a Pseudo Huber cost function
	}
	else if(method == "RANSAC")
	{
		result = findEssentialMat(E, p1, p2, cv::RANSAC, 0.999, threshold, mask, refine); //RANSAC with possible subsequent least squares solution
	}
	else if(method == "LMEDS")//Only LMEDS without least squares solution
	{
		result = findEssentialMat(E, p1, p2, cv::LMEDS, 0.999, threshold, mask);
	}
	else if (method == "USAC")
	{
		cout << "USAC must be executed by function estimateEssentialOrPoseUSAC as it needs additional paramters! Exiting." << endl;
		exit(1);
	}
	else
	{
		cout << "Either there is a typo in the specified robust estimation method or the method is not supported. Exiting." << endl;
		exit(1);
	}

	return result;
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

/* Bundle adjustment (BA) on motion (=extrinsics) and structure with or without camera metrices. For refinement of motion and structure without intrinsics,
 * the correspondences p1 and p2 must be in the camera coordinate system (pointsInImgCoords=false). Otherwise the correspondences must be
 * in the image coordinate system (pointsInImgCoords=true). If a mask is provided, only structure elements and correspondences marked as valid
 * used for BA. If BA fails or the refined motion differs too much from the initial motion (thresholds can be modified with angleThresh and t_norm_tresh),
 * the initial motion and structure are stored and the refined data is rejected.
 *
 * InputArray p1						Input  -> Observed point coordinates of the first image in the camera (pointsInImgCoords=false) or image
 *												  (pointsInImgCoords=true) coordinate system (n rows, 2 cols). If pointsInImgCoords=false or 
 												  dist1 and dist2 are not provided, points must be undistorted.
 * InputArray p2						Input  -> Observed point coordinates of the second image in the camera (pointsInImgCoords=false) or image
 *												  (pointsInImgCoords=true) coordinate system (n rows, 2 cols). If pointsInImgCoords=false or 
 												  dist1 and dist2 are not provided, points must be undistorted.
 * InputOutputArray R					I/O	   -> Rotation matrix
 * InputOutputArray t					I/O	   -> Translation vector (3 rows x 1 column)
 * InputOutputArray Q					I/O	   -> Triangulated 3D-points (n rows x 3 columns)
 * InputOutputArray K1					I/O	   -> Camera matrix of the first camera
 * InputOutputArray K2					I/O	   -> Camera matrix of the second camera
 * bool pointsInImgCoords				Input  -> Must be true if p1 and p2 are in image coordinates (pixels) or false if they are in camera coordinates
 *												  [Default=false]. If true, BA is always performed with intrinsics.
 * InputOutputArray mask				Input  -> Inlier mask [Default=cv::noArray()]
 * double angleThresh					Input  -> Threshold on the angular difference in degrees between the initial and refined rotation matrices [Default=1.25]
 * double t_norm_tresh					Input  -> Threshold on the norm of the difference between initial and refined translation vectors [Default=0.05]
 * double huber_thresh					Input  -> Threshold for the Pseudo-Huber cost function in the image coordinate system (pixels)
 * bool optimFocalOnly					Input  -> If pointsInImgCoords=true, only the focal length is optimized (in addition to extrinsics and optionally
 * 												  structure and distortion)
 * bool optimMotionOnly					Input  -> Only extrinsics and intrinsics are optimized, structure remains fixed.
 * bool fixCamMat						Input  -> All parameters within camera matrices are fixed - only extrinsics, distortion, and/or structure are optimized.
 * InputOutputArray dist1				I/O	   -> Exactly 5 distortion coeffitients (k1, k2, t1, t2, k3) of the first camera. 
 * 												  If provided, also dist2 has to be provided. Only if both are provided and pointsInImgCoords=true, 
 * 												  they are optimized.
 * InputOutputArray dist2				I/O	   -> Exactly 5 distortion coeffitients (k1, k2, t1, t2, k3) of the second camera. 
 * 												  If provided, also dist1 has to be provided. Only if both are provided and pointsInImgCoords=true, 
 * 												  they are optimized.
 *
 * Return value:						true :	Success
 *										false:	Failed
 */
bool refineStereoBA(cv::InputArray p1,
					cv::InputArray p2,
					cv::InputOutputArray R,
					cv::InputOutputArray t,
					cv::InputOutputArray Q,
					cv::InputOutputArray K1,
					cv::InputOutputArray K2,
					bool pointsInImgCoords,
					cv::InputArray mask,
					const double angleThresh,
					const double t_norm_tresh,
					const double huber_thresh,
					const bool optimFocalOnly,
					const bool optimMotionOnly,
					const bool fixCamMat,
					cv::InputOutputArray dist1,
					cv::InputOutputArray dist2)
{
    if(p1.empty() || p2.empty() || Q.empty()){
        return false;
    }
    if((p1.rows() != p2.rows()) || (p1.rows() != Q.rows())){
        return false;
    }
	/*const double angleThresh = 0.3;
	const double t_norm_tresh = 0.05;*/
	BAinfo info;
	double t_norm;
	std::vector<double *> t_vec;
	std::vector<double *> R_vec;
	Eigen::Matrix3d R1e;
	Eigen::Vector4d R1quat, R1quat_old;
	double R0[4] = {1.0,0.0,0.0,0.0};
	double t0[3] = {0.0,0.0,0.0};
	std::vector<double *> pts2D_vec;
	std::vector<int> num2Dpts;
	Mat t_ = t.getMat();
	Mat Rq_new;
	double r_diff, t_diff;
	Eigen::Vector3d t1e, t2e;
	Mat t_after_refine;
	Mat K1_tmp, K2_tmp;
	Mat p1_tmp, p2_tmp, Q_tmp;

	int optPars = BA_MOTSTRUCT;
	if (optimMotionOnly){
		optPars = BA_MOT;
	}

	double th = huber_thresh;
	if (th < 0)
	{
		th = ROBUST_THRESH;
	}

	//Prepare input data for the BA interface
	double *pts3D_vec = nullptr;

	cv::cv2eigen(R.getMat(),R1e);
	MatToQuat(R1e, R1quat);
	R1quat_old = R1quat;

	t_vec.push_back(t0);
	t_vec.push_back((double*)t_.data);

	R_vec.push_back(R0);
	R_vec.push_back((double *)R1quat.data());

	if(mask.empty())
	{
		Q_tmp = Q.getMat().clone();
		pts3D_vec = (double*)Q_tmp.data;

		num2Dpts.push_back((int)(p1.getMat().rows));
		num2Dpts.push_back((int)(p2.getMat().rows));

		pts2D_vec.push_back((double *)p1.getMat().data);
		pts2D_vec.push_back((double *)p2.getMat().data);
	}
	else
	{
		Mat mask_ = mask.getMat();
		Mat p1_ = p1.getMat();
		Mat p2_ = p2.getMat();
		Mat Q_ = Q.getMat();
		int n = p1_.rows;

		for(int i = 0; i < n; i++)
		{
			if(mask_.at<unsigned char>(i))
			{
				p1_tmp.push_back(p1_.row(i));
				p2_tmp.push_back(p2_.row(i));
				Q_tmp.push_back(Q_.row(i));
			}
		}
		pts3D_vec = (double*)Q_tmp.data;

		num2Dpts.push_back((int)(p1_tmp.rows));
		num2Dpts.push_back((int)(p2_tmp.rows));

		pts2D_vec.push_back((double *)p1_tmp.data);
		pts2D_vec.push_back((double *)p2_tmp.data);
	}

	t_after_refine = t_.clone();

	if(!pointsInImgCoords) //BA for extrinsics and structure
	{
		Eigen::Vector4d R1quat_old2;

		//Convert the threshold into the camera coordinate system
		th = 4.0*th/(std::sqrt((double)2)*(K1.getMat().at<double>(1,1)+K1.getMat().at<double>(2,2)+K2.getMat().at<double>(1,1)+K2.getMat().at<double>(2,2)));

		SBAdriver optiMotStruct(true, optPars, COST_PSEUDOHUBER, th, optPars);

		if(optiMotStruct.perform_sba(R_vec,t_vec,pts2D_vec,num2Dpts,pts3D_vec,Q_tmp.rows) < 0)
		{
			t_ = t_after_refine.clone();
			return false; //BA failed
		}

		info = optiMotStruct.getFinalSBAinfo();
	}
	else //BA with internals
	{
		Mat K1_ = K1.getMat();
		Mat K2_ = K2.getMat();

		vector<double *> intr_vec;
		double intr1[5] = {K1_.at<double>(0, 0), K1_.at<double>(0, 2), K1_.at<double>(1, 2), K1_.at<double>(1, 1) / K1_.at<double>(0, 0), 0.};
		intr_vec.push_back(intr1);

		double intr2[5] = {K2_.at<double>(0, 0), K2_.at<double>(0, 2), K2_.at<double>(1, 2), K2_.at<double>(1, 1) / K2_.at<double>(0, 0), 0.};
		intr_vec.push_back(intr2);

		int optimInternals = 2;
		if (optimFocalOnly)
		{
			optimInternals = 4;
		}

		double dist1_arr[5] = {0, 0, 0, 0, 0};
		double dist2_arr[5] = {0, 0, 0, 0, 0};
		vector<double *> dist_vec, *dist_vec_ptr = nullptr;
		if (!dist1.empty() && !dist2.empty())
		{
			Mat dist1_ = dist1.getMat();
			Mat dist2_ = dist2.getMat();
			if (!nearZero(cv::sum(dist1_)[0]))
			{
				for (int i = 0; i < 5; i++)
				{
					dist1_arr[i] = dist1_.at<double>(i);
				}
			}
			dist_vec.push_back(dist1_arr);
			if (!nearZero(cv::sum(dist2_)[0]))
			{
				for (int i = 0; i < 5; i++)
				{
					dist2_arr[i] = dist2_.at<double>(i);
				}
			}
			dist_vec.push_back(dist2_arr);
			dist_vec_ptr = &dist_vec;
			if (fixCamMat)
			{
				optimInternals = 5;
			}
		}

		SBAdriver optiMotStruct(false, optPars, COST_PSEUDOHUBER, th, optPars, optimInternals, 0, 0, 0, true);

		if (optiMotStruct.perform_sba(R_vec, t_vec, pts2D_vec, num2Dpts, pts3D_vec, Q_tmp.rows, nullptr, &intr_vec, dist_vec_ptr) < 0)
		{
			/*this->t = oldRTs.back().second.clone(); //-------------------------------> these two lines are executed within uncalibratedRectify for bundle adjustment with internal paramters (if it fails)
			delLastPose(true);*/
			t_ = t_after_refine.clone();
			return false; //BA failed
		}

		info = optiMotStruct.getFinalSBAinfo();

		K1_tmp = cv::Mat::eye(3,3,CV_64FC1);
		K2_tmp = cv::Mat::eye(3,3,CV_64FC1);

		K1_tmp.at<double>(0,0) = intr1[0];
		K1_tmp.at<double>(0,2) = intr1[1];
		K1_tmp.at<double>(1,2) = intr1[2];
		K1_tmp.at<double>(1,1) = intr1[3] * intr1[0];

		K2_tmp.at<double>(0,0) = intr2[0];
		K2_tmp.at<double>(0,2) = intr2[1];
		K2_tmp.at<double>(1,2) = intr2[2];
		K2_tmp.at<double>(1,1) = intr2[3] * intr2[0];

		if (!dist1.empty() && !dist2.empty()){
			Mat dist1_ = dist1.getMat();
			Mat dist2_ = dist2.getMat();
			for (int i = 0; i < 5; i++)
			{
				dist1_.at<double>(i) = dist1_arr[i];
				dist2_.at<double>(i) = dist2_arr[i];
			}
		}

		// free(intr1);
		// free(intr2);
	}

	//Normalize the translation vector
	t_norm = cv::norm(t_);
	if(!nearZero(t_norm) && std::abs(t_norm - 1.0) > 1e-4)
		t_ /= t_norm;

	//Check for the difference of extrinsics before and after BA to detect possible local minimas
	t1e << t_after_refine.at<double>(0), t_after_refine.at<double>(1), t_after_refine.at<double>(2);
	t2e << t_.at<double>(0), t_.at<double>(1), t_.at<double>(2);

	getRTQuality(R1quat_old, R1quat, t1e, t2e, &r_diff, &t_diff);
	r_diff = r_diff/PI * 180.0;
	if((abs(r_diff) > angleThresh) || (t_diff > t_norm_tresh) ||
		(info.terminatingReason > 5))
	{
		/*this->t = oldRTs.back().second.clone(); //-------------------------------> these two lines are executed within uncalibratedRectify for bundle adjustment with internal paramters (if it fails)
		delLastPose(true);*/
		t_ = t_after_refine.clone();
		return false; //BA failed
	}

	//Store the refined paramteres
	if (!optimMotionOnly){
		if (mask.empty())
		{
			Q_tmp.copyTo(Q.getMat());
		}
		else
		{
			Mat mask_ = mask.getMat();
			Mat Q_ = Q.getMat();
			int j = 0, n = Q_.rows;

			for (int i = 0; i < n; i++)
			{
				if (mask_.at<unsigned char>(i))
				{
					Q_tmp.row(j).copyTo(Q_.row(i));
					j++;
				}
			}
		}
	}

	if(pointsInImgCoords)
	{
		K1_tmp.copyTo(K1.getMat());
		K2_tmp.copyTo(K2.getMat());
	}

	cv::eigen2cv(R1quat,Rq_new);
	Mat R_new = R.getMat();
	quatToMatrix(R_new,Rq_new);

	return true;
}

/* Bundle adjustment (BA) on motion (=extrinsics) and structure with or without camera metrices and/or distortion. 
 * For refinement of motion and structure without intrinsics, the correspondences p1 and p2 must be in the 
 * camera coordinate system (pointsInImgCoords=false). Otherwise the correspondences must be
 * in the image coordinate system (pointsInImgCoords=true). If a mask is provided, only structure elements and correspondences marked as valid
 * are used for BA. If BA fails or the refined motion differs too much from the initial motion 
 * (thresholds can be modified with angleThresh and t_norm_tresh), the initial motion and structure are stored and the refined data is rejected.
 *
 * InputArray ps						Input  -> Observed point coordinates of all images in the camera (pointsInImgCoords=false) or image
 *												  (pointsInImgCoords=true) coordinate system (vector<Mat>(n rows, 2 cols)). 
 *												  If pointsInImgCoords=false or dists are not provided, points must be undistorted. 
 *												  The ordering must be the same as in "Qs".
 * InputArray map3D						Input  -> Masks of type char with the same size as Qs for every camera (vector<Mat>(n rows/cols)).
 * 												  The ordering of corresponding 2D points in "ps" must be the same as in "Qs"
 * InputOutputArray Rs					I/O	   -> Rotation matrices (vector<Mat>(3 rows, 3 cols))
 * InputOutputArray ts					I/O	   -> Translation vectors (vector<Mat>(3 rows x 1 column))
 * InputOutputArray Qs					I/O	   -> Triangulated 3D-points (Mat(n rows x 3 columns))
 * InputOutputArray Ks					I/O	   -> Camera matrices (vector<Mat>(3 rows, 3 cols))

 * bool pointsInImgCoords				Input  -> Must be true if ps are in image coordinates (pixels) or false if they are in camera coordinates
 *												  [Default=false]. If true, BA is always performed with intrinsics.
 * InputArray masks						Input  -> Inlier masks [Default=cv::noArray()]: vector<Mat<unsigned char>>(n rows)
 * double angleThresh					Input  -> Threshold on the angular difference in degrees between the initial and refined rotation matrices [Default=1.25]
 * double t_norm_tresh					Input  -> Threshold on the norm of the difference between initial and refined translation vectors [Default=0.05]
 * double huber_thresh					Input  -> Threshold for the Pseudo-Huber cost function in the image coordinate system (pixels)
 * bool optimFocalOnly					Input  -> If pointsInImgCoords=true, only the focal length is optimized (in addition to extrinsics and optionally
 * 												  structure and distortion)
 * bool optimMotionOnly					Input  -> Only extrinsics and intrinsics are optimized, structure remains fixed.
 * bool fixCamMat						Input  -> All parameters within camera matrices are fixed - only extrinsics, distortion, and/or structure are optimized.
 * InputOutputArray dists				I/O	   -> Exactly 5 distortion coeffitients (k1, k2, t1, t2, k3) for all cameras (vector<Mat>(5 coeffitients)). 
 * 												  Only if pointsInImgCoords=true, they are optimized.
 *
 * Return value:						true :	Success
 *										false:	Failed
 */
bool refineMultCamBA(cv::InputArray ps,
					 cv::InputArray map3D,
					 cv::InputOutputArray Rs,
					 cv::InputOutputArray ts,
					 cv::InputOutputArray Qs,
					 cv::InputOutputArray Ks,
					 bool pointsInImgCoords,
					 cv::InputArray masks,
					 const double angleThresh,
					 const double t_norm_tresh,
					 const double huber_thresh,
					 const bool optimFocalOnly,
					 const bool optimMotionOnly,
					 const bool fixCamMat,
					 cv::InputOutputArray dists)
{
	if (ps.empty() || Qs.empty() || map3D.empty() || Rs.empty() || ts.empty() || Ks.empty())
	{
		cerr << "Some input variables to BA are empty. Skipping!" << endl;
		return false;
	}
	if (!ps.isMatVector() || !map3D.isMatVector() || !Rs.isMatVector() || !ts.isMatVector() || !Ks.isMatVector())
	{
		cerr << "Inputs must be of type vector<Mat>. Skipping!" << endl;
		return false;
	}
	if(!Qs.isMat())
	{
		cerr << "Input Qs must be of type Mat. Skipping!" << endl;
		return false;
	}
	Mat Q = Qs.getMat();
	int nr_Qs = Q.size().height;
	if(nr_Qs < 15){
		cerr << "Too less 3D points. Skipping!" << endl;
		return false;
	}

	vector<Mat> p2D_vec, map3D_vec, R_mvec, t_mvec, K_vec;
	ps.getMatVector(p2D_vec);
	map3D.getMatVector(map3D_vec);
	Rs.getMatVector(R_mvec);
	ts.getMatVector(t_mvec);
	Ks.getMatVector(K_vec);
	const size_t vecSi = p2D_vec.size();
	if (map3D_vec.size() != vecSi || R_mvec.size() != vecSi || t_mvec.size() != vecSi || K_vec.size() != vecSi)
	{
		cerr << "Inputs of type vector<Mat> must be of same size. Skipping!" << endl;
		return false;
	}
	bool haveMasks = false;
	vector<Mat> kpMask_vec;
	if (!masks.empty())
	{
		masks.getMatVector(kpMask_vec);
		if (kpMask_vec.size() != vecSi)
		{
			cerr << "Input masks of type vector<Mat> must be of same size as other inputs. Skipping!" << endl;
			return false;
		}
		for (size_t i = 0; i < kpMask_vec.size(); ++i)
		{
			const int mSi = kpMask_vec[i].size().area();
			const int pSi = p2D_vec[i].size().height;
			if (pSi != mSi)
			{
				cerr << "Input mask " << i << " does not have the same size as corresponding 2D features. Skipping BA!" << endl;
				return false;
			}
		}
		haveMasks = true;
	}
	for (size_t i = 0; i < map3D_vec.size(); ++i)
	{
		const int mapSi = map3D_vec[i].size().area();
		const int ones = cv::countNonZero(map3D_vec[i]);
		int pSi = p2D_vec[i].size().height;
		if (haveMasks){
			pSi = kpMask_vec[i].size().area();
		}
		if (nr_Qs != mapSi){
			cerr << "3D map " << i << " does not have the same size as number of available 3D points. Skipping BA!" << endl;
			return false;
		}
		if (ones != pSi){
			cerr << "3D map " << i << " does not contain the same number of non-zero values as provided 2D correspondences. Skipping BA!" << endl;
			return false;
		}
		if (map3D_vec[i].type() != CV_8SC1){
			cerr << "3D map " << i << " must be of type char. Skipping BA!" << endl;
			return false;
		}
	}
	vector<Mat> dist_mvec;
	bool haveDists = false;
	if (!dists.empty())
	{
		dists.getMatVector(dist_mvec);
		if (dist_mvec.size() != vecSi)
		{
			cerr << "Input dists of type vector<Mat> must be of same size as other inputs. Skipping!" << endl;
			return false;
		}
		haveDists = true;
	}
	BAinfo info;
	std::vector<double> t_norm;
	std::vector<double *> t_vec;
	std::vector<double *> R_vec;
	std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> Rquat(vecSi), Rquat_old;
	// double R0[4] = {1.0, 0.0, 0.0, 0.0};
	// double t0[3] = {0.0, 0.0, 0.0};
	std::vector<double *> pts2D_vec;
	std::vector<int> num2Dpts;
	std::vector<char *> map3D_vec_ptr;
	std::vector<Mat> t_save, K_save;
	Mat Q_tmp;
	std::vector<Mat> dist_tmp;
	std::vector<double> f_rel_diff;

	int optPars = BA_MOTSTRUCT;
	if (optimMotionOnly)
	{
		optPars = BA_MOT;
	}

	double th = huber_thresh;
	if (th < 0)
	{
		th = ROBUST_THRESH;
	}

	//Prepare input data for the BA interface
	vector<Mat> pts2d_masked;
	if (haveMasks){
		for (size_t i = 0; i < vecSi; ++i){
			Mat p_ = p2D_vec[i];
			int n = p_.rows;
			Mat mask = kpMask_vec[i];
			Mat p_tmp1;
			for (int j = 0; j < n; j++)
			{
				if (mask.at<unsigned char>(j))
				{
					p_tmp1.push_back(p_.row(j));
				}
			}
			pts2d_masked.emplace_back(p_tmp1.clone());
		}
	}else{
		pts2d_masked = p2D_vec;
	}

	double *pts3D_vec = nullptr;
	for (size_t i = 0; i < vecSi; ++i)
	{
		Eigen::Matrix3d Rei;
		Eigen::Vector4d Rquati;
		cv::cv2eigen(R_mvec[i], Rei);
		MatToQuat(Rei, Rquat[i]);
		// Rquat.push_back(Rquati);
		R_vec.push_back((double *)Rquat[i].data());
		t_vec.push_back((double *)t_mvec[i].data);
		t_save.emplace_back(t_mvec[i].clone());
	}
	Rquat_old = Rquat;

	Q_tmp = Q.clone();
	pts3D_vec = (double *)Q_tmp.data;

	for (size_t i = 0; i < vecSi; ++i){
		num2Dpts.push_back(pts2d_masked[i].rows);
		pts2D_vec.push_back((double *)pts2d_masked[i].data);
		map3D_vec_ptr.push_back((char *)map3D_vec[i].data);
	}

	int err = 0;
	bool failed = false;
	if (!pointsInImgCoords) //BA for extrinsics and structure
	{
		Eigen::Vector4d R1quat_old2;

		double f_mean = 0;
		for (auto &Kx : K_vec){
			f_mean += (Kx.at<double>(1, 1) + Kx.at<double>(2, 2)) / 2.0;
		}
		f_mean /= static_cast<double>(vecSi);
		th /= f_mean;

		SBAdriver optiMotStruct(true, optPars, COST_PSEUDOHUBER, th, optPars);

		err = optiMotStruct.perform_sba(R_vec, t_vec, pts2D_vec, num2Dpts, pts3D_vec, Q_tmp.rows, &map3D_vec_ptr);
		if(err >= 0){
			info = optiMotStruct.getFinalSBAinfo();
			if (info.terminatingReason > 5){
				failed = true;
			}
		}else{
			failed = true;
		}		
	}
	else //BA with internals
	{
		vector<std::array<double, 5>> intr_vec_data;
		vector<double *> intr_vec;
		for (auto &Kx : K_vec)
		{
			intr_vec_data.push_back(std::array<double, 5>{{Kx.at<double>(0, 0), Kx.at<double>(0, 2), Kx.at<double>(1, 2), Kx.at<double>(1, 1) / Kx.at<double>(0, 0), Kx.at<double>(0, 1)}});
			K_save.emplace_back(Kx.clone());
		}
		for (auto &iv : intr_vec_data)
		{
			intr_vec.push_back(iv.data());
		}

		int optimInternals = 2;
		if (optimFocalOnly)
		{
			optimInternals = 4;
		}

		vector<double *> dist_vec, *dist_vec_ptr = nullptr;
		if (haveDists)
		{
			for (auto &distX : dist_mvec)
			{
				dist_tmp.emplace_back(distX.clone());
				dist_vec.push_back((double *)distX.data);
			}
			dist_vec_ptr = &dist_vec;
			if (fixCamMat)
			{
				optimInternals = 5;
			}
		}

		SBAdriver optiMotStruct(false, optPars, COST_PSEUDOHUBER, th, optPars, optimInternals, 0, 0, 0, true);
		//optiMotStruct.setVerbosityLevel(5);

		err = optiMotStruct.perform_sba(R_vec, t_vec, pts2D_vec, num2Dpts, pts3D_vec, Q_tmp.rows, &map3D_vec_ptr, &intr_vec, dist_vec_ptr);
		if (err >= 0)
		{
			info = optiMotStruct.getFinalSBAinfo();
			if (info.terminatingReason > 5)
			{
				failed = true;
			}else{
				for (size_t i = 0; i < vecSi; ++i)
				{
					std::array<double, 5> &arr = intr_vec_data[i];
					Mat K = K_vec[i];
					K.at<double>(0, 0) = arr[0];
					K.at<double>(0, 1) = arr[4];
					K.at<double>(0, 2) = arr[1];
					K.at<double>(1, 2) = arr[2];
					K.at<double>(1, 1) = arr[3] * arr[0];
					if(K_save[i].at<double>(0, 0) > arr[0]){
						f_rel_diff.push_back(K_save[i].at<double>(0, 0) / arr[0]);
					}else{
						f_rel_diff.push_back(arr[0] / K_save[i].at<double>(0, 0));
					}
				}
			}
		}
		else
		{
			failed = true;
		}
	}

	if (!failed){
		for (size_t i = 0; i < vecSi; ++i)
		{
			Mat t_new = t_mvec[i].clone();
			Mat t_old = t_save[i].clone();
			//Normalize the translation vectors
			double t_norm1 = cv::norm(t_new);
			if (!nearZero(t_norm1) && std::abs(t_norm1 - 1.0) > 1e-4)
			{
				t_new /= t_norm1;
			}
			double t_norm2 = cv::norm(t_save[i]);
			if (!nearZero(t_norm2) && std::abs(t_norm2 - 1.0) > 1e-4)
			{
				t_old /= t_norm2;
			}

			//Check for the difference of extrinsics before and after BA to detect possible local minimas
			Eigen::Vector3d t1e, t2e;
			t1e << t_old.at<double>(0), t_old.at<double>(1), t_old.at<double>(2);
			t2e << t_new.at<double>(0), t_new.at<double>(1), t_new.at<double>(2);

			double r_diff, t_diff;
			getRTQuality(Rquat_old[i], Rquat[i], t1e, t2e, &r_diff, &t_diff);
			r_diff = r_diff / PI * 180.0;
			double rf_factor = 1.0, tf_factor = 1.0;
			if (!f_rel_diff.empty())
			{
				tf_factor = std::min(f_rel_diff[i], 2.0);
				rf_factor = std::max(1.0, 0.9 * tf_factor);
				tf_factor = std::min(1.5 * tf_factor, 2.0);
			}
			const double angleThresh2 = rf_factor * angleThresh;
			const double t_norm_tresh2 = tf_factor * t_norm_tresh;
			if ((abs(r_diff) > angleThresh2) || (t_diff > t_norm_tresh2))
			{
				failed = true;
				break;
			}
		}
	}

	if(failed){
		int i = 0;
		for (auto &&old_t : t_save)
		{
			t_mvec[i++] = std::move(old_t);
		}
		if (pointsInImgCoords && haveDists)
		{
			i = 0;
			for (auto &&distX : dist_tmp)
			{
				dist_mvec[i++] = std::move(distX);
			}
		}
		if(!K_save.empty()){
			for (size_t i = 0; i < vecSi; ++i){
				K_save[i].copyTo(K_vec[i]);
			}
		}
		return false; //BA failed
	}

	//Store the refined paramteres
	if (!optimMotionOnly)
	{
		Q_tmp.copyTo(Q);
	}

	Eigen::Matrix3d Rei;
	Eigen::Vector4d Rquati;

	for (size_t i = 0; i < vecSi; ++i)
	{
		Mat Rq_new;
		cv::eigen2cv(Rquat[i], Rq_new);
		quatToMatrix(R_mvec[i], Rq_new);
	}

	return true;
}

int estimateEssentialOrPoseUSAC(const cv::Mat &p1,
								const cv::Mat &p2,
								cv::OutputArray E,
								double th,
								ConfigUSAC &cfg,
								bool &isDegenerate,
								cv::OutputArray inliers,
								cv::OutputArray R_degenerate,
								cv::OutputArray inliers_degenerate_R,
								cv::OutputArray R,
								cv::OutputArray t,
								bool verbose)
{
	//const double degenerateDecisionThreshold = 0.85;//Used for the option UsacChkDegenType::DEGEN_USAC_INTERNAL: Threshold on the fraction of degenerate inliers compared to the E-inlier fraction
	USACConfig::EssentialMatEstimatorUsed used_estimator;
	USACConfig::RefineAlgorithm	refineMethod;
	unsigned int nrInliers = 0;
	double prosac_beta = 0.09, sprt_delta = 0.05, sprt_epsilon = 0.15, sprt_delta_res = 0, sprt_epsilon_res = 0;
	static double sprt_delta_old = 0, sprt_delta_new = 0, sprt_epsilon_old = 0, sprt_epsilon_new = 0;
	const int historylength = 20;//If this value is changed, also the array sizes of sprt_delta_history and sprt_epsilon_history have to be changed to the same value
	static double sprt_delta_history[20], sprt_epsilon_history[20];
	static int historyCnt = 0;
	static bool historyBufFull = false;
	static statVals sprt_delta_stat, sprt_epsilon_stat;
	static bool statistic_valid = false;
	std::vector<unsigned int> sortedMatchIdx, *sortedMatchIdxPtr = nullptr;
	switch (cfg.estimator)
	{
	case(PoseEstimator::POSE_EIG_KNEIP):
		used_estimator = USACConfig::EssentialMatEstimatorUsed::ESTIM_EIG_KNEIP;
		break;
	case(PoseEstimator::POSE_NISTER):
		used_estimator = USACConfig::EssentialMatEstimatorUsed::ESTIM_NISTER;
		break;
	case(PoseEstimator::POSE_STEWENIUS):
		used_estimator = USACConfig::EssentialMatEstimatorUsed::ESTIM_STEWENIUS;
		break;
	default:
		cout << "Estimator not supported!" << endl;
		return -1;
	}
	switch (cfg.refinealg)
	{
	case(RefineAlg::REF_8PT_PSEUDOHUBER):
		refineMethod = USACConfig::RefineAlgorithm::REFINE_8PT_PSEUDOHUBER;
		break;
	case(RefineAlg::REF_EIG_KNEIP):
		refineMethod = USACConfig::RefineAlgorithm::REFINE_EIG_KNEIP;
		break;
	case(RefineAlg::REF_EIG_KNEIP_WEIGHTS):
		refineMethod = USACConfig::RefineAlgorithm::REFINE_EIG_KNEIP_WEIGHTS;
		break;
	case(RefineAlg::REF_WEIGHTS):
		refineMethod = USACConfig::RefineAlgorithm::REFINE_WEIGHTS;
		break;
	case(RefineAlg::REF_STEWENIUS):
		refineMethod = USACConfig::RefineAlgorithm::REFINE_STEWENIUS;
		break;
	case(RefineAlg::REF_STEWENIUS_WEIGHTS):
		refineMethod = USACConfig::RefineAlgorithm::REFINE_STEWENIUS_WEIGHTS;
		break;
	case(RefineAlg::REF_NISTER):
		refineMethod = USACConfig::RefineAlgorithm::REFINE_NISTER;
		break;
	case(RefineAlg::REF_NISTER_WEIGHTS):
		refineMethod = USACConfig::RefineAlgorithm::REFINE_NISTER_WEIGHTS;
		break;
	default:
		cout << "Refinement algorithm not supported!" << endl;
		return -1;
	}

	//Estimate initial values for delta and epsilon of the SPRT test
	const double statStdDivTh[2] = { 0.1, 0.2 };//Threshold for delta and epsilon on their history standard deviation
	const double relativeDifferenceTh[2] = { 0.33, 0.4 };//Threshold for delta and epsilon on their relative difference of the last two values
	const double relDiffArithToStd = 0.45;//Used to choose if the statistic should be used: The standard deviation must be smaller than relDiffArithToStd times the arithmetic mean value
	const double ignoreRelDiffATSTh[2] = { 0.1, 0.1 };//The relative threshold using relDiffArithToStd is ignored, if the arithmetic mean is below this value for delta and epsilon
	if (cfg.automaticSprtInit == SprtInit::SPRT_DELTA_AUTOM_INIT)
	{
		if (sprt_delta_old == 0 || sprt_delta_new == 0)
		{
			sprt_delta = estimateSprtDeltaInit(*cfg.matches, *cfg.keypoints1, *cfg.keypoints2, cfg.th_pixels, cfg.imgSize);
			sprt_delta_new = sprt_delta;
		}
		else if (!statistic_valid || (statistic_valid && ((sprt_delta_stat.arithStd > statStdDivTh[0]) || ((sprt_delta_stat.arithErr > ignoreRelDiffATSTh[0]) && (sprt_delta_stat.arithStd > relDiffArithToStd * sprt_delta_stat.arithErr)))))
		{
			if (abs((sprt_delta_old - sprt_delta_new) / sprt_delta_old) < relativeDifferenceTh[0])
			{
				sprt_delta = sprt_delta_new;
			}
			else
			{
				sprt_delta = estimateSprtDeltaInit(*cfg.matches, *cfg.keypoints1, *cfg.keypoints2, cfg.th_pixels, cfg.imgSize);
			}
		}
		else
		{
			sprt_delta = sprt_delta_stat.arithErr;
		}

		/*if (statistic_valid &&
			(sprt_epsilon_stat.arithStd <= statStdDivTh[1]) &&
			((sprt_epsilon_stat.arithErr <= ignoreRelDiffATSTh[1]) ||
			(sprt_epsilon_stat.arithStd <= relDiffArithToStd * sprt_epsilon_stat.arithErr)))
		{
			sprt_epsilon = sprt_epsilon_stat.arithErr;
		}
		else
		{
			sprt_epsilon = 0.15;
		}*/
	}
	else if (cfg.automaticSprtInit == SprtInit::SPRT_EPSILON_AUTOM_INIT)
	{
		if (sprt_epsilon_old == 0 || sprt_epsilon_new == 0)
		{
			sprt_epsilon = estimateSprtEpsilonInit(*cfg.matches, cfg.nrMatchesVfcFiltered);
			sprt_epsilon_new = sprt_epsilon;
		}
		else if (!statistic_valid || 
			(statistic_valid && 
			((sprt_epsilon_stat.arithStd > statStdDivTh[1]) || 
				((sprt_epsilon_stat.arithErr > ignoreRelDiffATSTh[1]) && 
				(sprt_epsilon_stat.arithStd > relDiffArithToStd * sprt_epsilon_stat.arithErr)))))
		{
			if (abs((sprt_epsilon_old - sprt_epsilon_new) / sprt_epsilon_old) < relativeDifferenceTh[1])
			{
				sprt_epsilon = sprt_epsilon_new;
			}
			else
			{
				sprt_epsilon = estimateSprtEpsilonInit(*cfg.matches, cfg.nrMatchesVfcFiltered);
			}
		}
		else
		{
			sprt_epsilon = sprt_epsilon_stat.arithErr;
		}

		/*if (statistic_valid &&
			(sprt_delta_stat.arithStd <= statStdDivTh[0]) &&
			((sprt_delta_stat.arithErr <= ignoreRelDiffATSTh[0]) ||
			(sprt_delta_stat.arithStd <= relDiffArithToStd * sprt_delta_stat.arithErr)))
		{
			sprt_delta = sprt_delta_stat.arithErr;
			prosac_beta = sprt_delta;
		}
		else
		{
			sprt_delta = 0.05;
			prosac_beta = 0.09;
		}*/
	}
	else if (cfg.automaticSprtInit == (SprtInit::SPRT_DELTA_AUTOM_INIT | SprtInit::SPRT_EPSILON_AUTOM_INIT))
	{
		if (sprt_delta_old == 0 || sprt_delta_new == 0)
		{
			sprt_delta = estimateSprtDeltaInit(*cfg.matches, *cfg.keypoints1, *cfg.keypoints2, cfg.th_pixels, cfg.imgSize);
			sprt_delta_new = sprt_delta;
		}
		else if (!statistic_valid || 
			(statistic_valid && 
			((sprt_delta_stat.arithStd > statStdDivTh[0]) || 
				((sprt_delta_stat.arithErr > ignoreRelDiffATSTh[0]) && 
				(sprt_delta_stat.arithStd > relDiffArithToStd * sprt_delta_stat.arithErr)))))
		{
			if (abs((sprt_delta_old - sprt_delta_new) / sprt_delta_old) < relativeDifferenceTh[0])
			{
				sprt_delta = sprt_delta_new;
			}
			else
			{
				sprt_delta = estimateSprtDeltaInit(*cfg.matches, *cfg.keypoints1, *cfg.keypoints2, cfg.th_pixels, cfg.imgSize);
			}
		}
		else
		{
			sprt_delta = sprt_delta_stat.arithErr;
		}
		if (sprt_epsilon_old == 0 || sprt_epsilon_new == 0)
		{
			sprt_epsilon = estimateSprtEpsilonInit(*cfg.matches, cfg.nrMatchesVfcFiltered);
			sprt_epsilon_new = sprt_epsilon;
		}
		else if (!statistic_valid || 
			(statistic_valid && 
			((sprt_epsilon_stat.arithStd > statStdDivTh[1]) || 
				((sprt_epsilon_stat.arithErr > ignoreRelDiffATSTh[1]) && 
				(sprt_epsilon_stat.arithStd > relDiffArithToStd * sprt_epsilon_stat.arithErr)))))
		{
			if (abs((sprt_epsilon_old - sprt_epsilon_new) / sprt_epsilon_old) < relativeDifferenceTh[1])
			{
				sprt_epsilon = sprt_epsilon_new;
			}
			else
			{
				sprt_epsilon = estimateSprtEpsilonInit(*cfg.matches, cfg.nrMatchesVfcFiltered);
			}
		}
		else
		{
			sprt_epsilon = sprt_epsilon_stat.arithErr;
		}
	}
	/*else if (cfg.automaticSprtInit == SprtInit::SPRT_DEFAULT_INIT)
	{
		if (statistic_valid && 
			(sprt_delta_stat.arithStd <= statStdDivTh[0]) && 
			((sprt_delta_stat.arithErr <= ignoreRelDiffATSTh[0]) || 
			(sprt_delta_stat.arithStd <= relDiffArithToStd * sprt_delta_stat.arithErr)))
		{
			sprt_delta = sprt_delta_stat.arithErr;
			prosac_beta = sprt_delta;
		}
		else
		{
			sprt_delta = 0.05;
			prosac_beta = 0.09;
		}
		if (statistic_valid && 
			(sprt_epsilon_stat.arithStd <= statStdDivTh[1]) && 
			((sprt_epsilon_stat.arithErr <= ignoreRelDiffATSTh[1]) || 
			(sprt_epsilon_stat.arithStd <= relDiffArithToStd * sprt_epsilon_stat.arithErr)))
		{
			sprt_epsilon = sprt_epsilon_stat.arithErr;
		}
		else
		{
			sprt_epsilon = 0.15;
		}
	}*/
	else if (cfg.automaticSprtInit != SprtInit::SPRT_DEFAULT_INIT)
	{
		cout << "Method for SPRT initialization not supported!" << endl;
		return -1;
	}
	if (!cfg.noAutomaticProsacParamters)
	{
		prosac_beta = sprt_delta;
	}
	if (cfg.matches)
	{
		getSortedMatchIdx(*cfg.matches, sortedMatchIdx);
		sortedMatchIdxPtr = &sortedMatchIdx;
	}

	if (cfg.degeneracyCheck == UsacChkDegenType::DEGEN_NO_CHECK)
	{
		Mat R_degen, inl_degen, inliers_degenerate_noMotion, R_kneip, t_kneip;
		double fraction_degen_inliers_R = 0, fraction_degen_inliers_noMot = 0;
		if (estimateEssentialMatUsac(p1,
			p2,
			E,
			sprt_delta_res,
			sprt_epsilon_res,
			th,
			cfg.focalLength,
			cfg.th_pixels,
			prosac_beta,
			sprt_delta,
			sprt_epsilon,
			false,
			used_estimator,
			refineMethod,
			inliers,
			&nrInliers,
			cv::noArray(),
			cv::noArray(),
			R_degen,
			inl_degen,
			cv::noArray(),
			cv::noArray(),
			inliers_degenerate_noMotion,
                                     nullptr,
			&fraction_degen_inliers_R,
                                     nullptr,
			&fraction_degen_inliers_noMot,
			*sortedMatchIdxPtr,
			R_kneip,
			t_kneip,
			verbose) == EXIT_FAILURE)
		{
			cout << "USAC failed!" << endl;
			return -2;
		}
		isDegenerate = false;
		if ((used_estimator == USACConfig::ESTIM_EIG_KNEIP
		|| refineMethod == USACConfig::REFINE_EIG_KNEIP
		|| refineMethod == USACConfig::REFINE_EIG_KNEIP_WEIGHTS) &&
			!R_kneip.empty() && !t_kneip.empty() && R.needed() && t.needed())
		{
			if (R.empty())
				R.create(3, 3, CV_64F);
			R_kneip.copyTo(R.getMat());
			if (t.empty())
				t.create(3, 1, CV_64F);
			t_kneip.copyTo(t.getMat());
		}
		else if (R.needed() || t.needed())
		{
			if (R.needed() && !R.empty())
				R.clear();
			if (t.needed() && !t.empty())
				t.clear();
		}
	}
	else if (cfg.degeneracyCheck == UsacChkDegenType::DEGEN_USAC_INTERNAL)
	{
		Mat R_degen, inl_degen, inliers_degenerate_noMotion, R_kneip, t_kneip;
		double fraction_degen_inliers_R = 0, fraction_degen_inliers_noMot = 0, fraction_inliers = 0;
		if (estimateEssentialMatUsac(p1,
			p2,
			E,
			sprt_delta_res,
			sprt_epsilon_res,
			th,
			cfg.focalLength,
			cfg.th_pixels,
			prosac_beta,
			sprt_delta,
			sprt_epsilon,
			true,
			used_estimator,
			refineMethod,
			inliers,
			&nrInliers,
			cv::noArray(),
			cv::noArray(),
			R_degen,
			inl_degen,
			cv::noArray(),
			cv::noArray(),
			inliers_degenerate_noMotion,
                                     nullptr,
			&fraction_degen_inliers_R,
                                     nullptr,
			&fraction_degen_inliers_noMot,
			*sortedMatchIdxPtr,
			R_kneip,
			t_kneip,
			verbose) == EXIT_FAILURE)
		{
			cout << "USAC failed!" << endl;
			return -2;
		}

		if ((used_estimator == USACConfig::ESTIM_EIG_KNEIP
		|| refineMethod == USACConfig::REFINE_EIG_KNEIP
		|| refineMethod == USACConfig::REFINE_EIG_KNEIP_WEIGHTS) &&
			!R_kneip.empty() && !t_kneip.empty() && R.needed() && t.needed())
		{
			if (R.empty())
				R.create(3, 3, CV_64F);
			R_kneip.copyTo(R.getMat());
			if (t.empty())
				t.create(3, 1, CV_64F);
			t_kneip.copyTo(t.getMat());
		}
		else if (R.needed() || t.needed())
		{
			if (R.needed() && !R.empty())
				R.clear();
			if (t.needed() && !t.empty())
				t.clear();
		}
		fraction_inliers = (double)nrInliers / (double)p1.rows;
		if (cfg.degenDecisionTh * fraction_inliers < fraction_degen_inliers_R)
		{
			isDegenerate = true;
			if (R_degenerate.needed() && !R_degen.empty())
			{
				if(R_degenerate.empty())
					R_degenerate.create(3, 3, CV_64F);
				R_degen.copyTo(R_degenerate.getMat());
			}
			if (inliers_degenerate_R.needed() && !inl_degen.empty())
			{
				if (inliers_degenerate_R.empty())
					inliers_degenerate_R.create(1, inl_degen.cols, CV_8U);
				inl_degen.copyTo(inliers_degenerate_R.getMat());
			}
		}
		else if (cfg.degenDecisionTh * fraction_inliers < fraction_degen_inliers_noMot)
		{
			isDegenerate = true;
			if (R_degenerate.needed())
			{
				R_degen = cv::Mat::eye(3, 3, CV_64FC1);
				if (R_degenerate.empty())
					R_degenerate.create(3, 3, CV_64F);
				R_degen.copyTo(R_degenerate.getMat());
			}
			if (inliers_degenerate_R.needed() && !inliers_degenerate_noMotion.empty())
			{
				if (inliers_degenerate_R.empty())
					inliers_degenerate_R.create(1, inliers_degenerate_noMotion.cols, CV_8U);
				inliers_degenerate_noMotion.copyTo(inliers_degenerate_R.getMat());
			}
		}
	}
	else if (cfg.degeneracyCheck == UsacChkDegenType::DEGEN_QDEGSAC)
	{
		Mat R_degen, inl_degen, R_kneip, t_kneip, E_, inliers_;
		double fraction_degen_inliers_R = 0;
		if (estimateEssentialQDEGSAC(p1,
			p2,
			E_,
			sprt_delta_res,
			sprt_epsilon_res,
			th,
			cfg.focalLength,
			cfg.th_pixels,
			prosac_beta,
			sprt_delta,
			sprt_epsilon,
			0.5,//Inlier ratio threshold for detecting degeneracy (should be between 0.5 and 0.8)
			used_estimator,
			refineMethod,
			inliers_,
			&nrInliers,
			R_degen,
			inl_degen,
			&fraction_degen_inliers_R,
			*sortedMatchIdxPtr,
			R_kneip,
			t_kneip,
			verbose) == EXIT_FAILURE)
		{
			cout << "USAC failed!" << endl;
			return -2;
		}
		if ((used_estimator == USACConfig::ESTIM_EIG_KNEIP
		|| refineMethod == USACConfig::REFINE_EIG_KNEIP
		|| refineMethod == USACConfig::REFINE_EIG_KNEIP_WEIGHTS) &&
			!R_kneip.empty() && !t_kneip.empty() && R.needed() && t.needed())
		{
			if (R.empty())
				R.create(3, 3, CV_64F);
			R_kneip.copyTo(R.getMat());
			if (t.empty())
				t.create(3, 1, CV_64F);
			t_kneip.copyTo(t.getMat());
		}
		else if (R.needed() || t.needed())
		{
			if (R.needed() && !R.empty())
				R.clear();
			if (t.needed() && !t.empty())
				t.clear();
		}
		if (E_.empty() || inliers_.empty())
		{
			isDegenerate = true;
			if (R_degenerate.needed() && !R_degen.empty())
			{
				if (R_degenerate.empty())
					R_degenerate.create(3, 3, CV_64F);
				R_degen.copyTo(R_degenerate.getMat());
			}
			if (inliers_degenerate_R.needed() && !inl_degen.empty())
			{
				if (inliers_degenerate_R.empty())
					inliers_degenerate_R.create(1, inl_degen.cols, CV_8U);
				inl_degen.copyTo(inliers_degenerate_R.getMat());
			}
			if (!E.empty())
				E.clear();
			if (inliers.needed() && !inliers.empty())
				inliers.clear();
			if (R.needed() && !R.empty())
				R.clear();
			if (t.needed() && !t.empty())
				t.clear();
		}
		else
		{
			isDegenerate = false;
			if (E.empty())
				E.create(3, 3, CV_64F);
			E_.copyTo(E.getMat());
			if (inliers.needed())
			{
				if (inliers.empty())
					inliers.create(1, inliers_.cols, CV_8U);
				inliers_.copyTo(inliers.getMat());
			}
			if (inliers_degenerate_R.needed() && !inliers_degenerate_R.empty())
				inliers_degenerate_R.clear();
			if (R_degenerate.needed() && !R_degenerate.empty())
				R_degenerate.clear();
		}
	}
	else
	{
		cout << "Mothod for checking degeneracy not available!" << endl;
		return -1;
	}
	sprt_delta_old = sprt_delta_new;
	sprt_delta_new = sprt_delta_res;
	sprt_epsilon_old = sprt_epsilon_new;
	sprt_epsilon_new = sprt_epsilon_res;

	//History
	if (!nearZero(sprt_delta_res) && !nearZero(sprt_epsilon_res))
	{
		sprt_delta_history[historyCnt] = sprt_delta_res;
		sprt_epsilon_history[historyCnt] = sprt_epsilon_res;
		historyCnt = (historyCnt + 1) % historylength;
		if (historyCnt == 0) historyBufFull = true;

		if (historyBufFull)
		{
			getStatsfromVec(std::vector<double>(sprt_delta_history, sprt_delta_history + historylength), &sprt_delta_stat, true);
			getStatsfromVec(std::vector<double>(sprt_epsilon_history, sprt_epsilon_history + historylength), &sprt_epsilon_stat, true);
		}
		else if (historyCnt > 5)
		{
			getStatsfromVec(std::vector<double>(sprt_delta_history, sprt_delta_history + historyCnt), &sprt_delta_stat, true);
			getStatsfromVec(std::vector<double>(sprt_epsilon_history, sprt_epsilon_history + historyCnt), &sprt_epsilon_stat, true);
			statistic_valid = true;
		}
	}

	return 0;
}
}
