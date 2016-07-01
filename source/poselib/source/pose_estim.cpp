/**********************************************************************************************************
 FILE: pose_estim.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: May 2016

 LOCATION: TechGate Vienna, Donau-City-Straße 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for the estimation and optimization of poses between
			  two camera views (images).
**********************************************************************************************************/

#include "pose_estim.h"
#include "pose_helper.h"
#include "five-point-nister\five-point.hpp"
#include "BA_driver.h"

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
	//double pixToCamFact = 4.0/(std::sqrt((double)2)*(this->K1.at<double>(1,1)+this->K1.at<double>(2,2)+this->K2.at<double>(1,1)+this->K2.at<double>(2,2)));

	getReprojErrors(_E,_p1,_p2,useImgCoordSystem,NULL,&error);

	for(size_t i = 0; i < error.size(); i++)
		error[i] = std::sqrt(error[i]);

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
	//double pixToCamFact = 4.0/(std::sqrt((double)2)*(this->K1.at<double>(1,1)+this->K1.at<double>(2,2)+this->K2.at<double>(1,1)+this->K2.at<double>(2,2)));

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
 *
 * Return value:					none
 */
void robustEssentialRefine(cv::InputArray points1, cv::InputArray points2, cv::InputArray E_init, cv::Mat & E_refined,
						  double th, unsigned int iters, bool makeClosestE, double *sumSqrErr_init,
						  double *sumSqrErr, cv::OutputArray errors, cv::InputOutputArray mask, int model)
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
		pointsRedu1 = pointsRedu1.t();
		pointsRedu1 = pointsRedu1.reshape(2,1);
		pointsRedu2 = pointsRedu2.t();
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
			pointsRedu1 = _points1.t();
			pointsRedu1 = pointsRedu1.reshape(2,1);
			pointsRedu2 = _points2.t();
			pointsRedu2 = pointsRedu2.reshape(2,1);
		}
	}

	if(npoints < 50)
	{
		E.copyTo(E_refined);
		cout << "There are too less points for a refinement left!" << endl;
		return;
	}
	E = E_init.getMat().clone();

	//Thresholds for iterative refinement
	const double minSumSqrErrDiff = th/10;
	const double minSumSqrErr = th*th/100 * npoints;
	const unsigned int maxIters = 50;

    CvPoint2D64f m0c = {0,0}, m1c = {0,0};
    double t, scale0 = 0, scale1 = 0;

	const CvPoint2D64f* _m1 = (const CvPoint2D64f*)pointsRedu1.data;
    const CvPoint2D64f* _m2 = (const CvPoint2D64f*)pointsRedu2.data;

    int i;

    // compute centers and average distances for each of the two point sets
    for( i = 0; i < npoints; i++ )
    {
        double x = _m1[i].x, y = _m1[i].y;
        m0c.x += x; m0c.y += y;

        x = _m2[i].x, y = _m2[i].y;
        m1c.x += x; m1c.y += y;
    }

    // calculate the normalizing transformations for each of the point sets:
    // after the transformation each set will have the mass center at the coordinate origin
    // and the average distance from the origin will be ~sqrt(2).
    t = 1./npoints;
	if(model != 2)
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

    for( i = 0; i < npoints; i++ )
    {
        double x = _m1[i].x - m0c.x, y = _m1[i].y - m0c.y;
        scale0 += sqrt(x*x + y*y);

        x = _m2[i].x - m1c.x, y = _m2[i].y - m1c.y;
        scale1 += sqrt(x*x + y*y);
    }

	if(model != 2)
	{
		scale0 *= t;
		scale1 *= t;
	}
	else
	{
		scale1 = scale0 = (scale0 + scale1) * t;
	}

    if( scale0 < FLT_EPSILON || scale1 < FLT_EPSILON )
        return;

    scale0 = sqrt(2.)/scale0;
    scale1 = sqrt(2.)/scale1;

	Mat H0 = (Mat_<double>(3, 3) << scale0, 0, -scale0*m0c.x,
									0, scale0, -scale0*m0c.y,
									0, 0, 1);

	Mat H1 = (Mat_<double>(3, 3) << scale1, 0, -scale1*m1c.x,
									0, scale1, -scale1*m1c.y,
									0, 0, 1);

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
		for( i = 0; i < npoints; i++ )
		{
			double denom1, num;
			SampsonL1(pointsRedu1.col(i).reshape(1,2), pointsRedu2.col(i).reshape(1,2), F3, denom1, num);
			const double weight = costPseudoHuber(num*denom1,th);


			double x0 = (_m1[i].x - m0c.x)*scale0;
			double y0 = (_m1[i].y - m0c.y)*scale0;
			double x1 = (_m2[i].x - m1c.x)*scale1;
			double y1 = (_m2[i].y - m1c.y)*scale1;

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

			A1.row(i) *= denom1*weight; /* multiply by 1/dominator of the Sampson L1-distance to eliminate
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
			Eigen::Matrix<double, 9, 1 > lastCol = svdA.matrixV().col(8);*/

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
					validE = validateEssential(_points1, _points2, F2, false, mask);
				else
					validE = validateEssential(_points1, _points2, F2);
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
					validE = validateEssential(_points1, _points2, F2, true, mask);
				else
					validE = validateEssential(_points1, _points2, F2, true);
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
				validE = validateEssential(_points1, _points2, F2, false, mask);
			else
				validE = validateEssential(_points1, _points2, F2);
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
				validE = validateEssential(_points1, _points2, F2, false, mask);
			else
				validE = validateEssential(_points1, _points2, F2);
			if(!validE)
			{
				cout << "E is no valid essential matrix - taking last valid E!" << endl;
				break;
			}
		}

		cv::eigen2cv(F2,F3);
		F3 = H1.t() * F3 * H0;

		if(!iters)
		{
			err = (A1 * lastCol).squaredNorm();
			//if(err < err_min)
			//{
			//	E_refined = F3.clone();
			//	err_min = err;
			//}
			const double errdiff = err_old - err;
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
	if(model != 2)
		E_refined /= E_refined.at<double>(2,2);
	else
		cv::normalize(E_refined, E_refined);
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
bool estimateEssentialMat(cv::OutputArray E, cv::InputArray p1, cv::InputArray p2, std::string method, double threshold, bool refine, cv::OutputArray mask)
{
	bool result = false;
	if(!method.compare("ARRSAC"))
	{
		result = findEssentialMat(E, p1, p2, ARRSAC, 0.999, threshold, mask, refine, robustEssentialRefine); //ARRSAC with optional robust refinement using a Pseudo Huber cost function
	}
	else if(!method.compare("RANSAC"))
	{
		result = findEssentialMat(E, p1, p2, CV_RANSAC, 0.999, threshold, mask, refine); //RANSAC with possible subsequent least squares solution
	}
	else if(!method.compare("LMEDS"))//Only LMEDS without least squares solution
	{
		result = findEssentialMat(E, p1, p2, CV_LMEDS, 0.999, threshold, mask);
	}
	else
	{
		cout << "Either there is a typo in the specified robust estimation method or the method is not supported. Exiting." << endl;
		exit(0);
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
int getPoseTriangPts(cv::InputArray E, cv::InputArray p1, cv::InputArray p2, cv::OutputArray R, cv::OutputArray t, cv::OutputArray Q, cv::InputOutputArray mask, const double dist, bool translatE)
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
	int npoints = points1.checkVector(2);
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
 *												  (pointsInImgCoords=true) coordinate system (n rows, 2 cols). Points must be undistorted.
 * InputArray p2						Input  -> Observed point coordinates of the second image in the camera (pointsInImgCoords=false) or image 
 *												  (pointsInImgCoords=true) coordinate system (n rows, 2 cols). Points must be undistorted.
 * InputOutputArray R					I/O	   -> Rotation matrix
 * InputOutputArray t					I/O	   -> Translation vector (3 rows x 1 column)
 * InputOutputArray Q					I/O	   -> Triangulated 3D-points (n rows x 3 columns)
 * InputOutputArray K1					I/O	   -> Camera matrix of the first camera
 * InputOutputArray K2					I/O	   -> Camera matrix of the second camera
 * bool pointsInImgCoords				Input  -> Must be true if p1 and p2 are in image coordinates (pixels) or false if they are in camera coordinates
 *												  [Default=false]. If true, BA is always performed with intrinsics.
 * InputOutputArray mask				Input  -> Inlier mask [Default=cv::noArray()]
 * double angleThresh					Input  -> Threshold on the angular difference in degrees between the initial and refined rotation matrices [Default=0.3]
 * double t_norm_tresh					Input  -> Threshold on the norm of the difference between initial and refined translation vectors [Default=0.05]
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
					const double t_norm_tresh)
{
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

	//Prepare input data for the BA interface
	double * pts3D_vec = NULL;

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
		double th = ROBUST_THRESH;
		th = 4*th/(std::sqrt((double)2)*(K1.getMat().at<double>(1,1)+K1.getMat().at<double>(2,2)+K2.getMat().at<double>(1,1)+K2.getMat().at<double>(2,2)));

		SBAdriver optiMotStruct(true,BA_MOTSTRUCT,COST_PSEUDOHUBER,th,true,BA_MOTSTRUCT);

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
		double *intr1, *intr2;
		intr1 = (double*)malloc(5*sizeof(double));
		if(intr1 == NULL) exit(1);

		intr2 = (double*)malloc(5*sizeof(double));
		if(intr2 == NULL) exit(1);

		intr1[0] = K1_.at<double>(0,0);
		intr1[1] = K1_.at<double>(0,2);
		intr1[2] = K1_.at<double>(1,2);
		intr1[3] = K1_.at<double>(1,1)/K1_.at<double>(0,0);
		intr1[4] = 0.0;
		intr_vec.push_back(intr1);

		intr2[0] = K2_.at<double>(0,0);
		intr2[1] = K2_.at<double>(0,2);
		intr2[2] = K2_.at<double>(1,2);
		intr2[3] = K2_.at<double>(1,1)/K2_.at<double>(0,0);
		intr2[4] = 0.0;
		intr_vec.push_back(intr2);

		SBAdriver optiMotStruct(false,BA_MOTSTRUCT,COST_PSEUDOHUBER,ROBUST_THRESH,true,BA_MOTSTRUCT);

		if(optiMotStruct.perform_sba(R_vec,t_vec,pts2D_vec,num2Dpts,pts3D_vec,Q_tmp.rows,NULL,&intr_vec) < 0)
		{
			/*this->t = oldRTs.back().second.clone(); //-------------------------------> these two lines are executed within uncalibratedRectify for bundle adjustment with internal paramters (if it fails)
			delLastPose(true);*/
			t_ = t_after_refine.clone();
			return false; //BA failed
		}

		info = optiMotStruct.getFinalSBAinfo();

		K1_tmp.create(3,3,CV_64FC1);
		K2_tmp.create(3,3,CV_64FC1);

		K1_tmp.at<double>(0,0) = intr1[0];
		K1_tmp.at<double>(0,2) = intr1[1];
		K1_tmp.at<double>(1,2) = intr1[2];
		K1_tmp.at<double>(1,1) = intr1[3] * intr1[0];
		
		K2_tmp.at<double>(0,0) = intr2[0];
		K2_tmp.at<double>(0,2) = intr2[1];
		K2_tmp.at<double>(1,2) = intr2[2];
		K2_tmp.at<double>(1,1) = intr2[3] * intr2[0];

		free(intr1);
		free(intr2);
	}

	//Normalize the translation vector
	t_norm = normFromVec(t_);
	if(std::abs(t_norm-1.0) > 1e-4)
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
	if(mask.empty())
	{
		Q_tmp.copyTo(Q.getMat());
	}
	else
	{
		Mat mask_ = mask.getMat();
		Mat Q_ = Q.getMat();
		int j = 0, n = Q_.rows;

		for(int i = 0; i < n; i++)
		{
			if(mask_.at<unsigned char>(i))
			{
				Q_tmp.row(j).copyTo(Q_.row(i));
				j++;
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
}