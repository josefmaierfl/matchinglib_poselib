/**********************************************************************************************************
 FILE: pose_helper.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: May 2016

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides helper functions for the estimation and optimization of poses between
			  two camera views (images).
**********************************************************************************************************/

#include "pose_helper.h"

#include <Eigen/SVD>
#include <Eigen/Dense>

using namespace cv;
using namespace std;

/* --------------------------- Defines --------------------------- */



/* --------------------- Function prototypes --------------------- */



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
void SampsonL1(const cv::Mat x1, const cv::Mat x2, const cv::Mat E, double & denom1, double & num)
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

	if(!nearZero(svdE.singularValues()[2])) return -1; // E is no essential matrix
	else if((svdE.singularValues()[0]/svdE.singularValues()[1] > 1.5) ||
			(svdE.singularValues()[0]/svdE.singularValues()[1] < 0.66)) return -1; // E is no essential matrix

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
 * InputOutputArray _mask			Input  -> If provided, a mask marking invalid correspondences
 *											  is returned
 *
 * Return value:					true:		  Essential/Fundamental matrix is valid
 *									false:		  Essential/Fundamental matrix is invalid
 */
bool validateEssential(const cv::Mat p1, const cv::Mat p2, Eigen::Matrix3d E, bool EfullCheck, cv::InputOutputArray _mask)
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

	tryagain:
	if(EfullCheck)
	{
		Eigen::Matrix3d V;
		Eigen::JacobiSVD<Eigen::Matrix3d > svdE(E.transpose(), Eigen::ComputeFullV);

		if(svdE.singularValues()(0) / svdE.singularValues()(1) > 1.2)
			return false;
		if(!nearZero(0.01*svdE.singularValues()(2)/svdE.singularValues()(1)))
			return false;

		V = svdE.matrixV();
		e2 = V.col(2);
	}
	else
	{
		Eigen::MatrixXd ker = E.transpose().fullPivLu().kernel();
		if(ker.cols() != 1)
			return false;
		e2 = ker.col(0);
	}

	//Does this improve something or only need time?
	for(int i = 0; i < n; i++)
	{
		Eigen::Vector3d e2_line1, e2_line2;
		x1 << _p1.at<double>(i,0),
			  _p1.at<double>(i,1),
			  1.0;
		x2 << _p2.at<double>(i,0),
			  _p2.at<double>(i,1),
			  1.0;
		e2_line1 = e2.cross(x2);
		e2_line2 = E * x1;
		for(int j = 0; j < 3; j++)
		{
			if(nearZero(0.1*e2_line1(j)) || nearZero(0.1*e2_line2(j)))
				continue;
			if(e2_line1(j)*e2_line2(j) < 0)
			{
				if(cnt < 3)
					cnt++;
				else if((cnt == 3) && (i == cnt))
				{
					E = E * -1.0;
					for(int k = 0; k < cnt; k++)
						mask.at<bool>(k) = 1;
					cnt++;
					goto tryagain;
				}
				mask.at<bool>(i) = 0;
				break;
			}
		}
	}
	badPtsRatio = 1.0f - (float)cv::countNonZero(mask) / (float)n;
	if(badPtsRatio > 0.4f)
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
inline bool nearZero(double d)
{
    //Decide if determinants, etc. are too close to 0 to bother with
    const double EPSILON = 1e-3;
    return (d<EPSILON) && (d>-EPSILON);
}


/* Calculates statistical parameters for the given values in the vector. The following parameters
 * are calculated: median, arithmetic mean value, standard deviation and median absolute deviation (MAD).
 *
 * vector<double> vals		Input  -> Input vector from which the statistical parameters should be calculated
 * statVals* stats			Output -> Structure holding the statistical parameters
 * bool rejQuartiles		Input  -> If true, the lower and upper quartiles are rejected before calculating
 *									  the parameters
 *
 * Return value:		 none
 */
void getStatsfromVec(const std::vector<double> vals, statVals *stats, bool rejQuartiles)
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

	if(std::abs(hlp) < 1e-6)
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
double normFromVec(cv::Mat vec)
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
	unsigned int n = vec.size();
	double norm = 0;
	
	for(unsigned int i = 0; i < n; i++)
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
void getReprojErrors(cv::Mat Essential, cv::InputArray p1, cv::InputArray p2, bool takeImageCoords, statVals* qp, std::vector<double> *repErr, cv::InputArray K1, cv::InputArray K2, bool EisF)
{
	CV_Assert(!p1.empty() && !p2.empty());
	CV_Assert((takeImageCoords && !K1.empty() && !K2.empty()) || EisF);
	CV_Assert(!((qp == NULL) && (repErr == NULL)));

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

	if(repErr != NULL)
		computeReprojError2(x1, x2, FE, *repErr);
	
	if(qp != NULL)
	{
		if(repErr == NULL)
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
void computeReprojError1(cv::Mat X1, cv::Mat X2, cv::Mat E, std::vector<double> & error, double *error1)
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
void computeReprojError2(cv::Mat X1, cv::Mat X2, cv::Mat E, std::vector<double> & error, double *error1)
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
	const double radDegConv = 180 / PI;

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
	pitch *= radDegConv;
	roll *= radDegConv;
	yaw *= radDegConv;
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
cv::Mat getEfromRT(cv::Mat R, cv::Mat t)
{
	return getSkewSymMatFromVec(t/normFromVec(t)) * R;
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
};