/**********************************************************************************************************
FILE: helper_funcs.cpp

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: March 2018

LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides some helper functions.
**********************************************************************************************************/

#include "helper_funcs.h"
#include <chrono>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

/* -------------------------- Functions -------------------------- */

long int randSeed(std::default_random_engine& rand_generator)
{
	// construct a trivial random generator engine from a time-based seed:
	long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	rand_generator = std::default_random_engine(seed);

	srand((unsigned int)time(nullptr));

    return seed;
}

void randSeed(std::mt19937& rand_generator, long int seed){
	rand_generator = std::mt19937(seed);
}

double getRandDoubleValRng(double lowerBound, double upperBound, std::default_random_engine rand_generator)
{
	std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
	/*if (rand_generator == NULL)
	{
		rand_generator = &(std::default_random_engine((unsigned int)std::rand()));
	}*/
	return distribution(rand_generator);
}

cv::Mat eulerAnglesToRotationMatrix(double x, double y, double z)
{
	//Calculate rotation about x axis
	Mat R_x = (Mat_<double>(3, 3) << 1.0, 0, 0,
		0, cos(x), -sin(x),
		0, sin(x), cos(x));

	//Calculate rotation about y axis
	Mat R_y = (Mat_<double>(3, 3) << cos(y), 0, sin(y),
		0, 1, 0,
		-sin(y), 0, cos(y));

	//Calculate rotation about z axis
	Mat R_z = (Mat_<double>(3, 3) << cos(z), -sin(z), 0,
		sin(z), cos(z), 0,
		0, 0, 1);


	//Combined rotation matrix
	//return R_z * R_y * R_x;
	return R_y * R_z * R_x;
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
    const double radDegConv = 180.0 / M_PI;

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
        yaw = M_PI_2;
        roll = 0;
    }
    else if (m.at<double>(1,0) < -0.998) { // singularity at south pole
        pitch = std::atan2(m.at<double>(0,2),m.at<double>(2,2));
        yaw = -M_PI_2;
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
    }
}

bool any_vec_cv(const cv::Mat& bin)
{
	CV_Assert(((bin.rows == 1) || (bin.cols == 1)) && (bin.type() == CV_8UC1));
	int ln = bin.rows > bin.cols ? bin.rows : bin.cols;
	for (int i = 0; i < ln; i++)
	{
		if (bin.at<bool>(i))
		{
			return true;
		}
	}
	return false;
}

bool isfinite_vec_cv(const cv::Mat& bin)
{
	CV_Assert(((bin.rows == 1) || (bin.cols == 1)) && (bin.type() == CV_64FC1));
	int ln = bin.rows > bin.cols ? bin.rows : bin.cols;
	for (int i = 0; i < ln; i++)
	{
		if (!isfinite(bin.at<double>(i)))
			return false;
	}
	return true;
}

/*Calculates a line represented by a join of a point 'a' and a direction
* 'b'. As the camera centre of camera 1 is identical to the origin, 'a'
* corresponds to the origin and it is not returned.
*		K is the camera matrix of camera 1
*		x is a homogeneous point within the image in pixel coordinates
*/
cv::Mat getLineCam1(const cv::Mat& K, const cv::Mat& x)
{
	return K.inv() * x;
}

/* Calculates a line represented by a join of a point 'a' and a direction
* 'b'.
*    R corresponds to the rotation matrix from cam1 to cam2
*    t corresponds to the translation vector such that x2 = R*x1 + t. If the
*    camera 2 corresponds to the right or top camera, the largest component
*    within t must be negative.
*    K is the camera matrix of camera 2
*  x is a homogeneous point within the image in pixel coordinates
*/
void getLineCam2(const cv::Mat& R, const cv::Mat& t, const cv::Mat& K, const cv::Mat& x, cv::Mat& a, cv::Mat& b)
{
	a = -1.0 * R.t() * t;
	b = R.t() * K.inv() * x;
}

/* Calculate the z - distance of the intersection of 2 3D lines or the mean
* z - distance at the shortest perpendicular between 2 skew lines in 3D.
* The lines are represented by the join of 2 points 'a' and direction 'b'.
* For the first line no 'a1' must be provided as it is identical to the
* origin of the coordinate system which is also the camera centre of the
* left / bottom camera.
*/
double getLineIntersect(const cv::Mat& b1, const cv::Mat& a2, const cv::Mat& b2)
{
	//First, check if the 2 lines are linear dependent
	//Check if line 2 contains the origin
	double s = -a2.at<double>(0) / b2.at<double>(0);
	Mat x0 = a2 + s*b2;
	if (nearZero(sum(x0)[0]))
	{
		s = b1.at<double>(0) / b2.at<double>(0);
		x0 = s*b2 - b1;
		if (nearZero(sum(x0)[0]))
			return 0;
	}

	//Check, if they intersect
	double ld = a2.dot(b1.cross(b2));
	if (nearZero(ld))
	{
		Mat A, x;
		hconcat(b1, -b2, A);
		if (!cv::solve(A, a2, x, DECOMP_NORMAL))
			return 0;
		Mat S = x.at<double>(0)*b1;
		return S.at<double>(2);
	}

	//Check, if they are parallel
	s = b1.at<double>(0) / b2.at<double>(0);
	x0 = s * b2 - b1;
	if (nearZero(100.0 * sum(x0)[0]))
	{
		return 1000000.0;
	}

	//Get mean z - distance to the perpendicular between the 2 skew lines
	Mat A, x;
	hconcat(b1, -b2, A);
	hconcat(A, b1.cross(b2), A);
	if (!solveLinEqu(A, a2, x))
		return 0;
	Mat S1 = x.at<double>(0) * b1;
	Mat S2 = a2 + x.at<double>(1) * b2;
	return (S1.at<double>(2) + S2.at<double>(2)) / 2.0;
}

/*
 * Calculates the angle between 2 directional vectors b0 = (a, b, c) and b1 of skew lines
 */
double getAngleSkewLines(const cv::Mat& b0, const cv::Mat& b1, bool useDegrees){
    double rad = b0.dot(b1) / (cv::norm(b0) * cv::norm(b1));
    if (rad > 1.){
        return 0;
    }else if(rad < -1.){
        if(useDegrees){
            return 180.;
        }
        return M_PI;
    }
    rad = std::acos(rad);
    if(useDegrees){
        return 180. * rad / M_PI;
    }
}

/*
 * Calculates the angle between viewing rays of 2 cameras using principal points p0 and p1, camera matrices K0 and K1,
 * and relative pose R, t
 */
double getViewAngleRelativeCams(const cv::Mat& R, const cv::Mat& t,
                                const cv::Mat& K1, bool useDegrees){
    CV_Assert((K1.type() == CV_64FC1) && (K1.rows == 3) && (K1.cols == 3));
    CV_Assert((t.type() == CV_64FC1) && (t.rows == 3) && (t.cols == 1));
    cv::Mat p1 = (cv::Mat_<double>(3, 1) << K1.at<double>(0, 2), K1.at<double>(1, 2), 1.);
    cv::Mat b0 = (cv::Mat_<double>(3, 1) << 0, 0, 1.);
    cv::Mat a1, b1;
    getLineCam2(R, t, K1, p1, a1, b1);
    return getAngleSkewLines(b0, b1, useDegrees);
}

/*
 * Calculates a relative pose [R, t] between cameras given two absolute poses [R0, t0] and [R1, t1] in the same world frame
 */
void getRelativeFromAbsPoses(const cv::Mat& R0, const cv::Mat& t0, const cv::Mat& R1, const cv::Mat& t1,
                             cv::Mat& R, cv::Mat& t){
    R = R1.t() * R0;
    t = R1.t() * (t0 - t1);
}

/*
 * Calculates a relative pose [R, t] between cameras given two absolute poses [R0, t0] and [R1, t1] in the same world frame
 */
void getRelativeFromAbsPoses(const Eigen::Matrix3d& R0, const Eigen::Vector3d& t0, const Eigen::Matrix3d& R1, const Eigen::Vector3d& t1,
                             Eigen::Matrix3d& R, Eigen::Vector3d& t){
    R = R1.transpose() * R0;
    t = R1.transpose() * (t0 - t1);
}

/*
 * Calculates the angle between viewing rays of 2 cameras using principal points p0 and p1, camera matrices K0 and K1,
 * and absolute camera poses [R0, t0] and [R1, t1]
 */
double getViewAngleAbsoluteCams(const cv::Mat& R0, const cv::Mat& t0,
                                const cv::Mat& K1, const cv::Mat& R1, const cv::Mat& t1, bool useDegrees){
    cv::Mat R, t;
    getRelativeFromAbsPoses(R0, t0, R1, t1, R, t);
    return getViewAngleRelativeCams(R, t, K1, useDegrees);
}

/*
 * Calculates the angle between viewing rays of 2 cameras using principal points p0 and p1, camera matrices K0 and K1,
 * and absolute camera poses [R0, t0] and [R1, t1]
 */
double getViewAngleAbsoluteCams(const Eigen::Matrix3d& R0, const Eigen::Vector3d& t0,
                                const Eigen::Matrix3d& K1, const Eigen::Matrix3d& R1, const Eigen::Vector3d& t1, bool useDegrees){
    cv::Mat R, t, K1cv, R0cv, R1cv, t0cv, t1cv;
    cv::eigen2cv(K1, K1cv);
    cv::eigen2cv(R0, R0cv);
    cv::eigen2cv(R1, R1cv);
    cv::eigen2cv(t0, t0cv);
    cv::eigen2cv(t1, t1cv);
    getRelativeFromAbsPoses(R0cv, t0cv, R1cv, t1cv, R, t);
    return getViewAngleRelativeCams(R, t, K1cv, useDegrees);
}

bool solveLinEqu(const cv::Mat& A, const cv::Mat& b, cv::Mat& x)
{
	if (!cv::solve(A, b, x, DECOMP_LU))
	{
		if (!cv::solve(A, b, x, DECOMP_SVD))
		{
			if (!cv::solve(A, b, x, DECOMP_QR))
			{
				if (!cv::solve(A, b, x, DECOMP_NORMAL))//DECOMP_NORMAL | DECOMP_QR
				{
					return false;
				}
			}
		}
	}
	return true;
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
	}
	else {
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

		fRoot = sqrt(rot(i, i) - rot(j, j) - rot(k, k) + (double) 1.0);
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

/* Checks if a 3x3 matrix is a rotation matrix
*
* cv::Mat R				Input  -> Rotation matrix
*
* Return:				true or false
*/
bool isMatRotationMat(const cv::Mat& R)
{
	CV_Assert(!R.empty());

	Eigen::Matrix3d Re;
	cv::cv2eigen(R, Re);

	return isMatRotationMat(Re);
}

/* Checks if a 3x3 matrix is a rotation matrix
*
* Matrix3d R			Input  -> Rotation matrix
*
* Return:				true or false
*/
bool isMatRotationMat(Eigen::Matrix3d R)
{
	//Check if R is a rotation matrix
	Eigen::Matrix3d R_check = (R.transpose() * R) - Eigen::Matrix3d::Identity();
	double r_det = R.determinant() - 1.0;

	return R_check.isZero(1e-3) && nearZero(r_det);
}

/* Calculates the difference (rotation angle) between two rotation matrices.
*
* Mat R1	Input  -> First rotation matrix
* Mat R2	Input  -> Second rotation matrix
*
* Return value:			Rotation angle (from Angle-axis-representation) between the two rotations
*/
double rotDiff(const cv::Mat& R1, const cv::Mat& R2)
{
	Eigen::Matrix3d R1e, R2e;
	Eigen::Vector4d q1, q2;
	cv::cv2eigen(R1, R1e);
	cv::cv2eigen(R2, R2e);

	MatToQuat(R1e, q1);
	MatToQuat(R2e, q2);

	return rotDiff(q1, q2);
}

/* Calculates the difference (rotation angle) between two rotation matrices.
*
* Eigen::Matrix3d R1	Input  -> First rotation matrix
* Eigen::Matrix3d R2	Input  -> Second rotation matrix
*
* Return value:			Rotation angle (from Angle-axis-representation) between the two rotations
*/
double rotDiff(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2)
{
	Eigen::Vector4d q1, q2;

	MatToQuat(R1, q1);
	MatToQuat(R2, q2);

	return rotDiff(q1, q2);
}

/* Calculates the difference (rotation angle) between two camera projection matrices.
*
* Eigen::Matrix4d R1	Input  -> First projection matrix
* Eigen::Matrix4d R2	Input  -> Second projection matrix
*
* Return value:			Rotation angle (from Angle-axis-representation) between the two rotations
*/
double rotDiff(const Eigen::Matrix4d& R1, const Eigen::Matrix4d& R2)
{
	Eigen::Vector4d q1, q2;
	Eigen::Matrix3d R1e, R2e;
	R1e = R1.block<3, 3>(0, 0);
	R2e = R2.block<3, 3>(0, 0);

	MatToQuat(R1e, q1);
	MatToQuat(R2e, q2);

	return rotDiff(q1, q2);
}

/* Calculates the difference (rotation angle) between two camera projection matrices.
*
* Eigen::Matrix4f R1	Input  -> First projection matrix
* Eigen::Matrix4f R2	Input  -> Second projection matrix
*
* Return value:			Rotation angle (from Angle-axis-representation) between the two rotations
*/
double rotDiff(const Eigen::Matrix4f& R1, const Eigen::Matrix4f& R2)
{
	Eigen::Vector4d q1, q2;
	Eigen::Matrix4d R1e, R2e;

	R1e = R1.cast<double>();
	R2e = R2.cast<double>();

	return rotDiff(R1e, R2e);
}

/* Calculates the difference (rotation angle) between two rotation quaternions.
*
* Eigen::Vector4d R1	Input  -> First rotation quaternion
* Eigen::Vector4d R2	Input  -> Second rotation quaternion
*
* Return value:			Rotation angle (from Angle-axis-representation) between the two rotations
*/
double rotDiff(Eigen::Vector4d & R1, Eigen::Vector4d & R2)
{
	Eigen::Vector4d Rdiff1;
	quatMultConj(R1, R2, Rdiff1);
	quatNormalise(Rdiff1);
	return quatAngle(Rdiff1);
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
	if (ang < 0)
		cout << "acos returning val less than 0" << endl;
	//if(isnan(ang))
	//	cout << "acos returning nan" << endl;
	if (ang > M_PI) ang -= 2 * M_PI;
	return ang;
}

/* Rounds every entry of the matrix to the nearest integer
*
* Mat m				Input -> A floating point matrix
*
* Return value:		Rounded floating point matrix
*/
cv::Mat roundMat(const cv::Mat& m)
{
	Mat tmp, tmp1;
	m.convertTo(tmp, CV_32S, 1.0, 0.5);
	tmp.convertTo(tmp1, m.type());

	return tmp1;
}

//Checks if the given matrix is an identity matrix
bool isIdentityMat(const cv::Mat& m){
    Mat diff = m - Mat::eye(m.size(), m.type());
    double sum = cv::sum(diff)[0];
    return nearZero(sum);
}

//Calculate the descriptor distance between 2 descriptors
double getDescriptorDistance(const cv::Mat &descriptor1, const cv::Mat &descriptor2){
    if(descriptor1.type() == CV_8U){
        return norm(descriptor1, descriptor2, NORM_HAMMING);
    }

    return norm(descriptor1, descriptor2, NORM_L2);
}

FileStorage& operator << (FileStorage& fs, bool &value)
{
    if(value){
        return (fs << 1);
    }

    return (fs << 0);
}

void operator >> (const FileNode& n, bool& value)
{
    int bVal;
    n >> bVal;
    if(bVal){
        value = true;
    }else{
        value = false;
    }
}

FileStorage& operator << (FileStorage& fs, int64_t &value)
{
    string strVal = std::to_string(value);
    return (fs << strVal);
}

void operator >> (const FileNode& n, int64_t& value)
{
    string strVal;
    n >> strVal;
    value = std::stoll(strVal);
}

FileNodeIterator& operator >> (FileNodeIterator& it, int64_t & value)
{
    *it >> value;
    return ++it;
}
