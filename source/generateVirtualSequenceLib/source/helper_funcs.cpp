/**********************************************************************************************************
FILE: helper_funcs.cpp

PLATFORM: Windows 7, MS Visual Studio 2015, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: March 2018

LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides some helper functions.
**********************************************************************************************************/

#include "helper_funcs.h"
#include <chrono>
#include <time.h>

using namespace std;
using namespace cv;

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

/* -------------------------- Functions -------------------------- */

void randSeed(std::default_random_engine& rand_generator)
{
	// construct a trivial random generator engine from a time-based seed:
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	rand_generator = std::default_random_engine(seed);

	srand(time(NULL));
}

inline double getRandDoubleValRng(double lowerBound, double upperBound, std::default_random_engine rand_generator)
{
	std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
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

bool any_vec_cv(cv::Mat bin)
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

bool isfinite_vec_cv(cv::Mat bin)
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
cv::Mat getLineCam1(cv::Mat K, cv::Mat x)
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
void getLineCam2(cv::Mat R, cv::Mat t, cv::Mat K, cv::Mat x, cv::Mat& a, cv::Mat& b)
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
double getLineIntersect(cv::Mat b1, cv::Mat a2, cv::Mat b2)
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

bool solveLinEqu(cv::Mat& A, cv::Mat& b, cv::Mat& x)
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