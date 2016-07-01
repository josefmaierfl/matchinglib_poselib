/**********************************************************************************************************
 FILE: HomographyAlignment.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: July 2015

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functions for calculating the extrinsic camera parameters R & t based
			  on multiple homography alignment.
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "glob_includes.h"


/* --------------------- Function prototypes --------------------- */

//Estimation of R & t based on multi homography alignment.
int ComputeHomographyMotion(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
							std::vector<cv::Mat> Hs,
							std::vector<unsigned int> num_inl,
							cv::Mat & R1_2,
							cv::Mat & t1_2,
							double tol,
							cv::Mat & N,
							std::vector<cv::Mat> Hs_out);
//Diagonalization of the matrix W=(H^T)H with the unknown orthogonal matrix U (where H is a homography) to solve UWU^T=Diag(d1, d2, d3)
int jacobi33(double a[3][3], double d[3], double v[3][3], int *nrot);
//Estimation of R, t & the plane normal vector from a homography (two solutions)
int LonguetHigginsSolution( cv::InputArray H, std::vector<cv::Mat> & R1_2, std::vector<cv::Mat> & dt2in1, std::vector<cv::Mat> & norm);
//Estimation of R, t & the plane normal vector from a homography with initialization (one solution)
int LonguetHigginsSolution_with_initial( cv::InputArray H, cv::Mat & R2_1, cv::Mat & dt1, cv::Mat & norm1);
double Check_motion_error(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
						  std::vector<unsigned int> num_inl,
						  cv::Mat R1_2,
						  cv::Mat t);
//Estimation of R & t based on multi homography alignment without initialization.
int  HomographysAlignment(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
						  std::vector<unsigned int> num_inl,
						  cv::InputArray H,
						  cv::Mat & homo,
						  cv::Mat & R2_1,
						  cv::Mat & t1_2,
						  cv::Mat & norms,
						  double tol);
//Estimation of R & t based on multi homography alignment with initialization.
int HomographysAlignment_initial_rotation(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
						  std::vector<unsigned int> num_inl,
						  cv::InputArray H,
						  cv::Mat & homo,
						  cv::Mat & R2_1,
						  cv::Mat & t1_2,
						  cv::Mat & norms,
						  double tol);
int Rays_ClosestPoints(cv::Mat pt1, cv::Mat ray1, cv::Mat pt2, cv::Mat ray2, cv::Mat & p1, cv::Mat & p2);
void LinearTransformD(
	const double		*L,		/* The left matrix */
	const double		*R,		/* The right matrix */
	register double	*P,		/* The resultant matrix */
	long			nRows,	/* The number of rows of the left and resultant matrices */
	long			lCol,	/* The number of columns in the left matrix */
	long			rCol	/* The number of columns in the resultant matrix */
);
int update_h0_rt(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
				  std::vector<unsigned int> num_inl,
				  cv::Mat & homo, 
				  cv::Mat dn,
				  cv::Mat & rt);
int update_dn(cv::Mat points1, cv::Mat points2, int num_pts, cv::Mat h0, cv::Mat k0, cv::Mat & dn);
void AddMatrixD (double *A, double *B, double *result, long m, long n);
void ScaleMatrixD (double scale, double *from, double *to, long m, long n);
long InvertMatrixD (const double *M, double *Minv, long nRows, register long n);
long LUDecomposeD(
	register const double	*a,		/* the (n x n) coefficient matrix */
	register double		*lu, 	/* the (n x n) lu matrix augmented by an (n x 1) pivot sequence */
	register long			n		/* the order of the matrix */
);
void LUSolveD(
	register const double	*lu,	/* the decomposed LU matrix */
	register const double	*b,		/* the constant vector */
	register double			*x,		/* the solution vector */
	register long			n		/* the order of the equation */
);
double Check_error(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
				  std::vector<unsigned int> num_inl,
				  cv::Mat base_homo, 
				  cv::Mat dn,
				  cv::Mat rt);
int homographyTransfer33D(double h[3][3], double ip[2], double op[2]);
int convertToHomo(double ip[2], double op[3]);
int convertToImage(double ip[3], double op[2]);
int constructAnalyticHomography(cv::Mat rot, cv::Mat t, cv::Mat plane, cv::Mat & h);