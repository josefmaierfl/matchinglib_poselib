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
 FILE: HomographyAlignment.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: July 2015

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functions for calculating the extrinsic camera parameters R & t based
			  on multiple homography alignment.
**********************************************************************************************************/

#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "poselib/glob_includes.h"


/* --------------------- Function prototypes --------------------- */

//Estimation of R & t based on multi homography alignment.
int ComputeHomographyMotion(const std::vector<std::pair<cv::Mat,cv::Mat>>& inl_points,
							std::vector<cv::Mat> Hs,
							const std::vector<unsigned int>& num_inl,
							cv::Mat & R1_2,
							cv::Mat & t1_2,
							double tol,
							cv::Mat & N,
							std::vector<cv::Mat> Hs_out);
//Diagonalization of matrix W=(H^T)H with the unknown orthogonal matrix U (where H is a homography) to solve UWU^T=Diag(d1, d2, d3)
int jacobi_mat_33(double a[3][3], double *d, double v[3][3], int *nrot);
//Estimation of R, t & the plane normal vector from a homography (two solutions)
int Longuet_Higgins_Solution(cv::InputArray H, std::vector<cv::Mat> & R1_2, std::vector<cv::Mat> & dt2in1, std::vector<cv::Mat> & norm);
//Estimation of R, t & the plane normal vector from a homography with initialization (one solution)
int Longuet_Higgins_Solution_with_initial(cv::InputArray H, cv::Mat & R2_1, cv::Mat & dt1, cv::Mat & norm1);
double Check_motion_error(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
						  std::vector<unsigned int> num_inl,
						  const cv::Mat& R1_2,
						  const cv::Mat& t);
//Estimation of R & t based on multiple homography alignment without initialization.
int  Homographys_Alignment(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
                           std::vector<unsigned int> num_inl,
                           cv::InputArray H,
                           cv::Mat & homo,
                           cv::Mat & R2_1,
                           cv::Mat & t1_2,
                           cv::Mat & norms,
                           double tol);
//Estimation of R & t based on multiple homography alignment with initialization.
int Homographys_Alignment_initial_rotation(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
                                           std::vector<unsigned int> num_inl,
                                           cv::InputArray H,
                                           cv::Mat & homo,
                                           cv::Mat & R2_1,
                                           cv::Mat & t1_2,
                                           cv::Mat & norms,
                                           double tol);
int Rays_Closest_Points(const cv::Mat& pt1, const cv::Mat& ray1, const cv::Mat& pt2, const cv::Mat& ray2, cv::Mat & p1, cv::Mat & p2);
void Linear_Transform_D(
	const double *L,	//The left mat
	const double *R,	//The right mat
	double	*P,	//The result mat
	int nRows, //Number of rows of the left and result matrices
	int lCol, //Number of columns in the left matrix
	int rCol); //The number of columns in the result matrix

int update_h0_rt(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
				  std::vector<unsigned int> num_inl,
				  cv::Mat & homo,
				  const cv::Mat& dn,
				  cv::Mat & rt);
int update_dn(cv::Mat points1, cv::Mat points2, int num_pts, const cv::Mat& h0, const cv::Mat& k0, cv::Mat & dn);
void add_matrix_D (double *A, double *B, double *result, int m, int n);
void scale_matrix_D (double scale, double *from, double *to, int m, int n);
size_t invert_matrix_D (const double *M, double *Minv, int nRows, int n);
size_t LU_decompose_D(
	const double *a,	//n x n coefficient matrix
	double *lu, //n x n LU matrix augmented by an n x 1 pivot sequence
	int n); //Order of the matrix
void LU_solve_D(
	const double *lu, //decomposed LU matrix
	const double *b, //constant vector
	double *x, //solution vector
	int n); //order of the equation
double check_error(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
                   std::vector<unsigned int> num_inl,
                   const cv::Mat& base_homo,
                   cv::Mat dn,
                   cv::Mat rt);
int homography_transfer_33D(double h[3][3], double *ip, double *op);
int convertToHomography(const double *ip, double *op);
int convertToImage(const double ip[3], double op[2]);
int construct_analytic_homography(const cv::Mat& rot, cv::Mat t, cv::Mat plane, cv::Mat & h);
