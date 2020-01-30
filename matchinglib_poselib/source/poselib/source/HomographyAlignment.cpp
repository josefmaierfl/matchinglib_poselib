/**********************************************************************************************************
 FILE: RectStructMot.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: July 2015

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functions for calculating the extrinsic camera parameters R & t based
			  on multiple homography alignment.
**********************************************************************************************************/

#include "HomographyAlignment.h"
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include "poselib/pose_helper.h"

using namespace std;
using namespace cv;

/* --------------------- Functions --------------------- */

/* Estimation of R & t based on multi homography alignment.
 *
 * vector<pair<Mat,Mat>> inl_points		Input  -> Vector containing the correspondences of the different planes in the camera
 *												  coordinate system. Each vector element contains the correspondeces of one
 *												  plane where the first points are coordinates of the left camera and the second
 *												  points are coordinates of the right camera. The first vector element must contain
 *												  the largest correspondence set for the dominatest plane in the images.
 * vector<Mat> Hs						Input  -> Homographies of the planes in the camera coordinate system. The vector-ordering
 *												  of the homographys must be in the same order than their correspondences in
 *												  inl_points.
 * vector<unsigned int> num_inl			Input  -> Number of inliers (correspondences) for each plane. The vector-ordering
 *												  must be in the same order than Hs.
 * Mat R1_2								Input & Output -> If a rotation matrix is provided, the homography alignment is initialized
 *														  with this rotation and with t1_2 (in this case both, R1_2 & t1_2 have to
 *														  be provided. Be careful to use the right rotation matrix and not its
 *														  inverse R1_2.t(). If no initialization should be performed R1_2 must be empty.
 *														  The resulting rotation matrix after homography alignment is stored in R1_2.
 * Mat t1_2								Input & Output -> If a translation vector is provided, the homography alignment is initialized
 *														  with this translation and with R1_2 (in this case both, R1_2 & t1_2 have to
 *														  be provided. Be careful to use the right rotation matrix and not its
 *														  inverse -t1_2. If no initialization should be performed t1_2 must be empty.
 *														  The resulting translation vector after homography alignment is stored in t1_2.
 * double tol							Input  -> Inlier threshold in the camera coordinate system.
 * Mat N								Output -> The resulting plane normal vectors after refinement in the same order than Hs.
 * vector<Mat> Hs_out					Output -> The refined homography matrices in the same order than Hs.
 *
 * Return value:						1 :		Everything ok
 *										0 :		Homography alignment not possible/failed
 */
int ComputeHomographyMotion(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
							std::vector<cv::Mat> Hs,
							std::vector<unsigned int> num_inl,
							cv::Mat & R1_2,
							cv::Mat & t1_2,
							double tol,
							cv::Mat & N,
							std::vector<cv::Mat> Hs_out)
{
	Mat norms, homo, rot2_1;
	vector<Mat> rot_b2, dt_b2, norm2, inv_rot_b2;
	int num_planes;

	int i;
	double error[2];

	num_planes = (int)num_inl.size();

	norms = Mat(3,num_planes,CV_64F);

	if(num_planes > 1)
	{
		//rot is rotation from frame 1 to frame 2
		//PRT_INT("---- nr of planes", num_planes)
		if(!t1_2.empty() && cv::norm(t1_2) > 0.0)
		{
			if(!R1_2.empty())
				rot2_1 = R1_2.t();
			HomographysAlignment_initial_rotation(inl_points,num_inl,Hs[0],homo,rot2_1,t1_2,norms,tol);
		}
		else
		{
			HomographysAlignment(inl_points,num_inl,Hs[0],homo,rot2_1,t1_2,norms,tol);
		}
		R1_2 = rot2_1.t();
		norms.col(0).copyTo(N);
		for(i = 0; i < num_planes; ++i)
		{
			constructAnalyticHomography(rot2_1, t1_2, norms.col(i), homo);
			Hs_out.push_back(homo.clone());
		}
		//PRT_TIME("---- constructAnalyticHomography", start)
	}
	else if(num_planes == 1)
	{
		if(t1_2.empty() || cv::norm(t1_2) < 1e-6)
			return 0;
		//PRT_MSG("---- nr of planes: 1")

		//PRT_TIME("---- getHomographyFromPoints", start)
		LonguetHigginsSolution(Hs[0], rot_b2, dt_b2, norm2);
		//PRT_TIME("---- LonguetHigginsSolution", start)
		inv_rot_b2.push_back(rot_b2[0].t());
		inv_rot_b2.push_back(rot_b2[1].t());
		error[0] = 0.0;
		error[1] = 0.0;

		error[0] = t1_2.dot(dt_b2[0]);
		error[1] = t1_2.dot(dt_b2[1]);
		if(error[0] > error[1])
		{
			R1_2 = rot_b2[0].t();
			dt_b2[0].copyTo(t1_2);
			norm2[0].copyTo(N);
			//xyzrot(rot, &er[0], &er[1], &er[2]);
			//this motion can be used in bundle adjustment input
			//printf("%f %f %f %f %f %f\n", dt_b2[0][0], dt_b2[0][1],  dt_b2[0][2], er[0], er[1], er[2]);
		}
		else
		{
			R1_2 = rot_b2[1].t();
			dt_b2[1].copyTo(t1_2);
			norm2[1].copyTo(N);
			//xyzrot(rot, &er[0], &er[1], &er[2]);
			//printf("%f %f %f %f %f %f\n", dt_b2[1][0], dt_b2[1][1],  dt_b2[1][2], er[0], er[1], er[2]);
		}
	}
	else
	{
		//PRT_MSG("there is no plane in this data set")
		return 0;
	}

//#define  CHECK_REPROJECTION_ERROR
#ifdef CHECK_REPROJECTION_ERROR
  double *h_a = (double*)Hs_out[0].data;
  Mat test_pts2, p_diff;
  for(j = 0; j < (int)num_inl[0]; ++j)
  {
    double op[2];
	test_pts2 = Mat(2,1,CV_64F,op);
	double *test_pts = (double*)inl_points[0].first.row(j).data;
	homographyTransfer33D(h_a, test_pts, op);
	p_diff = test_pts2 - inl_points[0].first.row(j).t();
	cout << "reprojection error " << p_diff.at<double>(0) << ", " << p_diff.at<double>(1) << endl;
  }
#endif

  //PRT_MAT33F("rotation", rot2_1)
  //PRT_VEC3F("translation", t1_2)

  return 1;//*numm;
}

/* Estimation of R & t based on multi homography alignment without initialization.
 *
 * vector<pair<Mat,Mat>> inl_points		Input  -> Vector containing the correspondences of the different planes in the camera
 *												  coordinate system. Each vector element contains the correspondeces of one
 *												  plane where the first points are coordinates of the left camera and the second
 *												  points are coordinates of the right camera. The first vector element must contain
 *												  the largest correspondence set for the dominatest plane in the images.
 * vector<unsigned int> num_inl			Input  -> Number of inliers (correspondences) for each plane. The vector-ordering
 *												  must be in the same order than inl_points.
 * InputArray H							Input  -> Homography corresponding to the first entry (correspondences) of inl_points. This
 *												  homography should have the largest support set of correspondences.
 * Mat homo								Output -> The refined input homography H after homography alignment.
 * Mat R2_1								Output -> The resulting rotation matrix from camera 1 to camera 2.
 * Mat t1_2								Output -> The resulting translation vector from camera 2 to camera 1.
 * Mat & norms							Output -> The resulting plane normal vectors after refinement for all planes.
 * double tol							Input  -> Inlier threshold in the camera coordinate system.
 *
 * Return value:						1 :		Everything ok
 *										0 :		Homography alignment not possible/failed
 */
#ifndef MAX_ITERATION
#define MAX_ITERATION 40
//estimated rotation is from camera 1 to camera 2
//estimate translation if the camera 2 in camera 1 frame
int  HomographysAlignment(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
						  std::vector<unsigned int> num_inl,
						  cv::InputArray H,
						  cv::Mat & homo,
						  cv::Mat & R2_1,
						  cv::Mat & t1_2,
						  cv::Mat & norms,
						  double tol)
{

	Mat _H = H.getMat();
	std::vector<Mat> rot_b2, dt_b2, norm2, base_rt2, rot2, homo2, t2/*, rt2*/;
	Mat dn, dn2, h0, rt0, rot_b, dt_b, norm_b, _hh;
	rot2.push_back(Mat(3,3,CV_64F));
	rot2.push_back(Mat(3,3,CV_64F));
	homo2.push_back(Mat(3,3,CV_64F));
	homo2.push_back(Mat(3,3,CV_64F));
	t2.push_back(Mat(3,1,CV_64F));
	t2.push_back(Mat(3,1,CV_64F));
	base_rt2.push_back(Mat(3,1,CV_64F));
	base_rt2.push_back(Mat(3,1,CV_64F));
	int i ;
	int iter;
	double hh[3][3];
	double e1, e2;
	int nump;
	int q;
	int iter1;

	double errors[2];
	double s1;

	int num_patches;
	num_patches = (int)num_inl.size();
	_hh = Mat(3,3,CV_64F,hh);

	nump = 0;
	for(i = 0; i < num_patches; ++i)
	{
		nump += (int)num_inl[i];
	}

	dn = Mat(num_patches,3,CV_64F);
	dn2 = Mat(2*num_patches,3,CV_64F);

	_H.copyTo(h0);

	LonguetHigginsSolution(_H, rot_b2, dt_b2, norm2);
	errors[0] = Check_motion_error(inl_points, num_inl, rot_b2[0], dt_b2[0]);
	errors[1] = Check_motion_error(inl_points, num_inl, rot_b2[1], dt_b2[1]);
	//printf("error %f dt %f %f %f\n", errors[0], dt_b2[0][0], dt_b2[0][1], dt_b2[0][2]);
	//printf("error %f dt %f %f %f\n", errors[1], dt_b2[1][0], dt_b2[1][1], dt_b2[1][2]);

	dn.row(0) = Mat::zeros(1,3,CV_64F);

	base_rt2[0] = rot_b2[0] * dt_b2[0];
	base_rt2[1] = rot_b2[1] * dt_b2[1];

	//q is the index of which solution is used for HA
	for(q = 0; q < 2; ++q)
	{
		_H.copyTo(h0);
		base_rt2[q].copyTo(rt0);
		rot_b2[q].copyTo(rot_b);
		dt_b2[q].copyTo(dt_b);
		e2 = 100000.0;
		for(iter1 = 0; iter1 < 4; ++iter1)
		{
			//this is to update the h0 and dn
			for(iter = 0 ; iter < MAX_ITERATION; ++iter)
			{
				dn.row(0) = Mat::zeros(1,3,CV_64F);//first update the dn here ---------> warum wird immer der erste 0 gesetzt -> ist das nicht ein Fehler -> nein (weil in der folgenden for-Schleife nie der erste ein update bekommt (die anderen werden dort hin getrimmt)

				for(i = 1; i < num_patches; ++i)
				{
					cv::Mat tmp = dn.row(i);
					update_dn(inl_points[i].first, inl_points[i].second, (int)num_inl[i], h0, rt0, tmp);
				}

				update_h0_rt(inl_points,num_inl,h0,dn,rt0);
				e1 = Check_error(inl_points, num_inl, h0, dn, rt0);
				//printf("q = %d iter %d e1 = %f e2 = %f\n", q, iter, e1, e2);

				if((fabs(e1-e2) < 0.00001 || e1 < tol)&& iter > 2)
				{
					break;
				}
				else
				{
					e2 = e1;
				}
				//update k
				//compute the total error
			}

			LonguetHigginsSolution_with_initial(h0, rot_b, dt_b, norm_b);
			rt0 = rot_b * dt_b;

			e1 = Check_error(inl_points, num_inl, h0, dn, rt0);

			//printf("iter %d e1 = %f e2 = %f\n", iter, e1, e2);
			if((fabs(e1-e2) < 0.000001 || e1 < tol)&& iter > 2)
			{
				break;
			}
			else
			{
				e2 = e1;
			}
			//update k
			//compute the total error
		}

		rot_b.copyTo(rot2[q]);
		norm_b.copyTo(norm2[q]);
		h0.copyTo(homo2[q]);
		dt_b.copyTo(t2[q]);

		errors[q] = e1;
		for(i = 0; i < num_patches; ++i)
		{
			dn.row(i).copyTo(dn2.row(q*num_patches + i));//keep the old dn for the future uses
		}
	}  //end of q loop
	//printf("homo 1\n");
	//prt33(homo2[0]);
	//printf("homo 2\n");
	//prt33(homo2[1]);

	//LonguetHigginsSolution(h0, rot_b2, dt_b2, norm2);
	errors[0] = Check_motion_error(inl_points, num_inl, rot2[0], t2[0]);
	errors[1] = Check_motion_error(inl_points, num_inl, rot2[1], t2[1]);
	//printf("error %f dt %f %f %f\n", errors[0], t2[0][0], t2[0][1],  t2[0][2]);
	//printf("error %f dt %f %f %f\n", errors[1], t2[1][0], t2[1][1], t2[1][2]);

	if(errors[0] < errors[1])
	{
		norm2[0].copyTo(norms.col(0));
		rot2[0].copyTo(R2_1);
		homo2[0].copyTo(homo);
		t2[0].copyTo(t1_2);
	}
	else
	{
		norm2[1].copyTo(norms.col(0));
		rot2[1].copyTo(R2_1);
		homo2[1].copyTo(homo);
		t2[1].copyTo(t1_2);
	}
	dn2.rowRange(0, num_patches).copyTo(dn);

	//compute the scale factor of the first homography and mulple it to rest dns
	for(int j = 0; j < 3; j++)
	{
		for(int j1 = 0; j1 < 3; j1++)
		{
			hh[j][j1] = t1_2.at<double>(j) * norms.at<double>(j1,0);
		}
	}
	h0 = Mat::eye(3,3,CV_64F);
	_hh = h0 - _hh;
	h0 = R2_1 * _hh;
	s1 = h0.at<double>(2,2);

	// update the dn and surface normal vectors

	for(i = 1; i < num_patches; ++i)
	{
		dn.row(i) = s1 * dn.row(i);
		norms.col(i) = norms.col(0) - dn.row(i).t();
	}

	return 1;
}
#undef MAX_ITERATION
#endif


/* Estimation of R & t based on multi homography alignment with initialization.
 *
 * vector<pair<Mat,Mat>> inl_points		Input  -> Vector containing the correspondences of the different planes in the camera
 *												  coordinate system. Each vector element contains the correspondeces of one
 *												  plane where the first points are coordinates of the left camera and the second
 *												  points are coordinates of the right camera. The first vector element must contain
 *												  the largest correspondence set for the dominatest plane in the images.
 * vector<unsigned int> num_inl			Input  -> Number of inliers (correspondences) for each plane. The vector-ordering
 *												  must be in the same order than inl_points.
 * InputArray H							Input  -> Homography corresponding to the first entry (correspondences) of inl_points. This
 *												  homography should have the largest support set of correspondences.
 * Mat homo								Output -> The refined input homography H after homography alignment.
 * Mat R2_1								Input & Output -> The homography alignment is initialized with this rotation. The resulting
 *														  rotation matrix from camera 1 to camera 2 after homography alignment is
 *														  copied back to R2_1.
 * Mat t1_2								Output -> The resulting translation vector from camera 2 to camera 1.
 * Mat & norms							Output -> The resulting plane normal vectors after refinement for all planes.
 * double tol							Input  -> Inlier threshold in the camera coordinate system.
 *
 * Return value:						1 :		Everything ok
 *										0 :		Homography alignment not possible/failed
 */
#ifndef MAX_ITERATION
#define MAX_ITERATION 30

int HomographysAlignment_initial_rotation(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
						  std::vector<unsigned int> num_inl,
						  cv::InputArray H,
						  cv::Mat & homo,
						  cv::Mat & R2_1,
						  cv::Mat & t1_2,
						  cv::Mat & norms,
						  double tol)
{
	Mat h0, _H, R, dR, rot_b, dn, dn2, rt0, rt00, dt_b, norm_b, _hh;
	std::vector<Mat> rot_b2, dt_b2, norm2, base_rt2, rot2, homo2, t2;
	_H = H.getMat();
	base_rt2.push_back(Mat(3,1,CV_64F));
	base_rt2.push_back(Mat(3,1,CV_64F));
	rot2.push_back(Mat(3,3,CV_64F));
	rot2.push_back(Mat(3,3,CV_64F));
	homo2.push_back(Mat(3,3,CV_64F));
	homo2.push_back(Mat(3,3,CV_64F));
	t2.push_back(Mat(3,1,CV_64F));
	t2.push_back(Mat(3,1,CV_64F));
	int i ;
	int iter;
	double hh[3][3];
	double e1, e2;

	int nump;
	int q;
	int iter1;
	double errors[2];
	double s1;
	int num_patches;
	num_patches = (int)num_inl.size();
	_hh = Mat(3,3,CV_64F,hh);
	nump = 0;
	for(i = 0; i < num_patches; ++i)
	{
		nump += (int)num_inl[i];
	}

	dn = Mat(num_patches,3,CV_64F);
	dn2 = Mat(2*num_patches,3,CV_64F);

	_H.copyTo(h0);

	LonguetHigginsSolution(h0, rot_b2, dt_b2, norm2);
	R = R2_1.t();
	dR = R * rot_b2[0];
	Eigen::Matrix3d rot_e;
	Eigen::Vector4d quat_e;
	Eigen::Vector3d axis_e;
	cv2eigen(dR, rot_e);
	poselib::MatToQuat(rot_e, quat_e);
	poselib::QuatToAxisAngle(quat_e, axis_e, errors[0]);
	dR = R * rot_b2[1];
	cv2eigen(dR, rot_e);
	poselib::MatToQuat(rot_e, quat_e);
	poselib::QuatToAxisAngle(quat_e, axis_e, errors[1]);

	//	there are two solutions here
	//  pick up this solution, in which, the estimated surface faces more or less to the first camera
	//  because the camera 1 optical point is 0 0 1, therefore the larger abs(n[2]) one is the real solution

	if(errors[0] < errors[1])
	{
		base_rt2[0] = rot_b2[0] * dt_b2[0];
		base_rt2[1] = rot_b2[1] * dt_b2[1];
		rot_b2[0].copyTo(rot_b);
		dt_b2[0].copyTo(dt_b);
		norm2[0].copyTo(norm_b);
	}
	else
	{
		base_rt2[0] = rot_b2[1] * dt_b2[1];
		base_rt2[1] = rot_b2[0] * dt_b2[0];
		rot_b2[1].copyTo(rot_b);
		dt_b2[1].copyTo(dt_b);
		norm2[1].copyTo(norm_b);
	}
	dn.row(0) = Mat::zeros(1,3,CV_64F);

	//q is the index of which solution is used for HA
	for(q = 0; q < 1; ++q)
	{
		_H.copyTo(h0);
		base_rt2[q].copyTo(rt0);
		rt0.copyTo(rt00);
		e2 = 100000.0;
		for(iter1 = 0; iter1 < 4; ++iter1)
		{
			for(iter = 0 ; iter < MAX_ITERATION; ++iter)
			{
				dn.row(0) = Mat::zeros(1,3,CV_64F); //first update the dn here ---------> warum wird immer der erste 0 gesetzt -> ist das nicht ein Fehler -> nein (weil in der folgenden for-Schleife nie der erste ein update bekommt (die anderen werden dort hin getrimmt)

				for(i = 1; i < num_patches; ++i)
				{
					cv::Mat tmp = dn.row(i);
					update_dn(inl_points[i].first, inl_points[i].second, (int)num_inl[i], h0, rt0, tmp);
				}

				update_h0_rt(inl_points,num_inl,h0,dn,rt0);
				e1 = Check_error(inl_points, num_inl, h0, dn, rt0);
				//printf("q = %d iter %d e1 = %f e2 = %f\n", q, iter, e1, e2);

				if((fabs(e1-e2) < 0.00001 || e1 < tol)&& iter > 2)
				{
					break;
				}
				else
				{
					e2 = e1;
				}
				//update k
				//compute the total error
			}
			//update rt0
			LonguetHigginsSolution_with_initial(h0, rot_b, dt_b, norm_b);

			//this may no tbe the right solution
			rt0 = rot_b * dt_b;
			rot_b.copyTo(rot2[q]);
			norm_b.copyTo(norm2[q]);
			h0.copyTo(homo2[q]);
			dt_b.copyTo(t2[q]);

			e1 = Check_error(inl_points, num_inl, h0, dn, rt0);
			if((fabs(e1-e2) < 0.000001 || e1 < tol)&& iter > 2)
			{
				break;
			}
			else
			{
				e2 = e1;
			}
			//update k
			//compute the total error
		}
		errors[q] = e1;
		for(i = 0; i < num_patches; ++i)
		{
			dn.row(i).copyTo(dn2.row(q*num_patches + i));//keep the old dn for the future uses
		}
	}

	norm2[0].copyTo(norms.col(0));
	rot2[0].copyTo(R2_1);
	homo2[0].copyTo(homo);
	t2[0].copyTo(t1_2);

	dn2.rowRange(0, num_patches).copyTo(dn);

	//compute the scale factor of the first homography and mulple it to rest dns
	for(size_t j = 0; j < 3; j++)
	{
		for(size_t j1 = 0; j1 < 3; j1++)
		{
			hh[j][j1] = t1_2.at<double>(j) * norms.at<double>(j1,0);
		}
	}
	h0 = Mat::eye(3,3,CV_64F);
	_hh = h0 - _hh;
	h0 = R2_1 * _hh;
	s1 = h0.at<double>(2,2);

	// update the dn and surface normal vectors

	for(i = 1; i < num_patches; ++i)
	{
		dn.row(i) = s1 * dn.row(i);
		norms.col(i) = norms.col(0) - dn.row(i).t();
	}

	return 1;
}
#undef MAX_ITERATION
#endif

/* Estimation of R, t & the plane normal vector (two possible solutions) based on the algorithm from Longuet-Higgine "A computer
 * algorithm for reconstructing a scene from two projections", 1981. These parameters are estimted from a homography H and its
 * inliers (in the camera coordinate system).
 *
 * InputArray H							Input  -> Homography between images and a world plane in the camera coordinate system.
 * vector<Mat> R1_2						Output -> Two possible solutions for the rotation matrix.
 * vector<Mat> dt2in1					Output -> Two possible solutions for the translation vector.
 * vector<Mat> norm						Output -> Two possible solutions for the plane normal vector.
 *
 * Return value:						1 :		Everything ok
 *										0 :		Estimation not possible/failed
 */
int LonguetHigginsSolution( cv::InputArray H, std::vector<cv::Mat> & R1_2, std::vector<cv::Mat> & dt2in1, std::vector<cv::Mat> & norm)
{
    Mat _H = H.getMat();
	if(!R1_2.empty())
		R1_2.clear();

	{
		Mat tmp = Mat(3,3,CV_64F);
		R1_2.push_back(tmp.clone());
		R1_2.push_back(tmp.clone());
	}
	if(!dt2in1.empty())
		dt2in1.clear();

	{
		Mat tmp = Mat(3,1,CV_64F);
		dt2in1.push_back(tmp.clone());
		dt2in1.push_back(tmp.clone());
	}
	if(!norm.empty())
		norm.clear();

	{
		Mat tmp = Mat(3,1,CV_64F);
		norm.push_back(tmp.clone());
		norm.push_back(tmp.clone());
	}

	double hh[3][3], d[3], u[3][3], n[3], check, d2;
    double p[3], t[3], tn[3][3];
    double ts[3];
    double ps[3] ;
    double check1;
    int s1, s2, iter, k, i;

	Mat ht, ut, h_norm, x, r, _ts, _n, trans, plane, invtn, rot;
	Mat _tn = Mat(3,3,CV_64F,tn);
	Mat _hh = Mat(3,3,CV_64F,hh);
	Mat _u = Mat(3,3,CV_64F,u);
	ht = _H.t();
	_hh = ht * _H;
    //u here is the Ut on the right side
    jacobi33(hh, d, u, &iter);
	ut = _u.t();

	_hh = d[1] * _hh;
    d2 = 1.0/sqrt(d[1]);
	h_norm = d2 * _H;//the homo_norm is the homo transform with
    d[0] = d[0]/d[1];
    d[2] = d[2]/d[1];
    d[1] = 1.0;

    if(d[0] - d[1] < 0.000001 && d[1]- d[2] < 0.000001)
    {
        // the two images are too close each other
        //printf("the images are two close each other and the slope estimation is not valid\n");
        //printf("the rotation estimation is ok\n");
		dt2in1[0] = Mat::zeros(3,1,CV_64F);
		dt2in1[1] = Mat::zeros(3,1,CV_64F);
        //the homography degerate to a rotation
		h_norm.copyTo(R1_2[0]);
		h_norm.copyTo(R1_2[1]);

        //the surface normal cannot be estimated here
		norm[0] = Mat::zeros(3,1,CV_64F);
		norm[1] = Mat::zeros(3,1,CV_64F);
        return 1;
    }

    if(d[0] > 1.0 && d[2] <= 1.0)
    {
        t[0] = sqrt((d[0] - 1.0)*d[2]/(d[0] - d[2]));
        t[1] = 0.0;
        t[2] = sqrt((1.0 - d[2])*d[0]/(d[0] - d[2]));
        p[0] = sqrt((d[0] - 1.0)*d[0]/(d[0] - d[2]));
        p[1] = 0.0;
        p[2] = sqrt((1.0 - d[2])*d[2]/(d[0] - d[2]));
    }
    else
    {
        return 0;
    }
	x = Mat::zeros(3,1,CV_64F);
	x.at<double>(2) = 1.0;
    k = 0;
	r = _u * x;
	_ts = Mat(3,1,CV_64F,ts);
	_ts = Mat::zeros(3,1,CV_64F);
	_n = Mat(3,1,CV_64F,n);

    for(s1 =-1; s1 <=1; s1 +=2)
    {
        for(s2 =-1; s2 <=1; s2 +=2)
        {
		    ps[0] = -s1*p[0];
			ps[1] = 0.0;
			ps[2] = -s2*p[2];
		    n[0] = s1*t[0] + ps[0];
		    n[1] = 0.0;
		    n[2] = s2*t[2] + ps[2];
			ts[0] = s1*t[0];
			ts[1] = 0.0;
			ts[2] = s2*t[2];

			trans = _u * _ts;
			plane = _u * _n;
			check1 = _n.dot(r);
			check = trans.dot(plane) - 1.0;

			if(plane.at<double>(2) > 0.0 )
			{
				trans.copyTo(dt2in1[k]);
				plane.copyTo(norm[k]);
				k++;
			}
         }
    }


    for(i = 0; i < k; ++i)
    {
		for(size_t j = 0; j < 3; j++)
		{
			for(size_t j1 = 0; j1 < 3; j1++)
			{
				tn[j][j1] = dt2in1[i].at<double>(j) * norm[i].at<double>(j1);
			}
		}

		tn[0][0] -=1.0;
		tn[1][1] -=1.0;
		tn[2][2] -=1.0;
		_tn = -1.0 * _tn;
		invtn = _tn.inv();
        rot = h_norm * invtn;
		//printf("det rot %f\n", det33(rot));
		if(determinant(rot) < 0.0)
		{
			rot = -1.0 * rot;
		}
		rot.copyTo(R1_2[i]);
    }

    return 1;
}

/* Estimation of R, t & the plane normal vector (two possible solutions) based on the algorithm from Longuet-Higgine "A computer
 * algorithm for reconstructing a scene from two projections", 1981. These parameters are estimted from a homography H and its
 * inliers (in the camera coordinate system).
 *
 * InputArray H							Input  -> Homography between images and a world plane in the camera coordinate system.
 * vector<Mat> R1_2						Output -> The resulting rotation matrix.
 * vector<Mat> dt1						Input & Output -> As input an initial translation vector has to be specified. The resulting
 *														  translation vector is copied back to dt1.
 * vector<Mat> norm						Output -> The resulting plane normal vector.
 *
 * Return value:						1 :		Everything ok
 *										0 :		Estimation not possible/failed
 */
int LonguetHigginsSolution_with_initial( cv::InputArray H, cv::Mat & R2_1, cv::Mat & dt1, cv::Mat & norm1)
{
    Mat _H = H.getMat();
	double hh[3][3], d[3], u[3][3], n[3], check, d2;
    double p[3], t[3], tn[3][3];
    double homo_norm[3][3];

    double ts[3];
    double ps[3] ;

    double check1;

    int s1, s2, iter, i;

	Mat ht, h_norm, rot, x, r, _ts, trans, plane, _ps, _n, invtn;
	vector<Mat> dt, norm;
	Mat _tn = Mat(3,3,CV_64F,tn);
	Mat _hh = Mat(3,3,CV_64F,hh);
	Mat _u = Mat(3,3,CV_64F,u);

	ht = _H.t();
	_hh = ht * _H;
    //u here is the Ut on the right side
    jacobi33(hh, d, u, &iter);


    //if d[1] != 1.0 make a correction
	_hh = -1.0 * _hh;
    d2 = 1.0/sqrt(d[1]);
	h_norm = d2 * _H;//the homo_norm is the homo transform with
    d[0] = d[0]/d[1];
    d[2] = d[2]/d[1];
    d[1] = 1.0;
    if(d[0] - d[1] < 0.000001 && d[1]- d[2] < 0.000001)
    {
        // the two images are too close each other
        //printf("the images are two close each other and the slope estimation is not valid\n");
        //printf("the rotation estimation is ok\n");
		dt1 = Mat::zeros(3,1,CV_64F);
        //the homography degerate to a rotation
		h_norm.copyTo(rot);
		h_norm.copyTo(R2_1);

        //the surface normal cannot be estimated here
		norm1 = Mat::zeros(3,1,CV_64F);
        return 1;
    }

    if(d[0] > 1.0 && d[2] <= 1.0)
    {
        t[0] = sqrt((d[0] - 1.0)*d[2]/(d[0] - d[2]));
        t[1] = 0.0;
        t[2] = sqrt((1.0 - d[2])*d[0]/(d[0] - d[2]));
        p[0] = sqrt((d[0] - 1.0)*d[0]/(d[0] - d[2]));
        p[1] = 0.0;
        p[2] = sqrt((1.0 - d[2])*d[2]/(d[0] - d[2]));
    }
    else
    {
        return 0;
    }

	x = Mat::zeros(3,1,CV_64F);
	x.at<double>(2) = 1.0;
	r = _u * x;
	_ts = Mat(3,1,CV_64F,ts);
	_ts = Mat::zeros(3,1,CV_64F);
	_n = Mat(3,1,CV_64F,n);

    for(s1 =-1; s1 <=1; s1 +=2)
    {
        for(s2 =-1; s2 <=1; s2 +=2)
        {
		    ps[0] = -s1*p[0];
			ps[1] = 0.0;
			ps[2] = -s2*p[2];
		    n[0] = s1*t[0] + ps[0];
		    n[1] = 0.0;
		    n[2] = s2*t[2] + ps[2];
			ts[0] = s1*t[0];
			ts[1] = 0.0;
			ts[2] = s2*t[2];

			trans = _u * _ts;
			plane = _u * _n;
			check1 = _n.dot(r);
			check = trans.dot(plane) - 1.0;

			if(plane.at<double>(2) > 0.0 )
			{
				dt.push_back(trans.clone());
				norm.push_back(plane.clone());
			}
        }
    }
	if(dt[0].dot(dt1) > dt[1].dot(dt1))
    {
		i = 0;
		dt[0].copyTo(dt1);
		norm[0].copyTo(norm1);
    }
    else
    {
	    i = 1;
		dt[1].copyTo(dt1);
		norm[1].copyTo(norm1);
    }

	for(size_t j = 0; j < 3; j++)
	{
		for(size_t j1 = 0; j1 < 3; j1++)
		{
			tn[j][j1] = dt[i].at<double>(j) * norm[i].at<double>(j1);
		}
	}
    tn[0][0] -=1.0;
    tn[1][1] -=1.0;
    tn[2][2] -=1.0;
	_tn = -1.0 * _tn;
	invtn = _tn.inv();
	rot = h_norm * invtn;

    if(determinant(rot) < 0.0)
    {
		rot = -1.0 * rot;
    }
    rot.copyTo(R2_1);

    return 1;
}

/* Diagonalization of the matrix W=(H^T)H with the unknown orthogonal matrix U (where H is a homography) to solve UWU^T=Diag(d1, d2, d3)
 * as described in the paper "Real-time Surface Estimation by Homography Alignment for Spacecraft Safe Landing" from Yang Cheng
 * in 2010. The input to this function is a=W=(H^T)H. The output are the 3 diagonal elemnts d and the matrix v=U^T.
 *
 * double a[3][3]						Input  -> Matrix W=(H^T)H, where H is a homography matrix
 * double d[3]							Output -> The resulting diagonal entries of the diagonal matrix Diag(d1, d2, d3)
 * double v[3][3]						Output -> The resulting orthogonal matrix U^T
 * int *nrot							Output -> Number of iterations needed for estimating d and v
 *
 * Return value:						1 :		Everything ok
 *										0 :		Estimation failed (max. number of iterations exceeded)
 */
#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
	a[k][l]=h+s*(g-h*tau);

int jacobi33(double a[3][3], double d[3], double v[3][3], int *nrot)
{
	int j,iq,ip,i;
	double tresh,theta,tau,t,sm,s,h,g,c, b[3],z[3];
    double dt[3][3], abc[3][3], tmp[3][3], tmp1[3][3];
	memcpy(abc,a,9*sizeof(double));

	for (ip=0;ip<3;ip++) {
		for (iq=0;iq<3;iq++) v[ip][iq]=0.0;
		v[ip][ip]=1.0;
	}
	for (ip=0;ip< 3;ip++) {
		b[ip]=d[ip]=a[ip][ip];
		z[ip]=0.0;
	}
	*nrot=0;
	for (i=1;i<=50;i++) {
		sm=0.0;
		for (ip=0;ip<2;ip++) {
			for (iq=ip+1;iq< 3;iq++)
				sm += fabs(a[ip][iq]);
		}
		if (sm < 0.000001) {
		    /*rearange the eigenvalue and eigenvector*/
/*		trans33(v, dt);
			mult333(dt, abc, tmp);
			mult333(tmp, v, tmp1);
*/
			if(d[1] > d[0])
			{
				c = d[0];
				d[0] = d[1];
				d[1] = c;
				//zero33(dt);
				memset(dt,0,9*sizeof(double));
				dt[0][1] = 1.0;
				dt[1][0] = 1.0;
				dt[2][2] = 1.0;

				//mult333(v, dt, tmp);
				Mat _v = Mat(3,3,CV_64F,v);
				Mat _dt = Mat(3,3,CV_64F,dt);
				Mat _tmp = Mat(3,3,CV_64F,tmp);
				_tmp = _v * _dt;

				memcpy(v,tmp,9*sizeof(double));
				/*
				for(i = 0; i< 3; ++i)
				{
					c = v[0][i];
					v[0][i] = v[1][i];
					v[1][i] = c;
				}
				*/
			}
			if(d[2] > d[1])
			{
				c = d[1];
				d[1] = d[2];
				d[2] = c;
				memset(dt,0,9*sizeof(double));
				dt[0][0] = 1.0;
				dt[1][2] = 1.0;
				dt[2][1] = 1.0;

				Mat _v = Mat(3,3,CV_64F,v);
				Mat _dt = Mat(3,3,CV_64F,dt);
				Mat _tmp = Mat(3,3,CV_64F,tmp);
				_tmp = _v * _dt;

				memcpy(v,tmp,9*sizeof(double));
				/*
				for(i = 0; i< 3; ++i)
				{
					c = v[1][i];
					v[1][i] = v[2][i];
					v[2][i] = c;
				}*/
			}
			if(d[1] > d[0])
			{
				c = d[0];
				d[0] = d[1];
				d[1] = c;
				memset(dt,0,9*sizeof(double));
				dt[0][1] = 1.0;
				dt[1][0] = 1.0;
				dt[2][2] = 1.0;

				Mat _v = Mat(3,3,CV_64F,v);
				Mat _dt = Mat(3,3,CV_64F,dt);
				Mat _tmp = Mat(3,3,CV_64F,tmp);
				_tmp = _v * _dt;

				memcpy(v,tmp,9*sizeof(double));
				/*
				for(i = 0; i< 3; ++i)
				{
					c = v[0][i];
					v[0][i] = v[1][i];
					v[1][i] = c;
				}*/
			}
			Mat _v = Mat(3,3,CV_64F,v);
			Mat _dt = Mat(3,3,CV_64F,dt);
			_dt = _v.t();
			Mat _abc = Mat(3,3,CV_64F,abc);
			Mat _tmp = Mat(3,3,CV_64F,tmp);
			_tmp = _dt * _abc;
			Mat _tmp1 = Mat(3,3,CV_64F,tmp1);
			_tmp1 = _tmp * _v;
			memcpy(a,abc,9*sizeof(double));//copy back the a
			return 1;
		}
		if (i < 4)
			tresh=0.2*sm/9;
		else
			tresh=0.0;
		for (ip=0;ip<3-1;ip++) {
			for (iq=ip+1;iq<3;iq++) {
				g=100.0*fabs(a[ip][iq]);
				if (i > 4 && (fabs(d[ip])+g) == fabs(d[ip])
					&& (fabs(d[iq])+g) == fabs(d[iq]))
					a[ip][iq]=0.0;
				else if (fabs(a[ip][iq]) > tresh) {
					h=d[iq]-d[ip];
					if ((fabs(h)+g) == fabs(h))
						t=(a[ip][iq])/h;
					else {
						theta=0.5*h/(a[ip][iq]);
						t=1.0/(fabs(theta)+sqrt(1.0+theta*theta));
						if (theta < 0.0) t = -t;
					}
					c=1.0/sqrt(1+t*t);
					s=t*c;
					tau=s/(1.0+c);
					h=t*a[ip][iq];
					z[ip] -= h;
					z[iq] += h;
					d[ip] -= h;
					d[iq] += h;
					a[ip][iq]=0.0;
					for (j=0;j<=ip-1;j++) {
						ROTATE(a,j,ip,j,iq)
					}
					for (j=ip+1;j<=iq-1;j++) {
						ROTATE(a,ip,j,j,iq)
					}
					for (j=iq+1;j<3;j++) {
						ROTATE(a,ip,j,iq,j)
					}
					for (j=0;j<3;j++) {
						ROTATE(v,j,ip,j,iq)
					}
					++(*nrot);
				}
			}
		}
		for (ip=0;ip<3;ip++) {
			b[ip] += z[ip];
			d[ip]=b[ip];
			z[ip]=0.0;
		}
	}
	return 0;
}
#undef ROTATE

double Check_motion_error(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
						  std::vector<unsigned int> num_inl,
						  cv::Mat R1_2,
						  cv::Mat t)
{
	Mat c0, R2_1, p, r1, r2, p1, p2, dp;
	int num_homo = (int)num_inl.size();
	int i, j;
	double error, error1;
	int total_pts;

	c0 = Mat::zeros(3,1,CV_64F);
	R2_1 = R1_2.t();
	total_pts = 0;
	error = 0.0;
	for(i = 0; i < num_homo; ++i)
	{
		error1 = 0;
		for(j = 0; j < num_inl[i]; ++j)
		{
			p = (Mat_<double>(3, 1) << inl_points[i].first.at<double>(j,0), inl_points[i].first.at<double>(j,1), 1.0);

			r1 = p / cv::norm(p);

			p = (Mat_<double>(3, 1) << inl_points[i].second.at<double>(j,0), inl_points[i].second.at<double>(j,1), 1.0);

			p1 = R2_1 * p;
			r2 = p1 / cv::norm(p1);

			if(Rays_ClosestPoints(c0, r1, t, r2, p1, p2)!= 0)
			{
				//printf("i = %d j = %d det %f %f\n", i, j,  det33(epi));
				dp = p1 - p2;
				error1 += cv::norm(dp);

				total_pts++;
			}
		}
		//printf("patch %d  %d error %18.12f\n", i, num_pts[i],  error1/(float)num_pts[i]);
		error +=error1;
	}
	return error/(double)total_pts;
}

int Rays_ClosestPoints(cv::Mat pt1, cv::Mat ray1, cv::Mat pt2, cv::Mat ray2, cv::Mat & p1, cv::Mat & p2)
{
    double m1, m2, dotp, dotbv1, dotbv2;
	Mat b;

	if(fabs(ray1.dot(ray2)) >0.9999999)
	{
		//two ray are close to parallel
		return 0;
	}
	b = pt2 - pt1;
	dotp = ray1.dot(ray2);
	dotbv1 = b.dot(ray1);
	dotbv2 = b.dot(ray2);

	if(dotp == 0.0)
	{
		m1 = dotbv1;
		m2 = dotbv2;
	}
	else
	{
		m1 = (dotbv1 - dotbv2*dotp)/(1.0-dotp*dotp);
		m2 = dotp*m1 - dotbv2;

		//m1 = dotbv1/(1+dotp);
	}
	p1 = m1 * ray1;
	p2 = m2 * ray2;
	p1 = pt1 + p1;
	p2 = pt2 + p2;
	return 1;
}

int update_dn(cv::Mat points1, cv::Mat points2, int num_pts, cv::Mat h0, cv::Mat k0, cv::Mat & dn)
{
	Mat A, B, invA, p, p1, t, t1, _hp, _tt, ty;
	int i;

	double hp[3];
	double tt[3][3];
	double y;
	double d;
	double *_k0;
	_k0 = (double*)k0.data;
	_tt = Mat(3,3,CV_64F,tt);
	_hp = Mat(3,1,CV_64F,hp);
	B = Mat::zeros(3,1,CV_64F);
	A = Mat::zeros(3,3,CV_64F);
	for(i = 0; i < num_pts; ++i)
	{
		p = (Mat_<double>(3, 1) << points1.at<double>(i,0), points1.at<double>(i,1), 1.0);
		p1 = (Mat_<double>(3, 1) << points2.at<double>(i,0), points2.at<double>(i,1), 1.0);
		t = _k0[0] * p;
		d = _k0[2] * points2.at<double>(i,0);
		t1 = d * p;
		t = t - t1;
		_hp = h0 * p;
		y = points2.at<double>(i,0) * hp[2] - hp[0];
		for(int j = 0; j < 3; j++)
		{
			for(int j1 = 0; j1 < 3; j1++)
			{
				tt[j][j1] = t.at<double>(j) * t.at<double>(j1);
			}
		}
		ty = y * t;
		A = _tt + A;
		B = ty + B;

		t = _k0[1] * p;
		d = _k0[2] * points2.at<double>(i,1);
		t1 = d * p;
		t = t - t1;
		_hp = h0 * p;
		y = points2.at<double>(i,1) * hp[2] - hp[1];
		for(int j = 0; j < 3; j++)
		{
			for(int j1 = 0; j1 < 3; j1++)
			{
				tt[j][j1] = t.at<double>(j) * t.at<double>(j1);
			}
		}
		ty = y * t;
		A = _tt + A;
		B = ty + B;

		/*
		p[0] = points1[i*2];
			p[1] = points1[i*2+1];
		p[2] = 1.0;
		mult331(h0, p, hp);
		mult313(k0, p, kp);
		scale3(points2[i*2], kp[2], t);
		sub3(kp[0], t, t);
		y = points2[i*2]*hp[2] - hp[0];
		mult313(t, t, tt);
		scale3(y, t, ty);
		add33(tt, A, A);
		add3(ty, B, B);

		scale3(points2[i*2+1], kp[2], t);
		sub3(kp[1], t, t);
		y = points2[i*2+1]*hp[2] - hp[1];
		mult313(t, t, tt);
		scale3(y, t, ty);
		add33(tt, A, A);
		add3(ty, B, B);
		*/
	}
	invA = A.inv();
	dn = (invA * B).t();

	return 1;
}

 int update_h0_rt(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
				  std::vector<unsigned int> num_inl,
				  cv::Mat & homo,
				  cv::Mat dn,
				  cv::Mat & rt)
{
	Mat rt_back, _p1;
	int num_homo = (int)num_inl.size();
	double *_homo;
	double *_rt;
	_homo = (double*)homo.data;
	_rt = (double*)rt.data;
	int i, j;
	double r[11];
	double a[121], inva[121], b[11], m[11];
	double c[121];
	double p1[3];
	double p2[3];
	double d;
	double dnp;
	_p1 = Mat(3,1,CV_64F,p1);

	rt.copyTo(rt_back);
	for(i = 0; i<11; ++i)
	{
		b[i] = 0.0;
		m[i] = 0.0;
	}
	for(i = 0; i < 121; ++i)
	{
		a[i] = 0.0;
	}
	for(i = 0; i < num_homo; ++i)
	{
		for(j = 0; j < num_inl[i]; ++j)
		{
			p1[0] = inl_points[i].first.at<double>(j,0);
			p1[1] = inl_points[i].first.at<double>(j,1);
			p1[2] = 1.0;

			dnp = _p1.dot(dn.row(i).t());
			p2[0] = inl_points[i].second.at<double>(j,0);
			p2[1] = inl_points[i].second.at<double>(j,1);
			//printf("i = %d point from %f %f to  %f %f\n", i, ipoint[0], ipoint[1], ipoint1[0], ipoint1[1]);
			r[0] = p1[0];
			r[1] = p1[1];
			r[2] = 1.0;
			r[3] = 0.0;
			r[4] = 0.0;
			r[5] = 0.0;

			r[6] = (double)(-p1[0]*p2[0]);
			r[7] = (double)(-p1[1]*p2[0]);

			r[8] = dnp;
			r[9] = 0.0;
			r[10] = -p2[0]*dnp;
			LinearTransformD(r, r, c, 11, 1, 11);
			AddMatrixD (c, a, a, 11, 11);
			d = (double)p2[0];
			ScaleMatrixD (d, r, r, 11, 1);
			AddMatrixD(r, b, b, 11, 1);

			r[0] = 0.0;
			r[1] = 0.0;
			r[2] = 0.0;
			r[3] = p1[0];
			r[4] = p1[1];
			r[5] = 1.0;

			r[6] = (double)(-p1[0]*p2[1]);
			r[7] = (double)(-p1[1]*p2[1]);

			r[8] = 0.0;
			r[9] = dnp;
			r[10] = -p2[1]*dnp;
			LinearTransformD(r, r, c, 11, 1, 11);
			AddMatrixD (c, a, a, 11, 11);
			d = (double)p2[1];
			ScaleMatrixD (d, r, r, 11, 1);
			AddMatrixD(r, b, b, 11, 1);
		}
	}
	if(InvertMatrixD(a, inva, 11, 11) == 0)
	{
		return (0);
	}

	LinearTransformD(inva, b, m, 11, 11, 1);
	_homo[0] = m[0];
	_homo[1] = m[1];
	_homo[2] = m[2];

	_homo[3] = m[3];
	_homo[4] = m[4];
	_homo[5] = m[5];

	_homo[6] = m[6];
	_homo[7] = m[7];
	_homo[8] = 1.0;
	_rt[0] = m[8];
	_rt[1] = m[9];
	_rt[2] = m[10];
	return 1;
}

/********************************************************************************
 * LinearTransform
 * Linear transformations, for transforming vectors and matrices.
 * This works for row vectors and column vectors alike.
 *	L[nRows][lCol]	- input (left) matrix
 *	rg[lCol][rCol]	- transformation (right) matrix
 *	P[nRows][rCol]	- output (product) matrix
 *
 * Examples:
 * v[3] * M[3][3] -> w[3] :			MLLinearTransform(&v[0], &M[0][0], &w[0], 1, 3, 3);
 * M[3][3] * v[3] -> w[3] :			MLLinearTransform(&M[0][0], &v[0], &w[0], 3, 3, 1);
 * M[4][4] * N[4][4] -> P[4][4]:	MLLinearTransform(&M[0][0], &N[0][0], &P[0][0], 4, 4, 4);
 * v[4] * M[4][3] -> w[3]:			MLLinearTransform(&v[0], &M[0][0], &w[0], 1, 4, 3);
 * v[3] tensor w[3] -> T[3][3]:		MLLinearTransform(&v[0], &w[0], T[3][3], 3, 1, 3);
 * This can be used In Place, i.e.,
 * to transform the left matrix
 * by the right matrix, placing the result back in the left.  By its nature,
 * then, this can only be used for transforming row vectors or concatenating
 * matrices from the right.
 ********************************************************************************/
#define MAXDIM	32			/* The maximum dimension of a matrix */

void LinearTransformD(
	const double		*L,		/* The left matrix */
	const double		*R,		/* The right matrix */
	register double	*P,		/* The resultant matrix */
	long			nRows,	/* The number of rows of the left and resultant matrices */
	long			lCol,	/* The number of columns in the left matrix */
	long			rCol	/* The number of columns in the resultant matrix */
)
{
	register const double *lp;		/* Left matrix pointer for dot product */
	register const char *rp;		/* Right matrix pointer for dot product */
	register long k;				/* Loop counter */
	register double sum;			/* Extended precision for intermediate results */
	register long rowBytes = lCol * sizeof(double);
	register long rRowBytes = rCol * sizeof(double);
	register long j, i;				/* Loop counters */
	register long lRowBytes = lCol * sizeof(double);
	const char *lb = (const char*)L;
	double temp[MAXDIM*MAXDIM]; // Temporary storage for in-place transformations
	register double *tp;

	if (P == L) {  // IN PLACE
		double *op = P;				/* Output geometry */
		for (i = nRows; i--; lb += rowBytes) {	/* Each row in L */
			{
				for (k = lCol, lp = (double*)lb, tp = &temp[0]; k--; )
					*tp++ = *lp++;			/* Copy one input vector to temp storage */
			}
			for (j = 0; j < lCol; j++) {		/* Each column in R */
				lp = &temp[0];				/* Left of ith row of L */
				rp = (const char *)(R + j);	/* Top of jth column of R */
				sum = 0;
				for (k = lCol; k--; rp += rowBytes)
					sum += *lp++ * (*((const double*)rp));	/* *P += L[i'][k'] * R[k'][j] */
				*op++ = sum;
			}
		}
	} else if (P != R) {
		for (i = nRows; i--; lb += lRowBytes) {	/* Each row in L */
			for (j = 0; j < rCol; j++) {	/* Each column in R */
				lp = (const double *)lb;		/* Left of ith row of L */
				rp = (const char *)(R + j);	/* Top of jth column of R */
				sum = 0;
				for (k = lCol; k--; rp += rRowBytes)
					sum += *lp++ * (*((const double*)rp));	/* *P += L[i'][k'] * R[k'][j] */
				*P++ = sum;
			}
		}
	} else { // P == R
		for (tp = temp, i = lCol * rCol; i--; ) *tp++ = *R++;  // copy R
		for (i = nRows; i--; lb += lRowBytes) {	/* Each row in L */
			for (j = 0; j < rCol; j++) {	/* Each column in R */
				lp = (const double *)lb;		/* Left of ith row of L */
				rp = (const char *)(temp + j);	/* Top of jth column of R (now in temp) */
				sum = 0;
				for (k = lCol; k--; rp += rRowBytes)
					sum += *lp++ * (*((const double*)rp));	/* *P += L[i'][k'] * R[k'][j] */
				*P++ = sum;
			}
		}
	}
}

void AddMatrixD (double *A, double *B, double *result, long m, long n)
{
	long i;
	for (i = m * n; i --; A++, B++, result ++)
		*result = (*A) + (*B);
}

void ScaleMatrixD (double scale, double *from, double *to, long m, long n)
{
	long i;
	for (i = m * n; i--; from ++, to++)
		*to = scale * (*from);
}

/********************************************************************************
 * InvertMatrix()
 *	Inverts square matrices
 *	With tall matrices, invert upper part and transform the bottom
 *	rows as would be expected if embedded into a larger matrix.
 *	Undefined for wide matrices.
 * M^(-1) --> Minv
 * IN PLACE SUPPORTED, no performance difference
 *
 * 1 is returned if the matrix was non-singular and the inversion was successful;
 * 0 is returned if the matrix was singular and the inversion failed.
 ********************************************************************************/

long InvertMatrixD (const double *M, double *Minv, long nRows, register long n)
{
	double *m;
	long tallerBy = nRows - n;		/* Excess of rows over columns */
	register long j, i;
	double b[MAXDIM];
	double lu[MAXDIM*MAXDIM+MAXDIM];

	/* Decompose matrix into L and U triangular matrices */
// was	if ((tallerBy < 0) || (MLLUdecompose(M, lu, n) == 0)) {
	if ((tallerBy < 0) || (LUDecomposeD(M, lu, n) == 0)) {
		return(0);		/* Singular */
	}

	/* Invert matrix by solving n simultaneous equations n times */
	for (i = 0, m = Minv; i < n; i++, m += n) {
		for(j = 0; j < n; j++)
			b[j] = 0;
		b[i] = 1;
	// was	MLLUsolve(lu, m, b, n);	/* Into a row of m */

		LUSolveD(lu, b, m, n);	/* Into a row of m */
	}

	/* Special post-processing for affine transformations (e.g. 4x3) */
	if (tallerBy) {			/* Affine transformation */
		register double *t = Minv+n*n;			/* Translation vector */
		m = Minv;			/* Reset m */
	// was	MLLinearTransformInPlace(t, m, tallerBy, n);	/* Invert translation */
		LinearTransformD(t, m, t, tallerBy, n, n);	/* Invert translation */
		for (j = tallerBy * n; n--; t++)
			*t = -*t;				/* Negate translation vector */
	}

	return(1);
}

/********************************************************************************
 * LUDecompose() decomposes the coefficient matrix A into upper and lower
 * triangular matrices, the composite being the LU matrix.
 * This is then followed by multiple applications of FELUSolve(),
 * to solve several problems with the same system matrix.
 *
 * 1 is returned if the matrix is non-singular and the decomposition was successful;
 * 0 is returned if the matrix is singular and the decomposition failed.
 ********************************************************************************/
#define luel(i, j)  lu[(i)*n+(j)]
#define ael(i, j)	a[(i)*n+(j)]

long LUDecomposeD(
	register const double	*a,		/* the (n x n) coefficient matrix */
	register double		*lu, 	/* the (n x n) lu matrix augmented by an (n x 1) pivot sequence */
	register long			n		/* the order of the matrix */
)
{
	register long i, j, k;
	short pivotindex;
	double pivot, biggest, mult, tempf;
	register long *ps;
	double scales[MAXDIM];

	ps = (long *)(&lu[n*n]); /* Memory for ps[] comes after LU[][] */

	for (i = 0; i < n; i++) {	/* For each row */
		/* Find the largest element in each row for row equilibration */
		biggest = 0.0;
		for (j = 0; j < n; j++)
			if (biggest < (tempf = fabs(luel(i,j) = ael(j,i)))) /* A transposed for row vectors */
				biggest = tempf;
		if (biggest != 0.0)
			scales[i] = 1.0 / biggest;
		else {
			scales[i] = 0.0;
			return(0);	/* Zero row: singular matrix */
		}

		ps[i] = i;		/* Initialize pivot sequence */
	}

	for (k = 0; k < n-1; k++) { /* For each column */
		/* Find the largest element in each column to pivot around */
		biggest = 0.0;
		for (i = k; i < n; i++) {
			if (biggest < (tempf = fabs(luel(ps[i],k)) * scales[ps[i]])) {
				biggest = tempf;
				pivotindex = (short)i;
			}
		}
		if (biggest == 0.0)
			return(0);	/* Zero column: singular matrix */
		if (pivotindex != k) {	/* Update pivot sequence */
			j = ps[k];
			ps[k] = ps[pivotindex];
			ps[pivotindex] = j;
		}

		/* Pivot, eliminating an extra variable each time */
		pivot = luel(ps[k],k);
		for (i = k+1; i < n; i++) {
			luel(ps[i],k) = mult = luel(ps[i],k) / pivot;
			if (mult != 0.0) {
				for (j = k+1; j < n; j++)
					luel(ps[i],j) -= mult * luel(ps[k],j);
			}
		}
	}
	return(luel(ps[n-1],n-1) != 0.0);	/* 0 if singular, 1 if not */
}	/* Decompose */

/********************************************************************************
 * Solve() solves the linear equation (xA = b) after the matrix A has
 * been decomposed with LUDecompose() into the lower and upper triangular
 * matrices L and U, giving the equivalent equation (xUL = b).
 ********************************************************************************/
void LUSolveD(
	register const double	*lu,	/* the decomposed LU matrix */
	register const double	*b,		/* the constant vector */
	register double			*x,		/* the solution vector */
	register long			n		/* the order of the equation */
)
{
	register long i, j;
	double dot;
	register const long *ps;

	ps = (const long *)(&lu[n*n]); /* Memory for ps[] comes after LU[][] */

	/* Vector reduction using U triangular matrix */
	for (i = 0; i < n; i++) {
		dot = 0.0;
		for (j = 0; j < i; j++)
			dot += luel(ps[i],j) * x[j];
		x[i] = b[ps[i]] - dot;
	}

	/* Back substitution, in L triangular matrix */
	for (i = n-1; i >= 0; i--) {
		dot = 0.0;
		for (j = i+1; j < n; j++)
			dot += luel(ps[i],j) * x[j];
		x[i] = (x[i] - dot) / luel(ps[i],i);
	}
}	/* LUSolve */

//compute mean of reprojection error
double Check_error(std::vector<std::pair<cv::Mat,cv::Mat>> inl_points,
				  std::vector<unsigned int> num_inl,
				  cv::Mat base_homo,
				  cv::Mat dn,
				  cv::Mat rt)
	//double **points1, double **points2, int *num_pts, int num_homo, double base_homo[3][3], double *dn, double rt[3])
{
	int num_homo = (int)num_inl.size();
	double *points1, *points2;
	Mat _h;
	int i, j;
	double error, error1;
	double dx, dy;
	double op[2];
	double h[3][3];
	double s;
	int total_pts;
	_h = Mat(3,3,CV_64F,h);

	total_pts = 0;
	error = 0.0;
	for(i = 0; i < num_homo; ++i)
	{
		for(size_t j1 = 0; j1 < 3; j1++)
		{
			for(size_t j2 = 0; j2 < 3; j2++)
			{
				h[j1][j2] = rt.at<double>(j1) * dn.at<double>(i,j2);
			}
		}
		_h = base_homo + _h;
		s = 1.0/h[2][2];
		_h = s * _h;
		error1 = 0;
		for(j = 0; j < num_inl[i]; ++j)
		{
			points1 = (double*)inl_points[i].first.row(j).data;
			points2 = (double*)inl_points[i].second.row(j).data;
			homographyTransfer33D(h, points1, op);
			dx = op[0] - points2[0];
			dy = op[1] - points2[1];
			//printf("i = %d dx dy %f %f\n", i, dx, dy);
			error1 +=sqrt(dx*dx + dy*dy);
			total_pts++;
		}
		//printf("patch %d error %18.12f\n", i, error1/(float)num_pts[i]);
		error +=error1;
	}
	return error/(float)total_pts;
}

int homographyTransfer33D(double h[3][3], double ip[2], double op[2])
{
	double  p[3];
	double hp[3];
	Mat _h, _p, _hp;
	_h = Mat(3,3,CV_64F,h);
	_hp = Mat(3,1,CV_64F,hp);
	_p = Mat(3,1,CV_64F,p);
	convertToHomo(ip, p);
	_hp = _h * _p;
	convertToImage(hp, op);
	return 1;
}

int convertToHomo(double ip[2], double op[3])
{
	op[0] = ip[0];
	op[1] = ip[1];
	op[2] = 1.0;
	return 1;
}

int convertToImage(double ip[3], double op[2])
{
	op[0] = ip[0]/ip[2];
	op[1] = ip[1]/ip[2];
	return 1;
}

//rot is from camera1 to camera 2
//t is the translation in camera 1 frame, plane is in the camera 1 frame
//in this formula, the 3d point translation is
//P2 = RP1 + dt
//the argument input is
//P2 = R(P1-t)

int constructAnalyticHomography(cv::Mat rot, cv::Mat t, cv::Mat plane, cv::Mat & h)
{
	Mat _tp, im;
	double tp[3][3], s;
	_tp = Mat(3,3,CV_64F,tp);

	for(size_t j1 = 0; j1 < 3; j1++)
	{
		for(size_t j2 = 0; j2 < 3; j2++)
		{
			tp[j1][j2] = t.at<double>(j1) * plane.at<double>(j2);
		}
	}
	im = Mat::eye(3,3,CV_64F);
	_tp = im - _tp;
	h = rot * _tp;
	if(fabs(h.at<double>(2,2)) > 0.0)
	{
		s = 1.0/h.at<double>(2,2);
		h = s * h;
		return 1;
	}
	else
	{
		return 0;
	}
}
