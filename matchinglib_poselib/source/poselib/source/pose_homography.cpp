/**********************************************************************************************************
 FILE: pose_homography.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: June 2016

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionality to robustly estimate a pose out of multiple homographies
 which in turn are generated using point correspondences and a homography estimation algorithm embedded
 in the ARRSAC algorithm.
**********************************************************************************************************/

#include "poselib/pose_homography.h"
#include "arrsac/arrsac.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "HomographyAlignment.h"
#include "poselib/pose_helper.h"
#include "opencv2/calib3d/calib3d_c.h"

using namespace cv;
using namespace std;

namespace poselib
{

/* --------------------------- Defines --------------------------- */



/* --------------------- Function prototypes --------------------- */

//Homography estimation kernel function.
bool runHomogrophyKernel( const cv::Mat* m1, const cv::Mat* m2, cv::Mat* H );
//Calculation of reprojection errors for a given set of correspondences and a homography.
void computeHomographyReprojError(const cv::Mat* m1, const cv::Mat* m2, const cv::Mat* model, std::vector<double> & err);
//Finds inliers to a given homography.
void findHomographyInliers(cv::InputArray p1, cv::InputArray p2, cv::InputArray H, double th, cv::OutputArray mask);
//Refines a given homography
bool refineHomography( const cv::Mat* m1, const cv::Mat* m2, cv::Mat* model, int maxIters = 10 );


/* --------------------------- Classes --------------------------- */

//ARRSAC algorithm wich embeds homography estimation algorithms for robust homography estimations
class ArrHomogrophyEstimator: public theia::Estimator<size_t,cv::Mat>
{
public:
	ArrHomogrophyEstimator(const cv::Mat points1, const cv::Mat points2):points1_(points1),
		 points2_(points2) {}

	~ArrHomogrophyEstimator() {}

	bool EstimateModel(const std::vector<size_t>& data,
		std::vector<cv::Mat>* model) const;

	bool EstimateModelNonminimal(const std::vector<size_t>& data,
		std::vector<cv::Mat>* model) const;

	double Error(const size_t& data, const cv::Mat & model) const;

private:
	cv::Mat points1_, points2_;
};

/* --------------------- Functions --------------------- */

/* Estimates an essential matrix as well as rotation and translation using homography alignment. It robustly
 * estimates a pose out of multiple homographies which in turn are generated using point correspondences and
 * a homography estimation algorithm embedded in the ARRSAC algorithm.
 *
 * InputArray p1							Input  -> Left image projections (n rows x 2 columns)
 * InputArray p2							Input  -> Right image projections (n rows x 2 columns)
 * OutputArray R							Output -> Rotation matrix
 * OutputArray t							Output -> Translation vector
 * OutputArray E							Output -> Essential matrix
 * double th								Input  -> Inlier/outlier threshold
 * int inliers								Output -> Number of inliers
 * InputOutputArray mask					I/O	   -> Mask for inliers of correspondences p1 & p2
 * bool checkPlaneStrength					Input  -> If true [Default=false], the pose is only estimated if the
 *													  planes are dominant enough
 * bool varTh								Input  -> If true [default=false], a variable threshold is used for the homography
 *													  estimation with th the minimum threshold. The threshold is
 *													  only changed if no plane was found in an iteration.
 * vector<pair<Mat,Mat>> *inlier_points		Output -> Optional correspondences (left, right) for every found plane
 * vector<unsigned int> *numbersHinliers	Output -> Optional number of inliers for every found plane
 * vector<Mat> *homographies				Output -> Optional homography for every found plane
 * vector<double> *planeStrength			Output -> Optional strength of each found plane
 *													  ( th * number_inliers / (actual_th * number_correspondences) )
 *
 * Return value:						 0:	Everything ok
 *										-1:	Estimation of multiple homographies failed
 *										-2: Sum of homography strengths too low
 *										-3: Homography alignment failed
 *										-4: Pose variables are necessary
 */
int estimatePoseHomographies(cv::InputArray p1,
							 cv::InputArray p2,
							 cv::OutputArray R,
							 cv::OutputArray t,
							 cv::OutputArray E,
							 double th,
							 int & inliers,
							 cv::InputOutputArray mask,
							 bool checkPlaneStrength,
							 bool varTh,
							 std::vector<std::pair<cv::Mat,cv::Mat>>* inlier_points,
							 std::vector<unsigned int>* numbersHinliers,
							 std::vector<cv::Mat>* homographies,
							 std::vector<double>* planeStrengths)
{

	Mat p_tmp1 = p1.getMat();
	Mat p_tmp2 = p2.getMat();
	Mat mask_;
	Mat p_tmp21, p_tmp22;
	vector<pair<Mat,Mat>> inl_points;
	//std::vector<cv::Mat> inl_mask;
	std::vector<unsigned int> num_inl;
	vector<Mat> Hs;
	vector<double> planeStrength;

	if(!mask.empty())
	{
		mask_ = mask.getMat();
		for(int i = 0; i < p_tmp1.rows; i++)
		{
			if(mask_.at<bool>(i))
			{
				p_tmp21.push_back(p_tmp1.row(i));
				p_tmp22.push_back(p_tmp2.row(i));
			}
		}
	}
	else
	{
		p_tmp21 = p_tmp1;
		p_tmp22 = p_tmp2;
	}
	if(!estimateMultHomographys(p_tmp21, p_tmp22, th, NULL, &inl_points, &Hs, &num_inl, varTh, &planeStrength, MAX_PLANES_PER_PAIR, MIN_PTS_PLANE))
	{
		double planeStrSum = 0;
		if(checkPlaneStrength)
		{
			for(size_t i = 0; i < planeStrength.size(); i++)
			{
				if(planeStrength[i] > 0.1)
					planeStrSum += planeStrength[i];
			}
		}

		if((planeStrSum > 0.5) || !checkPlaneStrength)
		{
			//Estimate the pose using the homography alignment methode
			Mat R1_2, t1_2, N;
			vector<Mat> Hs_out;
			//double roll, pitch, yaw;
			if(ComputeHomographyMotion(inl_points, Hs, num_inl, R1_2, t1_2, th, N, Hs_out))
			{
				//getAnglesRotMat(R1_2, roll, pitch, yaw);
				//R1_2.copyTo(this->R_homo);
				std::vector<double> repr_error;
				double th2 = th * th;
				Mat mask2_tmp, E_tmp;
				t1_2 = -1.0 * t1_2;
				double normt = normFromVec(t1_2);
				if(std::abs(normt-1.0) > 1e-4)
					t1_2 /= normt;
				if(!t.needed())
					return -4; //Pose variables are necessary

				t.create(3,1, CV_64FC1);
				Mat t_ = t.getMat();
				t1_2.copyTo(t_);

				if(!R.needed())
					return -4; //Pose variables are necessary

				R.create(3, 3, CV_64FC1);
				Mat R_ = R.getMat();
				R_ = R1_2.t();

				E_tmp = getEfromRT(R_, t_);
				if(E.needed())
				{
					E.create(3,3, CV_64FC1);
					Mat E_ = E.getMat();
					E_tmp.copyTo(E_);
				}

				computeReprojError2(p_tmp1, p_tmp2, E_tmp, repr_error);
				mask2_tmp = Mat::zeros(1, p_tmp1.rows, CV_8U);
				inliers = 0;
				for(int i = 0; i < p_tmp1.rows; i++)
				{
					if(repr_error[i] < th2)
					{
						mask2_tmp.at<bool>(0,i) = true;
						inliers++;
					}
				}
				if(mask.needed())
				{
					if(mask.empty())
					{
						mask.create(1, p_tmp1.rows, CV_8U);
						mask_ = mask.getMat();
					}
					mask2_tmp.copyTo(mask_);
				}
			}
			else
				return -3; //Homography alignment failed
		}
		else
			return -2; //Sum of homography strengths too low

		if(planeStrengths != nullptr)
		{
			*planeStrengths = planeStrength;
		}
		if(inlier_points != nullptr)
		{
			*inlier_points = inl_points;
		}
		if(numbersHinliers != nullptr)
		{
			*numbersHinliers = num_inl;
		}
		if(homographies != nullptr)
		{
			*homographies = Hs;
		}
	}
	else
		return -1; //Estimation of multiple homographies failed

	return 0;
}

/* Estimates n planes and their homographies in the given set of point correspondences.
 *
 * InputArray p1						Input  -> Left image projections (n_p rows x 2 columns)
 * InputArray p2						Input  -> Right image projections (n_p rows x 2 columns)
 * double th							Input  -> Inlier/outlier threshold (used for ARRSAC)
 * vector<Mat> *inl_mask				Output -> Optional correspondence mask for every found plane
 * vector<pair<Mat,Mat>> *inl_points	Output -> Optional correspondences (left, right) for every found plane
 * vector<Mat> *Hs						Output -> Optional homography for every found plane
 * vector<unsigned int> *num_inl		Output -> Optional number of inliers for every found plane
 * bool varTh							Input  -> If true [default], a variable threshold is used with
 *												  th the minimum threshold. The threshold is only changed
 *												  if no plane was found in an iteration.
 * vector<double> *planeStrength		Output -> Optional strength of each found plane
 *												  ( th * number_inliers / (actual_th * number_correspondences) )
 * unsigned int maxPlanes				Input  -> To specify a maximum number of planes (default = 0 = infinity)
 * unsigned int minPtsPerPlane			Input  -> Minimum number of points per valid plane (default = 4)
 *
 * Return value:						 0:	Everything ok
 *										-1:	No homography found
 */
int estimateMultHomographys(cv::InputArray p1,
							cv::InputArray p2,
							double th,
							std::vector<cv::Mat> *inl_mask,
							std::vector<std::pair<cv::Mat,cv::Mat>> *inl_points,
							std::vector<cv::Mat> *Hs,
							std::vector<unsigned int> *num_inl,
							bool varTh,
							std::vector<double> *planeStrength,
							unsigned int maxPlanes,
							unsigned int minPtsPerPlane)
{
	std::vector<cv::Mat> inl_mask_tmp;
	std::vector<std::pair<cv::Mat,cv::Mat>> inl_points_tmp;
	std::vector<cv::Mat> Hs_tmp;
	std::vector<unsigned int> num_inl_tmp;
	std::vector<double> planeStrength_tmp;
	vector<pair<unsigned int, unsigned int>> num_inl_idx;
	Mat p1_, p2_, mask, mask_init;
	unsigned int n, n_inl, n_rest, i = 0;
	const double maxTh = th * 6.0;
	const double th_mult = 1.5;
	const double th_mult_base = 1.5; //Multiplication factor for the initial threshold which is applied by default
	p1_ = p1.getMat().clone();
	p2_ = p2.getMat().clone();
	n = n_inl = p1_.rows;

	CV_Assert((n == p2_.rows) && (p1_.cols == 2) && (p2_.cols == 2) && (n > 3));

	p1_ = p1_.reshape(2);
	p2_ = p2_.reshape(2);
	n_rest = n;

	while((n_rest > 3) && (n_inl >= minPtsPerPlane) && (!maxPlanes || (i < maxPlanes)))
	{
		Mat p1_new, p2_new, p1_tmp, p2_tmp, H;
		double th_tmp = th * th_mult_base;
		bool result;

		do
		{
			result = computeHomographyArrsac(p1_, p2_, H, th_tmp, mask, p1_new, p2_new);
			//H = findHomography(p1_, p2_, CV_RANSAC, th_tmp, mask);
			th_tmp *= th_mult;
		}
		while(varTh && !result && (th_tmp < maxTh));

		if(H.empty())
		{
			if(!i)
				return -1; //No homography found
			break;
		}

		n_inl = countNonZero(mask);
		n_rest -= n_inl;

		if(num_inl && (n_inl >= minPtsPerPlane))
			num_inl_tmp.push_back(n_inl);

		if(planeStrength && (n_inl >= minPtsPerPlane))
		{
			th_tmp /= th_mult;
			planeStrength_tmp.push_back(th * th_mult_base * (double)n_inl / (th_tmp * (double)n));
		}

		for(int j = 0; j < p1_.rows; j++)
		{
			if(!mask.at<bool>(j))
			{
				p1_tmp.push_back(p1_.row(j));
				p2_tmp.push_back(p2_.row(j));
			}
		}
		if(inl_points && (n_inl >= minPtsPerPlane))
		{
			p1_new = p1_new.reshape(1);
			p2_new = p2_new.reshape(1);
			inl_points_tmp.push_back(make_pair<cv::Mat, cv::Mat>(p1_new.clone(),p2_new.clone()));
		}
		if(Hs && (n_inl >= minPtsPerPlane))
			Hs_tmp.push_back(H.clone());

		if(inl_mask)
		{
			if(!i)
			{
				if(n_inl >= minPtsPerPlane)
					inl_mask_tmp.push_back(mask.clone());
				mask.copyTo(mask_init);
				for(int j = 0; j < p1_.rows; j++)
				{
					if(mask_init.at<bool>(j))
						mask_init.at<bool>(j) = false;
					else
						mask_init.at<bool>(j) = true;
				}
			}
			else
			{
				size_t k = 0;
				if(n_inl >= minPtsPerPlane)
					inl_mask_tmp.push_back(mask_init.clone());
				for(size_t j = 0; j < n; j++)
				{
					if(mask_init.at<bool>(j))
					{
						if(mask.at<bool>(k))
							mask_init.at<bool>(j) = false;
						else if(n_inl >= minPtsPerPlane)
							inl_mask_tmp.back().at<bool>(j) = false;
						k++;
					}
				}
			}
		}

		p1_ = p1_tmp;
		p2_ = p2_tmp;
		i++;
	}

	if(num_inl && (num_inl_tmp.size() > 1))
	{
		for(unsigned int i = 0; i < num_inl_tmp.size(); i++)
		{
			num_inl_idx.push_back(make_pair(i,num_inl_tmp.at(i)));
		}
		std::sort(num_inl_idx.begin(),num_inl_idx.end(),[](std::pair<unsigned int, unsigned int> const & first, std::pair<unsigned int, unsigned int> const & second){
			return first.second > second.second;});

		if(inl_mask_tmp.size() > 1)
		{
			for(unsigned int i = 0; i < num_inl_idx.size(); i++)
				inl_mask->push_back(inl_mask_tmp[num_inl_idx[i].first]);
		}
		if(inl_points_tmp.size() > 1)
		{
			for(unsigned int i = 0; i < num_inl_idx.size(); i++)
				inl_points->push_back(inl_points_tmp[num_inl_idx[i].first]);
		}
		if(Hs_tmp.size() > 1)
		{
			for(unsigned int i = 0; i < num_inl_idx.size(); i++)
				Hs->push_back(Hs_tmp[num_inl_idx[i].first]);
		}
		if(num_inl_tmp.size() > 1)
		{
			for(unsigned int i = 0; i < num_inl_idx.size(); i++)
				num_inl->push_back(num_inl_tmp[num_inl_idx[i].first]);
		}
		if(planeStrength_tmp.size() > 1)
		{
			for(unsigned int i = 0; i < num_inl_idx.size(); i++)
				planeStrength->push_back(planeStrength_tmp[num_inl_idx[i].first]);
		}
	}
	else
	{
		if(inl_mask)
			*inl_mask = inl_mask_tmp;
		if(inl_points)
			*inl_points = inl_points_tmp;
		if(Hs)
			*Hs = Hs_tmp;
		if(num_inl)
			*num_inl = num_inl_tmp;
		if(planeStrength)
			*planeStrength = planeStrength_tmp;
	}

	return 0;
}

/* Robust homography estimation using ARRSAC.
 *
 * InputArray p1						Input  -> Left image projections (n_p rows x 1 columns x 2 channels)
 * InputArray p2						Input  -> Right image projections (n_p rows x 1 columns x 2 channels)
 * OutputArray H						Output -> Estimated homography matrix
 * double th							Input  -> Inlier/outlier threshold
 * OutputArray mask						Output -> Optional correspondence mask
 * OutputArray p_filtered1				Output -> Optional left correspondences for the found plane
 * OutputArray p_filtered2				Output -> Optional right correspondences for the found plane
 *
 * Return value:						true:	Everything ok
 *										false:	No homography found
 */
bool computeHomographyArrsac(cv::InputArray points1, cv::InputArray points2, cv::OutputArray H, double th, cv::OutputArray mask, cv::OutputArray p_filtered1, cv::OutputArray p_filtered2)
{
	Mat p1 = points1.getMat(), p2 = points2.getMat();

	CV_Assert((p1.cols == 1) && (p1.cols == p2.cols) &&
			  (p1.rows == p2.rows) && (p1.type() == CV_64FC2) &&
			  (p2.type() == CV_64FC2));
	int n = p1.rows;
	Mat H_, H_tmp, mask_, p1_, p2_;

	if(n < 4)
		return false;
	else if(n == 4)
	{
		cv::Mat m1 = p1_ = p1, m2 = p2_ = p2, H__;
		H__ = H_tmp = Mat(3,3,CV_64FC1);
		if(!runHomogrophyKernel( &m1, &m2, &H__ ))
			return false;
		mask_ = Mat::ones(1,4,CV_8U);
	}
	else
	{
		std::vector<size_t> input_data;
		H_tmp = Mat(3,3,CV_64FC1);
		for(int i = 0; i < n; i++)
			input_data.push_back(i);
		ArrHomogrophyEstimator esti(p1, p2);
		theia::Arrsac<size_t,cv::Mat> arrsac_estimator(4,th * th,500,100,12);
		if(arrsac_estimator.Estimate(input_data, esti, &H_tmp))
		{
			cv::Mat p1__, p2__, H__, H__tmp;
			findHomographyInliers(p1, p2, H_tmp, th, mask_);
			for(int i = 0; i < n; i++)
			{
				if(mask_.at<bool>(i))
				{
					p1_.push_back(p1.row(i));
					p2_.push_back(p2.row(i));
				}
			}
			p1__ = p1_;
			p2__ = p2_;
			H__tmp = H_tmp;
			H__ = H__tmp.clone();
			if(refineHomography( &p1__, &p2__, &H__, 10)) {
//                H_tmp = cv::cvarrToMat(&H__);
                H__.copyTo(H_tmp);
            }
		}
		else
			return false;
	}

	if(p_filtered1.needed() && p_filtered2.needed())
	{
		Mat p1_tmp, p2_tmp;
		p_filtered1.create(p1_.size(),p1_.type());
		p_filtered2.create(p2_.size(),p2_.type());
		p1_tmp = p_filtered1.getMat();
		p2_tmp = p_filtered2.getMat();
		p1_.copyTo(p1_tmp);
		p2_.copyTo(p2_tmp);
	}

	if(mask.needed())
	{
		Mat mask_tmp;
		mask.create(mask_.size(), mask_.type());
		mask_tmp = mask.getMat();
		mask_.copyTo(mask_tmp);
	}

	H.create(3,3,CV_64FC1);
	H_ = H.getMat();
	H_tmp.copyTo(H_);

	return true;
}

/* Homography estimation function called by ARRSAC using a minimal set of correspondences (4).
 *
 * vector<size_t> data					Input  -> Reference (index) values to the correspondences
 * vector<cv::Mat>* model				Output -> Estimated homography
 *
 * Return value:						true:	Everything ok
 *										false:	No valid homography found
 */
bool ArrHomogrophyEstimator::EstimateModel(const std::vector<size_t>& data,
										  std::vector<cv::Mat>* model) const
{
	CV_Assert((points1_.cols == 1) && (points1_.cols == points2_.cols) &&
			  (points1_.rows == points2_.rows) && (points1_.type() == CV_64FC2) &&
			  (points2_.type() == CV_64FC2));
	cv::Mat model_;
	Mat model__;
	cv::Mat m1((int)data.size(), points1_.cols, points1_.type());
	cv::Mat m2((int)data.size(), points2_.cols, points2_.type());
	cv::Mat cvm1;
	cv::Mat cvm2;

	for(size_t i = 0; i < data.size(); i++)
	{
		points1_.row(data[i]).copyTo(m1.row(i));
		points2_.row(data[i]).copyTo(m2.row(i));
	}

	cvm1 = m1;
	cvm2 = m2;

	model_ = Mat( 3, 3, CV_64FC1 );

	if(!runHomogrophyKernel( &cvm1, &cvm2, &model_ ))
        return false;

//	model__ = cv::cvarrToMat(model_);
    model_.copyTo(model__);
	model->push_back(model__.clone()); //Hopefully there is no memory-problem with that
//	cvReleaseMat(&model_);

	return true;
}

/* Homography estimation function called by ARRSAC using a nonminimal set of correspondences (typically 12).
 *
 * vector<size_t> data					Input  -> Reference (index) values to the correspondences
 * vector<cv::Mat>* model				Output -> Estimated homography
 *
 * Return value:						true:	Everything ok
 *										false:	No valid homography found
 */
bool ArrHomogrophyEstimator::EstimateModelNonminimal(const std::vector<size_t>& data,
													std::vector<cv::Mat>* model) const
{
	CV_Assert((points1_.cols == 1) && (points1_.cols == points2_.cols) &&
			  (points1_.rows == points2_.rows) && (points1_.type() == CV_64FC2) &&
			  (points2_.type() == CV_64FC2));
	cv::Mat m1((int)data.size(), points1_.cols, points1_.type());
	cv::Mat m2((int)data.size(), points2_.cols, points2_.type());
	cv::Mat H;

	for(size_t i = 0; i < data.size(); i++)
	{
		points1_.row(data[i]).copyTo(m1.row(i));
		points2_.row(data[i]).copyTo(m2.row(i));
	}
	/*m1.reshape(1);
	m2.reshape(1);*/

	H = findHomography(m1, m2, 0);
	if(H.empty() || (H.rows < 3) || (H.cols < 3) || (H.at<double>(0) == 0.0) || (H.at<double>(8) == 0.0))
		return false;

	model->push_back(H.clone());

	return true;
}

/* Calculation of the reprojection error for a given correspondence and homography.
 *
 * size_t data							Input  -> Reference (index) value to the correspondence
 * Mat model							Input  -> Estimated homography
 *
 * Return value:						Error value
 */
double ArrHomogrophyEstimator::Error(const size_t& data, const cv::Mat & model) const
{
	cv::Mat m1, m2;
	cv::Mat mc1, mc2;
	mc1 = points1_.row(data);
	mc2 = points2_.row(data);
	m1 = mc1;
	m2 = mc2;

//	const CvPoint2D64f* M = (const CvPoint2D64f*)m1.data.ptr;
//    const CvPoint2D64f* m = (const CvPoint2D64f*)m2.data.ptr;
    const cv::Point2d M = cv::Point2d(m1.reshape(1));
    const cv::Point2d m = cv::Point2d(m2.reshape(1));
	cv::Mat model_ = model;
	const double* H = (double*)model_.data;

    double ww = 1./(H[6]*M.x + H[7]*M.y + 1.);
    double dx = (H[0]*M.x + H[1]*M.y + H[2])*ww - m.x;
    double dy = (H[3]*M.x + H[4]*M.y + H[5])*ww - m.y;

	return dx*dx + dy*dy;
}

/* Homography estimation kernel function (extracted from the OpenCV).
 *
 * cv::Mat* m1							Input  -> Pointer to the corresponding left image projections
 * cv::Mat* m2							Input  -> Pointer to the corresponding right image projections
 * cv::Mat* H								Output -> Estimated homography
 *
 * Return value:						true:	Everything ok
 *										false:	Estimation failed
 */
bool runHomogrophyKernel( const cv::Mat* m1, const cv::Mat* m2, cv::Mat* H )
{
    int i, count = m1->rows*m1->cols;
//    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
//    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
    std::vector<cv::Point2d> M = (std::vector<cv::Point2d>)(m1->reshape(2));
    std::vector<cv::Point2d> m = (std::vector<cv::Point2d>)(m2->reshape(2));

    double LtL[9][9], W[9][1], V[9][9];
    cv::Mat _LtL = cv::Mat( 9, 9, CV_64F, LtL );
    cv::Mat matW = cv::Mat( 9, 1, CV_64F, W );
    cv::Mat matV = cv::Mat( 9, 9, CV_64F, V );
    cv::Mat _H0 = cv::Mat( 3, 3, CV_64F, V[8] );
    cv::Mat _Htemp = cv::Mat( 3, 3, CV_64F, V[7] );
    cv::Point2d cM={0,0}, cm={0,0}, sM={0,0}, sm={0,0};

    for( i = 0; i < count; i++ )
    {
        cm.x += m[i].x; cm.y += m[i].y;
        cM.x += M[i].x; cM.y += M[i].y;
    }

    cm.x /= count; cm.y /= count;
    cM.x /= count; cM.y /= count;

    for( i = 0; i < count; i++ )
    {
        sm.x += fabs(m[i].x - cm.x);
        sm.y += fabs(m[i].y - cm.y);
        sM.x += fabs(M[i].x - cM.x);
        sM.y += fabs(M[i].y - cM.y);
    }

    if( fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
        fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON )
        return false;
    sm.x = count/sm.x; sm.y = count/sm.y;
    sM.x = count/sM.x; sM.y = count/sM.y;

    double invHnorm[9] = { 1./sm.x, 0, cm.x, 0, 1./sm.y, cm.y, 0, 0, 1 };
    double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };
    cv::Mat _invHnorm = cv::Mat( 3, 3, CV_64FC1, invHnorm );
    cv::Mat _Hnorm2 = cv::Mat( 3, 3, CV_64FC1, Hnorm2 );

//    cvZero( &_LtL );
    _LtL.setTo(cv::Scalar::all(0));
    for( i = 0; i < count; i++ )
    {
        double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
        double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
        double Lx[] = { X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x };
        double Ly[] = { 0, 0, 0, X, Y, 1, -y*X, -y*Y, -y };
        int j, k;
        for( j = 0; j < 9; j++ )
            for( k = j; k < 9; k++ )
                LtL[j][k] += Lx[j]*Lx[k] + Ly[j]*Ly[k];
    }
    cv::completeSymm( _LtL );

    //cvSVD( &_LtL, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T );
    cv::eigen( _LtL, matW, matV );
    _Htemp = _invHnorm * _H0;
    _H0 = _Htemp * _Hnorm2;
    _H0.convertTo(_H0, CV_64FC1, 1./_H0.at<double>(2,2));
    _H0.copyTo(*H);
//    cvMatMul( &_invHnorm, &_H0, &_Htemp );
//    cvMatMul( &_Htemp, &_Hnorm2, &_H0 );
//    cvConvertScale( &_H0, H, 1./_H0.data.db[8] );

    return true;
}

/* Calculation of reprojection errors for a given set of correspondences and a homography.
 *
 * cv::Mat* m1							Input  -> Pointer to the corresponding left image projections
 * cv::Mat* m2							Input  -> Pointer to the corresponding right image projections
 * cv::Mat* model							Input  -> Estimated homography
 * vector<double> err					Output -> Vector of reprojection errors
 *
 *
 * Return value:						none
 */
void computeHomographyReprojError(const cv::Mat* m1, const cv::Mat* m2, const cv::Mat* model, std::vector<double> & err)
{
    int i, count = m1->rows*m1->cols;
    vector<cv::Point2d> M = (vector<cv::Point2d>)(m1->reshape(2));
    vector<cv::Point2d> m = (vector<cv::Point2d>)(m2->reshape(2));
//    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
//    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
    const double* H = (double*)model->data;

    for( i = 0; i < count; i++ )
    {
        double ww = 1./(H[6]*M[i].x + H[7]*M[i].y + 1.);
        double dx = (H[0]*M[i].x + H[1]*M[i].y + H[2])*ww - m[i].x;
        double dy = (H[3]*M[i].x + H[4]*M[i].y + H[5])*ww - m[i].y;
        err.push_back(dx*dx + dy*dy);
    }
}

/* Finds inliers to a given homography.
 *
 * InputArray p1						Input  -> Corresponding left image projections
 * cv::Mat* m2							Input  -> Corresponding right image projections
 * cv::Mat* model							Input  -> Estimated homography
 * double th							Input  -> Inlier/Outlier treshold
 * OutputArray mask						Output -> Correspondence mask marking inliers/ouliers
 *
 *
 * Return value:						none
 */
void findHomographyInliers(cv::InputArray p1, cv::InputArray p2, cv::InputArray H, double th, cv::OutputArray mask)
{
	Mat p1_ = p1.getMat();
	Mat p2_ = p2.getMat();
	Mat H_ = H.getMat();

	CV_Assert((p1_.cols == 1) && (p1_.cols == p2_.cols) &&
			  (p1_.rows == p2_.rows) && (p1_.type() == CV_64FC2) &&
			  (p2_.type() == CV_64FC2));
	CV_Assert((H_.cols == 3) && (H_.rows == 3) && (H_.type() == CV_64FC1));

	cv::Mat m1 = p1_, m2 = p2_, model = H_;
	vector<double> err;

	computeHomographyReprojError(&m1, &m2, &model, err);

	if(mask.needed())
	{
		double th2 = th * th;
		Mat mask_;
		mask.create(1, p1_.rows, CV_8U);
		mask_ = mask.getMat();
		mask_ = Mat::ones(1,p1_.rows, CV_8U);

		for(size_t i = 0; i < err.size(); i++)
			if(err[i] > th2)
				mask_.at<bool>(i) = false;
	}
}

/* Refines a given homography (extracted from the OpenCV).
 *
 * cv::Mat* m1							Input  -> Pointer to the corresponding left image projections
 * cv::Mat* m2							Input  -> Pointer to the corresponding right image projections
 * cv::Mat* model							Input/Output -> Homography
 * int maxIters							Input  -> Maximum number of iterations within Levenberg-Marquardt
 *
 * Return value:						true:	Everything ok
 *										false:	Refinement failed
 */
bool refineHomography( const cv::Mat* m1, const cv::Mat* m2, cv::Mat* model, int maxIters )
{
    CvLevMarq solver(8, 0, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, maxIters, DBL_EPSILON));
//    CvLevMarq solver(8, 0, cv::TermCriteria(cv::TermCriteria::Type::COUNT + cv::TermCriteria::Type::EPS, maxIters, DBL_EPSILON));
    //TERMCRIT_ITER+CV_TERMCRIT_EPS
    int i, j, k, count = m1->rows*m1->cols;
    std::vector<cv::Point2d> M = (const std::vector<cv::Point2d>)(m1->reshape(2));
    std::vector<cv::Point2d> m = (const std::vector<cv::Point2d>)(m2->reshape(2));
//    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data;
//    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data;
//    cv::Mat modelPart = cv::Mat( solver.param->rows, solver.param->cols, model->type(), model->data );
//    cv::Mat tmp1 = modelPart.clone();
//    double tmp1_d[9];
//    CvMat tmp = cvMat( solver.param->rows, solver.param->cols, model->type(), tmp1_d);
//    memcpy((void *)tmp.data.db, (void *)(model->data), sizeof(double) * 9);
    memcpy((void *)solver.param->data.db, (void *)(model->data), sizeof(double) * 8);
//    solver.param = &tmp;
//    cvCopy( &modelPart, solver.param );

    for(;;)
    {
        const CvMat* _param = 0;
        CvMat *_JtJ = 0, *_JtErr = 0;
        double* _errNorm = 0;

        if( !solver.updateAlt( _param, _JtJ, _JtErr, _errNorm ))
            break;

        for( i = 0; i < count; i++ )
        {
            const double* h = _param->data.db;
            double Mx = M[i].x, My = M[i].y;
            double ww = h[6]*Mx + h[7]*My + 1.;
            ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
            double _xi = (h[0]*Mx + h[1]*My + h[2])*ww;
            double _yi = (h[3]*Mx + h[4]*My + h[5])*ww;
            double err[] = { _xi - m[i].x, _yi - m[i].y };
            if( _JtJ || _JtErr )
            {
                double J[][8] =
                {
                    { Mx*ww, My*ww, ww, 0, 0, 0, -Mx*ww*_xi, -My*ww*_xi },
                    { 0, 0, 0, Mx*ww, My*ww, ww, -Mx*ww*_yi, -My*ww*_yi }
                };

                for( j = 0; j < 8; j++ )
                {
                    for( k = j; k < 8; k++ )
                        _JtJ->data.db[j*8+k] += J[0][j]*J[0][k] + J[1][j]*J[1][k];
                    _JtErr->data.db[j] += J[0][j]*err[0] + J[1][j]*err[1];
                }
            }
            if( _errNorm )
                *_errNorm += err[0]*err[0] + err[1]*err[1];
        }
    }

//    cvCopy( solver.param, &modelPart );
    memcpy((void *)model->data, (void *)(solver.param->data.db), sizeof(double) * 8);
    return true;
}

}
