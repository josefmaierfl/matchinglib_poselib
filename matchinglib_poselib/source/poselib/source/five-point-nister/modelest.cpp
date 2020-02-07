/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "five-point-nister/precomp.hpp"
#include "five-point-nister/_modelest.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <iostream>

using namespace std;


CvModelEstimator3::CvModelEstimator3(int _modelPoints, cv::Size _modelSize, int _maxBasicSolutions)
{
    modelPoints = _modelPoints;
    modelSize = _modelSize;
    maxBasicSolutions = _maxBasicSolutions;
    checkPartialSubsets = true;
    std::srand(std::time(nullptr));
}

CvModelEstimator3::~CvModelEstimator3()=default;

void CvModelEstimator3::setSeed( int64 seed )
{
    std::srand(seed);
}


int CvModelEstimator3::findInliers( const cv::Mat &m1, const cv::Mat &m2,
                                    const cv::Mat &model, cv::Mat &_err,
                                    cv::Mat &_mask, double threshold )
{
    int i, count = _err.rows*_err.cols, goodCount = 0;
    auto err = (const float*)_err.data;
    auto mask = (uchar*)_mask.data;

    computeReprojError3( m1, m2, model, _err );

	threshold *= threshold;
    for( i = 0; i < count; i++ )
        goodCount += mask[i] = err[i] <= threshold;
    return goodCount;
}


CV_IMPL int
cvRANSACUpdateNumIters1( double p, double ep,
                        int model_points, int max_iters )
{
    if( model_points <= 0 )
        CV_Error( CV_StsOutOfRange, "the number of model points should be positive" );

    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    double num = MAX(1. - p, DBL_MIN);
    double denom = 1. - pow(1. - ep,model_points);
    if( denom < DBL_MIN )
        return 0;

    num = log(num);
    denom = log(denom);

    return denom >= 0 || -num >= (double)max_iters*(-denom) ?
        max_iters : (int)std::round(num/denom);
}

bool EssentialMatEstimatorTheia::EstimateModel(const std::vector<size_t>& data,
										  std::vector<cv::Mat>* model) const
{
	CV_Assert((points1_.rows == 1) && (points1_.rows == points2_.rows) &&
			  (points1_.cols == points2_.cols));
	//cv::Ptr<CvMat> models;
	cv::Mat models;
	cv::Mat m1(points1_.rows,(int)data.size(), points1_.type());
	cv::Mat m2(points2_.rows,(int)data.size(), points2_.type());
	cv::Mat cvm1;
	cv::Mat cvm2;

	for(int i = 0; i < (int)data.size(); i++)
	{
		points1_.col(data[i]).copyTo(m1.col(i));
		points2_.col(data[i]).copyTo(m2.col(i));
	}

	models = cv::Mat( modelEstimator_->getModelSize().height*modelEstimator_->getMaxBasicSolutions(),
	        modelEstimator_->getModelSize().width, CV_64FC1 );

	int nmodels = modelEstimator_->runKernel( m1, m2, models );
    if( nmodels <= 0 )
        return false;

	for(int i = 0; i < nmodels; i++ )
    {
        cv::Mat model_i;
        model_i = models.rowRange(i*modelEstimator_->getModelSize().height, (i+1)*modelEstimator_->getModelSize().height);
		if(!modelEstimator_->ValidModel(m1, m2, model_i))
			continue;
		model->emplace_back(model_i.clone()); //Hopefully there is no memory-problem with that
	}

	if(model->size() == 0)
		return false;

	return true;
}


bool EssentialMatEstimatorTheia::EstimateModelNonminimal(const std::vector<size_t>& data,
													std::vector<cv::Mat>* model) const
{
	CV_Assert((points1_.rows == 1) && (points1_.rows == points2_.rows) &&
			  (points1_.cols == points2_.cols));
	cv::Mat m1(points1_.rows,(int)data.size(), points1_.type());
	cv::Mat m2(points2_.rows,(int)data.size(), points2_.type());
	cv::Mat E;

	for(int i = 0; i < (int)data.size(); i++)
	{
		points1_.col(data[i]).copyTo(m1.col(i));
		points2_.col(data[i]).copyTo(m2.col(i));
	}
	m1.reshape(1);
	m2.reshape(1);

	E = cv::findFundamentalMat(m1, m2, cv::FM_8POINT);
	if(E.empty())
		return false;

	if(!modelEstimator_->ValidModel(m1, m2, E))
		return false;

	model->emplace_back(E.clone());

	return true;
}


double EssentialMatEstimatorTheia::Error(const size_t& data, const cv::Mat& model) const
{
	cv::Mat m1, m2;
	cv::Mat mc1, mc2;
	cv::Mat _err = cv::Mat(1,1,CV_32F);
	mc1 = points1_.col((int)data);
	mc2 = points2_.col((int)data);
	m1 = mc1;
	m2 = mc2;

	modelEstimator_->computeReprojError3( mc1, mc2, model, _err );

	return (double)_err.at<float>(0, 0);
}

bool CvModelEstimator3::runARRSAC( const cv::Mat &m1, const cv::Mat &m2, cv::Mat &model,
								   cv::Mat &mask0, double reprojThreshold,
								   bool lesqu,
								   void (*refineEssential)(cv::InputArray points1, cv::InputArray points2, cv::InputArray E_init,
														   cv::Mat & E_refined, double th, unsigned int iters, bool makeClosestE,
														   double *sumSqrErr_init, double *sumSqrErr,
														   cv::OutputArray errors, cv::InputOutputArray mask, int model, bool tryOrientedEpipolar, bool normalizeCorrs))
{
	bool result = false;
	int goodCount = 0;
	cv::Mat err;
    cv::Mat mask = mask0.clone();

    int count = m1.rows*m1.cols, maxGoodCount = 0;
    CV_Assert( m1.size() == m2.size() && m1.cols == mask.cols );

	err = cv::Mat( 1, count, CV_32FC1 );

    if( count < modelPoints )
        return false;

    if( count == modelPoints )
    {
		cv::Mat models, tmask;
        cv::Mat ms1 = m1.clone();
        cv::Mat ms2 = m2.clone();

		models = cv::Mat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
		tmask = cv::Mat( 1, count, CV_8UC1 );

		int nmodels = runKernel( ms1, ms2, models );
		if( nmodels <= 0 )
		{
			return false;
		}
		else
		{
			auto errminsum = DBL_MAX;
			for(int i = 0; i < nmodels; i++ )
			{
				cv::Mat model_i;
				model_i = models.rowRange(i*modelSize.height, (i+1)*modelSize.height);
				goodCount = findInliers( m1, m2, model_i, err, tmask, reprojThreshold );

				if( goodCount > MAX(maxGoodCount, modelPoints-1) )
				{
					std::swap(tmask, mask);
                    model_i.copyTo(model);
					maxGoodCount = goodCount;
					//errminsum = cv::sum(cv::Mat(err)).val[0];
					errminsum = cv::sum(err).val[0];
					result = true;
				}
				else if( (goodCount == MAX(maxGoodCount, modelPoints)) && (errminsum < DBL_MAX)
						&& (errminsum > cv::sum(err).val[0]) )
				{
					std::swap(tmask, mask);
                    model_i.copyTo(model);
					errminsum = cv::sum(err).val[0];
				}
			}
		}
		if(result)
		    mask.copyTo(mask0);
		return result;
    }

	std::vector<size_t> input_data;
	for(size_t i = 0; i < (size_t)m1.cols; i++)
		input_data.push_back(i);

	cv::Mat bestmodel = model.clone();
	EssentialMatEstimatorTheia esti(this, m1, m2);
	theia::Arrsac<size_t,cv::Mat> arrsac_estimator(5, reprojThreshold * reprojThreshold, 500, 100, 14, 8);
	result = arrsac_estimator.Estimate(input_data, esti, &bestmodel);
	if(result)
	{
		goodCount = findInliers( m1, m2, bestmodel, err, mask, reprojThreshold );
		if(((goodCount < 50) && (input_data.size() > 200)) || (goodCount < 15))
		{
			result = false;
		}

		if(lesqu && result)
		{
			auto maskp = (uchar*)mask.data;
			cv::Mat inl_points1(1, goodCount, m1.type());
			cv::Mat inl_points2(1, goodCount, m1.type());
			int j = 0;
			for(int i = 0; i < mask.cols; i++)
			{
				if(maskp[i])
				{
                    m1.col(i).copyTo(inl_points1.col(j));
					m2.col(i).copyTo(inl_points2.col(j));
					j++;
				}
			}
			if(refineEssential)
			{
				cv::Mat E_refined;
				double err_i = 999.0, err_f = 999.0;
				int goodCount_tmp;
				cv::Mat mask_tmp = mask.clone();

				//cvCopy( mask_tmp, mask );

				if(inl_points1.cols < 50)
				{
					mask.copyTo( mask0 );
					bestmodel.copyTo( model);
					return result;
				}
				inl_points1.convertTo(inl_points1,CV_64FC2);
				inl_points2.convertTo(inl_points2,CV_64FC2);
				refineEssential(inl_points1,inl_points2,bestmodel,E_refined, reprojThreshold/50.0,0,true,&err_i,&err_f,cv::noArray(),cv::noArray(),0,false,false);

				goodCount_tmp = findInliers( m1, m2, bestmodel, err, mask_tmp, reprojThreshold );
				if(((float)goodCount_tmp / (float)goodCount > 0.66f) || (err_i > err_f))
				{
					bestmodel = E_refined.clone();
					goodCount = goodCount_tmp;
					mask.copyTo( mask_tmp );
				}
				else
				{
					cv::Mat model_ls = cv::findFundamentalMat(inl_points1, inl_points2, cv::FM_8POINT);
					bestmodel = model_ls.clone();
					goodCount = findInliers( m1, m2, bestmodel, err, mask, reprojThreshold );
				}
			}
			else
			{
				//cvleastsquares:
				cv::Mat model_ls = cv::findFundamentalMat(inl_points1, inl_points2, cv::FM_8POINT);
                bestmodel = model_ls.clone();
				goodCount = findInliers( m1, m2, bestmodel, err, mask, reprojThreshold );
			}
		}
		mask.copyTo( mask0 );
		bestmodel.copyTo(model);
	}

	return result;
}

bool CvModelEstimator3::runRANSAC( const cv::Mat &m1, const cv::Mat &m2, cv::Mat &model,
                                    cv::Mat &mask0, double reprojThreshold,
                                    double confidence, int maxIters, bool lesqu )
{
    bool result = false;
    cv::Mat mask = mask0.clone();
    cv::Mat models, err, tmask;
    cv::Mat ms1, ms2;

    int iter, niters = maxIters;
    int count = m1.rows*m1.cols, maxGoodCount = 0;
    CV_Assert( m1.size() == m2.size() && m1.cols == mask.cols );

    if( count < modelPoints )
        return false;

    models = cv::Mat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
    err = cv::Mat( 1, count, CV_32FC1 );
    tmask = cv::Mat( 1, count, CV_8UC1 );

	auto errminsum = DBL_MAX;

    if( count > modelPoints )
    {
        ms1 = cv::Mat( 1, modelPoints, m1.type() );
        ms2 = cv::Mat( 1, modelPoints, m2.type() );
    }
    else
    {
        niters = 1;
        ms1 = m1.clone();
        ms2 = m2.clone();
    }

    for( iter = 0; iter < niters; iter++ )
    {
        int i, goodCount, nmodels;
        if( count > modelPoints )
        {
            bool found = getSubset( m1, m2, ms1, ms2, 300 );
            if( !found )
            {
                if( iter == 0 )
                    return false;
                break;
            }
        }

        nmodels = runKernel( ms1, ms2, models );
        if( nmodels <= 0 )
            continue;
        for( i = 0; i < nmodels; i++ )
        {
            cv::Mat model_i;
            model_i = models.rowRange(i*modelSize.height, (i+1)*modelSize.height);
            goodCount = findInliers( m1, m2, model_i, err, tmask, reprojThreshold );

            if( goodCount > MAX(maxGoodCount, modelPoints-1) )
            {
                std::swap(tmask, mask);
                model_i.copyTo( model );
                maxGoodCount = goodCount;
                niters = cvRANSACUpdateNumIters1( confidence,
                    (double)(count - goodCount)/count, modelPoints, niters );
				errminsum = cv::sum(err).val[0];
            }else if( (goodCount == MAX(maxGoodCount, modelPoints)) && (errminsum < DBL_MAX)
					&& (errminsum > cv::sum(err).val[0]) )
			{
				std::swap(tmask, mask);
                model_i.copyTo( model );
				errminsum = cv::sum(err).val[0];
			}
        }
    }


	//Calculate the least squares solution with all inliers
	if(lesqu && (maxGoodCount > 0))
	{
		cv::Mat mask1 = mask;
		mask1.convertTo(mask1,CV_8U);

		cv::Mat m11 = m1, m21 = m2,ms11, ms21;
		m11.reshape(1,2);
		m21.reshape(1,2);

		for(int i = 0;i<count;i++)
		{
			if(mask1.at<bool>(i))
			{
				ms11.push_back(m11.col(i));
				ms21.push_back(m21.col(i));
			}
		}
		ms11 = ms11.t();
		ms21 = ms21.t();

		int i, goodCount, nmodels;
		nmodels = runKernel( ms11, ms21, models );
		if( nmodels <= 0 )
			return result;
		for( i = 0; i < nmodels; i++ )
		{
			cv::Mat model_i;
			model_i = models.rowRange(i*modelSize.height, (i+1)*modelSize.height);
			goodCount = findInliers( m1, m2, model_i, err, tmask, reprojThreshold );

			if( goodCount > MAX(maxGoodCount, modelPoints-1) )
			{
				std::swap(tmask, mask);
				model_i.copyTo( model );
				maxGoodCount = goodCount;
				errminsum = cv::sum(err).val[0];
			}
			else if( (goodCount == maxGoodCount) && (errminsum < DBL_MAX) && (errminsum > cv::sum(err).val[0]) )
			{
				std::swap(tmask, mask);
                model_i.copyTo( model );
				errminsum = cv::sum(err).val[0];
			}
		}
	}

	if( maxGoodCount > 0 )
    {
        if( sum(mask != mask0) != cv::Scalar(0,0,0,0) )
            mask.copyTo( mask0 );
        result = true;
    }

    return result;
}


//static CV_IMPLEMENT_QSORT( icvSortDistances, int, CV_LT )
static void icvSortDistances( int *array, size_t total, int /*unused*/ )
{
   std::sort( &array[0], &array[total] );
}

bool CvModelEstimator3::runLMeDS( const cv::Mat &m1, const cv::Mat &m2, cv::Mat &model,
                                  cv::Mat &mask, double confidence, int maxIters )
{
    const double outlierRatio = 0.45;
    bool result = false;
    cv::Mat models;
    cv::Mat ms1, ms2;
    cv::Mat err;

    int iter, niters = maxIters;
    int count = m1.rows*m1.cols;
    double minMedian = DBL_MAX, sigma;

    CV_Assert( m1.size() == m2.size() && m1.cols == mask.cols );

    if( count < modelPoints )
        return false;

    models = cv::Mat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
    err = cv::Mat( 1, count, CV_32FC1 );

    if( count > modelPoints )
    {
        ms1 = cv::Mat( 1, modelPoints, m1.type() );
        ms2 = cv::Mat( 1, modelPoints, m2.type() );
    }
    else
    {
        niters = 1;
        ms1 = m1.clone();
        ms2 = m2.clone();
    }

    niters = (int)std::round(log(1. - confidence)/log(1. - pow(1. - outlierRatio,(double)modelPoints)));
    niters = MIN( MAX(niters, 3), maxIters );

    for( iter = 0; iter < niters; iter++ )
    {
        int i, nmodels;
        if( count > modelPoints )
        {
            bool found = getSubset( m1, m2, ms1, ms2, 300 );
            if( !found )
            {
                if( iter == 0 )
                    return false;
                break;
            }
        }

        nmodels = runKernel( ms1, ms2, models );
        if( nmodels <= 0 )
            continue;
        for( i = 0; i < nmodels; i++ )
        {
            cv::Mat model_i;
            model_i = models.rowRange(i*modelSize.height, (i+1)*modelSize.height);
            computeReprojError3( m1, m2, model_i, err );
            icvSortDistances( (int*)err.data, count, 0 );

            double median = count % 2 != 0 ?
                            (double)err.at<float>(count/2) : (double)(err.at<float>(count/2-1) + err.at<float>(count/2))*0.5;

            if( median < minMedian )
            {
                minMedian = median;
                model_i.copyTo( model );
            }
        }
    }

    if( minMedian < DBL_MAX )
    {
        sigma = 2.5*1.4826*(1 + 5./(count - modelPoints))*sqrt(minMedian);
        sigma = MAX( sigma, 0.001 );

        count = findInliers( m1, m2, model, err, mask, sigma );
        result = count >= modelPoints;
    }

    return result;
}


bool CvModelEstimator3::getSubset( const cv::Mat &m1, const cv::Mat &m2,
                                   cv::Mat &ms1, cv::Mat &ms2, int maxAttempts )
{
    cv::AutoBuffer<int> _idx(modelPoints);
    int* idx = _idx;
    int i = 0, j, k, idx_i, iters = 0;
    int type = CV_MAT_TYPE(m1.type()), elemSize = CV_ELEM_SIZE(type);
    const int *m1ptr = (const int*)m1.data, *m2ptr = (const int*)m2.data;
    int *ms1ptr = (int*)ms1.data, *ms2ptr = (int*)ms2.data;
    int count = m1.cols*m1.rows;

    assert( m1.isContinuous() && m2.isContinuous() && (elemSize % sizeof(int) == 0) );
    elemSize /= sizeof(int);

    for(; iters < maxAttempts; iters++)
    {
        for( i = 0; i < modelPoints && iters < maxAttempts; )
        {
            idx[i] = idx_i = std::rand() % count;
            for( j = 0; j < i; j++ )
                if( idx_i == idx[j] )
                    break;
            if( j < i )
                continue;
            for( k = 0; k < elemSize; k++ )
            {
                ms1ptr[i*elemSize + k] = m1ptr[idx_i*elemSize + k];
                ms2ptr[i*elemSize + k] = m2ptr[idx_i*elemSize + k];
            }
            if( checkPartialSubsets && (!checkSubset( ms1, i+1 ) || !checkSubset( ms2, i+1 )))
            {
                iters++;
                continue;
            }
            i++;
        }
        if( !checkPartialSubsets && i == modelPoints &&
            (!checkSubset( ms1, i ) || !checkSubset( ms2, i )))
            continue;
        break;
    }

    return i == modelPoints && iters < maxAttempts;
}


bool CvModelEstimator3::checkSubset( const cv::Mat &m, int count )
{
    int j, k, i, i0, i1;
    std::vector<cv::Point2d> ptr;
    ptr = (std::vector<cv::Point2d>)(m.reshape(2));
//    CvPoint2D64f* ptr = (CvPoint2D64f*)m->data.ptr;

    assert( CV_MAT_TYPE(m.type()) == CV_64FC2 );

    if( checkPartialSubsets )
        i0 = i1 = count - 1;
    else
        i0 = 0, i1 = count - 1;

    for( i = i0; i <= i1; i++ )
    {
        // check that the i-th selected point does not belong
        // to a line connecting some previously selected points
        for( j = 0; j < i; j++ )
        {
            double dx1 = ptr[j].x - ptr[i].x;
            double dy1 = ptr[j].y - ptr[i].y;
            for( k = 0; k < j; k++ )
            {
                double dx2 = ptr[k].x - ptr[i].x;
                double dy2 = ptr[k].y - ptr[i].y;
                if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                    break;
            }
            if( k < j )
                break;
        }
        if( j < i )
            break;
    }

    return i >= i1;
}
