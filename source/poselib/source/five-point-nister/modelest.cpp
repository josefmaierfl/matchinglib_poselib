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

#include "five-point-nister\precomp.hpp"
#include "five-point-nister\_modelest.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <iostream>

using namespace std;


CvModelEstimator3::CvModelEstimator3(int _modelPoints, CvSize _modelSize, int _maxBasicSolutions)
{
    modelPoints = _modelPoints;
    modelSize = _modelSize;
    maxBasicSolutions = _maxBasicSolutions;
    checkPartialSubsets = true;
    rng = cvRNG(-1);
}

CvModelEstimator3::~CvModelEstimator3()
{
}

void CvModelEstimator3::setSeed( int64 seed )
{
    rng = cvRNG(seed);
}


int CvModelEstimator3::findInliers( const CvMat* m1, const CvMat* m2,
                                    const CvMat* model, CvMat* _err,
                                    CvMat* _mask, double threshold )
{
    int i, count = _err->rows*_err->cols, goodCount = 0;
    const float* err = _err->data.fl;
    uchar* mask = _mask->data.ptr;

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

    return denom >= 0 || -num >= max_iters*(-denom) ?
        max_iters : cvRound(num/denom);
}

bool EssentialMatEstimatorTheia::EstimateModel(const std::vector<size_t>& data,
										  std::vector<CvMat>* model) const
{
	CV_Assert((points1_.rows == 1) && (points1_.rows == points2_.rows) &&
			  (points1_.cols == points2_.cols));
	//cv::Ptr<CvMat> models;
	CvMat* models;
	cv::Mat m1(points1_.rows,(int)data.size(), points1_.type());
	cv::Mat m2(points2_.rows,(int)data.size(), points2_.type());
	CvMat cvm1;
	CvMat cvm2;

	for(int i = 0; i < (int)data.size(); i++)
	{
		points1_.col(data[i]).copyTo(m1.col(i));
		points2_.col(data[i]).copyTo(m2.col(i));
	}
	cvm1 = m1;
	cvm2 = m2;

	models = cvCreateMat( modelEstimator_->getModelSize().height*modelEstimator_->getMaxBasicSolutions(),
						  modelEstimator_->getModelSize().width, CV_64FC1 );

	int nmodels = modelEstimator_->runKernel( &cvm1, &cvm2, models );
    if( nmodels <= 0 )
        return false;

	for(int i = 0; i < nmodels; i++ )
    {
        CvMat model_i;
        cvGetRows( models, &model_i, i*modelEstimator_->getModelSize().height,
				  (i+1)*modelEstimator_->getModelSize().height );
		if(!modelEstimator_->ValidModel(&cvm1, &cvm2, &model_i))
			continue;
		model->push_back(*cvCloneMat(&model_i)); //Hopefully there is no memory-problem with that
	}
	cvReleaseMat(&models);

	if(model->size() == 0)
		return false;

	return true;
}


bool EssentialMatEstimatorTheia::EstimateModelNonminimal(const std::vector<size_t>& data,
													std::vector<CvMat>* model) const
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

	E = cv::findFundamentalMat(m1, m2, CV_FM_8POINT);
	if(E.empty())
		return false;

	CvMat E1 = E, p1 = m1, p2 = m2;
	if(!modelEstimator_->ValidModel(&p1, &p2, &E1))
		return false;

	model->push_back(*cvCloneMat(&E1));

	return true;
}


double EssentialMatEstimatorTheia::Error(const size_t& data, const CvMat& model) const
{
	CvMat m1, m2;
	cv::Mat mc1, mc2;
	cv::Ptr<CvMat> _err = cvCreateMat(1,1,CV_32F);
	mc1 = points1_.col((int)data);
	mc2 = points2_.col((int)data);
	m1 = mc1;
	m2 = mc2;

	modelEstimator_->computeReprojError3( &m1, &m2, &model, _err );

	return (double)_err->data.fl[0];
}

bool CvModelEstimator3::runARRSAC( const CvMat* m1, const CvMat* m2, CvMat* model,
								   CvMat* mask0, double reprojThreshold,
								   bool lesqu, 
								   void (*refineEssential)(cv::InputArray points1, cv::InputArray points2, cv::InputArray E_init, 
														   cv::Mat & E_refined, double th, unsigned int iters, bool makeClosestE, 
														   double *sumSqrErr_init, double *sumSqrErr, 
														   cv::OutputArray errors, cv::InputOutputArray mask, int model, bool tryOrientedEpipolar, bool normalizeCorrs))
{
	bool result = false;
	int goodCount = 0;
	cv::Ptr<CvMat> err;
    cv::Ptr<CvMat> mask = cvCloneMat(mask0);

    int count = m1->rows*m1->cols, maxGoodCount = 0;
    CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );

	err = cvCreateMat( 1, count, CV_32FC1 );

    if( count < modelPoints )
        return false;
    
    if( count == modelPoints )
    {
		cv::Ptr<CvMat> models, tmask;
        cv::Ptr<CvMat> ms1 = cvCloneMat(m1);
        cv::Ptr<CvMat> ms2 = cvCloneMat(m2);

		models = cvCreateMat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
		tmask = cvCreateMat( 1, count, CV_8UC1 );

		int nmodels = runKernel( ms1, ms2, models );
		if( nmodels <= 0 )
		{
			return false;
		}
		else
		{
			double errminsum = DBL_MAX;
			for(int i = 0; i < nmodels; i++ )
			{
				CvMat model_i;
				cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );
				goodCount = findInliers( m1, m2, &model_i, err, tmask, reprojThreshold );

				cv::Mat err_tmp = cvarrToMat(err);
				if( goodCount > MAX(maxGoodCount, modelPoints-1) )
				{
					std::swap(tmask, mask);
					cvCopy( &model_i, model );
					maxGoodCount = goodCount;
					//errminsum = cv::sum(cv::Mat(err)).val[0];
					errminsum = cv::sum(err_tmp).val[0];
					result = true;
				}
				else if( (goodCount == MAX(maxGoodCount, modelPoints)) && (errminsum < DBL_MAX) 
						&& (errminsum > cv::sum(err_tmp).val[0]) )
				{
					std::swap(tmask, mask);
					cvCopy( &model_i, model );
					errminsum = cv::sum(err_tmp).val[0];
				}
			}
		}
		if(result)
			cvCopy( mask, mask0 );
		return result;
    }

	std::vector<size_t> input_data;
	for(size_t i = 0; i < (size_t)m1->cols; i++)
		input_data.push_back(i);
	
	cv::Ptr<CvMat> bestmodel = cvCloneMat(model);
	cv::Mat m1_tmp = cv::cvarrToMat(m1);
	cv::Mat m2_tmp = cv::cvarrToMat(m2);
	EssentialMatEstimatorTheia esti(this, m1_tmp, m2_tmp);
	theia::Arrsac<size_t,CvMat> arrsac_estimator(5, reprojThreshold * reprojThreshold, 500, 100, 14, 8);
	result = arrsac_estimator.Estimate(input_data, esti, bestmodel);
	if(result)
	{
		goodCount = findInliers( m1, m2, bestmodel, err, mask, reprojThreshold );
		if(((goodCount < 50) && (input_data.size() > 200)) || (goodCount < 15))
		{
			result = false;
		}

		if(lesqu && result)
		{
			uchar* maskp = mask->data.ptr;
			cv::Mat inl_points1(1, goodCount, m1->type);
			cv::Mat inl_points2(1, goodCount, m1->type);
			cv::Mat tmp1 = m1_tmp;
			cv::Mat tmp2 = m2_tmp;
			int j = 0;
			for(int i = 0; i < mask->cols; i++)
			{
				if(maskp[i])
				{
					tmp1.col(i).copyTo(inl_points1.col(j));
					tmp2.col(i).copyTo(inl_points2.col(j));
					j++;
				}
			}
			if(refineEssential)
			{
				cv::Mat E_refined;
				double err_i = 999.0, err_f = 999.0;
				int goodCount_tmp;
				cv::Ptr<CvMat> mask_tmp = cvCloneMat(mask);;

				//cvCopy( mask_tmp, mask );

				if(inl_points1.cols < 50)
				{
					cvCopy( mask, mask0 );
					cvCopy( bestmodel, model);
					return result;
				}
				inl_points1.convertTo(inl_points1,CV_64FC2);
				inl_points2.convertTo(inl_points2,CV_64FC2);
				cv::Mat bestmodel_tmp = cvarrToMat(bestmodel);
				refineEssential(inl_points1,inl_points2,bestmodel_tmp,E_refined, reprojThreshold/50.0,0,true,&err_i,&err_f,cv::noArray(),cv::noArray(),0,false,false);
				
				goodCount_tmp = findInliers( m1, m2, bestmodel, err, mask_tmp, reprojThreshold );
				if(((float)goodCount_tmp / (float)goodCount > 0.66f) || (err_i > err_f))
				{
					CvMat E_ref = E_refined;
					bestmodel = cvCloneMat(&E_ref);
					goodCount = goodCount_tmp;
					cvCopy( mask, mask_tmp );
				}
				else
				{
					cv::Mat model_ls = cv::findFundamentalMat(inl_points1, inl_points2, CV_FM_8POINT);
					CvMat model_ls2 = model_ls;
					bestmodel = cvCloneMat(&model_ls2);
					goodCount = findInliers( m1, m2, bestmodel, err, mask, reprojThreshold );
				}

				//if((err_i > err_f) || ((err_i < 0.1) && (err_f < 0.1)))
				//{
				//	//E_refined.copyTo(cv::Mat(bestmodel));
				//	CvMat E_ref = E_refined;
				//	bestmodel = cvCloneMat(&E_ref);
				//	//*bestmodel = E_refined;
				//	goodCount = findInliers( m1, m2, bestmodel, err, mask, reprojThreshold );
				//}
				//else 
				//{
				//	//goto cvleastsquares;
				//	cv::Mat model_ls = cv::findFundamentalMat(inl_points1, inl_points2, CV_FM_8POINT);
				//	CvMat model_ls2 = model_ls;
				//	bestmodel = cvCloneMat(&model_ls2);
				//	goodCount = findInliers( m1, m2, bestmodel, err, mask, reprojThreshold );
				//}
			}
			else
			{
				//cvleastsquares:
				cv::Mat model_ls = cv::findFundamentalMat(inl_points1, inl_points2, CV_FM_8POINT);
				CvMat model_ls2 = model_ls;
				bestmodel = cvCloneMat(&model_ls2);
				goodCount = findInliers( m1, m2, bestmodel, err, mask, reprojThreshold );
			}
		}
		cvCopy( mask, mask0 );
		cvCopy( bestmodel, model);
	}

	return result;
}

bool CvModelEstimator3::runRANSAC( const CvMat* m1, const CvMat* m2, CvMat* model,
                                    CvMat* mask0, double reprojThreshold,
                                    double confidence, int maxIters, bool lesqu )
{
    bool result = false;
    cv::Ptr<CvMat> mask = cvCloneMat(mask0);
    cv::Ptr<CvMat> models, err, tmask;
    cv::Ptr<CvMat> ms1, ms2;

    int iter, niters = maxIters;
    int count = m1->rows*m1->cols, maxGoodCount = 0;
    CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );

    if( count < modelPoints )
        return false;

    models = cvCreateMat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
    err = cvCreateMat( 1, count, CV_32FC1 );
    tmask = cvCreateMat( 1, count, CV_8UC1 );

	double errminsum = DBL_MAX;

    if( count > modelPoints )
    {
        ms1 = cvCreateMat( 1, modelPoints, m1->type );
        ms2 = cvCreateMat( 1, modelPoints, m2->type );
    }
    else
    {
        niters = 1;
        ms1 = cvCloneMat(m1);
        ms2 = cvCloneMat(m2);
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
            CvMat model_i;
            cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );
            goodCount = findInliers( m1, m2, &model_i, err, tmask, reprojThreshold );

			cv::Mat err_tmp = cvarrToMat(err);
            if( goodCount > MAX(maxGoodCount, modelPoints-1) )
            {
                std::swap(tmask, mask);
                cvCopy( &model_i, model );
                maxGoodCount = goodCount;
                niters = cvRANSACUpdateNumIters1( confidence,
                    (double)(count - goodCount)/count, modelPoints, niters );
				errminsum = cv::sum(err_tmp).val[0];
            }
			else if( (goodCount == MAX(maxGoodCount, modelPoints)) && (errminsum < DBL_MAX) 
					&& (errminsum > cv::sum(err_tmp).val[0]) )
			{
				std::swap(tmask, mask);
                cvCopy( &model_i, model );
				errminsum = cv::sum(err_tmp).val[0];
			}
        }
    }


	//Calculate the least squares solution with all inliers
	if(lesqu && (maxGoodCount > 0))
	{
		cv::Mat mask1 = cvarrToMat(mask);
		mask1.convertTo(mask1,CV_8U);

		cv::Mat m11 = cv::cvarrToMat(m1), m21 = cv::cvarrToMat(m2),ms11, ms21;
		m11.reshape(1,2);
		m21.reshape(1,2);

		for(int i = 0;i<count;i++)
		{
			if(mask1.at<bool>(i) == true)
			{
				ms11.push_back(m11.col(i));
				ms21.push_back(m21.col(i));
			}
		}
		ms11 = ms11.t();
		ms21 = ms21.t();
		CvMat ms12 = ms11;
		CvMat ms22 = ms21;
	

		int i, goodCount, nmodels;
		nmodels = runKernel( &ms12, &ms22, models );
		if( nmodels <= 0 )
			return result;
		for( i = 0; i < nmodels; i++ )
		{
			CvMat model_i;
			cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );
			goodCount = findInliers( m1, m2, &model_i, err, tmask, reprojThreshold );

			cv::Mat err_tmp = cvarrToMat(err);
			if( goodCount > MAX(maxGoodCount, modelPoints-1) )
			{
				std::swap(tmask, mask);
				cvCopy( &model_i, model );
				maxGoodCount = goodCount;
				errminsum = cv::sum(err_tmp).val[0];
			}
			else if( (goodCount == maxGoodCount) && (errminsum < DBL_MAX) && (errminsum > cv::sum(err_tmp).val[0]) )
			{
				std::swap(tmask, mask);
				cvCopy( &model_i, model );
				errminsum = cv::sum(err_tmp).val[0];
			}
		}
	}

	if( maxGoodCount > 0 )
    {
        if( mask != mask0 )
            cvCopy( mask, mask0 );
        result = true;
    }

    return result;
}


//static CV_IMPLEMENT_QSORT( icvSortDistances, int, CV_LT )
static void icvSortDistances( int *array, size_t total, int /*unused*/ )
{
   std::sort( &array[0], &array[total] );
}

bool CvModelEstimator3::runLMeDS( const CvMat* m1, const CvMat* m2, CvMat* model,
                                  CvMat* mask, double confidence, int maxIters )
{
    const double outlierRatio = 0.45;
    bool result = false;
    cv::Ptr<CvMat> models;
    cv::Ptr<CvMat> ms1, ms2;
    cv::Ptr<CvMat> err;

    int iter, niters = maxIters;
    int count = m1->rows*m1->cols;
    double minMedian = DBL_MAX, sigma;

    CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );

    if( count < modelPoints )
        return false;

    models = cvCreateMat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
    err = cvCreateMat( 1, count, CV_32FC1 );

    if( count > modelPoints )
    {
        ms1 = cvCreateMat( 1, modelPoints, m1->type );
        ms2 = cvCreateMat( 1, modelPoints, m2->type );
    }
    else
    {
        niters = 1;
        ms1 = cvCloneMat(m1);
        ms2 = cvCloneMat(m2);
    }

    niters = cvRound(log(1-confidence)/log(1-pow(1-outlierRatio,(double)modelPoints)));
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
            CvMat model_i;
            cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );
            computeReprojError3( m1, m2, &model_i, err );
            icvSortDistances( err->data.i, count, 0 );

            double median = count % 2 != 0 ?
                err->data.fl[count/2] : (err->data.fl[count/2-1] + err->data.fl[count/2])*0.5;

            if( median < minMedian )
            {
                minMedian = median;
                cvCopy( &model_i, model );
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


bool CvModelEstimator3::getSubset( const CvMat* m1, const CvMat* m2,
                                   CvMat* ms1, CvMat* ms2, int maxAttempts )
{
    cv::AutoBuffer<int> _idx(modelPoints);
    int* idx = _idx;
    int i = 0, j, k, idx_i, iters = 0;
    int type = CV_MAT_TYPE(m1->type), elemSize = CV_ELEM_SIZE(type);
    const int *m1ptr = m1->data.i, *m2ptr = m2->data.i;
    int *ms1ptr = ms1->data.i, *ms2ptr = ms2->data.i;
    int count = m1->cols*m1->rows;

    assert( CV_IS_MAT_CONT(m1->type & m2->type) && (elemSize % sizeof(int) == 0) );
    elemSize /= sizeof(int);

    for(; iters < maxAttempts; iters++)
    {
        for( i = 0; i < modelPoints && iters < maxAttempts; )
        {
            idx[i] = idx_i = cvRandInt(&rng) % count;
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


bool CvModelEstimator3::checkSubset( const CvMat* m, int count )
{
    int j, k, i, i0, i1;
    CvPoint2D64f* ptr = (CvPoint2D64f*)m->data.ptr;

    assert( CV_MAT_TYPE(m->type) == CV_64FC2 );

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


/*namespace cv
{

class Affine3DEstimator : public CvModelEstimator3
{
public:
    Affine3DEstimator() : CvModelEstimator3(4, cvSize(4, 3), 1) {}
    virtual int runKernel( const CvMat* m1, const CvMat* m2, CvMat* model );
protected:
    virtual void computeReprojError( const CvMat* m1, const CvMat* m2, const CvMat* model, CvMat* error );
    virtual bool checkSubset( const CvMat* ms1, int count );
};

}

int cv::Affine3DEstimator::runKernel( const CvMat* m1, const CvMat* m2, CvMat* model )
{
    const Point3d* from = reinterpret_cast<const Point3d*>(m1->data.ptr);
    const Point3d* to   = reinterpret_cast<const Point3d*>(m2->data.ptr);

    Mat A(12, 12, CV_64F);
    Mat B(12, 1, CV_64F);
    A = Scalar(0.0);

    for(int i = 0; i < modelPoints; ++i)
    {
        *B.ptr<Point3d>(3*i) = to[i];

        double *aptr = A.ptr<double>(3*i);
        for(int k = 0; k < 3; ++k)
        {
            aptr[3] = 1.0;
            *reinterpret_cast<Point3d*>(aptr) = from[i];
            aptr += 16;
        }
    }

    CvMat cvA = A;
    CvMat cvB = B;
    CvMat cvX;
    cvReshape(model, &cvX, 1, 12);
    cvSolve(&cvA, &cvB, &cvX, CV_SVD );

    return 1;
}

void cv::Affine3DEstimator::computeReprojError( const CvMat* m1, const CvMat* m2, const CvMat* model, CvMat* error )
{
    int count = m1->rows * m1->cols;
    const Point3d* from = reinterpret_cast<const Point3d*>(m1->data.ptr);
    const Point3d* to   = reinterpret_cast<const Point3d*>(m2->data.ptr);
    const double* F = model->data.db;
    float* err = error->data.fl;

    for(int i = 0; i < count; i++ )
    {
        const Point3d& f = from[i];
        const Point3d& t = to[i];

        double a = F[0]*f.x + F[1]*f.y + F[ 2]*f.z + F[ 3] - t.x;
        double b = F[4]*f.x + F[5]*f.y + F[ 6]*f.z + F[ 7] - t.y;
        double c = F[8]*f.x + F[9]*f.y + F[10]*f.z + F[11] - t.z;

        err[i] = (float)sqrt(a*a + b*b + c*c);
    }
}

bool cv::Affine3DEstimator::checkSubset( const CvMat* ms1, int count )
{
    CV_Assert( CV_MAT_TYPE(ms1->type) == CV_64FC3 );

    int j, k, i = count - 1;
    const Point3d* ptr = reinterpret_cast<const Point3d*>(ms1->data.ptr);

    // check that the i-th selected point does not belong
    // to a line connecting some previously selected points

    for(j = 0; j < i; ++j)
    {
        Point3d d1 = ptr[j] - ptr[i];
        double n1 = norm(d1);

        for(k = 0; k < j; ++k)
        {
            Point3d d2 = ptr[k] - ptr[i];
            double n = norm(d2) * n1;

            if (fabs(d1.dot(d2) / n) > 0.996)
                break;
        }
        if( k < j )
            break;
    }

    return j == i;
}

int cv::estimateAffine3D(InputArray _from, InputArray _to,
                         OutputArray _out, OutputArray _inliers,
                         double param1, double param2)
{
    Mat from = _from.getMat(), to = _to.getMat();
    int count = from.checkVector(3, CV_32F);

    CV_Assert( count >= 0 && to.checkVector(3, CV_32F) == count );

    _out.create(3, 4, CV_64F);
    Mat out = _out.getMat();

    _inliers.create(count, 1, CV_8U, -1, true);
    Mat inliers = _inliers.getMat();
    inliers = Scalar::all(1);

    Mat dFrom, dTo;
    from.convertTo(dFrom, CV_64F);
    to.convertTo(dTo, CV_64F);

    CvMat F3x4 = out;
    CvMat mask  = inliers;
    CvMat m1 = dFrom;
    CvMat m2 = dTo;

    const double epsilon = numeric_limits<double>::epsilon();
    param1 = param1 <= 0 ? 3 : param1;
    param2 = (param2 < epsilon) ? 0.99 : (param2 > 1 - epsilon) ? 0.99 : param2;

    return Affine3DEstimator().runRANSAC(&m1, &m2, &F3x4, &mask, param1, param2 );
}*/
