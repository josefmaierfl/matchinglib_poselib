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


#ifndef _CV_MODEL_EST_H_
#define _CV_MODEL_EST_H_

#include "precomp.hpp"
#include "arrsac/arrsac.h"
#include <random>
//#include "../../Theia-master/src/theia/solvers/estimator.h"


class CvModelEstimator3
{
public:
	CvModelEstimator3(std::mt19937 &_mt, const int &_modelPoints, const cv::Size &_modelSize, const int &_maxBasicSolutions);
	virtual ~CvModelEstimator3();

    virtual int runKernel( const cv::Mat &m1, const cv::Mat &m2, cv::Mat &model )=0;
	virtual bool ValidModel(const cv::Mat &m1, const cv::Mat &m2, const cv::Mat &model) {return true;}
    virtual bool runLMeDS( const cv::Mat &m1, const cv::Mat &m2, cv::Mat &model, 
                           cv::Mat &mask, double confidence, int maxIters );
    virtual bool runRANSAC( const cv::Mat &m1, const cv::Mat &m2, cv::Mat &model,
                            cv::Mat &mask0, double reprojThreshold,
                            double confidence, int maxIters, bool lesqu );
	virtual bool runARRSAC( const cv::Mat &m1, const cv::Mat &m2, cv::Mat &model,
							cv::Mat &mask0, double reprojThreshold,
							bool lesqu,
							void (*refineEssential)(cv::InputArray points1, cv::InputArray points2, cv::InputArray E_init,
													cv::Mat & E_refined, double th, unsigned int iters, bool makeClosestE,
													double *sumSqrErr_init, double *sumSqrErr,
													cv::OutputArray errors, cv::InputOutputArray mask, int model, bool tryOrientedEpipolar, bool normalizeCorrs));
    virtual bool refine( const cv::Mat&, const cv::Mat&, cv::Mat&, int ) { return true; }
    // virtual void setSeed( int64 seed );
	virtual void computeReprojError3( const cv::Mat &m1, const cv::Mat &m2,
                                     const cv::Mat &model, cv::Mat &error ) = 0;

	int getModelPoints(){return modelPoints;}
	cv::Size getModelSize(){return modelSize;}
	int getMaxBasicSolutions(){return maxBasicSolutions;}

protected:

    virtual int findInliers( const cv::Mat &m1, const cv::Mat &m2,
                             const cv::Mat &model, cv::Mat &error,
                             cv::Mat &mask, double threshold );
    virtual bool getSubset( const cv::Mat &m1, const cv::Mat &m2,
                            cv::Mat &ms1, cv::Mat &ms2, int maxAttempts );
    virtual bool checkSubset( const cv::Mat &ms1, int count );

	std::mt19937 &mt;
	int modelPoints;
    cv::Size modelSize;
    int maxBasicSolutions;
    bool checkPartialSubsets;

	friend class EssentialMatEstimatorTheia;
};

class EssentialMatEstimatorTheia: public theia::Estimator<size_t,cv::Mat>
{
public:
	EssentialMatEstimatorTheia(CvModelEstimator3* modelEstimator,
						  const cv::Mat &points1, const cv::Mat &points2)
		:modelEstimator_(modelEstimator),
		 points1_(points1),
		 points2_(points2) {}

	~EssentialMatEstimatorTheia() {}

	bool EstimateModel(const std::vector<size_t>& data,
		std::vector<cv::Mat>* model) const;

	bool EstimateModelNonminimal(const std::vector<size_t>& data,
		std::vector<cv::Mat>* model) const;

	double Error(const size_t& data, const cv::Mat& model) const;

private:
	CvModelEstimator3* modelEstimator_;
	cv::Mat points1_, points2_;
};

#endif // _CV_MODEL_EST_H_
