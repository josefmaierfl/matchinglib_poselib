/**********************************************************************************************************
 FILE: match_statOptFlow.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: September 2015

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for matching features
**********************************************************************************************************/

#pragma once

#define _WIN32_WINNT     0x0601

#include "glob_includes.h"

#include <Eigen/Core>

//#include "opencv2/core/types.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "nanoflann.hpp"


/* --------------------------- Defines --------------------------- */
typedef Eigen::Matrix<float,Eigen::Dynamic,2, Eigen::RowMajor> EMatFloat2;
typedef nanoflann::KDTreeEigenMatrixAdaptor< EMatFloat2, 2,nanoflann::metric_L2_Simple>  KDTree_D2float;

#define INITMATCHQUALEVAL 0 //If set to 1, the GMbSOF algorithm is aborted after generation of the SOF and the initial and filtered matches are given back for analysis (ensure that the code calling the method AdvancedMatching provides the necessary input/output variables)

#define PI 3.14159265
#define MIN_FLOW_TH 3.0 //Threshold for filtering matches with a flow smaller than this
#define MIN_INIT_KEYPS_FOR_MATCHING 40 //Minimum number of keypoints for which matching is performed with AdvancedMatching
#define MIN_FINAL_MATCHES 2 //Minimum number of final matches
#define MIN_INIT_INL_RATIO_S 0.2 //If the inlier ratio after initial matching with highdimensional descriptors (e.g. SIFT) is below this value, tree-based matching is performed instead of GMbSOF
#define MIN_INIT_INL_RATIO_F 0.08 //If the inlier ratio after initial matching with binary descriptors (e.g. FREAK) is below this value, tree-based matching is performed instead of GMbSOF
#define AUTOTH 1 //If 1, the SOF validation threshold is calculated using the estimated inlier ratio after initial matching. If 0, the provided or default=0.3 threshold is used
#define FILTER_WITH_CD_RATIOS 0 //If 2, the final matches are filtered according to their cost and distance ratios (cost / med. cost of surrounding matches, dist. to SOF-estimated pos. / med. of surrounding dists).
                                //If it is only one, the ratios are calculated but no filtering is performed. This is used for testing.

/* --------------------- Function prototypes --------------------- */

//Matches large sets of keypoints very effective and finds more correct matches than with BF matching
int AdvancedMatching( cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
                      std::vector<cv::KeyPoint> const& keypoints1, std::vector<cv::KeyPoint> const& keypoints2,
                      cv::Mat const& descriptors1, cv::Mat const& descriptors2, cv::Size imgSi,
                      std::vector<cv::DMatch>& filteredMatches12, bool finalCrossCheck = true, double validationTH = 0.3,
                      double stdMult = 3.5, int BFknn = 2, bool filterSmallFlow = false,
                      std::vector<float> *costRatios = NULL, std::vector<float> *distRatios = NULL,
                      double *estiInlRatio = NULL, std::vector<cv::DMatch> *initFilteredMatches = NULL, cv::InputArray img1 = cv::noArray(), cv::InputArray img2 = cv::noArray());
//Generates a grid-based statistical flow for many areas over the whole image.
int getStatisticalMatchingPositions(const EMatFloat2 keyP1, const EMatFloat2 keyP2, cv::Size imgSi,
                                    std::vector<std::vector<cv::Point3f>> & gridSearchParams, float *gridElemSize,
                                    cv::OutputArray mask = cv::noArray(), bool filterSmallFlow = false,
                                    unsigned int minYGridSize = 0, double validationTH = 0.3, KDTree_D2float *keypts1idxall = NULL, double stdMult_ = 3.5);
