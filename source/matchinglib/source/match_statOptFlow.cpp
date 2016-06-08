/**********************************************************************************************************
 FILE: match_statOptFlow.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: September 2015

 LOCATION: TechGate Vienna, Donau-City-Straße 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for matching features
**********************************************************************************************************/

#include "match_statOptFlow.h"
//#include "..\include\glob_includes.h"
//#include <iterator>
#include <list>

#include "PfeInlineQsort.h"

//#include <Eigen/Dense>
//#include <Eigen/StdVector>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include <opencv2/core/eigen.hpp>
//#include "opencv2/nonfree/nonfree.hpp"
//#include <opencv2/nonfree/features2d.hpp>

#include "ratioMatches_Flann.h"

#include "omp.h"

#include "vfcMatches.h"
#include <utility>
#include <vector>
#include <nmmintrin.h>
#include <bitset>
#ifdef __linux__
#include <inttypes.h>
#include <cpuid.h>
#endif

using namespace cv;
using namespace std;
using std::make_pair;

/* --------------------------- Defines --------------------------- */

/*
 * medErr ...	median of the reprojection errors masked as inliers
 * arithErr ... arithmetic mean value of the reprojection errors masked as inliers
 * arithStd	... standard deviation of the reprojection errors masked as inliers
 * medStd ... standard deviation of the reprojection errors masked as inliers using the median instead of the mean value
*/
typedef struct qualityParm {
		double medErr, arithErr, arithStd, medStd;
} qualityParm;

typedef struct mCostDist {
	float distance;//Distance to the estimated position
	float costs;//Descriptor distance
	unsigned int x;//Bin position (column index) in the SOF
	unsigned int y;//Bin position (row index) in the SOF
} mCostDist;

/* --------------------- Function prototypes --------------------- */

//This function matches keypoints from two images and performs the ratio test for filtering the matches
bool ratioTestMatches(cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
                         const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                         std::vector<cv::DMatch>& filteredMatches12);
//Calculates statistical parameters for the given values in the vector
void getStatisticfromVec(const std::vector<double> vals, qualityParm *stats, bool rejQuartiles = false);
//Calculates statistical parameters for a vector of angular values including values near 0 and 2*pi
void getAngularStatistic(const std::vector<double> vals, qualityParm *stats, bool rejQuartiles = false);
//This function performs a guided matching on the basis of precalculated statistical optical flow (output of one match per keypoint)
int guidedMatching(std::vector<std::vector<cv::Point3f>> gridSearchParams,
				   float gridElemSize,
				   std::vector<cv::KeyPoint> keypoints,
				   cv::Mat descriptors1,
				   cv::Mat descriptors2,
				   KDTree_D2float &keypointtree,
				   std::vector<int> keypIndexes,
				   cv::Size imgSi,
				   std::vector<cv::DMatch> &matches,
                   std::vector<mCostDist> mprops = std::vector<mCostDist>());
//This function performs a guided matching on the basis of precalculated statistical optical flow (output of knn)
int guidedMatching(std::vector<std::vector<cv::Point3f>> gridSearchParams,
				   float gridElemSize,
				   std::vector<cv::KeyPoint> keypoints,
				   cv::Mat descriptors1,
				   cv::Mat descriptors2,
				   KDTree_D2float &keypointtree,
				   std::vector<int> keypIndexes,
				   cv::Size imgSi,
				   std::vector<std::vector<cv::DMatch>> &matches,
                   unsigned int knn = 0,
                   std::vector<std::vector<mCostDist>> mprops = std::vector<std::vector<mCostDist>>());
//Generates a sparse set of keypoints from a large keypoint set.
void get_Sparse_KeypointField(std::vector<std::pair<cv::KeyPoint,int>> &keypInit,
							  KDTree_D2float &keypointtree,
							  EMatFloat2 eigkeypts,
							  EMatFloat2 gridPoints,
							  std::vector<cv::KeyPoint> keypoints,
							  int divx,
							  int divy,
							  float imgpart,
							  float lastwidthpart,
							  const int remainGridPix);
//This function compares the response of two keypoints to be able to sort them accordingly while keeping track of the index.
bool sortKeyPointsIdx(std::pair<cv::KeyPoint,int> first, std::pair<cv::KeyPoint,int> second);
//This function calculates the L1-norm of the hamming distance between two column vectors using a LUT.
unsigned int getHammingL1(cv::Mat vec1, cv::Mat vec2);
//This function calculates the L1-norm of the hamming distance between two column vectors using the CPU popcnt instruction.
unsigned int getHammingL1PopCnt(cv::Mat vec1, cv::Mat vec2, unsigned char byte8width);
//This function calculates the L2-norm of two descriptors
float getL2Distance(cv::Mat vec1, cv::Mat vec2);
//This function interpolates a grid of bins with statistical optical flow (SOF) values to get a smoother transition between the elements.
void interpolStatOptFlow(std::vector<std::vector<cv::Point3f>> & gridSearchParams, float & gridElemSize, cv::Size imgSi);
//Used to weight 2 or 3 optical flow elements out of the grid
cv::Point3f interpolFlowRad(cv::Point3f *f1, cv::Point3f *f2, cv::Point3f *f3 = NULL);
//Calculates for every match the cost and distance ratio to their local median
#if FILTER_WITH_CD_RATIOS
void getMeanDistCostFactors(std::vector<float> & Dfactors, std::vector<float> & Cfactors,
							std::vector<float> & quartDfactors, std::vector<float> & quartCfactors,
							std::vector<mCostDist> mprops, int si_x, int si_y);
#endif

/* --------------------- Functions --------------------- */

/* This function matches keypoints from two images. This function first calculates a grid in the image. For
 * every cell in this grid the strongest keypoints are chosen in the first and second image by using a KD tree. These
 * keypoints are matched with a NN matcher (FLANN). From these matches, the statistical optical flow is calculated.
 * Afterwards the approximated position for every keypoint in the first image is calculated in the second image. There,
 * the keypoints within an uncertainty radius are selected (by using a KD tree) for matching. Next, the descriptors of
 * the found keypoints are matched and filtered. If crosschecking is enabled, a final crosscheck or ratio test is
 * performed on the matches. This depends on the number (BFknn) of nearest neighbors (NN):
 * |	finalCrossCheck		|	BFknn	|	Description
 * |		false			|	 0-n	|	Independent of Bfknn only one nearest neighbor is searched at the estimated position.
 * |		true			|	  0		|	The number of NN is chosen for each match seperatly depending on the difference in
 *											descriptor distance. A crosscheck is performed on this varying # of NN from left to right and back
 * |		true			|	  1		|	Crosscheck for 1 NN from left to right and back
 * |		true			|	  2		|	Ratio test (2 NN)
 * |		true			|	  >2	|	Crosscheck for BFknn NN from left to right and back
 *
 * Ptr<DescriptorMatcher>& descriptorMatcher	Input  -> If not empty, the provided matcher instead of the FLANN-matcher is used for
 *														  initial matching. Standard should be empty.
 * vector<KeyPoint> keypoints1					Input  -> The keypoints in the left (first) image
 * vector<KeyPoint> keypoints2					Input  -> The keypoints in the right (second) image
 * Mat descriptors1								Input  -> The descriptors of the keypoints in the left (first) image
 * Mat descriptors2								Input  -> The descriptors of the keypoints in the right (second) image
 * Size imgSi									Input  -> The size of the image
 * vector<DMatch>& filteredMatches12			Output -> Fitered matches. If INITMATCHQUALEVAL = 1, the initial matches are returned.
 * bool finalCrossCheck=true					Input  -> If true, a final crosscheck or ratio test is performed on the matches
 * double validationTH = 0.3					Input  -> Threshold, above which the ratio (mean - median)/mean signals
 *														  an invalid flow field. The adjustable range lies between 0.1 and 1.0.
 *														  [Default = 0.3]. By default this value is estimated automatically according
 *														  to the estimated inlier ratio after initial matching. If you want to use the
 *														  provided threshold you have to recompile the code with the define AUTOTH set to 0.
 * double stdMult								Input  -> Multiplication factor for the standard deviation which is used to generate
 *														  the thresholds and the search range (e.g. range = mean + stdMult * sigma)
 *														  within the stimation of the statistical optical flow.
 *														  The adjustable range lies between 1.0 and 7.0. [Default = 3.5]
 * int BFknn=0									Input  -> Maximum number of matching train descriptos per query descriptor.
 *														  If BFknn=0, the number of matching train descriptors for one query
 *														  index depends on the distance (Hamming) to the next (in sorted
 *														  order (Hamming)) descriptor.
 * bool filterSmallFlow							Input  -> If true [default=false], matches with a too small flow are marked as invalid
 *														  and are excluded during the SOF estimation. Use with care! Use only, if you are
 *														  sure that matches cant have a flow of only a few pixels (MIN_FLOW_TH).
 * vector<float> *costRatios					Output -> If not NULL [Default = NULL], the ratio between the desrciptor distance
 *														  and its local median is returned for every match (ordering corresponds to
 *														  "filteredMatches12").
 * vector<float> *distRatios					Output -> If not NULL [Default = NULL], the ratio between the distance to the estimated
 *														  position (SOF) and its local median is returned for every match (ordering
 *														  corresponds to "filteredMatches12").
 * double *estiInlRatio							Output -> It is only allowed to be used if the define INITMATCHQUALEVAL = 1.
 *														  If INITMATCHQUALEVAL = 1, it must be used. This variable returns the
 *														  estimated inlier ratio of the data computed after initial matching
 * vector<DMatch> *initFilteredMatches			Output -> It is only allowed to be used if the define INITMATCHQUALEVAL = 1.
 *														  If INITMATCHQUALEVAL = 1, it must be used. This vector holds the
 *														  filtered initial matches after calculating the SOF.
 * InputArray img1 = cv::noArray()				Input  -> Only for visualization -> If the first (img1) and second (img2)
 *														  image are specified, the matches are displayed
 * InputArray img2 = cv::noArray()				Input  -> Only for visualization -> If the first (img1) and second (img2)
 *														  image are specified, the matches are displayed
 *
 * Return value:								 0:		  Everything ok
 *												-1:		  Too less keypoints -> do better a normal NN matching
 *												-2:		  Too less initial keypoints -> do better a normal NN matching
 *												-3:		  Prefiltering was not effective -> do better a normal NN matching
 *												-4:		  Calculation of flow statistic failed
 *												-5:		  Too less inliers after calculating SOF
 *												-6:		  Error during the guided matching (too less matches remaining)
 *												-7:		  Too less remaining matches -> There might be a problem with your data
 *												-8:		  Descriptor type not supported
 *												-9:		  Wrong SOF grid format
 */
int AdvancedMatching( cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
					  std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2,
					  cv::Mat descriptors1, cv::Mat descriptors2, cv::Size imgSi,
                      std::vector<DMatch>& filteredMatches12, bool finalCrossCheck, double validationTH,
					  double stdMult, int BFknn, bool filterSmallFlow,
					  std::vector<float> *costRatios, std::vector<float> *distRatios,
					  double *estiInlRatio, std::vector<cv::DMatch> *initFilteredMatches, cv::InputArray img1, cv::InputArray img2)
{
	if((keypoints1.size() < MIN_INIT_KEYPS_FOR_MATCHING) || (keypoints2.size() < MIN_INIT_KEYPS_FOR_MATCHING)) return -1; //too less keypoints -> do better a normal BF matching

#if INITMATCHQUALEVAL
	if((estiInlRatio == NULL) || (initFilteredMatches == NULL))
	{
		cout << "GMbSOF is not configured for evaluating the initial matches! Exiting." << endl;
		exit(0);
	}
	std::vector<cv::DMatch> initMatchesValidIdx;
#else
	if(estiInlRatio || initFilteredMatches)
	{
		cout << "GMbSOF is configured for evaluating the initial matches but the calling function is not! Exiting." << endl;
		exit(0);
	}
#endif

	filteredMatches12.clear();
    vector<vector<DMatch> > matches12, matches21;
	std::vector<std::pair<cv::KeyPoint,int>> keypInit1, keypInit2;
	vector<int> index1, index2;
	const int remainGridPix = 25; //If the last column of the grid in the image is smaller than this, it is ignored
	double estim_inlRatio;
	int err_SOF = 0;
#if FILTER_WITH_CD_RATIOS
	vector<mCostDist> mprops_init;
	vector<float> costRatios_tmp, distRatios_tmp;
#endif

	//Generate grid for sparse flow init
	int divx, divy = 7, idx;
	float imgpart, lastwidthpart, xpos, imgpart2, lwidth2;
	if(imgSi.height >= 800)
		divy = (int)floor((float)imgSi.height/100.0);
	imgpart = (float)imgSi.height/(float)divy;
	divx = (int)floor((float)imgSi.width/imgpart);
	lastwidthpart = (float)imgSi.width-(float)divx*imgpart;
	imgpart2 = imgpart/2;
	lwidth2 = lastwidthpart/2;
	if(lastwidthpart > remainGridPix) //if the remaining column of the image is too small forget it
		divx++;

	EMatFloat2 gridPoints(divx*divy,2);
	Eigen::Matrix<float,Eigen::Dynamic,1> gridX(divx,1);
	Eigen::Matrix<float,1,1> gridY(1,1);
	gridX(0,0) = gridY(0,0) = xpos = imgpart2;
	for(int i = 0; i<divy;i++)
	{
		idx = i*divx;
		gridPoints.block(idx,1,divx,1) = gridY.replicate(divx,1);
		gridY(0,0) += imgpart;
	}
	for(int j = 1;j < ((lastwidthpart <= imgpart2) && (lastwidthpart > remainGridPix) ? (divx-1):divx);j++)
	{
		xpos += imgpart;
		gridX(j,0) = xpos;
	}
	if((lastwidthpart <= imgpart2) && (lastwidthpart > remainGridPix))
	{
		gridX(divx-1,0) = xpos + imgpart2 + lwidth2;
	}
	else
	{
		gridX(divx-1,0) = xpos;
	}

	gridPoints.col(0) = gridX.replicate(divy,1);


	//Prepare the coordinates of the keypoints for the KD-tree
	EMatFloat2 eigkeypts1(keypoints1.size(),2), eigkeypts2(keypoints2.size(),2);

	for(unsigned int i = 0;i<keypoints1.size();i++)
	{
		eigkeypts1(i,0) = keypoints1.at(i).pt.x;
		eigkeypts1(i,1) = keypoints1.at(i).pt.y;
	}

	for(unsigned int i = 0;i<keypoints2.size();i++)
	{
		eigkeypts2(i,0) = keypoints2.at(i).pt.x;
		eigkeypts2(i,1) = keypoints2.at(i).pt.y;
	}


	//Generate the KD-tree index for the keypoint coordinates
	const int maxLeafNum     = 20;
	KDTree_D2float keypts1idx(2,eigkeypts1,maxLeafNum);
	keypts1idx.index->buildIndex();

	KDTree_D2float keypts2idx(2,eigkeypts2,maxLeafNum);
	keypts2idx.index->buildIndex();

	//Get the sparse set of keypoints of the images
	get_Sparse_KeypointField(keypInit1, keypts1idx, eigkeypts1, gridPoints, keypoints1, divx, divy, imgpart, lastwidthpart, remainGridPix);
	get_Sparse_KeypointField(keypInit2, keypts2idx, eigkeypts2, gridPoints, keypoints2, divx, divy, imgpart, lastwidthpart, remainGridPix);

	// FOR DEBUGGING - SHOW THE SPARSE KEYPOINTS
	if(!img1.empty() && !img2.empty())
	{
		Mat img1c, img2c;
		vector<KeyPoint> keyp1, keyp2;

		for(unsigned int i = 0; i<keypInit1.size();i++)
			keyp1.push_back(keypInit1[i].first);
		for(unsigned int i = 0; i<keypInit2.size();i++)
			keyp2.push_back(keypInit2[i].first);

		drawKeypoints( img1.getMat(), keyp1, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
		drawKeypoints( img2.getMat(), keyp2, img2c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

		imshow("Keypoints 1", img1c );
		imshow("Keypoints 2", img2c );

		waitKey(0);
	}

	if((keypInit1.size() < 30) || (keypInit2.size() < 30)) return -2; // Too less initial keypoints
	//if(((float)keypoints1.size()/(float)keypInit1.size() < 3.0) || ((float)keypoints2.size()/(float)keypInit2.size() < 3.0)) return -3; // Prefiltering was not effective

	//Get the descriptors for the sparse keypoints and do the matching with a given matcher (with subsequent ratio test) or match them with a KD-tree matcher from the flann lib and perform a ratio test [default]
	Mat initDescriptors1, initDescriptors2;
	vector<DMatch> initMatches;
	for(unsigned int i = 0; i<keypInit1.size();i++)
		initDescriptors1.push_back(descriptors1.row(keypInit1[i].second));
	for(unsigned int i = 0; i<keypInit2.size();i++)
		initDescriptors2.push_back(descriptors2.row(keypInit2[i].second));
	if(descriptorMatcher.empty())
	{
		ratioTestFlannMatches(initDescriptors1, initDescriptors2, initMatches, &estim_inlRatio);
	}
	else
	{
		ratioTestMatches(descriptorMatcher,initDescriptors1, initDescriptors2, initMatches);
		estim_inlRatio = (double)initMatches.size() / (double)initDescriptors1.rows;
	}

    vector<int> queryIdxs( initMatches.size() ), trainIdxs( initMatches.size() );
    for( unsigned int i = 0; i < initMatches.size(); i++ )
    {
        queryIdxs[i] = initMatches[i].queryIdx;
        trainIdxs[i] = initMatches[i].trainIdx;
    }

	// FOR DEBUGGING - SHOW THE MATCHES
	if(!img1.empty() && !img2.empty())
	{
		Mat drawImg;
		vector<KeyPoint> keyp1, keyp2;

		for(unsigned int i = 0; i<keypInit1.size();i++)
			keyp1.push_back(keypInit1[i].first);
		for(unsigned int i = 0; i<keypInit2.size();i++)
			keyp2.push_back(keypInit2[i].first);

		drawMatches( img1.getMat(), keyp1, img2.getMat(), keyp2, initMatches, drawImg );
		cv::imshow( "Initial Matches", drawImg );
		cv::waitKey(0);
	}

	EMatFloat2 keyP1(queryIdxs.size(),2), keyP2(queryIdxs.size(),2);
	std::vector<std::vector<cv::Point3f>> gridSearchParams;
	Mat inliers;
	float gridElemSize;
	do
	{
		//If the inlier ratio after initial matches is below a treshold perform a tree-based matching and skip GMbSOF
		if(((estim_inlRatio < MIN_INIT_INL_RATIO_S) && (descriptors1.type() == CV_32F)) || ((estim_inlRatio < MIN_INIT_INL_RATIO_F) && (descriptors1.type() == CV_8U)))
		{
			bool nobadInitialMatching = true;
			//Check if the initial matches are matches that survived the ratio test
			if(std::abs((double)initMatches.size() / (double)initDescriptors1.rows - estim_inlRatio) > 0.01)
			{
				ratioTestFlannMatches(descriptors1, descriptors2, filteredMatches12, NULL, true);
				nobadInitialMatching = false;
			}

			vector<int> keypIndexes1,keypIndexes2, keypIndexes12,keypIndexes22;
			if(nobadInitialMatching)
			{
				//Get original matching indices from initial matches
				for(unsigned int i = 0; i<queryIdxs.size();i++)
				{
					DMatch m_tmp = initMatches[i];
					keypIndexes1.push_back(keypInit1[queryIdxs[i]].second);
					keypIndexes2.push_back(keypInit2[trainIdxs[i]].second);
					m_tmp.queryIdx = keypIndexes1.back();
					m_tmp.trainIdx = keypIndexes2.back();
					filteredMatches12.push_back(m_tmp);
				}
			}
			else
			{
				for(unsigned int i = 0; i<filteredMatches12.size();i++)
				{
					keypIndexes1.push_back(filteredMatches12[i].queryIdx);
					keypIndexes2.push_back(filteredMatches12[i].trainIdx);
				}
			}

			//Search for matches that match the same right keypoint and delete them
			{
				vector<size_t> mdelIdx;
				for(size_t i = 0; i < keypIndexes2.size(); i++)
				{
					int matchtrainIdx = keypIndexes2[i];
					bool foundduplicate = false;
					for(size_t j = i + 1; j < keypIndexes2.size(); j++)
					{
						if(matchtrainIdx == keypIndexes2[j])
						{
							foundduplicate = true;
							mdelIdx.push_back(j);
						}
					}
					if(foundduplicate)
						mdelIdx.push_back(i);
				}
				if(!mdelIdx.empty())
				{
					sort(mdelIdx.begin(), mdelIdx.end(),
						[](size_t first, size_t second){return first > second;});
					//Remove duplicates
					for(int i = 0; i < (int)mdelIdx.size() - 1; i++)
					{
						while(mdelIdx[i] == mdelIdx[i+1])
						{
							mdelIdx.erase(mdelIdx.begin() + i + 1);
							if(i + 1 == (int)mdelIdx.size())
								break;
						}
					}
					for(int i = 0; i < mdelIdx.size(); i++)
					{
						keypIndexes1.erase(keypIndexes1.begin() + mdelIdx[i]);
						keypIndexes2.erase(keypIndexes2.begin() + mdelIdx[i]);
						filteredMatches12.erase(filteredMatches12.begin() + mdelIdx[i]);
					}
				}
			}

			if(nobadInitialMatching)
			{
				//Extract left descriptors that are not included in the initial matches
				cv::Mat descriptorrest1, descriptorrest2;
				vector<DMatch> restMatches, restmatches2;
				if(!filteredMatches12.empty())
				{
					std::sort(keypIndexes1.begin(),keypIndexes1.end());
					size_t j = 0;
					for(unsigned int i = 0; i < keypoints1.size(); i++)
					{
						if(keypIndexes1[j] == i)
						{
							if(j < (keypIndexes1.size() - 1))
								j++;
							continue;
						}
						keypIndexes12.push_back(i);
						descriptorrest1.push_back(descriptors1.row(i));
					}

					//Extract right descriptors that are not included in the initial matches
					std::sort(keypIndexes2.begin(),keypIndexes2.end());
					j = 0;
					for(unsigned int i = 0; i < keypoints2.size(); i++)
					{
						if(keypIndexes2[j] == i)
						{
							if(j < (keypIndexes2.size() - 1))
								j++;
							continue;
						}
						keypIndexes22.push_back(i);
						descriptorrest2.push_back(descriptors2.row(i));
					}
					keypIndexes1.clear();
					keypIndexes2.clear();

					//Match the rest of the features
					ratioTestFlannMatches(descriptorrest1, descriptorrest2, restMatches, NULL, true);

					//Get original matching indices from the matches
					for(unsigned int i = 0; i<restMatches.size();i++)
					{
						DMatch m_tmp = restMatches[i];
						keypIndexes1.push_back(keypIndexes12[restMatches[i].queryIdx]);
						keypIndexes2.push_back(keypIndexes22[restMatches[i].trainIdx]);
						m_tmp.queryIdx = keypIndexes1.back();
						m_tmp.trainIdx = keypIndexes2.back();
						restmatches2.push_back(m_tmp);
					}
				}
				else
				{
					ratioTestFlannMatches(descriptors1, descriptors2, filteredMatches12, NULL, true);
					nobadInitialMatching = false;

					for(unsigned int i = 0; i<filteredMatches12.size();i++)
					{
						keypIndexes1.push_back(filteredMatches12[i].queryIdx);
						keypIndexes2.push_back(filteredMatches12[i].trainIdx);
					}
				}

				//Search for matches that match the same right keypoint and delete them
				{
					vector<size_t> mdelIdx;
					for(size_t i = 0; i < keypIndexes2.size(); i++)
					{
						int matchtrainIdx = keypIndexes2[i];
						bool foundduplicate = false;
						for(size_t j = i + 1; j < keypIndexes2.size(); j++)
						{
							if(matchtrainIdx == keypIndexes2[j])
							{
								foundduplicate = true;
								mdelIdx.push_back(j);
							}
						}
						if(foundduplicate)
							mdelIdx.push_back(i);
					}
					if(!mdelIdx.empty())
					{
						sort(mdelIdx.begin(), mdelIdx.end(),
							[](size_t first, size_t second){return first > second;});
						//Remove duplicates
						for(int i = 0; i < (int)mdelIdx.size() - 1; i++)
						{
							while(mdelIdx[i] == mdelIdx[i+1])
							{
								mdelIdx.erase(mdelIdx.begin() + i + 1);
								if(i + 1 == (int)mdelIdx.size())
									break;
							}
						}
						for(int i = 0; i < mdelIdx.size(); i++)
						{
							keypIndexes1.erase(keypIndexes1.begin() + mdelIdx[i]);
							keypIndexes2.erase(keypIndexes2.begin() + mdelIdx[i]);
							if(nobadInitialMatching)
								restmatches2.erase(restmatches2.begin() + mdelIdx[i]);
							else
								filteredMatches12.erase(filteredMatches12.begin() + mdelIdx[i]);
						}
					}
				}

				//Concatenate initial matches and the rest of the matches
				if(nobadInitialMatching)
					filteredMatches12.insert(filteredMatches12.end(), restmatches2.begin(), restmatches2.end());
			}

			//Filter final matches with VFC
			{
				vector<DMatch> vfcfilteredMatches;
				if(!filterWithVFC(keypoints1, keypoints2, filteredMatches12, vfcfilteredMatches))
				{
					if((vfcfilteredMatches.size() > 8) || (filteredMatches12.size() < 24))
						filteredMatches12 = vfcfilteredMatches;
				}
			}

			// FOR DEBUGGING - SHOW ALL MATCHES
			if(!img1.empty() && !img2.empty())
			{
				Mat drawImg;

				drawMatches( img1.getMat(), keypoints1, img2.getMat(), keypoints2, filteredMatches12, drawImg );
				//imwrite("C:\\work\\matches_final.jpg", drawImg);
				cv::imshow( "All Matches", drawImg );
				cv::waitKey(0);

				//Show reduced set of matches
				{
					Mat img_match;
					std::vector<cv::KeyPoint> keypL_reduced;//Left keypoints
					std::vector<cv::KeyPoint> keypR_reduced;//Right keypoints
					std::vector<cv::DMatch> matches_reduced;
					std::vector<cv::KeyPoint> keypL_reduced1;//Left keypoints
					std::vector<cv::KeyPoint> keypR_reduced1;//Right keypoints
					std::vector<cv::DMatch> matches_reduced1;
					int j = 0;
					size_t keepNMatches = 100;
					if(filteredMatches12.size() > keepNMatches)
					{
						size_t keepXthMatch = filteredMatches12.size() / keepNMatches;
						for (unsigned int i = 0; i < filteredMatches12.size(); i++)
						{
							int idx = filteredMatches12[i].queryIdx;
							keypL_reduced.push_back(keypoints1[idx]);
							matches_reduced.push_back(filteredMatches12[i]);
							matches_reduced.back().queryIdx = i;
							keypR_reduced.push_back(keypoints2[matches_reduced.back().trainIdx]);
							matches_reduced.back().trainIdx = i;
						}
						j = 0;
						for (unsigned int i = 0; i < matches_reduced.size(); i++)
						{
							if((i % (int)keepXthMatch) == 0)
							{
								keypL_reduced1.push_back(keypL_reduced[i]);
								matches_reduced1.push_back(matches_reduced[i]);
								matches_reduced1.back().queryIdx = j;
								keypR_reduced1.push_back(keypR_reduced[i]);
								matches_reduced1.back().trainIdx = j;
								j++;
							}
						}
						drawMatches(img1.getMat(), keypL_reduced1, img2.getMat(), keypR_reduced1, matches_reduced1, img_match);
						imshow("Approx. 100 found matches", img_match);
						waitKey(0);
					}
				}
			}

			if(filteredMatches12.size() < MIN_FINAL_MATCHES) return -7; //Too less remaining matches -> There might be a problem with your data

			return 0;
		}

		//Order the keypoints according to the matching
		/*Mat unFiltPoints1, unFiltPoints2;
		Mat H,inliers,points;

		cv::eigen2cv(eigkeypts1,points);
		for(unsigned int i = 0; i<queryIdxs.size();i++)
			unFiltPoints1.push_back(points.row(keypInit1[queryIdxs[i]].second));

		cv::eigen2cv(eigkeypts2,points);
		for(unsigned int i = 0; i<trainIdxs.size();i++)
			unFiltPoints2.push_back(points.row(keypInit2[trainIdxs[i]].second));*/

		for(unsigned int i = 0; i<queryIdxs.size();i++)
			keyP1.row(i) = eigkeypts1.row(keypInit1[queryIdxs[i]].second);

		for(unsigned int i = 0; i<trainIdxs.size();i++)
			keyP2.row(i) = eigkeypts2.row(keypInit2[trainIdxs[i]].second);

#if AUTOTH
		if(descriptors1.type() == CV_32F)
		{
			if(estim_inlRatio >= 0.75)
			{
				validationTH = 0.75;
			}
			else if(estim_inlRatio <= 0.3)
			{
				validationTH = 0.3;
			}
			else
			{
				validationTH = estim_inlRatio;
			}
		}
		else if(descriptors1.type() == CV_8U)
		{
			if(estim_inlRatio >= 0.45)
			{
				validationTH = 0.75;
			}
			else if(estim_inlRatio <= 0.15)
			{
				validationTH = 0.3;
			}
			else
			{
				validationTH = 1.5 * estim_inlRatio + 0.075;
				if(validationTH > 0.75)
				{
					validationTH = 0.75;
				}
			}
		}
#endif

		if(getStatisticalMatchingPositions(keyP1, keyP2, imgSi, gridSearchParams, &gridElemSize, inliers, filterSmallFlow, 0, validationTH, &keypts1idx, stdMult))
		{
			if((MIN_INIT_INL_RATIO_S > 0.0) || (MIN_INIT_INL_RATIO_F > 0.0))
			{
				estim_inlRatio = 0.001;
				err_SOF++;
				if(err_SOF > 1)
					return -4; //Calculation of flow statistic failed
				continue;
			}
			else
			{
				return -5; // Too less inliers after filtering
			}
		}

		//H = findHomography(unFiltPoints1, unFiltPoints2,CV_RANSAC,5,inliers);
		//if( H.empty() ) return -4; //It was not possible to calculate H

		int nrIn = 0;
		for( unsigned int i = 0; i < trainIdxs.size(); i++)
			if(inliers.at<bool>(i,0) == true ) nrIn++;
		if(nrIn < 15)
		{
			if((MIN_INIT_INL_RATIO_S > 0.0) || (MIN_INIT_INL_RATIO_F > 0.0))
			{
				estim_inlRatio = 0.001;
				err_SOF++;
				if(err_SOF > 1)
					return -5; // Too less inliers after filtering
				continue;
			}
			else
			{
				return -5; // Too less inliers after filtering
			}
		}
	}
	while(err_SOF == 1);

	//Interpolate the statistical optical flow at the grid borders (the number of grid elements can be 25 times larger afterwards)
	interpolStatOptFlow(gridSearchParams, gridElemSize, imgSi);

	// FOR DEBUGGING - SHOW THE MATCHES AFTER FILTERING
	if(!img1.empty() && !img2.empty())
	{
		vector<char> matchesMask( initMatches.size(), 0 );
		vector<KeyPoint> keyp1, keyp2;
		Mat drawImg;

		for( unsigned int i = 0; i < trainIdxs.size(); i++)
			if( inliers.at<bool>(i,0) == true )
			{
				matchesMask[i] = 1;
			}

		for(unsigned int i = 0; i<keypInit1.size();i++)
			keyp1.push_back(keypInit1[i].first);
		for(unsigned int i = 0; i<keypInit2.size();i++)
			keyp2.push_back(keypInit2[i].first);

		drawMatches( img1.getMat(), keyp1, img2.getMat(), keyp2, initMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask);
		//imwrite("C:\\work\\matches_init_filtered.jpg", drawImg);
		cv::imshow( "Filtered initial matches", drawImg );
		cv::waitKey(0);
	}

	//Filter out wrong matches
	vector<KeyPoint> keys1, keys2;
	vector<int> keypIndexes1,keypIndexes2;
	for(unsigned int i = 0; i<queryIdxs.size();i++)
	{
		if(inliers.at<bool>(i) == true)
		{
			DMatch m_tmp = initMatches[i];
			keys1.push_back(keypInit1[queryIdxs[i]].first);
			keys2.push_back(keypInit2[trainIdxs[i]].first);
			keypIndexes1.push_back(keypInit1[queryIdxs[i]].second);
			keypIndexes2.push_back(keypInit2[trainIdxs[i]].second);
			m_tmp.queryIdx = keypIndexes1.back();
			m_tmp.trainIdx = keypIndexes2.back();
			filteredMatches12.push_back(m_tmp);
		}
#if INITMATCHQUALEVAL
		DMatch m_tmp1 = initMatches[i];
		m_tmp1.queryIdx = keypInit1[queryIdxs[i]].second;
		m_tmp1.trainIdx = keypInit2[trainIdxs[i]].second;
		initMatchesValidIdx.push_back(m_tmp1);
#endif
	}
	keypInit1.clear();
	keypInit2.clear();

	//Search for matches that match the same right keypoint and delete them
	{
		vector<size_t> mdelIdx;
		for(size_t i = 0; i < keypIndexes2.size(); i++)
		{
			int matchtrainIdx = keypIndexes2[i];
			bool foundduplicate = false;
			for(size_t j = i + 1; j < keypIndexes2.size(); j++)
			{
				if(matchtrainIdx == keypIndexes2[j])
				{
					foundduplicate = true;
					mdelIdx.push_back(j);
				}
			}
			if(foundduplicate)
				mdelIdx.push_back(i);
		}
		if(!mdelIdx.empty())
		{
			sort(mdelIdx.begin(), mdelIdx.end(),
				[](size_t first, size_t second){return first > second;});
			//Remove duplicates
			for(int i = 0; i < (int)mdelIdx.size() - 1; i++)
			{
				while(mdelIdx[i] == mdelIdx[i+1])
				{
					mdelIdx.erase(mdelIdx.begin() + i + 1);
					if(i + 1 == (int)mdelIdx.size())
						break;
				}
			}
			for(int i = 0; i < mdelIdx.size(); i++)
			{
				keys1.erase(keys1.begin() + mdelIdx[i]);
				keys2.erase(keys2.begin() + mdelIdx[i]);
				keypIndexes1.erase(keypIndexes1.begin() + mdelIdx[i]);
				keypIndexes2.erase(keypIndexes2.begin() + mdelIdx[i]);
				filteredMatches12.erase(filteredMatches12.begin() + mdelIdx[i]);
			}
		}
	}

#if INITMATCHQUALEVAL
	*estiInlRatio = estim_inlRatio;
	*initFilteredMatches = filteredMatches12;
	filteredMatches12 = initMatchesValidIdx;
	return 0;
#endif

	//Calculate the matching costs for the initial keypoints
	if(descriptors1.type() == CV_8U)
	{
		int cpuInfo[4];
// __linux__
    // __cpuid(cpuInfo,0x00000001);
// #endif
		std::bitset<32> f_1_ECX_ = cpuInfo[2];
		bool popcnt_available = f_1_ECX_[23];
		if(popcnt_available)
		{
			unsigned char byte8width = (unsigned char)descriptors1.cols / 8;
			for(size_t i = 0; i < filteredMatches12.size(); i++)
			{
				filteredMatches12[i].distance = (float)getHammingL1PopCnt(descriptors1.row(queryIdxs[i]), descriptors2.row(trainIdxs[i]), byte8width);
			}
		}
		else
		{
			for(size_t i = 0; i < filteredMatches12.size(); i++)
			{
				filteredMatches12[i].distance = (float)getHammingL1(descriptors1.row(queryIdxs[i]), descriptors2.row(trainIdxs[i]));
			}
		}
	}
	else if(descriptors1.type() == CV_32F)
	{
		for(size_t i = 0; i < filteredMatches12.size(); i++)
		{
			filteredMatches12[i].distance = getL2Distance(descriptors1.row(queryIdxs[i]), descriptors2.row(trainIdxs[i]));
		}
	}
	else
	{
		cout << "Descriptor type not supported!" << endl;
		return -8; //Descriptor type not supported
	}

	//Calculate the SOF grid position index and distance of each initial match to its corresponding SOF estimate
#if FILTER_WITH_CD_RATIOS
	{
		mCostDist mprop_init_tmp;
		const unsigned int gridSPsize[2] = {gridSearchParams[0].size(), gridSearchParams.size()};//Format [x,y] -> gridSearchParams[y][x]
		const unsigned int gridSPsizeIdx[2] = {gridSPsize[0] - 1, gridSPsize[1] - 1};
		unsigned int grid_x, grid_y;

		if(gridSearchParams[0].size() != gridSearchParams.back().size())
			return -9; //Size in dimension 1 (x and bin cols, respectively) must be the same for all bin rows (y coordinates) in gridSearchParams

		for(size_t i = 0; i < filteredMatches12.size(); i++)
		{
			float esti_pos[2], keypDiff[2];
			int singleQueryIdx = filteredMatches12[i].queryIdx;
			int singleTrainIdx = filteredMatches12[i].trainIdx;
			grid_x = (unsigned int)floor(keypoints1[singleQueryIdx].pt.x / gridElemSize);
			grid_y = (unsigned int)floor(keypoints1[singleQueryIdx].pt.y / gridElemSize);
			if(grid_y < gridSPsize[1])
			{
				if(grid_x < gridSPsize[0])
				{
					esti_pos[0] = keypoints1[singleQueryIdx].pt.x + gridSearchParams[grid_y][grid_x].x;
					esti_pos[1] = keypoints1[singleQueryIdx].pt.y + gridSearchParams[grid_y][grid_x].y;
				}
				else
				{
					esti_pos[0] = keypoints1[singleQueryIdx].pt.x + gridSearchParams[grid_y].back().x;
					esti_pos[1] = keypoints1[singleQueryIdx].pt.y + gridSearchParams[grid_y].back().y;
					grid_x = gridSPsizeIdx[0];
				}
			}
			else
			{
				if(grid_x < gridSPsize[0])
				{
					esti_pos[0] = keypoints1[singleQueryIdx].pt.x + gridSearchParams.back()[grid_x].x;
					esti_pos[1] = keypoints1[singleQueryIdx].pt.y + gridSearchParams.back()[grid_x].y;
					grid_y = gridSPsizeIdx[1];
				}
				else
				{
					esti_pos[0] = keypoints1[singleQueryIdx].pt.x + gridSearchParams.back().back().x;
					esti_pos[1] = keypoints1[singleQueryIdx].pt.y + gridSearchParams.back().back().y;
					grid_x = gridSPsizeIdx[0];
					grid_y = gridSPsizeIdx[1];
				}
			}
			keypDiff[0] = keypoints2[singleTrainIdx].pt.x - esti_pos[0];
			keypDiff[1] = keypoints2[singleTrainIdx].pt.y - esti_pos[1];
			keypDiff[0] *= keypDiff[0];
			keypDiff[1] *= keypDiff[1];
			mprop_init_tmp.distance = sqrtf(keypDiff[0] + keypDiff[1]);
			mprop_init_tmp.costs = filteredMatches12[i].distance;
			mprop_init_tmp.x = grid_x;
			mprop_init_tmp.y = grid_y;
			mprops_init.push_back(mprop_init_tmp);
		}
	}
#endif

	//Guided matching of all keypoints
	vector<DMatch> matches;
#if FILTER_WITH_CD_RATIOS
	vector<mCostDist> mprops;
#endif
	int gmerr;
	const int maxDepthSearch = 32;
	const int hammL1tresh = 160;
	const int searchRadius = (int)powf(10,2); //The radius is squared because the L2-norm is used within the KD-tree

	idx = 0;
	std::sort(keypIndexes1.begin(),keypIndexes1.end());

	if(!finalCrossCheck)
	{
		if(keypoints1.size() < 40)//Perform matching on single CPU
		{
#if FILTER_WITH_CD_RATIOS
			gmerr = guidedMatching(gridSearchParams, gridElemSize, keypoints1, descriptors1, descriptors2, keypts2idx, keypIndexes1, imgSi, matches, mprops);
#else
            gmerr = guidedMatching(gridSearchParams, gridElemSize, keypoints1, descriptors1, descriptors2, keypts2idx, keypIndexes1, imgSi, matches);
#endif
			if(gmerr != 0) return -6; //Error during the guided matching (too less matches remaining)
		}
		else//Perform matching on multiple CPU cores
		{
			const size_t nr_threads = 8;
			size_t nrFirstXKeys = keypoints1.size() / nr_threads;
			size_t keyPsIdx = 0;
			vector<size_t> nr_keypoints;
			vector<vector<cv::KeyPoint>> threadkeyPs;
			vector<cv::Mat> threadDescriptors;
			vector<vector<int>> threadkeypIndexes1;
			vector<vector<DMatch>> threadmatches12(nr_threads);
#if FILTER_WITH_CD_RATIOS
			vector<vector<mCostDist>> threadmprops12(nr_threads);
#endif
			for(size_t i = 0; i < nr_threads - 1; i++)
			{
				nr_keypoints.push_back(nrFirstXKeys);
			}
			nr_keypoints.push_back(keypoints1.size() - (nr_threads - 1) * nrFirstXKeys);
			for(size_t i = 0; i < nr_threads; i++)
			{
				threadkeyPs.push_back(vector<cv::KeyPoint>(keypoints1.begin() + keyPsIdx, keypoints1.begin() + keyPsIdx + nr_keypoints[i]));
				threadDescriptors.push_back(descriptors1.rowRange(keyPsIdx, keyPsIdx + nr_keypoints[i]).clone());
				threadkeypIndexes1.push_back(keypIndexes1);
				if(i > 0)
				{
					size_t delnegidx = 0;
					for(size_t j = 0; j < threadkeypIndexes1.back().size(); j++)
					{
						threadkeypIndexes1.back()[j] -= (int)keyPsIdx;
						if(threadkeypIndexes1.back()[j] < 0)
							delnegidx++;
					}
					threadkeypIndexes1.back().erase(threadkeypIndexes1.back().begin(), threadkeypIndexes1.back().begin() + delnegidx);
				}
				keyPsIdx += nr_keypoints[i];
			}
			//Perform multithreaded matching
			vector<int> errvec(nr_threads);
			#pragma omp parallel for
			for(int i = 0; i < (const int)nr_threads; i++)
			{
#if FILTER_WITH_CD_RATIOS
				errvec[i] = guidedMatching(gridSearchParams, gridElemSize, threadkeyPs[i], threadDescriptors[i], descriptors2, keypts2idx, threadkeypIndexes1[i], imgSi, threadmatches12[i], threadmprops12[i]);
#else
                errvec[i] = guidedMatching(gridSearchParams, gridElemSize, threadkeyPs[i], threadDescriptors[i], descriptors2, keypts2idx, threadkeypIndexes1[i], imgSi, threadmatches12[i]);
#endif
			}
			{
				int errcnt = 0;
				for(size_t i = 1; i < nr_threads; i++)
				{
					if(errvec[i] != 0)
						errcnt++;
				}
				if(errcnt == nr_threads)
					return -6; //Error during the guided matching (too less matches remaining)
			}
			keyPsIdx = nr_keypoints[0];
			for(size_t i = 1; i < nr_threads; i++)
			{
				for(size_t j = 0; j < threadmatches12[i].size(); j++)
				{
					threadmatches12[i][j].queryIdx += (int)keyPsIdx;
				}
				keyPsIdx += nr_keypoints[i];
			}
			matches = threadmatches12[0];
#if FILTER_WITH_CD_RATIOS
			mprops = threadmprops12[0];
#endif
			for(size_t i = 1; i < nr_threads; i++)
			{
				matches.insert(matches.end(), threadmatches12[i].begin(), threadmatches12[i].end());
#if FILTER_WITH_CD_RATIOS
				mprops.insert(mprops.end(), threadmprops12[i].begin(), threadmprops12[i].end());
#endif
			}
		}
	}
	else
	{
		std::sort(keypIndexes2.begin(),keypIndexes2.end());
		int sizeDiff;
		std::vector<std::vector<cv::Point3f>> gridSearchParams_inv;

		for(size_t i = 0; i < gridSearchParams.size(); i++)
		{
			Point3f inv_flow(-1.0f * gridSearchParams[i][0].x, -1.0f * gridSearchParams[i][0].y, gridSearchParams[i][0].z);
			gridSearchParams_inv.push_back(vector<Point3f>(1,inv_flow));
			for(size_t j = 1; j < gridSearchParams.back().size(); j++)
			{
				inv_flow.x = -1.0f * gridSearchParams[i][j].x;
				inv_flow.y = -1.0f * gridSearchParams[i][j].y;
				inv_flow.z = gridSearchParams[i][j].z;
				gridSearchParams_inv.back().push_back(inv_flow);
			}
		}

		if(BFknn == 1)
		{
			vector<DMatch> matches12,matches21;
#if FILTER_WITH_CD_RATIOS
			vector<mCostDist> mprops12/*,mprops21*/;
#endif
			if(keypoints1.size() < 40)//Perform matching on single CPU
			{
#if FILTER_WITH_CD_RATIOS
				gmerr = guidedMatching(gridSearchParams, gridElemSize, keypoints1, descriptors1, descriptors2, keypts2idx, keypIndexes1, imgSi, matches12, mprops12);
#else
                gmerr = guidedMatching(gridSearchParams, gridElemSize, keypoints1, descriptors1, descriptors2, keypts2idx, keypIndexes1, imgSi, matches12);
#endif
				if(gmerr != 0) return -6; //Error during the guided matching (too less matches remaining)
			}
			else//Perform matching on multiple CPU cores
			{
				const size_t nr_threads = 8;
				size_t nrFirstXKeys = keypoints1.size() / nr_threads;
				size_t keyPsIdx = 0;
				vector<size_t> nr_keypoints;
				vector<vector<cv::KeyPoint>> threadkeyPs;
				vector<cv::Mat> threadDescriptors;
				vector<vector<int>> threadkeypIndexes1;
				vector<vector<DMatch>> threadmatches12(nr_threads);
#if FILTER_WITH_CD_RATIOS
				vector<vector<mCostDist>> threadmprops12(nr_threads);
#endif
				for(size_t i = 0; i < nr_threads - 1; i++)
				{
					nr_keypoints.push_back(nrFirstXKeys);
				}
				nr_keypoints.push_back(keypoints1.size() - (nr_threads - 1) * nrFirstXKeys);
				for(size_t i = 0; i < nr_threads; i++)
				{
					threadkeyPs.push_back(vector<cv::KeyPoint>(keypoints1.begin() + keyPsIdx, keypoints1.begin() + keyPsIdx + nr_keypoints[i]));
					threadDescriptors.push_back(descriptors1.rowRange(keyPsIdx, keyPsIdx + nr_keypoints[i]).clone());
					threadkeypIndexes1.push_back(keypIndexes1);
					if(i > 0)
					{
						size_t delnegidx = 0;
						for(size_t j = 0; j < threadkeypIndexes1.back().size(); j++)
						{
							threadkeypIndexes1.back()[j] -= (int)keyPsIdx;
							if(threadkeypIndexes1.back()[j] < 0)
								delnegidx++;
						}
						threadkeypIndexes1.back().erase(threadkeypIndexes1.back().begin(), threadkeypIndexes1.back().begin() + delnegidx);
					}
					keyPsIdx += nr_keypoints[i];
				}
				//Perform multithreaded matching
				vector<int> errvec(nr_threads);
				#pragma omp parallel for
				for(int i = 0; i < (const int)nr_threads; i++)
				{
#if FILTER_WITH_CD_RATIOS
                    errvec[i] = guidedMatching(gridSearchParams, gridElemSize, threadkeyPs[i], threadDescriptors[i], descriptors2, keypts2idx, threadkeypIndexes1[i], imgSi, threadmatches12[i], threadmprops12[i]);
#else
                    errvec[i] = guidedMatching(gridSearchParams, gridElemSize, threadkeyPs[i], threadDescriptors[i], descriptors2, keypts2idx, threadkeypIndexes1[i], imgSi, threadmatches12[i]);
#endif
				}
				{
					int errcnt = 0;
					for(size_t i = 1; i < nr_threads; i++)
					{
						if(errvec[i] != 0)
							errcnt++;
					}
					if(errcnt == nr_threads)
						return -6; //Error during the guided matching (too less matches remaining)
				}
				keyPsIdx = nr_keypoints[0];
				for(size_t i = 1; i < nr_threads; i++)
				{
					for(size_t j = 0; j < threadmatches12[i].size(); j++)
					{
						threadmatches12[i][j].queryIdx += (int)keyPsIdx;
					}
					keyPsIdx += nr_keypoints[i];
				}
				matches12 = threadmatches12[0];
#if FILTER_WITH_CD_RATIOS
				mprops12 = threadmprops12[0];
#endif
				for(size_t i = 1; i < nr_threads; i++)
				{
					matches12.insert(matches12.end(), threadmatches12[i].begin(), threadmatches12[i].end());
#if FILTER_WITH_CD_RATIOS
					mprops12.insert(mprops12.end(), threadmprops12[i].begin(), threadmprops12[i].end());
#endif
				}
			}
			if(keypoints2.size() < 40)//Perform matching on single CPU
			{
                gmerr = guidedMatching(gridSearchParams_inv, gridElemSize, keypoints2, descriptors2, descriptors1, keypts1idx, keypIndexes2, imgSi, matches21/*, mprops21*/);
                if(gmerr != 0) return -6; //Error during the guided matching (too less matches remaining)
			}
			else//Perform matching on multiple CPU cores
			{
				const size_t nr_threads = 8;
				size_t nrFirstXKeys = keypoints2.size() / nr_threads;
				size_t keyPsIdx = 0;
				vector<size_t> nr_keypoints;
				vector<vector<cv::KeyPoint>> threadkeyPs;
				vector<cv::Mat> threadDescriptors;
				vector<vector<int>> threadkeypIndexes1;
				vector<vector<DMatch>> threadmatches12(nr_threads);
                //vector<vector<mCostDist>> threadmprops12(nr_threads);
				for(size_t i = 0; i < nr_threads - 1; i++)
				{
					nr_keypoints.push_back(nrFirstXKeys);
				}
				nr_keypoints.push_back(keypoints2.size() - (nr_threads - 1) * nrFirstXKeys);
				for(size_t i = 0; i < nr_threads; i++)
				{
					threadkeyPs.push_back(vector<cv::KeyPoint>(keypoints2.begin() + keyPsIdx, keypoints2.begin() + keyPsIdx + nr_keypoints[i]));
					threadDescriptors.push_back(descriptors2.rowRange(keyPsIdx, keyPsIdx + nr_keypoints[i]).clone());
					threadkeypIndexes1.push_back(keypIndexes2);
					if(i > 0)
					{
						size_t delnegidx = 0;
						for(size_t j = 0; j < threadkeypIndexes1.back().size(); j++)
						{
							threadkeypIndexes1.back()[j] -= (int)keyPsIdx;
							if(threadkeypIndexes1.back()[j] < 0)
								delnegidx++;
						}
						threadkeypIndexes1.back().erase(threadkeypIndexes1.back().begin(), threadkeypIndexes1.back().begin() + delnegidx);
					}
					keyPsIdx += nr_keypoints[i];
				}
				//Perform multithreaded matching
				vector<int> errvec(nr_threads);
				#pragma omp parallel for
				for(int i = 0; i < (const int)nr_threads; i++)
				{
                    errvec[i] = guidedMatching(gridSearchParams_inv, gridElemSize, threadkeyPs[i], threadDescriptors[i], descriptors1, keypts1idx, threadkeypIndexes1[i], imgSi, threadmatches12[i]/*, threadmprops12[i]*/);
				}
				{
					int errcnt = 0;
					for(size_t i = 1; i < nr_threads; i++)
					{
						if(errvec[i] != 0)
							errcnt++;
					}
					if(errcnt == nr_threads)
						return -6; //Error during the guided matching (too less matches remaining)
				}
				keyPsIdx = nr_keypoints[0];
				for(size_t i = 1; i < nr_threads; i++)
				{
					for(size_t j = 0; j < threadmatches12[i].size(); j++)
					{
						threadmatches12[i][j].queryIdx += (int)keyPsIdx;
					}
					keyPsIdx += nr_keypoints[i];
				}
				matches21 = threadmatches12[0];
				//mprops21 = threadmprops12[0];
				for(size_t i = 1; i < nr_threads; i++)
				{
					matches21.insert(matches21.end(), threadmatches12[i].begin(), threadmatches12[i].end());
					//mprops21.insert(mprops21.end(), threadmprops12[i].begin(), threadmprops12[i].end());
				}
			}

			sizeDiff = (int)(keypoints2.size()-matches21.size());
			for( size_t m = 0; m < matches12.size(); m++ )
			{
				for( int bk0 = matches12[m].trainIdx > sizeDiff ? matches12[m].trainIdx-sizeDiff:0; bk0 <= matches12[m].trainIdx; bk0++ )
				{
					if( matches21[bk0].trainIdx == matches12[m].queryIdx )
					{
						matches.push_back(matches12[m]);
#if FILTER_WITH_CD_RATIOS
						mprops.push_back(mprops12[m]);
#endif
						break;
					}
				}
			}
		}
		else
		{
			vector<vector<DMatch>> matches12,matches21;
#if FILTER_WITH_CD_RATIOS
			vector<vector<mCostDist>> mprops12/*,mprops21*/;
#endif
			size_t m21si;

			if(keypoints1.size() < 40) //Perform matching on single CPU
			{
				// ----->BE CAREFUL: the distance values in the DMatch structs represent the weights of the features and not the distances to the query keypoints
#if FILTER_WITH_CD_RATIOS
				gmerr = guidedMatching(gridSearchParams, gridElemSize, keypoints1, descriptors1, descriptors2, keypts2idx, keypIndexes1, imgSi, matches12, BFknn, mprops12);
#else
                gmerr = guidedMatching(gridSearchParams, gridElemSize, keypoints1, descriptors1, descriptors2, keypts2idx, keypIndexes1, imgSi, matches12, BFknn);
#endif
				if(gmerr != 0) return -6; //Error during the guided matching (too less matches remaining)
			}
			else //Perform matching on multiple CPU cores
			{
				const size_t nr_threads = 8;
				size_t nrFirstXKeys = keypoints1.size() / nr_threads;
				size_t keyPsIdx = 0;
				vector<size_t> nr_keypoints;
				vector<vector<cv::KeyPoint>> threadkeyPs;
				vector<cv::Mat> threadDescriptors;
				vector<vector<int>> threadkeypIndexes1;
				vector<vector<vector<DMatch>>> threadmatches12(nr_threads);
#if FILTER_WITH_CD_RATIOS
				vector<vector<vector<mCostDist>>> threadmprops12(nr_threads);
#endif
				for(size_t i = 0; i < nr_threads - 1; i++)
				{
					nr_keypoints.push_back(nrFirstXKeys);
				}
				nr_keypoints.push_back(keypoints1.size() - (nr_threads - 1) * nrFirstXKeys);
				for(size_t i = 0; i < nr_threads; i++)
				{
					threadkeyPs.push_back(vector<cv::KeyPoint>(keypoints1.begin() + keyPsIdx, keypoints1.begin() + keyPsIdx + nr_keypoints[i]));
					threadDescriptors.push_back(descriptors1.rowRange(keyPsIdx, keyPsIdx + nr_keypoints[i]).clone());
					threadkeypIndexes1.push_back(keypIndexes1);
					if(i > 0)
					{
						size_t delnegidx = 0;
						for(size_t j = 0; j < threadkeypIndexes1.back().size(); j++)
						{
							threadkeypIndexes1.back()[j] -= (int)keyPsIdx;
							if(threadkeypIndexes1.back()[j] < 0)
								delnegidx++;
						}
						threadkeypIndexes1.back().erase(threadkeypIndexes1.back().begin(), threadkeypIndexes1.back().begin() + delnegidx);
					}
					keyPsIdx += nr_keypoints[i];
				}
				//Perform multithreaded matching
				vector<int> errvec(nr_threads);
				#pragma omp parallel for
				for(int i = 0; i < (const int)nr_threads; i++)
				{
#if FILTER_WITH_CD_RATIOS
					errvec[i] = guidedMatching(gridSearchParams, gridElemSize, threadkeyPs[i], threadDescriptors[i], descriptors2, keypts2idx, threadkeypIndexes1[i], imgSi, threadmatches12[i], BFknn, threadmprops12[i]);
#else
                    errvec[i] = guidedMatching(gridSearchParams, gridElemSize, threadkeyPs[i], threadDescriptors[i], descriptors2, keypts2idx, threadkeypIndexes1[i], imgSi, threadmatches12[i], BFknn);
#endif
				}
				{
					int errcnt = 0;
					for(size_t i = 1; i < nr_threads; i++)
					{
						if(errvec[i] != 0)
							errcnt++;
					}
					if(errcnt == nr_threads)
						return -6; //Error during the guided matching (too less matches remaining)
				}
				keyPsIdx = nr_keypoints[0];
				for(size_t i = 1; i < nr_threads; i++)
				{
					for(size_t j = 0; j < threadmatches12[i].size(); j++)
					{
						for(size_t k = 0; k < threadmatches12[i][j].size(); k++)
						{
							threadmatches12[i][j][k].queryIdx += (int)keyPsIdx;
						}
					}
					keyPsIdx += nr_keypoints[i];
				}
				matches12 = threadmatches12[0];
#if FILTER_WITH_CD_RATIOS
				mprops12 = threadmprops12[0];
#endif
				for(size_t i = 1; i < nr_threads; i++)
				{
					matches12.insert(matches12.end(), threadmatches12[i].begin(), threadmatches12[i].end());
#if FILTER_WITH_CD_RATIOS
					mprops12.insert(mprops12.end(), threadmprops12[i].begin(), threadmprops12[i].end());
#endif
				}
			}

			//Ratio test
			/*{
				vector<vector<DMatch>> matches12_tmp;
#if FILTER_WITH_CD_RATIOS
				vector<vector<mCostDist>> mprops12_tmp;
#endif
				for(size_t q = 0; q < matches12.size(); q++)
				{
					if(matches12[q].size() > 1)
						if(matches12[q][0].distance < (0.75 * matches12[q][1].distance))
						{
							matches12_tmp.push_back(matches12[q]);
#if FILTER_WITH_CD_RATIOS
							mprops12_tmp.push_back(mprops12[q]);
#endif
						}
				}
				matches12 = matches12_tmp;
#if FILTER_WITH_CD_RATIOS
				mprops12 = mprops12_tmp;
#endif
			}*/
			if(BFknn != 2)
			{
				if(keypoints2.size() < 40)//Perform matching on single CPU
				{
                    gmerr = guidedMatching(gridSearchParams_inv, gridElemSize, keypoints2, descriptors2, descriptors1, keypts1idx, keypIndexes2, imgSi, matches21, BFknn/*, mprops21*/);
                    if(gmerr != 0) return -6; //Error during the guided matching (too less matches remaining)
				}
				else//Perform matching on multiple CPU cores
				{
					const size_t nr_threads = 8;
					size_t nrFirstXKeys = keypoints2.size() / nr_threads;
					size_t keyPsIdx = 0;
					vector<size_t> nr_keypoints;
					vector<vector<cv::KeyPoint>> threadkeyPs;
					vector<cv::Mat> threadDescriptors;
					vector<vector<int>> threadkeypIndexes1;
					vector<vector<vector<DMatch>>> threadmatches12(nr_threads);
					//vector<vector<vector<mCostDist>>> threadmprops12(nr_threads);
					for(size_t i = 0; i < nr_threads - 1; i++)
					{
						nr_keypoints.push_back(nrFirstXKeys);
					}
					nr_keypoints.push_back(keypoints2.size() - (nr_threads - 1) * nrFirstXKeys);
					for(size_t i = 0; i < nr_threads; i++)
					{
						threadkeyPs.push_back(vector<cv::KeyPoint>(keypoints2.begin() + keyPsIdx, keypoints2.begin() + keyPsIdx + nr_keypoints[i]));
						threadDescriptors.push_back(descriptors2.rowRange(keyPsIdx, keyPsIdx + nr_keypoints[i]).clone());
						threadkeypIndexes1.push_back(keypIndexes2);
						if(i > 0)
						{
							size_t delnegidx = 0;
							for(size_t j = 0; j < threadkeypIndexes1.back().size(); j++)
							{
								threadkeypIndexes1.back()[j] -= (int)keyPsIdx;
								if(threadkeypIndexes1.back()[j] < 0)
									delnegidx++;
							}
							threadkeypIndexes1.back().erase(threadkeypIndexes1.back().begin(), threadkeypIndexes1.back().begin() + delnegidx);
						}
						keyPsIdx += nr_keypoints[i];
					}
					//Perform multithreaded matching
					vector<int> errvec(nr_threads);
					#pragma omp parallel for
					for(int i = 0; i < (const int)nr_threads; i++)
					{
                        errvec[i] = guidedMatching(gridSearchParams_inv, gridElemSize, threadkeyPs[i], threadDescriptors[i], descriptors1, keypts1idx, threadkeypIndexes1[i], imgSi, threadmatches12[i], BFknn/*, threadmprops12[i]*/);
					}
					{
						int errcnt = 0;
						for(size_t i = 1; i < nr_threads; i++)
						{
							if(errvec[i] != 0)
								errcnt++;
						}
						if(errcnt == nr_threads)
							return -6; //Error during the guided matching (too less matches remaining)
					}
					keyPsIdx = nr_keypoints[0];
					for(size_t i = 1; i < nr_threads; i++)
					{
						for(size_t j = 0; j < threadmatches12[i].size(); j++)
						{
							for(size_t k = 0; k < threadmatches12[i][j].size(); k++)
							{
								threadmatches12[i][j][k].queryIdx += (int)keyPsIdx;
							}
						}
						keyPsIdx += nr_keypoints[i];
					}
					matches21 = threadmatches12[0];
					//mprops21 = threadmprops12[0];
					for(size_t i = 1; i < nr_threads; i++)
					{
						matches21.insert(matches21.end(), threadmatches12[i].begin(), threadmatches12[i].end());
						//mprops21.insert(mprops21.end(), threadmprops12[i].begin(), threadmprops12[i].end());
					}
				}

				//Ratio test
				/*{
					vector<vector<DMatch>> matches21_tmp;
					//vector<vector<mCostDist>> mprops21_tmp;
					for(size_t q = 0; q < matches21.size(); q++)
					{
						if(matches21[q].size() > 1)
							if(matches21[q][0].distance < (0.75 * matches21[q][1].distance))
							{
								matches21_tmp.push_back(matches21[q]);
								//mprops21_tmp.push_back(mprops21[q]);
							}
					}
					matches21 = matches21_tmp;
					//mprops21 = mprops21_tmp;
				}*/

				m21si = matches21.size()-1;
				sizeDiff = (int)(keypoints2.size() - matches21.size());

				for( size_t m = 0; m < matches12.size(); m++ )
				{
					bool findCrossCheck = false;
					for( size_t fk = 0; fk < matches12[m].size(); fk++ )
					{
						DMatch forward = matches12[m][fk];
#if FILTER_WITH_CD_RATIOS
						mCostDist mpforward = mprops12[m][fk];
#endif

						for( int bk0 = forward.trainIdx > sizeDiff ? forward.trainIdx-sizeDiff:0; bk0 <= (forward.trainIdx > (int)m21si ? (int)m21si:forward.trainIdx); bk0++ )
						{
							if(matches21[bk0][0].queryIdx == forward.trainIdx)
							{
								for( size_t bk = 0; bk < matches21[bk0].size(); bk++ )
								{
									DMatch backward = matches21[bk0][bk];
									if( backward.trainIdx == forward.queryIdx )
									{
										matches.push_back(forward);
#if FILTER_WITH_CD_RATIOS
										mprops.push_back(mpforward);
#endif
										findCrossCheck = true;
										break;
									}
								}
								break;
							}
						}
						if( findCrossCheck ) break;
					}
				}
			}
			else
			{
				//Ratio test
				{
					for(size_t q = 0; q < matches12.size(); q++)
					{
						if(matches12[q].size() > 1)
						{
							if(matches12[q][0].distance < (0.75f * matches12[q][1].distance))
							{
								matches.push_back(matches12[q][0]);
#if FILTER_WITH_CD_RATIOS
								mprops.push_back(mprops12[q][0]);
#endif
							}
						}
						else //If there is only 1 match, try to make a crosscheck to verify the correctness of the match
						{
							vector<int> keypIdxEmpty;
							vector<vector<DMatch>> matches212;
							Mat descriptors21 = descriptors2.row(matches12[q][0].trainIdx);
							vector<cv::KeyPoint> keypoints21;
							keypoints21.push_back(keypoints2[matches12[q][0].trainIdx]);
                            guidedMatching(gridSearchParams_inv, gridElemSize, keypoints21, descriptors21, descriptors1, keypts1idx, keypIdxEmpty, imgSi, matches212, BFknn);
							if(!matches212.empty() && (matches212[0].size() > 1))
							{
								//If the best match from backward search is the same match as from forward search and if the ratio test holds, save the match
								if((matches212[0][0].trainIdx == matches12[q][0].queryIdx) && (matches212[0][0].distance < (0.75f * matches212[0][1].distance)))
								{
									matches.push_back(matches12[q][0]);
#if FILTER_WITH_CD_RATIOS
									mprops.push_back(mprops12[q][0]);
#endif
								}
							}
							else //If only one match was found, discard it if the distance the the estimated position is too high
							{
#if FILTER_WITH_CD_RATIOS == 0
								unsigned int grid_x, grid_y;
								float esti_pos[3], keypDiff[2];
								int singleQueryIdx = matches12[q][0].queryIdx;
								grid_x = (unsigned int)floor(keypoints1[singleQueryIdx].pt.x / gridElemSize);
								grid_y = (unsigned int)floor(keypoints1[singleQueryIdx].pt.y / gridElemSize);
								if(grid_y < gridSearchParams.size())
								{
									if(grid_x < gridSearchParams[grid_y].size())
									{
										esti_pos[0] = keypoints1[singleQueryIdx].pt.x + gridSearchParams[grid_y][grid_x].x;
										esti_pos[1] = keypoints1[singleQueryIdx].pt.y + gridSearchParams[grid_y][grid_x].y;
										esti_pos[2] = gridSearchParams[grid_y][grid_x].z;
									}
									else
									{
										esti_pos[0] = keypoints1[singleQueryIdx].pt.x + gridSearchParams[grid_y].back().x;
										esti_pos[1] = keypoints1[singleQueryIdx].pt.y + gridSearchParams[grid_y].back().y;
										esti_pos[2] = gridSearchParams[grid_y].back().z;
									}
								}
								else
								{
									if(grid_x < gridSearchParams[grid_y].size())
									{
										esti_pos[0] = keypoints1[singleQueryIdx].pt.x + gridSearchParams.back()[grid_x].x;
										esti_pos[1] = keypoints1[singleQueryIdx].pt.y + gridSearchParams.back()[grid_x].y;
										esti_pos[2] = gridSearchParams.back()[grid_x].z;
									}
									else
									{
										esti_pos[0] = keypoints1[singleQueryIdx].pt.x + gridSearchParams.back().back().x;
										esti_pos[1] = keypoints1[singleQueryIdx].pt.y + gridSearchParams.back().back().y;
										esti_pos[2] = gridSearchParams.back().back().z;
									}
								}
								keypDiff[0] = keypoints21[0].pt.x - esti_pos[0];
								keypDiff[1] = keypoints21[0].pt.y - esti_pos[1];
								keypDiff[0] *= keypDiff[0];
								keypDiff[1] *= keypDiff[1];
								keypDiff[0] = std::sqrt(keypDiff[0] + keypDiff[1]);
								keypDiff[1] = keypDiff[0] / (esti_pos[2] + 0.5f);
#else
								float keypDiff[2];
								keypDiff[0] = sqrtf(mprops12[q][0].distance);
								keypDiff[1] = keypDiff[0] / (gridSearchParams[mprops12[q][0].y][mprops12[q][0].x].z + 0.5f);
#endif

								if(keypDiff[1] < 0.66f) //Accept it if the distance to the estimated position is smaller 66% of the search range (= 1 sigma if the range was calculated with 3.5 * sigma)
								{
									matches.push_back(matches12[q][0]);
#if FILTER_WITH_CD_RATIOS
									mprops.push_back(mprops12[q][0]);
#endif
								}
							}
						}
					}
				}
			}
		}
	}

#if FILTER_WITH_CD_RATIOS
	//Calculate the distances from the squared distances for every matching right keypoint to its SOF estimate
	for(size_t i = 0; i < mprops.size(); i++)
	{
		mprops[i].distance = std::sqrtf(mprops[i].distance);
	}
#endif

	/*//Calculate the flow between the correspondences
	vector<Point2f> flowInit;
	for(int i = 0; i<keypInit1Filt.size();i++)
		flowInit.push_back(keypInit2Filt[i].first.pt-keypInit1Filt[i].first.pt);

	//Filter the keypoints according to the optical flow result
	vector<pair<Point2f,int>> flowNormAngle;
	float flowNormMed, flowAngleMed, flowNormSdev=0, flowAngleSdev=0;
	for(int i = 0; i<flowInit.size();i++)
		flowNormAngle.push_back(make_pair(Point2f(powf(flowInit[i].x,2)+powf(flowInit[i].y,2),
								atan2f(flowInit[i].y,flowInit[i].x)),i));

	std::sort(flowNormAngle.begin(),flowNormAngle.end(),sortFlowNormIdx);
	if(flowNormAngle.size() % 2)
		flowNormMed = flowNormAngle[(flowNormAngle.size()-1)/2].first.x;
	else
		flowNormMed = (flowNormAngle[flowNormAngle.size()/2].first.x + flowNormAngle[flowNormAngle.size()/2-1].first.x)/2;

	for(int i = (int)floorf((float)flowNormAngle.size()*0.1);i<(int)ceilf((float)flowNormAngle.size()*0.9);i++)
		flowNormSdev += powf(flowNormAngle[i].first.x - flowNormMed,2.0);
	flowNormSdev = std::sqrtf(flowNormSdev/(flowNormAngle.size()-1));

	idx = 0;
	float thresh[2] = {flowNormMed-3.5*flowNormSdev, flowNormMed+3.5*flowNormSdev};
	while(idx < flowNormAngle.size())
	{
		if((flowNormAngle[idx].first.x < thresh[0]) || (flowNormAngle[idx].first.x > thresh[1]))
		{
			flowNormAngle.erase(flowNormAngle.begin()+idx,flowNormAngle.begin()+idx+1);
		}
		else idx++;
	}

	std::sort(flowNormAngle.begin(),flowNormAngle.end(),sortFlowAngleIdx);
	if(flowNormAngle.size() % 2)
		flowAngleMed = flowNormAngle[(flowNormAngle.size()-1)/2].first.y;
	else
		flowAngleMed = (flowNormAngle[flowNormAngle.size()/2].first.y + flowNormAngle[flowNormAngle.size()/2-1].first.y)/2;

	for(int i = (int)floorf((float)flowNormAngle.size()*0.1);i<(int)ceilf((float)flowNormAngle.size()*0.9);i++)
		flowAngleSdev += powf(flowNormAngle[i].first.y - flowAngleMed,2.0);
	flowAngleSdev = std::sqrtf(flowAngleSdev/(flowNormAngle.size()-1));

	idx = 0;
	thresh[0] = flowAngleMed-3.5*flowAngleSdev;
	thresh[1] = flowAngleMed+3.5*flowAngleSdev;
	while(idx < flowNormAngle.size())
	{
		if((flowNormAngle[idx].first.y < thresh[0]) || (flowNormAngle[idx].first.y > thresh[1]))
		{
			flowNormAngle.erase(flowNormAngle.begin()+idx,flowNormAngle.begin()+idx+1);
		}
		else idx++;
	}*/


	/*//Filter points according to their distances to the corresponding epipolar lines
	idx = 0;
	F = findFundamentalMat(filtPoints1, filtPoints2, CV_FM_8POINT);
	Mat Ft = F.t();
	vector<int> delIdx;
	while(idx < keypInit1Filt.size())
	{
		double err;
		static double thresh = 2*std::pow(3.0,2.0);
		Mat l1, l2, x1(1,3,F.type()), x2(1,3,F.type());
		filtPoints1.row(idx).convertTo(x1.colRange(0,2),F.type());
		x1.at<double>(2) = 1.0;
		filtPoints2.row(idx).convertTo(x2.colRange(0,2),F.type());
		x2.at<double>(2) = 1.0;
		l2 = x1 * Ft; // Epipolar lines in the second image
		l1 = x2 * F; // Epipolar lines in the first image
		err = pow(x1.dot(l1)/sqrt(pow(l1.at<double>(0),2.0)+pow(l1.at<double>(1),2.0)),2.0)+
			  pow(x2.dot(l2)/sqrt(pow(l2.at<double>(0),2.0)+pow(l2.at<double>(1),2.0)),2.0);
		if(err > thresh)
		{
			keypInit1Filt.erase(keypInit1Filt.begin()+idx,keypInit1Filt.begin()+idx+1);
			keypInit2Filt.erase(keypInit2Filt.begin()+idx,keypInit2Filt.begin()+idx+1);
			points = filtPoints1.rowRange(0,idx);
			points.push_back(filtPoints1.rowRange(idx+1,filtPoints1.size().height));
			filtPoints1 = points;
			points = filtPoints2.rowRange(0,idx);
			points.push_back(filtPoints2.rowRange(idx+1,filtPoints2.size().height));
			filtPoints2 = points;
			delIdx.push_back(idx+delIdx.size());
		}
		else idx++;
	}


	// FOR DEBUGGING - SHOW THE MATCHES AFTER FILTERING
	if(!img1.empty() && !img2.empty())
	{
		vector<char> matchesMask( initMatches.size(), 0 );
		vector<KeyPoint> keyp1, keyp2;
		Mat drawImg;
		int idx1 = 0;

		for( int i = 0; i < trainIdxs.size(); i++)
			if( inliers.at<bool>(i,0) == true )
			{
				if(!delIdx.empty())
					if(delIdx[idx1] == i)
					{
							if(delIdx.size()> idx1-1) idx1++;
							continue;
					}
				matchesMask[i] = 1;
			}

		for(int i = 0; i<keypInit1.size();i++)
			keyp1.push_back(keypInit1[i].first);
		for(int i = 0; i<keypInit2.size();i++)
			keyp2.push_back(keypInit2[i].first);

		drawMatches( img1.getMat(), keyp1, img2.getMat(), keyp2, initMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask);
		cv::imshow( "Second filtering step of initial matches", drawImg );
		cv::waitKey(0);
	}
	delIdx.clear();


	keypInit1.clear();
	keypInit2.clear();*/




	//A filter could be inserted here to ensure the same density of matches throughout the whole image (remove matches in areas of high density of matches)

	filteredMatches12.reserve(matches.size()+filteredMatches12.size());
	filteredMatches12.insert(filteredMatches12.end(),matches.begin(),matches.end());

#if FILTER_WITH_CD_RATIOS
	std::vector<float> quartDfactors, quartCfactors;
	mprops_init.reserve(mprops.size() + mprops_init.size());
	mprops_init.insert(mprops_init.end(), mprops.begin(), mprops.end());
	getMeanDistCostFactors(distRatios_tmp, costRatios_tmp, quartDfactors, quartCfactors, mprops_init, gridSearchParams[0].size(), gridSearchParams.size());
#endif

#if FILTER_WITH_CD_RATIOS == 2
	//float cost_th = 1.25f, dist_th = 2.0f;
	float cost_th = 1.5f, dist_th = 2.0f;
	//vector<bool> match_ndel_vec(filteredMatches12.size(), false);
	int delsize = 0;
	vector<cv::DMatch> filteredMatches12_tmp;
	vector<float> costRatios_tmp1, distRatios_tmp1;
	/*if(descriptors1.type() == CV_32F)
	{
		if(estim_inlRatio >= 0.75)
		{
			cost_th = 3.0f;
			dist_th = 3.25f;
		}
		else if(estim_inlRatio > 0.45)
		{
			cost_th = 5.833f * (float)estim_inlRatio - 1.375f;
			dist_th = 4.167f * (float)estim_inlRatio + 0.125f;
		}
	}
	else if(descriptors1.type() == CV_8U)
	{
		if(estim_inlRatio >= 0.44)
		{
			cost_th = 3.0f;
			dist_th = 3.25f;
		}
		else if(estim_inlRatio > 0.23)
		{
			cost_th = 8.333f * (float)estim_inlRatio - 0.667f;
			dist_th = 5.952f * (float)estim_inlRatio + 0.631f;
		}
	}*/
	filteredMatches12_tmp.reserve(filteredMatches12.size());
	costRatios_tmp1.reserve(filteredMatches12.size());
	distRatios_tmp1.reserve(filteredMatches12.size());
	for(size_t i = 0; i < filteredMatches12.size(); i++)
	{
		if(((distRatios_tmp[i] <= dist_th) && (costRatios_tmp[i] <= cost_th) &&
			(distRatios_tmp[i] <= 1.25 * quartDfactors[i]) &&
			(costRatios_tmp[i] <= 1.25 * quartCfactors[i])) ||
			((quartDfactors[i] > 3.0f) && (quartCfactors[i] > 2.0f)))
		{
			filteredMatches12_tmp.push_back(filteredMatches12[i]);
			costRatios_tmp1.push_back(costRatios_tmp[i]);
			distRatios_tmp1.push_back(distRatios_tmp[i]);
		}
		//if(((distRatios_tmp[i] > dist_th) && (costRatios_tmp[i] > cost_th) &&
		//	(distRatios_tmp[i] > 1.25 * quartDfactors[i]) &&
		//	(costRatios_tmp[i] > 1.25 * quartCfactors[i])) /*&&
		//	((quartDfactors[i] <= 3.0f) && (quartCfactors[i] <= 2.0f))*/)
		//{
		//	match_ndel_vec[i] = true;
		//	delsize++;
		//}
	}
	/*Eigen::Vector2f x1_tmp;
	EMatFloat2 eigdelpts(delsize,2);
	std::vector<std::pair<size_t,float> >   del_neighbors;
	int idxe = 0;
	float search_delrad;
	int maxareadels = (int)std::floor(0.2f * (float)delsize + 0.5f);
	if(maxareadels == 0) maxareadels = 1;
	for(unsigned int i = 0;i < filteredMatches12.size();i++)
	{
		if(match_ndel_vec[i])
		{
			eigdelpts(idxe,0) = keypoints1.at(filteredMatches12[i].queryIdx).pt.x;
			eigdelpts(idxe,1) = keypoints1.at(filteredMatches12[i].queryIdx).pt.y;
			idxe++;
		}
	}

	KDTree_D2float keyptsdelidx(2,eigdelpts,maxLeafNum);
	keyptsdelidx.index->buildIndex();

	idxe = 0;
	for(unsigned int i = 0;i < filteredMatches12.size();i++)
	{
		if(match_ndel_vec[i])
		{
			x1_tmp(0) = eigdelpts(idxe,0);
			x1_tmp(1) = eigdelpts(idxe,1);
			search_delrad = 1.2 * gridSearchParams[mprops_init[i].y][mprops_init[i].x].z;
			if(search_delrad < 10) search_delrad = 10;
			if(search_delrad > 20) search_delrad = 20;
			search_delrad *= search_delrad;
			del_neighbors.clear();
			keyptsdelidx.index->radiusSearch(&x1_tmp(0),search_delrad,del_neighbors,nanoflann::SearchParams(maxDepthSearch));
			idxe++;
			if(del_neighbors.size() > maxareadels)
				match_ndel_vec[i] = false;
		}
	}
	for(unsigned int i = 0;i < filteredMatches12.size();i++)
	{
		if(!match_ndel_vec[i])
		{
			filteredMatches12_tmp.push_back(filteredMatches12[i]);
			costRatios_tmp1.push_back(costRatios_tmp[i]);
			distRatios_tmp1.push_back(distRatios_tmp[i]);
		}
	}*/
	filteredMatches12 = filteredMatches12_tmp;
	costRatios_tmp = costRatios_tmp1;
	distRatios_tmp = distRatios_tmp1;
#endif
#if FILTER_WITH_CD_RATIOS
	if((costRatios != NULL) && (distRatios != NULL))
	{
		*costRatios = costRatios_tmp;
		*distRatios = distRatios_tmp;
	}
#endif

	// FOR DEBUGGING - SHOW ALL MATCHES
	if(!img1.empty() && !img2.empty())
	{
		Mat drawImg;

		drawMatches( img1.getMat(), keypoints1, img2.getMat(), keypoints2, filteredMatches12, drawImg );
		//imwrite("C:\\work\\matches_final.jpg", drawImg);
		cv::imshow( "All Matches", drawImg );
		cv::waitKey(0);

		//Show reduced set of matches
		{
			Mat img_match;
			std::vector<cv::KeyPoint> keypL_reduced;//Left keypoints
			std::vector<cv::KeyPoint> keypR_reduced;//Right keypoints
			std::vector<cv::DMatch> matches_reduced;
			std::vector<cv::KeyPoint> keypL_reduced1;//Left keypoints
			std::vector<cv::KeyPoint> keypR_reduced1;//Right keypoints
			std::vector<cv::DMatch> matches_reduced1;
			int j = 0;
			size_t keepNMatches = 100;
			if(filteredMatches12.size() > keepNMatches)
			{
				size_t keepXthMatch = filteredMatches12.size() / keepNMatches;
				for (unsigned int i = 0; i < filteredMatches12.size(); i++)
				{
					int idx = filteredMatches12[i].queryIdx;
					keypL_reduced.push_back(keypoints1[idx]);
					matches_reduced.push_back(filteredMatches12[i]);
					matches_reduced.back().queryIdx = i;
					keypR_reduced.push_back(keypoints2[matches_reduced.back().trainIdx]);
					matches_reduced.back().trainIdx = i;
				}
				j = 0;
				for (unsigned int i = 0; i < matches_reduced.size(); i++)
				{
					if((i % (int)keepXthMatch) == 0)
					{
						keypL_reduced1.push_back(keypL_reduced[i]);
						matches_reduced1.push_back(matches_reduced[i]);
						matches_reduced1.back().queryIdx = j;
						keypR_reduced1.push_back(keypR_reduced[i]);
						matches_reduced1.back().trainIdx = j;
						j++;
					}
				}
				drawMatches(img1.getMat(), keypL_reduced1, img2.getMat(), keypR_reduced1, matches_reduced1, img_match);
				imshow("Approx. 100 found matches", img_match);
				waitKey(0);
			}
		}
	}

	if( (filteredMatches12.size() < MIN_FINAL_MATCHES) /*|| (filteredMatches12.size() < keypoints1.size()/3)*/) return -7; //Too less remaining matches -> There might be a problem with your data

	return 0;
}


/* This function interpolates a grid of bins with statistical optical flow (SOF) values to get a smoother transition between the elements.
 * One bin within the grid is split up into a subgrid with nxn sub-bins, whereas all sub-bins situated at the border of the original
 * bin are interpolated and the other sub-bins contain the same SOF as in the original bin.
 *
 * vector<vector<Point3f>> gridSearchParams		Input & Output  ->	The vector-structure corresponds to the size of a grid in the image
 *																	(the outer vector corresponds to the rows or
 *																	y-coordinates and the inner vector corresponds to the
 *																	columns or x-coordinates). Each vector-element holds
 *																	3 values. The first corresponds to the average flow
 *																	in x- and the second to the average flow in y-direction
 *																	within the grid element. The third value is a multiple of
 *																	the standard deviation of the flow.
 * float gridElemSize							Input & Output  ->	Size of one grid element in the image used by gridSearchParams
 * Size imgSi									Input			->	The size of the image
 *
 * Return value:								none
 */
void interpolStatOptFlow(std::vector<std::vector<cv::Point3f>> & gridSearchParams, float & gridElemSize, cv::Size imgSi)
{
	if(gridSearchParams.size() == 1)
		return;

	const unsigned int gigs = 5; //grid in grid size e.g. 5x5 grid in one original bin
	unsigned int idx_max = gigs - 1;
	float remaining_pix = fmod((float)imgSi.width, gridElemSize);
	float gridSi = gridElemSize / (float)gigs;
	size_t x_last_max;
	size_t origGridSi[2];
	origGridSi[0] = gridSearchParams[0].size() - 1;
	origGridSi[1] = gridSearchParams.size() - 1;
	if(remaining_pix < 25.0f)
	{
		x_last_max = gigs;
	}
	else
	{
		remaining_pix = remaining_pix / gridSi;
		x_last_max = (size_t)floor(remaining_pix);
		if(remaining_pix - floor(remaining_pix) >= 0.5f)
		{
			x_last_max++;
		}
		if(x_last_max == 0)
		{
			if((remaining_pix - floor(remaining_pix)) * gridSi < 25.0f)
			{
				cout << "This shouldnt happen during SOF interpolation!" << endl;
				return;
			}
			x_last_max++;
		}
	}

	std::vector<std::vector<cv::Point3f>> gridSearchParams_new;//(gridSearchParams.size() * gigs);

	for(size_t y1 = 0; y1 < gridSearchParams.size() * gigs; y1++)
	{
		gridSearchParams_new.push_back(std::vector<cv::Point3f>((gridSearchParams[0].size() - 1) * gigs + x_last_max));
	}

	size_t idx_x, idx_y;
	for(size_t y = 0; y < gridSearchParams.size(); y++)
	{
		for(size_t x = 0; x < gridSearchParams[0].size(); x++)
		{
			for(size_t y1 = 0; y1 < gigs; y1++)
			{
				idx_y = y * gigs + y1;
				for(size_t x1 = 0; x1 < gigs; x1++)
				{
					if((x == origGridSi[0]) && (x1 == x_last_max))
					{
						break;
					}
					idx_x = x * gigs + x1;
					if(((x1 > 0) && (x1 < idx_max)) && ((y1 > 0) && (y1 < idx_max))) //new bins not at the borders of a old bin
					{
						gridSearchParams_new[idx_y][idx_x] = gridSearchParams[y][x];
					}
					else if((x1 == 0) && ((y1 > 0) && (y1 < idx_max))) //new bins at the left border exclusive upper and lower border
					{
						if(x == 0) //if there is no left neighbor
						{
							gridSearchParams_new[idx_y][idx_x] = gridSearchParams[y][x];
						}
						else
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y][x - 1]);
						}
					}
					else if((x1 == idx_max) && ((y1 > 0) && (y1 < idx_max))) //new bins at the right border exclusive upper and lower border
					{
						if(x == origGridSi[0]) //if there is no right neighbor
						{
							gridSearchParams_new[idx_y][idx_x] = gridSearchParams[y][x];
						}
						else
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y][x + 1]);
						}
					}
					else if((y1 == 0) && ((x1 > 0) && (x1 < idx_max))) //new bins at the upper border exclusive left and right border
					{
						if(y == 0) //if there is no upper neighbor
						{
							gridSearchParams_new[idx_y][idx_x] = gridSearchParams[y][x];
						}
						else
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y - 1][x]);
						}
					}
					else if((y1 == idx_max) && ((x1 > 0) && (x1 < idx_max))) //new bins at the lower border exclusive left and right border
					{
						if(y == origGridSi[1]) //if there is no lower neighbor
						{
							gridSearchParams_new[idx_y][idx_x] = gridSearchParams[y][x];
						}
						else
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y + 1][x]);
						}
					}
					else if((x1 == 0) && (y1 == 0)) //new bins in the left upper corner
					{
						if((x == 0) && (y == 0)) //if it is in the most left and upper corner of the whole grid
						{
							gridSearchParams_new[idx_y][idx_x] = gridSearchParams[y][x];
						}
						else if(x == 0)
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y - 1][x]);
						}
						else if(y == 0)
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y][x - 1]);
						}
						else
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y][x - 1], &gridSearchParams[y - 1][x]);
						}
					}
					else if((x1 == idx_max) && (y1 == 0)) //new bins in the right upper corner
					{
						if((x == origGridSi[0]) && (y == 0))
						{
							gridSearchParams_new[idx_y][idx_x] = gridSearchParams[y][x];
						}
						else if(x == origGridSi[0])
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y - 1][x]);
						}
						else if(y == 0)
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y][x + 1]);
						}
						else
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y][x + 1], &gridSearchParams[y - 1][x]);
						}
					}
					else if((x1 == 0) && (y1 == idx_max)) //new bins in the left lower corner
					{
						if((x == 0) && (y == origGridSi[1]))
						{
							gridSearchParams_new[idx_y][idx_x] = gridSearchParams[y][x];
						}
						else if(x == 0)
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y + 1][x]);
						}
						else if(y == origGridSi[1])
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y][x - 1]);
						}
						else
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y][x - 1], &gridSearchParams[y + 1][x]);
						}
					}
					else if((x1 == idx_max) && (y1 == idx_max)) //new bins in the right lower corner
					{
						if((x == origGridSi[0]) && (y == origGridSi[1]))
						{
							gridSearchParams_new[idx_y][idx_x] = gridSearchParams[y][x];
						}
						else if(x == origGridSi[0])
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y + 1][x]);
						}
						else if(y == origGridSi[1])
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y][x + 1]);
						}
						else
						{
							gridSearchParams_new[idx_y][idx_x] = interpolFlowRad(&gridSearchParams[y][x], &gridSearchParams[y][x + 1], &gridSearchParams[y + 1][x]);
						}
					}
				}
			}
		}
	}

	gridSearchParams = gridSearchParams_new;
	gridElemSize = gridSi;
}


/* This function interpolates calculates a mean value for 2 or 3 statistical optical flow (vector & search radius) values.
 * If 2 values are given, the adress of f3 must be NULL. If 3 values are given, the first value is weighted two times more
 * than the other values (For corners in the stat. optical flow (SOF) grid where 2 border lines adjoin the same SOF bin).
 *
 * cv::Point3f *f1					Input  -> First SOF element (is weighted 2 times more than the other if 3 elements are given)
 * cv::Point3f *f2					Input  -> Second SOF element
 * cv::Point3f *f3					Input  -> Third SOF element (if NULL, only the first two are used)
 *
 * Return value:					The interpolated SOF
 */
cv::Point3f interpolFlowRad(cv::Point3f *f1, cv::Point3f *f2, cv::Point3f *f3)
{
	cv::Point3f tmp;
	if(f3 == NULL)
	{
		cv::Point2f r_neu;
		tmp.x = (f1->x + f2->x) / 2.0f;
		tmp.y = (f1->y + f2->y) / 2.0f;

		r_neu.x = f1->x - f2->x;
		r_neu.y = f1->y - f2->y;
        tmp.z = std::max(f1->z, f2->z) + sqrtf(r_neu.x * r_neu.x + r_neu.y * r_neu.y);
	}
	else
	{
		cv::Point2f r_neu1, r_neu2, r_neu3;
		float rn[3];
		tmp.x = (2.0f * f1->x + f2->x + f3->x) / 4.0f;
		tmp.y = (2.0f * f1->y + f2->y + f3->y) / 4.0f;
		r_neu1.x = f1->x - tmp.x;
		r_neu1.y = f1->y - tmp.y;
		r_neu2.x = f2->x - tmp.x;
		r_neu2.y = f2->y - tmp.y;
		r_neu3.x = f3->x - tmp.x;
		r_neu3.y = f3->y - tmp.y;
        rn[0] = sqrtf(r_neu1.x * r_neu1.x + r_neu1.y * r_neu1.y);
        rn[1] = sqrtf(r_neu2.x * r_neu2.x + r_neu2.y * r_neu2.y);
        rn[2] = sqrtf(r_neu3.x * r_neu3.x + r_neu3.y * r_neu3.y);
		rn[0] += f1->z;
		rn[1] += f2->z;
		rn[2] += f3->z;
		tmp.z = std::max(rn[0], std::max(rn[1], rn[2]));
	}

	return tmp;
}


/* This function compares the response of two keypoints to be able to sort them accordingly
 * while keeping track of the index.
 *
 * KeyPoint first				Input  -> First pair of keypoint and index
 * KeyPoint second				Input  -> Second pair of keypoint and index
 */
bool sortKeyPointsIdx(std::pair<cv::KeyPoint,int> first, std::pair<cv::KeyPoint,int> second)
{
	return first.first.response > second.first.response;
}


/* This function matches keypoints from two images and returns only matches for which the ratio between the
 * distances of the best and second-closest match are below a threshold.
 *
 * Ptr<DescriptorMatcher>& descriptorMatcher	Input  -> The matcher-object
 * const Mat& descriptors1						Input  -> The descriptors of the keypoints in the left (first) image
 * const Mat& descriptors2						Input  -> The descriptors of the keypoints in the right (second) image
 * vector<DMatch>& filteredMatches12			Output -> Fitered matches
 *
 * Return value:								true:		  Everything ok
 *												false:		  Too less matches are left
 */
bool ratioTestMatches(cv::Ptr<DescriptorMatcher>& descriptorMatcher,
                         const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                         std::vector<DMatch>& filteredMatches12)
{
	filteredMatches12.clear();
    vector<vector<DMatch> > matches12;
	descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, 2 );

	//Ratio test
	for(size_t q = 0; q < matches12.size(); q++)
	{
		if(matches12[q][0].distance < (0.75 * matches12[q][1].distance))
			filteredMatches12.push_back(matches12[q][0]);
	}

	if(filteredMatches12.size() < 15)
		return false;

	return true;
}


/* Generates a grid-based statistical flow for many areas over the whole image. Moreover, a map for filtering
 * the pre-matched keypoints keyP1 and keyP2 is generated based on the statistical result. The pre-matching can
 * e.g. be performed using the Bruteforce matcher.
 *
 * EMatFloat2 keyP1								Input  -> Matched keypoints of the left image in the correct
 *														  order to match keyP2
 * EMatFloat2 keyP2								Input  -> Matched keypoints of the right image in the correct
 *														  order to match keyP1
 * Size imgSi									Input  -> Size of the image
 * vector<vector<Point3f>> gridSearchParams		Output -> The vector-structure corresponds to the size of the
 *														  grid (the outer vector corresponds to the rows or
 *														  y-coordinates and the inner vector corresponds to the
 *														  columns or x-coordinates). Each vector-element holds
 *														  3 values. The first corresponds to the average flow
 *														  in x- and the second to the average flow in y-direction
 *														  within the grid element. The third value is a multiple of
 *														  the standard deviation of the flow.
 * float *gridElemSize							Output -> Size of one grid element (pixel bin)
 * OutputArray mask								Output -> If provided, a mask marking invalid correspondences
 *														  is returned
 * bool filterSmallFlow							Input  -> If true [default=false], matches with a too small flow are marked
 *														  as invalid in the mask
 * unsigned int minYGridSize					Input  -> If provided, the minimal grid size in y-direction can
 *														  be specified (The output grid can differ if too less
 *														  correspondences are provided). If this value is set
 *														  to 0, the grid size is computed automatically using
 *														  a 100 pixel grid [standard].
 * double validationTH							Input  -> Threshold, above which the ratio (mean - median)/mean signals
 *														  an invalid flow field, The adjustable range lies between 0.1 and 1.0.
 *														  [Default = 0.3]
 * KDTree_D2float *keypts1idxall				Input  -> Pointer to the KD-tree (nanoflann) which includes all keypoint positions
 *														  in the left image to get image regions where no statistical flow must be
 *														  calculated. If NULL [Default] the statistical flow is generated for the
 *														  whole image (useful for only filtering correspondences).
 * double stdMult_								Input  -> Multiplication factor for the standard deviation which is used to generate
 *														  the thresholds and the search range (e.g. range = mean + stdMult * sigma).
 *														  The adjustable range lies between 1.0 and 7.0. [Default = 3.5]
 *
 * Return value:								0:		  Everything ok
 *												-1:		  Calculation of the statistic failed
 */
int getStatisticalMatchingPositions(const EMatFloat2 keyP1, const EMatFloat2 keyP2, cv::Size imgSi,
									std::vector<std::vector<cv::Point3f>> & gridSearchParams, float *gridElemSize,
									cv::OutputArray mask, bool filterSmallFlow, unsigned int minYGridSize,
									double validationTH, KDTree_D2float *keypts1idxall, double stdMult_)
{
	const int remainGridPix = 25; //If the last column of the grid in the image is smaller than this, it is ignored -> dont touch this (the same value is used in interpolStatOptFlow)
	const int maxDepthSearch = 32;
	const int minPtsPerSqr = 16;
	const int maxLeafNum = 20;
	double stdMult = 3.5; //Multiplier for the standard deviation to generate the search range (r = stdMult * sigma)
	const double stdMult_th = 4.0; //Multiplier for the standard deviation to generate a threshold (th = mu + stdMult_th * sigma)
	const double validThAngDivFact = 6.0; //Conversion factor for the validation threshold to validate the angle statistic
	const double minAbsDistDiff = 0.5; //Minimum absolut difference between mean and median distance -> Differences below are not filtered out
	const float th_smallFlow = MIN_FLOW_TH; //Threshold for filtering matches with a too small flow (less than this value)
	int roundNewGridCalcs = 0;
	double validationTH_; //Validation threshold for distance statistics
	double validationTHang; //Validation threshold for angle statistics
	if(validationTH > 1.0)
		validationTH_ = 1.0;
	else if(validationTH < 0.1)
		validationTH_ = 0.1;
	else
		validationTH_ = validationTH;
	validationTHang = validationTH_ / validThAngDivFact;
	if(stdMult_ > 7.0)
		stdMult = 7.0;
	else if(stdMult_ < 1.0)
		stdMult = 1.0;
	else
		stdMult = stdMult_;

	if(keyP1.rows() < minPtsPerSqr)
		return -1;

	if(keyP1.rows() != keyP2.rows())
		return -1;

	//vector<pair<Point3i,vector<std::pair<size_t,float>>>> validSqrs, lessPtSqrs;
	qualityParm stat_dist_glob, stat_ang_glob;
	vector<vector<std::pair<size_t,float>>> gridTreeMatches;
	vector<qualityParm> stat_dist, stat_ang;
	vector<vector<double>> flow_dist, flow_ang;
	bool globStatCalculated = false;
	std::vector<std::pair<size_t,float> >   ret_matches;
	int divx, divy, idx;
	float imgpart, lastwidthpart, xpos, imgpart2, imgpart22, radius1;
	EMatFloat2 gridPoints;//(divx*divy,2);
	Eigen::Matrix<float,Eigen::Dynamic,1> gridX;//(divx,1);
	Eigen::Matrix<float,1,1> gridY;
	Eigen::MatrixXi gridElemType;
	int usedGridElems = 0;

	//Generate the KD-tree index for the keypoint coordinates
	KDTree_D2float keypts1idx(2,keyP1,maxLeafNum);
	keypts1idx.index->buildIndex();

	//Generate image grid
	if(keyP1.rows() < 2 * minPtsPerSqr)
	{
		divy = 1;
	}
	else
	{
		if(!minYGridSize)
			divy = (int)floor((float)imgSi.height/100.0);
		else
			divy = (int)minYGridSize;
		float xyRatio = (float)imgSi.width/(float)imgSi.height;
		if(keypts1idxall)
		{
			imgpart = (float)imgSi.height/(float)divy;
			*gridElemSize = imgpart;
			divx = (int)floor((float)imgSi.width/imgpart);
			lastwidthpart = (float)imgSi.width-(float)divx*imgpart;
			imgpart2 = imgpart/2;
			if(lastwidthpart > remainGridPix) //if the remaining column of the image is too small forget it
				divx++;
			gridElemType = Eigen::MatrixXi::Ones(divy,divx);
			usedGridElems = 0;
			gridPoints.resize(divx*divy,2);
			gridX.resize(divx,1);
			gridX(0,0) = gridY(0,0) = xpos = imgpart2;
			for(int i = 0; i<divy;i++)
			{
				idx = i*divx;
				gridPoints.block(idx,1,divx,1) = gridY.replicate(divx,1);
				gridY(0,0) += imgpart;
			}
			for(int j = 1;j < divx;j++)
			{
				xpos += imgpart;
				gridX(j,0) = xpos;
			}
			gridPoints.col(0) = gridX.replicate(divy,1);
			imgpart22 = imgpart2 * imgpart2;

			//Check which grid bins must not contain stat. optical flow to get a finer grid
			for(int i = 0; i < divy; i++)
			{
				idx = i*divx;
				for(int j = 0; j < divx; j++)
				{
					ret_matches.clear();
                    keypts1idxall->index->radiusSearch(&gridPoints(idx+j,0),imgpart22,ret_matches,nanoflann::SearchParams(maxDepthSearch));
					int resSize = (int)ret_matches.size();
					if(resSize < 5)
					{
						usedGridElems++;
					}
				}
			}
			if((gridPoints.rows() - usedGridElems) < 2)
			{
				divy = 1;
			}
			else if((float)keyP1.rows()/((float)(divy * divx) * ((float)(gridPoints.rows() - usedGridElems) / (float)gridPoints.rows())) < (float)minPtsPerSqr)
			{
				divy = (int)floor(std::sqrt((float)keyP1.rows() * ((float)gridPoints.rows() / (float)(gridPoints.rows() - usedGridElems))/((float)minPtsPerSqr * xyRatio)));
				if(!divy || (divy < 0)) divy = 1;
				if(divy > (imgSi.height / 10)) divy = imgSi.height / 10;
			}
		}
		else
		{
			if((float)keyP1.rows()/((float)(divy * divy) * xyRatio) < (float)minPtsPerSqr)
			{
				divy = (int)floor(std::sqrt((float)keyP1.rows()/((float)minPtsPerSqr * xyRatio)));
				if(!divy || (divy < 0)) divy = 1;
				if(divy > (imgSi.height / 10)) divy = imgSi.height / 10;
			}
		}
	}

	bool calcNewGrid = false;
//calcNewGrid:
	do{
        vector<std::pair<Point3i,vector<std::pair<size_t,float>>>> validSqrs, lessPtSqrs;
		gridTreeMatches.clear();
		flow_dist.clear();
		flow_ang.clear();
		stat_dist.clear();
		stat_ang.clear();
		if(divy > 1)
		{
			calcNewGrid = false;
			imgpart = (float)imgSi.height/(float)divy;
			*gridElemSize = imgpart;
			divx = (int)floor((float)imgSi.width/imgpart);
			lastwidthpart = (float)imgSi.width-(float)divx*imgpart;
			imgpart2 = imgpart/2;
			if(lastwidthpart > remainGridPix) //if the remaining column of the image is too small forget it
				divx++;
			gridElemType = Eigen::MatrixXi::Ones(divy,divx);
			usedGridElems = 0;
			gridPoints.resize(divx*divy,2);
			gridX.resize(divx,1);
			gridX(0,0) = gridY(0,0) = xpos = imgpart2;
			for(int i = 0; i<divy;i++)
			{
				idx = i*divx;
				gridPoints.block(idx,1,divx,1) = gridY.replicate(divx,1);
				gridY(0,0) += imgpart;
			}
			for(int j = 1;j < divx;j++)
			{
				xpos += imgpart;
				gridX(j,0) = xpos;
			}
			gridPoints.col(0) = gridX.replicate(divy,1);

			imgpart22 = imgpart2 * imgpart2;
			radius1 = 2*imgpart22;

			// do a radius search
			for(int i = 0; i < divy; i++)
			{
				idx = i*divx;
				for(int j = 0; j < divx; j++)
				{
					ret_matches.clear();
                    keypts1idx.index->radiusSearch(&gridPoints(idx+j,0),radius1,ret_matches,nanoflann::SearchParams(maxDepthSearch));
					int resSize = (int)ret_matches.size();
					if(!resSize)
					{
						gridElemType(i,j) = 0;
					}
					else if(resSize < minPtsPerSqr)
					{
                        lessPtSqrs.push_back(make_pair(cv::Point3i(j,i,resSize),ret_matches));
						gridElemType(i,j) = 2;
					}
                    else
                        validSqrs.push_back(make_pair(cv::Point3i(j,i,resSize),ret_matches));
				}
			}

			//Check if information is available according to the number of not used grid elements and use it for recalculation of the grid size
			if(keypts1idxall)
			{
				for(int i = 0; i < divy; i++)
				{
					idx = i*divx;
					for(int j = 0; j < divx; j++)
					{
						ret_matches.clear();
                        keypts1idxall->index->radiusSearch(&gridPoints(idx+j,0),imgpart22,ret_matches,nanoflann::SearchParams(maxDepthSearch));
						int resSize = (int)ret_matches.size();
						if(resSize < 5)
						{
							usedGridElems++;
						}
					}
				}

				//Check if the mesh is too fine (if true, recalculate the mesh)
				if((float)lessPtSqrs.size() >= 0.5 * (float)validSqrs.size())
				{
					float meanPts = 0;
					int divy1;
					for(unsigned int i = 0; i < lessPtSqrs.size(); i++)
					{
						meanPts += (float)lessPtSqrs[i].first.z;
					}
					for(unsigned int i = 0; i < validSqrs.size(); i++)
					{
						meanPts += (float)validSqrs[i].first.z;
					}
					meanPts /= (float)(lessPtSqrs.size() + validSqrs.size());
					if((gridPoints.rows() - usedGridElems) < 2)
					{
						divy1 = (int)floor((float)divy * meanPts / (float)minPtsPerSqr);
					}
					else
					{
						divy1 = (int)floor((float)divy * ((float)gridPoints.rows() / (float)(gridPoints.rows() - usedGridElems)) * meanPts / (float)minPtsPerSqr);
					}
					if(!divy1 || (divy1 < 0)) divy1 = 1;
					if(divy1 > (imgSi.height / 10)) divy1 = imgSi.height / 10;
					if(divy1 < divy)
					{
						divy = divy1;
						calcNewGrid = true;
						continue;
						//goto calcNewGrid;
					}
				}
			}
			else
			{
				//Check if the mesh is too fine (if true, recalculate the mesh)
				if((float)lessPtSqrs.size() >= 0.5 * (float)validSqrs.size())
				{
					float meanPts = 0;
					for(unsigned int i = 0; i < lessPtSqrs.size(); i++)
					{
						meanPts += (float)lessPtSqrs[i].first.z;
					}
					for(unsigned int i = 0; i < validSqrs.size(); i++)
					{
						meanPts += (float)validSqrs[i].first.z;
					}
					meanPts /= (float)(lessPtSqrs.size() + validSqrs.size());
					int divy1 = (int)floor((float)divy * meanPts / (float)minPtsPerSqr);
					if(!divy1 || (divy1 < 0)) divy1 = 1;
					if(divy1 > (imgSi.height / 10)) divy1 = imgSi.height / 10;
					if(divy1 < divy)
					{
						divy = divy1;
						calcNewGrid = true;
						continue;
						//goto calcNewGrid;
					}
				}
			}

			//Fill the grid with matches (fills also grid elements with too less matches with surrounding matches)
			int idx1 = 0, idx2 = 0;
			int sumpts;
			int addArea;
			Eigen::Matrix2i cornerValid;
			for(int i = 0; i < divy; i++)
			{
				for(int j = 0; j < divx; j++)
				{
					if(gridElemType(i,j) == 1)
					{
						gridTreeMatches.push_back(validSqrs[idx1].second);
						idx1++;
					}
					else if(!gridElemType(i,j) || (gridElemType(i,j) == 2))
					{
						bool missSqrNotFilled;
						bool noBigSearchWindow = true;
						int addArea_old = 0;
						Eigen::Matrix2i sideValid = Eigen::Matrix2i::Zero(); //directions in matrix: [+x, -x; +y, -y] ->element is 1 if valid. Only 2 elements can be valid at a time (Eigen accesses with idxs (y,x))
						if(!gridElemType(i,j))
						{
							sumpts = 0;
							missSqrNotFilled = true;
						}
						else
						{
							missSqrNotFilled = false;
							gridTreeMatches.push_back(lessPtSqrs[idx2].second);
							sumpts = lessPtSqrs[idx2].first.z;
						}
						addArea = 0;
						while(sumpts < minPtsPerSqr)
						{
							if(noBigSearchWindow)
								cornerValid = Eigen::Matrix2i::Ones();
							addArea++;

							for(int k = 0; k < 4; k++)
							{
								int yrange, xrange, xstart, ystart;
								switch(k)
								{
								case 0:
									if(noBigSearchWindow)
									{
										yrange = 1;
										xrange = 2 * addArea + 1;
										xstart = -1 * addArea;
										ystart = addArea;
									}
									else
									{
										if(!sideValid(1,0))
											continue;
										yrange = 1;
										xrange = 2 * addArea_old + 1;
										xstart = -1 * addArea_old;
										ystart = addArea;
									}
									sideValid(1,0) = 0;
									break;
								case 1:
									if(noBigSearchWindow)
									{
										yrange = 1;
										xrange = 2 * addArea + 1;
										xstart = -1 * addArea;
										ystart = -1 * addArea;
									}
									else
									{
										if(!sideValid(1,1))
											continue;
										yrange = 1;
										xrange = 2 * addArea_old + 1;
										xstart = -1 * addArea_old;
										ystart = -1 * addArea;
									}
									sideValid(1,1) = 0;
									break;
								case 2:
									if(noBigSearchWindow)
									{
										yrange = 2 * (addArea - 1) + 1;
										xrange = 1;
										xstart = addArea;
										ystart = -1 * addArea + 1;
									}
									else
									{
										if(!sideValid(0,0))
											continue;
										yrange = 2 * (addArea_old - 1) + 1;
										xrange = 1;
										xstart = addArea;
										ystart = -1 * addArea_old + 1;
									}
									sideValid(0,0) = 0;
									break;
								case 3:
									if(noBigSearchWindow)
									{
										yrange = 2 * (addArea - 1) + 1;
										xrange = 1;
										xstart = -1 * addArea;
										ystart = -1 * addArea + 1;
									}
									else
									{
										if(!sideValid(0,1))
											continue;
										yrange = 2 * (addArea_old - 1) + 1;
										xrange = 1;
										xstart = -1 * addArea;
										ystart = -1 * addArea_old + 1;
									}
									sideValid(0,1) = 0;
								}

								for(int i2 = ystart; i2 < (ystart + yrange); i2++)
								{
									if(i + i2 >= divy)
									{
										cornerValid(0,0) = cornerValid(0,1) = 0;
										break;
									}
									if(i + i2 < 0)
									{
										cornerValid(1,0) = cornerValid(1,1) = 0;
										continue;
									}
									for(int j2 = xstart; j2 < (xstart + xrange); j2++)
									{
										if(j + j2 >= divx)
										{
											cornerValid(0,1) = cornerValid(1,1) = 0;
											break;
										}
										if(j + j2 < 0)
										{
											cornerValid(0,0) = cornerValid(1,0) = 0;
											continue;
										}

										int cnt = -1;
										switch(gridElemType(i + i2, j + j2))
										{
										case 0:
											switch(k)
											{
											case 0:
												sideValid(1, 0) = 1;
												break;
											case 1:
												sideValid(1, 1) = 1;
												break;
											case 2:
												sideValid(0, 0) = 1;
												break;
											case 3:
												sideValid(0, 1) = 1;
											}
											break;
										case 1:
											for(int i1 = 0; i1 <= i + i2; i1++)
												for(int j1 = 0; j1 <= j + j2; j1++)
													if(gridElemType(i1,j1) == 1)
														cnt++;
											if(missSqrNotFilled)
											{
												missSqrNotFilled = false;
												gridTreeMatches.push_back(validSqrs[cnt].second);
											}
											else
											{
												//Only add the index if it is no duplicate
												int gTMsi = (int)gridTreeMatches.back().size();
												int vSqsi = validSqrs[cnt].first.z;
												for(int j1 = 0; j1 < vSqsi; j1++)
												{
													bool duplNFnd = true;
													for(int i1 = 0; i1 < gTMsi; i1++)
													{
														if(gridTreeMatches.back()[i1].first == validSqrs[cnt].second[j1].first)
														{
															duplNFnd = false;
															sumpts--;
															break;
														}
													}
													if(duplNFnd)
														gridTreeMatches.back().push_back(validSqrs[cnt].second[j1]);
												}
												//gridTreeMatches.back().insert(gridTreeMatches.back().end(),validSqrs[cnt].second.begin(),validSqrs[cnt].second.end());

											}
											sumpts += validSqrs[cnt].first.z;
											switch(k)
											{
											case 0:
												sideValid(1, 0) = 1;
												break;
											case 1:
												sideValid(1, 1) = 1;
												break;
											case 2:
												sideValid(0, 0) = 1;
												break;
											case 3:
												sideValid(0, 1) = 1;
											}
											break;
										case 2:
											for(int i1 = 0; i1 <= i + i2; i1++)
												for(int j1 = 0; j1 <= j + j2; j1++)
													if(gridElemType(i1,j1) == 2)
														cnt++;
											if(missSqrNotFilled)
											{
												missSqrNotFilled = false;
												gridTreeMatches.push_back(lessPtSqrs[cnt].second);
											}
											else
											{
												//Only add the index if it is no duplicate
												int gTMsi = (int)gridTreeMatches.back().size();
												int vSqsi = lessPtSqrs[cnt].first.z;
												for(int j1 = 0; j1 < vSqsi; j1++)
												{
													bool duplNFnd = true;
													for(int i1 = 0; i1 < gTMsi; i1++)
													{
														if(gridTreeMatches.back()[i1].first == lessPtSqrs[cnt].second[j1].first)
														{
															duplNFnd = false;
															sumpts--;
															break;
														}
													}
													if(duplNFnd)
														gridTreeMatches.back().push_back(lessPtSqrs[cnt].second[j1]);
												}
												//gridTreeMatches.back().insert(gridTreeMatches.back().end(),lessPtSqrs[cnt].second.begin(),lessPtSqrs[cnt].second.end());
											}
											sumpts += lessPtSqrs[cnt].first.z;
											switch(k)
											{
											case 0:
												sideValid(1, 0) = 1;
												break;
											case 1:
												sideValid(1, 1) = 1;
												break;
											case 2:
												sideValid(0, 0) = 1;
												break;
											case 3:
												sideValid(0, 1) = 1;
											}
										}
									}
								}
							}
							if(!cornerValid.sum())
							{
								if(!sideValid.sum())
									break;
								if(noBigSearchWindow)
								{
									addArea_old = addArea;
									noBigSearchWindow = false;
								}
							}
						}
						if(gridElemType(i,j))
							idx2++;
					}
				}
			}

			//Check if some grid elements couldnt be filled
			if(divx * divy != (int)gridTreeMatches.size())
			{
				divy = 1;
				calcNewGrid = true;
				continue;
			}

			//Calculate the flow for the matches
			for(unsigned int i = 0; i < gridTreeMatches.size(); i++)
			{
				vector<double> tmp;
				float dist2, ang, xdist, ydist;
				xdist = keyP2(gridTreeMatches[i][0].first,0) - keyP1(gridTreeMatches[i][0].first,0);
				ydist = keyP2(gridTreeMatches[i][0].first,1) - keyP1(gridTreeMatches[i][0].first,1);
				dist2 = std::sqrt(xdist * xdist + ydist * ydist);
				if(!ydist && !xdist)
					ang = 0.0;
				else
					ang = std::atan2(ydist, xdist);
				if(ang < 0.0)
					ang += (float)(2.0 * PI);
				tmp.push_back((double)dist2);
				flow_dist.push_back(tmp);
				tmp[0] = (double)ang;
				flow_ang.push_back(tmp);
				for(unsigned int j = 1; j < gridTreeMatches[i].size(); j++)
				{
					xdist = keyP2(gridTreeMatches[i][j].first,0) - keyP1(gridTreeMatches[i][j].first,0);
					ydist = keyP2(gridTreeMatches[i][j].first,1) - keyP1(gridTreeMatches[i][j].first,1);
					dist2 = std::sqrt(xdist * xdist + ydist * ydist);
					if(!ydist && !xdist)
						ang = 0.0;
					else
						ang = std::atan2(ydist, xdist);
					if(ang < 0.0)
						ang += (float)(2.0 * PI);
					flow_dist.back().push_back((double)dist2);
					flow_ang.back().push_back((double)ang);
				}
			}

			//Calculate and validate the statistics for the flow
			Mat validGridElem = Mat::ones(divy,divx,CV_8U);
			Mat angDistGridValid = Mat::ones(divy,divx,CV_8U);
			for(int i = 0; i < divy; i++)
			{
				for(int j = 0; j < divx; j++)
				{
					idx = i * divx + j;
					qualityParm tmp;
					getStatisticfromVec(flow_dist[idx], &tmp, true);
					double statdiff = abs(tmp.arithErr - tmp.medErr);
					if((statdiff/(tmp.arithErr + 0.1) > validationTH_) && (statdiff > minAbsDistDiff))
					{
						//validGridElem.at<unsigned char>(i,j) = 0;
						angDistGridValid.at<unsigned char>(i,j) = 0;
					}
					stat_dist.push_back(tmp);
					getAngularStatistic(flow_ang[idx], &tmp, true);
					statdiff = abs(tmp.arithErr - tmp.medErr) / PI;
					if(statdiff > 1.0) statdiff -= 1.0;
					if(statdiff > validationTHang)
					{
						if(angDistGridValid.at<unsigned char>(i,j))
						{
							angDistGridValid.at<unsigned char>(i,j) = 3;
							//validGridElem.at<unsigned char>(i,j) = 0;
						}
					}
					else if(!angDistGridValid.at<unsigned char>(i,j))
						angDistGridValid.at<unsigned char>(i,j) = 2;
					stat_ang.push_back(tmp);
				}
			}

			//Filter the statistic
			vector<double> ang_stat_stat, dist_stat_stat;
			for(int i = 0; i < divy; i++)
			{
				for(int j = 0; j < divx; j++)
				{
					if(!angDistGridValid.at<unsigned char>(i,j))
						continue;
					idx = i * divx + j;
					ang_stat_stat.push_back(stat_ang[idx].medErr);
					dist_stat_stat.push_back(stat_dist[idx].medErr);
				}
			}

			//Filter the flow according to the statistic from the statistic and recalculate the initial statistic
			if(ang_stat_stat.size() < 3)
			{
				divy = 1;
				calcNewGrid = true;
				continue;
				//goto calcNewGrid;
			}
			else
			{
				qualityParm tmp_ang, tmp_dist;
				float ang_th[3], dist_th[2];
				double minStdAng;
				vector<vector<double>> flow_dist_filtered, flow_ang_filtered;

				//Calculate the statistic over the statistic from the grid elements
				getAngularStatistic(ang_stat_stat, &tmp_ang);
				getStatisticfromVec(dist_stat_stat, &tmp_dist);

				minStdAng = 1.07 * std::atan(1.0 / (tmp_dist.arithErr + 0.1)) / stdMult_th;
				if(tmp_ang.arithStd < minStdAng)
					tmp_ang.arithStd = minStdAng;
				if(tmp_dist.arithStd < 0.5)
					tmp_dist.arithStd = 0.5;

				//Filter the flow according to the statistic from the statistic
				ang_th[0] = (float)(tmp_ang.arithErr - stdMult_th * tmp_ang.arithStd);
				ang_th[1] = (float)(tmp_ang.arithErr + stdMult_th * tmp_ang.arithStd);
				dist_th[0] = (float)(tmp_dist.arithErr - stdMult_th * tmp_dist.arithStd);
				dist_th[1] = (float)(tmp_dist.arithErr + stdMult_th * tmp_dist.arithStd);
				if(ang_th[0] >= PI)
				{
					ang_th[0] -= 2.0f * PI;
				}
				if(ang_th[1] > 2.0f * PI)
				{
					if(ang_th[1] - ang_th[0] >= 2.0f * PI)
					{
						ang_th[1] = 2.0f * PI;
						ang_th[0] = 0;
					}
					else
						ang_th[1] = 2.0f * PI;
				}
				if(ang_th[0] < 0)
				{
					ang_th[2] = 2.0f * PI + ang_th[0];
					ang_th[0] = 0;
				}
				else
					ang_th[2] = 2.0f * PI;

				for(unsigned int i = 0; i < flow_ang.size(); i++)
				{
					vector<double> flow_ang_tmp, flow_dist_tmp;
					for(unsigned int j = 0; j < flow_ang[i].size(); j++)
					{
						if((flow_dist[i][j] <= dist_th[1]) && (flow_dist[i][j] >= dist_th[0]) &&
							((flow_ang[i][j] <= ang_th[1]) || (flow_ang[i][j] >= ang_th[2])) &&
							(flow_ang[i][j] >= ang_th[0]))
						{
							flow_ang_tmp.push_back(flow_ang[i][j]);
							flow_dist_tmp.push_back(flow_dist[i][j]);
						}
					}
					flow_dist_filtered.push_back(flow_dist_tmp);
					flow_ang_filtered.push_back(flow_ang_tmp);
				}

				//Check if some grid elements (flow) are missing
				bool escapefor = false;
				for(int i = 0; i < divy; i++)
				{
					for(int j = 0; j < divx; j++)
					{
						idx = i * divx + j;
						if(flow_dist_filtered[idx].empty())
						{
							int i_tmp, j_tmp = j;
							int idx_tmp;
							if(i < (divy - 1))
							{
								i_tmp = i + 1;
								idx_tmp = i_tmp * divx + j_tmp;
								while((i_tmp < (divy - 1)) && flow_dist_filtered[idx_tmp].empty())
								{
									i_tmp++;
									idx_tmp = i_tmp * divx + j_tmp;
								}
								if(!flow_dist_filtered[idx_tmp].empty())
								{
									flow_dist_filtered[idx] = flow_dist_filtered[idx_tmp];
									flow_ang_filtered[idx] = flow_ang_filtered[idx_tmp];
								}
							}
							if(i > 0)
							{
								i_tmp = i - 1;
								idx_tmp = i_tmp * divx + j_tmp;
								while(flow_dist_filtered[idx_tmp].empty() && (i_tmp > 0))
								{
									i_tmp--;
									idx_tmp = i_tmp * divx + j_tmp;
								}
								if(!flow_dist_filtered[idx_tmp].empty())
								{
									if(flow_dist_filtered[idx].empty())
									{
										flow_dist_filtered[idx] = flow_dist_filtered[idx_tmp];
										flow_ang_filtered[idx] = flow_ang_filtered[idx_tmp];
									}
									else
									{
										flow_dist_filtered[idx].insert(flow_dist_filtered[idx].end(), flow_dist_filtered[idx_tmp].begin(), flow_dist_filtered[idx_tmp].end());
										flow_ang_filtered[idx].insert(flow_ang_filtered[idx].end(), flow_ang_filtered[idx_tmp].begin(), flow_ang_filtered[idx_tmp].end());
									}
								}
							}
							i_tmp = i;
							if(j < (divx - 1))
							{
								j_tmp = j + 1;
								idx_tmp = i_tmp * divx + j_tmp;
								while(flow_dist_filtered[idx_tmp].empty() && (j_tmp < (divx - 1)))
								{
									j_tmp++;
									idx_tmp = i_tmp * divx + j_tmp;
								}
								if(!flow_dist_filtered[idx_tmp].empty())
								{
									if(flow_dist_filtered[idx].empty())
									{
										flow_dist_filtered[idx] = flow_dist_filtered[idx_tmp];
										flow_ang_filtered[idx] = flow_ang_filtered[idx_tmp];
									}
									else
									{
										flow_dist_filtered[idx].insert(flow_dist_filtered[idx].end(), flow_dist_filtered[idx_tmp].begin(), flow_dist_filtered[idx_tmp].end());
										flow_ang_filtered[idx].insert(flow_ang_filtered[idx].end(), flow_ang_filtered[idx_tmp].begin(), flow_ang_filtered[idx_tmp].end());
									}
								}
							}
							if(j > 0)
							{
								j_tmp = j - 1;
								idx_tmp = i_tmp * divx + j_tmp;
								while(flow_dist_filtered[idx_tmp].empty() && (j_tmp > 0))
								{
									j_tmp--;
									idx_tmp = i_tmp * divx + j_tmp;
								}
								if(!flow_dist_filtered[idx_tmp].empty())
								{
									if(flow_dist_filtered[idx].empty())
									{
										flow_dist_filtered[idx] = flow_dist_filtered[idx_tmp];
										flow_ang_filtered[idx] = flow_ang_filtered[idx_tmp];
									}
									else
									{
										flow_dist_filtered[idx].insert(flow_dist_filtered[idx].end(), flow_dist_filtered[idx_tmp].begin(), flow_dist_filtered[idx_tmp].end());
										flow_ang_filtered[idx].insert(flow_ang_filtered[idx].end(), flow_ang_filtered[idx_tmp].begin(), flow_ang_filtered[idx_tmp].end());
									}
								}
							}
							if(flow_dist_filtered[idx].empty())
							{
								if(roundNewGridCalcs > 0)
								{
									return -1; //Calculation of statistic failed
								}
								escapefor = true;
								roundNewGridCalcs++;
								break;
							}
						}
					}
					if(escapefor)
						break;
				}
				if(escapefor)
				{
					divy = 1;
					calcNewGrid = true;
					continue;
				}

				//Recalculate the initial statistic for the flow
				stat_dist.clear();
				stat_ang.clear();
				int gridSi = divy * divx;
				int validGridElemSi = gridSi;
				for(int i = 0; i < divy; i++)
				{
					for(int j = 0; j < divx; j++)
					{
						idx = i * divx + j;
						qualityParm tmp;
						getStatisticfromVec(flow_dist_filtered[idx], &tmp, true);
						double statdiff = abs(tmp.arithErr - tmp.medErr);
						if((statdiff/(tmp.arithErr + 0.1) > validationTH_)  && (statdiff > minAbsDistDiff))
						{
							validGridElem.at<unsigned char>(i,j) = 0;
							validGridElemSi--;
						}

						double minStdAng[2];
						minStdAng[0] = 1.07 * std::atan(1.0 / (tmp.arithErr + 0.1)) / stdMult_th;
						minStdAng[1] = 1.07 * std::atan(1.0 / (tmp.medErr + 0.1)) / stdMult_th;
						if(tmp.arithStd < 0.5)
							tmp.arithStd = 0.5;
						if(tmp.medStd < 0.5)
							tmp.medStd = 0.5;

						stat_dist.push_back(tmp);
						getAngularStatistic(flow_ang_filtered[idx], &tmp, true);
						statdiff = abs(tmp.arithErr - tmp.medErr) / PI;
						if(statdiff > 1.0) statdiff -= 1.0;
						if((statdiff > validationTHang) &&
							validGridElem.at<unsigned char>(i,j))
						{
							validGridElem.at<unsigned char>(i,j) = 0;
							validGridElemSi--;
						}

						if(tmp.arithStd < minStdAng[0])
							tmp.arithStd = minStdAng[0];
						if(tmp.medStd < minStdAng[1])
							tmp.medStd = minStdAng[1];

						stat_ang.push_back(tmp);
					}
				}
				if((gridSi - validGridElemSi)/gridSi > 0.5)
				{
					divy = 1;
					calcNewGrid = true;
					continue;
					//goto calcNewGrid;
				}
			}

			//Use only valid statistics (if one is not valid, take a valid one from a neighbor or the global one)
			for(int i = 0; i < divy; i++)
			{
				for(int j = 0; j < divx; j++)
				{
					if(!validGridElem.at<unsigned char>(i,j))
					{
						vector<qualityParm> validNeighbors_ang, validNeighbors_dist;
						vector<double> neighbor_diff;
						int idx_min = 0;
						double vect_diff[2], divStdInv;
						for(int i1 = 0; i1 < 3; i1++)
						{
							const int ynew = i + i1 - 1;
							if(ynew >= divy)
								break;
							if(ynew < 0)
								continue;
							for(int j1 = 0; j1 < 3; j1++)
							{

								if((i1 == 1) && (j1 == 1))
									continue;
								const int xnew = j + j1 - 1;
								if(xnew >= divx)
									break;
								if(xnew < 0)
									continue;

								if(validGridElem.at<unsigned char>(ynew,xnew))
								{
									validNeighbors_ang.push_back(stat_ang[ynew * divx + xnew]);
									validNeighbors_dist.push_back(stat_dist[ynew * divx + xnew]);
								}
							}
						}
						if(!globStatCalculated)
						{
							vector<double> flow_dist_glob, flow_ang_glob;
							for(int i1 = 0; i1 < (int)keyP2.rows();i1++)
							{
								float dist2, ang, xdist, ydist;
								xdist = keyP2(i1,0) - keyP1(i1,0);
								ydist = keyP2(i1,1) - keyP1(i1,1);
								dist2 = std::sqrt(xdist * xdist + ydist * ydist);
								if(!ydist && !xdist)
									ang = 0.0;
								else
									ang = std::atan2(ydist, xdist);
								if(ang < 0.0)
									ang += (float)(2.0 * PI);
								flow_dist_glob.push_back((double)dist2);
								flow_ang_glob.push_back((double)ang);
							}
							getStatisticfromVec(flow_dist_glob, &stat_dist_glob, true);
							getAngularStatistic(flow_ang_glob, &stat_ang_glob, true);
							globStatCalculated = true;
						}
						validNeighbors_ang.push_back(stat_ang_glob);
						validNeighbors_dist.push_back(stat_dist_glob);
						idx = i * divx + j;
						if(validNeighbors_dist.size() > 1)
						{
							for(unsigned int k = 0; k < validNeighbors_ang.size(); k++)
							{
								double act_diff[2];
								act_diff[0] = std::abs(validNeighbors_ang[k].medErr - stat_ang[idx].medErr + 0.05236) / PI;
								if(act_diff[0] > 1.0) act_diff[0] -= 1.0;
								act_diff[0] *= validThAngDivFact;
								act_diff[1] = std::abs(validNeighbors_dist[k].medErr - stat_dist[idx].medErr + 0.01) / (stat_dist[idx].medErr + 0.1);
								neighbor_diff.push_back(act_diff[0] * act_diff[1]);
							}
							idx_min = std::distance(neighbor_diff.begin(), std::min_element(neighbor_diff.begin(), neighbor_diff.end()));
						}
						vect_diff[0] = std::cos(validNeighbors_ang[idx_min].medErr) * validNeighbors_dist[idx_min].medErr -
										std::cos(stat_ang[idx].medErr) * stat_dist[idx].medErr;
						vect_diff[1] = std::sin(validNeighbors_ang[idx_min].medErr) * validNeighbors_dist[idx_min].medErr -
										std::sin(stat_ang[idx].medErr) * stat_dist[idx].medErr;
						vect_diff[0] *= vect_diff[0];
						vect_diff[1] *= vect_diff[1];
						vect_diff[0] += vect_diff[1];
						vect_diff[0] = std::sqrt(vect_diff[0]) / stdMult;
						stat_dist[idx] = validNeighbors_dist[idx_min];
						stat_dist[idx].arithStd += vect_diff[0];

						divStdInv = stat_ang[idx].arithStd / (validNeighbors_ang[idx_min].arithStd + 0.001);
						if(divStdInv > 1.5)
							divStdInv = 1.5;
						else if(divStdInv < 1.0)
							divStdInv = 1.0;
						stat_ang[idx] = validNeighbors_ang[idx_min];
						stat_ang[idx].arithStd *= divStdInv;
					}
				}
			}
		}
		else
		{
			calcNewGrid = false;
			divx = 1;
			*gridElemSize = (float)imgSi.width;
			vector<double> flow_dist_glob, flow_ang_glob;
			for(int i1 = 0; i1 < (int)keyP2.rows();i1++)
			{
				float dist2, ang, xdist, ydist;
				xdist = keyP2(i1,0) - keyP1(i1,0);
				ydist = keyP2(i1,1) - keyP1(i1,1);
				dist2 = std::sqrt(xdist * xdist + ydist * ydist);
				if(!ydist && !xdist)
					ang = 0.0;
				else
					ang = std::atan2(ydist, xdist);
				if(ang < 0.0)
					ang += (float)(2.0 * PI);
				flow_dist_glob.push_back((double)dist2);
				flow_ang_glob.push_back((double)ang);
			}
			getStatisticfromVec(flow_dist_glob, &stat_dist_glob, true);

			if(stat_dist_glob.arithStd < 0.5)
				stat_dist_glob.arithStd = 0.5;
			if(stat_dist_glob.medStd < 0.5)
				stat_dist_glob.medStd = 0.5;

			getAngularStatistic(flow_ang_glob, &stat_ang_glob, true);

			double minStdAng[2];
			minStdAng[0] = 1.07 * std::atan(1.0 / (stat_dist_glob.arithErr + 0.1)) / stdMult_th;
			minStdAng[1] = 1.07 * std::atan(1.0 / (stat_dist_glob.medErr + 0.1)) / stdMult_th;
			if(stat_ang_glob.arithStd < minStdAng[0])
				stat_ang_glob.arithStd = minStdAng[0];
			if(stat_ang_glob.medStd < minStdAng[1])
				stat_ang_glob.medStd = minStdAng[1];

			stat_dist.push_back(stat_dist_glob);
			stat_ang.push_back(stat_ang_glob);
			flow_dist.push_back(flow_dist_glob);
			flow_ang.push_back(flow_ang_glob);
		}
	}while(calcNewGrid == true);

	//Calculate the thresholds and searchradius and generate the inlier mask
	float flow_x, flow_y, r_std, ang_th[3], dist_th[2];
	Mat mask_;
	if(mask.needed())
	{
		mask.create((int)keyP2.rows(),1, CV_8U);
		mask_ = mask.getMat();
		mask_ = Mat::ones((int)keyP2.rows(),1, CV_8U);
	}
	if(divy == 1)
	{
		flow_x = (float)(std::cos(stat_ang[0].medErr) * stat_dist[0].medErr);
		flow_y = (float)(std::sin(stat_ang[0].medErr) * stat_dist[0].medErr);
		r_std = (float)(stdMult * stat_dist[0].medStd);
		gridSearchParams.push_back(vector<Point3f>(1,Point3f(flow_x, flow_y, r_std)));
		if(mask.needed())
		{
			ang_th[0] = stat_ang[0].medErr - stdMult * stat_ang[0].medStd;
			ang_th[1] = stat_ang[0].medErr + stdMult * stat_ang[0].medStd;
			dist_th[0] = stat_dist[0].medErr - r_std;
			dist_th[1] = stat_dist[0].medErr + r_std;
			if(ang_th[0] >= PI)
			{
				ang_th[0] -= 2.0f * PI;
			}
			if(ang_th[1] > 2.0f * PI)
			{
				if(ang_th[1] - ang_th[0] >= 2.0f * PI)
				{
					ang_th[1] = 2.0f * PI;
					ang_th[0] = 0;
				}
				else
					ang_th[1] = 2.0f * PI;
			}
			if(ang_th[0] < 0)
			{
				ang_th[2] = 2 * PI + ang_th[0];
				ang_th[0] = 0;
			}
			else
				ang_th[2] = 2 * PI;
			for(int i1 = 0; i1 < (int)keyP2.rows(); i1++)
			{
				if((flow_dist[0][i1] > dist_th[1]) || (flow_dist[0][i1] < dist_th[0]) ||
					(flow_ang[0][i1] < ang_th[0]) ||
					((flow_ang[0][i1] <= ang_th[2]) && (flow_ang[0][i1] > ang_th[1])) ||
					(filterSmallFlow && (flow_dist[0][i1] < th_smallFlow)))
					mask_.at<unsigned char>(i1) = 0;
			}
		}
	}
	else
	{
		for(int i = 0; i < divy; i++)
		{
			idx = i * divx;
			flow_x = std::cos(stat_ang[idx].arithErr) * stat_dist[idx].arithErr;
			flow_y = std::sin(stat_ang[idx].arithErr) * stat_dist[idx].arithErr;
			r_std = stdMult * stat_dist[idx].arithStd;
			gridSearchParams.push_back(vector<Point3f>(1,Point3f(flow_x, flow_y, r_std)));
			if(mask.needed())
			{
				ang_th[0] = stat_ang[idx].arithErr - stdMult * stat_ang[idx].arithStd;
				ang_th[1] = stat_ang[idx].arithErr + stdMult * stat_ang[idx].arithStd;
				dist_th[0] = stat_dist[idx].arithErr - r_std;
				dist_th[1] = stat_dist[idx].arithErr + r_std;
				if(ang_th[0] >= PI)
				{
					ang_th[0] -= 2.0f * PI;
				}
				if(ang_th[1] > 2.0f * PI)
				{
					if(ang_th[1] - ang_th[0] >= 2.0f * PI)
					{
						ang_th[1] = 2.0f * PI;
						ang_th[0] = 0;
					}
					else
						ang_th[1] = 2.0f * PI;
				}
				if(ang_th[0] < 0)
				{
					ang_th[2] = 2 * PI + ang_th[0];
					ang_th[0] = 0;
				}
				else
					ang_th[2] = 2 * PI;
				for(unsigned int i1 = 0; i1 < gridTreeMatches[idx].size(); i1++)
				{
					if((flow_dist[idx][i1] > dist_th[1]) || (flow_dist[idx][i1] < dist_th[0]) ||
						(flow_ang[idx][i1] < ang_th[0]) ||
						((flow_ang[idx][i1] <= ang_th[2]) && (flow_ang[idx][i1] > ang_th[1])) ||
						(filterSmallFlow && (flow_dist[idx][i1] < th_smallFlow)))
						mask_.at<unsigned char>(gridTreeMatches[idx][i1].first) = 0;
				}
			}
			for(int j = 1; j < divx; j++)
			{
				const int idx1 = idx + j;
				flow_x = std::cos(stat_ang[idx1].arithErr) * stat_dist[idx1].arithErr;
				flow_y = std::sin(stat_ang[idx1].arithErr) * stat_dist[idx1].arithErr;
				r_std = stdMult * stat_dist[idx1].arithStd;
				gridSearchParams.back().push_back(Point3f(flow_x, flow_y, r_std));
				if(mask.needed())
				{
					ang_th[0] = stat_ang[idx1].arithErr - stdMult * stat_ang[idx1].arithStd;
					ang_th[1] = stat_ang[idx1].arithErr + stdMult * stat_ang[idx1].arithStd;
					dist_th[0] = stat_dist[idx1].arithErr - r_std;
					dist_th[1] = stat_dist[idx1].arithErr + r_std;
					if(ang_th[0] >= PI)
					{
						ang_th[0] -= 2.0f * PI;
					}
					if(ang_th[1] > 2.0f * PI)
					{
						if(ang_th[1] - ang_th[0] >= 2.0f * PI)
						{
							ang_th[1] = 2.0f * PI;
							ang_th[0] = 0;
						}
						else
							ang_th[1] = 2.0f * PI;
					}
					if(ang_th[0] < 0)
					{
						ang_th[2] = 2 * PI + ang_th[0];
						ang_th[0] = 0;
					}
					else
						ang_th[2] = 2 * PI;
					for(unsigned int i1 = 0; i1 < gridTreeMatches[idx1].size(); i1++)
					{
						if((flow_dist[idx1][i1] > dist_th[1]) || (flow_dist[idx1][i1] < dist_th[0]) ||
							(flow_ang[idx1][i1] < ang_th[0]) ||
							((flow_ang[idx1][i1] <= ang_th[2]) && (flow_ang[idx1][i1] > ang_th[1])) ||
							(filterSmallFlow && (flow_dist[idx1][i1] < th_smallFlow)))
							mask_.at<unsigned char>(gridTreeMatches[idx1][i1].first) = 0;
					}
				}
			}
		}
	}
	if(gridSearchParams.empty())
		cout << "something went wrong while calc stat opt flow" << endl;

	return 0;
}

/* Calculates statistical parameters for angular values in the vector. The following parameters
 * are calculated: median, arithmetic mean value, standard deviation and standard deviation using the median.
 * This function is a workaround for the cases where angles near zero and 2*pi are included in the data set,
 * which are quite near/equal to each other but are numerically far from each other
 *
 * vector<double> vals		Input  -> Input vector from which the statistical parameters should be calculated
 * qualityParm* stats		Output -> Structure holding the statistical parameters
 * bool rejQuartiles		Input  -> If true, the lower and upper quartiles are rejected before calculating
 *									  the parameters
 *
 * Return value:		 none
 */
void getAngularStatistic(const std::vector<double> vals, qualityParm *stats, bool rejQuartiles)
{
	std::vector<double> vals1, vals2, vals_tmp, vals1_2, vals2_2;
	qualityParm hyp1Stats, hyp2Stats;
	const double ang_th = 3 * PI;
	const double pi2 = 2 * PI;

	vals_tmp = vals;

	//Split angles to eliminate the influence of the angles 0 and 2pi which are equal but numerically far from each other
	for(int i = 0; i < (int)vals_tmp.size(); i++)
	{
		vals_tmp[i] += pi2;//Add 2pi to prevent negative values at later angular operations
		if(vals_tmp[i] < ang_th)
		{
			vals1.push_back(vals_tmp[i]);
		}
		else
		{
			vals2.push_back(vals_tmp[i]);
		}
	}

	if((vals1.size() < 2) || (vals2.size() < 2))
	{
		getStatisticfromVec(vals, stats, rejQuartiles);
		return;
	}
	if((float)vals2.size() / (float)vals1.size() < 0.1)
	{
		for(int i = 0; i < (int)vals1.size(); i++)
		{
			vals1[i] -= pi2;
		}
		getStatisticfromVec(vals1, stats, rejQuartiles);
		return;
	}
	if((float)vals1.size() / (float)vals2.size() < 0.1)
	{
		for(int i = 0; i < (int)vals2.size(); i++)
		{
			vals2[i] -= pi2;
		}
		getStatisticfromVec(vals2, stats, rejQuartiles);
		return;
	}

	//Generate 2 hypotheses
	getStatisticfromVec(vals1, &hyp1Stats);
	getStatisticfromVec(vals2, &hyp2Stats);

	//Check for both hypotheses which values from the other dataset are farer away than pi
	vals1_2 = vals1;
	vals2_2 = vals2;
	for(int i = 0; i < (int)vals1.size(); i++)
	{
		const double ang_diff = abs(hyp2Stats.medErr - vals1[i]);
		if(ang_diff > PI)
			vals2_2.push_back(vals1[i] + pi2);
		else
			vals2_2.push_back(vals1[i]);
	}
	for(int i = 0; i < (int)vals2.size(); i++)
	{
		const double ang_diff = abs(vals2[i] - hyp1Stats.medErr);
		if(ang_diff > PI)
			vals1_2.push_back(vals2[i] - pi2);
		else
			vals1_2.push_back(vals2[i]);
	}

	//Reestimate the statistic for both hypotheses
	getStatisticfromVec(vals1_2, &hyp1Stats, rejQuartiles);
	getStatisticfromVec(vals2_2, &hyp2Stats, rejQuartiles);

	if((hyp1Stats.arithStd + hyp1Stats.medStd) <= (hyp2Stats.arithStd + hyp2Stats.medStd))
		*stats = hyp1Stats;
	else
		*stats = hyp2Stats;

	if((stats->arithErr >= pi2) && (stats->medErr >= pi2))
	{
		stats->arithErr -= pi2;
		stats->medErr -= pi2;
		if((stats->arithErr >= pi2) && (stats->medErr >= pi2))
		{
			stats->arithErr -= pi2;
			stats->medErr -= pi2;
		}
	}
}


/* Calculates statistical parameters for the given values in the vector. The following parameters
 * are calculated: median, arithmetic mean value, standard deviation and standard deviation using the median.
 *
 * vector<double> vals		Input  -> Input vector from which the statistical parameters should be calculated
 * qualityParm* stats		Output -> Structure holding the statistical parameters
 * bool rejQuartiles		Input  -> If true, the lower and upper quartiles are rejected before calculating
 *									  the parameters
 *
 * Return value:		 none
 */
void getStatisticfromVec(const std::vector<double> vals, qualityParm *stats, bool rejQuartiles)
{
	if(vals.empty())
	{
		stats->arithErr = 0;
		stats->arithStd = 0;
		stats->medErr = 0;
		stats->medStd = 0;
		return;
	}
	int n = vals.size();
	int qrt_si = (int)floor(0.25 * (double)n);
	std::vector<double> vals_tmp(vals);

	std::sort(vals_tmp.begin(),vals_tmp.end(),[](double const & first, double const & second){
		return first < second;});
	//INLINEQSORTSIMPLE(double,vals_tmp.data(),n,Pfe32u_lt);

	if(n % 2)
		stats->medErr = vals_tmp[(n-1)/2];
	else
		stats->medErr = (vals_tmp[n/2] + vals_tmp[n/2-1]) / 2.0;

	stats->arithErr = 0.0;
	double err2sum = 0.0;
	double medstdsum = 0.0;
	double hlp;
	for(int i = rejQuartiles ? qrt_si:0; i < (rejQuartiles ? (n-qrt_si):n); i++)
	{
		stats->arithErr += vals_tmp[i];
		err2sum += vals_tmp[i] * vals_tmp[i];
		hlp = (vals_tmp[i] - stats->medErr);
		medstdsum += hlp * hlp;
	}
	if(rejQuartiles)
		n -= 2 * qrt_si;
	stats->arithErr /= (double)n;

	hlp = err2sum - (double)n * (stats->arithErr) * (stats->arithErr);
	if(std::abs(hlp) < 1e-6)
		stats->arithStd = 0.0;
	else
		stats->arithStd = std::sqrt(hlp/((double)n - 1.0));

	if(std::abs(medstdsum) < 1e-6)
		stats->medStd = 0.0;
	else
		stats->medStd = std::sqrt(medstdsum/((double)n - 1.0));
}


/* This function calculates the L1-norm of the hamming distance between two column vectors
 * using a LUT.
 *
 * Mat vec1						Input  -> First vector which must be of type uchar or CV_8U
 * Mat vec1						Input  -> Second vector which must be the same size as vec1
 *										  and of type uchar or CV_8U
 *
 * Return value:				L1-norm of the hamming distance
 */
inline unsigned int getHammingL1(cv::Mat vec1, cv::Mat vec2)
{
	static const unsigned char BitsSetTable256[256] =
	{
	#   define B2(n) n,     n+1,     n+1,     n+2
	#   define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
	#   define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
		B6(0), B6(1), B6(1), B6(2)
	};
	unsigned int l1norm = 0;

	for(int i = 0; i<vec1.size().width;i++)
	{
        l1norm += (unsigned int)BitsSetTable256[(vec1.at<uchar>(i)^vec2.at<uchar>(i))];
	}

	//Slower version below
	/*Mat v12xor;
	bitwise_xor(vec1, vec2, v12xor);
	for(int i = 0;i<v12xor.size().width;i++)
	{
		while(v12xor.at<uchar>(i))
		{
			v12xor.at<uchar>(i) &= v12xor.at<uchar>(i) - 1;
			l1norm++;
		}
	}*/

	return l1norm;
}

/* This function calculates the L1-norm of the hamming distance between two column vectors using
 * the popcnt CPU-instruction
 *
 * Mat vec1						Input  -> First vector which must be of type uchar or CV_8U
 * Mat vec1						Input  -> Second vector which must be the same size as vec1
 *										  and of type uchar or CV_8U
 * unsigned char byte8width		Input  -> Number of bytes devided by 8 (64 bit) for one descriptor
 *
 * Return value:				L1-norm of the hamming distance
 */
inline unsigned int getHammingL1PopCnt(cv::Mat vec1, cv::Mat vec2, unsigned char byte8width)
{
#ifdef __linux__
  __uint64_t hamsum1 = 0;
  const __uint64_t *inputarr1 = reinterpret_cast<const __uint64_t*>(vec1.data);
  const __uint64_t *inputarr2 = reinterpret_cast<const __uint64_t*>(vec2.data);
#else
  unsigned __int64 hamsum1 = 0;
  const unsigned __int64 *inputarr1 = reinterpret_cast<const unsigned __int64*>(vec1.data);
  const unsigned __int64 *inputarr2 = reinterpret_cast<const unsigned __int64*>(vec2.data);
#endif
	for(unsigned char i = 0; i < byte8width; i++)
	{
        hamsum1 += _mm_popcnt_u64(*(inputarr1 + i) ^ *(inputarr2 + i));
	}
	return (unsigned int)hamsum1;
}

/* This function calculates the L2-norm between two column vectors of desriptors
 *
 * Mat vec1						Input  -> First vector which must be of type uchar or CV_32F
 * Mat vec1						Input  -> Second vector which must be the same size as vec1
 *										  and of type uchar or CV_8U
 *
 * Return value:				L1-norm of the hamming distance
 */
inline float getL2Distance(cv::Mat vec1, cv::Mat vec2)
{
	static int descrCols = vec1.cols;
	float vecsum = 0;

	for(int i = 0; i < descrCols; i++)
	{
		float hlp = vec1.at<float>(i) - vec2.at<float>(i);
		vecsum += hlp * hlp;
	}
	return vecsum;
}



/* This function finds the best matches between descriptors by estimating the position of the corresponding keypoint with a
 * precalculated flow in the first place and selecting the nearest neighbors within a specified radius (given by the
 * standard deviation of the flow) for descriptor matching in the second place. For matching the descriptors, the L1-norm
 * (Manhattan distance) of the Hamming distance is calculated.
 *
 * vector<vector<Point3f>> gridSearchParams		Input  -> The vector-structure corresponds to the size of a grid in the image
 *														  (the outer vector corresponds to the rows or
 *														  y-coordinates and the inner vector corresponds to the
 *														  columns or x-coordinates). Each vector-element holds
 *														  3 values. The first corresponds to the average flow
 *														  in x- and the second to the average flow in y-direction
 *														  within the grid element. The third value is a multiple of
 *														  the standard deviation of the flow.
 * float gridElemSize							Input  -> Size of one grid element in the image used by gridSearchParams
 * vector<KeyPoint> keypoints					Input  -> The keypoints of the view for which correspondences in the other
 *														  view should be searched for
 * Mat descriptors1								Input  -> The descriptors of the view corresponding to "keypoints"
 * Mat descriptors2								Input  -> The descriptors of the second view
 * KDTree_D2float &keypointtree					Input  -> The KD-tree (indexes) of the keypoints of the view for which
 *														  correspondences in the other view should be searched for
 * vector<int> keypIndexes						Input  -> A sorted (from low to high) index of keypoints corresponding to
 *														  "keypoints" that were already matched during the calculation of "H"
 * Size imgSi									Input  -> The size of the image
 * vector<DMatch> &matches						Output -> Array of matches
 * vector<mCostDist> mprops						Output -> Holds for every match in "matches" the squared distance to the estimated position (SOF)
 *														  as well as the position (index) of the left keypoint of the match in "gridSearchParams"
 *
 * Return value:								 0:		  Everything ok
 *												-1:		  Too less remaining matches
 *												-2:		  Too less remaining matches (maybe because of too many duplicates)
 *												-3:		  Size in dimension 1 (x and bin cols, respectively) must be the same for all bin rows (y coordinates) in gridSearchParams
 */
int guidedMatching(std::vector<std::vector<cv::Point3f>> gridSearchParams,
				   float gridElemSize,
				   std::vector<cv::KeyPoint> keypoints,
				   cv::Mat descriptors1,
				   cv::Mat descriptors2,
				   KDTree_D2float &keypointtree,
				   std::vector<int> keypIndexes,
				   cv::Size imgSi,
				   std::vector<cv::DMatch> &matches,
                   std::vector<mCostDist> mprops)
{
	const int maxDepthSearch = 32;
	const int hammL1tresh = 160;
	const int searchRadius = 10;

	const int NormIdxSize = 25;
	unsigned int l1norms[NormIdxSize];
	float l2norms[NormIdxSize];
	int var_searchRadius;
	const unsigned int gridSPsize[2] = {gridSearchParams[0].size(), gridSearchParams.size()};//Format [x,y] -> gridSearchParams[y][x]
	const unsigned int gridSPsizeIdx[2] = {gridSPsize[0] - 1, gridSPsize[1] - 1};
#if FILTER_WITH_CD_RATIOS
	mCostDist mprop_tmp = {0,0,0,0};
#endif
	unsigned int grid_x, grid_y;

	if(gridSearchParams[0].size() != gridSearchParams.back().size())
		return -3; //Size in dimension 1 (x and bin cols, respectively) must be the same for all bin rows (y coordinates) in gridSearchParams

	vector<DMatch> matches_knn1;
#if FILTER_WITH_CD_RATIOS
	vector<mCostDist> mprops_knn1;
#endif

	int idx = 0;
	Size imgSi2 = cv::Size(imgSi.width/2,imgSi.height/2);
	float imgDiag = powf((float)imgSi2.width,2)+powf((float)imgSi.height,2);
	//Eigen::Matrix3f He;
	//Eigen::Vector3f x1e;
	Eigen::Vector2f x2e;
	//H.convertTo(H,CV_32F);
	//cv2eigen(H,He);

	//Check type of descriptor
	static bool BinaryOrVector = true;
	static bool useBinPopCnt = false;
	static bool fixProperties = true;
	static unsigned char byte8width = 8;
	static unsigned char descrCols = 64;
	if(fixProperties)
	{
		if(descriptors1.type() == CV_32F)
		{
			BinaryOrVector = false;
		}

		int cpuInfo[4];
//		__cpuid(cpuInfo,0x00000001);
		std::bitset<32> f_1_ECX_ = cpuInfo[2];
		static bool popcnt_available = f_1_ECX_[23];
		if(popcnt_available)
			useBinPopCnt = true;

		if(descriptors1.cols != 64)
		{
			byte8width = (unsigned char)(descriptors1.cols / 8);
			descrCols = descriptors1.cols;
		}

		fixProperties = false;
	}

	if(gridSPsize[1] == 1)
	{
		gridSearchParams[0][0].z = (float)searchRadius > gridSearchParams[0][0].z ? (float)searchRadius:gridSearchParams[0][0].z;
		gridSearchParams[0][0].z *= gridSearchParams[0][0].z; //The radius is squared because the L2-norm is used within the KD-tree
		var_searchRadius = (int)ceil(gridSearchParams[0][0].z);
	}
	else
	{
		for(unsigned int i = 0; i < gridSPsize[1]; i++)
		{
			for(unsigned int j = 0; j < gridSPsize[0]; i++)
			{
				gridSearchParams[i][j].z = (float)searchRadius > gridSearchParams[i][j].z ? (float)searchRadius:gridSearchParams[i][j].z;
				gridSearchParams[i][j].z *= gridSearchParams[i][j].z; //The radius is squared because the L2-norm is used within the KD-tree
				gridSearchParams[i][j].z = ceil(gridSearchParams[i][j].z);
			}
		}
	}

	if(gridSPsize[1] == 1)
	{
		grid_x = grid_y = 0;
	}

	//float pDistM, xdistnorm,ydistnorm;
	for(int i = 0; i<(int)keypoints.size();i++)
	{
		std::vector<std::pair<size_t,float> >   ret_matches;
		unsigned int* idx1;
		float* idx2;
		//Point2f distp;
		if(!keypIndexes.empty())
		{
			if(keypIndexes[idx] == i) //continue for already found correspondences
			{
				if(idx < ((int)keypIndexes.size()-1)) idx++;
					continue;
			}
		}
		//Calculate the approximate position of the corresponding keypoint in the second image
		//x1e << keypoints[i].pt.x, keypoints[i].pt.y, 1.0;
		//x2e = He*x1e;
		if(gridSPsize[1] == 1)
		{
			x2e(0) = keypoints[i].pt.x + gridSearchParams[0][0].x;
			x2e(1) = keypoints[i].pt.y + gridSearchParams[0][0].y;
		}
		else
		{
			grid_x = (unsigned int)floor(keypoints[i].pt.x / gridElemSize);
			grid_y = (unsigned int)floor(keypoints[i].pt.y / gridElemSize);
			if(grid_y < gridSPsize[1])
			{
				if(grid_x < gridSPsize[0])
				{
					x2e(0) = keypoints[i].pt.x + gridSearchParams[grid_y][grid_x].x;
					x2e(1) = keypoints[i].pt.y + gridSearchParams[grid_y][grid_x].y;
					var_searchRadius = (int)gridSearchParams[grid_y][grid_x].z;
				}
				else
				{
					x2e(0) = keypoints[i].pt.x + gridSearchParams[grid_y].back().x;
					x2e(1) = keypoints[i].pt.y + gridSearchParams[grid_y].back().y;
					var_searchRadius = (int)gridSearchParams[grid_y].back().z;
					grid_x = gridSPsizeIdx[0];
				}
			}
			else
			{
				if(grid_x < gridSPsize[0])
				{
					x2e(0) = keypoints[i].pt.x + gridSearchParams.back()[grid_x].x;
					x2e(1) = keypoints[i].pt.y + gridSearchParams.back()[grid_x].y;
					var_searchRadius = (int)gridSearchParams.back()[grid_x].z;
					grid_y = gridSPsizeIdx[1];
				}
				else
				{
					x2e(0) = keypoints[i].pt.x + gridSearchParams.back().back().x;
					x2e(1) = keypoints[i].pt.y + gridSearchParams.back().back().y;
					var_searchRadius = (int)gridSearchParams.back().back().z;
					grid_x = gridSPsizeIdx[0];
					grid_y = gridSPsizeIdx[1];
				}
			}
		}
		/*x2e(0) /= x2e(2);
		x2e(1) /= x2e(2);*/
		if((imgSi.width <= x2e(0)) || (imgSi.height <= x2e(1))) continue;
		/*xdistnorm = x2e(0)-imgSi2.width;
		ydistnorm = x2e(1)-imgSi2.height;
		pDistM = (xdistnorm*xdistnorm+ydistnorm*ydistnorm)/imgDiag;*/
		keypointtree.index->radiusSearch(&x2e(0),/*(1+pDistM)**/(float)var_searchRadius,ret_matches,nanoflann::SearchParams(maxDepthSearch)); //The larger the distance from the image center, the larger the search radius (compensate for radial distortion)
		if(ret_matches.empty()) continue; //in this case there could be a nearest neighbor search (for image regions with large distortions)

		if(ret_matches.size() > NormIdxSize)
			ret_matches.erase(ret_matches.begin()+NormIdxSize,ret_matches.end());

		if(BinaryOrVector)
		{
			if(useBinPopCnt)
			{
				for(unsigned int k = 0; k<ret_matches.size();k++)
				{
					l1norms[k] = getHammingL1PopCnt(descriptors1.row(i), descriptors2.row((int)ret_matches[k].first), byte8width);
				}
			}
			else
			{
				//double* weights = new double[ret_matches.size()];
				for(unsigned int k = 0; k<ret_matches.size();k++)
				{
					l1norms[k] = getHammingL1(descriptors1.row(i), descriptors2.row((int)ret_matches[k].first));
					//*(weights+k) = pow((double)(*(l1norms+k)),2)+36*(1-0.6*pDistM)*ret_matches[k].second; //The larger the distance from the image center, the less important is the distance to the estimated point (compensate for radial distortion)
				}
			}
			idx1 = min_element(l1norms,l1norms + ret_matches.size());
			if(*idx1 > hammL1tresh) continue;

			//Be careful the distance values in the DMatch structs represent the weights of the features and not the distances to the query keypoints
			matches_knn1.push_back(DMatch(i,(int)ret_matches[idx1-l1norms].first,(float)(*idx1)/*sqrtf(powf(distp.x,2)+powf(distp.y,2))*/));
			//delete [] weights;
#if FILTER_WITH_CD_RATIOS
			mprop_tmp.costs = (float)(*idx1);
			mprop_tmp.distance = ret_matches[idx1-l1norms].second;
			mprop_tmp.x = grid_x;
			mprop_tmp.y = grid_y;
			mprops_knn1.push_back(mprop_tmp);
#endif
		}
		else
		{
			for(unsigned int k = 0; k<ret_matches.size();k++)
			{
				l2norms[k] = getL2Distance(descriptors1.row(i), descriptors2.row((int)ret_matches[k].first));
			}
			idx2 = min_element(l2norms,l2norms + ret_matches.size() * sizeof(float));

			//Be careful the distance values in the DMatch structs represent the weights of the features and not the distances to the query keypoints
			matches_knn1.push_back(DMatch(i,(int)ret_matches[idx2-l2norms].first,*idx2));
#if FILTER_WITH_CD_RATIOS
			mprop_tmp.costs = *idx2;
			mprop_tmp.distance = ret_matches[idx2-l2norms].second;
			mprop_tmp.x = grid_x;
			mprop_tmp.y = grid_y;
			mprops_knn1.push_back(mprop_tmp);
#endif
		}
	}

	//if((matches_knn1.size() < 8)/* || (matches_knn1.size() < keypoints.size()/2)*/) return -1; // Too less remaining matches

	for(unsigned int i = 0;i<matches_knn1.size();i++)
	{
		DMatch goodMatch = matches_knn1[i];
#if FILTER_WITH_CD_RATIOS
		mCostDist mprop_good = mprops_knn1[i];
#endif
		for(unsigned int k = i+1;k<matches_knn1.size();k++)
		{
			if(matches_knn1[i].trainIdx == matches_knn1[k].trainIdx)
			{
				if(goodMatch.distance > matches_knn1[k].distance)
				{
					goodMatch = matches_knn1[k];
#if FILTER_WITH_CD_RATIOS
					mprop_good = mprops_knn1[k];
#endif
				}
				matches_knn1.erase(matches_knn1.begin()+k,matches_knn1.begin()+k+1);
#if FILTER_WITH_CD_RATIOS
				mprops_knn1.erase(mprops_knn1.begin()+k,mprops_knn1.begin()+k+1);
#endif
			}
		}
		matches.push_back(goodMatch);
#if FILTER_WITH_CD_RATIOS
		mprops.push_back(mprop_good);
#endif
	}

	if((matches.size() < MIN_FINAL_MATCHES)/* || (matches.size() < keypoints.size()/2)*/) return -2; // Too less remaining matches (maybe because of too many duplicates)

	return 0;
}

/* This function finds the best matches between descriptors by estimating the position of the corresponding keypoint with a
 * precalculated flow in the first place and selecting the nearest neighbors within a specified radius (given by the
 * standard deviation of the flow) for descriptor matching in the second place. For matching the descriptors, the L1-norm
 * (Manhattan distance) of the Hamming distance is calculated.
 *
 * vector<vector<Point3f>> gridSearchParams		Input  -> The vector-structure corresponds to the size of a grid in the image
 *														  (the outer vector corresponds to the rows or
 *														  y-coordinates and the inner vector corresponds to the
 *														  columns or x-coordinates). Each vector-element holds
 *														  3 values. The first corresponds to the average flow
 *														  in x- and the second to the average flow in y-direction
 *														  within the grid element. The third value is a multiple of
 *														  the standard deviation of the flow.
 * float gridElemSize							Input  -> Size of one grid element in the image used by gridSearchParams
 * vector<KeyPoint> keypoints					Input  -> The keypoints of the view for which correspondences in the other
 *														  view should be searched for
 * Mat descriptors1								Input  -> The descriptors of the view corresponding to "keypoints"
 * Mat descriptors2								Input  -> The descriptors of the second view
 * KDTree_D2float &keypointtree					Input  -> The KD-tree (indexes) of the keypoints of the view for which
 *														  correspondences in the other view should be searched for
 * vector<int> keypIndexes						Input  -> A sorted (from low to high) index of keypoints corresponding to
 *														  "keypoints" that were already matched during the calculation of "H"
 * Size imgSi									Input  -> The size of the image
 * vector<vector<DMatch>> &matches				Output -> Array of matches
 * unsigned int knn = 0							Input  -> This variable specifies, how many train indexes should be stored for one query
 *														  index (of the matches) in sorted order (L1-norm of hamming distance). If knn=0,
 *														  the number of train indexes stored for one query index depends on the distance
 *														  (Hamming) to the next (in sorted order (Hamming)) descriptor. The treshold of
 *														  this distance can be adjusted in the code by the variable "knn0hammL1tresh".
 *														  It is not allowed to set knn=1. If you want that, only use a output vector of
 *														  the type vector<DMatch> and no knn.
 * vector<vector<mCostDist>> mprops				Output -> Holds for every match in "matches" the squared distance to the estimated position (SOF)
 *														  as well as the position (index) of the left keypoint of the match in "gridSearchParams"
 *
 * Return value:								 0:		  Everything ok
 *												-1:		  knn must be 0 or >1 or change the type of the variable matches to vector<DMatch>
 *												-2:		  Too less remaining matches
 *												-3:		  Size in dimension 1 (x and bin cols, respectively) must be the same for all bin rows (y coordinates) in gridSearchParams
 */
int guidedMatching(std::vector<std::vector<cv::Point3f>> gridSearchParams,
				   float gridElemSize,
				    std::vector<cv::KeyPoint> keypoints,
				    cv::Mat descriptors1,
				    cv::Mat descriptors2,
				    KDTree_D2float &keypointtree,
				    std::vector<int> keypIndexes,
				    cv::Size imgSi,
				    std::vector<std::vector<cv::DMatch>> &matches,
					unsigned int knn,
                    std::vector<std::vector<mCostDist>> mprops)
{

	if(knn == 1)
		return -1; //knn must be 0 or >1 or change the type of the variable matches to vector<DMatch>

	const int maxDepthSearch = 32;
	const int hammL1tresh = 160;
	const int knn0hammL1tresh = 25;
	const int searchRadius = 10;

	//#define Pfe32u_gt(a,b) ((*a)<(*b))
	#define N_BIT_SHIFTS_IN 6 //only values from 1 to 8 are possible
	#define N_BIT_SHIFTS ((((N_BIT_SHIFTS_IN < 1) ? 1:N_BIT_SHIFTS_IN) > 8) ? 8:N_BIT_SHIFTS_IN)
	#define NormIdxSize (1 << N_BIT_SHIFTS) //If you change the number, also change the values in HamL1NormIdx

	const static unsigned int HamL1NormIdx[NormIdxSize] =
	{
		#define I2(n) n, n+1
		#define I4(n) I2(n), I2(n+2)
		#define I8(n) I4(n), I4(n+4)
		#define I16(n) I8(n), I8(n+8)
		#define I32(n) I16(n), I16(n+16)
		#define I64(n) I32(n), I32(n+32)
		#define I128(n) I64(n), I64(n+64)
		#define I256(n) I128(n), I128(n+128)
		#if N_BIT_SHIFTS == 1
			#define IVAR(n) I2(n)
		#elif N_BIT_SHIFTS == 2
			#define IVAR(n) I4(n)
		#elif N_BIT_SHIFTS == 3
			#define IVAR(n) I8(n)
		#elif N_BIT_SHIFTS == 4
			#define IVAR(n) I16(n)
		#elif N_BIT_SHIFTS == 5
			#define IVAR(n) I32(n)
		#elif N_BIT_SHIFTS == 6
			#define IVAR(n) I64(n)
		#elif N_BIT_SHIFTS == 7
			#define IVAR(n) I128(n)
		#else
			#define IVAR(n) I256(n)
		#endif
		IVAR(0)
		//I16(0), I4(16) //for values 0-19 (NormIdxSize = 20)

	};

	#define MAX_ENLARGE_ITS 2
	unsigned int normIdx[NormIdxSize];
	unsigned int l1norms[NormIdxSize];
	float l2norms[NormIdxSize];
	float var_searchRadius;
	int max_search_its = MAX_ENLARGE_ITS;
	const static float enlarge_sRadius = 1.265625; //1.125*1.125=1.265625 (is squared because the squared range is used within the KD-tree)
	const unsigned int gridSPsize[2] = {gridSearchParams[0].size(), gridSearchParams.size()};//Format [x,y] -> gridSearchParams[y][x]
	const unsigned int gridSPsizeIdx[2] = {gridSPsize[0] - 1, gridSPsize[1] - 1};
#if FILTER_WITH_CD_RATIOS
	mCostDist mprop_tmp = {0,0,0,0};
#endif
	unsigned int grid_x, grid_y;

	if(gridSearchParams[0].size() != gridSearchParams.back().size())
		return -3; //Size in dimension 1 (x and bin cols, respectively) must be the same for all bin rows (y coordinates) in gridSearchParams


	vector<pair<DMatch,double>> matches_knn1;

	int idx = 0;
	Size imgSi2 = cv::Size(imgSi.width/2,imgSi.height/2);
	float imgDiag = (float)(imgSi2.width * imgSi2.width) + (float)(imgSi.height * imgSi.height);//powf((float)imgSi2.width,2)+powf((float)imgSi.height,2);
	Eigen::Matrix3f He;
	//Eigen::Vector3f x1e, x2e;
	Eigen::Vector2f x2e;
	//H.convertTo(H,CV_32F);
	//cv2eigen(H,He);

	//Check type of descriptor
	static bool BinaryOrVector = true;
	static bool useBinPopCnt = false;
	static bool fixProperties = true;
	static unsigned char byte8width = 8;
	static unsigned char descrCols = 64;
	if(fixProperties)
	{
		if(descriptors1.type() == CV_32F)
		{
			BinaryOrVector = false;
		}

		int cpuInfo[4];
// __linux__
		// __cpuid(cpuInfo,0x00000001);
// #endif
		std::bitset<32> f_1_ECX_ = cpuInfo[2];
		bool popcnt_available = f_1_ECX_[23];
		if(popcnt_available)
			useBinPopCnt = true;

		if(descriptors1.cols != 64)
		{
			byte8width = (unsigned char)(descriptors1.cols / 8);
			descrCols = descriptors1.cols;
		}

		fixProperties = false;
	}

	if(gridSPsize[1] == 1)
	{
		gridSearchParams[0][0].z = (float)searchRadius > gridSearchParams[0][0].z ? (float)searchRadius:gridSearchParams[0][0].z;
		gridSearchParams[0][0].z *= gridSearchParams[0][0].z; //The radius is squared because the L2-norm is used within the KD-tree
		gridSearchParams[0][0].z = ceil(gridSearchParams[0][0].z);
		//var_searchRadius = ceil(gridSearchParams[0][0].z);
	}
	else
	{
		for(unsigned int i = 0; i < gridSPsize[1]; i++)
		{
			for(unsigned int j = 0; j < gridSPsize[0]; j++)
			{
				gridSearchParams[i][j].z = (float)searchRadius > gridSearchParams[i][j].z ? (float)searchRadius:gridSearchParams[i][j].z;
				gridSearchParams[i][j].z *= gridSearchParams[i][j].z; //The radius is squared because the L2-norm is used within the KD-tree
				gridSearchParams[i][j].z = ceil(gridSearchParams[i][j].z);
			}
		}
	}

	if(gridSPsize[1] == 1)
	{
		grid_x = grid_y = 0;
	}

	//float pDistM, xdistnorm,ydistnorm;
	for(int i = 0; i<(int)keypoints.size();i++)
	{
		std::vector<std::pair<size_t,float> >   ret_matches;
		max_search_its = MAX_ENLARGE_ITS;
		if(!keypIndexes.empty())
		{
			if(keypIndexes[idx] == i) //continue for already found correspondences
			{
				if(idx < ((int)keypIndexes.size()-1)) idx++;
				continue;
			}
		}
		//Calculate the approximate position of the corresponding keypoint in the second image
		/*x1e << keypoints[i].pt.x, keypoints[i].pt.y, 1.0;
		x2e = He*x1e;
		x2e(0) /= x2e(2);
		x2e(1) /= x2e(2);*/

		if(gridSPsize[1] == 1)
		{
			x2e(0) = keypoints[i].pt.x + gridSearchParams[0][0].x;
			x2e(1) = keypoints[i].pt.y + gridSearchParams[0][0].y;
			var_searchRadius = gridSearchParams[0][0].z;
		}
		else
		{
			grid_x = (unsigned int)floor(keypoints[i].pt.x / gridElemSize);
			grid_y = (unsigned int)floor(keypoints[i].pt.y / gridElemSize);
			if(grid_y < gridSPsize[1])
			{
				if(grid_x < gridSPsize[0])
				{
					x2e(0) = keypoints[i].pt.x + gridSearchParams[grid_y][grid_x].x;
					x2e(1) = keypoints[i].pt.y + gridSearchParams[grid_y][grid_x].y;
					var_searchRadius = gridSearchParams[grid_y][grid_x].z;
				}
				else
				{
					x2e(0) = keypoints[i].pt.x + gridSearchParams[grid_y].back().x;
					x2e(1) = keypoints[i].pt.y + gridSearchParams[grid_y].back().y;
					var_searchRadius = gridSearchParams[grid_y].back().z;
					grid_x = gridSPsizeIdx[0];
				}
			}
			else
			{
				if(grid_x < gridSPsize[0])
				{
					x2e(0) = keypoints[i].pt.x + gridSearchParams.back()[grid_x].x;
					x2e(1) = keypoints[i].pt.y + gridSearchParams.back()[grid_x].y;
					var_searchRadius = gridSearchParams.back()[grid_x].z;
					grid_y = gridSPsizeIdx[1];
				}
				else
				{
					x2e(0) = keypoints[i].pt.x + gridSearchParams.back().back().x;
					x2e(1) = keypoints[i].pt.y + gridSearchParams.back().back().y;
					var_searchRadius = gridSearchParams.back().back().z;
					grid_x = gridSPsizeIdx[0];
					grid_y = gridSPsizeIdx[1];
				}
			}
		}

		if((imgSi.width <= x2e(0)) || (imgSi.height <= x2e(1))) continue;
		/*xdistnorm = x2e(0)-imgSi2.width;
		ydistnorm = x2e(1)-imgSi2.height;
		pDistM = (xdistnorm*xdistnorm+ydistnorm*ydistnorm)/imgDiag;*/

		while(max_search_its > 0)
		{
			keypointtree.index->radiusSearch(&x2e(0),/*(1+pDistM)**/var_searchRadius,ret_matches,nanoflann::SearchParams(maxDepthSearch)); //The larger the distance from the image center, the larger the search radius (compensate for radial distortion)
			if(ret_matches.size() == 1)//Enlarge the search radius if only one match was found so that a ratio test can be performed
			{
				var_searchRadius *= enlarge_sRadius;
				max_search_its--;
				if(max_search_its > 0)
					ret_matches.clear();
			}
			else
			{
				max_search_its = 0;
			}
		}
		if(ret_matches.empty()) continue; //in this case there could be a nearest neighbor search (for image regions with large distortions)

		if(ret_matches.size() > NormIdxSize)
			ret_matches.erase(ret_matches.begin()+NormIdxSize,ret_matches.end());

		//double* weights = new double[ret_matches.size()];

		if(BinaryOrVector)
		{
			if(useBinPopCnt)
			{
				for(unsigned int k = 0; k<ret_matches.size();k++)
				{
					l1norms[k] = getHammingL1PopCnt(descriptors1.row(i), descriptors2.row((int)ret_matches[k].first), byte8width);
				}
			}
			else
			{
				//double* weights = new double[ret_matches.size()];
				for(unsigned int k = 0; k<ret_matches.size();k++)
				{
					l1norms[k] = getHammingL1(descriptors1.row(i), descriptors2.row((int)ret_matches[k].first));
					//*(weights+k) = pow((double)(*(l1norms+k)),2)+36*(1-0.6*pDistM)*ret_matches[k].second; //The larger the distance from the image center, the less important is the distance to the estimated point (compensate for radial distortion)
				}
			}

			if(*min_element(l1norms,l1norms+ret_matches.size()) > hammL1tresh) continue;

			memcpy( normIdx, HamL1NormIdx, ret_matches.size()*sizeof(unsigned int) );

			/*Here the L1-norms are used for sorting, because the number of matching features (corresponding to
			features in the first image) depends on the differences between consecutive L1-norms in the sorted vector.
			If the difference to the next feature is above knn0hammL1tresh, the propability is high that one of the
			matches earlier in this vector is the right match.*/

			INLINEQSORTREORDER(unsigned int, l1norms, normIdx, ret_matches.size(), Pfe32u_lt);

			//vector<pair<double,unsigned int>> sorted;
			if(knn == 0)
			{

				//Be careful the distance values in the DMatch structs represent the L1-norms of the features and not the distances to the query keypoints
				matches.push_back(vector<DMatch>(1,DMatch(i,ret_matches[normIdx[0]].first,(float)l1norms[0])));
#if FILTER_WITH_CD_RATIOS
				mprop_tmp.costs = (float)l1norms[0];
				mprop_tmp.distance = ret_matches[normIdx[0]].second;
				mprop_tmp.x = grid_x;
				mprop_tmp.y = grid_y;
				mprops.push_back(vector<mCostDist>(1,mprop_tmp));
#endif
				for(unsigned int k = 1; k<ret_matches.size();k++)
				{
					if(l1norms[k] - l1norms[0] <= knn0hammL1tresh) //Be careful the distance values in the DMatch structs represent the weights of the features and not the distances to the query keypoints
					{
						matches.back().push_back(DMatch(i,ret_matches[normIdx[k]].first,(float)l1norms[k]));
#if FILTER_WITH_CD_RATIOS
						mprop_tmp.costs = (float)l1norms[k];
						mprop_tmp.distance = ret_matches[normIdx[k]].second;
						mprop_tmp.x = grid_x;
						mprop_tmp.y = grid_y;
						mprops.back().push_back(mprop_tmp);
#endif
					}
					else
						break;
				}
			}
			else
			{
				//Be careful the distance values in the DMatch structs represent the weights of the features and not the distances to the query keypoints
				matches.push_back(vector<DMatch>(1,DMatch(i,ret_matches[normIdx[0]].first,(float)l1norms[0])));
#if FILTER_WITH_CD_RATIOS
				mprop_tmp.costs = (float)l1norms[0];
				mprop_tmp.distance = ret_matches[normIdx[0]].second;
				mprop_tmp.x = grid_x;
				mprop_tmp.y = grid_y;
				mprops.push_back(vector<mCostDist>(1,mprop_tmp));
#endif
				for(unsigned int k = 1; (k<ret_matches.size()) && (k<knn);k++)
				{
					//Be careful the distance values in the DMatch structs represent the weights of the features and not the distances to the query keypoints
					matches.back().push_back(DMatch(i,ret_matches[normIdx[k]].first,(float)l1norms[k]));
#if FILTER_WITH_CD_RATIOS
					mprop_tmp.costs = (float)l1norms[k];
					mprop_tmp.distance = ret_matches[normIdx[k]].second;
					mprop_tmp.x = grid_x;
					mprop_tmp.y = grid_y;
					mprops.back().push_back(mprop_tmp);
#endif
				}
			}

		}
		else
		{
			for(unsigned int k = 0; k<ret_matches.size();k++)
			{
				l2norms[k] = getL2Distance(descriptors1.row(i), descriptors2.row((int)ret_matches[k].first));
			}

			memcpy( normIdx, HamL1NormIdx, ret_matches.size() * sizeof(unsigned int) );

			/*Here the L1-norms are used for sorting, because the number of matching features (corresponding to
			features in the first image) depends on the differences between consecutive L1-norms in the sorted vector.
			If the difference to the next feature is above knn0hammL1tresh, the propability is high that one of the
			matches earlier in this vector is the right match.*/

			INLINEQSORTREORDER(float, l2norms, normIdx, ret_matches.size(), Pfe32u_lt);
			/*vector<pair
			for(unsigned int k = 0; k < ret_matches.size(); k++)*/

			//vector<pair<double,unsigned int>> sorted;
			if(knn == 0)
			{

				//Be careful the distance values in the DMatch structs represent the L1-norms of the features and not the distances to the query keypoints
				matches.push_back(vector<DMatch>(1,DMatch(i,ret_matches[normIdx[0]].first,l2norms[0])));
#if FILTER_WITH_CD_RATIOS
				mprop_tmp.costs = l2norms[0];
				mprop_tmp.distance = ret_matches[normIdx[0]].second;
				mprop_tmp.x = grid_x;
				mprop_tmp.y = grid_y;
				mprops.push_back(vector<mCostDist>(1,mprop_tmp));
#endif
				for(unsigned int k = 1; k<ret_matches.size();k++)
				{
					if(l1norms[k] - l1norms[0] <= knn0hammL1tresh) //Be careful the distance values in the DMatch structs represent the weights of the features and not the distances to the query keypoints
					{
						matches.back().push_back(DMatch(i,ret_matches[normIdx[k]].first,l2norms[k]));
#if FILTER_WITH_CD_RATIOS
						mprop_tmp.costs = l2norms[k];
						mprop_tmp.distance = ret_matches[normIdx[k]].second;
						mprop_tmp.x = grid_x;
						mprop_tmp.y = grid_y;
						mprops.back().push_back(mprop_tmp);
#endif
					}
					else
						break;
				}
			}
			else
			{
				//Be careful the distance values in the DMatch structs represent the weights of the features and not the distances to the query keypoints
				matches.push_back(vector<DMatch>(1,DMatch(i,ret_matches[normIdx[0]].first,l2norms[0])));
#if FILTER_WITH_CD_RATIOS
				mprop_tmp.costs = l2norms[0];
				mprop_tmp.distance = ret_matches[normIdx[0]].second;
				mprop_tmp.x = grid_x;
				mprop_tmp.y = grid_y;
				mprops.push_back(vector<mCostDist>(1,mprop_tmp));
#endif
				for(unsigned int k = 1; (k<ret_matches.size()) && (k<knn);k++)
				{
					//Be careful the distance values in the DMatch structs represent the weights of the features and not the distances to the query keypoints
					matches.back().push_back(DMatch(i,ret_matches[normIdx[k]].first,l2norms[k]));
#if FILTER_WITH_CD_RATIOS
					mprop_tmp.costs = l2norms[k];
					mprop_tmp.distance = ret_matches[normIdx[k]].second;
					mprop_tmp.x = grid_x;
					mprop_tmp.y = grid_y;
					mprops.back().push_back(mprop_tmp);
#endif
				}
			}
		}

		//delete [] weights;

	}

	if((matches.size() < MIN_FINAL_MATCHES)/* || (matches.size() < keypoints.size()/2)*/) return -2; // Too less remaining matches

	return 0;
}


/* This function generates an sparse set of keypoints from a large keypoint set. The algorithm selects keypoints distributed
 * over the whole image with a density given by the grid size divx and divy. For every cell in the image the algorithm
 * selects the keypoints with the largest responses. Thus, only feature detectors can be used in advance that return a response
 * for every keypoint.
 *
 * vector<pair<KeyPoint,int>> &keypInit			Output -> The sparse set of keypoints and the index numbers to the selected keypoints
 * KDTree_D2float &keypointtree					Input  -> The KD-tree (indexes) of the keypoints
 * EMatFloat2 eigkeypts							Input  -> The coordinates of the keypoints organized in RowMajor order in an
 *														  Eigen matrix
 * EMatFloat2 gridPoints						Input  -> The image coordinates within the grid for which near keypoints are
 *														  searched (one coordinate per cell)
 * int divx										Input  -> The number of columns of the grid
 * int divy										Input  -> The number of rows of the grid
 * float imgpart								Input  -> The size (square) of each cell
 * float lastwidthpart							Input  -> The size (square) of the last column
 * const int remainGridPix						Input  -> Threshold for the smallest possible size of the last column of the grid
 *
 * Return value:								none
 */
void get_Sparse_KeypointField(std::vector<std::pair<cv::KeyPoint,int>> &keypInit,
							  KDTree_D2float &keypointtree,
							  EMatFloat2 eigkeypts,
							  EMatFloat2 gridPoints,
							  std::vector<cv::KeyPoint> keypoints,
							  int divx,
							  int divy,
							  float imgpart,
							  float lastwidthpart,
							  const int remainGridPix)
{
	float imgpart2, lwidth2, imgpart22, lwidth22, radius1, radius2;
	int idx;

	const int maxDepthSearch = 32;
	//const int response_thresh[] = {35,18,8}; //bei FAST f�r die Response-Bereiche >100, 50-100 und <50
	const float threshMultFactors[3] = {1.0/4.0, 1.0/8.0, 1.0/16.0}; // for the response-sorted upper, middel and lower third of the correspondences

	const int minNperGridElem = (int)std::ceil(50.0/((float)(divx * divy)));
	//const int maxNperGridElem = eigkeypts.rows()/3.0 < (float)minNperGridElem ? minNperGridElem:(int)std::floor(eigkeypts.rows()/3.0);

	imgpart2 = imgpart/2; //Devide the width/height of the grid element by 2 to get the radius
	lwidth2 = lastwidthpart/2;
	imgpart22 = powf(imgpart2,2); //The grid element radius is squared as nanoflann uses squared radii as input (when L2-norms are used)
	lwidth22 = powf(lwidth2,2);
	radius1 = 2*imgpart22;
	radius2 = 2*lwidth22; //This is the squared diagonal radius of an grid element (c�=a�+b�, a=b => c� = 2a�)

	// do a radius search
	std::vector<std::pair<size_t,float> >   ret_matches;

	std::list<std::pair<cv::KeyPoint,int>> idx_response;
	std::list<std::pair<cv::KeyPoint,int>>::iterator idx_response_it;
	for(int i = 0; i < divy; i++)
	{
		idx = i*divx;
		for(int j = 0; j < ((lastwidthpart <= imgpart2) && (lastwidthpart > remainGridPix) ? (divx-1):divx); j++)
		{
			ret_matches.clear();
			keypointtree.index->radiusSearch(&gridPoints(idx+j,0),radius1,ret_matches,nanoflann::SearchParams(maxDepthSearch));
			int resSize = ret_matches.size();
			for(int k = 0; k < resSize; k++)
			{
				if(ret_matches[k].second > imgpart22)
				{
					if((eigkeypts(ret_matches[k].first,0) > (gridPoints(idx+j,0) - imgpart2)) &&
					   (eigkeypts(ret_matches[k].first,0) < (gridPoints(idx+j,0) + imgpart2)) &&
					   (eigkeypts(ret_matches[k].first,1) > (gridPoints(idx+j,1) - imgpart2)) &&
					   (eigkeypts(ret_matches[k].first,1) < (gridPoints(idx+j,1) + imgpart2))) //Only consider points inside the grid element
					   continue;
					ret_matches.erase(ret_matches.begin()+k,ret_matches.begin()+k+1);
					k--;
					resSize--;
				}
			}
			if(resSize == 0) continue;
			for(int k1 = 0; k1 < resSize; k1++)
			{
				idx_response.push_back(make_pair(keypoints[ret_matches[k1].first],ret_matches[k1].first));
			}

			if(resSize > minNperGridElem)
			{
				std::list<std::pair<cv::KeyPoint,int>>::iterator idx_response_it_start;
				int kptCnt = 0;
				idx_response.sort(sortKeyPointsIdx);

				const float responseDiff = idx_response.front().first.response - idx_response.back().first.response;
				if(responseDiff && (idx_response.size() > 3))
				{
					const float response_thresh[3] = {responseDiff * threshMultFactors[0], responseDiff * threshMultFactors[1], responseDiff * threshMultFactors[2]};
					const int threshPosIndexs[2] = {resSize/3-1, 2*resSize/3-1};

					idx_response_it_start = idx_response.begin();
					std::advance(idx_response_it_start, minNperGridElem < 3 ? 2:(minNperGridElem - 1));
					idx_response_it = idx_response_it_start;
					idx_response_it++;
					for(idx_response_it;idx_response_it!=idx_response.end();idx_response_it++) //Only take a few strong (response) keypoints that have nearly the same response
					{
						if(kptCnt > threshPosIndexs[1])
						{
							if(idx_response_it_start->first.response - idx_response_it->first.response > response_thresh[2])
							{
								idx_response.erase(idx_response_it,idx_response.end());
								break;
							}
						}
						else if(kptCnt > threshPosIndexs[0])
						{
							if(idx_response_it_start->first.response - idx_response_it->first.response > response_thresh[1])
							{
								idx_response.erase(idx_response_it,idx_response.end());
								break;
							}
						}
						else
						{
							if(idx_response_it_start->first.response - idx_response_it->first.response > response_thresh[0])
							{
								idx_response.erase(idx_response_it,idx_response.end());
								break;
							}
						}
						kptCnt++;
					}
				}
			}
			keypInit.insert(keypInit.end(),idx_response.begin(),idx_response.end());
			idx_response.clear();
		}
		if((lastwidthpart <= imgpart2) && (lastwidthpart > remainGridPix))
		{
			ret_matches.clear();
			keypointtree.index->radiusSearch(&gridPoints(idx+divx-1,0),radius2,ret_matches,nanoflann::SearchParams(maxDepthSearch));
			int resSize = ret_matches.size();
			for(int k = 0; k < resSize; k++)
			{
				if(ret_matches[k].second > imgpart22)
				{
					if((eigkeypts(ret_matches[k].first,0) > (gridPoints(idx+divx-1,0) - lwidth2)) &&
					   (eigkeypts(ret_matches[k].first,0) < (gridPoints(idx+divx-1,0) + lwidth2)) &&
					   (eigkeypts(ret_matches[k].first,1) > (gridPoints(idx+divx-1,1) - lwidth2)) &&
					   (eigkeypts(ret_matches[k].first,1) < (gridPoints(idx+divx-1,1) + lwidth2)))
					   continue;
					ret_matches.erase(ret_matches.begin()+k,ret_matches.begin()+k+1);
					k--;
					resSize--;
				}
			}
			if(resSize == 0) continue;
			for(int k1 = 0; k1 < resSize; k1++)
			{
				idx_response.push_back(make_pair(keypoints[ret_matches[k1].first],ret_matches[k1].first));
			}
			if(resSize > minNperGridElem)
			{
				std::list<std::pair<cv::KeyPoint,int>>::iterator idx_response_it_start;
				int kptCnt = 0;
				idx_response.sort(sortKeyPointsIdx);

				const float responseDiff = idx_response.front().first.response - idx_response.back().first.response;
				const float response_thresh[3] = {responseDiff * threshMultFactors[0], responseDiff * threshMultFactors[1], responseDiff * threshMultFactors[2]};
				const int threshPosIndexs[2] = {resSize/3-1, 2*resSize/3-1};

				idx_response_it_start = idx_response.begin();
				std::advance(idx_response_it_start, minNperGridElem - 1);
				idx_response_it = idx_response_it_start;
				idx_response_it++;
				for(idx_response_it;idx_response_it!=idx_response.end();idx_response_it++)
				{
					if(kptCnt > threshPosIndexs[1])
					{
						if(idx_response_it_start->first.response - idx_response_it->first.response > response_thresh[2])
						{
							idx_response.erase(idx_response_it,idx_response.end());
							break;
						}
					}
					else if(kptCnt > threshPosIndexs[0])
					{
						if(idx_response_it_start->first.response - idx_response_it->first.response > response_thresh[1])
						{
							idx_response.erase(idx_response_it,idx_response.end());
							break;
						}
					}
					else
					{
						if(idx_response_it_start->first.response - idx_response_it->first.response > response_thresh[0])
						{
							idx_response.erase(idx_response_it,idx_response.end());
							break;
						}
					}
					kptCnt++;
				}
			}
			keypInit.insert(keypInit.end(),idx_response.begin(),idx_response.end());
			idx_response.clear();
		}
	}
}

#if FILTER_WITH_CD_RATIOS
void getMeanDistCostFactors(std::vector<float> & Dfactors, std::vector<float> & Cfactors,
							std::vector<float> & quartDfactors, std::vector<float> & quartCfactors,
							std::vector<mCostDist> mprops, int si_x, int si_y)
{
	vector<vector<vector<float>>> costBins(si_y, vector<vector<float>>(si_x)), distBins(si_y, vector<vector<float>>(si_x)), costBins_tmp, distBins_tmp;
	vector<vector<float>> costBinsMed(si_y, vector<float>(si_x)), distBinsMed(si_y, vector<float>(si_x));
	vector<vector<float>> costBinsQuart(si_y, vector<float>(si_x)), distBinsQuart(si_y, vector<float>(si_x));
	size_t sumpts;

	if(si_y == 1)
	{
		for(size_t i = 0; i < mprops.size(); i++)
		{
			costBins[0][0].push_back(mprops[i].costs);
			distBins[0][0].push_back(mprops[i].distance);
		}

		sumpts = costBins[0][0].size();
		if(!sumpts)
		{
			return;
		}
		else if(sumpts == 1)
		{
			costBinsMed[0][0] = costBins[0][0][0];
			distBinsMed[0][0] = distBins[0][0][0];
			costBinsQuart[0][0] = costBins[0][0][0];
			distBinsQuart[0][0] = distBins[0][0][0];
		}
		else if(sumpts == 2)
		{
			costBinsMed[0][0] = (costBins[0][0][0] + costBins[0][0][1]) / 2.0f;
			distBinsMed[0][0] = (distBins[0][0][0] + distBins[0][0][1]) / 2.0f;
			costBinsQuart[0][0] = std::max(costBins[0][0][0], costBins[0][0][1]);
			distBinsQuart[0][0] = std::max(distBins[0][0][0], distBins[0][0][1]);
		}
		else
		{
			std::sort(costBins[0][0].begin(),costBins[0][0].end(),[](double const & first, double const & second){
					return first < second;});
			std::sort(distBins[0][0].begin(),distBins[0][0].end(),[](double const & first, double const & second){
					return first < second;});
			if(sumpts % 2)
			{
				costBinsMed[0][0] = costBins[0][0][(sumpts-1)/2];
				distBinsMed[0][0] = distBins[0][0][(sumpts-1)/2];
			}
			else
			{
				costBinsMed[0][0] = (costBins[0][0][sumpts/2] + costBins[0][0][sumpts/2-1]) / 2.0f;
				distBinsMed[0][0] = (distBins[0][0][sumpts/2] + distBins[0][0][sumpts/2-1]) / 2.0f;
			}
			costBinsQuart[0][0] = costBins[0][0][(int)std::floor(0.75f * (float)sumpts)];
			distBinsQuart[0][0] = distBins[0][0][(int)std::floor(0.75f * (float)sumpts)];
		}

		if(costBinsMed[0][0] == 0)
		{
			int idx = (sumpts % 2) ? ((sumpts-1)/2):(sumpts/2-1);
			while((idx < (sumpts-1)) && (costBinsMed[0][0] == 0))
			{
				costBinsMed[0][0] = costBins[0][0][idx];
				idx++;
			}
			if(costBinsMed[0][0] == 0)
				costBinsMed[0][0] = 0.01f;
		}
		if(costBinsQuart[0][0] == 0)
		{
			int idx = (int)std::floor(0.75f * (float)sumpts);
			while((idx < (sumpts-1)) && (costBinsQuart[0][0] == 0))
			{
				costBinsQuart[0][0] = costBins[0][0][idx];
				idx++;
			}
			if(costBinsQuart[0][0] == 0)
				costBinsQuart[0][0] = 0.01f;
		}
		if(distBinsMed[0][0] == 0)
		{
			distBinsMed[0][0] = 0.1f;
		}
		if(distBinsQuart[0][0] == 0)
		{
			distBinsQuart[0][0] = 1.0f;
		}

		for(size_t i = 0; i < mprops.size(); i++)
		{
			Cfactors.push_back(mprops[i].costs / costBinsMed[0][0]);
			Dfactors.push_back(mprops[i].distance / distBinsMed[0][0]);
			quartCfactors.push_back(costBinsQuart[0][0] / costBinsMed[0][0]);
			quartDfactors.push_back(distBinsQuart[0][0] / distBinsMed[0][0]);
		}
	}
	else
	{
		for(size_t i = 0; i < mprops.size(); i++)
		{
			size_t x, y;
			x = mprops[i].x;
			y = mprops[i].y;
			costBins[y][x].push_back(mprops[i].costs);
			distBins[y][x].push_back(mprops[i].distance);
		}
		costBins_tmp = costBins;
		distBins_tmp = distBins;

		//Fill the grid with surrounding values if one bin holds less than 30
		int addArea;
		Eigen::Matrix2i cornerValid;
		size_t minPtsPerSqr = 30;
		for(int i = 0; i < si_y; i++)
		{
			for(int j = 0; j < si_x; j++)
			{
				sumpts = costBins[i][j].size();
				if((sumpts > minPtsPerSqr) || (sumpts == 0))
				{
					continue;
				}
				else
				{
					bool missSqrNotFilled;
					bool noBigSearchWindow = true;
					int addArea_old = 0;
					Eigen::Matrix2i sideValid = Eigen::Matrix2i::Zero(); //directions in matrix: [+x, -x; +y, -y] ->element is 1 if valid. Only 2 elements can be valid at a time (Eigen accesses with idxs (y,x))
					addArea = 0;
					while(sumpts < minPtsPerSqr)
					{
						cornerValid = Eigen::Matrix2i::Ones();
						addArea++;

						for(int k = 0; k < 4; k++)
						{
							int yrange, xrange, xstart, ystart;
							switch(k)
							{
							case 0:
								if(noBigSearchWindow)
								{
									yrange = 1;
									xrange = 2 * addArea + 1;
									xstart = -1 * addArea;
									ystart = addArea;
								}
								else
								{
									if(!sideValid(1,0))
										continue;
									yrange = 1;
									xrange = 2 * addArea_old + 1;
									xstart = -1 * addArea_old;
									ystart = addArea;
								}
								sideValid(1,0) = 0;
								break;
							case 1:
								if(noBigSearchWindow)
								{
									yrange = 1;
									xrange = 2 * addArea + 1;
									xstart = -1 * addArea;
									ystart = -1 * addArea;
								}
								else
								{
									if(!sideValid(1,1))
										continue;
									yrange = 1;
									xrange = 2 * addArea_old + 1;
									xstart = -1 * addArea_old;
									ystart = -1 * addArea;
								}
								sideValid(1,1) = 0;
								break;
							case 2:
								if(noBigSearchWindow)
								{
									yrange = 2 * (addArea - 1) + 1;
									xrange = 1;
									xstart = addArea;
									ystart = -1 * addArea + 1;
								}
								else
								{
									if(!sideValid(0,0))
										continue;
									yrange = 2 * (addArea_old - 1) + 1;
									xrange = 1;
									xstart = addArea;
									ystart = -1 * addArea_old + 1;
								}
								sideValid(0,0) = 0;
								break;
							case 3:
								if(noBigSearchWindow)
								{
									yrange = 2 * (addArea - 1) + 1;
									xrange = 1;
									xstart = -1 * addArea;
									ystart = -1 * addArea + 1;
								}
								else
								{
									if(!sideValid(0,1))
										continue;
									yrange = 2 * (addArea_old - 1) + 1;
									xrange = 1;
									xstart = -1 * addArea;
									ystart = -1 * addArea_old + 1;
								}
								sideValid(0,1) = 0;
							}

							for(int i2 = ystart; i2 < (ystart + yrange); i2++)
							{
								if(i + i2 >= si_y)
								{
									cornerValid(0,0) = cornerValid(0,1) = 0;
									break;
								}
								if(i + i2 < 0)
								{
									cornerValid(1,0) = cornerValid(1,1) = 0;
									continue;
								}
								for(int j2 = xstart; j2 < (xstart + xrange); j2++)
								{
									if(j + j2 >= si_x)
									{
										cornerValid(0,1) = cornerValid(1,1) = 0;
										break;
									}
									if(j + j2 < 0)
									{
										cornerValid(0,0) = cornerValid(1,0) = 0;
										continue;
									}

									if(!costBins[i + i2][j + j2].empty())
									{
										costBins_tmp[i][j].insert(costBins_tmp[i][j].end(), costBins[i + i2][j + j2].begin(), costBins[i + i2][j + j2].end());
										distBins_tmp[i][j].insert(distBins_tmp[i][j].end(), distBins[i + i2][j + j2].begin(), distBins[i + i2][j + j2].end());
										sumpts += costBins[i + i2][j + j2].size();
									}
									switch(k)
									{
									case 0:
										sideValid(1, 0) = 1;
										break;
									case 1:
										sideValid(1, 1) = 1;
										break;
									case 2:
										sideValid(0, 0) = 1;
										break;
									case 3:
										sideValid(0, 1) = 1;
									}
								}
							}
						}
						if(!cornerValid.sum())
						{
							if(!sideValid.sum())
								break;
							if(noBigSearchWindow)
							{
								addArea_old = addArea;
								noBigSearchWindow = false;
							}
						}
					}
				}
			}
		}

		for(int i = 0; i < si_y; i++)
		{
			for(int j = 0; j < si_x; j++)
			{
				sumpts = costBins_tmp[i][j].size();
				if(!sumpts)
				{
					continue;
				}
				else if(sumpts == 1)
				{
					costBinsMed[i][j] = costBins_tmp[i][j][0];
					distBinsMed[i][j] = distBins_tmp[i][j][0];
					costBinsQuart[i][j] = costBins_tmp[i][j][0];
					distBinsQuart[i][j] = distBins_tmp[i][j][0];
				}
				else if(sumpts == 2)
				{
					costBinsMed[i][j] = (costBins_tmp[i][j][0] + costBins_tmp[i][j][1]) / 2.0f;
					distBinsMed[i][j] = (distBins_tmp[i][j][0] + distBins_tmp[i][j][1]) / 2.0f;
					costBinsQuart[i][j] = std::max(costBins_tmp[i][j][0], costBins_tmp[i][j][1]);
					distBinsQuart[i][j] = std::max(distBins_tmp[i][j][0], distBins_tmp[i][j][1]);
				}
				else
				{
					std::sort(costBins_tmp[i][j].begin(),costBins_tmp[i][j].end(),[](double const & first, double const & second){
							return first < second;});
					std::sort(distBins_tmp[i][j].begin(),distBins_tmp[i][j].end(),[](double const & first, double const & second){
							return first < second;});
					if(sumpts % 2)
					{
						costBinsMed[i][j] = costBins_tmp[i][j][(sumpts-1)/2];
						distBinsMed[i][j] = distBins_tmp[i][j][(sumpts-1)/2];
					}
					else
					{
						costBinsMed[i][j] = (costBins_tmp[i][j][sumpts/2] + costBins_tmp[i][j][sumpts/2-1]) / 2.0f;
						distBinsMed[i][j] = (distBins_tmp[i][j][sumpts/2] + distBins_tmp[i][j][sumpts/2-1]) / 2.0f;
					}
					costBinsQuart[i][j] = costBins_tmp[i][j][(int)std::floor(0.75f * (float)sumpts)];
					distBinsQuart[i][j] = distBins_tmp[i][j][(int)std::floor(0.75f * (float)sumpts)];
				}
				if(costBinsMed[i][j] == 0)
				{
					int idx = (sumpts % 2) ? ((sumpts-1)/2):(sumpts/2-1);
					while((idx < (sumpts-1)) && (costBinsMed[i][j] == 0))
					{
						costBinsMed[i][j] = costBins_tmp[i][j][idx];
						idx++;
					}
					if(costBinsMed[i][j] == 0)
						costBinsMed[i][j] = 0.01f;
				}
				if(costBinsQuart[i][j] == 0)
				{
					int idx = (int)std::floor(0.75f * (float)sumpts);
					while((idx < (sumpts-1)) && (costBinsQuart[i][j] == 0))
					{
						costBinsQuart[i][j] = costBins_tmp[i][j][idx];
						idx++;
					}
					if(costBinsQuart[i][j] == 0)
						costBinsQuart[i][j] = 0.01f;
				}
				if(distBinsMed[i][j] == 0)
				{
					distBinsMed[i][j] = 0.1f;
				}
				if(distBinsQuart[i][j] == 0)
				{
					distBinsQuart[i][j] = 1.0f;
				}
			}
		}
		for(size_t i = 0; i < mprops.size(); i++)
		{
			size_t x, y;
			x = mprops[i].x;
			y = mprops[i].y;
			Cfactors.push_back(mprops[i].costs / costBinsMed[y][x]);
			Dfactors.push_back(mprops[i].distance / distBinsMed[y][x]);
			quartCfactors.push_back(costBinsQuart[y][x] / costBinsMed[y][x]);
			quartDfactors.push_back(distBinsQuart[y][x] / distBinsMed[y][x]);
		}
	}
}
#endif
