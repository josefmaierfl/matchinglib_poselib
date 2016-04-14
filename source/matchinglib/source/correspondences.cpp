/**********************************************************************************************************
 FILE: correspondences.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for generating matched feature sets out of image 
			  information.
**********************************************************************************************************/

#include "correspondences.h"
#include "features.h"
#include "matchers.h"
#include "match_statOptFlow.h"

//#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
//#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

namespace matchinglib
{

/* --------------------- Function prototypes --------------------- */



/* --------------------- Functions --------------------- */

/* Generation of features followed by matching, filtering, and subpixel-refinement.
 *
 * Mat img1						Input  -> First or left input image
 * Mat img2						Input  -> First or right input image
 * vector<DMatch> finalMatches	Output -> Matched features
 * string featuretype			Input  -> Name of feature detector. Possible detectors [Default=FAST] are:
 *										  FAST, STAR, SIFT, SURF, ORB, BRISK, MSER, GFTT, HARRIS, Dense, SimpleBlob
 *										  -> see the OpenCV documentation for further details on the different methods
 * string extractortype			Input  -> Methode for extracting the descriptors. The following inputs [Default=FREAK]
 *										  are possible:
 *										  FREAK, SIFT, SURF, ORB, BRISK, BriefDescriptorExtractor
 *										  -> see the OpenCV documentation for further details on the different methods
 * string matchertype			Input  -> The abbreviation of the used matcher [Default = GMBSOF].
 *										  Possible inputs are:
 *											CASHASH: Cascade Hashing matcher
 *											GMBSOF: Guided matching based on statistical optical flow
 *											HIRCLUIDX: Hirarchical Clustering Index Matching from the FLANN library
 *											HIRKMEANS: hierarchical k-means tree matcher from the FLANN library
 *											LINEAR: linear Matching algorithm (Brute force) from the FLANN library
 *											LSHIDX: LSH Index Matching algorithm from the FLANN library
 *											RANDKDTREE: randomized KD-trees matcher from the FLANN library
 * bool dynamicKeypDet			Input  -> If true [Default], the number of features is limited to a specific nr. of 
 *										  features using dynamic versions of the feature detectors. Only GFTT, SURF, 
 *										  FAST and STAR are supported using this option.
 * int limitNrfeatures			Input  -> Maximum number of features that should remain after filtering or dynamic 
 *										  feature detection [Default=8000].
 * bool VFCrefine				Input  -> If true [Default=false], the matches are refined using the vector field
 *										  consensus (VFC) algorithm. It is not recommended for non-rigid scenes or 
 *										  scenes with volatile depth changes (The optical flow should smmoothly change
 *										  over the whole image). Otherwise, matches at such objects or "borders" are
 *										  rejected. The filter performance works quite well for all other kind of 
 *										  scenes and inlier ratios above 20%.
 * bool ratioTest				Input  -> If true [Default=true], a ratio test is performed after matching. If 
 *										  VFCrefine=true, the ratio test is performed in advance to VFC. For
 *										  GMBSOF, a ratio test is always performed independent of the value of
 *										  ratioTest. If you wand to change this behaviour for GMBSOF, this has to
 *										  be changed directly at the call of function AdvancedMatching. Moreover,
 *										  no ratio test can be performed for the CASHASH matcher.
 * bool SOFrefine				Input  -> If true [Default=false], the matches are filtered using the statistical
 *										  optical flow (SOF). This only maks sence, if the GMBSOF algorithm is not
 *										  used, as GMBSOF depends on SOF. This filtering strategy is computational
 *										  expensive for a large amount of matches.
 * bool subPixRefine			Input  -> If true [Default = false], the subpixel-positions of matched keypoints are 
 *										  calculated by template matching. Be careful, if there are large rotations, 
 *										  changes in scale or other feature deformations between the matches, this 
 *										  method should not be called.
 *
 * Return value:				 0:		  Everything ok
 *								-1:		  Cannot create descriptor extractor
 */
int getCorrespondences(cv::Mat img1, 
					   cv::Mat img2,
					   std::vector<cv::DMatch> & finalMatches,
					   std::string featuretype, 
					   std::string extractortype, 
					   std::string matchertype, 
					   bool dynamicKeypDet,
					   int limitNrfeatures, 
					   bool VFCrefine, 
					   bool ratioTest,
					   bool SOFrefine,
					   bool subPixRefine)
{
	if(img1.empty() || img2.empty())
	{
		cout << "No image information provided!" << endl;
		return -1;
	}
	if(((img1.rows != img2.rows) || (img1.cols != img2.cols)) && !matchertype.compare("GMBSOF"))
	{
		cout << "Images should have the same size when using GMBSOF! There might be errors!" << endl;
		//return -1;
	}
	if(((img1.rows != img2.rows) || (img1.cols != img2.cols)) && SOFrefine)
	{
		cout << "Images should have the same size when using SOF-filtering! There might be errors!" << endl;
		//return -1;
	}
	if((!featuretype.compare("MSER") || !featuretype.compare("SimpleBlob")) && !matchertype.compare("GMBSOF"))
	{
		cout << "MSER and SimpleBlob do not provide response information which is necessary for GMBSOF!" << endl;
		return -1;
	}

	vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	cv::Size imgSi = img1.size();

	if(getKeypoints(img1, &keypoints1, featuretype, dynamicKeypDet, limitNrfeatures) != 0) return -2; //Error while calculating keypoints
	if(getKeypoints(img2, &keypoints2, featuretype, dynamicKeypDet, limitNrfeatures) != 0) return -2; //Error while calculating keypoints

	if(getDescriptors(img1, keypoints1, extractortype, descriptors1) != 0) return -3; //Error while extracting descriptors
	if(getDescriptors(img2, keypoints2, extractortype, descriptors2) != 0) return -3; //Error while extracting descriptors

	if(getMatches(keypoints1, keypoints2, descriptors1, descriptors2, imgSi, finalMatches, matchertype, VFCrefine, ratioTest) != 0) return -4; //Matching failed

	if(SOFrefine)
	{
		if(!matchertype.compare("GMBSOF"))
		{
			cout << "SOF-filtering makes no sence when using GMBSOF! Skipping the filtering step." << endl;
		}
		else
		{
			vector<int> queryIdxs( finalMatches.size() ), trainIdxs( finalMatches.size() );
			vector<cv::KeyPoint> keypoints1_tmp, keypoints2_tmp;
			cv::Mat inliers;
			std::vector<cv::DMatch> finalMatches_tmp;
			for( size_t i = 0; i < finalMatches.size(); i++ )
			{
				queryIdxs[i] = finalMatches[i].queryIdx;
				trainIdxs[i] = finalMatches[i].trainIdx;
			}

			for(size_t i = 0; i < finalMatches.size(); i++ )
			{
				keypoints1_tmp.push_back(keypoints1.at(queryIdxs[i]));
				keypoints2_tmp.push_back(keypoints2.at(trainIdxs[i]));
			}


			/* ONLY FOR DEBUGGING START */
			//Mat drawImg;
			//drawMatches( *img1, keypoints1, *img2, keypoints2, matches, drawImg );
			////imwrite("C:\\work\\bf_matches_cross-check.jpg", drawImg);
			//cv::namedWindow( "Source_1", CV_WINDOW_NORMAL );
			//cv::imshow( "Source_1", drawImg );
			//cv::waitKey(0);

			//vector<char> matchesMask( matches.size(), 0 );
			/* ONLY FOR DEBUGGING END */

			EMatFloat2 keyP1(keypoints1_tmp.size(),2), keyP2(keypoints2_tmp.size(),2);
			for(unsigned int i = 0; i<keypoints1_tmp.size();i++)
			{
				keyP1(i,0) = keypoints1_tmp[i].pt.x;
				keyP1(i,1) = keypoints1_tmp[i].pt.y;
			}

			for(unsigned int i = 0; i<keypoints2_tmp.size();i++)
			{
				keyP2(i,0) = keypoints2_tmp[i].pt.x;
				keyP2(i,1) = keypoints2_tmp[i].pt.y;
			}

			std::vector<std::vector<cv::Point3f>> gridSearchParams;
			float gridElemSize;
			if(getStatisticalMatchingPositions(keyP1, keyP2, imgSi, gridSearchParams, &gridElemSize, inliers) != 0)
			{
				//Calculation of flow statistic failed
				cout << "Filtering with SOF failed! Taking unfiltered matches." << endl;
			}
			else
			{
				
				/* ONLY FOR DEBUGGING START */
				//for( size_t i1 = 0; i1 < keypoints1_tmp.size(); i1++)
				//{
				//	if( inliers.at<bool>(i1,0) == true )
				//	{
				//		matchesMask[i1] = 1;
				//	}
				//}
				//drawMatches( *img1, keypoints1, *img2, keypoints2, matches, drawImg, Scalar::all(-1)/*CV_RGB(0, 255, 0)*/, CV_RGB(0, 0, 255), matchesMask);
				////imwrite("C:\\work\\bf_matches_filtered_stat_flow.jpg", drawImg);
				//cv::imshow( "Source_1", drawImg );
				//cv::waitKey(0);
				/* ONLY FOR DEBUGGING END */

				for( size_t i1 = 0; i1 < finalMatches.size(); i1++)
				{
					if( inliers.at<bool>((int)i1,0) == true )
					{
						finalMatches_tmp.push_back(finalMatches[i1]);
					}
				}
				if(finalMatches_tmp.size() >= MIN_FINAL_MATCHES)
				{
					finalMatches = finalMatches_tmp;
				}
			}
		}
	}

	if(subPixRefine)
	{
		std::vector<bool> inliers;
		vector<int> queryIdxs( finalMatches.size() ), trainIdxs( finalMatches.size() );
		vector<cv::KeyPoint> keypoints1_tmp, keypoints2_tmp;
		std::vector<cv::DMatch> finalMatches_tmp;
		for( size_t i = 0; i < finalMatches.size(); i++ )
		{
			queryIdxs[i] = finalMatches[i].queryIdx;
			trainIdxs[i] = finalMatches[i].trainIdx;
		}

		for(size_t i = 0; i < finalMatches.size(); i++ )
		{
			keypoints1_tmp.push_back(keypoints1.at(queryIdxs[i]));
			keypoints2_tmp.push_back(keypoints2.at(trainIdxs[i]));
		}

		if(getSubPixMatches(img1, img2, &keypoints1_tmp, &keypoints2_tmp, &inliers) != 0)
		{
			cout << "Subpixel refinement would have rejected too many matches -> skipped. Taking integer positions instead." << endl;
		}
		else
		{
			for(size_t i = 0; i < finalMatches.size(); i++)
			{
				keypoints2[trainIdxs[i]] = keypoints2_tmp[i];
			}
			for(int i = finalMatches.size() - 1; i >= 0; i--)
			{
				if(inliers[i])
				{
					finalMatches_tmp.push_back(finalMatches[i]);
				}
			}
			finalMatches = finalMatches_tmp;
		}
	}
	
	return 0;
}

}