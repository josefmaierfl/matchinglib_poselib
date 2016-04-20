/**********************************************************************************************************
 FILE: features.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Straße 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for extracting keypoints and generating descriptors as 
			  well as for sub-pixel refinement
**********************************************************************************************************/

#include "features.h"

//#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

namespace matchinglib
{

/* --------------------- Function prototypes --------------------- */

//Compares the response of two keypoints
bool sortKeyPoints(cv::KeyPoint first, cv::KeyPoint second);

/* --------------------- Functions --------------------- */

/* This function calculates the keypoints in an image
 *
 * Mat img							Input  -> Input image
 * vector<KeyPoint>* keypoints		Output -> Pointer to the keypoints
 * string featuretype				Input  -> Algorithm for calculating the features. The following inputs are possible:
 *											  FAST, STAR, SIFT, SURF, ORB, BRISK, MSER, GFTT, HARRIS, Dense, SimpleBlob
 *											  -> see the OpenCV documentation for further details on the different methods
 * bool dynamicKeypDet				Input  -> If true [Default], the number of features is limited to a specific nr. of 
 *											  features using dynamic versions of the feature detectors. Only GFTT, SURF, 
 *											  FAST and STAR are supported using this option.
 * int limitNrfeatures				Input  -> Maximum number of features that should remain after filtering or dynamic 
 *											  feature detection [Default=8000].
 *
 * Return value:					 0:		  Everything ok
 *									-1:		  Too less features detected
 *									-2:		  Error creating feature detector
 *									-3:		  No such feature detector
 */
int getKeypoints(cv::Mat img, std::vector<cv::KeyPoint>* keypoints, std::string featuretype, bool dynamicKeypDet, int limitNrfeatures)
{
	const int minnumfeatures = 10, maxnumfeatures = limitNrfeatures;

	if(!featuretype.compare("SIFT") || !featuretype.compare("SURF"))
		cv::initModule_nonfree();

	if(dynamicKeypDet == true)
	{
		if(!featuretype.compare("GFTT"))
			{
				std::vector<cv::Point2f> corners;
				int cornerCount = 0,cnt = 0, mindist = 16;
				double quallev = 0.01;
				bool qualdist = true;
				while ((cornerCount < minnumfeatures) && (cnt < 3))
				{
					cv::goodFeaturesToTrack(img,corners,maxnumfeatures,quallev,mindist);
					cornerCount = (int)corners.size();
					if( qualdist == true )
					{
						quallev /= 2;
						qualdist = false;
					}
					else
					{
						mindist /= 2;
						qualdist = true;
					}
					cnt++;
				}
				if(cnt >= 3)
				{
					fprintf(stderr,"Only %d corners were detected!\n",cornerCount);
					return -1; //Too less features detected
				}
				/*CvTermCriteria crita;
				crita.max_iter = 10;
				crita.epsilon = 0.01;
				crita.type = CV_TERMCRIT_EPS;
				cornerSubPix(img,corners,Size(3,3),Size(-1,-1),crita);*/
				KeyPoint::convert(corners,*keypoints,1.0f,0);
			}
		else if(!featuretype.compare("SURF"))
			{
				Ptr<FeatureDetector> detector(new SURF(500));
				if(detector.empty())
				{
					cout << "Cannot create feature detector!" << endl;
					return -2; //Error creating feature detector
				}
				detector->detect(img,*keypoints);
				if(keypoints->size() < minnumfeatures)
				{
					detector.release();
					/*detector = new DynamicAdaptedFeatureDetector(new SurfAdjuster(400,150,maxnumfeatures),
																			minnumfeatures,maxnumfeatures,10);*/
					int imgrows = 4, imgcols = 4, max_grid_features, min_grid_features;
					if((img).rows > 400)
					{
						imgrows = (int)ceilf(((float)((img).rows))/100.0f);
					}
					if((img).cols > 400)
					{
						imgcols = (int)ceilf(((float)((img).cols))/100.0f);
					}
					max_grid_features = (int)ceil((float)maxnumfeatures/((float)(imgrows * imgcols)));
					max_grid_features = max_grid_features > 200 ? max_grid_features:200;

					min_grid_features = (int)ceil((float)minnumfeatures/((float)(imgrows * imgcols)));
					min_grid_features = min_grid_features > 10 ? min_grid_features:10;
				
					detector = new GridAdaptedFeatureDetector(new DynamicAdaptedFeatureDetector(new SurfAdjuster(),
																min_grid_features,max_grid_features,10),maxnumfeatures,imgrows,imgcols);
					keypoints->clear();
					if(detector.empty())
					{
						cout << "Cannot create feature detector!" << endl;
						return -2; //Error creating feature detector
					}
					detector->detect(img,*keypoints);
					if(keypoints->size() < minnumfeatures)
					{
						return -1; //Too less features detected
					}
				}
				/*else if(keypoints->size() > maxnumfeatures)
					{
						std::sort(keypoints->begin(),keypoints->end(),sortKeyPoints);
						keypoints->erase(keypoints->begin()+maxnumfeatures,keypoints->begin()+keypoints->size());
					}*/
				
				//detector.release();
			}
		else if(!featuretype.compare("FAST"))
			{
				/*int imgrows = 7, imgcols = 7, max_grid_features, min_grid_features;
				if((img).rows > 700)
				{
					imgrows = (int)ceilf(((float)((img).rows))/100.0);
				}
				if((img).cols > 700)
				{
					imgcols = (int)ceilf(((float)((img).cols))/100.0);
				}

				max_grid_features = (int)ceil((float)maxnumfeatures/((float)(imgrows * imgcols)));
				max_grid_features = max_grid_features > 200 ? max_grid_features:200;

				min_grid_features = (int)ceil((float)minnumfeatures/((float)(imgrows * imgcols)));
				min_grid_features = min_grid_features > 10 ? min_grid_features:10;*/
				
				/*Ptr<FeatureDetector> detector = cv::FeatureDetector::create("FAST");*/
				/*Ptr<FeatureDetector> detector(new GridAdaptedFeatureDetector(new FastAdjuster(),maxnumfeatures,imgrows,imgcols));*/
				Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector(new FastAdjuster(),	minnumfeatures,maxnumfeatures,10));
				/*Ptr<FeatureDetector> detector(new GridAdaptedFeatureDetector(new DynamicAdaptedFeatureDetector(new FastAdjuster(),
																				min_grid_features,max_grid_features,10),maxnumfeatures,imgrows,imgcols));*/
				
				//cv::KeyPointsFilter::retainBest(*keypoints, maxnumfeatures); //--------------> auch andere Filter verfügbar

				if(detector.empty())
				{
					cout << "Cannot create feature detector!" << endl;
					return -2; //Error creating feature detector
				}
				detector->detect(img,*keypoints);

				if(keypoints->size() < minnumfeatures)
				{
					return -1; //Too less features detected
				}

				/*if(keypoints->size() > maxnumfeatures)
				{
					std::sort(keypoints->begin(),keypoints->end(),sortKeyPoints);
					keypoints->erase(keypoints->begin()+maxnumfeatures,keypoints->begin()+keypoints->size());
				}*/

				/*CvTermCriteria crita;
				crita.max_iter = 10;
				crita.epsilon = 0.01;
				crita.type = CV_TERMCRIT_EPS;
				std::vector<cv::Point2f> corners;
				KeyPoint::convert(*keypoints,corners);
				cornerSubPix(img,corners,Size(3,3),Size(-1,-1),crita);
				for(size_t i = 0;i<keypoints->size();i++)
					keypoints->at(i).pt = corners.at(i);*/
				//KeyPoint::convert(corners,*keypoints);

				detector.release();
			}
		else if(!featuretype.compare("STAR"))
			{
				Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector(new StarAdjuster(),
																				minnumfeatures,maxnumfeatures,10));
				if(detector.empty())
				{
					cout << "Cannot create feature detector!" << endl;
					return -2; //Error creating feature detector
				}
				detector->detect(img,*keypoints);
				if(keypoints->size() > maxnumfeatures)
				{
					cv::KeyPointsFilter::retainBest(*keypoints, maxnumfeatures); //--------------> auch andere Filter verfügbar
					/*std::sort(keypoints->begin(),keypoints->end(),sortKeyPoints);
					keypoints->erase(keypoints->begin()+maxnumfeatures,keypoints->begin()+keypoints->size());*/
				}
				if(keypoints->size() < minnumfeatures)
				{
					return -1; //Too less features detected
				}

				//detector.release();
			}
		else
			{
				cout << "Only GFTT, SURF, FAST and STAR are supported using the user specific version of the algorithms (parameters)!" << endl;
				return -3; //No such feature detector
			}
	}
	else
	{
		Ptr<FeatureDetector> detector = FeatureDetector::create( featuretype );
		if(detector.empty())
		{
			cout << "Cannot create feature detector!" << endl;
			return -2; //Error creating feature detector
		}
		detector->detect( img, *keypoints );

		if(keypoints->size() > maxnumfeatures)
		{
			cv::KeyPointsFilter::retainBest(*keypoints, maxnumfeatures); //--------------> auch andere Filter verfügbar
		}

		if(keypoints->size() < minnumfeatures)
		{
			return -1; //Too less features detected
		}
	}

	return 0;
}

/* Extraction of descriptors at given keypoint locations
 *
 * Mat img1						Input  -> Input image
 * vector<KeyPoint> keypoints	Input  -> Locations (keypoints) for which descriptors should be extracted
 * string extractortype			Input  -> Methode for extracting the descriptors
 *										  (FREAK, SIFT, SURF, ORB, BRISK, BriefDescriptorExtractor)
 * Mat descriptors				Output -> Extracted descriptors (row size corresponds to number of 
 *										  descriptors and features, respectively)
 *
 * Return value:				 0:		  Everything ok
 *								-1:		  Cannot create descriptor extractor
 */
int getDescriptors(cv::Mat img,
				   std::vector<cv::KeyPoint> & keypoints,
				   std::string extractortype,
				   cv::Mat & descriptors)
{

	if(!extractortype.compare("SIFT") || !extractortype.compare("SURF"))
		cv::initModule_nonfree();

	cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create(extractortype);

	if(extractor.empty())
	{
		fprintf(stderr,"Cannot create descriptor extractor!\n");
		return -1; //Cannot create descriptor extractor
	}

	extractor->compute(img,keypoints,descriptors);

	return 0;
}

/* This function compares the response of two keypoints to be able to sort them accordingly.
 * 
 * KeyPoint first				Input  -> First Keypoint
 * KeyPoint second				Input  -> Second Keypoint
 */
bool sortKeyPoints(cv::KeyPoint first, cv::KeyPoint second)
{
	return first.response > second.response;
}

}