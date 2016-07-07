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

#include "matchinglib_imagefeatures.h"
//#include "opencv2\xfeatures2d\nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <Eigen/Core>

//#include "opencv2/nonfree/nonfree.hpp"
//#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

namespace matchinglib
{

/* --------------------------- Defines --------------------------- */

struct KeypointResponseGreaterThanThreshold2
{
    KeypointResponseGreaterThanThreshold2(float _value) :
    value(_value)
    {
    }
    inline bool operator()(const KeyPoint& kpt) const
    {
        return kpt.response >= value;
    }
    float value;
};

struct KeypointResponseGreater2
{
    inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const
    {
        return kp1.response > kp2.response;
    }
};

typedef struct keyPIdx{
	keyPIdx(int _idx, float _response) :
	idx(_idx),
	response(_response)
	{
    }
    int idx;
    float response;
}keyPIdx;

struct ResponseGreaterThanThreshold
{
    ResponseGreaterThanThreshold(float _value) :
    value(_value)
    {
    }
    inline bool operator()(const keyPIdx& kpt) const
    {
        return kpt.response >= value;
    }
    float value;
};

/* --------------------- Function prototypes --------------------- */

//Compares the response of two keypoints
bool sortKeyPoints(cv::KeyPoint first, cv::KeyPoint second);
//Filters keypoints with a small response value within multiple grids.
void responseFilterGridBased(std::vector<cv::KeyPoint>& keys, cv::Size imgSi, int number);
//Sorts the input vector based on the response values (largest elements first) until the given number is reached.
int sortResponses(std::vector<keyPIdx>& keys, int number);

/* --------------------- Functions --------------------- */

/* This function calculates the keypoints in an image
 *
 * Mat img							Input  -> Input image
 * vector<KeyPoint>* keypoints		Output -> Pointer to the keypoints
 * string featuretype				Input  -> Algorithm for calculating the features. The following inputs are possible:
 *											  OpenCV 2.4.9: FAST, STAR, SIFT, SURF, ORB, BRISK, MSER, GFTT, HARRIS, Dense, SimpleBlob
 *											  OpenCV 3.0: FAST, STAR, (SIFT, SURF,) ORB, BRISK, MSER, KAZE, AKAZE
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
int getKeypoints(cv::Mat img, std::vector<cv::KeyPoint>& keypoints, std::string featuretype, bool dynamicKeypDet, int limitNrfeatures)
{
	const int minnumfeatures = 10, maxnumfeatures = limitNrfeatures;

	/*if(!featuretype.compare("SIFT") || !featuretype.compare("SURF"))
		cv::initModule_nonfree();*/

	if(dynamicKeypDet == true)
	{
		//if(!featuretype.compare("GFTT"))
		//	{
		//		std::vector<cv::Point2f> corners;
		//		int cornerCount = 0,cnt = 0, mindist = 16;
		//		double quallev = 0.01;
		//		bool qualdist = true;
		//		while ((cornerCount < minnumfeatures) && (cnt < 3))
		//		{
		//			cv::goodFeaturesToTrack(img,corners,maxnumfeatures,quallev,mindist);
		//			cornerCount = (int)corners.size();
		//			if( qualdist == true )
		//			{
		//				quallev /= 2;
		//				qualdist = false;
		//			}
		//			else
		//			{
		//				mindist /= 2;
		//				qualdist = true;
		//			}
		//			cnt++;
		//		}
		//		if(cnt >= 3)
		//		{
		//			fprintf(stderr,"Only %d corners were detected!\n",cornerCount);
		//			return -1; //Too less features detected
		//		}
		//		/*CvTermCriteria crita;
		//		crita.max_iter = 10;
		//		crita.epsilon = 0.01;
		//		crita.type = CV_TERMCRIT_EPS;
		//		cornerSubPix(img,corners,Size(3,3),Size(-1,-1),crita);*/
		//		KeyPoint::convert(corners,*keypoints,1.0f,0);
		//	}
		//else if(!featuretype.compare("SURF"))
		//	{
		//		Ptr<FeatureDetector> detector(new SURF(500));
		//		if(detector.empty())
		//		{
		//			cout << "Cannot create feature detector!" << endl;
		//			return -2; //Error creating feature detector
		//		}
		//		detector->detect(img,*keypoints);
		//		if(keypoints->size() < minnumfeatures)
		//		{
		//			detector.release();
		//			/*detector = new DynamicAdaptedFeatureDetector(new SurfAdjuster(400,150,maxnumfeatures),
		//																	minnumfeatures,maxnumfeatures,10);*/
		//			int imgrows = 4, imgcols = 4, max_grid_features, min_grid_features;
		//			if((img).rows > 400)
		//			{
		//				imgrows = (int)ceilf(((float)((img).rows))/100.0f);
		//			}
		//			if((img).cols > 400)
		//			{
		//				imgcols = (int)ceilf(((float)((img).cols))/100.0f);
		//			}
		//			max_grid_features = (int)ceil((float)maxnumfeatures/((float)(imgrows * imgcols)));
		//			max_grid_features = max_grid_features > 200 ? max_grid_features:200;

		//			min_grid_features = (int)ceil((float)minnumfeatures/((float)(imgrows * imgcols)));
		//			min_grid_features = min_grid_features > 10 ? min_grid_features:10;
		//		
		//			detector = new GridAdaptedFeatureDetector(new DynamicAdaptedFeatureDetector(new SurfAdjuster(),
		//														min_grid_features,max_grid_features,10),maxnumfeatures,imgrows,imgcols);
		//			keypoints->clear();
		//			if(detector.empty())
		//			{
		//				cout << "Cannot create feature detector!" << endl;
		//				return -2; //Error creating feature detector
		//			}
		//			detector->detect(img,*keypoints);
		//			if(keypoints->size() < minnumfeatures)
		//			{
		//				return -1; //Too less features detected
		//			}
		//		}
		//		/*else if(keypoints->size() > maxnumfeatures)
		//			{
		//				std::sort(keypoints->begin(),keypoints->end(),sortKeyPoints);
		//				keypoints->erase(keypoints->begin()+maxnumfeatures,keypoints->begin()+keypoints->size());
		//			}*/
		//		
		//		//detector.release();
		//	}
		//else if(!featuretype.compare("FAST"))
		//	{
		//		/*int imgrows = 7, imgcols = 7, max_grid_features, min_grid_features;
		//		if((img).rows > 700)
		//		{
		//			imgrows = (int)ceilf(((float)((img).rows))/100.0);
		//		}
		//		if((img).cols > 700)
		//		{
		//			imgcols = (int)ceilf(((float)((img).cols))/100.0);
		//		}

		//		max_grid_features = (int)ceil((float)maxnumfeatures/((float)(imgrows * imgcols)));
		//		max_grid_features = max_grid_features > 200 ? max_grid_features:200;

		//		min_grid_features = (int)ceil((float)minnumfeatures/((float)(imgrows * imgcols)));
		//		min_grid_features = min_grid_features > 10 ? min_grid_features:10;*/
		//		
		//		/*Ptr<FeatureDetector> detector = cv::FeatureDetector::create("FAST");*/
		//		/*Ptr<FeatureDetector> detector(new GridAdaptedFeatureDetector(new FastAdjuster(),maxnumfeatures,imgrows,imgcols));*/
		//		Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector(new FastAdjuster(),	minnumfeatures,maxnumfeatures,10));
		//		/*Ptr<FeatureDetector> detector(new GridAdaptedFeatureDetector(new DynamicAdaptedFeatureDetector(new FastAdjuster(),
		//																		min_grid_features,max_grid_features,10),maxnumfeatures,imgrows,imgcols));*/
		//		
		//		//cv::KeyPointsFilter::retainBest(*keypoints, maxnumfeatures); //--------------> auch andere Filter verfügbar

		//		if(detector.empty())
		//		{
		//			cout << "Cannot create feature detector!" << endl;
		//			return -2; //Error creating feature detector
		//		}
		//		detector->detect(img,*keypoints);

		//		if(keypoints->size() < minnumfeatures)
		//		{
		//			return -1; //Too less features detected
		//		}

		//		/*if(keypoints->size() > maxnumfeatures)
		//		{
		//			std::sort(keypoints->begin(),keypoints->end(),sortKeyPoints);
		//			keypoints->erase(keypoints->begin()+maxnumfeatures,keypoints->begin()+keypoints->size());
		//		}*/

		//		/*CvTermCriteria crita;
		//		crita.max_iter = 10;
		//		crita.epsilon = 0.01;
		//		crita.type = CV_TERMCRIT_EPS;
		//		std::vector<cv::Point2f> corners;
		//		KeyPoint::convert(*keypoints,corners);
		//		cornerSubPix(img,corners,Size(3,3),Size(-1,-1),crita);
		//		for(size_t i = 0;i<keypoints->size();i++)
		//			keypoints->at(i).pt = corners.at(i);*/
		//		//KeyPoint::convert(corners,*keypoints);

		//		detector.release();
		//	}
		//else if(!featuretype.compare("STAR"))
		//	{
		//		Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector(new StarAdjuster(),
		//																		minnumfeatures,maxnumfeatures,10));
		//		if(detector.empty())
		//		{
		//			cout << "Cannot create feature detector!" << endl;
		//			return -2; //Error creating feature detector
		//		}
		//		detector->detect(img,*keypoints);
		//		if(keypoints->size() > maxnumfeatures)
		//		{
		//			cv::KeyPointsFilter::retainBest(*keypoints, maxnumfeatures); //--------------> auch andere Filter verfügbar
		//			/*std::sort(keypoints->begin(),keypoints->end(),sortKeyPoints);
		//			keypoints->erase(keypoints->begin()+maxnumfeatures,keypoints->begin()+keypoints->size());*/
		//		}
		//		if(keypoints->size() < minnumfeatures)
		//		{
		//			return -1; //Too less features detected
		//		}

		//		//detector.release();
		//	}
		//else
		//	{
		//		cout << "Only GFTT, SURF, FAST and STAR are supported using the user specific version of the algorithms (parameters)!" << endl;
		//		return -3; //No such feature detector
		//	}

		//cout << "Dynamic keypoint detection is since OpenCV 3.0 not available! Performing response filtering." << endl;

		Ptr<FeatureDetector> detector;

		if(!featuretype.compare("FAST"))
		{
			detector = FastFeatureDetector::create();
		}
		else if(!featuretype.compare("MSER"))
		{
			detector = MSER::create();
		}
		else if(!featuretype.compare("ORB"))
		{
			detector = ORB::create();
		}
		else if(!featuretype.compare("BRISK"))
		{
			detector = BRISK::create();
		}
		else if(!featuretype.compare("KAZE"))
		{
			detector = KAZE::create();
		}
		else if(!featuretype.compare("AKAZE"))
		{
			detector = AKAZE::create();
		}
		/*else if(!featuretype.compare("SIFT"))
		{
			detector = xfeatures2d::SIFT::create();
		}
		else if(!featuretype.compare("SURF"))
		{
			detector = xfeatures2d::SURF::create();
		}*/
		else if(!featuretype.compare("STAR"))
		{
			detector = xfeatures2d::StarDetector::create();
		}
		else
		{
			cout << "This feature detector is not supported!" << endl;
			return -3;
		}
				
		if(detector.empty())
		{
			cout << "Cannot create feature detector!" << endl;
			return -2; //Error creating feature detector
		}
		detector->detect( img, keypoints );

		responseFilterGridBased(keypoints, img.size(), maxnumfeatures);

	}
	else
	{
		Ptr<FeatureDetector> detector;// = FeatureDetector::create( featuretype );

		if(!featuretype.compare("FAST"))
		{
			detector = FastFeatureDetector::create();
		}
		else if(!featuretype.compare("MSER"))
		{
			detector = MSER::create();
		}
		else if(!featuretype.compare("ORB"))
		{
			detector = ORB::create();
		}
		else if(!featuretype.compare("BRISK"))
		{
			detector = BRISK::create();
		}
		else if(!featuretype.compare("KAZE"))
		{
			detector = KAZE::create();
		}
		else if(!featuretype.compare("AKAZE"))
		{
			detector = AKAZE::create();
		}
		/*else if(!featuretype.compare("SIFT"))
		{
			detector = xfeatures2d::SIFT::create();
		}
		else if(!featuretype.compare("SURF"))
		{
			detector = xfeatures2d::SURF::create();
		}*/
		else if(!featuretype.compare("STAR"))
		{
			detector = xfeatures2d::StarDetector::create();
		}
		else
		{
			cout << "This feature detector is not supported!" << endl;
			return -3;
		}
				
		if(detector.empty())
		{
			cout << "Cannot create feature detector!" << endl;
			return -2; //Error creating feature detector
		}

		detector->detect( img, keypoints );

        if((int)keypoints.size() > maxnumfeatures)
		{
			cv::KeyPointsFilter::retainBest(keypoints, maxnumfeatures); //--------------> auch andere Filter verfügbar
		}

		if(keypoints.size() < minnumfeatures)
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
 *										  (OpenCV 2.4.9: FREAK, SIFT, SURF, ORB, BRISK, BriefDescriptorExtractor)
 *										  (OpenCV 3.0: FREAK, SIFT, SURF, ORB, BRISK, KAZE, AKAZE, DAISY, LATCH)
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

	/*if(!extractortype.compare("SIFT") || !extractortype.compare("SURF"))
	{
		cv::initModule_nonfree();
	}*/

	cv::Ptr<cv::DescriptorExtractor> extractor;// = cv::DescriptorExtractor::create(extractortype);

	if(!extractortype.compare("BRISK"))
	{
		extractor = BRISK::create();
	}
	else if(!extractortype.compare("ORB"))
	{
		extractor = ORB::create();
	}
	else if(!extractortype.compare("KAZE"))
	{
		extractor = KAZE::create();
	}
	else if(!extractortype.compare("AKAZE"))
	{
		extractor = AKAZE::create();
	}
	else if(!extractortype.compare("FREAK"))
	{
		extractor = xfeatures2d::FREAK::create();
	}
	/*else if(!extractortype.compare("SIFT"))
	{
		extractor = xfeatures2d::SIFT::create();
	}
	else if(!extractortype.compare("SURF"))
	{
		extractor = xfeatures2d::SURF::create();
	}*/
	else if(!extractortype.compare("DAISY"))
	{
		extractor = xfeatures2d::DAISY::create();
	}
	else if(!extractortype.compare("LATCH"))
	{
		extractor = xfeatures2d::LATCH::create();
	}
	/*else if(!extractortype.compare("LUCID"))
	{
		extractor = xfeatures2d::LUCID::create(1,2);
		cout << "Be careful: The lucid_kernel and blur_kernel size are set arbitrary! RGB images should be used!" << endl;
	}*/
    else {
        cout << "This extractor type is not supported!" << endl;
        return -3;
    }

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

/* Filters keypoints with a small response value within multiple grids. The image is devided into multiple grids
 * depending on the size of the image. The grid sizes range from an 1x1 grid (whole image) to a grid with a maximum
 * cell size of 100 pixels whereas the reduction of the grid size is in the range of 1.5 to 3.0 depending again on
 * the size of the image. Thus, an optimal reduction factor is found by reducing the image border overlap of the
 * cells. Moreover, each grid (size) is used 4 times - it is shifted in x- and y- direction to reduce effects that
 * would change the result due to cell border effects. Each keypoint is scored within each grid and the cells it 
 * belongs to, respectively. The scoring is based on its local/global response and the percentage of keypoints that
 * should be kept. Thus, keypoints with the highest score within the specified number of keypoints are kept.
 *
 * vector<KeyPoint> keys		Input/Output  -> Keypoints
 * Size imgSi					Input		  -> Size of the image
 * int number					Input		  -> Number of keypoints that should be kept
 *
 * Return value:				none
 */
void responseFilterGridBased(std::vector<cv::KeyPoint>& keys, cv::Size imgSi, int number)
{
	vector<int> gridsizes, halfgridsizes;
	int minGrSi = imgSi.height / 4; //minimum grid size of the larger grid (2 times the smaller grid)
	int maxGrSi = imgSi.height / 2; //maximum grid size of the larger grid (2 times the smaller grid)
	float sizeOptions = (float)maxGrSi - (float)minGrSi + 1.0f;
	float dimx = (float)imgSi.width;
	float dimy = (float)imgSi.height;
	int minElem = INT_MAX;
	vector<keyPIdx> scores;
	float reduction = (float)number / (float)keys.size();
	vector<cv::KeyPoint> keys_tmp;

	if((number >= (int)keys.size()) || (number == 0))
		return;

	if(reduction > 0.875) //if only 12.5% (=1/8) should be removed
	{
		cv::KeyPointsFilter::retainBest(keys, number);
		return;
	}

	while(minElem >= 100) //Generates new grid sizes until a grid size smaller than 100 pixels was estimated before
	{
		vector<float> errterm;
		vector<float>::iterator minElemIt;
		if(minElem <  imgSi.height)
		{
			minGrSi = minElem / 3;
			maxGrSi = (int)floor((float)minElem / 1.5f);
			sizeOptions = (float)maxGrSi - (float)minGrSi + 1.0f;
		}
		//Find the grid size with the smallest remaining pixels at the borders
		for(float i = 0; i < sizeOptions; i++)
		{
			float d = (float)minGrSi + i;
			float tmp1 = dimx - floor(dimx/d) * d;
			float tmp2 = dimy - floor(dimy/d) * d;
			errterm.push_back(tmp1 * tmp1 + tmp2 * tmp2);
		}
		//Find the grid size with smallest border overlap (if there are multiple -> the first in the vector)
		minElemIt = ++min_element(errterm.begin(), errterm.end());//take the element above the minimum
		//Get the largest grid size with the same overlap
		for(vector<float>::iterator i = minElemIt--; i < errterm.end(); ++i)//set i to the element above the minimum and set minElemIt to the minimum
		{
			if(*i <= *minElemIt)
			{
				minElemIt = i;
			}
		}
		minElem = minGrSi + (minElemIt - errterm.begin());
		gridsizes.push_back(minElem);
	}

	for(size_t i = 0; i < gridsizes.size(); ++i)
	{
		halfgridsizes.push_back(gridsizes[i] / 2);
	}
	
	//Get the strongest responses in the whole image
	{
		//first use nth element to partition the keypoints into the best and worst.
		std::nth_element(keys.begin(), keys.begin() + number, keys.end(), KeypointResponseGreater2());
		//this is the boundary response, and in the case of FAST may be ambigous
		float ambiguous_response = keys[number - 1].response;
		//use std::partition to grab all of the keypoints with the boundary response.
		std::vector<KeyPoint>::const_iterator new_end =
		std::partition(keys.begin() + number, keys.end(),
						KeypointResponseGreaterThanThreshold2(ambiguous_response));

		//Generate a scoring vector for the keypoints and increment the score for keypoints with the highest response in the image
		int accept_keys = new_end - keys.begin();
		for(int i = 0; i < (int)keys.size(); ++i)
		{
			if(i < accept_keys)
			{
				scores.push_back(keyPIdx(i, 1.0f));
			}
			else
			{
				scores.push_back(keyPIdx(i, 0));
			}
		}
	}

	//Generate vectors for all grids
	vector<vector<vector<vector<keyPIdx>>>> gridsOrig((int)gridsizes.size(), vector<vector<vector<keyPIdx>>>());
	vector<vector<vector<vector<keyPIdx>>>> gridsLeft((int)gridsizes.size(), vector<vector<vector<keyPIdx>>>()); //Start of grid shifted half cell size to the left & added one additional cell at the right side
	vector<vector<vector<vector<keyPIdx>>>> gridsUp((int)gridsizes.size(), vector<vector<vector<keyPIdx>>>()); //Start of grid shifted half cell size below 0 & added one additional cell in y-dimension
	vector<vector<vector<vector<keyPIdx>>>> gridsCenter((int)gridsizes.size(), vector<vector<vector<keyPIdx>>>()); //Combination of the two other shifted grids
	for(size_t i = 0; i < gridsizes.size(); ++i)
	{
		int siX = (int)floor(dimx / (float)gridsizes[i]);
		int siY = (int)floor(dimy / (float)gridsizes[i]);
		gridsOrig[i].resize(siX, vector<vector<keyPIdx>>(siY));
		gridsLeft[i].resize(siX + 1, vector<vector<keyPIdx>>(siY));
		gridsUp[i].resize(siX, vector<vector<keyPIdx>>(siY + 1));
		gridsCenter[i].resize(siX + 1, vector<vector<keyPIdx>>(siY + 1));
	}

	//Assign keypoint indices and responses to cells of all grids
	for(int i = 0; i < (int)keys.size(); ++i)
	{
		float x = keys[i].pt.x;
		float y = keys[i].pt.y;
		float response = keys[i].response;
		for(size_t j = 0; j < gridsizes.size(); ++j)
		{
			int posX = (int)floor(x/(float)gridsizes[j]);
			int posY = (int)floor(y/(float)gridsizes[j]);
			int posX1 = (int)floor((x - (float)halfgridsizes[j])/(float)gridsizes[j]) + 1;
			int posY1 = (int)floor((y - (float)halfgridsizes[j])/(float)gridsizes[j]) + 1;

			//Check if the keypoint is near the border
			if(posX >= (int)gridsOrig[j].size())
			{
				posX--;
			}
			if(posY >= (int)gridsOrig[j][0].size())
			{
				posY--;
			}
			
			gridsOrig[j][posX][posY].push_back(keyPIdx(i,response));
			gridsLeft[j][posX1][posY].push_back(keyPIdx(i,response));
			gridsUp[j][posX][posY1].push_back(keyPIdx(i,response));
			gridsCenter[j][posX1][posY1].push_back(keyPIdx(i,response));
		}
	}

	//Get for every cell the strongest percentage (variable reduction) of responses and score the keypoints
	for(size_t i = 0; i < gridsOrig.size(); ++i)
	{
		for(size_t j = 0; j < gridsOrig[i].size(); ++j)
		{
			for(size_t k = 0; k < gridsOrig[i][j].size(); ++k)
			{
				int first_bad = sortResponses(gridsOrig[i][j][k], (int)floor((float)gridsOrig[i][j][k].size() * reduction + 0.5f));
                for(int k1 = 0; k1 < first_bad; k1++)
				{
					scores[gridsOrig[i][j][k][k1].idx].response++;
				}
			}
			for(size_t k = 0; k < gridsLeft[i][j].size(); ++k)
			{
				int first_bad = sortResponses(gridsLeft[i][j][k], (int)floor((float)gridsLeft[i][j][k].size() * reduction + 0.5f));
                for(int k1 = 0; k1 < first_bad; k1++)
				{
					scores[gridsLeft[i][j][k][k1].idx].response++;
				}
			}
			for(size_t k = 0; k < gridsUp[i][j].size(); ++k)
			{
				int first_bad = sortResponses(gridsUp[i][j][k], (int)floor((float)gridsUp[i][j][k].size() * reduction + 0.5f));
                for(int k1 = 0; k1 < first_bad; k1++)
				{
					scores[gridsUp[i][j][k][k1].idx].response++;
				}
			}
			for(size_t k = 0; k < gridsCenter[i][j].size(); ++k)
			{
				int first_bad = sortResponses(gridsCenter[i][j][k], (int)floor((float)gridsCenter[i][j][k].size() * reduction + 0.5f));
                for(int k1 = 0; k1 < first_bad; k1++)
				{
					scores[gridsCenter[i][j][k][k1].idx].response++;
				}
			}
		}
	}

	//Sort the score
	std::sort(scores.begin(), scores.end(), [](keyPIdx first, keyPIdx second){return first.response > second.response;});

	{
		int idx1, idx2;
		idx1 = idx2 = number - 1;
		float hlp = scores[idx1].response;
		idx2++;
		while(std::abs(scores[idx2].response - hlp) < 1e-3)
		{
			idx2++;
		}
		idx2--;
		if(idx2 == idx1) //If all elements with the same score would be accepted ("number" is at the boundary between two scores)
		{
			for(int i = 0; i <= idx2; i++)
			{
				keys_tmp.push_back(keys[scores[i].idx]);
			}
		}
		else
		{
			vector<keyPIdx> scoredResp;
			idx1--;
			while(std::abs(scores[idx1].response - hlp) < 1e-3)
			{
				idx1--;
			}
			idx1++;

			int remaining = number - idx1;

			for(int i = idx1; i <= idx2; i++)
			{
				scoredResp.push_back(keyPIdx(scores[i].idx, keys[scores[i].idx].response));
			}

			remaining = sortResponses(scoredResp, remaining);

			for(int i = 0; i < idx1; i++)
			{
				keys_tmp.push_back(keys[scores[i].idx]);
			}
			for(int i = 0; i < remaining; i++)
			{
				keys_tmp.push_back(keys[scoredResp[i].idx]);
			}
		}		
	}

	keys = keys_tmp;
}

/* Sorts the input vector based on the response values (largest elements first) until the given number is reached.
 * The remaining elements (at the end of the vector) stay unsorted.
 *
 * vector<keyPIdx> keys			Input/Output  -> Indices and responses of the keypoints
 * int number					Input		  -> Number of elements that should be sorted
 *
 * Return value:				Last position within the vector that holds the response equal to the position at 
 *								"number" of the sorted vector. E.g. number=4, sorted responses=[9,8,4,4,4,4,1,3,2,1,3],
 *								return value=5
 */
int sortResponses(std::vector<keyPIdx>& keys, int number)
{
	if(keys.empty() || (number == 0))
		return 0;
	//first use nth element to partition the keypoints into the best and worst.
	std::nth_element(keys.begin(), keys.begin() + number, keys.end(), [](keyPIdx first, keyPIdx second){return first.response > second.response;});
	//this is the boundary response, and in the case of FAST may be ambigous
	float ambiguous_response = keys[number - 1].response;
	//use std::partition to grab all of the keypoints with the boundary response.
	std::vector<keyPIdx>::const_iterator new_end = std::partition(keys.begin() + number, keys.end(), ResponseGreaterThanThreshold(ambiguous_response));

	return new_end - keys.begin();
}

bool IsFeatureTypeSupported(const std::string &type)
{
    std::vector<std::string> vecSupportedTypes = GetSupportedFeatureTypes();
    if(std::find(vecSupportedTypes.begin(), vecSupportedTypes.end(), type) != vecSupportedTypes.end()) return true;
    return false;
}

std::vector<std::string> GetSupportedFeatureTypes()
{
    static std::string types [] = {"FAST", "MSER", "ORB", "BRISK", "KAZE", "AKAZE", "SIFT", "SURF", "STAR"};
    return std::vector<std::string>(types, types + 9);
}

bool IsExtractorTypeSupported(const std::string &type)
{
    std::vector<std::string> vecSupportedTypes = GetSupportedExtractorTypes();
    if(std::find(vecSupportedTypes.begin(), vecSupportedTypes.end(), type) != vecSupportedTypes.end()) return true;
    return false;
}

std::vector<std::string> GetSupportedExtractorTypes()
{
    static std::string types [] = {"BRISK", "ORB", "KAZE", "AKAZE", "FREAK", "SIFT", "SURF", "DAISY", "LATCH", "LUCID"};
    return std::vector<std::string>(types, types + 10);
}


} // namepace matchinglib
