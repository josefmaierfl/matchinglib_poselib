/**********************************************************************************************************
 FILE: ratioMatches_Flann.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: October 2015

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: Workaround for using the flann lib with OpenCV lib using the cv namespace in other files
**********************************************************************************************************/

#include "../include/ratioMatches_Flann.h"
//#include "flann/flann.hpp"
#include "opencv2/flann.hpp"

using namespace std;

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

/* --------------------- Functions --------------------- */

/* This function matches keypoints from two images and returns only matches for which the ratio between the
 * distances of the best and second-closest match are below a threshold. If too less matches are remaining
 * after ratio test (<30), 50% of the matches with the best ratio are returned.
 *
 * const Mat& descriptors1						Input  -> The descriptors of the keypoints in the left (first) image
 * const Mat& descriptors2						Input  -> The descriptors of the keypoints in the right (second) image
 * vector<DMatch>& filteredMatches12			Output -> Fitered matches
 * double *estim_inlRatio						Output -> If this pointer is not NULL [Default=NULL], the estimated inlier
 *														  ratio of the data is returned
 * bool onlyRatTestMatches						Input  -> If true [Default = false], only matches that passed the ratio
 *														  test are returned regardless of the number of matches
 *
 * Return value:								true:		  Everything ok
 *												false:		  Too less matches are left
 */
bool ratioTestFlannMatches(const cv::Mat descriptors1, const cv::Mat descriptors2,
                         std::vector<cv::DMatch>& filteredMatches12, double *estim_inlRatio, bool onlyRatTestMatches)
{
	int nn = 2;
	filteredMatches12.clear();
	if(descriptors1.type() == CV_32F)
	{
		cvflann::Matrix<float> dataset((float*)descriptors2.data, descriptors2.rows, descriptors2.cols);
		cvflann::Matrix<float> query((float*)descriptors1.data, descriptors1.rows, descriptors1.cols);

		cvflann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
		cvflann::Matrix<float> dists(new float[query.rows*nn], query.rows, nn);
		/*vector<vector<int>> indices;
		vector<vector<float>> dists;*/

		// construct an randomized kd-tree index using 8 kd-trees
		cvflann::Index<cvflann::L2<float>> index(dataset, cvflann::KDTreeIndexParams(8));
		index.buildIndex();

		// do a knn search, using 32 checks (higher values lead to higher precision, but slower performance)
		index.knnSearch(query, indices, dists, nn, cvflann::SearchParams(64));

		//Ratio test
		for(size_t q = 0; q < indices.rows; q++)
		{
			if(dists[q][0] < (0.75f * dists[q][1]))
			{
				cv::DMatch match;
				match.distance = dists[q][0];
				match.queryIdx = q;
				match.trainIdx = indices[q][0];
				filteredMatches12.push_back(match);
			}
		}

		if(estim_inlRatio)
		{
			*estim_inlRatio = (double)filteredMatches12.size() / (double)indices.rows;
		}

		if((filteredMatches12.size() < 30) && !onlyRatTestMatches) //Get 50% (if 50% are larger than 30) of the matches with the best ratio
		{
			vector<pair<size_t,float>> matches2;
			size_t matchesSize = indices.rows;
			filteredMatches12.clear();
			for(size_t q = 0; q < matchesSize; q++)
			{
                matches2.push_back(make_pair(q, dists[q][0] / dists[q][1]));
			}
			//Sort ratios to get the smaller values first (better ratio)
			sort(matches2.begin(), matches2.end(),
				[](pair<size_t,float> first, pair<size_t,float> second){return first.second < second.second;});
			if((matchesSize > 60) && (matchesSize < 120))
				matchesSize /= 2;
			else if(matchesSize > 120)
				matchesSize = 60;
			while((matchesSize > 0) && (matches2[matchesSize - 1].second > 0.85f))
				matchesSize--;
			for(size_t q = 0; q < matchesSize; q++)
			{
				cv::DMatch match;
				int idx = (int)matches2[q].first;
				match.distance = dists[idx][0];
				match.queryIdx = idx;
				match.trainIdx = indices[idx][0];
				filteredMatches12.push_back(match);
			}
		}
		/*delete[] indices.ptr();
		delete[] dists.ptr();*/
		delete[] indices.data;
		delete[] dists.data;
	}
	else if(descriptors1.type() == CV_8U)
	{
		cvflann::Matrix<unsigned char> dataset(descriptors2.data, descriptors2.rows, descriptors2.cols);
		cvflann::Matrix<unsigned char> query(descriptors1.data, descriptors1.rows, descriptors1.cols);
		
		cvflann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
		cvflann::Matrix<int> dists(new int[query.rows*nn], query.rows, nn);
		/*vector<vector<int>> indices;
		vector<vector<int>> dists;*/

		// construct a hierarchical clustering index
		cvflann::Index<cvflann::HammingLUT> index(dataset, cvflann::HierarchicalClusteringIndexParams());
		index.buildIndex();

		// do a knn search, using 64 checks (higher values lead to higher precision, but slower performance)
		index.knnSearch(query, indices, dists, nn, cvflann::SearchParams(64));

		//Ratio test
		for(size_t q = 0; q < indices.rows; q++)
		{
			if((float)dists[q][0] < (0.75f * (float)dists[q][1]))
			{
				cv::DMatch match;
				match.distance = (float)dists[q][0];
				match.queryIdx = q;
				match.trainIdx = indices[q][0];
				filteredMatches12.push_back(match);
			}
		}

		if(estim_inlRatio)
		{
			*estim_inlRatio = (double)filteredMatches12.size() / (double)indices.rows;
		}

		if(filteredMatches12.size() < 30) //Get 50% (if 50% are larger than 30) of the matches with the best ratio
		{
			vector<pair<size_t,float>> matches2;
			size_t matchesSize = indices.rows;
			filteredMatches12.clear();
			for(size_t q = 0; q < matchesSize; q++)
			{
                matches2.push_back(make_pair(q, (float)dists[q][0] / (float)dists[q][1]));
			}
			//Sort ratios to get the smaller values first (better ratio)
			sort(matches2.begin(), matches2.end(),
				[](pair<size_t,float> first, pair<size_t,float> second){return first.second < second.second;});
			if((matchesSize > 60) && (matchesSize < 120))
				matchesSize /= 2;
			else if(matchesSize > 120)
				matchesSize = 60;
			while((matchesSize > 0) && (matches2[matchesSize - 1].second > 0.85f))
				matchesSize--;
			for(size_t q = 0; q < matchesSize; q++)
			{
				cv::DMatch match;
				int idx = (int)matches2[q].first;
				match.distance = (float)dists[idx][0];
				match.queryIdx = idx;
				match.trainIdx = indices[idx][0];
				filteredMatches12.push_back(match);
			}
		}
		/*delete[] indices.ptr();
		delete[] dists.ptr();*/
		delete[] indices.data;
		delete[] dists.data;
	}
	else
	{
		cout << "Format of descriptors not supported!" << endl;
		return false;
	}

	return true;
}
