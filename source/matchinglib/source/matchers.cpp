/**********************************************************************************************************
 FILE: match_statOptFlow.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Straße 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for matching features
**********************************************************************************************************/
#include "matchinglib_matchers.h"

#include "match_statOptFlow.h"
#include "matchinglib_imagefeatures.h"

#include "CascadeHash/Share.h"
#include "CascadeHash/DataPreProcessor.h"
#include "CascadeHash/HashConvertor.h"
#include "CascadeHash/BucketBuilder.h"
#include "CascadeHash/MatchPairLoader.h"
#include "CascadeHash/CasHashMatcher.h"

#include "nmslib\nmslib_matchers.h"

//#include "flann/flann.hpp"
#include "opencv2/flann.hpp"

#include <map>
#include <algorithm>

#include "vfcMatches.h"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//using namespace cv;
using namespace std;

namespace matchinglib
{

  /* --------------------- Function prototypes --------------------- */

//This function compares the weights of the matches to be able to sort them accordingly while keeping track of the index.
  bool sortMatchWeightIdx(std::pair<double,unsigned int>& first, std::pair<double,unsigned int>& second);
//The Cascade Hashing matcher
  int cashashMatching(cv::Mat const& descrL, cv::Mat const& descrR, std::vector<cv::DMatch> & matches);

  /* --------------------- Functions --------------------- */

  /* Matches 2 feature sets with an user selectable matching algorithm.
   *
   * vector<KeyPoint> keypoints1    Input  -> Keypoints in the first or left image
   * vector<KeyPoint>* keypoints    Input  -> Keypoints in the second or right image
   * Mat descriptors1         Input  -> Descriptors of the keypoints in the first or left image
   * Mat descriptors2         Input  -> Descriptors of the keypoints in the second or right image
   * Size imgSi           Input  -> Size of the image (first and second image must have the same size
   *                        (width and height) if the GMbSOF algorithm is used)
   * vector<DMatch> finalMatches    Output -> Vector of matched features
   * string matcher_name        Input  -> The abbreviation of the used matcher [Default = GMBSOF].
   *                        Possible inputs are:
   *                        CASHASH: Cascade Hashing matcher
   *                        GMBSOF: Guided matching based on statistical optical flow
   *                        HIRCLUIDX: Hirarchical Clustering Index Matching from the FLANN library
   *                        HIRKMEANS: hierarchical k-means tree matcher from the FLANN library
   *                        LINEAR: linear Matching algorithm (Brute force) from the FLANN library
   *                        LSHIDX: LSH Index Matching algorithm from the FLANN library
   *                        RANDKDTREE: randomized KD-trees matcher from the FLANN library
   * bool VFCrefine         Input  -> If true [Default=false], the matches are refined using the vector field
   *                        consensus (VFC) algorithm. It is not recommended for non-rigid scenes or
   *                        scenes with volatile depth changes (The optical flow should smmoothly change
   *                        over the whole image). Otherwise, matches at such objects or "borders" are
   *                        rejected. The filter performance works quite well for all other kind of
   *                        scenes and inlier ratios above 20%.
   * bool ratioTest         Input  -> If true [Default=true], a ratio test is performed after matching. If
   *                        VFCrefine=true, the ratio test is performed in advance to VFC. For
   *                        GMBSOF, a ratio test is always performed independent of the value of
   *                        ratioTest. If you wand to change this behaviour for GMBSOF, this has to
   *                        be changed directly at the call of function AdvancedMatching. Moreover,
   *                        no ratio test can be performed for the CASHASH matcher.
   * string idxPars_NMSLIB		Input  -> Index parameters for matchers of the NMSLIB. See manual of NMSLIB for details.
   *					  [Default = ""]
   * string queryPars_NMSLIB	Input  -> Query-time parameters for matchers of the NMSLIB. See manual of NMSLIB for details.
   *					  [Default = ""]
   *
   * Return value:           0:     Everything ok
   *                  -1:     Wrong input data
   *                  -2:     Matcher not supported
   *                  -3:     Matching algorithm failed
   *                  -4:     Too less keypoits
   */
  int getMatches(std::vector<cv::KeyPoint> const& keypoints1, std::vector<cv::KeyPoint> const& keypoints2,
                 const cv::Mat &descriptors1, const cv::Mat &descriptors2, cv::Size imgSi, std::vector<cv::DMatch> & finalMatches,
                 const string &matcher_name, bool VFCrefine, bool ratioTest, std::string idxPars_NMSLIB, std::string queryPars_NMSLIB)
  {
    CV_Assert(descriptors1.type() == descriptors2.type());
    int err;

    if((keypoints1.size() < 15) || (keypoints2.size() < 15))
    {
      cout << "Too less keypoits!" << endl;
      return -4;
    }

    if(((int)keypoints1.size() != descriptors1.rows) || ((int)keypoints2.size() != descriptors2.rows))
    {
      cout << "Number of descriptors must be equal to the number of keypoints!" << endl;
      return -1;
    }

    finalMatches.clear();

    if(!matcher_name.compare("GMBSOF"))
    {
      cv::Ptr<cv::DescriptorMatcher> matcher;// = NULL;//DescriptorMatcher::create( "BruteForce-Hamming" );

      /*if(descriptors1.cols % 8)
      {
        cout << "For GMBSOF, the number of descriptor bytes must be a multiple of 8!" << endl;
        return -1;
      } */
      if(!(keypoints1[0].response > FLT_MIN) || !(keypoints2[0].response > FLT_MIN))
      {
        cout << "Keypoint response must be available for GMBSOF!" << endl;
        return -1;
      }

      if(!ratioTest)
      {
        cout << "GMbSOF is executed by default with ratio test. If you want to change this, you have to edit the options of function AdvancedMatching and recompile."
             << endl;
        cout << "Performing GMbSOF with ratio test!" << endl;
      }

      err = AdvancedMatching( matcher, keypoints1, keypoints2, descriptors1, descriptors2, imgSi,
                              finalMatches);//, true, 0.3, 3.5,  2, false, NULL, NULL, NULL, NULL, imgs[0], imgs[1]);

      if(err)
      {
        cout << "GMbSOF failed with code " << err << endl;
        return -3;
      }
    }
    else if(!matcher_name.compare("CASHASH"))
    {
      if(descriptors1.type() != CV_32F)
      {
        cout << "Wrong descriptor data type for CASHASH! Must be 32bit float." << endl;
        return -1;
      }

      err = cashashMatching(descriptors1, descriptors2, finalMatches);

      if(err)
      {
        cout << "CASHASH failed with code " << err << endl;
        return -3;
      }
    }
	else if (!matcher_name.compare("SWGRAPH"))
	{
		if (descriptors1.type() == CV_32F)
		{
			if (idxPars_NMSLIB.empty() || queryPars_NMSLIB.empty())
			{
				nmslibMatching<float>(descriptors1,
					descriptors2,
					finalMatches,
					"sw-graph",
					"l2",
					"NN=3,initIndexAttempts=5",
					"initSearchAttempts=2,efSearch=2,efConstruction=2",
					ratioTest,
					8);
			}
			else
			{
				nmslibMatching<float>(descriptors1,
					descriptors2,
					finalMatches,
					"sw-graph",
					"l2",
					idxPars_NMSLIB,
					queryPars_NMSLIB,
					ratioTest,
					8);
			}
		}
		else if (descriptors1.type() == CV_8U)
		{
			if (idxPars_NMSLIB.empty() || queryPars_NMSLIB.empty())
			{
				nmslibMatching<int>(descriptors1,
					descriptors2,
					finalMatches,
					"sw-graph",
					"bit_hamming",
					"NN=3,initIndexAttempts=5",
					"initSearchAttempts=2,efSearch=2,efConstruction=2",
					ratioTest,
					8);
			}
			else
			{
				nmslibMatching<int>(descriptors1,
					descriptors2,
					finalMatches,
					"sw-graph",
					"bit_hamming",
					idxPars_NMSLIB,
					queryPars_NMSLIB,
					ratioTest,
					8);
			}
		}
	}
    else if(!matcher_name.compare("HIRCLUIDX") || !matcher_name.compare("HIRKMEANS") ||
            !matcher_name.compare("LINEAR") || !matcher_name.compare("LSHIDX") ||
            !matcher_name.compare("RANDKDTREE"))
    {
      int nn;

      if(ratioTest)
      {
        nn = 2;
      }
      else
      {
        nn = 1;
      }

      if(!matcher_name.compare("HIRCLUIDX") || !matcher_name.compare("LINEAR"))
      {
        if((descriptors1.type() != CV_32F) && (descriptors1.type() != CV_8U))
        {
          cout << "Format of descriptors not supported!" << endl;
          return -1;
        }
      }
      else if(!matcher_name.compare("HIRKMEANS") || !matcher_name.compare("RANDKDTREE"))
      {
        if(descriptors1.type() != CV_32F)
        {
          cout << "Format of descriptors not supported!" << endl;
          return -1;
        }
      }
      else
      {
        if(descriptors1.type() != CV_8U)
        {
          cout << "Format of descriptors not supported!" << endl;
          return -1;
        }
      }

      if(descriptors1.type() == CV_8U)
      {
        cvflann::Matrix<unsigned char> dataset(descriptors2.data, descriptors2.rows, descriptors2.cols);
        cvflann::Matrix<unsigned char> query(descriptors1.data, descriptors1.rows, descriptors1.cols);
        cvflann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
        cvflann::Matrix<int> dists(new int[query.rows*nn], query.rows, nn);

        if(!matcher_name.compare("HIRCLUIDX"))
        {
          // construct a hierarchical clustering index
          cvflann::Index<cvflann::HammingLUT> index(dataset, cvflann::HierarchicalClusteringIndexParams());
          index.buildIndex();

          // do a knn search, using 64 checks (higher values lead to higher precision, but slower performance) -> 32 can also be used
          index.knnSearch(query, indices, dists, nn, cvflann::SearchParams(64));
        }
        else if(!matcher_name.compare("LINEAR"))
        {
          // construct a linear index
          cvflann::Index<cvflann::HammingLUT> index(dataset, cvflann::LinearIndexParams());
          index.buildIndex();

          // do a knn search, using 64 checks (higher values lead to higher precision, but slower performance) -> 32 can also be used
          index.knnSearch(query, indices, dists, nn, cvflann::SearchParams(64));
        }
        else
        {
          // construct a LSH index
          cvflann::Index<cvflann::HammingLUT> index(dataset, cvflann::LshIndexParams());
          index.buildIndex();

          // do a knn search, using 64 checks (higher values lead to higher precision, but slower performance) -> 32 can also be used
          index.knnSearch(query, indices, dists, nn, cvflann::SearchParams(64));
        }

        //Ratio test
        if(ratioTest)
        {
          for(size_t q = 0; q < indices.rows; q++)
          {
            if(dists[q][0] < (0.75f * dists[q][1]))
            {
              cv::DMatch match;
              match.distance = (float)dists[q][0];
              match.queryIdx = q;
              match.trainIdx = indices[q][0];
              finalMatches.push_back(match);
            }
          }
        }
        else
        {
          for(size_t q = 0; q < indices.rows; q++)
          {
            cv::DMatch match;
            match.distance = (float)dists[q][0];
            match.queryIdx = q;
            match.trainIdx = indices[q][0];
            finalMatches.push_back(match);
          }
        }

        /*delete[] indices.ptr();
        delete[] dists.ptr();*/
        delete[] indices.data;
        delete[] dists.data;
      }
      else
      {
        cvflann::Matrix<float> dataset((float*)descriptors2.data, descriptors2.rows, descriptors2.cols);
        cvflann::Matrix<float> query((float*)descriptors1.data, descriptors1.rows, descriptors1.cols);
        cvflann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
        cvflann::Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

        if(!matcher_name.compare("HIRCLUIDX"))
        {
          // construct a hierarchical clustering index
          cvflann::Index<cvflann::L2<float>> index(dataset, cvflann::HierarchicalClusteringIndexParams());
          index.buildIndex();

          // do a knn search, using 64 checks (higher values lead to higher precision, but slower performance) -> 32 can also be used
          index.knnSearch(query, indices, dists, nn, cvflann::SearchParams(64));
        }
        else if(!matcher_name.compare("HIRKMEANS"))
        {
          // construct a hierarchical k-means tree
          cvflann::Index<cvflann::L2<float>> index(dataset, cvflann::KMeansIndexParams());
          index.buildIndex();

          // do a knn search, using 64 checks (higher values lead to higher precision, but slower performance) -> 32 can also be used
          index.knnSearch(query, indices, dists, nn, cvflann::SearchParams(64));
        }
        else if(!matcher_name.compare("LINEAR"))
        {
          // construct a linear index
          cvflann::Index<cvflann::L2<float>> index(dataset, cvflann::LinearIndexParams());
          index.buildIndex();

          // do a knn search, using 64 checks (higher values lead to higher precision, but slower performance) -> 32 can also be used
          index.knnSearch(query, indices, dists, nn, cvflann::SearchParams(64));
        }
        else
        {
          // construct a randomized KD-tree index
          cvflann::Index<cvflann::L2<float>> index(dataset, cvflann::KDTreeIndexParams(8));
          index.buildIndex();

          // do a knn search, using 64 checks (higher values lead to higher precision, but slower performance) -> 32 can also be used
          index.knnSearch(query, indices, dists, nn, cvflann::SearchParams(64));
        }

        //Ratio test
        if(ratioTest)
        {
          for(size_t q = 0; q < indices.rows; q++)
          {
            if(dists[q][0] < (0.75f * dists[q][1]))
            {
              cv::DMatch match;
              match.distance = dists[q][0];
              match.queryIdx = (int)q;
              match.trainIdx = indices[q][0];
              finalMatches.push_back(match);
            }
          }
        }
        else
        {
          for(size_t q = 0; q < indices.rows; q++)
          {
            cv::DMatch match;
            match.distance = dists[q][0];
            match.queryIdx = (int)q;
            match.trainIdx = indices[q][0];
            finalMatches.push_back(match);
          }
        }

        /*delete[] indices.ptr();
        delete[] dists.ptr();*/
        delete[] indices.data;
        delete[] dists.data;
      }

      if(finalMatches.size() < MIN_FINAL_MATCHES)
      {
        cout << "Too less remaining matches using the " << matcher_name << " matcher." << endl;
        return -3; //Too less matches left
      }
    }
    else
    {
      cout << "Matcher " << matcher_name << " is not supported." << endl;
      return -2;
    }

    //Filter final matches with VFC
    if(VFCrefine)
    {
      vector<cv::DMatch> vfcfilteredMatches;

      if(!filterWithVFC(keypoints1, keypoints2, finalMatches, vfcfilteredMatches))
      {
        if((vfcfilteredMatches.size() > 8) || (finalMatches.size() < 24))
        {
          finalMatches = vfcfilteredMatches;
        }
      }
    }

    return 0;
  }

  int cashashMatching(cv::Mat const& descrL, cv::Mat const& descrR, std::vector<cv::DMatch> & matches)
  {
    ImageData imgdata1, imgdata2;

    std::auto_ptr<std::vector<ImageData>> stImageDataList(new std::vector<ImageData>());
    std::auto_ptr<HashConvertor> stHashConvertor(new HashConvertor);
    std::auto_ptr<MatchPairLoader> stMatchPairLoader(new MatchPairLoader);
    std::auto_ptr<CasHashMatcher> stCasHashMatcher(new CasHashMatcher);

    if(kDimSiftData != descrL.cols)
    {
      std::cout << "Dimension of SIFT descriptor not compatible with CASHASH!" << endl;
      return -1;
    }

    //Get descriptors from left image
    imgdata1.cntPoint = descrL.rows;
    // allocate space for SIFT feature vector pointers
    imgdata1.siftDataPtrList = (SiftDataPtr*)malloc(sizeof(SiftDataPtr*) * descrL.rows);

    for (int dataIndex = 0; dataIndex < imgdata1.cntPoint; dataIndex++)
    {
      // allocate space for each SIFT feature vector
      imgdata1.siftDataPtrList[dataIndex] = (int*)malloc(sizeof(int) * kDimSiftData);

      for (int dimIndex = 0; dimIndex < kDimSiftData; dimIndex++)
      {
        *(imgdata1.siftDataPtrList[dataIndex] + dimIndex) = cv::saturate_cast<int>(descrL.at<float>(dataIndex, dimIndex));
      }
    }

    stImageDataList->push_back(imgdata1);

    //Get descriptors from right image
    imgdata2.cntPoint = descrR.rows;
    // allocate space for SIFT feature vector pointers
    imgdata2.siftDataPtrList = (SiftDataPtr*)malloc(sizeof(SiftDataPtr*) * descrR.rows);

    for (int dataIndex = 0; dataIndex < imgdata2.cntPoint; dataIndex++)
    {
      // allocate space for each SIFT feature vector
      imgdata2.siftDataPtrList[dataIndex] = (int*)malloc(sizeof(int) * kDimSiftData);

      for (int dimIndex = 0; dimIndex < kDimSiftData; dimIndex++)
      {
        *(imgdata2.siftDataPtrList[dataIndex] + dimIndex) = cv::saturate_cast<int>(descrR.at<float>(dataIndex, dimIndex));
      }
    }

    stImageDataList->push_back(imgdata2);

    stMatchPairLoader->matchPairList.clear();

    DataPreProcessor::PreProcess(*stImageDataList);// adjust each SIFT feature vector to zero-mean

    for (int imageIndex = 0; imageIndex < 2; imageIndex++)
    {
      stHashConvertor->SiftDataToHashData((*stImageDataList)[imageIndex]); // convert SIFT feature to Hash code and CompHash code
      BucketBuilder::Build((*stImageDataList)[imageIndex]); // construct multiple groups of buckets for each image
    }

    stMatchPairLoader->LoadMatchPairList(2);

    for (int imageIndex_1 = 0; imageIndex_1 < 2; imageIndex_1++)
    {
      // obtain the list of images to be matched with the current image
      std::vector<int> matchPairList = stMatchPairLoader->GetMatchPairList(imageIndex_1);

      for (std::vector<int>::const_iterator iter = matchPairList.begin(); iter != matchPairList.end(); iter++)
      {
        int imageIndex_2 = *iter;

        // perform image matching with CasHash Matching Technique
        MatchList matchList = stCasHashMatcher->MatchSpFast((*stImageDataList)[imageIndex_2], (*stImageDataList)[imageIndex_1]);

        // if the number of successfully matched SIFT points exceeds the required minimal threshold, then write the result to file
        for (MatchList::const_iterator iter = matchList.begin(); iter != matchList.end(); iter++)
        {
          cv::DMatch match_tmp;
          match_tmp.queryIdx = iter->first;
          match_tmp.trainIdx = iter->second;
          matches.push_back(match_tmp);
        }
      }
    }

    //Free allocated memory of bucketList
    for (int groupIndex = 0; groupIndex < kCntBucketGroup; groupIndex++)
    {
      for (int bucketID = 0; bucketID < kCntBucketPerGroup; bucketID++)
      {
        for(int k1 = 0; k1 < (int)stImageDataList->size(); k1++)
        {
          free((*stImageDataList)[k1].bucketList[groupIndex][bucketID]);
        }
      }
    }

    //Free allocated memory of bucketIDList
    for (int groupIndex = 0; groupIndex < kCntBucketGroup; groupIndex++)
    {
      for(int k1 = 0; k1 < (int)stImageDataList->size(); k1++)
      {
        free((*stImageDataList)[k1].bucketIDList[groupIndex]);
      }
    }

    //Free allocated memory of hashDataPtrList and compHashDataPtrList
    for(int k1 = 0; k1 < (int)stImageDataList->size(); k1++)
    {
      for(int k = 0; k < (*stImageDataList)[k1].cntPoint; k++)
      {
        free((*stImageDataList)[k1].hashDataPtrList[k]);
        free((*stImageDataList)[k1].compHashDataPtrList[k]);
      }

      free((*stImageDataList)[k1].hashDataPtrList);
      free((*stImageDataList)[k1].compHashDataPtrList);
    }

    if(matches.size() < MIN_FINAL_MATCHES)
    {
      for(int k = 0; k < imgdata1.cntPoint; k++)
      {
        free(imgdata1.siftDataPtrList[k]);
      }

      free(imgdata1.siftDataPtrList);

      for(int k = 0; k < imgdata2.cntPoint; k++)
      {
        free(imgdata2.siftDataPtrList[k]);
      }

      free(imgdata2.siftDataPtrList);

      return -2; //Too less matches left
    }

    for(int k = 0; k < imgdata1.cntPoint; k++)
    {
      free(imgdata1.siftDataPtrList[k]);
    }

    free(imgdata1.siftDataPtrList);

    for(int k = 0; k < imgdata2.cntPoint; k++)
    {
      free(imgdata2.siftDataPtrList[k]);
    }

    free(imgdata2.siftDataPtrList);

    return 0;
  }

  /* This function calculates the subpixel-position of matched keypoints by template matching. Be careful,
   * if there are large rotations, changes in scale or other feature deformations between the matches, this
   * method should not be called. The two keypoint sets keypoints1 and keypoints2 must be matched keypoints,
   * where matches have the same index.
   *
   * Mat img1                 Input  -> First image
   * Mat img2                 Input  -> Second image
   * vector<KeyPoint> *keypoints1       Input  -> Keypoints in the first (left) image
   *                            which match to keypoints2 (a match must have the same index
   *                            in keypoints1 and keypoints2)
   * vector<KeyPoint> *keypoints2       Input  -> Keypoints in the second (right) image
   *                            which match to keypoints1 (a match must have the same index
   *                            in keypoints1 and keypoints2)
   * vector<bool> inliers           Output -> Inlier mask. For matches with subpixel-locations
   *                            farer than 4 pixels to the original position are
   *                            marked as outliers.
   *
   * Return value:                 0:     Everything ok
   *                        -1:     Subpixel refinement failed for too many keypoints
   *                        -2:     Size of keypoint sets not equal
   */
  int getSubPixMatches(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> *keypoints1, std::vector<cv::KeyPoint> *keypoints2,
                       std::vector<bool> *inliers)
  {
    cv::Mat results;
    int featuresize1, featuresize2, diff1, diff2;
    unsigned int nrInliers = 0;
    /*float diffsum=0; //only for quality estimation
    Point2f keypcopy, posdiff_f; //only for quality estimation*/
    float featuresize_2, nx, ny, valPxy, valPxp, valPxn, valPyp, valPyn;
    cv::Point minLoc;
    cv::Point2f newKeyP;
    vector<bool> InliersMask( keypoints1->size(), false );

    if(keypoints1->size() != keypoints2->size())
    {
      cout << "For subpixel-refinement the number of left and right keypoints must be the same as they must match!" << endl;
      return -2;
    }

    //const int maxDepthSearch = 32;
    //const int maxLeafNum = 20;



    ////Prepare the coordinates of the keypoints for the KD-tree
    //EMatFloat2 eigkeypts1(keypoints1->size(),2), eigkeypts2(keypoints2->size(),2);
    //const size_t num_results = 6;
    //vector<size_t>   ret_indexes(num_results);
    //vector<float> out_dists_sqr(num_results);
    //nanoflann::KNNResultSet<float> resultSet(num_results);
    ////resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );

    //vector<Mat> homos;
    //Mat H, hp1, hp2, imgpart1_warped;

    //for(unsigned int i = 0;i<keypoints1->size();i++)
    //{
    //  eigkeypts1(i,0) = keypoints1->at(i).pt.x;
    //  eigkeypts1(i,1) = keypoints1->at(i).pt.y;
    //  eigkeypts2(i,0) = keypoints2->at(i).pt.x;
    //  eigkeypts2(i,1) = keypoints2->at(i).pt.y;
    //}

    //{
    //  float xrange[2] = {eigkeypts1.col(0).minCoeff(), eigkeypts1.col(0).maxCoeff()};
    //  float yrange[2] = {eigkeypts1.col(1).minCoeff(), eigkeypts1.col(1).maxCoeff()};
    //  float xrange_4 = (xrange[1] - xrange[0])/4.0;
    //  float yrange_4 = (yrange[1] - yrange[0])/4.0;
    //  if(((xrange_4 > 40.0) || (yrange_4 > 40.0)) && (eigkeypts1.rows() > 150.0))
    //  {
    //    const int maxRangeTrys = 10;
    //    static cv::RNG rng;
    //    for(
    //    int rand_number[2] = {rng.uniform((int)xrange[0],(int)xrange[1]),rng.uniform((int)yrange[0],(int)yrange[1])};
    //  }
    //}

    ////Generate the KD-tree index for the keypoint coordinates
    //KDTree_D2float keypts1idx(2,eigkeypts1,maxLeafNum);
    //keypts1idx.index->buildIndex();

    for(size_t i=0; i<keypoints1->size(); i++)
    {
      featuresize1 = (int)((keypoints1->at(i).size > keypoints2->at(i).size) ? keypoints1->at(i).size : keypoints2->at(i).size);
      featuresize1 = featuresize1 + 6;

      if(featuresize1 < 18)
      {
        featuresize1 = 18;
      }

      featuresize_2 = (float)featuresize1/2.0f;
      diff1 = (((float)cvRound(featuresize_2)-featuresize_2) != 0.0) ? (int)featuresize_2 : ((int)featuresize_2-1);

      if(((float)cvRound(featuresize_2)-featuresize_2) == 0.0)
      {
        featuresize1 -= 1;
      }

      featuresize2 = featuresize1+10;
      diff2 = diff1+5;

      cv::Rect rec2 = cv::Rect(cvRound(keypoints2->at(i).pt.x)-diff2, cvRound(keypoints2->at(i).pt.y)-diff2, featuresize2, featuresize2);
      cv::Rect rec1 = cv::Rect(cvRound(keypoints1->at(i).pt.x)-diff1, cvRound(keypoints1->at(i).pt.y)-diff1, featuresize1,featuresize1);



      //// do a knn search
      //resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
      //keypts1idx.index->findNeighbors(resultSet, &eigkeypts1(i,0), nanoflann::SearchParams(maxDepthSearch));
      ////Check if the keypoints lie on the same line
      //double dx1 = eigkeypts1(ret_indexes[1],0) - eigkeypts1(ret_indexes[0],0);
      //double dy1 = eigkeypts1(ret_indexes[1],1) - eigkeypts1(ret_indexes[0],1);
      //size_t noLine_idx = 2;
      //for(size_t k = 2; k < num_results; k++ )
      //{
      //  double dx2 = eigkeypts1(ret_indexes[k],0) - eigkeypts1(ret_indexes[0],0);
      //  double dy2 = eigkeypts1(ret_indexes[k],1) - eigkeypts1(ret_indexes[0],1);
      //  if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
      //    noLine_idx++;
      //  else if(k >= 2)
      //    break;
      //}
      //if(noLine_idx >= num_results)
      //  matchTemplate((img2)(rec2), (img1)(rec1), results, CV_TM_SQDIFF);
      //else
      //{
      //  dx1 = eigkeypts2(ret_indexes[1],0) - eigkeypts2(ret_indexes[0],0);
      //  dy1 = eigkeypts2(ret_indexes[1],1) - eigkeypts2(ret_indexes[0],1);
      //  for(size_t k = 2; k < num_results; k++ )
      //  {
      //    double dx2 = eigkeypts2(ret_indexes[k],0) - eigkeypts2(ret_indexes[0],0);
      //    double dy2 = eigkeypts2(ret_indexes[k],1) - eigkeypts2(ret_indexes[0],1);
      //    if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
      //      noLine_idx++;
      //    else if(k >= 2)
      //      break;
      //  }
      //  if(noLine_idx >= num_results)
      //    matchTemplate((img2)(rec2), (img1)(rec1), results, CV_TM_SQDIFF);
      //  else
      //  {
      //    hp1 = (Mat_<float>(3, 2) << eigkeypts1(ret_indexes[0],0) - rec1.x, eigkeypts1(ret_indexes[0],1) - rec1.y,
      //                  eigkeypts1(ret_indexes[1],0) - rec1.x, eigkeypts1(ret_indexes[1],1) - rec1.y,
      //                  eigkeypts1(ret_indexes[noLine_idx],0) - rec1.x, eigkeypts1(ret_indexes[noLine_idx],1) - rec1.y);
      //    hp2 = (Mat_<float>(3, 2) << eigkeypts2(ret_indexes[0],0) - rec1.x, eigkeypts2(ret_indexes[0],1) - rec1.y,
      //                  eigkeypts2(ret_indexes[1],0) - rec1.x, eigkeypts2(ret_indexes[1],1) - rec1.y,
      //                  eigkeypts2(ret_indexes[noLine_idx],0) - rec1.x, eigkeypts2(ret_indexes[noLine_idx],1) - rec1.y);
      //    H = getAffineTransform(hp1,hp2);
      //    /*H.at<double>(0,2) = 0.0;
      //    H.at<double>(1,2) = 0.0;*/
      //    warpAffine((img1)(rec1),imgpart1_warped,H,Size(120,120));//,WARP_INVERSE_MAP);//,INTER_LINEAR,BORDER_REPLICATE);
      //    homos.push_back(H);

      //    imshow("not_warped", (img1)(rec1) );
      //    imshow("warped", imgpart1_warped );
      //    waitKey(0);

      //    //matchTemplate((img2)(rec2), (img1)(rec1), results, CV_TM_SQDIFF);
      //    matchTemplate((img2)(rec2), imgpart1_warped, results, CV_TM_SQDIFF);
      //  }
      //}
      cv::matchTemplate((img2)(rec2), (img1)(rec1), results, CV_TM_SQDIFF);
      cv::minMaxLoc(results,(double *)0,(double *)0,&minLoc);

      newKeyP = cv::Point2f((float)(rec2.x+minLoc.x+(featuresize1-1)/2), (float)(rec2.y+minLoc.y+(featuresize1-1)/2));

      if( std::sqrt(std::pow(newKeyP.x-cvRound(keypoints2->at(i).pt.x),2)+std::pow(newKeyP.y-cvRound(keypoints2->at(i).pt.y),2)) <= 4)
      {
        InliersMask[(int)i] = true;

        valPxy = results.at<float>(minLoc.y,minLoc.x);
        valPxp = results.at<float>(minLoc.y,minLoc.x+1);
        valPxn = results.at<float>(minLoc.y,minLoc.x-1);
        valPyp = results.at<float>(minLoc.y+1,minLoc.x);
        valPyn = results.at<float>(minLoc.y-1,minLoc.x);

        //keypcopy = keypoints2->at(i).pt; //only for quality estimation

        nx = 2*(2*valPxy-valPxn-valPxp);
        ny = 2*(2*valPxy-valPyn-valPyp);

        if((nx != 0) && (ny != 0))
        {
          nx = (valPxp-valPxn)/nx;
          ny = (valPyp-valPyn)/ny;

          keypoints2->at(i).pt = cv::Point2f(newKeyP.x+nx,
                                             newKeyP.y+ny);
          nrInliers++;
        }

        /*// Below -> Only for quality estimation
        posdiff_f = Point2f(keypcopy.x-keypoints2->at(i).pt.x+keypoints1->at(i).pt.x-cvRound(keypoints1->at(i).pt.x),
                  keypcopy.y-keypoints2->at(i).pt.y+keypoints1->at(i).pt.y-cvRound(keypoints1->at(i).pt.y));
        cout << posdiff_f.x << ", " << posdiff_f.y << endl;
        diffsum += std::abs(posdiff_f.x)+std::abs(posdiff_f.y);
        // Above -> Only for quality estimation*/
      }
    }

    //cout << "Sum of differences from keypoint to matching position: " << diffsum << endl; //only for quality estimation

    if(inliers != NULL)
    {
      *inliers = InliersMask;
    }

    if((nrInliers < keypoints1->size() / 3) || (nrInliers < MIN_FINAL_MATCHES))
    {
      return -1;  //Subpixel refinement failed for too many keypoints
    }

    return 0;
  }

  /* This function compares the weights of the matches to be able to sort them accordingly
   * while keeping track of the index.
   *
   * KeyPoint first       Input  -> First pair of match and index
   * KeyPoint second        Input  -> Second pair of match and index
   */
  bool sortMatchWeightIdx(std::pair<double, unsigned int> &first, std::pair<double, unsigned int> &second)
  {
    return first.first < second.first;
  }

  bool IsMatcherSupported(const std::string &type)
  {
    std::vector<std::string> vecSupportedTypes = GetSupportedMatcher();

    if(std::find(vecSupportedTypes.begin(), vecSupportedTypes.end(), type) != vecSupportedTypes.end())
    {
      return true;
    }

    return false;
  }

  std::vector<std::string> GetSupportedMatcher()
  {
    static std::string types [] = {"CASHASH", "GMBSOF","HIRCLUIDX", "HIRKMEANS", "LINEAR", "LSHIDX", "RANDKDTREE", "LKOF", "ALKOF", "LKOFT", "ALKOFT", "SWGRAPH"};
    return std::vector<std::string>(types, types + 11);
  }

}
