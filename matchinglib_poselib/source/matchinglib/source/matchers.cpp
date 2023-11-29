//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2019 AIT Austrian Institute of Technology GmbH
//
//Permission is hereby granted, free of charge, to any person obtaining
//a copy of this software and associated documentation files (the "Software"),
//to deal in the Software without restriction, including without limitation
//the rights to use, copy, modify, merge, publish, distribute, sublicense,
//and/or sell copies of the Software, and to permit persons to whom the
//Software is furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included
//in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//Author: Josef Maier (josefjohann-dot-maier-at-gmail-dot-at)

#include "matchinglib_matchers.h"

#include "match_statOptFlow.h"
#include "matchinglib_imagefeatures.h"

#include "CascadeHash/Share.h"
#include "CascadeHash/DataPreProcessor.h"
#include "CascadeHash/HashConvertor.h"
#include "CascadeHash/BucketBuilder.h"
#include "CascadeHash/MatchPairLoader.h"
#include "CascadeHash/CasHashMatcher.h"

#include "nmslib/nmslib_matchers.h"

//#include "flann/flann.hpp"
#include "opencv2/flann.hpp"

#include <map>
#include <algorithm>
#include <cmath>

#include "vfcMatches.h"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "annoy/src/kissrandom.h"
#include "annoy/src/annoylib.h"

//using namespace cv;
using namespace std;

namespace matchinglib
{

#define NMSLIB_SEARCH_THREADS 8

  /* --------------------- Function prototypes --------------------- */

//This function compares the weights of the matches to be able to sort them accordingly while keeping track of the index.
  bool sortMatchWeightIdx(std::pair<double,unsigned int>& first, std::pair<double,unsigned int>& second);
//The Cascade Hashing matcher
  int cashashMatching(cv::Mat const& descrL, cv::Mat const& descrR, std::vector<cv::DMatch> & matches);
  int annoyMatching(cv::Mat const& descrL, cv::Mat const& descrR, std::vector<cv::DMatch> & matches, bool ratioTest = true, unsigned int n_trees = 0, unsigned int search_k = 0);

  /* --------------------- Functions --------------------- */

  int getMatches(std::vector<cv::KeyPoint> const &keypoints1, std::vector<cv::KeyPoint> const &keypoints2,
                 const cv::Mat &descriptors1, const cv::Mat &descriptors2, cv::Size imgSi, std::vector<cv::DMatch> &finalMatches,
                 const std::string &matcher_name, bool VFCrefine, bool ratioTest, const std::string &descriptor_name, std::string idxPars_NMSLIB, std::string queryPars_NMSLIB, const size_t nr_threads)
  {
    std::random_device rd;
    std::mt19937 g(rd());
    return getMatches(keypoints1, keypoints2, descriptors1, descriptors2, imgSi, finalMatches, g, matcher_name, VFCrefine, ratioTest, descriptor_name, idxPars_NMSLIB, queryPars_NMSLIB, nr_threads);
  }

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
   * string &descriptor_name	Input  -> Name of the used descriptor type which is used to adjust the paramters of the
   *						VPTREE matcher of the NMSLIB
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
  int getMatches(std::vector<cv::KeyPoint> const &keypoints1, std::vector<cv::KeyPoint> const &keypoints2,
                 const cv::Mat &descriptors1, const cv::Mat &descriptors2, cv::Size imgSi, std::vector<cv::DMatch> &finalMatches, std::mt19937 &mt,
                 const std::string &matcher_name, bool VFCrefine, bool ratioTest, const std::string &descriptor_name, std::string idxPars_NMSLIB, std::string queryPars_NMSLIB, const size_t nr_threads)
  {
    CV_Assert(descriptors1.type() == descriptors2.type());
    int err;
    const size_t nr_threads_ = nr_threads > 0 ? nr_threads : NMSLIB_SEARCH_THREADS;

    if ((keypoints1.size() < 15) || (keypoints2.size() < 15))
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

      err = AdvancedMatching(matcher, keypoints1, keypoints2, descriptors1, descriptors2, imgSi, finalMatches, mt, true, 0.3, 3.5, 2, false, nullptr, nullptr, nullptr, nullptr, cv::noArray(), cv::noArray(), nr_threads_); //, imgs[0], imgs[1]);

      if (err)
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
        string idxPars = (idxPars_NMSLIB.empty() || queryPars_NMSLIB.empty()) ? "NN=10,efConstruction=10,indexThreadQty=8" : idxPars_NMSLIB;
        string queryPars = (idxPars_NMSLIB.empty() || queryPars_NMSLIB.empty()) ? "efSearch=10" : queryPars_NMSLIB;
        if (descriptors1.type() == CV_32F)
        {
          nmslibMatching<float>(descriptors1,
                                descriptors2,
                                finalMatches,
                                "sw-graph",
                                "l2",
                                idxPars,
                                queryPars,
                                mt,
                                ratioTest,
                                nr_threads_);
        }
        else if (descriptors1.type() == CV_8U)
        {
          nmslibMatching<int>(descriptors1,
                              descriptors2,
                              finalMatches,
                              "sw-graph",
                              "bit_hamming",
                              idxPars,
                              queryPars,
                              mt,
                              ratioTest,
                              nr_threads_);
        }
        else if (descriptors1.type() == CV_64F)
        {
          cv::Mat desc1_float;
          cv::Mat desc2_float;
          descriptors1.convertTo(desc1_float, CV_32F);
          descriptors2.convertTo(desc2_float, CV_32F);
          nmslibMatching<float>(desc1_float,
                                desc2_float,
                                finalMatches,
                                "sw-graph",
                                "l2",
                                idxPars,
                                queryPars,
                                mt,
                                ratioTest,
                                nr_threads_);
        }
        else
        {
            cout << "Wrong descriptor data type for SWGRAPH! Must be 32bit float, 64bit double or 8bit unsigned char." << endl;
            return -1;
        }

    }
    else if (!matcher_name.compare("HNSW"))
    {
        string idxPars = (idxPars_NMSLIB.empty() || queryPars_NMSLIB.empty()) ? "M=50,efConstruction=10,delaunay_type=1,indexThreadQty=8" : idxPars_NMSLIB;
        string queryPars = (idxPars_NMSLIB.empty() || queryPars_NMSLIB.empty()) ? "efSearch=10" : queryPars_NMSLIB;
        if (descriptors1.type() == CV_32F)
        {
          nmslibMatching<float>(descriptors1,
                                descriptors2,
                                finalMatches,
                                "hnsw",
                                "l2",
                                idxPars,
                                queryPars,
                                mt,
                                ratioTest,
                                nr_threads_);
        }
        else if (descriptors1.type() == CV_8U)
        {
          nmslibMatching<int>(descriptors1,
                              descriptors2,
                              finalMatches,
                              "hnsw",
                              "bit_hamming",
                              idxPars,
                              queryPars,
                              mt,
                              ratioTest,
                              nr_threads_);
        }
        else if (descriptors1.type() == CV_64F)
        {
          cv::Mat desc1_float;
          cv::Mat desc2_float;
          descriptors1.convertTo(desc1_float, CV_32F);
          descriptors2.convertTo(desc2_float, CV_32F);
          nmslibMatching<float>(desc1_float,
                                desc2_float,
                                finalMatches,
                                "hnsw",
                                "l2",
                                idxPars,
                                queryPars,
                                mt,
                                ratioTest,
                                nr_threads_);
        }
        else
        {
            cout << "Wrong descriptor data type for HNSW! Must be 32bit float, 64bit double or 8bit unsigned char." << endl;
            return -1;
        }
    }
    else if (!matcher_name.compare("VPTREE"))
    {
        string idxPars = "";
        string queryPars = "";
        if (idxPars_NMSLIB.empty() || queryPars_NMSLIB.empty())
        {
            idxPars = "chunkBucket=1,bucketSize=20";
            if (!descriptor_name.compare("SIFT"))
                queryPars = "alphaLeft=3.66802,alphaRight=3.01833,expLeft=1,expRight=1"; //for a recall of 0.98
            else if (!descriptor_name.compare("SURF"))
                queryPars = "alphaLeft=2.22949,alphaRight=1.78183,expLeft=1,expRight=1"; //for a recall of 0.98
            else if (!descriptor_name.compare("KAZE"))
                queryPars = "alphaLeft=2.9028,alphaRight=1.94528,expLeft=1,expRight=1"; //for a recall of 0.98
            else if (!descriptor_name.compare("DAISY"))
                queryPars = "alphaLeft=2.82843,alphaRight=2.18102,expLeft=1,expRight=1"; //for a recall of 0.98
            else if (!descriptor_name.compare("LBGM"))
                queryPars = "alphaLeft=2.92572,alphaRight=2.24031,expLeft=1,expRight=1"; //for a recall of 0.98
            else if (!descriptor_name.compare("VGG_120"))
                queryPars = "alphaLeft=3.45397,alphaRight=2.91922,expLeft=1,expRight=1"; //for a recall of 0.98
            else if (!descriptor_name.compare("VGG_80"))
                queryPars = "alphaLeft=3.01833,alphaRight=2.56574,expLeft=1,expRight=1"; //for a recall of 0.98
            else if (!descriptor_name.compare("VGG_64"))
                queryPars = "alphaLeft=2.77269,alphaRight=2.33744,expLeft=1,expRight=1"; //for a recall of 0.98
            else if (!descriptor_name.compare("VGG_48"))
                queryPars = "alphaLeft=2.57083,alphaRight=2.12745,expLeft=1,expRight=1"; //for a recall of 0.98
            else
            {
                queryPars = "alphaLeft=1,alphaRight=1,expLeft=1,expRight=1"; //Default parameters
                cout << "No tuned parameters are available for VPTREE matcher and the given descriptor! Using default values!" << endl;
            }
        }
        else
        {
            idxPars = idxPars_NMSLIB;
            queryPars = queryPars_NMSLIB;
        }
        if (descriptors1.type() == CV_32F)
        {
          nmslibMatching<float>(descriptors1,
                                descriptors2,
                                finalMatches,
                                "vptree",
                                "l2",
                                idxPars,
                                queryPars,
                                mt,
                                ratioTest,
                                nr_threads_);
        }
        else if (descriptors1.type() == CV_64F)
        {
          cv::Mat desc1_float;
          cv::Mat desc2_float;
          descriptors1.convertTo(desc1_float, CV_32F);
          descriptors2.convertTo(desc2_float, CV_32F);
          nmslibMatching<float>(desc1_float,
                                desc2_float,
                                finalMatches,
                                "vptree",
                                "l2",
                                idxPars,
                                queryPars,
                                mt,
                                ratioTest,
                                nr_threads_);
        }
        else
        {
            cout << "Wrong descriptor data type for VPTREE! Must be 32bit float or 64bit double." << endl;
            return -1;
        }
    }
    else if (!matcher_name.compare("BRUTEFORCENMS"))
    {
        if (descriptors1.type() == CV_32F)
        {
          nmslibMatching<float>(descriptors1,
                                descriptors2,
                                finalMatches,
                                "seq_search",
                                "l2",
                                "",
                                "",
                                mt,
                                ratioTest,
                                nr_threads_);
        }
        else if (descriptors1.type() == CV_8U)
        {
          nmslibMatching<int>(descriptors1,
                              descriptors2,
                              finalMatches,
                              "seq_search",
                              "bit_hamming",
                              "",
                              "",
                              mt,
                              ratioTest,
                              nr_threads_);
        }
        else if (descriptors1.type() == CV_64F)
        {
          cv::Mat desc1_float;
          cv::Mat desc2_float;
          descriptors1.convertTo(desc1_float, CV_32F);
          descriptors2.convertTo(desc2_float, CV_32F);
          nmslibMatching<float>(desc1_float,
                                desc2_float,
                                finalMatches,
                                "seq_search",
                                "l2",
                                "",
                                "",
                                mt,
                                ratioTest,
                                nr_threads_);
        }
        else
        {
            cout << "Wrong descriptor data type for BRUTEFORCENMS! Must be 32bit float, 64bit double or 8bit unsigned char." << endl;
            return -1;
        }
    }
    else if (!matcher_name.compare("ANNOY"))//exact search method
    {
        if (annoyMatching(descriptors1, descriptors2, finalMatches, ratioTest) != 0)
            return -1;
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

      if(!filterWithVFC(keypoints1, keypoints2, finalMatches, vfcfilteredMatches, mt))
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
        for (MatchList::const_iterator iter1 = matchList.begin(); iter1 != matchList.end(); iter1++)
        {
          cv::DMatch match_tmp;
          match_tmp.queryIdx = iter1->first;
          match_tmp.trainIdx = iter1->second;
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

  /* Wrapper function for the Annoy matching library
  *
  * Mat descrL              Input  -> Descriptors within the first or left image
  * Mat descrR              Input  -> Descriptors within the second or right image
  * vector<DMatch> matches  Output -> Matches
  * bool ratioTest          Input  -> If true [Default=true], a ratio test is performed on the results.
  * unsigned int n_trees    Input  -> If provided, the build process of the index is influenced in the following way:
  *									  n_trees is provided during build time and affects the build time and the index
  *									  size. A larger value will give more accurate results, but larger indexes.
  * unsigned int search_k   Input  -> If provided, the search process influenced in the following way:
  *									  search_k is provided in runtime and affects the search performance. A larger
  *									  value will give more accurate results, but will take longer time to return.
  *
  * Return value:           0:     Everything ok
  *                  -1:     Discriptor type not supported
  */
  int annoyMatching(cv::Mat const& descrL, cv::Mat const& descrR, std::vector<cv::DMatch> & matches, bool ratioTest, unsigned int n_trees, unsigned int search_k)
  {
      unsigned int descrDim = (unsigned int)descrR.cols;
      unsigned int nrEntriesR = (unsigned int)descrR.rows;
      unsigned int nrEntriesL = (unsigned int)descrL.rows;
      // using namespace annoy;
      if (descrR.type() == CV_32F)
      {
          //AnnoyIndex<index format, descriptor type,> index(dimension of descriptor)
          AnnoyIndex<unsigned int, float, Euclidean, Kiss32Random, AnnoyIndexMultiThreadedBuildPolicy> index(descrR.cols);

          for (unsigned int i = 0; i < nrEntriesR; i++)
          {
              index.add_item(i, (float*)descrR.data + i * descrDim);
          }
          //index.verbose(true);

          if (n_trees)
              index.build(n_trees);
          else
              index.build(10);

          for (unsigned int i = 0; i < nrEntriesL; i++)
          {
              vector<unsigned int> idxs;
              vector<float> distances;
              cv::DMatch match;

              if(search_k)
                  index.get_nns_by_vector((float*)descrL.data + i * descrDim, 2, search_k, &idxs, &distances);
              else
                  index.get_nns_by_vector((float*)descrL.data + i * descrDim, 2, (size_t)-1, &idxs, &distances);

              if (ratioTest && (idxs.size() > 1))
              {
                if (distances[0] < (0.75f * distances[1]))
                {
                    match.distance = distances[0];
                    match.queryIdx = (int)i;
                    match.trainIdx = (int)idxs[0];
                    matches.push_back(match);
                }
              }
              else if(!idxs.empty())
              {
                  match.distance = distances[0];
                  match.queryIdx = (int)i;
                  match.trainIdx = (int)idxs[0];
                  matches.push_back(match);
              }
          }
      }
      else if (descrR.type() == CV_64F)
      {
          //AnnoyIndex<index format, descriptor type,> index(dimension of descriptor)
          AnnoyIndex<unsigned int, double, Euclidean, Kiss32Random, AnnoyIndexMultiThreadedBuildPolicy> index(descrR.cols);

          for (unsigned int i = 0; i < nrEntriesR; i++)
          {
              index.add_item(i, (double*)descrR.data + i * descrDim);
          }
          //index.verbose(true);

          if (n_trees)
              index.build(n_trees);
          else
              index.build(10);

          for (unsigned int i = 0; i < nrEntriesL; i++)
          {
              vector<unsigned int> idxs;
              vector<double> distances;
              cv::DMatch match;

              if (search_k)
                  index.get_nns_by_vector((double*)descrL.data + i * descrDim, 2, search_k, &idxs, &distances);
              else
                  index.get_nns_by_vector((double*)descrL.data + i * descrDim, 2, (size_t)-1, &idxs, &distances);

              if (ratioTest && (idxs.size() > 1))
              {
                  if (distances[0] < (0.75f * distances[1]))
                  {
                      match.distance = distances[0];
                      match.queryIdx = (int)i;
                      match.trainIdx = (int)idxs[0];
                      matches.push_back(match);
                  }
              }
              else if (!idxs.empty())
              {
                  match.distance = distances[0];
                  match.queryIdx = (int)i;
                  match.trainIdx = (int)idxs[0];
                  matches.push_back(match);
              }
          }
      }
      // else if (descrR.type() == CV_8UC1)
      // {
      //     //AnnoyIndex<index format, descriptor type,> index(dimension of descriptor)
      //     AnnoyIndex<unsigned int, unsigned char, annoy::Hamming, Kiss32Random, AnnoyIndexMultiThreadedBuildPolicy> index(descrR.cols);

      //     for (unsigned int i = 0; i < nrEntriesR; i++)
      //     {
      //         index.add_item(i, (unsigned char*)descrR.data + i * descrDim);
      //     }
      //     //index.verbose(true);

      //     if (n_trees)
      //         index.build(n_trees);
      //     else
      //         index.build(10);

      //     for (unsigned int i = 0; i < nrEntriesL; i++)
      //     {
      //         vector<unsigned int> idxs;
      //         vector<unsigned char> distances;
      //         cv::DMatch match;

      //         if (search_k)
      //             index.get_nns_by_vector((unsigned char*)descrL.data + i * descrDim, 2, search_k, &idxs, &distances);
      //         else
      //             index.get_nns_by_vector((unsigned char*)descrL.data + i * descrDim, 2, (size_t)-1, &idxs, &distances);

      //         if (ratioTest && (idxs.size() > 1))
      //         {
      //             if (distances[0] < (0.75f * static_cast<float>(distances[1])))
      //             {
      //                 match.distance = static_cast<float>(distances[0]);
      //                 match.queryIdx = (int)i;
      //                 match.trainIdx = (int)idxs[0];
      //                 matches.push_back(match);
      //             }
      //         }
      //         else if (!idxs.empty())
      //         {
      //             match.distance = static_cast<float>(distances[0]);
      //             match.queryIdx = (int)i;
      //             match.trainIdx = (int)idxs[0];
      //             matches.push_back(match);
      //         }
      //     }
      // }
      else
      {
          cout << "Wrong descriptor data type for ANNOY! Must be 32bit float or 64bit double." << endl;
          return -1;
      }


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
      //  matchTemplate((img2)(rec2), (img1)(rec1), results, cv::TM_SQDIFF);
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
      //    matchTemplate((img2)(rec2), (img1)(rec1), results, cv::TM_SQDIFF);
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

      //    //matchTemplate((img2)(rec2), (img1)(rec1), results, cv::TM_SQDIFF);
      //    matchTemplate((img2)(rec2), imgpart1_warped, results, cv::TM_SQDIFF);
      //  }
      //}

      Mat img_border[2];
      const int border_size = 100;
      copyMakeBorder(img1, img_border[0], border_size, border_size, border_size, border_size, BORDER_CONSTANT, Scalar(0, 0, 0));
      copyMakeBorder(img2, img_border[1], border_size, border_size, border_size, border_size, BORDER_CONSTANT, Scalar(0, 0, 0));
      if (rec2.x + rec2.width >= img2.cols || rec2.y + rec2.height >= img2.rows || rec2.x < 0 || rec2.y < 0 ||
          rec1.x + rec1.width >= img1.cols || rec1.y + rec1.height >= img1.rows || rec1.x < 0 || rec1.y < 0)
      {
            cv::Rect rec2_tmp = rec2, rec1_tmp = rec1;
            rec2_tmp.x += border_size;
            rec2_tmp.y += border_size;
            rec1_tmp.x += border_size;
            rec1_tmp.y += border_size;
            cv::matchTemplate((img_border[1])(rec2_tmp), (img_border[0])(rec1_tmp), results, cv::TM_SQDIFF);
      }
      else
      {
          cv::matchTemplate((img2)(rec2), (img1)(rec1), results, cv::TM_SQDIFF);
      }
      cv::minMaxLoc(results,(double *)0,(double *)0,&minLoc);

      newKeyP = cv::Point2f((float)(rec2.x+minLoc.x+(featuresize1-1)/2), (float)(rec2.y+minLoc.y+(featuresize1-1)/2));

      if((std::pow(newKeyP.x-cvRound(keypoints2->at(i).pt.x),2)+std::pow(newKeyP.y-cvRound(keypoints2->at(i).pt.y),2)) <= 16)
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

  /* This function calculates the subpixel-position of each keypoint in both images seperately using the OpenCV function
  * cv::cornerSubPix(). This function is also suitable for large rotations and scale changes between images.
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
  int getSubPixMatches_seperate_Imgs(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> *keypoints1, std::vector<cv::KeyPoint> *keypoints2,
      std::vector<bool> *inliers)
  {
      unsigned int nrInliers = 0;
      vector<bool> InliersMask(keypoints1->size(), true);
      float minKeySize[2] = { FLT_MAX, FLT_MAX };

      if (keypoints1->size() != keypoints2->size())
      {
          cout << "For subpixel-refinement the number of left and right keypoints must be the same as they must match!" << endl;
          return -2;
      }

      size_t n = keypoints1->size();
      vector<cv::Point2f> p1s(keypoints1->size()), p2s(keypoints2->size());
      vector<cv::Point2f> p1s_old, p2s_old;

      for (size_t i = 0; i < n; i++)
      {
          p1s[i] = keypoints1->at(i).pt;
          p2s[i] = keypoints2->at(i).pt;
          if ((keypoints1->at(i).size > 0) && (keypoints1->at(i).size < minKeySize[0]))
              minKeySize[0] = keypoints1->at(i).size;
          if ((keypoints2->at(i).size > 0) && (keypoints2->at(i).size < minKeySize[1]))
              minKeySize[1] = keypoints2->at(i).size;
      }
      p1s_old = p1s;
      p2s_old = p2s;
      minKeySize[0] = minKeySize[0] > 20.f ? 20.f : minKeySize[0];
      minKeySize[0] = minKeySize[0] < 6.f ? 6.f : minKeySize[0];
      minKeySize[0] /= 2.f;
      minKeySize[0] = std::ceil(minKeySize[0]);
      minKeySize[1] = minKeySize[1] > 20.f ? 20.f : minKeySize[1];
      minKeySize[1] = minKeySize[1] < 6.f ? 6.f : minKeySize[1];
      minKeySize[1] /= 2.f;
      minKeySize[1] = std::ceil(minKeySize[1]);

      TermCriteria criteria = TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::COUNT, 40, 0.001);

      cv::cornerSubPix(img1, p1s, cv::Size((int)minKeySize[0], (int)minKeySize[0]), cv::Size(-1, -1), criteria);
      cv::cornerSubPix(img2, p2s, cv::Size((int)minKeySize[1], (int)minKeySize[1]), cv::Size(-1, -1), criteria);

      cv::Point2f diff[2];
      float diffd[2];
      for (size_t i = 0; i < n; i++)
      {
          diff[0] = p1s_old[i] - p1s[i];
          diffd[0] = diff[0].x * diff[0].x + diff[0].y * diff[0].y;
          diff[1] = p2s_old[i] - p2s[i];
          diffd[1] = diff[1].x * diff[1].x + diff[1].y * diff[1].y;
          if (diffd[0] > 12 || diffd[1] > 12)
          {
              InliersMask[i] = false;
          }
          else
          {
              keypoints1->at(i).pt = p1s[i];
              keypoints2->at(i).pt = p2s[i];
              nrInliers++;
          }
      }

      if (inliers != NULL)
      {
          *inliers = InliersMask;
      }

      if ((nrInliers < n / 3) || (nrInliers < MIN_FINAL_MATCHES))
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
    static std::string types [] = {"CASHASH", "GMBSOF","HIRCLUIDX", "HIRKMEANS", "LINEAR", "LSHIDX", "RANDKDTREE", "LKOF", "ALKOF", "LKOFT", "ALKOFT", "SWGRAPH", "HNSW", "VPTREE", "BRUTEFORCENMS", "ANNOY"};
    return std::vector<std::string>(types, std::end(types));
  }

}
