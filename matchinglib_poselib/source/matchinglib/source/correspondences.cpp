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
/**********************************************************************************************************
 FILE: correspondences.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.9

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: April 2016

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for generating matched feature sets out of image
              information.
**********************************************************************************************************/

#include "matchinglib_correspondences.h"
#include "matchinglib_imagefeatures.h"
#include "matchinglib_matchers.h"
#include "match_statOptFlow.h"
#include "gms.h"

//#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
//#include <opencv2/nonfree/features2d.hpp>

#define USE_FSTREAM 0
#if USE_FSTREAM
#include <fstream>
#include <bitset>
#include <type_traits>
#endif

using namespace cv;
using namespace std;

namespace matchinglib
{

  /* --------------------- Function prototypes --------------------- */
#if USE_FSTREAM
  // SFINAE for safety. Sue me for putting it in a macro for brevity on the function
#define IS_INTEGRAL(T) typename std::enable_if< std::is_integral<T>::value >::type* = 0
	void writeDescriptorsToDisk(cv::Mat descriptors, std::string file);
	template<class T>
	std::string integral_to_binary_string(T byte, IS_INTEGRAL(T))
	{
		std::bitset<sizeof(T) * CHAR_BIT> bs(byte);
		return bs.to_string();
	}
#endif

  /* --------------------- Functions --------------------- */

  /* Generation of features followed by matching, filtering, and subpixel-refinement.
   *
   * Mat img1           Input  -> First or left input image
   * Mat img2           Input  -> Second or right input image
   * vector<DMatch> finalMatches  Output -> Matched features
   * vector<KeyPoint> kp1     Output -> All detected keypoints in the first or left image
   * vector<KeyPoint> kp1     Output -> All detected keypoints in the second or right image
   * string featuretype     Input  -> Name of feature detector. Possible detectors [Default=FAST] are:
   *                      FAST,11 STAR, SIFT, SURF, ORB, BRISK, MSER, GFTT, HARRIS, Dense, SimpleBlob
   *                      -> see the OpenCV documentation for further details on the different methods
   * string extractortype     Input  -> Methode for extracting the descriptors. The following inputs [Default=FREAK]
   *                      are possible:
   *                      FREAK, SIFT, SURF, ORB, BRISK, BriefDescriptorExtractor
   *                      -> see the OpenCV documentation for further details on the different methods
   * string matchertype     Input  -> The abbreviation of the used matcher [Default = GMBSOF].
   *                      Possible inputs are:
   *                      CASHASH: Cascade Hashing matcher
   *                      GMBSOF: Guided matching based on statistical optical flow
   *                      HIRCLUIDX: Hirarchical Clustering Index Matching from the FLANN library
   *                      HIRKMEANS: hierarchical k-means tree matcher from the FLANN library
   *                      LINEAR: linear Matching algorithm (Brute force) from the FLANN library
   *                      LSHIDX: LSH Index Matching algorithm from the FLANN library
   *                      RANDKDTREE: randomized KD-trees matcher from the FLANN library
   * bool dynamicKeypDet      Input  -> If true [Default], the number of features is limited to a specific nr. of
   *                      features using dynamic versions of the feature detectors. Only GFTT, SURF,
   *                      FAST and STAR are supported using this option.
   * int limitNrfeatures      Input  -> Maximum number of features that should remain after filtering or dynamic
   *                      feature detection [Default=8000].
   * bool VFCrefine       Input  -> If true [Default=false], the matches are refined using the vector field
   *                      consensus (VFC) algorithm. It is not recommended for non-rigid scenes or
   *                      scenes with volatile depth changes (The optical flow should smmoothly change
   *                      over the whole image). Otherwise, matches at such objects or "borders" are
   *                      rejected. The filter performance works quite well for all other kind of
   *                      scenes and inlier ratios above 20%.
   * bool ratioTest       Input  -> If true [Default=true], a ratio test is performed after matching. If
   *                      VFCrefine=true, the ratio test is performed in advance to VFC. For
   *                      GMBSOF, a ratio test is always performed independent of the value of
   *                      ratioTest. If you wand to change this behaviour for GMBSOF, this has to
   *                      be changed directly at the call of function AdvancedMatching. Moreover,
   *                      no ratio test can be performed for the CASHASH matcher.
   * bool SOFrefine       Input  -> If true [Default=false], the matches are filtered using the statistical
   *                      optical flow (SOF). This only maks sence, if the GMBSOF algorithm is not
   *                      used, as GMBSOF depends on SOF. This filtering strategy is computational
   *                      expensive for a large amount of matches.
   * bool subPixRefine      Input  -> If larger 0 [Default = 0], the subpixel-positions of matched keypoints are
   *                      calculated by either template matching (subPixRefine = 1) or corner refinement using cv::cornerSubPix (subPixRefine > 1). Be careful, if there are large rotations,
   *                      changes in scale or other feature deformations between the matches, the template matching
   *                      method should not be called.
   * int verbose          Input  -> Set the verbose level [Default = 0]:
   *                      0:  no information
   *                      1:  Display matching time
   *                      2:  Display feature detection times and matching time
   *                      3:  Display number of features and matches in addition to all temporal values
   * string idxPars_NMSLIB		Input  -> Index parameters for matchers of the NMSLIB. See manual of NMSLIB for details.
   *					  [Default = ""]
   * string queryPars_NMSLIB	Input  -> Query-time parameters for matchers of the NMSLIB. See manual of NMSLIB for details.
   *					  [Default = ""]
   *
   * Return value:         0:     Everything ok
   *                -1:     Wrong, corrupt or missing imput data
   *                -2:     Error while calculating keypoints
   *                -3:     Error while extracting descriptors
   *                -4:     Incompatible feature detector and matcher
   *                -5:     Matching failed
   *				-6:		Incompatible feature detector and descriptor
   */
  int getCorrespondences(Mat &img1,
                         Mat &img2,
                         std::vector<cv::DMatch> & finalMatches,
                         std::vector<cv::KeyPoint> & kp1,
                         std::vector<cv::KeyPoint> & kp2,
                         std::string featuretype,
                         std::string extractortype,
                         std::string matchertype,
                         bool dynamicKeypDet,
                         int limitNrfeatures,
                         bool VFCrefine,
						 bool GMSrefine,
                         bool ratioTest,
                         bool SOFrefine,
                         int subPixRefine,
                         int verbose,
						 std::string idxPars_NMSLIB,
						 std::string queryPars_NMSLIB)
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
      return -4;
    }

	if (featuretype.compare("KAZE") && !extractortype.compare("KAZE"))
	{
		cout << "KAZE descriptors are only compatible with KAZE keypoints!" << endl;
		return -6;
	}

	if (featuretype.compare("AKAZE") && !extractortype.compare("AKAZE"))
	{
		cout << "AKAZE descriptors are only compatible with AKAZE keypoints!" << endl;
		return -6;
	}

	if (!featuretype.compare("SIFT") && !extractortype.compare("ORB"))
	{
		cout << "ORB descriptors are not compatible with SIFT keypoints!" << endl;
		return -6;
	}

	if (!featuretype.compare("MSD") && !extractortype.compare("SIFT"))
	{
		cout << "SIFT descriptors are not compatible with MSD keypoints!" << endl;
		return -6;
	}

    vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    cv::Size imgSi = img1.size();
    int err = 0;
    double t_mea = 0.0, t_oa = 0.0;

    if(verbose > 1)
    {
      t_mea = (double)getTickCount(); //Start time measurement
    }

    if(getKeypoints(img1, keypoints1, featuretype, dynamicKeypDet, limitNrfeatures) != 0)
    {
      return -2;  //Error while calculating keypoints
    }

    if(getKeypoints(img2, keypoints2, featuretype, dynamicKeypDet, limitNrfeatures) != 0)
    {
      return -2;  //Error while calculating keypoints
    }

    if(verbose > 1)
    {
      t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
      cout << "Time for keypoint detection (2 imgs): " << t_mea << "ms" << endl;
      t_oa = t_mea;
      t_mea = (double)getTickCount(); //Start time measurement
    }

    bool onlyKeypoints = false;

    if(matchertype == "LKOF" || matchertype == "LKOFT")
    {
      onlyKeypoints = true;

      if(matchertype == "LKOF")
      {
        matchinglib::getMatches_OpticalFlow(keypoints1, keypoints2,
                                            img1, img2,
                                            finalMatches,
                                            false, false,
                                            cv::Size(11, 11), 5.0f);

      }

      if(matchertype == "LKOFT")
      {
        matchinglib::getMatches_OpticalFlowTracker(keypoints1, keypoints2, cv::Mat(),
            img1, img2, finalMatches,
            "LKOFT", "ORB",
            false, false,
            cv::Size(11, 11));

        keypoints2 = keypoints1;  // SAME ASSIGNEMNENT!
      }


      if(verbose > 0)
      {
        if(verbose > 1)
        {
          t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
          cout << "getMatches_OpticalFlow: " << t_mea << "ms" << endl;
          t_oa += t_mea;
          cout << "getMatches_OpticalFlow + keypoint detection: " << t_oa << "ms" << endl;

          if(verbose > 2)
          {
            cout << "# of features (1st img): " << keypoints1.size() << endl;
            cout << "# of features (2nd img): " << keypoints2.size() << endl;
          }
        }

        t_mea = (double)getTickCount(); //Start time measurement
      }
    }
    else
    {

      if(getDescriptors(img1, keypoints1, extractortype, descriptors1, featuretype) != 0)
      {
        return -3;  //Error while extracting descriptors
      }

      if(getDescriptors(img2, keypoints2, extractortype, descriptors2, featuretype) != 0)
      {
        return -3;  //Error while extracting descriptors
      }

#if USE_FSTREAM
	  //Function to write descriptor entries to disk -> for external tuning of the paramters of matcher VPTREE
	  writeDescriptorsToDisk(descriptors1, "C:\\work\\THIRDPARTY\\NMSLIB-1.6\\trunk\\sample_data\\descriptors_" + extractortype + ".txt");
	  return -1;
#endif

      if(verbose > 0)
      {
        if(verbose > 1)
        {
          t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
          cout << "Time for descriptor extraction (2 imgs): " << t_mea << "ms" << endl;
          t_oa += t_mea;
          cout << "Time for feature detection (2 imgs): " << t_oa << "ms" << endl;

          if(verbose > 2)
          {
            cout << "# of features (1st img): " << keypoints1.size() << endl;
            cout << "# of features (2nd img): " << keypoints2.size() << endl;
          }
        }

        t_mea = (double)getTickCount(); //Start time measurement
      }
    }

    if(matchertype == "ALKOF")
    {
      matchinglib::getMatches_OpticalFlowAdvanced(keypoints1, keypoints2,
          descriptors1, descriptors2,
          img1, img2,
          finalMatches,
          "ALKOF",
          false, false,
          cv::Size(11, 11), 5.0f, 3);
    }
    else if(matchertype == "ALKOFT")
    {
      matchinglib::getMatches_OpticalFlowTracker(keypoints1, keypoints2, descriptors1,
          img1, img2,
          finalMatches,
          "ALKOFT",
          "ORB",
          false, false,
          cv::Size(11, 11));

    }
    else if(!onlyKeypoints)
    {
      err = getMatches(keypoints1, keypoints2, descriptors1, descriptors2, imgSi, finalMatches, matchertype, VFCrefine, ratioTest, extractortype, idxPars_NMSLIB, queryPars_NMSLIB);
    }

    if(err != 0)
    {
      return (-4 + err);  //Matching failed
    }

    if(verbose > 0)
    {
      t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
      cout << "Time for matching: " << t_mea << "ms" << endl;

      if(verbose > 1)
      {
        t_oa += t_mea;
        cout << "Time for feature detection (2 imgs) and matching: " << t_oa << "ms" << endl;

        if(verbose > 2)
        {
          cout << "# of matches: " << finalMatches.size() << endl;
        }
      }
    }

	if (GMSrefine)
	{
		if (verbose > 1)
		{
			t_mea = (double)getTickCount(); //Start time measurement
		}

		std::vector<bool> inlierMask;
		int n_gms = filterMatchesGMS(keypoints1, imgSi,	keypoints2, imgSi, finalMatches, inlierMask, false, false);

		if (n_gms >= MIN_FINAL_MATCHES)
		{
			std::vector<cv::DMatch> matches_tmp(n_gms);
			for (size_t i = 0, j = 0; i < inlierMask.size(); i++)
			{
				if (inlierMask[i])
					matches_tmp[j++] = finalMatches[i];
			}
			finalMatches = matches_tmp;
		}

		if (verbose > 1)
		{
			t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
			cout << "Time GMS refinement: " << t_mea << "ms" << endl;
			t_oa += t_mea;
			cout << "Overall time with GMS refinement: " << t_oa << "ms" << endl;

			if (verbose > 2)
			{
				cout << "# of matches after GMS refinement: " << finalMatches.size() << endl;
			}
		}


	}

    if(SOFrefine)
    {
      if(!matchertype.compare("GMBSOF"))
      {
        cout << "SOF-filtering makes no sence when using GMBSOF! Skipping the filtering step." << endl;
      }
      else
      {
        if(verbose > 1)
        {
          t_mea = (double)getTickCount(); //Start time measurement
        }

        filterMatchesSOF(keypoints1, keypoints2, imgSi, finalMatches);

        if(verbose > 1)
        {
          t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
          cout << "Time SOF refinement: " << t_mea << "ms" << endl;
          t_oa += t_mea;
          cout << "Overall time with SOF refinement: " << t_oa << "ms" << endl;

          if(verbose > 2)
          {
            cout << "# of matches after SOF refinement: " << finalMatches.size() << endl;
          }
        }
      }
    }

    if(subPixRefine)
    {
      if(verbose > 1)
      {
        t_mea = (double)getTickCount(); //Start time measurement
      }

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
	  
	  int err_subpix;
	  if (subPixRefine == 1)
		  err_subpix = getSubPixMatches(img1, img2, &keypoints1_tmp, &keypoints2_tmp, &inliers);
	  else
		  err_subpix = getSubPixMatches_seperate_Imgs(img1, img2, &keypoints1_tmp, &keypoints2_tmp, &inliers);
	  if (err_subpix != 0)
      {
        cout << "Subpixel refinement would have rejected too many matches -> skipped. Taking integer positions instead." << endl;
      }
      else
      {
        for(size_t i = 0; i < finalMatches.size(); i++)
        {
          keypoints2[trainIdxs[i]] = keypoints2_tmp[i];
        }

        for(int i = (int)finalMatches.size() - 1; i >= 0; i--)
        {
          if(inliers[i])
          {
            finalMatches_tmp.push_back(finalMatches[i]);
          }
        }

        finalMatches = finalMatches_tmp;
      }

      if(verbose > 1)
      {
        t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
        cout << "Time subpixel refinement: " << t_mea << "ms" << endl;
        t_oa += t_mea;
        cout << "Overall time: " << t_oa << "ms" << endl;

        if(verbose > 2)
        {
          cout << "# of matches after subpixel refinement: " << finalMatches.size() << endl;
        }
      }
    }

    if(verbose > 0)
    {
      cout << endl;
    }

    kp1 = keypoints1;
    kp2 = keypoints2;

    return 0;
  }

  void filterMatchesSOF(const std::vector<cv::KeyPoint> &keypoints1,
                       const std::vector<cv::KeyPoint> &keypoints2,
                       const cv::Size &imgSi,
                       std::vector<cv::DMatch> &matches){
      vector<int> queryIdxs( matches.size() ), trainIdxs( matches.size() );
      vector<cv::KeyPoint> keypoints1_tmp, keypoints2_tmp;
      cv::Mat inliers;
      std::vector<cv::DMatch> finalMatches_tmp;

      for( size_t i = 0; i < matches.size(); i++ )
      {
          queryIdxs[i] = matches[i].queryIdx;
          trainIdxs[i] = matches[i].trainIdx;
      }

      for(size_t i = 0; i < matches.size(); i++ )
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

      for(unsigned int i = 0; i<keypoints1_tmp.size(); i++)
      {
          keyP1(i,0) = keypoints1_tmp[i].pt.x;
          keyP1(i,1) = keypoints1_tmp[i].pt.y;
      }

      for(unsigned int i = 0; i<keypoints2_tmp.size(); i++)
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
          //  if( inliers.at<bool>(i1,0) == true )
          //  {
          //    matchesMask[i1] = 1;
          //  }
          //}
          //drawMatches( *img1, keypoints1, *img2, keypoints2, matches, drawImg, Scalar::all(-1)/*CV_RGB(0, 255, 0)*/, CV_RGB(0, 0, 255), matchesMask);
          ////imwrite("C:\\work\\bf_matches_filtered_stat_flow.jpg", drawImg);
          //cv::imshow( "Source_1", drawImg );
          //cv::waitKey(0);
          /* ONLY FOR DEBUGGING END */

          for( size_t i1 = 0; i1 < matches.size(); i1++)
          {
              if( inliers.at<bool>((int)i1,0) )
              {
                  finalMatches_tmp.push_back(matches[i1]);
              }
          }

          if(finalMatches_tmp.size() >= MIN_FINAL_MATCHES)
          {
              matches = finalMatches_tmp;
          }
      }
  }

#if USE_FSTREAM
  //Function to write descriptor entries to disk -> for external tuning of the paramters of matcher VPTREE
  void writeDescriptorsToDisk(cv::Mat descriptors, std::string file)
  {
	  std::ofstream ofile(file, std::ofstream::app);

	  for (int i = 0; i < descriptors.rows; i++)
	  {
		if (descriptors.type() == CV_8U)
		{
			for (int j = 0; j < descriptors.cols - 1; j = j + 2)
			{
				ofile << integral_to_binary_string(descriptors.at<unsigned char>(i, j));
			}
		}
		else if (descriptors.type() == CV_32F)
		{
			for (int j = 0; j < descriptors.cols - 1; j = j + 2)
			{
				ofile << ' ' << descriptors.at<float>(i, j);
			}
		}
		else if (descriptors.type() == CV_64F)
		{
			for (int j = 0; j < descriptors.cols - 1; j = j + 2)
			{
				ofile << ' ' << descriptors.at<double>(i, j);
			}
		}
		else
		{
			ofile.close();
			return;
		}
		ofile << endl;
	  }
	  
	  ofile.close();
  }
#endif

} // namepace matchinglib
